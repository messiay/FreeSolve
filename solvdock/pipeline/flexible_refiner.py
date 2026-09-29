"""Flexible receptor induced-fit docking and side-chain relaxation engine."""

import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rdkit import Chem

from solvdock.core.charges import assign_charges, get_partial_charges
from solvdock.core.grid_engine import SpatialGridEngine
from solvdock.core.kinematics import (
    DifferentiableSE3,
    DifferentiableTorsionTree,
    build_downstream_subgraphs,
    find_rotatable_bonds,
)
from solvdock.core.solvation_pde import SolvationPDESolver
from solvdock.core.topology import build_topological_scale_matrix
from solvdock.pipeline.energy import CombinedPotential, ATOM_PARAMS, DEFAULT_PARAMS


class FlexibleRefiner:
    """Induced-fit flexible receptor pose refiner.

    Simultaneously relaxes ligand conformation/pose and receptor pocket side-chains
    under harmonic backbone restraints and covalent bond preservation potentials.
    """

    def __init__(
        self,
        potential: Optional[CombinedPotential] = None,
        k_backbone: float = 20.0,
        k_bond: float = 100.0,
        v_barrier: float = 1.5,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.k_backbone = float(k_backbone)
        self.k_bond = float(k_bond)
        self.v_barrier = float(v_barrier)

        if potential is None:
            grid_engine = SpatialGridEngine(box_size=32, grid_spacing=1.0).to(self.device)
            pde_solver = SolvationPDESolver(
                grid_spacing=1.0,
                steps=2,
                calibrated_constants_path="configs/calibrated_constants.yaml",
                strict=False,
                disable_residual_mlp=True,
            ).to(self.device)
            self.potential = CombinedPotential(pde_solver, grid_engine).to(self.device)
        else:
            self.potential = potential.to(self.device)

    @staticmethod
    def identify_backbone_mask(mol: Chem.Mol) -> torch.Tensor:
        """Identifies protein backbone atoms (N, CA, C, O) via PDB residue info."""
        n_atoms = mol.GetNumAtoms()
        bb_names = {"N", "CA", "C", "O"}
        is_bb = torch.zeros(n_atoms, dtype=torch.bool)
        for i, atom in enumerate(mol.GetAtoms()):
            info = atom.GetPDBResidueInfo()
            if info and info.GetName().strip() in bb_names:
                is_bb[i] = True
        return is_bb

    @staticmethod
    def extract_covalent_bonds(
        mol: Chem.Mol, coords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extracts covalent bond pairs (M, 2) and reference equilibrium bond lengths (M,)."""
        b_idx, b_r0 = [], []
        c_np = coords.detach().cpu().numpy()
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            r0 = float(np.linalg.norm(c_np[i] - c_np[j]))
            b_idx.append((i, j))
            b_r0.append(r0)
        if not b_idx:
            return torch.empty((0, 2), dtype=torch.int64), torch.empty((0,), dtype=torch.float32)
        return torch.tensor(b_idx, dtype=torch.int64), torch.tensor(b_r0, dtype=torch.float32)

    def refine_induced_fit(
        self,
        ligand_mol: Chem.Mol,
        pocket_mol: Chem.Mol,
        backbone_mask: Optional[torch.Tensor] = None,
        freeze_backbone: bool = False,
        mode: str = "cartesian",
        max_steps: int = 40,
        lr: float = 0.05,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Performs gradient-driven induced-fit relaxation of pocket side chains + ligand.

        Args:
            ligand_mol: Conformation of ligand.
            pocket_mol: Conformation of receptor pocket.
            backbone_mask: Boolean mask indicating backbone atoms.
            freeze_backbone: If True, backbone atoms are held completely stationary.
            mode: 'cartesian' (spring-restrained displacement) or 'torsional'
                  (exact articulated joint physics with locked bond lengths/angles).
            max_steps: Number of optimizer steps.
            lr: Learning rate.
            verbose: Verbosity flag.

        Returns:
            Dictionary of energetic and geometric convergence metrics.
        """
        # Ensure partial charges
        assign_charges(ligand_mol, scheme="gasteiger")
        assign_charges(pocket_mol, scheme="gasteiger")

        conf_lig = ligand_mol.GetConformer()
        conf_poc = pocket_mol.GetConformer()

        n_lig = ligand_mol.GetNumAtoms()
        n_poc = pocket_mol.GetNumAtoms()

        coords_lig_0 = torch.tensor(
            [[conf_lig.GetAtomPosition(i).x, conf_lig.GetAtomPosition(i).y, conf_lig.GetAtomPosition(i).z]
             for i in range(n_lig)],
            dtype=torch.float32,
            device=self.device,
        )
        coords_poc_0 = torch.tensor(
            [[conf_poc.GetAtomPosition(i).x, conf_poc.GetAtomPosition(i).y, conf_poc.GetAtomPosition(i).z]
             for i in range(n_poc)],
            dtype=torch.float32,
            device=self.device,
        )

        q_lig = get_partial_charges(ligand_mol).to(self.device)
        q_poc = get_partial_charges(pocket_mol).to(self.device)
        z_lig = torch.tensor([a.GetAtomicNum() for a in ligand_mol.GetAtoms()], dtype=torch.int64, device=self.device)
        z_poc = torch.tensor([a.GetAtomicNum() for a in pocket_mol.GetAtoms()], dtype=torch.int64, device=self.device)

        if backbone_mask is None:
            backbone_mask = self.identify_backbone_mask(pocket_mol)
        backbone_mask = backbone_mask.to(self.device)
        is_sidechain = ~backbone_mask

        if mode == "torsional":
            # --- Articulated Torsional Kinematics (Joint Physics) ---
            fixed_indices = {i for i in range(n_poc) if backbone_mask[i].item()}
            rot_poc = find_rotatable_bonds(pocket_mol, fixed_atom_indices=fixed_indices)
            masks_poc = build_downstream_subgraphs(pocket_mol, rot_poc, root_indices=fixed_indices)
            torsion_poc = DifferentiableTorsionTree(coords_poc_0, rot_poc, masks_poc).to(self.device)

            rot_lig = find_rotatable_bonds(ligand_mol)
            masks_lig = build_downstream_subgraphs(ligand_mol, rot_lig)
            torsion_lig = DifferentiableTorsionTree(coords_lig_0, rot_lig, masks_lig).to(self.device)
            se3_lig = DifferentiableSE3(coords_lig_0.mean(dim=0)).to(self.device)

            # Build topological scaling matrices for intramolecular non-bonded evaluations
            # Excludes 1-2, 1-3 pairs and scales 1-4 dihedrals by 0.5 to prevent false self-clashes
            topo_poc = build_topological_scale_matrix(pocket_mol, device=self.device)
            topo_lig = build_topological_scale_matrix(ligand_mol, device=self.device)

            opt_params = [p for p in list(se3_lig.parameters()) + list(torsion_lig.parameters()) + list(torsion_poc.parameters())
                          if p.requires_grad and p.numel() > 0]
            optimizer = optim.Adam(opt_params, lr=lr)

            def compute_torsional_loss() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                curr_lig = se3_lig(torsion_lig())
                curr_poc = torsion_poc()
                e_direct, _, _ = self.potential.compute_direct_energy(
                    curr_lig, q_lig, z_lig,
                    curr_poc, q_poc, z_poc,
                )
                e_intra_poc, _, _ = self.potential.compute_intramolecular_energy(
                    curr_poc, q_poc, z_poc, topo_poc
                )
                e_intra_lig, _, _ = self.potential.compute_intramolecular_energy(
                    curr_lig, q_lig, z_lig, topo_lig
                )

                if torsion_poc.n_torsions > 0:
                    e_strain_poc = torch.sum(self.v_barrier * (1.0 - torch.cos(3.0 * torsion_poc.thetas)))
                else:
                    e_strain_poc = torch.tensor(0.0, device=self.device)

                if torsion_lig.n_torsions > 0:
                    e_strain_lig = torch.sum(self.v_barrier * (1.0 - torch.cos(3.0 * torsion_lig.thetas)))
                else:
                    e_strain_lig = torch.tensor(0.0, device=self.device)

                total_loss = e_direct + e_intra_poc + e_intra_lig + e_strain_poc + e_strain_lig
                return total_loss, e_direct, e_intra_poc, (e_strain_poc + e_strain_lig)

            with torch.no_grad():
                init_loss, init_direct, init_intra_poc, init_strain = compute_torsional_loss()

            for step in range(max_steps):
                optimizer.zero_grad()
                loss, _, _, _ = compute_torsional_loss()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                final_loss, final_direct, final_intra_poc, final_strain = compute_torsional_loss()
                final_lig = se3_lig(torsion_lig()).detach()
                final_poc = torsion_poc().detach()

                lig_rmsd = float(torch.sqrt(torch.mean(torch.sum((final_lig - coords_lig_0) ** 2, dim=-1))))
                bb_rmsd = 0.0  # Backbone mathematically locked
                sc_rmsd = float(torch.sqrt(torch.mean(torch.sum((final_poc[is_sidechain] - coords_poc_0[is_sidechain]) ** 2, dim=-1)))) if is_sidechain.sum() > 0 else 0.0
                max_sc_disp = float(torch.max(torch.sqrt(torch.sum((final_poc[is_sidechain] - coords_poc_0[is_sidechain]) ** 2, dim=-1)))) if is_sidechain.sum() > 0 else 0.0

            return {
                "initial_loss": float(init_loss.item()),
                "initial_direct_energy": float(init_direct.item()),
                "initial_intra_pocket_energy": float(init_intra_poc.item()),
                "final_loss": float(final_loss.item()),
                "final_direct_energy": float(final_direct.item()),
                "final_intra_pocket_energy": float(final_intra_poc.item()),
                "final_torsional_strain_energy": float(final_strain.item()),
                "final_backbone_restraint_energy": 0.0,
                "final_bond_strain_energy": 0.0,  # Exact 0.000 by kinematic construction
                "ligand_rmsd": lig_rmsd,
                "backbone_rmsd": bb_rmsd,
                "sidechain_rmsd": sc_rmsd,
                "max_sidechain_displacement": max_sc_disp,
                "final_ligand_coords": final_lig.cpu(),
                "final_pocket_coords": final_poc.cpu(),
                "n_sidechain_torsions": len(rot_poc),
                "n_ligand_torsions": len(rot_lig),
            }

        # --- Default Cartesian Mode (Spring Restraints) ---
        bonds_lig_idx, bonds_lig_r0 = self.extract_covalent_bonds(ligand_mol, coords_lig_0)
        bonds_poc_idx, bonds_poc_r0 = self.extract_covalent_bonds(pocket_mol, coords_poc_0)
        bonds_lig_idx = bonds_lig_idx.to(self.device)
        bonds_lig_r0 = bonds_lig_r0.to(self.device)
        bonds_poc_idx = bonds_poc_idx.to(self.device)
        bonds_poc_r0 = bonds_poc_r0.to(self.device)

        delta_lig = nn.Parameter(torch.zeros_like(coords_lig_0))
        delta_poc = nn.Parameter(torch.zeros_like(coords_poc_0))

        optimizer = optim.Adam([delta_lig, delta_poc], lr=lr)

        def compute_loss() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            curr_lig = coords_lig_0 + delta_lig
            curr_poc = coords_poc_0 + delta_poc

            e_direct, e_lj, e_coulomb = self.potential.compute_direct_energy(
                curr_lig, q_lig, z_lig,
                curr_poc, q_poc, z_poc,
            )

            disp_bb = delta_poc[backbone_mask]
            if disp_bb.shape[0] > 0 and not freeze_backbone:
                e_restraint_bb = 0.5 * self.k_backbone * torch.sum(disp_bb ** 2)
            else:
                e_restraint_bb = torch.tensor(0.0, device=self.device)

            e_bond = torch.tensor(0.0, device=self.device)
            if bonds_lig_idx.shape[0] > 0:
                d_lig = torch.norm(curr_lig[bonds_lig_idx[:, 0]] - curr_lig[bonds_lig_idx[:, 1]], dim=-1)
                e_bond = e_bond + 0.5 * self.k_bond * torch.sum((d_lig - bonds_lig_r0) ** 2)
            if bonds_poc_idx.shape[0] > 0:
                d_poc = torch.norm(curr_poc[bonds_poc_idx[:, 0]] - curr_poc[bonds_poc_idx[:, 1]], dim=-1)
                e_bond = e_bond + 0.5 * self.k_bond * torch.sum((d_poc - bonds_poc_r0) ** 2)

            total = e_direct + e_restraint_bb + e_bond
            return total, e_direct, e_restraint_bb, e_bond

        with torch.no_grad():
            init_loss, init_direct, _, _ = compute_loss()

        for step in range(max_steps):
            optimizer.zero_grad()
            total_loss, _, _, _ = compute_loss()
            total_loss.backward()
            if freeze_backbone and backbone_mask.any():
                delta_poc.grad[backbone_mask] = 0.0
            optimizer.step()
            if freeze_backbone and backbone_mask.any():
                delta_poc.data[backbone_mask] = 0.0

        with torch.no_grad():
            final_loss, final_direct, final_bb_res, final_bond = compute_loss()
            final_lig = coords_lig_0 + delta_lig
            final_poc = coords_poc_0 + delta_poc

            lig_rmsd = float(torch.sqrt(torch.mean(torch.sum(delta_lig ** 2, dim=-1))))
            bb_rmsd = float(torch.sqrt(torch.mean(torch.sum(delta_poc[backbone_mask] ** 2, dim=-1)))) if backbone_mask.sum() > 0 else 0.0
            sc_rmsd = float(torch.sqrt(torch.mean(torch.sum(delta_poc[is_sidechain] ** 2, dim=-1)))) if is_sidechain.sum() > 0 else 0.0
            max_sc_disp = float(torch.max(torch.sqrt(torch.sum(delta_poc[is_sidechain] ** 2, dim=-1)))) if is_sidechain.sum() > 0 else 0.0

        return {
            "initial_loss": float(init_loss.item()),
            "initial_direct_energy": float(init_direct.item()),
            "final_loss": float(final_loss.item()),
            "final_direct_energy": float(final_direct.item()),
            "final_backbone_restraint_energy": float(final_bb_res.item()),
            "final_bond_strain_energy": float(final_bond.item()),
            "ligand_rmsd": lig_rmsd,
            "backbone_rmsd": bb_rmsd,
            "sidechain_rmsd": sc_rmsd,
            "max_sidechain_displacement": max_sc_disp,
            "final_ligand_coords": final_lig.detach().cpu(),
            "final_pocket_coords": final_poc.detach().cpu(),
        }
