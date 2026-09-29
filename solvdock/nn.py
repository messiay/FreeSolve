"""SolvDock PyTorch Neural Network Integration Layer.

Exposes SolvDock's physical force field as a standard torch.nn.Module, allowing
generative AI and molecular deep learning models to backpropagate physical
solvation, electrostatic, and Lennard-Jones gradients directly into their weights.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from rdkit import Chem

from solvdock.core.charges import assign_charges, get_partial_charges
from solvdock.core.grid_engine import SpatialGridEngine
from solvdock.core.solvation_pde import SolvationPDESolver
from solvdock.core.topology import build_topological_scale_matrix
from solvdock.pipeline.energy import CombinedPotential


class SolvDockPhysicsLoss(nn.Module):
    """Differentiable physics loss layer for deep learning models.

    Computes intermolecular non-bonded forces (soft-core Lennard-Jones and
    screened Debye-Hückel electrostatics in physiological saline) as well
    as intramolecular conformational strain.

    Args:
        salt_concentration: Ionic strength in mol/L (default: 0.150 M physiological saline).
        dielectric_water: Bulk solvent dielectric constant (default: 78.4).
        clash_penalty_weight: Multiplier on severe steric overlap (< 2.0 A).
        vdw_weight: Weight of Van der Waals interaction term.
        elec_weight: Weight of electrostatic interaction term.
    """

    def __init__(
        self,
        salt_concentration: float = 0.150,
        dielectric_water: float = 78.4,
        clash_penalty_weight: float = 2.0,
        vdw_weight: float = 1.0,
        elec_weight: float = 1.0,
        use_pde_grid: bool = False,
    ):
        super().__init__()
        self.salt_concentration = salt_concentration
        self.dielectric_water = dielectric_water
        self.clash_penalty_weight = clash_penalty_weight
        self.vdw_weight = vdw_weight
        self.elec_weight = elec_weight

        # Debye screening parameter kappa = sqrt(2 * I * F^2 / (eps0 * eps_r * R * T))
        # At 300K, 0.15 M: kappa ~ 0.127 A^-1 (Debye length ~ 7.9 A)
        self.register_buffer("kappa", torch.tensor(0.127 * (salt_concentration / 0.15) ** 0.5, dtype=torch.float32))

        self.grid_engine = SpatialGridEngine(box_size=32, grid_spacing=1.0)
        self.pde_solver = None
        if use_pde_grid:
            self.pde_solver = SolvationPDESolver(
                grid_spacing=1.0, steps=2, strict=False, disable_residual_mlp=True
            )
        self.potential = CombinedPotential(self.pde_solver, self.grid_engine)

    def forward(
        self,
        ligand_coords: torch.Tensor,
        ligand_charges: torch.Tensor,
        ligand_elements: torch.Tensor,
        pocket_coords: torch.Tensor,
        pocket_charges: torch.Tensor,
        pocket_elements: torch.Tensor,
        ligand_topo_matrix: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Calculates differentiable physical energy terms.

        Args:
            ligand_coords: Shape (N, 3), requires_grad=True
            ligand_charges: Shape (N,)
            ligand_elements: Shape (N,) atomic numbers
            pocket_coords: Shape (M, 3)
            pocket_charges: Shape (M,)
            pocket_elements: Shape (M,) atomic numbers
            ligand_topo_matrix: Optional (N, N) 1-2, 1-3, 1-4 scaling matrix

        Returns:
            Dictionary containing:
                - 'total_loss': Scalar loss to call .backward() on.
                - 'vdw_energy': Van der Waals interaction energy.
                - 'elec_energy': Screened electrostatic energy.
                - 'clash_count': Number of atom pairs with distance < 2.0 A.
        """
        # 1. Direct Intermolecular Non-Bonded Energy
        e_direct, e_vdw, e_elec = self.potential.compute_direct_energy(
            ligand_coords, ligand_charges, ligand_elements,
            pocket_coords, pocket_charges, pocket_elements,
        )

        # 2. Severe Clash Penalty (< 2.0 A)
        dists = torch.cdist(ligand_coords, pocket_coords)
        clash_mask = dists < 2.0
        clash_count = torch.sum(clash_mask.float())
        # Smooth quadratic clash barrier for pairs under 2.0 A
        clash_penalty = torch.sum(torch.relu(2.0 - dists) ** 2)

        # 3. Optional Intramolecular Strain
        e_intra = torch.tensor(0.0, device=ligand_coords.device, dtype=ligand_coords.dtype)
        if ligand_topo_matrix is not None:
            e_intra, _, _ = self.potential.compute_intramolecular_energy(
                ligand_coords, ligand_charges, ligand_elements, ligand_topo_matrix
            )

        # Total Weighted Physical Loss
        total_loss = (
            self.vdw_weight * e_vdw
            + self.elec_weight * e_elec
            + self.clash_penalty_weight * clash_penalty
            + 0.5 * e_intra
        )

        return {
            "total_loss": total_loss,
            "vdw_energy": e_vdw,
            "elec_energy": e_elec,
            "clash_penalty": clash_penalty,
            "clash_count": clash_count,
        }

    @classmethod
    def from_molecules(
        cls,
        ligand_mol: Chem.Mol,
        pocket_mol: Chem.Mol,
        **kwargs
    ) -> Tuple["SolvDockPhysicsLoss", Dict[str, torch.Tensor]]:
        """Convenience factory extracting charges, elements, and topology from RDKit molecules."""
        assign_charges(ligand_mol, scheme="gasteiger")
        assign_charges(pocket_mol, scheme="gasteiger")

        q_lig = get_partial_charges(ligand_mol)
        q_poc = get_partial_charges(pocket_mol)

        z_lig = torch.tensor([a.GetAtomicNum() for a in ligand_mol.GetAtoms()], dtype=torch.int64)
        z_poc = torch.tensor([a.GetAtomicNum() for a in pocket_mol.GetAtoms()], dtype=torch.int64)

        c_poc = torch.tensor(pocket_mol.GetConformer().GetPositions(), dtype=torch.float32)
        topo_lig = build_topological_scale_matrix(ligand_mol)

        layer = cls(**kwargs)
        tensors = {
            "pocket_coords": c_poc,
            "pocket_charges": q_poc,
            "pocket_elements": z_poc,
            "ligand_charges": q_lig,
            "ligand_elements": z_lig,
            "ligand_topo_matrix": topo_lig,
        }
        return layer, tensors


# Alias for bioRxiv preprint terminology
FreeSolvEPhysicsLoss = SolvDockPhysicsLoss
