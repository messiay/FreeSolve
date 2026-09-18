"""Differentiable gradient-descent pose optimizer over torsion angles and so(3) rigid coordinates."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.optim as optim
from rdkit import Chem
from rdkit.Geometry import Point3D

from solvdock.core.topology import MolecularTopology
from solvdock.core.kinematics import apply_torsions, apply_rigid_transform
from solvdock.pipeline.energy import CombinedPotential


class PoseOptimizer:
    """Gradient-descent pose refiner over scalar torsion angles and so(3) rigid pose.

    Minimizes the combined potential E_direct + ΔG_solv by propagating
    gradients through the Rodrigues kinematic chain, the so(3) matrix exponential,
    the 3D Poisson solve, and the unrolled Ginzburg-Landau solvation PDE.
    """

    def __init__(self, combined_potential: CombinedPotential, device: str = "cpu"):
        self.potential = combined_potential
        self.device = torch.device(device)
        self.potential.to(self.device)

    def extract_mol_tensors(self, mol: Chem.Mol) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extracts coordinates, charges, and atomic numbers from an RDKit Mol."""
        num_atoms = mol.GetNumAtoms()
        conf = mol.GetConformer()

        coords = torch.zeros((num_atoms, 3), dtype=torch.float32, device=self.device)
        charges = torch.zeros((num_atoms,), dtype=torch.float32, device=self.device)
        atomic_nums = torch.zeros((num_atoms,), dtype=torch.int64, device=self.device)

        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            coords[i] = torch.tensor([pos.x, pos.y, pos.z], device=self.device)
            atomic_nums[i] = atom.GetAtomicNum()
            try:
                q = float(atom.GetProp("_GasteigerCharge"))
                if np.isnan(q) or np.isinf(q):
                    q = 0.0
            except KeyError:
                q = 0.0
            charges[i] = q

        return coords, charges, atomic_nums

    def refine(
        self,
        initial_mol: Chem.Mol,
        pocket_mol: Optional[Chem.Mol] = None,
        steps: int = 20,
        lr: float = 0.05,
        optimizer_type: str = "adam",
        thetas_init: Optional[torch.Tensor] = None,
        omega_init: Optional[torch.Tensor] = None,
        translation_init: Optional[torch.Tensor] = None,
        fixed_grid_origin: Optional[torch.Tensor] = None,
        precomputed_dG_pocket: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Refines the ligand pose in the pocket by gradient descent on the combined potential.

        Args:
            initial_mol: Starting ligand conformer (RDKit Mol).
            pocket_mol: Optional pocket/receptor structure (RDKit Mol).
            steps: Number of gradient descent steps (default 20).
            lr: Learning rate (default 0.05).
            optimizer_type: 'adam' or 'lbfgs'.
            thetas_init: Optional starting torsion angles tensor (K,).
            omega_init: Optional starting so(3) vector tensor (3,).
            translation_init: Optional starting translation vector tensor (3,).
            fixed_grid_origin: Optional precomputed spatial grid origin.
            precomputed_dG_pocket: Optional precomputed static pocket solvation energy.

        Returns:
            Dict containing refined RDKit Mol, delta_G_bind, and component breakdown.
        """
        # 1. Topological decomposition of the ligand
        topology = MolecularTopology(initial_mol)
        lig_coords, lig_charges, lig_z = (
            topology.atom_coords.to(self.device),
            topology.partial_charges.to(self.device),
            topology.atomic_numbers.to(self.device),
        )

        # 2. Pocket atom tensors
        if pocket_mol is not None and pocket_mol.GetNumAtoms() > 0:
            poc_coords, poc_charges, poc_z = self.extract_mol_tensors(pocket_mol)
        else:
            poc_coords = torch.empty((0, 3), dtype=torch.float32, device=self.device)
            poc_charges = torch.empty((0,), dtype=torch.float32, device=self.device)
            poc_z = torch.empty((0,), dtype=torch.int64, device=self.device)

        # Compute fixed grid origin centered on the initial complex if not pre-passed
        if fixed_grid_origin is None:
            all_initial = torch.cat([lig_coords, poc_coords], dim=0) if poc_coords.shape[0] > 0 else lig_coords
            fixed_grid_origin = self.potential.grid_engine.get_grid_origin(all_initial)
        ligand_center = lig_coords.mean(dim=0, keepdim=True)

        # 3. Precompute static pocket solvation (invariant across all steps for rigid receptor)
        if precomputed_dG_pocket is not None:
            dG_pocket = precomputed_dG_pocket
        else:
            dG_pocket = None
            if poc_coords.shape[0] > 0:
                with torch.no_grad():
                    dG_pocket, _ = self.potential.compute_solvation(poc_coords, poc_charges)
                    # Invariance verification check
                    dG_pocket_verify, _ = self.potential.compute_solvation(poc_coords, poc_charges)
                    assert torch.allclose(dG_pocket, dG_pocket_verify, atol=1e-6), "Pocket solvation must be strictly invariant"

        # Precompute static ligand solvation if rigid (K == 0)
        dG_ligand_static = None
        K = topology.rotatable_bonds.shape[0]
        if K == 0:
            with torch.no_grad():
                dG_ligand_static, _ = self.potential.compute_solvation(lig_coords, lig_charges)

        # 4. Parameters to optimize: thetas (K,), omega (3,), translation (3,)
        if thetas_init is not None and K > 0:
            thetas = thetas_init.clone().detach().to(device=self.device, dtype=torch.float32).requires_grad_(True)
        else:
            thetas = torch.zeros(K, dtype=torch.float32, device=self.device, requires_grad=(K > 0))

        if omega_init is not None:
            omega = omega_init.clone().detach().to(device=self.device, dtype=torch.float32).requires_grad_(True)
        else:
            omega = torch.zeros(3, dtype=torch.float32, device=self.device, requires_grad=True)

        if translation_init is not None:
            translation = translation_init.clone().detach().to(device=self.device, dtype=torch.float32).requires_grad_(True)
        else:
            translation = torch.zeros(3, dtype=torch.float32, device=self.device, requires_grad=True)

        params = [omega, translation]
        if K > 0:
            params.append(thetas)

        if optimizer_type.lower() == "adam":
            optimizer = optim.Adam(params, lr=lr)
        elif optimizer_type.lower() == "lbfgs":
            optimizer = optim.LBFGS(params, lr=lr, max_iter=steps)
        else:
            optimizer = optim.Adam(params, lr=lr)

        best_loss = float("inf")
        best_coords = lig_coords.clone()
        best_components = {}
        steps_taken = 0

        # 5. Gradient descent optimization loop
        for step in range(steps):
            steps_taken += 1

            def closure():
                optimizer.zero_grad()
                # Forward kinematics
                curr_coords = lig_coords
                if K > 0:
                    curr_coords = apply_torsions(curr_coords, thetas, topology)
                transformed_coords = apply_rigid_transform(
                    curr_coords, omega, translation, center=ligand_center
                )

                # If flexible (K > 0), compute ligand self-solvation dynamically
                # so gradients propagate through internal dihedrals.
                # If rigid (K == 0), reuse precomputed dG_ligand_static.
                dG_lig = dG_ligand_static
                if K > 0:
                    dG_lig, _ = self.potential.compute_solvation(curr_coords, lig_charges)

                delta_G_bind, comp = self.potential(
                    transformed_coords, lig_charges, lig_z,
                    poc_coords, poc_charges, poc_z,
                    grid_origin=fixed_grid_origin,
                    dG_pocket=dG_pocket,
                    dG_ligand=dG_lig,
                    num_rotatable_bonds=K,
                )
                delta_G_bind.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=0.5)
                return delta_G_bind

            loss = optimizer.step(closure)

            # Check convergence via gradient norm
            grad_sq_sum = (
                torch.sum(omega.grad ** 2) + torch.sum(translation.grad ** 2)
            )
            if K > 0 and thetas.grad is not None:
                grad_sq_sum += torch.sum(thetas.grad ** 2)
            grad_norm = torch.sqrt(grad_sq_sum).item()

            if loss.item() < best_loss:
                best_loss = loss.item()
                with torch.no_grad():
                    c = lig_coords
                    if K > 0:
                        c = apply_torsions(c, thetas, topology)
                    best_coords = apply_rigid_transform(c, omega, translation, center=ligand_center)
                    dG_lig_best = dG_ligand_static
                    if K > 0:
                        dG_lig_best, _ = self.potential.compute_solvation(c, lig_charges)
                    _, best_components = self.potential(
                        best_coords, lig_charges, lig_z,
                        poc_coords, poc_charges, poc_z,
                        grid_origin=fixed_grid_origin,
                        dG_pocket=dG_pocket,
                        dG_ligand=dG_lig_best,
                        num_rotatable_bonds=K,
                    )

            if grad_norm < 1e-4:
                break

        # 6. Write refined coordinates back into an RDKit Mol
        refined_mol = Chem.Mol(initial_mol)
        conf = refined_mol.GetConformer()
        best_coords_np = best_coords.detach().cpu().numpy()
        for i in range(refined_mol.GetNumAtoms()):
            conf.SetAtomPosition(
                i,
                Point3D(
                    float(best_coords_np[i, 0]),
                    float(best_coords_np[i, 1]),
                    float(best_coords_np[i, 2]),
                ),
            )

        return {
            "mol": refined_mol,
            "delta_G_bind": float(best_loss),
            "components": {
                k: float(v.item()) if isinstance(v, torch.Tensor) else float(v)
                for k, v in best_components.items()
            },
            "optimized_thetas": thetas.detach().cpu().numpy() if K > 0 else np.array([]),
            "optimized_omega": omega.detach().cpu().numpy(),
            "optimized_translation": translation.detach().cpu().numpy(),
            "steps_taken": steps_taken,
        }
