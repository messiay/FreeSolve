"""Basin-hopping global simulation engine for blind docking and thermodynamic search."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from rdkit import Chem
from rdkit.Geometry import Point3D

from solvdock.core.pose_optimizer import PoseOptimizer
from solvdock.core.topology import MolecularTopology


@dataclass
class BasinHoppingResult:
    """Encapsulates the output of a global basin-hopping simulation."""

    best_mol: Chem.Mol
    best_delta_G_bind: float
    best_components: Dict[str, float]
    trajectories: List[Dict[str, Any]]
    top_modes: List[Dict[str, Any]]
    acceptance_rate: float
    metadata: Dict[str, Any] = field(default_factory=dict)


def compute_heavy_atom_rmsd(mol1: Chem.Mol, mol2: Chem.Mol) -> float:
    """Computes in-pocket Cartesian RMSD over heavy atoms without superposition.

    Superposition must NOT be applied in docking mode analysis because distinct
    docking modes are defined by their spatial position relative to the pocket.
    """
    conf1 = mol1.GetConformer()
    conf2 = mol2.GetConformer()
    n_atoms = mol1.GetNumAtoms()

    sq_dists = []
    for i in range(n_atoms):
        atom = mol1.GetAtomWithIdx(i)
        if atom.GetAtomicNum() > 1:  # Heavy atoms only
            p1 = conf1.GetAtomPosition(i)
            p2 = conf2.GetAtomPosition(i)
            dx = p1.x - p2.x
            dy = p1.y - p2.y
            dz = p1.z - p2.z
            sq_dists.append(dx * dx + dy * dy + dz * dz)

    # Fallback to all atoms if molecule has no heavy atoms
    if not sq_dists:
        for i in range(n_atoms):
            p1 = conf1.GetAtomPosition(i)
            p2 = conf2.GetAtomPosition(i)
            dx = p1.x - p2.x
            dy = p1.y - p2.y
            dz = p1.z - p2.z
            sq_dists.append(dx * dx + dy * dy + dz * dz)

    return float(np.sqrt(np.mean(sq_dists))) if sq_dists else 0.0


class BasinHoppingDockingEngine:
    """Autonomous global biophysical docking engine via stochastic basin-hopping.

    Combines large stochastic jumps across rigid so(3) x R^3 coordinates and
    torsion angles with inner-loop gradient descent on the full unrolled
    solvation PDE + direct interaction potential. Jumps across energy barriers
    are accepted or rejected via the Metropolis-Hastings criterion at temperature T.
    """

    KB: float = 0.00198720425864083  # kcal / (mol * K)

    def __init__(
        self,
        optimizer: PoseOptimizer,
        temperature: float = 300.0,
        step_size_trans: float = 2.0,
        step_size_rot: float = 0.5,
        step_size_dihedral: float = 0.5,
        box_radius: float = 8.0,
        local_steps: int = 8,
        local_lr: float = 0.05,
        rmsd_clustering_cutoff: float = 1.0,
        seed: Optional[int] = None,
    ):
        self.optimizer = optimizer
        self.temperature = float(temperature)
        self.step_size_trans = float(step_size_trans)
        self.step_size_rot = float(step_size_rot)
        self.step_size_dihedral = float(step_size_dihedral)
        self.box_radius = float(box_radius)
        self.local_steps = int(local_steps)
        self.local_lr = float(local_lr)
        self.rmsd_clustering_cutoff = float(rmsd_clustering_cutoff)
        self.seed = seed

    def _reflect_boundary(
        self,
        trial_coord: np.ndarray,
        center: np.ndarray,
        radius: float,
    ) -> np.ndarray:
        """Applies elastic reflection boundary condition within a bounding box."""
        bounded = trial_coord.copy()
        for i in range(3):
            c_min = center[i] - radius
            c_max = center[i] + radius
            if bounded[i] > c_max:
                overshoot = bounded[i] - c_max
                bounded[i] = c_max - overshoot
            elif bounded[i] < c_min:
                overshoot = c_min - bounded[i]
                bounded[i] = c_min + overshoot
            # Clamp in case of extreme double overshoot
            bounded[i] = np.clip(bounded[i], c_min, c_max)
        return bounded

    def run(
        self,
        initial_mol: Chem.Mol,
        pocket_mol: Optional[Chem.Mol] = None,
        n_trials: int = 15,
    ) -> BasinHoppingResult:
        """Executes global basin-hopping simulation over the molecular complex.

        Args:
            initial_mol: RDKit Mol of the ligand (with 3D conformer).
            pocket_mol: Optional RDKit Mol of the pocket receptor.
            n_trials: Number of outer basin-hopping Monte Carlo cycles.

        Returns:
            BasinHoppingResult containing best pose, free energy, mode clusters,
            and complete acceptance trajectory.
        """
        rng = np.random.default_rng(self.seed)
        if self.seed is not None:
            torch.manual_seed(self.seed)

        # 1. Topological analysis of ligand
        topology = MolecularTopology(initial_mol)
        K = len(topology.rotatable_bonds)
        lig_coords_initial, _, _ = self.optimizer.extract_mol_tensors(initial_mol)
        ligand_initial_center_t = torch.mean(lig_coords_initial, dim=0)
        ligand_initial_center = ligand_initial_center_t.detach().cpu().numpy()

        # 2. Pocket coordinates and reference center
        if pocket_mol is not None and pocket_mol.GetNumAtoms() > 0:
            poc_coords, poc_charges, _ = self.optimizer.extract_mol_tensors(pocket_mol)
            pocket_center_t = torch.mean(poc_coords, dim=0)
            pocket_center = pocket_center_t.detach().cpu().numpy()
            ref_grid_coords = poc_coords
        else:
            poc_coords = torch.empty((0, 3), dtype=torch.float32, device=self.optimizer.device)
            poc_charges = torch.empty((0,), dtype=torch.float32, device=self.optimizer.device)
            pocket_center = ligand_initial_center.copy()
            ref_grid_coords = lig_coords_initial

        # 3. Precompute fixed spatial grid origin and static pocket solvation
        fixed_grid_origin = self.optimizer.potential.grid_engine.get_grid_origin(ref_grid_coords)
        if poc_coords.shape[0] > 0:
            with torch.no_grad():
                dG_pocket, _ = self.optimizer.potential.compute_solvation(
                    poc_coords, poc_charges, grid_origin=fixed_grid_origin
                )
        else:
            dG_pocket = torch.tensor(0.0, device=self.optimizer.device)

        # 4. Initial local minimization at starting coordinates
        init_res = self.optimizer.refine(
            initial_mol=initial_mol,
            pocket_mol=pocket_mol,
            steps=self.local_steps,
            lr=self.local_lr,
            fixed_grid_origin=fixed_grid_origin,
            precomputed_dG_pocket=dG_pocket,
        )

        curr_thetas = init_res["optimized_thetas"].copy() if K > 0 else np.array([], dtype=np.float32)
        curr_omega = init_res["optimized_omega"].copy()
        curr_trans = init_res["optimized_translation"].copy()
        curr_energy = float(init_res["delta_G_bind"])
        curr_mol = init_res["mol"]
        curr_components = init_res["components"]

        best_energy = curr_energy
        best_mol = curr_mol
        best_components = curr_components

        trajectories = [
            {
                "step": 0,
                "energy": curr_energy,
                "delta_energy": 0.0,
                "accepted": True,
                "p_accept": 1.0,
                "temperature": self.temperature,
            }
        ]

        all_minima = [
            {
                "mol": curr_mol,
                "energy": curr_energy,
                "components": curr_components,
                "translation": curr_trans,
                "omega": curr_omega,
                "thetas": curr_thetas,
            }
        ]

        accepted_jumps = 0
        kt = self.KB * self.temperature

        # 5. Basin-Hopping Outer Loop
        for step in range(1, n_trials + 1):
            # A. Perturb translation with elastic reflection at search box boundary
            delta_trans = rng.uniform(-self.step_size_trans, self.step_size_trans, size=3).astype(np.float32)
            trial_trans = curr_trans + delta_trans
            trial_ligand_center = ligand_initial_center + trial_trans

            # Apply reflection relative to pocket center
            bounded_center = self._reflect_boundary(trial_ligand_center, pocket_center, self.box_radius)
            trial_trans = bounded_center - ligand_initial_center

            # B. Perturb Lie algebra so(3) axis-angle vector
            delta_omega = rng.normal(0.0, self.step_size_rot, size=3).astype(np.float32)
            trial_omega = curr_omega + delta_omega

            # C. Perturb dihedral angles
            if K > 0:
                delta_thetas = rng.uniform(-self.step_size_dihedral, self.step_size_dihedral, size=K).astype(np.float32)
                trial_thetas = curr_thetas + delta_thetas
            else:
                trial_thetas = None

            # D. Inner-loop gradient descent minimization on unrolled PDE potential
            trial_thetas_tensor = (
                torch.tensor(trial_thetas, dtype=torch.float32, device=self.optimizer.device)
                if K > 0
                else None
            )
            trial_omega_tensor = torch.tensor(trial_omega, dtype=torch.float32, device=self.optimizer.device)
            trial_trans_tensor = torch.tensor(trial_trans, dtype=torch.float32, device=self.optimizer.device)

            trial_res = self.optimizer.refine(
                initial_mol=initial_mol,
                pocket_mol=pocket_mol,
                steps=self.local_steps,
                lr=self.local_lr,
                thetas_init=trial_thetas_tensor,
                omega_init=trial_omega_tensor,
                translation_init=trial_trans_tensor,
                fixed_grid_origin=fixed_grid_origin,
                precomputed_dG_pocket=dG_pocket,
            )

            trial_energy = float(trial_res["delta_G_bind"])
            delta_e = trial_energy - curr_energy

            # E. Metropolis-Hastings acceptance test
            if delta_e <= 0.0:
                p_accept = 1.0
                accepted = True
            else:
                p_accept = float(np.exp(-delta_e / kt)) if kt > 1e-6 else 0.0
                accepted = bool(rng.uniform(0.0, 1.0) < p_accept)

            if accepted:
                accepted_jumps += 1
                curr_thetas = trial_res["optimized_thetas"].copy() if K > 0 else np.array([], dtype=np.float32)
                curr_omega = trial_res["optimized_omega"].copy()
                curr_trans = trial_res["optimized_translation"].copy()
                curr_energy = trial_energy
                curr_mol = trial_res["mol"]
                curr_components = trial_res["components"]

            # F. Track global best across all visited basins
            if trial_energy < best_energy:
                best_energy = trial_energy
                best_mol = trial_res["mol"]
                best_components = trial_res["components"]

            trajectories.append(
                {
                    "step": step,
                    "energy": trial_energy,
                    "delta_energy": delta_e,
                    "accepted": accepted,
                    "p_accept": min(1.0, p_accept),
                    "temperature": self.temperature,
                }
            )

            all_minima.append(
                {
                    "mol": trial_res["mol"],
                    "energy": trial_energy,
                    "components": trial_res["components"],
                    "translation": trial_res["optimized_translation"],
                    "omega": trial_res["optimized_omega"],
                    "thetas": trial_res["optimized_thetas"],
                }
            )

        # 6. Mode clustering by Cartesian heavy-atom RMSD
        # Sort all visited minima by binding free energy ascending
        all_minima.sort(key=lambda item: item["energy"])
        top_modes: List[Dict[str, Any]] = []

        for candidate in all_minima:
            is_distinct = True
            for mode in top_modes:
                rmsd = compute_heavy_atom_rmsd(candidate["mol"], mode["mol"])
                if rmsd < self.rmsd_clustering_cutoff:
                    is_distinct = False
                    break
            if is_distinct:
                top_modes.append(candidate)

        acceptance_rate = float(accepted_jumps / n_trials) if n_trials > 0 else 1.0

        return BasinHoppingResult(
            best_mol=best_mol,
            best_delta_G_bind=best_energy,
            best_components=best_components,
            trajectories=trajectories,
            top_modes=top_modes,
            acceptance_rate=acceptance_rate,
            metadata={
                "n_trials": n_trials,
                "temperature": self.temperature,
                "box_radius": self.box_radius,
                "local_steps": self.local_steps,
                "num_rotatable_bonds": K,
                "total_minima_evaluated": len(all_minima),
                "num_distinct_modes": len(top_modes),
            },
        )
