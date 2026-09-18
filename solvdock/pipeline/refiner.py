"""High-level SolvDock pose refiner API."""

import os
from typing import Any, Dict, Optional, Union
import yaml
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from solvdock.core.grid_engine import SpatialGridEngine
from solvdock.core.solvation_pde import SolvationPDESolver
from solvdock.core.pose_optimizer import PoseOptimizer
from solvdock.core.basin_hopping import BasinHoppingDockingEngine, BasinHoppingResult
from solvdock.pipeline.energy import CombinedPotential


class SolvDockRefiner:
    """High-level API wiring the physical grid, Poisson solve, calibrated PDE, and pose optimizer.

    Refuses to instantiate without calibrated physical constants and a trained
    OrientationalCorrectionMLP checkpoint when strict=True.
    """

    def __init__(
        self,
        constants_path: str = "configs/calibrated_constants.yaml",
        checkpoint_path: str = "checkpoints/residual_mlp.pt",
        device: str = "cpu",
        grid_spacing: float = 1.0,
        box_size: int = 25,
        pde_steps: int = 8,
        strict: bool = True,
        disable_residual_mlp: bool = True,
    ):
        self.device = str(device)
        self.strict = bool(strict)
        self.disable_residual_mlp = bool(disable_residual_mlp)

        # Enforce requirement: calibrated constants and trained checkpoint must exist
        if self.strict:
            if not os.path.exists(constants_path):
                raise RuntimeError(
                    f"Calibrated physical constants file '{constants_path}' not found. "
                    "Per SolvDock specification, please run Phase A calibration first:\n"
                    "  python -m solvdock.train.train_residual_mlp --phase A"
                )
            if not self.disable_residual_mlp and not os.path.exists(checkpoint_path):
                raise RuntimeError(
                    f"Trained residual MLP checkpoint '{checkpoint_path}' not found. "
                    "Per SolvDock specification, please run Phase B calibration first:\n"
                    "  python -m solvdock.train.train_residual_mlp --phase B"
                )

        # 1. Grid Engine
        self.grid_engine = SpatialGridEngine(
            grid_spacing=grid_spacing, box_size=box_size
        ).to(self.device)

        # 2. Calibrated Solvation PDE Solver
        self.pde_solver = SolvationPDESolver(
            grid_spacing=grid_spacing,
            steps=pde_steps,
            calibrated_constants_path=constants_path if os.path.exists(constants_path) else None,
            residual_mlp_path=checkpoint_path if (os.path.exists(checkpoint_path) and not self.disable_residual_mlp) else None,
            strict=self.strict and not self.disable_residual_mlp,
            disable_residual_mlp=self.disable_residual_mlp,
        ).to(self.device)

        gamma_rot = 0.50
        if os.path.exists(constants_path):
            import yaml
            with open(constants_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
                gamma_rot = float(cfg.get("gamma_rot", 0.50))

        # 3. Combined Potential
        self.potential = CombinedPotential(
            pde_solver=self.pde_solver,
            grid_engine=self.grid_engine,
            poisson_method="greens_function",
            gamma_rot=gamma_rot,
        ).to(self.device)

        # 4. Pose Optimizer
        self.optimizer = PoseOptimizer(self.potential, device=self.device)

    def _load_molecule(self, mol_input: Union[str, Chem.Mol], charge_model: str = "mmff94") -> Chem.Mol:
        """Helper to load a molecule from SMILES, file, or RDKit Mol."""
        if isinstance(mol_input, Chem.Mol):
            mol = Chem.Mol(mol_input)
        elif isinstance(mol_input, str):
            if os.path.exists(mol_input):
                ext = os.path.splitext(mol_input)[1].lower()
                if ext in (".sdf", ".mol"):
                    suppl = Chem.SDMolSupplier(mol_input, removeHs=False)
                    mol = suppl[0]
                elif ext == ".pdb":
                    mol = Chem.MolFromPDBFile(mol_input, removeHs=False)
                else:
                    mol = Chem.MolFromMolFile(mol_input, removeHs=False)
            else:
                # Interpret as SMILES string
                mol = Chem.MolFromSmiles(mol_input)
                if mol is not None:
                    mol = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(mol, randomSeed=42)
        else:
            raise ValueError(f"Unsupported molecule input: {mol_input}")

        if mol is None:
            raise ValueError(f"Failed to load molecule from {mol_input}")

        # Compute or preserve partial charges via unified charge engine
        from solvdock.core.charges import assign_charges, has_existing_charges
        if not has_existing_charges(mol):
            assign_charges(mol, scheme=charge_model)

        return mol

    def refine_pose(
        self,
        ligand_input: Union[str, Chem.Mol],
        pocket_input: Optional[Union[str, Chem.Mol]] = None,
        output_path: Optional[str] = None,
        max_steps: int = 20,
        lr: float = 0.05,
        charge_model: str = "mmff94",
    ) -> Dict[str, Any]:
        """Refines the ligand pose in the pocket binding site.

        Args:
            ligand_input: Ligand file path (SDF, PDB), SMILES string, or RDKit Mol.
            pocket_input: Optional receptor pocket file path (PDB, SDF) or RDKit Mol.
            output_path: Optional output file path (SDF or PDB) to save refined pose.
            max_steps: Maximum gradient descent iterations.
            lr: Learning rate for pose optimization.
            charge_model: Charge scheme ('mmff94', 'am1bcc', 'gasteiger', 'preserve').

        Returns:
            Dict containing:
                'mol': Refined RDKit Mol
                'delta_G_bind': Final predicted binding free energy (kcal/mol)
                'components': Dictionary of energy breakdown (E_LJ, E_Coulomb, ΔG_solv, etc.)
                'steps_taken': Number of steps executed before convergence
        """
        lig_mol = self._load_molecule(ligand_input, charge_model=charge_model)
        poc_mol = self._load_molecule(pocket_input, charge_model=charge_model) if pocket_input is not None else None

        result = self.optimizer.refine(
            initial_mol=lig_mol,
            pocket_mol=poc_mol,
            steps=max_steps,
            lr=lr,
        )

        # Write output file if requested
        if output_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            ext = os.path.splitext(output_path)[1].lower()
            if ext == ".pdb":
                with Chem.PDBWriter(output_path) as writer:
                    writer.write(result["mol"])
            else:
                with Chem.SDWriter(output_path) as writer:
                    writer.write(result["mol"])

        return result

    def dock_global(
        self,
        ligand_input: Union[str, Chem.Mol],
        pocket_input: Optional[Union[str, Chem.Mol]] = None,
        n_trials: int = 15,
        temperature: float = 300.0,
        box_radius: float = 8.0,
        step_size_trans: float = 2.0,
        step_size_rot: float = 0.5,
        step_size_dihedral: float = 0.5,
        local_steps: int = 8,
        local_lr: float = 0.05,
        rmsd_clustering_cutoff: float = 1.0,
        seed: Optional[int] = None,
        charge_model: str = "mmff94",
        output_path: Optional[str] = None,
    ) -> BasinHoppingResult:
        """Performs global stochastic basin-hopping docking over the molecular complex.

        Explores translational, rotational, and conformational space by coupling
        Metropolis-Hastings barrier jumping with local gradient descent on the full
        unrolled continuum solvation potential.

        Args:
            ligand_input: Ligand file path (SDF, PDB), SMILES string, or RDKit Mol.
            pocket_input: Optional receptor pocket file path (PDB, SDF) or RDKit Mol.
            n_trials: Number of outer basin-hopping Monte Carlo cycles.
            temperature: Simulation temperature in Kelvin (governs barrier crossing).
            box_radius: Half-width of search box around pocket center (Angstroms).
            step_size_trans: Maximum translational perturbation per trial (Angstroms).
            step_size_rot: Standard deviation of axis-angle rotation jump (radians).
            step_size_dihedral: Maximum torsion angle perturbation per trial (radians).
            local_steps: Inner-loop gradient descent iterations per trial.
            local_lr: Learning rate for inner-loop gradient descent.
            rmsd_clustering_cutoff: Heavy-atom RMSD cutoff for distinct docking modes (Angstroms).
            seed: Optional random seed for reproducible stochastic trajectories.
            charge_model: Charge scheme ('mmff94', 'am1bcc', 'gasteiger', 'preserve').
            output_path: Optional output file path (SDF or PDB) to save top docking pose.

        Returns:
            BasinHoppingResult containing best pose, free energy, mode clusters,
            and complete acceptance trajectory.
        """
        lig_mol = self._load_molecule(ligand_input, charge_model=charge_model)
        poc_mol = self._load_molecule(pocket_input, charge_model=charge_model) if pocket_input is not None else None

        engine = BasinHoppingDockingEngine(
            optimizer=self.optimizer,
            temperature=temperature,
            step_size_trans=step_size_trans,
            step_size_rot=step_size_rot,
            step_size_dihedral=step_size_dihedral,
            box_radius=box_radius,
            local_steps=local_steps,
            local_lr=local_lr,
            rmsd_clustering_cutoff=rmsd_clustering_cutoff,
            seed=seed,
        )

        result = engine.run(
            initial_mol=lig_mol,
            pocket_mol=poc_mol,
            n_trials=n_trials,
        )

        if output_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            ext = os.path.splitext(output_path)[1].lower()
            if ext == ".pdb":
                with Chem.PDBWriter(output_path) as writer:
                    writer.write(result.best_mol)
            else:
                with Chem.SDWriter(output_path) as writer:
                    writer.write(result.best_mol)

        return result
