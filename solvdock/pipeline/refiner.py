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

        # 3. Combined Potential
        self.potential = CombinedPotential(
            pde_solver=self.pde_solver,
            grid_engine=self.grid_engine,
            poisson_method="greens_function",
        ).to(self.device)

        # 4. Pose Optimizer
        self.optimizer = PoseOptimizer(self.potential, device=self.device)

    def _load_molecule(self, mol_input: Union[str, Chem.Mol]) -> Chem.Mol:
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
                    AllChem.ComputeGasteigerCharges(mol)
        else:
            raise ValueError(f"Unsupported molecule input: {mol_input}")

        if mol is None:
            raise ValueError(f"Failed to load molecule from {mol_input}")

        # Compute Gasteiger charges if not present
        needs_charges = any(not a.HasProp("_GasteigerCharge") for a in mol.GetAtoms())
        if needs_charges:
            AllChem.ComputeGasteigerCharges(mol)

        return mol

    def refine_pose(
        self,
        ligand_input: Union[str, Chem.Mol],
        pocket_input: Optional[Union[str, Chem.Mol]] = None,
        output_path: Optional[str] = None,
        max_steps: int = 20,
        lr: float = 0.05,
    ) -> Dict[str, Any]:
        """Refines the ligand pose in the pocket binding site.

        Args:
            ligand_input: Ligand file path (SDF, PDB), SMILES string, or RDKit Mol.
            pocket_input: Optional receptor pocket file path (PDB, SDF) or RDKit Mol.
            output_path: Optional output file path (SDF or PDB) to save refined pose.
            max_steps: Maximum gradient descent iterations.
            lr: Learning rate for pose optimization.

        Returns:
            Dict containing:
                'mol': Refined RDKit Mol
                'delta_G_bind': Final predicted binding free energy (kcal/mol)
                'components': Dictionary of energy breakdown (E_LJ, E_Coulomb, ΔG_solv, etc.)
                'steps_taken': Number of steps executed before convergence
        """
        lig_mol = self._load_molecule(ligand_input)
        poc_mol = self._load_molecule(pocket_input) if pocket_input is not None else None

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
