"""Unit and regression tests for macromolecular physics components:
topological exclusions, Debye-Hückel screening, SE(3) invariance, and flexible induced-fit relaxation
(evaluating static non-bonded potentials, strictly avoiding uncalibrated thermodynamic Delta G claims).
"""

import math
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from solvdock.core.topology import build_topological_scale_matrix, MolecularTopology
from solvdock.core.charges import assign_charges, get_partial_charges
from solvdock.core.grid_engine import SpatialGridEngine
from solvdock.core.solvation_pde import SolvationPDESolver
from solvdock.pipeline.energy import CombinedPotential
from solvdock.pipeline.flexible_refiner import FlexibleRefiner


def test_topological_exclusion_eliminates_steric_clash():
    """Validates that 1-2 and 1-3 topological exclusions eliminate false steric clashes from bonded angles."""
    # Propane: C1-C2 is 1.54 A (1-2), C1-C3 is ~2.5 A (1-3 angle)
    mol = Chem.MolFromSmiles("CCC")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.UFFOptimizeMolecule(mol)

    conf = mol.GetConformer()
    N = mol.GetNumAtoms()
    coords = torch.tensor([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in range(N)], dtype=torch.float32)
    charges = get_partial_charges(mol)
    z = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.int64)

    grid_engine = SpatialGridEngine(box_size=16, grid_spacing=1.0)
    pde_solver = SolvationPDESolver(grid_spacing=1.0, steps=2, strict=False, disable_residual_mlp=True)
    potential = CombinedPotential(pde_solver, grid_engine)

    # 1. Unmasked calculation (no topological exclusions)
    e_unmasked, lj_unmasked, _ = potential.compute_intramolecular_energy(coords, charges, z, topo_scale_matrix=None)

    # 2. Masked calculation (standard AMBER/MMFF94 1-2/1-3 exclusions and 1-4 scaling)
    scale_mat = build_topological_scale_matrix(mol)
    e_masked, lj_masked, _ = potential.compute_intramolecular_energy(coords, charges, z, topo_scale_matrix=scale_mat)

    # Without exclusions, 1-2 and 1-3 bonded pairs at ~1.5 A and ~2.5 A cause massive false LJ repulsion
    assert lj_unmasked > 100.0, f"Expected unmasked LJ to show massive repulsion, got {lj_unmasked:.2f} kcal/mol"
    # With exclusions, intramolecular non-bonded LJ is clean and free of bonded steric artifacts
    assert lj_masked < lj_unmasked, f"Masked LJ ({lj_masked:.2f}) must be substantially lower than unmasked ({lj_unmasked:.2f})"
    assert lj_masked < 10.0, f"Expected masked LJ to be reasonable (<10 kcal/mol), got {lj_masked:.2f}"


def test_debye_huckel_screening():
    """Validates that physiological ionic screening dampens Coulombic repulsion by exp(-kappa * r)."""
    grid_engine = SpatialGridEngine(box_size=16, grid_spacing=1.0)
    pde_solver = SolvationPDESolver(grid_spacing=1.0, steps=2, strict=False, disable_residual_mlp=True)

    # Two negative charges (-1.0e each) separated by 6.0 Angstroms
    coords1 = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    charges1 = torch.tensor([-1.0], dtype=torch.float32)
    z1 = torch.tensor([8], dtype=torch.int64)

    coords2 = torch.tensor([[6.0, 0.0, 0.0]], dtype=torch.float32)
    charges2 = torch.tensor([-1.0], dtype=torch.float32)
    z2 = torch.tensor([8], dtype=torch.int64)

    # Unscreened (vacuum/pure water without salt: debye_kappa = 0.0)
    pot_unscreened = CombinedPotential(pde_solver, grid_engine, debye_kappa=0.0)
    _, _, e_coulomb_unscreened = pot_unscreened.compute_direct_energy(coords1, charges1, z1, coords2, charges2, z2)

    # Screened at physiological salt (150 mM NaCl: kappa = 0.126 A^-1)
    kappa = 0.126
    pot_screened = CombinedPotential(pde_solver, grid_engine, debye_kappa=kappa)
    _, _, e_coulomb_screened = pot_screened.compute_direct_energy(coords1, charges1, z1, coords2, charges2, z2)

    expected_ratio = math.exp(-kappa * 6.0)
    actual_ratio = float((e_coulomb_screened / e_coulomb_unscreened).item())

    assert abs(actual_ratio - expected_ratio) < 1e-3, (
        f"Debye screening ratio {actual_ratio:.4f} did not match physical expectation {expected_ratio:.4f}"
    )


def test_intramolecular_energy_invariance_under_rigid_transform():
    """Validates that intramolecular non-bonded energy is strictly invariant under SE(3) rigid transforms."""
    mol = Chem.MolFromSmiles("CCCCCO")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    conf = mol.GetConformer()
    N = mol.GetNumAtoms()
    coords = torch.tensor([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in range(N)], dtype=torch.float32)
    charges = get_partial_charges(mol)
    z = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.int64)
    scale_mat = build_topological_scale_matrix(mol)

    grid_engine = SpatialGridEngine(box_size=16, grid_spacing=1.0)
    pde_solver = SolvationPDESolver(grid_spacing=1.0, steps=2, strict=False, disable_residual_mlp=True)
    potential = CombinedPotential(pde_solver, grid_engine)

    e_orig, _, _ = potential.compute_intramolecular_energy(coords, charges, z, topo_scale_matrix=scale_mat)

    # Apply rigid rotation (90 deg around z) + translation (+12.5, -4.2, +8.0)
    theta = torch.tensor(math.pi / 2.0)
    R = torch.tensor([
        [torch.cos(theta), -torch.sin(theta), 0.0],
        [torch.sin(theta),  torch.cos(theta), 0.0],
        [0.0,               0.0,              1.0],
    ])
    t = torch.tensor([12.5, -4.2, 8.0])
    coords_transformed = torch.matmul(coords, R.T) + t

    e_trans, _, _ = potential.compute_intramolecular_energy(coords_transformed, charges, z, topo_scale_matrix=scale_mat)

    assert abs(float((e_trans - e_orig).item())) < 1e-4, (
        f"Intramolecular energy changed under rigid transformation: delta = {abs(float(e_trans - e_orig)):.6f} kcal/mol"
    )


def test_flexible_refiner_induced_fit():
    """Validates that FlexibleRefiner relaxes pocket side chains while restraining backbone atoms."""
    # Synthetic pocket: 2 backbone atoms, 2 side-chain atoms
    # Backbone: atom 0, 1
    # Side-chain: atom 2, 3
    poc_mol = Chem.MolFromSmiles("CCCC")
    poc_mol = Chem.AddHs(poc_mol)
    AllChem.EmbedMolecule(poc_mol, randomSeed=42)

    # Ligand: methane
    lig_mol = Chem.MolFromSmiles("C")
    lig_mol = Chem.AddHs(lig_mol)
    AllChem.EmbedMolecule(lig_mol, randomSeed=42)

    # Place ligand close to side chain atom 3 to induce steric clash
    conf_poc = poc_mol.GetConformer()
    p3 = conf_poc.GetAtomPosition(3)
    conf_lig = lig_mol.GetConformer()
    # Move ligand right against atom 3 (1.2 A separation -> severe clash)
    conf_lig.SetAtomPosition(0, Chem.rdGeometry.Point3D(p3.x + 1.2, p3.y, p3.z))

    # Explicit backbone mask: atoms 0, 1 are backbone, 2, 3 are side chains
    bb_mask = torch.zeros(poc_mol.GetNumAtoms(), dtype=torch.bool)
    bb_mask[0] = True
    bb_mask[1] = True

    refiner = FlexibleRefiner(k_backbone=50.0, k_bond=100.0)
    res = refiner.refine_induced_fit(
        lig_mol, poc_mol, backbone_mask=bb_mask, freeze_backbone=True, max_steps=20, lr=0.05
    )

    # 1. Energy must decrease
    assert res["final_loss"] < res["initial_loss"], (
        f"Induced fit failed to reduce energy: init {res['initial_loss']:.2f} -> final {res['final_loss']:.2f}"
    )

    # 2. Frozen backbone must have exactly 0.0 RMSD displacement
    assert res["backbone_rmsd"] == 0.0, (
        f"Backbone RMSD ({res['backbone_rmsd']:.4f} A) must be exactly 0.0 under frozen scaffold"
    )

    # 3. Side chains must actively yield and displace
    assert res["sidechain_rmsd"] > 0.01, (
        f"Sidechain RMSD ({res['sidechain_rmsd']:.4f} A) did not respond to steric clash"
    )

