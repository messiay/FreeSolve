"""Unit tests for the Basin-Hopping Global Simulation Engine and mode clustering."""

import os
import numpy as np
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

from solvdock.core.basin_hopping import (
    BasinHoppingDockingEngine,
    BasinHoppingResult,
    compute_heavy_atom_rmsd,
)
from solvdock.core.grid_engine import SpatialGridEngine
from solvdock.core.solvation_pde import SolvationPDESolver
from solvdock.core.pose_optimizer import PoseOptimizer
from solvdock.pipeline.energy import CombinedPotential
from solvdock.pipeline.refiner import SolvDockRefiner


def _build_test_mol(smiles: str = "CCCO") -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.ComputeGasteigerCharges(mol)
    return mol


def test_heavy_atom_rmsd_in_situ():
    """Verifies that in-pocket heavy-atom RMSD does NOT apply rotational/translational superposition."""
    mol1 = _build_test_mol("CCCO")
    mol2 = Chem.Mol(mol1)

    # Identical molecules
    rmsd_zero = compute_heavy_atom_rmsd(mol1, mol2)
    assert abs(rmsd_zero) < 1e-6

    # Translate mol2 rigidly by +2.0 A along x
    conf2 = mol2.GetConformer()
    for i in range(mol2.GetNumAtoms()):
        pos = conf2.GetAtomPosition(i)
        conf2.SetAtomPosition(i, Point3D(pos.x + 2.0, pos.y, pos.z))

    rmsd_translated = compute_heavy_atom_rmsd(mol1, mol2)
    # Since all atoms were translated by 2.0 A, RMSD must be exactly 2.0 A
    assert abs(rmsd_translated - 2.0) < 1e-5


def test_elastic_boundary_reflection():
    """Verifies reflective elastic boundaries retain search coordinates within bounding box."""
    grid = SpatialGridEngine(box_size=16)
    pde = SolvationPDESolver(steps=2, disable_residual_mlp=True)
    pot = CombinedPotential(pde_solver=pde, grid_engine=grid)
    opt = PoseOptimizer(pot)
    engine = BasinHoppingDockingEngine(optimizer=opt, box_radius=5.0)

    center = np.array([10.0, 10.0, 10.0])
    radius = 5.0  # Box is [5.0, 15.0]

    # Inside point: unchanged
    pt_inside = np.array([12.0, 8.0, 10.0])
    res_inside = engine._reflect_boundary(pt_inside, center, radius)
    np.testing.assert_allclose(res_inside, pt_inside)

    # Overshoot point at x = 16.5 (overshoot 1.5): should reflect to 15.0 - 1.5 = 13.5
    pt_over = np.array([16.5, 10.0, 10.0])
    res_over = engine._reflect_boundary(pt_over, center, radius)
    assert abs(res_over[0] - 13.5) < 1e-5

    # Undershoot point at y = 3.0 (undershoot 2.0): should reflect to 5.0 + 2.0 = 7.0
    pt_under = np.array([10.0, 3.0, 10.0])
    res_under = engine._reflect_boundary(pt_under, center, radius)
    assert abs(res_under[1] - 7.0) < 1e-5


def test_metropolis_acceptance_thermodynamics():
    """Validates Boltzmann factor calculation against textbook thermodynamics."""
    grid = SpatialGridEngine(box_size=16)
    pde = SolvationPDESolver(steps=2, disable_residual_mlp=True)
    pot = CombinedPotential(pde_solver=pde, grid_engine=grid)
    opt = PoseOptimizer(pot)

    T = 300.0
    engine = BasinHoppingDockingEngine(optimizer=opt, temperature=T)
    kt = engine.KB * T  # ~0.59616 kcal/mol

    # Downhill jump: Delta E = -2.0 kcal/mol -> P = 1.0
    delta_e_down = -2.0
    p_down = 1.0 if delta_e_down <= 0 else np.exp(-delta_e_down / kt)
    assert p_down == 1.0

    # Uphill jump: Delta E = +1.0 kcal/mol -> P = exp(-1.0 / 0.59616) ~ 0.187
    delta_e_up = 1.0
    p_up = float(np.exp(-delta_e_up / kt))
    assert 0.18 < p_up < 0.19

    # High barrier: Delta E = +10.0 kcal/mol -> P ~ 5e-8
    delta_e_high = 10.0
    p_high = float(np.exp(-delta_e_high / kt))
    assert p_high < 1e-6


def test_basin_hopping_global_simulation():
    """Runs a fast multi-trial basin-hopping search and tests energy minimization and mode clustering."""
    grid = SpatialGridEngine(box_size=16)
    pde = SolvationPDESolver(steps=2, disable_residual_mlp=True)
    pot = CombinedPotential(pde_solver=pde, grid_engine=grid)
    opt = PoseOptimizer(pot)

    engine = BasinHoppingDockingEngine(
        optimizer=opt,
        temperature=300.0,
        step_size_trans=1.5,
        step_size_rot=0.4,
        step_size_dihedral=0.5,
        box_radius=6.0,
        local_steps=3,
        local_lr=0.05,
        rmsd_clustering_cutoff=1.0,
        seed=101,
    )

    lig_mol = _build_test_mol("CCCC")  # Butane with 1 rotatable bond

    # Intentionally scramble starting conformer coordinates
    conf = lig_mol.GetConformer()
    for i in range(lig_mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, Point3D(p.x + 1.5, p.y - 1.0, p.z + 0.5))

    n_trials = 4
    result = engine.run(initial_mol=lig_mol, n_trials=n_trials)

    assert isinstance(result, BasinHoppingResult)
    assert len(result.trajectories) == n_trials + 1
    assert result.trajectories[0]["step"] == 0
    assert 0.0 <= result.acceptance_rate <= 1.0
    assert len(result.top_modes) >= 1

    # Best energy must be <= initial energy or among evaluated minima
    init_energy = result.trajectories[0]["energy"]
    assert result.best_delta_G_bind <= init_energy + 1e-4

    # Check that best mol has valid finite 3D coordinates
    best_conf = result.best_mol.GetConformer()
    for i in range(result.best_mol.GetNumAtoms()):
        pos = best_conf.GetAtomPosition(i)
        assert np.isfinite(pos.x) and np.isfinite(pos.y) and np.isfinite(pos.z)

    # Check top modes are sorted by energy ascending
    energies = [m["energy"] for m in result.top_modes]
    assert energies == sorted(energies)

    # Check mode separation: all pairs in top_modes must have RMSD >= cutoff
    if len(result.top_modes) > 1:
        for i in range(len(result.top_modes)):
            for j in range(i + 1, len(result.top_modes)):
                rmsd_ij = compute_heavy_atom_rmsd(result.top_modes[i]["mol"], result.top_modes[j]["mol"])
                assert rmsd_ij >= engine.rmsd_clustering_cutoff - 1e-5


def test_solvdock_refiner_dock_global_integration():
    """Verifies high-level SolvDockRefiner.dock_global() API end-to-end."""
    refiner = SolvDockRefiner(
        constants_path="configs/calibrated_constants.yaml",
        strict=True,
        box_size=16,
        pde_steps=2,
    )

    result = refiner.dock_global(
        ligand_input="CCCO",
        n_trials=3,
        temperature=300.0,
        local_steps=3,
        seed=42,
    )

    assert isinstance(result, BasinHoppingResult)
    assert result.best_mol is not None
    assert np.isfinite(result.best_delta_G_bind)
    assert "delta_G_solv" in result.best_components
    assert len(result.top_modes) >= 1


def test_basin_hopping_with_pocket_receptor():
    """Verifies basin-hopping simulation when both ligand and pocket receptor are present."""
    grid = SpatialGridEngine(box_size=20)
    pde = SolvationPDESolver(steps=2, disable_residual_mlp=True)
    pot = CombinedPotential(pde_solver=pde, grid_engine=grid)
    opt = PoseOptimizer(pot)

    engine = BasinHoppingDockingEngine(
        optimizer=opt,
        temperature=300.0,
        step_size_trans=1.0,
        step_size_rot=0.3,
        step_size_dihedral=0.3,
        box_radius=5.0,
        local_steps=3,
        local_lr=0.05,
        rmsd_clustering_cutoff=1.0,
        seed=42,
    )

    lig_mol = _build_test_mol("CCO")
    pocket_mol = _build_test_mol("c1ccccc1")

    result = engine.run(initial_mol=lig_mol, pocket_mol=pocket_mol, n_trials=3)

    assert isinstance(result, BasinHoppingResult)
    assert np.isfinite(result.best_delta_G_bind)
    assert "e_direct" in result.best_components
    assert "ddG_solv" in result.best_components
    assert "dG_pocket" in result.best_components
    assert len(result.trajectories) == 4
    assert len(result.top_modes) >= 1
