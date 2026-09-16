"""Unit tests for CombinedPotential and soft-core energy terms."""

import pytest
import torch

from solvdock.core.grid_engine import SpatialGridEngine
from solvdock.core.solvation_pde import SolvationPDESolver
from solvdock.pipeline.energy import CombinedPotential


def test_combined_potential_forward_and_backward():
    """Validates complete potential pipeline and autograd differentiability."""
    grid_engine = SpatialGridEngine(grid_spacing=1.0, box_size=15)
    pde_solver = SolvationPDESolver(
        grid_spacing=1.0,
        steps=5,
        dt=0.1,
        alpha=1.0,
        beta=0.05,
        cs2=0.5,
        chi_e=0.8,
        strict=False,
    )
    potential = CombinedPotential(pde_solver, grid_engine, poisson_method="greens_function", r_min=0.8)

    lig_coords = torch.tensor([[5.0, 5.0, 5.0], [6.2, 5.0, 5.0]], dtype=torch.float32, requires_grad=True)
    lig_charges = torch.tensor([0.2, -0.2], dtype=torch.float32)
    lig_z = torch.tensor([6, 8], dtype=torch.int64)

    poc_coords = torch.tensor([[8.0, 5.0, 5.0], [9.5, 5.0, 5.0]], dtype=torch.float32)
    poc_charges = torch.tensor([-0.3, 0.3], dtype=torch.float32)
    poc_z = torch.tensor([7, 6], dtype=torch.int64)

    energy, comp = potential(lig_coords, lig_charges, lig_z, poc_coords, poc_charges, poc_z)

    assert "total_energy" in comp
    assert "e_direct" in comp
    assert "delta_G_solv" in comp
    assert torch.isfinite(energy)

    energy.backward()
    assert lig_coords.grad is not None
    assert torch.norm(lig_coords.grad).item() > 0.0


def test_soft_core_clipping():
    """Validates that soft-core clipping prevents energy divergence at overlapping coords."""
    grid_engine = SpatialGridEngine(grid_spacing=1.0, box_size=15)
    pde_solver = SolvationPDESolver(
        grid_spacing=1.0, steps=2, alpha=1.0, beta=0.05, cs2=0.5, chi_e=0.8, strict=False
    )
    potential = CombinedPotential(pde_solver, grid_engine, r_min=0.8)

    # Overlapping atoms: r = 0.0
    c_lig = torch.tensor([[5.0, 5.0, 5.0]], dtype=torch.float32)
    c_poc = torch.tensor([[5.0, 5.0, 5.0]], dtype=torch.float32)
    q = torch.tensor([0.5], dtype=torch.float32)
    z = torch.tensor([6], dtype=torch.int64)

    e_dir, e_lj, e_coul = potential.compute_direct_energy(c_lig, q, z, c_poc, q, z)
    assert torch.isfinite(e_dir)
    assert not torch.isnan(e_dir)
    assert not torch.isinf(e_dir)


def test_binding_free_energy_decomposition():
    """Validates that CombinedPotential computes the exact 3-state MM/PBSA decomposition."""
    grid_engine = SpatialGridEngine(grid_spacing=1.0, box_size=15)
    pde_solver = SolvationPDESolver(
        grid_spacing=1.0, steps=3, dt=0.1, alpha=1.0, beta=0.05, cs2=0.5, chi_e=0.8, strict=False
    )
    potential = CombinedPotential(pde_solver, grid_engine)

    lig_coords = torch.tensor([[4.0, 5.0, 5.0], [5.2, 5.0, 5.0]], dtype=torch.float32, requires_grad=True)
    lig_charges = torch.tensor([0.2, -0.2], dtype=torch.float32)
    lig_z = torch.tensor([6, 8], dtype=torch.int64)

    poc_coords = torch.tensor([[8.0, 5.0, 5.0], [9.5, 5.0, 5.0]], dtype=torch.float32)
    poc_charges = torch.tensor([-0.3, 0.3], dtype=torch.float32)
    poc_z = torch.tensor([7, 6], dtype=torch.int64)

    # 1. Full 3-state calculation
    dG_bind, comp = potential(lig_coords, lig_charges, lig_z, poc_coords, poc_charges, poc_z)

    # Verify all 3 solvation states exist
    assert "dG_complex" in comp
    assert "dG_pocket" in comp
    assert "dG_ligand" in comp
    assert "ddG_solv" in comp
    assert "e_direct" in comp

    # Verify exact thermodynamic balance: ΔG_bind = E_direct + dG_complex - dG_pocket - dG_ligand
    expected_dG_bind = comp["e_direct"] + comp["dG_complex"] - comp["dG_pocket"] - comp["dG_ligand"]
    assert torch.allclose(dG_bind, expected_dG_bind, atol=1e-5)
    assert torch.allclose(comp["ddG_solv"], comp["dG_complex"] - comp["dG_pocket"] - comp["dG_ligand"], atol=1e-5)

    # 2. Verify precomputed pocket invariance
    dG_bind_precomputed, comp_pre = potential(
        lig_coords, lig_charges, lig_z, poc_coords, poc_charges, poc_z, dG_pocket=comp["dG_pocket"]
    )
    assert torch.allclose(dG_bind, dG_bind_precomputed, atol=1e-6)

    # 3. Verify backpropagation propagates gradients to ligand coordinates
    dG_bind.backward()
    assert lig_coords.grad is not None
    assert torch.norm(lig_coords.grad).item() > 0.0
