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


def test_salt_bridge_physical_desolvation_penalty():
    """Validates physical sign/magnitude laws on a salt bridge toy system:
    1. Isolated ions must have negative solvation free energies (stabilized by dielectric solvent).
    2. Bound contact dipole must be less solvated than two separated ions (|dG_complex| < |dG_poc| + |dG_lig|).
    3. Net electrostatic desolvation ddG_solv must be strictly POSITIVE (energetic desolvation penalty).
    """
    grid_engine = SpatialGridEngine(grid_spacing=1.0, box_size=25)
    pde_solver = SolvationPDESolver(
        grid_spacing=1.0, steps=5, dt=0.1, alpha=1.0, beta=0.05, cs2=0.5, chi_e=0.8, strict=False
    )
    potential = CombinedPotential(pde_solver, grid_engine)

    # Cation (+0.5e) and Anion (-0.5e) at 2.5 A contact separation
    lig_coords = torch.tensor([[11.0, 12.0, 12.0]], dtype=torch.float32)
    lig_q = torch.tensor([0.5], dtype=torch.float32)
    lig_z = torch.tensor([7], dtype=torch.int64)

    poc_coords = torch.tensor([[13.5, 12.0, 12.0]], dtype=torch.float32)
    poc_q = torch.tensor([-0.5], dtype=torch.float32)
    poc_z = torch.tensor([8], dtype=torch.int64)

    dG_bind, comp = potential(lig_coords, lig_q, lig_z, poc_coords, poc_q, poc_z)

    # 1. Negative solvation of isolated charges
    assert comp["dG_ligand"].item() < 0.0, "Isolated ligand cation must have negative solvation free energy"
    assert comp["dG_pocket"].item() < 0.0, "Isolated pocket anion must have negative solvation free energy"

    # 2. Bound dipole complex magnitude is smaller than sum of isolated monopoles
    sum_unbound = abs(comp["dG_ligand"].item()) + abs(comp["dG_pocket"].item())
    assert abs(comp["dG_complex"].item()) < sum_unbound, "Contact dipole must generate less total polarization than separated ions"

    # 3. Desolvation cost must be strictly positive (penalty)
    assert comp["ddG_solv"].item() > 0.0, "Desolvating opposite charges to form a contact pair must carry a positive penalty"


def test_conformational_entropy_penalty():
    """Validates that conformational entropy:
    1. Adds exactly gamma_rot * N_rot to the binding free energy (destabilizing / positive penalty).
    2. Does not alter coordinate gradients (dL/dx is invariant to constant entropy offset).
    """
    grid_engine = SpatialGridEngine(grid_spacing=1.0, box_size=15)
    pde_solver = SolvationPDESolver(
        grid_spacing=1.0, steps=3, dt=0.1, alpha=1.0, beta=0.05, cs2=0.5, chi_e=0.8, strict=False
    )
    gamma_rot = 0.50
    potential = CombinedPotential(pde_solver, grid_engine, gamma_rot=gamma_rot)

    c_lig_0 = torch.tensor([[4.0, 5.0, 5.0], [5.2, 5.0, 5.0]], dtype=torch.float32, requires_grad=True)
    c_lig_4 = torch.tensor([[4.0, 5.0, 5.0], [5.2, 5.0, 5.0]], dtype=torch.float32, requires_grad=True)
    q = torch.tensor([0.2, -0.2], dtype=torch.float32)
    z = torch.tensor([6, 8], dtype=torch.int64)

    c_poc = torch.tensor([[8.0, 5.0, 5.0], [9.5, 5.0, 5.0]], dtype=torch.float32)
    q_poc = torch.tensor([-0.3, 0.3], dtype=torch.float32)
    z_poc = torch.tensor([7, 6], dtype=torch.int64)

    # 1. Evaluate with 0 rotatable bonds
    dG_0, comp_0 = potential(c_lig_0, q, z, c_poc, q_poc, z_poc, num_rotatable_bonds=0)
    dG_0.backward()

    # 2. Evaluate with 4 rotatable bonds
    dG_4, comp_4 = potential(c_lig_4, q, z, c_poc, q_poc, z_poc, num_rotatable_bonds=4)
    dG_4.backward()

    # Verify exact scalar offset
    expected_offset = 4 * gamma_rot  # 2.0 kcal/mol
    assert torch.allclose(comp_4["delta_G_rot"], torch.tensor(expected_offset), atol=1e-5)
    assert torch.allclose(dG_4 - dG_0, torch.tensor(expected_offset), atol=1e-5)

    # Verify exact gradient invariance
    assert torch.allclose(c_lig_0.grad, c_lig_4.grad, atol=1e-6), "Conformational entropy must not distort pose gradients"


