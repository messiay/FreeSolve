"""Unit tests for FFT Poisson solver and electric field computation."""

import math
import pytest
import torch

from solvdock.core.poisson_solver import solve_poisson, compute_field


def test_poisson_coulomb_reproduction():
    """Validates that solve_poisson reproduces analytic 1/r Coulomb potential."""
    D, H, W = 25, 25, 25
    dx = 1.0
    eps0 = 1.0
    sigma = 1.0

    grid = torch.zeros(1, 1, D, H, W, dtype=torch.float32)
    cx, cy, cz = W // 2, H // 2, D // 2

    z, y, x = torch.meshgrid(torch.arange(D), torch.arange(H), torch.arange(W), indexing="ij")
    r2 = ((z - cz)**2 + (y - cy)**2 + (x - cx)**2).float()
    rho = torch.exp(-r2 / (2.0 * sigma**2))
    rho = rho / (rho.sum() * (dx**3))  # Net charge = 1.0 e
    grid[0, 0] = rho

    phi = solve_poisson(grid, grid_spacing=dx, epsilon_0=eps0)
    assert phi.shape == (1, 1, D, H, W)

    # Test potential at r = 5.0 A along x-axis
    r_test = 5.0
    analytic_val = 1.0 / (4.0 * math.pi * eps0 * r_test)
    numerical_val = phi[0, 0, cz, cy, cx + int(r_test)].item()

    rel_error = abs(numerical_val - analytic_val) / analytic_val
    assert rel_error < 0.05, f"Coulomb potential relative error {rel_error:.2%} exceeds 5%!"


def test_compute_field_directions():
    """Validates that central-difference electric field points radially away from positive charge."""
    D, H, W = 21, 21, 21
    dx = 1.0
    cx, cy, cz = W // 2, H // 2, D // 2

    grid = torch.zeros(1, 1, D, H, W, dtype=torch.float32)
    z, y, x = torch.meshgrid(torch.arange(D), torch.arange(H), torch.arange(W), indexing="ij")
    r2 = ((z - cz)**2 + (y - cy)**2 + (x - cx)**2).float()
    rho = torch.exp(-r2 / 2.0)
    grid[0, 0] = rho / (rho.sum() * (dx**3))

    phi = solve_poisson(grid, grid_spacing=dx, epsilon_0=1.0)
    E = compute_field(phi, grid_spacing=dx)

    assert E.shape == (1, 3, D, H, W)
    # Along X-axis: E_x > 0 for x > cx and E_x < 0 for x < cx
    assert E[0, 0, cz, cy, cx + 2].item() > 0
    assert E[0, 0, cz, cy, cx - 2].item() < 0
    # Along Y-axis: E_y > 0 for y > cy and E_y < 0 for y < cy
    assert E[0, 1, cz, cy + 2, cx].item() > 0
    assert E[0, 1, cz, cy - 2, cx].item() < 0
    # Along Z-axis: E_z > 0 for z > cz and E_z < 0 for z < cz
    assert E[0, 2, cz + 2, cy, cx].item() > 0
    assert E[0, 2, cz - 2, cy, cx].item() < 0
