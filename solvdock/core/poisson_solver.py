"""FFT-based Poisson solver for electrostatic potential and electric field."""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def solve_poisson(
    charge_grid: torch.Tensor,
    grid_spacing: float = 1.0,
    epsilon_0: float = 1.0 / (4.0 * math.pi * 332.0637),
    method: str = "greens_function",
    ionic_strength: float = 0.0,
    dielectric_constant: float = 78.4,
) -> torch.Tensor:
    """Solves Poisson's or Linearized Poisson-Boltzmann equation on a 3D grid via FFT.

    When ionic_strength == 0.0 (default):
        ∇²Φ = -ρ_q / ε₀ (Standard Poisson equation in open boundary conditions).

    When ionic_strength > 0.0:
        ∇²Φ - κ²Φ = -ρ_q / ε₀ (Linearized Poisson-Boltzmann with Debye-Hückel screening).
        κ = sqrt(8 * pi * N_A * e^2 * I / (1000 * 4*pi*eps0 * eps_r * k_B * T)).
        Screened Green's function: G(r) = exp(-κ * r) / (4πε₀r).

    Args:
        charge_grid: Tensor of shape (1, 1, D, H, W) containing charge density ρ_q.
        grid_spacing: Grid spacing Δx in Angstroms.
        epsilon_0: Dielectric permittivity of vacuum (in consistent simulation units).
        method: Solver formulation ('greens_function' or 'discrete_laplacian').
        ionic_strength: Salt concentration / ionic strength in Molar (e.g. 0.15 for physiological saline).
        dielectric_constant: Solvent relative permittivity ε_r for Debye length calculation (default 78.4).

    Returns:
        phi: Electrostatic potential Φ of shape (1, 1, D, H, W).
    """
    if charge_grid.dim() != 5:
        raise ValueError(f"charge_grid must have shape (B, C, D, H, W), got {charge_grid.shape}")

    B, C, D, H, W = charge_grid.shape
    dx = float(grid_spacing)
    dV = dx ** 3
    eps0 = float(epsilon_0)

    if method == "greens_function":
        # Hockney-Eastwood method for open boundary Poisson solve
        # Pad grid by D, H, W on the right (size doubled in each dimension)
        pD, pH, pW = 2 * D, 2 * H, 2 * W
        padded_rho = F.pad(charge_grid, (0, W, 0, H, 0, D), mode="constant", value=0.0)

        device = charge_grid.device
        dtype = charge_grid.dtype

        # Coordinate axes for Green's function with periodic wrapping
        z_coords = torch.cat([torch.arange(0, D, device=device, dtype=dtype),
                              torch.arange(-D, 0, device=device, dtype=dtype)]) * dx
        y_coords = torch.cat([torch.arange(0, H, device=device, dtype=dtype),
                              torch.arange(-H, 0, device=device, dtype=dtype)]) * dx
        x_coords = torch.cat([torch.arange(0, W, device=device, dtype=dtype),
                              torch.arange(-W, 0, device=device, dtype=dtype)]) * dx

        Z, Y, X = torch.meshgrid(z_coords, y_coords, x_coords, indexing="ij")
        R = torch.sqrt(Z ** 2 + Y ** 2 + X ** 2)

        # Compute inverse Debye length kappa (in 1/Angstrom)
        # kappa = sqrt(8 * pi * N_A * e^2 * I / (1000 * 4*pi*eps0 * eps_r * k_B * T))
        if ionic_strength > 0.0:
            k_b_t = 0.001987204 * 298.15  # kcal/mol
            n_a_per_a3 = 6.02214076e-4    # molecules / Angstrom^3 per Molar
            coulomb_const = 332.0637      # kcal * A / mol
            kappa2 = (8.0 * math.pi * n_a_per_a3 * coulomb_const / (float(dielectric_constant) * k_b_t)) * float(ionic_strength)
            kappa = math.sqrt(max(kappa2, 0.0))
        else:
            kappa = 0.0

        # Free space or Debye-Huckel screened Green's function
        G = torch.zeros_like(R)
        nonzero_mask = R > 0
        r_eff = dx * ((3.0 / (4.0 * math.pi)) ** (1.0 / 3.0))

        if kappa > 0.0:
            screening = torch.exp(-kappa * R[nonzero_mask])
            G[nonzero_mask] = screening / (4.0 * math.pi * eps0 * R[nonzero_mask])
            # Attenuate self-interaction regularizer by average screening inside voxel
            self_factor = max(1.0 - 0.375 * kappa * r_eff, 0.20)
            G[~nonzero_mask] = (3.0 / (8.0 * math.pi * eps0 * r_eff)) * self_factor
        else:
            G[nonzero_mask] = 1.0 / (4.0 * math.pi * eps0 * R[nonzero_mask])
            G[~nonzero_mask] = 3.0 / (8.0 * math.pi * eps0 * r_eff)

        # 3D FFT convolution
        G_tensor = G.unsqueeze(0).unsqueeze(0)  # (1, 1, pD, pH, pW)
        rho_hat = torch.fft.fftn(padded_rho, dim=(-3, -2, -1))
        G_hat = torch.fft.fftn(G_tensor, dim=(-3, -2, -1))
        phi_padded = torch.fft.ifftn(rho_hat * G_hat, dim=(-3, -2, -1)).real * dV

        # Crop back to original domain
        phi = phi_padded[:, :, :D, :H, :W]
        return phi

    elif method == "discrete_laplacian":
        if ionic_strength > 0.0:
            k_b_t = 0.001987204 * 298.15
            n_a_per_a3 = 6.02214076e-4
            coulomb_const = 332.0637
            kappa2 = (8.0 * math.pi * n_a_per_a3 * coulomb_const / (float(dielectric_constant) * k_b_t)) * float(ionic_strength)
        else:
            kappa2 = 0.0

        # Discrete Laplacian eigenvalue inversion with zero-padding
        # Pad by 1 box width on both sides: shape 3D x 3H x 3W
        padded_rho = F.pad(charge_grid, (W, W, H, H, D, D), mode="constant", value=0.0)
        pD, pH, pW = padded_rho.shape[-3:]

        kz = 2.0 * math.pi * torch.fft.fftfreq(pD, d=dx).to(charge_grid.device)
        ky = 2.0 * math.pi * torch.fft.fftfreq(pH, d=dx).to(charge_grid.device)
        kx = 2.0 * math.pi * torch.fft.rfftfreq(pW, d=dx).to(charge_grid.device)
        Kz, Ky, Kx = torch.meshgrid(kz, ky, kx, indexing="ij")

        # Discrete 7-point Laplacian eigenvalues: 2*(cos(k*dx) - 1)/dx^2
        denom = (
            2.0 * (torch.cos(Kz * dx) - 1.0)
            + 2.0 * (torch.cos(Ky * dx) - 1.0)
            + 2.0 * (torch.cos(Kx * dx) - 1.0)
        ) / (dx ** 2)

        # (∇² - κ²)Φ = -ρ/ε₀ => (denom - κ²)Φ_hat = -ρ_hat / ε₀ => Φ_hat = ρ_hat / (ε₀ * (κ² - denom))
        rho_hat = torch.fft.rfftn(padded_rho, dim=(-3, -2, -1))
        phi_hat = torch.zeros_like(rho_hat)

        effective_denom = kappa2 - denom  # Always >= 0 since denom <= 0 and kappa2 >= 0
        mask = effective_denom > 1e-12
        phi_hat[..., mask] = (rho_hat[..., mask] / eps0) / effective_denom[mask]
        phi_hat[..., 0, 0, 0] = 0.0

        phi_padded = torch.fft.irfftn(phi_hat, s=(pD, pH, pW), dim=(-3, -2, -1))
        phi = phi_padded[:, :, D:2*D, H:2*H, W:2*W]
        return phi

    else:
        raise ValueError(f"Unknown Poisson solve method: {method}")


def compute_field(phi: torch.Tensor, grid_spacing: float = 1.0) -> torch.Tensor:
    """Computes electric field E(r) = -∇Φ(r) via 3D central differences.

    Uses a fixed, non-trainable 3D central difference convolution kernel
    with zero padding.

    Args:
        phi: Potential tensor of shape (1, 1, D, H, W).
        grid_spacing: Grid spacing Δx in Angstroms.

    Returns:
        E_field: Electric field of shape (1, 3, D, H, W) where channels
                 correspond to [E_x, E_y, E_z].
    """
    if phi.dim() != 5:
        raise ValueError(f"phi must have shape (B, 1, D, H, W), got {phi.shape}")

    dx = float(grid_spacing)
    device = phi.device
    dtype = phi.dtype

    # Fixed central-difference gradient kernels of shape (3, 1, 3, 3, 3)
    # E_x = - dPhi/dx, E_y = - dPhi/dy, E_z = - dPhi/dz
    # Dim 2 is Z, Dim 3 is Y, Dim 4 is X
    kernel = torch.zeros((3, 1, 3, 3, 3), device=device, dtype=dtype)

    # d/dx: along X axis (dim 4 of 5D tensor, index 2 of 3x3x3 kernel)
    # Central difference: (f[x+1] - f[x-1]) / (2*dx)
    # -d/dx: -(f[x+1] - f[x-1])/(2dx) = (f[x-1] - f[x+1])/(2dx)
    kernel[0, 0, 1, 1, 0] = 1.0 / (2.0 * dx)   # x - 1
    kernel[0, 0, 1, 1, 2] = -1.0 / (2.0 * dx)  # x + 1

    # d/dy: along Y axis (dim 3 of 5D tensor, index 1 of 3x3x3 kernel)
    kernel[1, 0, 1, 0, 1] = 1.0 / (2.0 * dx)   # y - 1
    kernel[1, 0, 1, 2, 1] = -1.0 / (2.0 * dx)  # y + 1

    # d/dz: along Z axis (dim 2 of 5D tensor, index 0 of 3x3x3 kernel)
    kernel[2, 0, 0, 1, 1] = 1.0 / (2.0 * dx)   # z - 1
    kernel[2, 0, 2, 1, 1] = -1.0 / (2.0 * dx)  # z + 1

    # Compute electric field components simultaneously
    E_field = F.conv3d(phi, kernel, padding=1)
    return E_field


if __name__ == "__main__":
    print("Testing Poisson Solver and Field Computation...")
    D, H, W = 25, 25, 25
    dx = 1.0
    eps0 = 1.0

    # Create point charge splatted as Gaussian blob at center
    sigma = 1.0
    grid = torch.zeros(1, 1, D, H, W, dtype=torch.float32)
    cx, cy, cz = W // 2, H // 2, D // 2

    z, y, x = torch.meshgrid(torch.arange(D), torch.arange(H), torch.arange(W), indexing="ij")
    r2 = ((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2).float()
    rho = torch.exp(-r2 / (2.0 * (sigma ** 2)))
    rho = rho / (rho.sum() * (dx ** 3))  # Total charge = 1.0 e
    grid[0, 0] = rho

    phi = solve_poisson(grid, grid_spacing=dx, epsilon_0=eps0)
    assert phi.shape == (1, 1, D, H, W), f"Expected (1, 1, 25, 25, 25), got {phi.shape}"

    # Unit test: compare numerical Φ against analytic 1/r Coulomb potential at r = 5.0 A
    r_test = 5.0
    analytic_phi = 1.0 / (4.0 * math.pi * eps0 * r_test)
    numerical_phi = phi[0, 0, cz, cy, cx + int(r_test)].item()
    rel_error = abs(numerical_phi - analytic_phi) / analytic_phi
    print(f"r = {r_test} A -> Analytic: {analytic_phi:.6f}, Numerical: {numerical_phi:.6f}, Rel err: {rel_error:.2%}")
    assert rel_error < 0.05, f"Relative error {rel_error:.2%} exceeds 5% threshold!"

    # Test electric field computation
    E = compute_field(phi, grid_spacing=dx)
    assert E.shape == (1, 3, D, H, W), f"Expected (1, 3, 25, 25, 25), got {E.shape}"
    # Electric field should point radially outward from positive charge (+x has Ex > 0, -x has Ex < 0)
    Ex_pos = E[0, 0, cz, cy, cx + 3].item()
    Ex_neg = E[0, 0, cz, cy, cx - 3].item()
    assert Ex_pos > 0 and Ex_neg < 0, f"Field direction inverted: Ex_pos={Ex_pos}, Ex_neg={Ex_neg}"

    print("Poisson solver and electric field tests passed successfully!")
