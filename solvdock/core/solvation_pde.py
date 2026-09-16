"""Differentiable Ginzburg-Landau Mean-Field Solvation PDE Solver."""

import os
from typing import Dict, Optional, Tuple, Union
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from solvdock.core.residual_mlp import OrientationalCorrectionMLP


class SolvationPDESolver(nn.Module):
    """Solves the non-linear Ginzburg-Landau mean-field solvation PDE.

    Unrolls damped gradient relaxation updates on the solvent polarization
    field P(r) subject to an external electrostatic field E_ext(r) generated
    by solute charges, holding the solvent density rho(r) fixed at its
    Boltzmann excluded-volume profile.

    Free energy functional:
        Delta_G_solv = \\int [ -1/2 P \\cdot E_ext + k_B T rho ln(rho/rho_0)
                              + f_phi(rho, ||P||, ||nabla P||) ] d^3r
    """

    def __init__(
        self,
        grid_spacing: float = 1.0,
        steps: int = 8,
        dt: float = 0.1,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        cs2: Optional[float] = None,
        chi_e: Optional[float] = None,
        residual_mlp_path: Optional[Union[str, os.PathLike]] = None,
        calibrated_constants_path: Optional[Union[str, os.PathLike]] = "configs/calibrated_constants.yaml",
        strict: bool = True,
        disable_residual_mlp: bool = False,
        P_sat: Optional[float] = None,
        temperature: float = 298.15,
        bulk_density: float = 0.0333,  # molecules / A^3 (bulk water)
    ):
        super().__init__()
        self.grid_spacing = float(grid_spacing)
        self.steps = int(steps)
        self.dt = float(dt)
        self.strict = bool(strict)
        self.disable_residual_mlp = bool(disable_residual_mlp)
        self.temperature = float(temperature)
        self.kbT = 0.001987204 * self.temperature  # kcal/mol (0.592 at 298.15 K)
        self.bulk_density = float(bulk_density)

        # 1. Load physical constants (refusing silent placeholders if strict=True)
        loaded = {}
        if any(param is None for param in (alpha, beta, cs2, chi_e, P_sat)):
            loaded = self._load_calibrated_constants(calibrated_constants_path)
            alpha = alpha if alpha is not None else loaded.get("alpha")
            beta = beta if beta is not None else loaded.get("beta")
            cs2 = cs2 if cs2 is not None else loaded.get("cs2")
            chi_e = chi_e if chi_e is not None else loaded.get("chi_e")
            P_sat = P_sat if P_sat is not None else loaded.get("P_sat")

        self.P_sat = float(P_sat) if P_sat is not None else None

        if any(param is None for param in (alpha, beta, cs2, chi_e, P_sat)):
            if self.strict:
                raise ValueError(
                    "Calibrated physical constants (alpha, beta, cs2, chi_e, P_sat) are not provided "
                    f"and could not be found in '{calibrated_constants_path}'. "
                    "SolvDock requires calibrated constants from Step 3 Phase A. "
                    "Run 'python -m solvdock.train.train_residual_mlp --phase A' or pass explicit constants."
                )
            else:
                # Fallback only if strict=False
                alpha = alpha or 1.0
                beta = beta or 0.1
                cs2 = cs2 or 1.0
                chi_e = chi_e or 0.8
                self.P_sat = self.P_sat or 0.002

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.cs2 = float(cs2)
        self.chi_e = float(chi_e)

        # 2. Register fixed 7-point discrete 3D Laplacian kernel
        # Applied independently to each vector component of P (3 channels, groups=3)
        # Kernel shape: (3, 1, 3, 3, 3)
        dx = self.grid_spacing
        lapl = torch.zeros((3, 1, 3, 3, 3), dtype=torch.float32)
        for c in range(3):
            # Center point
            lapl[c, 0, 1, 1, 1] = -6.0 / (dx ** 2)
            # 6-face neighbors
            lapl[c, 0, 0, 1, 1] = 1.0 / (dx ** 2)
            lapl[c, 0, 2, 1, 1] = 1.0 / (dx ** 2)
            lapl[c, 0, 1, 0, 1] = 1.0 / (dx ** 2)
            lapl[c, 0, 1, 2, 1] = 1.0 / (dx ** 2)
            lapl[c, 0, 1, 1, 0] = 1.0 / (dx ** 2)
            lapl[c, 0, 1, 1, 2] = 1.0 / (dx ** 2)

        self.register_buffer("laplacian_kernel", lapl)

        # 3. Load OrientationalCorrectionMLP
        if residual_mlp_path is not None and os.path.exists(residual_mlp_path):
            self.residual_mlp = OrientationalCorrectionMLP.load_pretrained(
                residual_mlp_path, strict=self.strict
            )
        else:
            if self.strict and residual_mlp_path is not None:
                raise FileNotFoundError(
                    f"Residual MLP checkpoint '{residual_mlp_path}' does not exist. "
                    "Run 'python -m solvdock.train.train_residual_mlp --phase B' first."
                )
            # Standalone physics-only mode (used in Step 2 unit tests and Step 3 Phase A)
            self.residual_mlp = OrientationalCorrectionMLP(strict=False)
            self.residual_mlp.is_pretrained = False

    def _load_calibrated_constants(self, path: Optional[Union[str, os.PathLike]]) -> Dict[str, float]:
        """Loads physical constants from YAML file if it exists."""
        if path is not None and os.path.exists(path):
            with open(path, "r") as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict):
                    return data
        return {}

    def compute_laplacian(self, P: torch.Tensor) -> torch.Tensor:
        """Computes discrete 3D Laplacian ∇²P with zero-Dirichlet boundary conditions."""
        # P shape: (B, 3, D, H, W)
        return F.conv3d(P, self.laplacian_kernel, padding=1, groups=3)

    def compute_grad_norm(self, P: torch.Tensor) -> torch.Tensor:
        """Computes Frobenius norm ||∇P|| for vector field P."""
        # P shape: (B, 3, D, H, W)
        dx = self.grid_spacing
        grad_norm_sq = torch.zeros(
            (P.shape[0], 1, P.shape[2], P.shape[3], P.shape[4]),
            device=P.device,
            dtype=P.dtype,
        )

        for c in range(3):
            pc = P[:, c : c + 1]
            # Finite differences with zero-padding
            pad_z = F.pad(pc, (0, 0, 0, 0, 1, 1), mode="replicate")
            pad_y = F.pad(pc, (0, 0, 1, 1, 0, 0), mode="replicate")
            pad_x = F.pad(pc, (1, 1, 0, 0, 0, 0), mode="replicate")

            dz = (pad_z[:, :, 2:, :, :] - pad_z[:, :, :-2, :, :]) / (2.0 * dx)
            dy = (pad_y[:, :, :, 2:, :] - pad_y[:, :, :, :-2, :]) / (2.0 * dx)
            dx_comp = (pad_x[:, :, :, :, 2:] - pad_x[:, :, :, :, :-2]) / (2.0 * dx)

            grad_norm_sq = grad_norm_sq + dz ** 2 + dy ** 2 + dx_comp ** 2

        return torch.sqrt(grad_norm_sq + 1e-10)

    def forward(
        self,
        E_field: torch.Tensor,
        rho_solute: Optional[torch.Tensor] = None,
        v_steric: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward relaxation of polarization P and computation of ΔG_solv.

        Args:
            E_field: External electric field of shape (1, 3, D, H, W).
            rho_solute: Optional solvent density field of shape (1, 1, D, H, W).
            v_steric: Optional steric repulsion potential of shape (1, 1, D, H, W).

        Returns:
            delta_G_solv: Integrated solvation free energy scalar (kcal/mol).
            components: Dictionary containing detailed energy components
                        ('enthalpy', 'trans_entropy', 'orient_entropy', 'P_final').
        """
        B, C, D, H, W = E_field.shape
        assert C == 3, f"E_field must have 3 channels [Ex, Ey, Ez], got {C}"
        dx = self.grid_spacing
        dV = dx ** 3

        # Compute or assign solvent density rho(r)
        if rho_solute is not None:
            rho = rho_solute
        elif v_steric is not None:
            # Boltzmann excluded-volume distribution
            rho = self.bulk_density * torch.exp(-torch.clamp(v_steric, min=0.0) / self.kbT)
        else:
            # Default: solvent fills domain outside strong electric fields
            rho = torch.full((B, 1, D, H, W), self.bulk_density, device=E_field.device, dtype=E_field.dtype)

        # 1. Initialize P^(0) = 0
        P = torch.zeros_like(E_field)
        rho_norm = torch.clamp(rho / (self.bulk_density + 1e-12), min=0.0, max=1.0)

        # 2. Unroll Ginzburg-Landau relaxation updates
        # Physical coupling: water polarization is driven only where solvent density exists
        for _ in range(self.steps):
            lapl_P = self.compute_laplacian(P)
            P_norm_sq = torch.sum(P ** 2, dim=1, keepdim=True)  # (B, 1, D, H, W)
            damping = self.alpha * P + self.beta * P_norm_sq * P
            driving = self.chi_e * rho_norm * E_field
            dP_dt = self.cs2 * lapl_P - damping + driving
            P = P + self.dt * dP_dt

            # Physical dielectric saturation: ||P|| <= P_sat * (rho / rho_0)
            if self.P_sat is not None:
                P_norm = torch.sqrt(torch.sum(P ** 2, dim=1, keepdim=True) + 1e-10)
                scale = torch.clamp(self.P_sat * rho_norm / P_norm, max=1.0)
                P = P * scale

        # 3. Compute Free Energy Terms per Voxel
        # Enthalpy density: -1/2 P · E_ext
        enthalpy_density = -0.5 * torch.sum(P * E_field, dim=1, keepdim=True)

        # Translational entropy density: kB * T * rho * ln(rho / rho_0)
        safe_ratio = torch.clamp(rho / (self.bulk_density + 1e-12), min=1e-8, max=10.0)
        trans_entropy_density = self.kbT * rho * torch.log(safe_ratio)

        # Orientational entropy correction: f_phi(rho, ||P||, ||nabla P||)
        P_norm = torch.sqrt(torch.sum(P ** 2, dim=1, keepdim=True) + 1e-10)
        grad_P_norm = self.compute_grad_norm(P)
        features = torch.cat([rho, P_norm, grad_P_norm], dim=1)  # (B, 3, D, H, W)

        if self.residual_mlp.is_pretrained and not self.disable_residual_mlp:
            orient_entropy_density = self.residual_mlp(features)
        else:
            orient_entropy_density = torch.zeros_like(enthalpy_density)

        # Integrate over spatial grid: sum * dV
        enthalpy = torch.sum(enthalpy_density) * dV
        trans_entropy = torch.sum(trans_entropy_density) * dV
        orient_entropy = torch.sum(orient_entropy_density) * dV

        delta_G_solv = enthalpy + trans_entropy + orient_entropy

        components = {
            "enthalpy": enthalpy,
            "trans_entropy": trans_entropy,
            "orient_entropy": orient_entropy,
            "delta_G_solv": delta_G_solv,
            "P_final": P,
            "rho": rho,
        }

        return delta_G_solv, components


if __name__ == "__main__":
    print("Testing SolvationPDESolver...")
    # Instantiate physics-only solver with explicit constants
    solver = SolvationPDESolver(
        grid_spacing=1.0,
        steps=8,
        dt=0.1,
        alpha=1.0,
        beta=0.1,
        cs2=0.5,
        chi_e=0.8,
        strict=False,
    )

    # Electric field with requires_grad=True to test autograd backward pass
    E = torch.randn(1, 3, 15, 15, 15, requires_grad=True)
    dG, comp = solver(E)

    print(f"delta_G_solv: {dG.item():.4f} kcal/mol")
    print(f"Enthalpy: {comp['enthalpy'].item():.4f} kcal/mol")
    print(f"Translational entropy: {comp['trans_entropy'].item():.4f} kcal/mol")

    # Autograd backward test: ensures d(Delta_G_solv) / d(E) flows through all unrolled steps
    dG.backward()
    assert E.grad is not None, "Gradient did not flow through unrolled PDE solver!"
    grad_norm = torch.norm(E.grad).item()
    print(f"Gradient norm ||d(dG)/dE||: {grad_norm:.6f}")
    assert grad_norm > 0, "Gradient norm is zero!"

    print("SolvationPDESolver autograd and relaxation tests passed successfully!")
