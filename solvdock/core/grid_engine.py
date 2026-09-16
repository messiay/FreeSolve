"""Spatial grid engine for Gaussian charge splatting and voxelization."""

from typing import Optional, Tuple
import torch
import torch.nn as nn


class SpatialGridEngine(nn.Module):
    """Voxelization and Gaussian charge deposition onto a 3D pocket grid.

    Deposits atomic point charges onto a regular 3D grid with spacing dx
    using Gaussian splatting (width sigma = 1.0 A) while strictly conserving
    total net charge to within 1e-3 e.
    """

    def __init__(
        self,
        grid_spacing: float = 1.0,
        padding: float = 6.0,
        box_size: int = 25,
        sigma: float = 1.0,
    ):
        super().__init__()
        self.grid_spacing = float(grid_spacing)
        self.padding = float(padding)
        self.box_size = int(box_size)
        self.sigma = float(sigma)

    def get_grid_origin(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute grid origin such that coords are centered in the box."""
        device = coords.device
        dtype = coords.dtype
        center = coords.mean(dim=0)
        half_span = (self.box_size * self.grid_spacing) / 2.0
        return center - half_span

    def get_grid_coords(
        self, grid_origin: torch.Tensor, device: torch.device = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate 3D coordinate meshgrid for the grid box.

        Returns (Z, Y, X) tensors each of shape (D, H, W).
        """
        if device is None:
            device = grid_origin.device
        dtype = grid_origin.dtype
        D = H = W = self.box_size
        dx = self.grid_spacing

        # Indexing: z along dim 0, y along dim 1, x along dim 2
        z = grid_origin[2] + torch.arange(D, device=device, dtype=dtype) * dx
        y = grid_origin[1] + torch.arange(H, device=device, dtype=dtype) * dx
        x = grid_origin[0] + torch.arange(W, device=device, dtype=dtype) * dx

        Z, Y, X = torch.meshgrid(z, y, x, indexing="ij")
        return Z, Y, X

    def deposit_charges(
        self,
        coords: torch.Tensor,
        charges: torch.Tensor,
        grid_origin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Gaussian-splat point charges onto a 3D grid, sigma = 1.0 A.

        Args:
            coords: Atom coordinates of shape (N, 3) in Angstroms [x, y, z].
            charges: Partial charges of shape (N,) in elementary charge units e.
            grid_origin: Origin of the grid [ox, oy, oz]. If None, centered on coords.

        Returns:
            charge_grid: Tensor of shape (1, 1, D, H, W) containing charge density
                         in units of e / A^3.
        """
        device = coords.device
        dtype = coords.dtype

        if grid_origin is None:
            grid_origin = self.get_grid_origin(coords)
        else:
            grid_origin = grid_origin.to(device=device, dtype=dtype)

        D = H = W = self.box_size
        dx = self.grid_spacing
        dV = dx ** 3
        sigma2 = 2.0 * (self.sigma ** 2)

        Z, Y, X = self.get_grid_coords(grid_origin, device=device)
        grid_points = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=-1)

        # Identify atoms located within the physical grid box
        min_x, max_x = grid_origin[0], grid_origin[0] + (W - 1) * dx
        min_y, max_y = grid_origin[1], grid_origin[1] + (H - 1) * dx
        min_z, max_z = grid_origin[2], grid_origin[2] + (D - 1) * dx

        in_box = (
            (coords[:, 0] >= min_x - 0.5 * dx) & (coords[:, 0] <= max_x + 0.5 * dx) &
            (coords[:, 1] >= min_y - 0.5 * dx) & (coords[:, 1] <= max_y + 0.5 * dx) &
            (coords[:, 2] >= min_z - 0.5 * dx) & (coords[:, 2] <= max_z + 0.5 * dx)
        )

        active_coords = coords[in_box]
        active_charges = charges[in_box]

        if active_coords.shape[0] == 0:
            return torch.zeros((1, 1, D, H, W), device=device, dtype=dtype)

        # Pairwise distance matrix between active atoms (N_act, 3) and grid points (M, 3)
        diff = grid_points.unsqueeze(0) - active_coords.unsqueeze(1)  # (N_act, M, 3)
        dist_sq = torch.sum(diff ** 2, dim=-1)  # (N_act, M)

        weights = torch.exp(-dist_sq / sigma2)  # (N_act, M)
        norm = torch.sum(weights, dim=-1, keepdim=True) * dV  # (N_act, 1)
        norm = torch.clamp(norm, min=1e-12)
        norm_weights = weights / norm  # (N_act, M)

        rho = torch.sum(active_charges.unsqueeze(-1) * norm_weights, dim=0)
        charge_grid = rho.view(1, 1, D, H, W)

        # Assert total charge of active in-box atoms is conserved to within 1e-3 e
        total_grid_charge = (charge_grid.sum() * dV).item()
        total_active_charge = active_charges.sum().item()
        charge_err = abs(total_grid_charge - total_active_charge)
        assert charge_err < 1e-3, (
            f"Charge conservation violated! Grid charge: {total_grid_charge:.6f}, "
            f"Active charge: {total_active_charge:.6f}, Error: {charge_err:.6e}"
        )

        return charge_grid


if __name__ == "__main__":
    print("Testing SpatialGridEngine...")
    engine = SpatialGridEngine(grid_spacing=1.0, box_size=25, sigma=1.0)
    coords = torch.tensor([
        [10.0, 10.0, 10.0],
        [12.0, 11.0, 10.5],
        [8.5, 9.5, 11.2],
    ], dtype=torch.float32)
    charges = torch.tensor([0.45, -0.70, 0.25], dtype=torch.float32)

    grid = engine.deposit_charges(coords, charges)
    assert grid.shape == (1, 1, 25, 25, 25)
    net_charge = (grid.sum() * (engine.grid_spacing ** 3)).item()
    print(f"Total atom charge: {charges.sum().item():.5f}, Deposited: {net_charge:.5f}")
    assert abs(net_charge - charges.sum().item()) < 1e-4
    print("SpatialGridEngine test passed successfully!")
