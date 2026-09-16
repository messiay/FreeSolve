"""Unit tests for SpatialGridEngine."""

import pytest
import torch

from solvdock.core.grid_engine import SpatialGridEngine


def test_grid_charge_conservation():
    """Validates Gaussian charge splatting conserves net charge to < 1e-3 e."""
    engine = SpatialGridEngine(grid_spacing=1.0, box_size=25, sigma=1.0)
    coords = torch.tensor([
        [10.0, 10.0, 10.0],
        [12.0, 11.5, 10.2],
        [8.5, 9.0, 11.0],
    ], dtype=torch.float32)
    charges = torch.tensor([0.45, -0.70, 0.25], dtype=torch.float32)

    grid = engine.deposit_charges(coords, charges)
    assert grid.shape == (1, 1, 25, 25, 25)

    dV = engine.grid_spacing ** 3
    net_charge = (grid.sum() * dV).item()
    target_charge = charges.sum().item()

    assert abs(net_charge - target_charge) < 1e-3


def test_grid_origin_centering():
    """Validates grid origin calculation centers coordinates."""
    engine = SpatialGridEngine(grid_spacing=1.0, box_size=25)
    coords = torch.tensor([
        [0.0, 0.0, 0.0],
        [10.0, 10.0, 10.0],
    ], dtype=torch.float32)
    origin = engine.get_grid_origin(coords)
    # Center is [5.0, 5.0, 5.0], half box is 12.5 -> origin should be [-7.5, -7.5, -7.5]
    assert torch.allclose(origin, torch.tensor([-7.5, -7.5, -7.5]))
