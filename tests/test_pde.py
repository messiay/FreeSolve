"""Unit tests for SolvationPDESolver and OrientationalCorrectionMLP."""

import pytest
import torch

from solvdock.core.solvation_pde import SolvationPDESolver
from solvdock.core.residual_mlp import OrientationalCorrectionMLP


def test_residual_mlp_strictness():
    """Validates that OrientationalCorrectionMLP refuses uncalibrated execution in strict mode."""
    mlp = OrientationalCorrectionMLP(strict=True)
    assert not mlp.is_pretrained

    dummy_x = torch.randn(1, 3, 10, 10, 10)
    with pytest.raises(RuntimeError) as exc_info:
        mlp(dummy_x)
    assert "running an untrained residual network at inference is prohibited" in str(exc_info.value)


def test_solvation_pde_autograd():
    """Validates unrolled Ginzburg-Landau relaxation and backward autograd gradient flow."""
    solver = SolvationPDESolver(
        grid_spacing=1.0,
        steps=6,
        dt=0.1,
        alpha=1.0,
        beta=0.05,
        cs2=0.5,
        chi_e=0.8,
        strict=False,
    )

    E = torch.randn(1, 3, 12, 12, 12, requires_grad=True)
    dG, comp = solver(E)

    assert dG.dim() == 0 or dG.numel() == 1
    assert "enthalpy" in comp
    assert "trans_entropy" in comp
    assert "orient_entropy" in comp

    dG.backward()
    assert E.grad is not None
    assert torch.norm(E.grad).item() > 0.0
