"""Orientational entropy residual neural network."""

import os
from typing import Optional, Union
import torch
import torch.nn as nn


class OrientationalCorrectionMLP(nn.Module):
    """Small per-voxel Delta-learning network for orientational entropy correction.

    Predicts the local orientational entropy deficit from voxel features:
    [rho, ||P||, ||nabla P||].

    Parameters: < 2000.
    Enforces strict calibration tracking: raises RuntimeError if used untrained
    with strict=True.
    """

    def __init__(self, strict: bool = True):
        super().__init__()
        self.strict = bool(strict)
        self.is_pretrained = False

        # Architecture: Linear(3, 32) -> SiLU -> Linear(32, 32) -> SiLU -> Linear(32, 1)
        # Total parameters: (3*32+32) + (32*32+32) + (32*1+1) = 128 + 1056 + 33 = 1217 (< 2000)
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass for orientational entropy prediction.

        Args:
            features: Tensor of shape (B, 3, D, H, W) or (N, 3) with channels
                      [rho, ||P||, ||nabla P||].

        Returns:
            correction: Tensor of shape (B, 1, D, H, W) or (N, 1) in kcal/(mol * A^3).
        """
        if self.strict and not self.is_pretrained:
            raise RuntimeError(
                "OrientationalCorrectionMLP has not been loaded with pretrained weights. "
                "Per SolvDock specification, running an untrained residual network at inference "
                "is prohibited. Please run Step 3 Phase B calibration or instantiate with strict=False."
            )

        if features.dim() == 5:
            # (B, 3, D, H, W) -> (B, D, H, W, 3)
            B, C, D, H, W = features.shape
            assert C == 3, f"Expected 3 channels, got {C}"
            perm = features.permute(0, 2, 3, 4, 1).contiguous()
            flat = perm.view(-1, 3)
            out_flat = self.net(flat)
            out = out_flat.view(B, D, H, W, 1).permute(0, 4, 1, 2, 3).contiguous()
            return out
        elif features.dim() == 2:
            assert features.shape[1] == 3, f"Expected (N, 3), got {features.shape}"
            return self.net(features)
        else:
            raise ValueError(f"Unsupported features tensor shape: {features.shape}")

    @classmethod
    def load_pretrained(
        cls,
        path: Union[str, os.PathLike],
        strict: bool = True,
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> "OrientationalCorrectionMLP":
        """Loads a pretrained model checkpoint and marks it as verified."""
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Checkpoint file not found at '{path}'. Please train the residual MLP first "
                "(python -m solvdock.train.train_residual_mlp --phase B)."
            )

        model = cls(strict=strict)
        state_dict = torch.load(path, map_location=map_location, weights_only=True)
        model.load_state_dict(state_dict)
        model.is_pretrained = True
        return model


if __name__ == "__main__":
    print("Testing OrientationalCorrectionMLP...")
    mlp = OrientationalCorrectionMLP(strict=True)
    num_params = sum(p.numel() for p in mlp.parameters())
    print(f"Total parameter count: {num_params} (< 2000 requirement satisfied)")
    assert num_params < 2000

    dummy_input = torch.randn(1, 3, 10, 10, 10)
    try:
        mlp(dummy_input)
        assert False, "Should have raised RuntimeError when uncalibrated!"
    except RuntimeError as e:
        print("Caught expected strictness error:", e)

    # Mark as pretrained for smoke test
    mlp.is_pretrained = True
    out = mlp(dummy_input)
    assert out.shape == (1, 1, 10, 10, 10)
    print("OrientationalCorrectionMLP test passed successfully!")
