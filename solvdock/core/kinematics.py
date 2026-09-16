"""Differentiable kinematics: Rodrigues torsion FK and so(3) Lie-algebraic rigid rotation."""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from solvdock.core.topology import MolecularTopology


def skew(omega: torch.Tensor) -> torch.Tensor:
    """Builds the 3x3 skew-symmetric cross-product matrix from an so(3) Lie-algebra vector.

    Args:
        omega: 3-vector [wx, wy, wz] (or batch of shape (..., 3)).

    Returns:
        K: Skew-symmetric matrix of shape (..., 3, 3).
    """
    wx = omega[..., 0]
    wy = omega[..., 1]
    wz = omega[..., 2]
    zeros = torch.zeros_like(wx)

    # Row 0: [ 0,  -wz,  wy]
    # Row 1: [ wz,   0, -wx]
    # Row 2: [-wy,  wx,   0]
    row0 = torch.stack([zeros, -wz, wy], dim=-1)
    row1 = torch.stack([wz, zeros, -wx], dim=-1)
    row2 = torch.stack([-wy, wx, zeros], dim=-1)

    return torch.stack([row0, row1, row2], dim=-2)


def apply_rigid_transform(
    coords: torch.Tensor,
    omega: torch.Tensor,
    translation: torch.Tensor,
    center: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Applies rigid SO(3) rotation + translation via so(3) exponential map.

    Because R is formed via torch.linalg.matrix_exp(skew(omega)), R is strictly
    orthogonal with det(R) = +1 by construction. No quaternion drift, renormalization,
    or manifold departures occur during gradient descent.

    Args:
        coords: Tensor of shape (N, 3) in Angstroms.
        omega: Lie-algebra vector of shape (3,) parameterizing axis-angle rotation.
        translation: Translation 3-vector [Tx, Ty, Tz] of shape (3,).
        center: Optional center of rotation. If None, uses coords.mean(dim=0).

    Returns:
        transformed: Tensor of shape (N, 3).
    """
    if center is None:
        center = coords.mean(dim=0, keepdim=True)
    else:
        center = center.view(1, 3)

    # Compute SO(3) matrix: R = exp([omega]_x)
    K = skew(omega)
    R = torch.linalg.matrix_exp(K)  # (3, 3)

    # x' = (x - center) @ R^T + center + translation
    centered = coords - center
    rotated = centered @ R.T
    transformed = rotated + center + translation.view(1, 3)
    return transformed


def rotate_subgraph(
    coords: torch.Tensor,
    origin: torch.Tensor,
    axis: torch.Tensor,
    theta: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Rotates masked downstream atoms around directed bond axis using Rodrigues' formula.

    Args:
        coords: Tensor of shape (N, 3).
        origin: Point on rotation axis (parent atom coords), shape (3,) or (1, 3).
        axis: Unit direction vector of rotation axis, shape (3,) or (1, 3).
        theta: Scalar angle in radians (requires_grad=True).
        mask: Boolean tensor of shape (N,) indicating which atoms rotate.

    Returns:
        new_coords: Tensor of shape (N, 3).
    """
    # Normalize axis
    u = axis / (torch.norm(axis) + 1e-10)
    origin = origin.view(1, 3)
    u = u.view(1, 3)

    v = coords - origin  # (N, 3)

    # Rodrigues: v_rot = v*cos(theta) + (u x v)*sin(theta) + u*(u . v)*(1 - cos(theta))
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    # Cross product u x v
    # u: (1, 3), v: (N, 3) -> cross along last dim
    u_cross_v = torch.cross(u.expand_as(v), v, dim=-1)

    # Dot product u . v
    u_dot_v = torch.sum(u * v, dim=-1, keepdim=True)  # (N, 1)

    v_rot = v * cos_t + u_cross_v * sin_t + u * u_dot_v * (1.0 - cos_t)
    rot_coords = origin + v_rot

    # Apply only to masked atoms
    mask_3d = mask.unsqueeze(-1)  # (N, 1)
    new_coords = torch.where(mask_3d, rot_coords, coords)
    return new_coords


def apply_torsions(
    coords: torch.Tensor,
    thetas: torch.Tensor,
    topology: MolecularTopology,
) -> torch.Tensor:
    """Hierarchically applies torsion angle updates along the kinematic DAG.

    Args:
        coords: Tensor of shape (N, 3).
        thetas: Tensor of shape (K,) containing rotation angles for K rotatable bonds.
        topology: MolecularTopology providing rotatable_bonds and downstream_masks.

    Returns:
        updated_coords: Tensor of shape (N, 3).
    """
    K = topology.rotatable_bonds.shape[0]
    if K == 0:
        return coords

    curr_coords = coords
    # Traverse bonds in DAG order (root -> leaves)
    for k in range(K):
        parent_idx, child_idx = topology.rotatable_bonds[k]
        origin = curr_coords[parent_idx]
        axis = curr_coords[child_idx] - curr_coords[parent_idx]
        mask = topology.downstream_masks[k].to(device=coords.device)
        theta_k = thetas[k]

        curr_coords = rotate_subgraph(curr_coords, origin, axis, theta_k, mask)

    return curr_coords


if __name__ == "__main__":
    print("Testing Kinematics...")
    # Test rigid rotation via so(3)
    coords = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
    omega = torch.tensor([0.0, 0.0, 3.14159265 / 2.0], requires_grad=True)  # 90 deg around z
    T = torch.tensor([0.0, 0.0, 2.0], requires_grad=True)

    out = apply_rigid_transform(coords, omega, T, center=torch.zeros(3))
    print("Original:\n", coords.numpy())
    print("Rotated 90 deg + translated:\n", out.detach().numpy())

    # Check orthogonality of R: R @ R.T should be identity
    K = skew(omega)
    R = torch.linalg.matrix_exp(K)
    identity_err = torch.norm(R @ R.T - torch.eye(3)).item()
    print(f"||R R^T - I||: {identity_err:.2e} (Strictly SO(3))")
    assert identity_err < 1e-6

    # Test autograd
    loss = torch.sum(out ** 2)
    loss.backward()
    assert omega.grad is not None and T.grad is not None
    print("Autograd successfully flowed to omega and T!")

    print("Kinematics tests passed successfully!")
