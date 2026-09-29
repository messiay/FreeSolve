"""Differentiable internal coordinate torsional kinematics and SE(3) mechanics."""

from collections import deque
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem

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

    row0 = torch.stack([zeros, -wz, wy], dim=-1)
    row1 = torch.stack([wz, zeros, -wx], dim=-1)
    row2 = torch.stack([-wy, wx, zeros], dim=-1)

    return torch.stack([row0, row1, row2], dim=-2)


def rodrigues_rotation_matrix(axis: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Computes a 3x3 rotation matrix using Rodrigues' formula in PyTorch.

    Args:
        axis: Normalized unit vector of shape (3,).
        theta: Rotation angle in radians (scalar or 0-dim/1-dim tensor).

    Returns:
        3x3 rotation matrix.
    """
    if axis.ndim == 2 and axis.shape[0] == 1:
        axis = axis.squeeze(0)
    theta = theta.squeeze()

    u_x, u_y, u_z = axis[0], axis[1], axis[2]
    zero = torch.zeros((), dtype=axis.dtype, device=axis.device)

    K = torch.stack([
        torch.stack([zero, -u_z, u_y]),
        torch.stack([u_z, zero, -u_x]),
        torch.stack([-u_y, u_x, zero]),
    ])

    I = torch.eye(3, dtype=axis.dtype, device=axis.device)
    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)

    R = I + sin_t * K + (1.0 - cos_t) * (K @ K)
    return R


def axis_angle_to_matrix(omega: torch.Tensor) -> torch.Tensor:
    """Converts a 3D axis-angle vector to a 3x3 rotation matrix.

    Args:
        omega: 3D vector of shape (3,) where norm is angle and direction is axis.
    """
    theta = torch.norm(omega)
    if theta < 1e-7:
        return torch.eye(3, dtype=omega.dtype, device=omega.device)
    axis = omega / theta
    return rodrigues_rotation_matrix(axis, theta)


def apply_rigid_transform(
    coords: torch.Tensor,
    omega: torch.Tensor,
    translation: torch.Tensor,
    center: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Applies rigid SO(3) rotation + translation via so(3) exponential map."""
    if center is None:
        center = coords.mean(dim=0, keepdim=True)
    else:
        center = center.view(1, 3)

    K = skew(omega)
    R = torch.linalg.matrix_exp(K)

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
    """Rotates masked downstream atoms around directed bond axis using Rodrigues' formula."""
    u = axis / (torch.norm(axis) + 1e-10)
    origin = origin.view(1, 3)
    u = u.view(1, 3)

    v = coords - origin
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    u_cross_v = torch.cross(u.expand_as(v), v, dim=-1)
    u_dot_v = torch.sum(u * v, dim=-1, keepdim=True)

    v_rot = v * cos_t + u_cross_v * sin_t + u * u_dot_v * (1.0 - cos_t)
    rot_coords = origin + v_rot

    mask_3d = mask.unsqueeze(-1)
    new_coords = torch.where(mask_3d, rot_coords, coords)
    return new_coords


def apply_torsions(
    coords: torch.Tensor,
    thetas: torch.Tensor,
    topology: MolecularTopology,
) -> torch.Tensor:
    """Hierarchically applies torsion angle updates along the kinematic DAG."""
    K = topology.rotatable_bonds.shape[0]
    if K == 0:
        return coords

    curr_coords = coords
    for k in range(K):
        parent_idx, child_idx = topology.rotatable_bonds[k]
        origin = curr_coords[parent_idx]
        axis = curr_coords[child_idx] - curr_coords[parent_idx]
        mask = topology.downstream_masks[k].to(device=coords.device)
        theta_k = thetas[k]

        curr_coords = rotate_subgraph(curr_coords, origin, axis, theta_k, mask)

    return curr_coords


def find_rotatable_bonds(
    mol: Chem.Mol,
    fixed_atom_indices: Optional[Set[int]] = None,
) -> List[Tuple[int, int]]:
    """Identifies single, acyclic rotatable bonds ordered topologically from fixed root.

    Args:
        mol: RDKit molecule.
        fixed_atom_indices: Set of atom indices considered fixed/root (e.g., backbone atoms).

    Returns:
        List of (begin_atom_idx, end_atom_idx) tuples directed strictly outward from root.
    """
    if fixed_atom_indices is None:
        fixed_atom_indices = set()

    min_fixed_dist = None
    if fixed_atom_indices and mol.GetNumAtoms() > 0:
        dist_matrix = Chem.GetDistanceMatrix(mol)
        fixed_list = [idx for idx in fixed_atom_indices if idx < mol.GetNumAtoms()]
        if fixed_list:
            min_fixed_dist = np.min([dist_matrix[idx] for idx in fixed_list], axis=0)

    candidate_bonds = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        if bond.IsInRing():
            continue

        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()

        if a1.GetDegree() <= 1 or a2.GetDegree() <= 1:
            continue

        # Check for amide bonds (conjugated with carbonyl: C(=O)-N)
        is_amide = False
        for a, b in [(a1, a2), (a2, a1)]:
            if a.GetAtomicNum() == 7:
                for neighbor in b.GetNeighbors():
                    if neighbor.GetIdx() != a.GetIdx() and neighbor.GetAtomicNum() == 8:
                        nb_bond = mol.GetBondBetweenAtoms(b.GetIdx(), neighbor.GetIdx())
                        if nb_bond and nb_bond.GetBondType() == Chem.BondType.DOUBLE:
                            is_amide = True
                            break
        if is_amide:
            continue

        i, j = a1.GetIdx(), a2.GetIdx()
        if min_fixed_dist is not None:
            # Skip if both atoms are in fixed root
            if i in fixed_atom_indices and j in fixed_atom_indices:
                continue
            d_i, d_j = min_fixed_dist[i], min_fixed_dist[j]
            if d_i < d_j:
                u, v, dist = i, j, int(d_i)
            elif d_j < d_i:
                u, v, dist = j, i, int(d_j)
            else:
                # Equidistant from root; skip to prevent ambiguous orientation
                continue

            # Check if downstream component reachable from v (without traversing u) connects back to fixed root
            # (e.g., disulfide bridges forming closed macrocyclic loops). If it reaches root, rotating u-v would
            # break the macrocycle and cause bond length distortions.
            visited_check = {u, v}
            queue_check = deque([v])
            reaches_root = False
            while queue_check:
                curr = queue_check.popleft()
                if curr in fixed_atom_indices:
                    reaches_root = True
                    break
                atom_curr = mol.GetAtomWithIdx(curr)
                for nb in atom_curr.GetNeighbors():
                    nb_idx = nb.GetIdx()
                    if nb_idx not in visited_check:
                        visited_check.add(nb_idx)
                        queue_check.append(nb_idx)

            if not reaches_root:
                candidate_bonds.append((u, v, dist))
        else:
            candidate_bonds.append((i, j, 0))

    if min_fixed_dist is not None:
        # Sort candidate bonds by distance from fixed root ascending
        # This guarantees parent joints are evaluated before child joints in kinematic chains
        candidate_bonds.sort(key=lambda x: x[2])

    return [(b[0], b[1]) for b in candidate_bonds]


def build_downstream_subgraphs(
    mol: Chem.Mol,
    rotatable_bonds: List[Tuple[int, int]],
    root_indices: Optional[Set[int]] = None,
) -> List[torch.Tensor]:
    """Finds boolean masks of downstream atoms that rotate with each bond.

    Strictly guarantees that no atom in root_indices is ever rotated.
    """
    n_atoms = mol.GetNumAtoms()
    subgraphs = []

    for i, j in rotatable_bonds:
        mask = torch.zeros(n_atoms, dtype=torch.bool)
        visited = {i, j}
        if root_indices:
            visited = visited | set(root_indices)
        queue = deque([j])
        mask[j] = True

        while queue:
            curr = queue.popleft()
            atom = mol.GetAtomWithIdx(curr)
            for neighbor in atom.GetNeighbors():
                nb_idx = neighbor.GetIdx()
                if nb_idx not in visited:
                    visited.add(nb_idx)
                    mask[nb_idx] = True
                    queue.append(nb_idx)

        # Enforce that no root/backbone atom is ever marked downstream
        if root_indices:
            for r_idx in root_indices:
                if r_idx < n_atoms:
                    mask[r_idx] = False

        subgraphs.append(mask)

    return subgraphs


class DifferentiableSE3(nn.Module):
    """Rigid-body SE(3) transformation module.

    Guarantees 100% exact conservation of internal geometry.
    """

    def __init__(self, init_center: torch.Tensor):
        super().__init__()
        self.register_buffer("center_0", init_center.clone().detach())
        self.omega = nn.Parameter(torch.zeros(3, dtype=torch.float32))
        self.trans = nn.Parameter(torch.zeros(3, dtype=torch.float32))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Applies rigid rotation around center and translation."""
        R = axis_angle_to_matrix(self.omega)
        centered = coords - self.center_0
        rotated = centered @ R.T
        return self.center_0 + rotated + self.trans


class DifferentiableTorsionTree(nn.Module):
    """Articulated torsional kinematics module in PyTorch.

    Computes exact forward kinematics by rotating downstream molecular subgraphs
    around rotatable bond axes. Strictly locks all bond lengths, bond angles,
    and ring structures to machine precision (< 1e-6 A).
    """

    def __init__(
        self,
        base_coords: torch.Tensor,
        rotatable_bonds: List[Tuple[int, int]],
        downstream_masks: List[torch.Tensor],
    ):
        super().__init__()
        self.register_buffer("base_coords", base_coords.clone().detach().float())
        self.rotatable_bonds = rotatable_bonds
        self.n_torsions = len(rotatable_bonds)

        for idx, mask in enumerate(downstream_masks):
            self.register_buffer(f"mask_{idx}", mask.clone().detach())

        if self.n_torsions > 0:
            self.thetas = nn.Parameter(torch.zeros(self.n_torsions, dtype=torch.float32))
        else:
            self.thetas = nn.Parameter(torch.empty(0, dtype=torch.float32))

    def forward(self, input_coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Applies forward kinematics to compute updated 3D coordinates."""
        coords = self.base_coords if input_coords is None else input_coords

        if self.n_torsions == 0:
            return coords

        curr_coords = coords.clone()

        for k in range(self.n_torsions):
            i, j = self.rotatable_bonds[k]
            mask = getattr(self, f"mask_{k}")

            theta = self.thetas[k]

            p_i = curr_coords[i]
            p_j = curr_coords[j]
            bond_vec = p_j - p_i
            bond_len = torch.norm(bond_vec) + 1e-9
            axis = bond_vec / bond_len

            R = rodrigues_rotation_matrix(axis, theta)

            sub_coords = curr_coords[mask]
            rel_coords = sub_coords - p_i
            rot_coords = rel_coords @ R.T
            curr_coords[mask] = p_i + rot_coords

        return curr_coords
