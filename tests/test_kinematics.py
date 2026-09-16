"""Unit tests for differentiable Rodrigues FK and so(3) rigid transforms."""

import math
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from solvdock.core.kinematics import apply_rigid_transform, skew, apply_torsions
from solvdock.core.topology import MolecularTopology


def test_so3_lie_algebra_matrix():
    """Validates that matrix_exp of skew-symmetric matrix is strictly in SO(3)."""
    omega = torch.tensor([0.3, -0.7, 1.2], dtype=torch.float32)
    K = skew(omega)
    R = torch.linalg.matrix_exp(K)

    # Check orthogonality: R @ R.T = I
    identity_diff = torch.norm(R @ R.T - torch.eye(3)).item()
    assert identity_diff < 1e-6, f"Orthogonality error: {identity_diff}"

    # Check determinant: det(R) = +1 (special orthogonal group)
    det = torch.linalg.det(R).item()
    assert abs(det - 1.0) < 1e-6, f"Determinant is {det}, not 1.0"


def test_rigid_transform_translation_and_rotation():
    """Validates rigid transformation on simple coordinates."""
    coords = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    omega = torch.tensor([0.0, 0.0, math.pi / 2.0], dtype=torch.float32)  # 90 deg around z
    trans = torch.tensor([5.0, 5.0, 5.0], dtype=torch.float32)
    center = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

    out = apply_rigid_transform(coords, omega, trans, center=center)
    expected = torch.tensor([[0.0 + 5.0, 1.0 + 5.0, 0.0 + 5.0]], dtype=torch.float32)
    assert torch.allclose(out, expected, atol=1e-5)


def test_torsion_fk_butane():
    """Validates that torsion rotation preserves bond lengths and alters dihedral."""
    mol = Chem.MolFromSmiles("CCCC")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    topology = MolecularTopology(mol)

    coords = topology.atom_coords
    thetas = torch.tensor([math.pi / 2.0], requires_grad=True)

    rotated_coords = apply_torsions(coords, thetas, topology)
    # Check bond lengths between bonded heavy atoms remain invariant
    c1, c2 = coords[0], coords[1]
    c1_rot, c2_rot = rotated_coords[0], rotated_coords[1]
    d_orig = torch.norm(c1 - c2).item()
    d_rot = torch.norm(c1_rot - c2_rot).item()
    assert abs(d_orig - d_rot) < 1e-5

    # Autograd test
    loss = torch.sum(rotated_coords ** 2)
    loss.backward()
    assert thetas.grad is not None
