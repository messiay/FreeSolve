"""Rigorous Gradient Faithfulness Benchmark: Analytical Autograd vs Numerical Finite Differences.

Validates that SolvDock's kinematic and potential gradients are exact, preserving
strict mathematical faithfulness required for training downstream neural networks.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem

from solvdock.core.charges import assign_charges, get_partial_charges
from solvdock.core.grid_engine import SpatialGridEngine
from solvdock.core.kinematics import (
    DifferentiableSE3,
    DifferentiableTorsionTree,
    axis_angle_to_matrix,
    build_downstream_subgraphs,
    find_rotatable_bonds,
    rodrigues_rotation_matrix,
)
from solvdock.pipeline.energy import CombinedPotential


def finite_difference_grad(func, param: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Computes two-point centered finite difference gradients."""
    grad = torch.zeros_like(param)
    orig_data = param.data.clone()

    for idx in range(param.numel()):
        # f(x + eps)
        param.data.copy_(orig_data)
        param.data.view(-1)[idx] += eps
        loss_plus = func()

        # f(x - eps)
        param.data.copy_(orig_data)
        param.data.view(-1)[idx] -= eps
        loss_minus = func()

        # Central difference: (f(x+eps) - f(x-eps)) / (2*eps)
        grad.view(-1)[idx] = (loss_plus - loss_minus) / (2.0 * eps)

    param.data.copy_(orig_data)
    return grad


@pytest.fixture
def flexible_ligand():
    """Generates a small flexible drug-like molecule with multiple rotatable bonds."""
    mol = Chem.MolFromSmiles("CCN(CC)CC(=O)Oc1ccccc1")  # Diethylaminoethyl benzoate
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=101)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def test_se3_translation_gradient_faithfulness(flexible_ligand):
    """Verifies that analytical gradients of rigid SE(3) translation match finite differences."""
    conf = flexible_ligand.GetConformer()
    coords = torch.tensor(conf.GetPositions(), dtype=torch.float64)
    center = coords.mean(dim=0)

    # Simple quadratic potential: E = 0.5 * sum((coords_transformed - target)^2)
    target = coords + torch.tensor([1.2, -0.8, 0.5], dtype=torch.float64)

    # Set parameters in double precision for exact numerical differentiation
    se3 = DifferentiableSE3(center.float())
    se3.trans = nn.Parameter(torch.tensor([0.2, -0.3, 0.4], dtype=torch.float64))
    se3.omega = nn.Parameter(torch.tensor([0.1, 0.2, -0.1], dtype=torch.float64))

    def compute_loss():
        # Rodrigues rotation in float64
        R = axis_angle_to_matrix(se3.omega)
        curr = (coords - center) @ R.T + center + se3.trans
        return 0.5 * torch.sum((curr - target) ** 2)

    # 1. Analytical Autograd
    loss = compute_loss()
    loss.backward()
    grad_auto_trans = se3.trans.grad.clone()
    grad_auto_omega = se3.omega.grad.clone()

    # 2. Numerical Finite Difference
    grad_num_trans = finite_difference_grad(compute_loss, se3.trans, eps=1e-5)
    grad_num_omega = finite_difference_grad(compute_loss, se3.omega, eps=1e-5)

    # 3. Compare Translation Gradients
    rel_err_trans = torch.norm(grad_auto_trans - grad_num_trans) / torch.norm(grad_num_trans)
    cos_sim_trans = torch.dot(grad_auto_trans, grad_num_trans) / (torch.norm(grad_auto_trans) * torch.norm(grad_num_trans))

    # 4. Compare Rotation Gradients
    rel_err_omega = torch.norm(grad_auto_omega - grad_num_omega) / torch.norm(grad_num_omega)
    cos_sim_omega = torch.dot(grad_auto_omega, grad_num_omega) / (torch.norm(grad_auto_omega) * torch.norm(grad_num_omega))

    print(f"\n[SE(3) Translation] Rel Error: {rel_err_trans.item():.2e} | Cosine Sim: {cos_sim_trans.item():.6f}")
    print(f"[SE(3) Rotation]    Rel Error: {rel_err_omega.item():.2e} | Cosine Sim: {cos_sim_omega.item():.6f}")

    assert rel_err_trans.item() < 1e-4, f"Translation relative error {rel_err_trans.item()} exceeds 1e-4"
    assert cos_sim_trans.item() > 0.9999, "Translation gradient direction mismatch"
    assert rel_err_omega.item() < 1e-4, f"Rotation relative error {rel_err_omega.item()} exceeds 1e-4"
    assert cos_sim_omega.item() > 0.9999, "Rotation gradient direction mismatch"


def test_torsion_tree_gradient_faithfulness(flexible_ligand):
    """Verifies that DifferentiableTorsionTree forward kinematics gradients match finite differences."""
    conf = flexible_ligand.GetConformer()
    coords = torch.tensor(conf.GetPositions(), dtype=torch.float64)

    rot_bonds = find_rotatable_bonds(flexible_ligand)
    masks = build_downstream_subgraphs(flexible_ligand, rot_bonds)
    assert len(rot_bonds) >= 4, "Fixture should have at least 4 rotatable dihedrals"

    tree = DifferentiableTorsionTree(coords.float(), rot_bonds, masks)
    # Cast to float64 for exact machine precision in finite difference check
    tree.base_coords = tree.base_coords.to(torch.float64)
    tree.thetas = nn.Parameter(torch.tensor([0.2 * ((-1) ** k) for k in range(len(rot_bonds))], dtype=torch.float64))

    # Objective: Pull terminal atoms toward a fixed point
    fixed_attractor = coords.mean(dim=0, keepdim=True) + torch.tensor([[2.0, 1.0, -1.5]], dtype=torch.float64)

    def compute_loss():
        curr_coords = tree()
        dists = torch.norm(curr_coords - fixed_attractor, dim=-1)
        return torch.sum(dists ** 2)

    # 1. Analytical Autograd
    tree.zero_grad()
    loss = compute_loss()
    loss.backward()
    grad_auto = tree.thetas.grad.clone()

    # 2. Numerical Finite Difference
    grad_num = finite_difference_grad(compute_loss, tree.thetas, eps=1e-6)

    rel_error = torch.norm(grad_auto - grad_num) / torch.norm(grad_num)
    cos_sim = torch.dot(grad_auto, grad_num) / (torch.norm(grad_auto) * torch.norm(grad_num))

    print(f"\n[Torsion Tree float64] N_torsions: {len(rot_bonds)} | Rel Error: {rel_error.item():.2e} | Cosine Sim: {cos_sim.item():.6f}")
    for idx, (g_a, g_n) in enumerate(zip(grad_auto, grad_num)):
        print(f"  Torsion {idx:2d} | Autograd: {g_a.item():+10.4f} | NumFD: {g_n.item():+10.4f} | Diff: {abs(g_a-g_n).item():.2e}")

    assert rel_error.item() < 1e-4, f"Torsion gradient relative error {rel_error.item()} exceeds tolerance"
    assert cos_sim.item() > 0.9999, f"Torsion gradient cosine similarity {cos_sim.item()} below 0.9999"


def test_full_molecular_energy_gradient_faithfulness():
    """Verifies that full non-bonded potential (LJ + Screened Coulomb) yields faithful gradients."""
    # Build two small interacting molecular fragments: Ethanol + Acetic Acid (H-bonding pair)
    m1 = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    m2 = Chem.AddHs(Chem.MolFromSmiles("CC(=O)O"))
    AllChem.EmbedMolecule(m1, randomSeed=42)
    AllChem.EmbedMolecule(m2, randomSeed=42)

    assign_charges(m1, scheme="gasteiger")
    assign_charges(m2, scheme="gasteiger")

    q1 = get_partial_charges(m1)
    q2 = get_partial_charges(m2)
    z1 = torch.tensor([a.GetAtomicNum() for a in m1.GetAtoms()], dtype=torch.int64)
    z2 = torch.tensor([a.GetAtomicNum() for a in m2.GetAtoms()], dtype=torch.int64)

    c1 = torch.tensor(m1.GetConformer().GetPositions(), dtype=torch.float32)
    c2 = torch.tensor(m2.GetConformer().GetPositions(), dtype=torch.float32) + torch.tensor([2.5, 0.0, 0.0])

    rot1 = find_rotatable_bonds(m1)
    masks1 = build_downstream_subgraphs(m1, rot1)

    tree1 = DifferentiableTorsionTree(c1, rot1, masks1)
    se3_1 = DifferentiableSE3(c1.mean(dim=0))

    grid = SpatialGridEngine(box_size=16, grid_spacing=1.0)
    potential = CombinedPotential(pde_solver=None, grid_engine=grid)

    def compute_energy():
        curr_c1 = se3_1(tree1())
        e_dir, _, _ = potential.compute_direct_energy(curr_c1, q1, z1, c2, q2, z2)
        return e_dir

    # Autograd
    se3_1.zero_grad()
    if tree1.n_torsions > 0:
        tree1.zero_grad()
    loss = compute_energy()
    loss.backward()

    grad_auto_trans = se3_1.trans.grad.clone()
    grad_num_trans = finite_difference_grad(compute_energy, se3_1.trans, eps=1e-4)

    rel_err = torch.norm(grad_auto_trans - grad_num_trans) / torch.norm(grad_num_trans)
    cos_sim = torch.dot(grad_auto_trans, grad_num_trans) / (torch.norm(grad_auto_trans) * torch.norm(grad_num_trans))

    print(f"\n[Intermolecular LJ + Coulomb] Rel Error: {rel_err.item():.2e} | Cosine Sim: {cos_sim.item():.6f}")
    assert rel_err.item() < 2e-3, f"Energy gradient relative error {rel_err.item()} exceeds 2e-3"
    assert cos_sim.item() > 0.9999, "Energy gradient direction is unfaithful"


def test_torch_official_gradcheck_rodrigues():
    """Validates Rodrigues rotation matrix using PyTorch's official gradcheck."""
    axis = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64)
    axis = axis / torch.norm(axis)

    theta = torch.tensor(0.785, dtype=torch.float64, requires_grad=True)

    def rot_func(th):
        return rodrigues_rotation_matrix(axis, th)

    # PyTorch official numerical gradient check
    assert torch.autograd.gradcheck(rot_func, (theta,), eps=1e-6, atol=1e-5, rtol=1e-4)


if __name__ == "__main__":
    pytest.main(["-v", "tests/test_gradient_faithfulness.py"])
