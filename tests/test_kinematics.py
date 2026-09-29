"""Unit tests for articulated torsional kinematics and SE(3) mechanics."""

import numpy as np
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from solvdock.core.kinematics import (
    DifferentiableSE3,
    DifferentiableTorsionTree,
    axis_angle_to_matrix,
    build_downstream_subgraphs,
    find_rotatable_bonds,
    rodrigues_rotation_matrix,
)


@pytest.fixture
def sample_molecule():
    """Creates a flexible molecule with rotatable bonds (n-butanol or phenethylamine)."""
    mol = Chem.MolFromSmiles("NCCCC(=O)O")  # GABA / 4-aminobutanoic acid
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


@pytest.fixture
def aromatic_molecule():
    """Creates a molecule with an aromatic ring (benzylamine)."""
    mol = Chem.MolFromSmiles("c1ccccc1CCN")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def test_rodrigues_rotation_orthonormality():
    """Test that Rodrigues rotation matrix is strictly orthonormal (R @ R.T == I, det(R) == 1)."""
    axis = torch.tensor([1.0, 2.0, -3.0])
    axis = axis / torch.norm(axis)
    theta = torch.tensor(1.234)

    R = rodrigues_rotation_matrix(axis, theta)

    I = torch.eye(3)
    assert torch.allclose(R @ R.T, I, atol=1e-6)
    assert torch.allclose(R.T @ R, I, atol=1e-6)
    assert torch.allclose(torch.det(R), torch.tensor(1.0), atol=1e-6)


def test_se3_internal_distance_conservation(sample_molecule):
    """Test that DifferentiableSE3 preserves 100% of pairwise internal distances."""
    conf = sample_molecule.GetConformer()
    coords = torch.tensor([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                           for i in range(sample_molecule.GetNumAtoms())], dtype=torch.float32)

    center = coords.mean(dim=0)
    se3 = DifferentiableSE3(center)

    # Random rotation and translation
    se3.omega.data = torch.tensor([0.7, -1.2, 0.5])
    se3.trans.data = torch.tensor([15.0, -3.2, 8.1])

    transformed = se3(coords)

    # Compute all pairwise distances before and after
    d_before = torch.cdist(coords, coords)
    d_after = torch.cdist(transformed, transformed)

    max_diff = torch.max(torch.abs(d_before - d_after)).item()
    assert max_diff < 1e-5, f"SE(3) altered internal distances: max error = {max_diff}"


def test_torsional_bond_length_conservation(sample_molecule):
    """Test that rotating internal dihedral angles preserves ALL covalent bond lengths (< 1e-5 A)."""
    conf = sample_molecule.GetConformer()
    coords = torch.tensor([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                           for i in range(sample_molecule.GetNumAtoms())], dtype=torch.float32)

    rot_bonds = find_rotatable_bonds(sample_molecule)
    assert len(rot_bonds) > 0, "Expected at least 1 rotatable bond"

    masks = build_downstream_subgraphs(sample_molecule, rot_bonds)
    tree = DifferentiableTorsionTree(coords, rot_bonds, masks)

    # Reference bond lengths
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in sample_molecule.GetBonds()]
    r0_list = [torch.norm(coords[i] - coords[j]).item() for i, j in bonds]

    # Test across 5 random dihedral angle configurations
    torch.manual_seed(42)
    for _ in range(5):
        random_thetas = (torch.rand(len(rot_bonds)) - 0.5) * 2.0 * np.pi
        tree.thetas.data = random_thetas

        new_coords = tree()

        for (i, j), r0 in zip(bonds, r0_list):
            r_new = torch.norm(new_coords[i] - new_coords[j]).item()
            diff = abs(r_new - r0)
            assert diff < 1e-5, f"Bond {i}-{j} stretched by {diff} A under torsional rotation!"


def test_torsional_valence_angle_conservation(sample_molecule):
    """Test that all valence angles (A-B-C) remain strictly invariant under torsional rotation."""
    conf = sample_molecule.GetConformer()
    coords = torch.tensor([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                           for i in range(sample_molecule.GetNumAtoms())], dtype=torch.float32)

    rot_bonds = find_rotatable_bonds(sample_molecule)
    masks = build_downstream_subgraphs(sample_molecule, rot_bonds)
    tree = DifferentiableTorsionTree(coords, rot_bonds, masks)

    # Collect valence angle triplets
    triplets = []
    for atom in sample_molecule.GetAtoms():
        b_idx = atom.GetIdx()
        neighbors = [nb.GetIdx() for nb in atom.GetNeighbors()]
        if len(neighbors) >= 2:
            for n1_idx in range(len(neighbors)):
                for n2_idx in range(n1_idx + 1, len(neighbors)):
                    triplets.append((neighbors[n1_idx], b_idx, neighbors[n2_idx]))

    def compute_angle(c, i, j, k):
        v1 = c[i] - c[j]
        v2 = c[k] - c[j]
        cos_ang = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2) + 1e-9)
        return torch.acos(torch.clamp(cos_ang, -1.0, 1.0)).item()

    ref_angles = [compute_angle(coords, a, b, c) for a, b, c in triplets]

    # Apply large non-trivial rotations
    tree.thetas.data = torch.linspace(0.5, 2.5, tree.n_torsions)
    new_coords = tree()

    for (a, b, c), ang_0 in zip(triplets, ref_angles):
        ang_new = compute_angle(new_coords, a, b, c)
        diff = abs(ang_new - ang_0)
        assert diff < 1e-4, f"Valence angle {a}-{b}-{c} distorted by {diff} radians!"


def test_aromatic_ring_preservation(aromatic_molecule):
    """Test that aromatic ring internal geometry is 100% rigid during sidechain rotation."""
    conf = aromatic_molecule.GetConformer()
    coords = torch.tensor([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                           for i in range(aromatic_molecule.GetNumAtoms())], dtype=torch.float32)

    rot_bonds = find_rotatable_bonds(aromatic_molecule)
    masks = build_downstream_subgraphs(aromatic_molecule, rot_bonds)
    tree = DifferentiableTorsionTree(coords, rot_bonds, masks)

    # Ring atom indices
    ring_atoms = [a.GetIdx() for a in aromatic_molecule.GetAromaticAtoms()]
    assert len(ring_atoms) == 6

    d_ring_0 = torch.cdist(coords[ring_atoms], coords[ring_atoms])

    # Rotate sidechains
    tree.thetas.data = torch.linspace(-1.0, 1.5, tree.n_torsions)
    new_coords = tree()

    d_ring_new = torch.cdist(new_coords[ring_atoms], new_coords[ring_atoms])
    max_ring_distortion = torch.max(torch.abs(d_ring_0 - d_ring_new)).item()
    assert max_ring_distortion < 1e-5, f"Aromatic ring was distorted: max diff = {max_ring_distortion}"


def test_torsional_differentiability(sample_molecule):
    """Test that PyTorch autograd computes smooth, non-zero gradients through torsional kinematics."""
    conf = sample_molecule.GetConformer()
    coords = torch.tensor([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                           for i in range(sample_molecule.GetNumAtoms())], dtype=torch.float32)

    rot_bonds = find_rotatable_bonds(sample_molecule)
    masks = build_downstream_subgraphs(sample_molecule, rot_bonds)
    tree = DifferentiableTorsionTree(coords, rot_bonds, masks)

    # Target: dummy potential pushing atoms away from origin
    new_coords = tree()
    target_loss = torch.sum(new_coords ** 2)
    target_loss.backward()

    assert tree.thetas.grad is not None
    assert not torch.isnan(tree.thetas.grad).any()
    assert not torch.isinf(tree.thetas.grad).any()
    # At least some gradients must be non-zero
    assert torch.norm(tree.thetas.grad).item() > 1e-5


def test_end_to_end_torsional_refinement_2v00():
    """Test FlexibleRefiner with mode='torsional' on 2v00, verifying exact bond preservation."""
    from solvdock.pipeline.flexible_refiner import FlexibleRefiner
    import os

    poc_path = "data/casf2016_core/2v00_pocket.pdb"
    lig_path = "data/casf2016_core/2v00_ligand.sdf"

    if not os.path.exists(poc_path) or not os.path.exists(lig_path):
        pytest.skip("CASF-2016 2v00 data not present")

    m_poc = Chem.MolFromPDBFile(poc_path, removeHs=False)
    m_lig = Chem.SDMolSupplier(lig_path, removeHs=False)[0]

    # Pre-calculate all covalent bond lengths in both molecules
    poc_bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m_poc.GetBonds()]
    lig_bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m_lig.GetBonds()]

    c_poc_0 = m_poc.GetConformer().GetPositions()
    c_lig_0 = m_lig.GetConformer().GetPositions()

    r0_poc = [np.linalg.norm(c_poc_0[i] - c_poc_0[j]) for i, j in poc_bonds]
    r0_lig = [np.linalg.norm(c_lig_0[i] - c_lig_0[j]) for i, j in lig_bonds]

    refiner = FlexibleRefiner()
    res = refiner.refine_induced_fit(m_lig, m_poc, mode="torsional", max_steps=15, lr=0.03)

    assert res["final_bond_strain_energy"] == 0.0
    assert res["backbone_rmsd"] == 0.0
    assert res["n_sidechain_torsions"] > 0
    assert res["final_loss"] < res["initial_loss"]

    # Verify that NO bond length in pocket or ligand changed by more than 1e-5 A
    final_poc = res["final_pocket_coords"].numpy()
    final_lig = res["final_ligand_coords"].numpy()

    for (i, j), r0 in zip(poc_bonds, r0_poc):
        r_after = np.linalg.norm(final_poc[i] - final_poc[j])
        assert abs(r_after - r0) < 1e-4, f"Pocket bond {i}-{j} stretched by {abs(r_after - r0)} A!"

    for (i, j), r0 in zip(lig_bonds, r0_lig):
        r_after = np.linalg.norm(final_lig[i] - final_lig[j])
        assert abs(r_after - r0) < 1e-4, f"Ligand bond {i}-{j} stretched by {abs(r_after - r0)} A!"

