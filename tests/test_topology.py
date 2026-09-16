"""Unit tests for MolecularTopology DAG decomposition and rotatable bond identification."""

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from solvdock.core.topology import MolecularTopology


def test_propanol_rotatable_bonds():
    """Propanol decomposes into exactly 1 rotatable bond (central C-C bond)."""
    mol = Chem.MolFromSmiles("CCCO")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    topology = MolecularTopology(mol)
    assert topology.rotatable_bonds.shape[0] == 1, (
        f"Expected 1 rotatable bond for propanol, got {topology.rotatable_bonds.shape[0]}"
    )


def test_ethanol_zero_rotatable_bonds():
    """Ethanol decomposes into exactly 0 rotatable bonds (both bonds have terminal heavy atom)."""
    mol = Chem.MolFromSmiles("CCO")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    topology = MolecularTopology(mol)
    assert topology.rotatable_bonds.shape[0] == 0, (
        f"Expected 0 rotatable bonds for ethanol (negative test), got {topology.rotatable_bonds.shape[0]}"
    )


def test_butane_rotatable_bonds():
    """Butane has 1 rotatable bond (C2-C3)."""
    mol = Chem.MolFromSmiles("CCCC")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    topology = MolecularTopology(mol)
    assert topology.rotatable_bonds.shape[0] == 1


def test_topology_tensors():
    """Validates tensor output shapes and data types."""
    mol = Chem.MolFromSmiles("CCCCO")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    topology = MolecularTopology(mol)

    coords, nums, charges, rot_bonds, masks = topology.get_tensors()
    N = mol.GetNumAtoms()
    assert coords.shape == (N, 3)
    assert nums.shape == (N,)
    assert charges.shape == (N,)
    assert rot_bonds.shape[1] == 2
    assert len(masks) == rot_bonds.shape[0]
