"""Unit tests for unified atomic charge assignment, AM1-BCC cache, and charge preservation."""

import pytest
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from solvdock.core.charges import (
    assign_charges,
    get_partial_charges,
    has_existing_charges,
    get_freesolv_am1bcc_cache,
)
from solvdock.core.pose_optimizer import PoseOptimizer
from solvdock.core.grid_engine import SpatialGridEngine
from solvdock.core.solvation_pde import SolvationPDESolver
from solvdock.pipeline.energy import CombinedPotential


def _make_mol(smiles: str = "CCO") -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    return mol


def test_mmff94_charge_assignment():
    """Verifies MMFF94 charge assignment, net charge conservation, and property tagging."""
    mol = _make_mol("CCO")
    charges, scheme = assign_charges(mol, scheme="mmff94")

    assert scheme == "mmff94"
    assert len(charges) == mol.GetNumAtoms()
    assert abs(float(torch.sum(charges).item())) < 1e-4
    assert all(a.HasProp("_MMFFCharge") for a in mol.GetAtoms())
    assert all(a.HasProp("_PartialCharge") for a in mol.GetAtoms())


def test_gasteiger_charge_assignment():
    """Verifies Gasteiger charge assignment fallback."""
    mol = _make_mol("CCO")
    charges, scheme = assign_charges(mol, scheme="gasteiger")

    assert scheme == "gasteiger"
    assert len(charges) == mol.GetNumAtoms()
    assert abs(float(torch.sum(charges).item())) < 1e-3
    assert all(a.HasProp("_GasteigerCharge") for a in mol.GetAtoms())


def test_am1bcc_cache_lookup_freesolv():
    """Verifies retrieval of authentic MobleyLab AM1-BCC charges from bundled cache."""
    cache = get_freesolv_am1bcc_cache()
    assert len(cache) == 642
    assert "mobley_2501588" in cache  # profluralin

    # Profluralin
    smiles = "CCC[N@@](CC1CC1)c2c(cc(cc2[N+](=O)[O-])C(F)(F)F)[N+](=O)[O-]"
    mol = _make_mol(smiles)

    charges, scheme = assign_charges(mol, scheme="am1bcc", compound_id="mobley_2501588")
    assert scheme == "am1bcc"
    assert len(charges) == 40
    assert abs(float(torch.sum(charges).item())) < 1e-3

    # Verify atom properties are set
    for atom in mol.GetAtoms():
        assert atom.HasProp("_AM1BCCCharge")
        assert atom.HasProp("_PartialCharge")


def test_charge_preservation():
    """Verifies that pre-existing quantum or custom charges are preserved strictly without overwriting."""
    mol = _make_mol("CCO")
    n = mol.GetNumAtoms()

    # Assign custom synthetic charges summing to 0
    custom_q = [0.1 * (i - n / 2) for i in range(n)]
    sum_q = sum(custom_q)
    custom_q[0] -= sum_q  # Force exact zero sum

    for i, a in enumerate(mol.GetAtoms()):
        a.SetProp("_PartialCharge", str(custom_q[i]))

    assert has_existing_charges(mol) is True

    # Call get_partial_charges
    extracted = get_partial_charges(mol)
    np.testing.assert_allclose(extracted.cpu().numpy(), np.array(custom_q, dtype=np.float32), atol=1e-5)

    # Calling assign_charges with scheme='preserve' should not alter them
    charges, scheme = assign_charges(mol, scheme="preserve", preserve_existing=True)
    assert scheme == "preserved"
    np.testing.assert_allclose(charges.cpu().numpy(), np.array(custom_q, dtype=np.float32), atol=1e-5)


def test_pose_optimizer_extracts_preserved_charges():
    """Verifies PoseOptimizer.extract_mol_tensors respects AM1-BCC charges rather than forcing Gasteiger."""
    grid = SpatialGridEngine(box_size=16)
    pde = SolvationPDESolver(steps=2, disable_residual_mlp=True)
    pot = CombinedPotential(pde_solver=pde, grid_engine=grid)
    opt = PoseOptimizer(pot)

    mol = _make_mol("CCO")
    for i, a in enumerate(mol.GetAtoms()):
        a.SetProp("_AM1BCCCharge", "0.12345")
        a.SetProp("_GasteigerCharge", "0.99999")

    _, charges, _ = opt.extract_mol_tensors(mol)

    # Must match _AM1BCCCharge (0.12345), NOT _GasteigerCharge (0.99999)
    assert abs(charges[0].item() - 0.12345) < 1e-4


def test_profluralin_push_pull_error_drop_with_am1bcc():
    """Verifies that AM1-BCC charges dramatically eliminate the push-pull conjugation error on profluralin."""
    from solvdock.solvation.api import predict_solvation

    smiles = "CCC[N@@](CC1CC1)c2c(cc(cc2[N+](=O)[O-])C(F)(F)F)[N+](=O)[O-]"
    expt = -2.45  # kcal/mol

    # MMFF94 prediction
    res_mmff = predict_solvation(smiles, compound_id="mobley_2501588", charge_model="mmff94")
    err_mmff = abs(res_mmff.delta_g_hyd - expt)

    # AM1-BCC prediction
    res_am1 = predict_solvation(smiles, compound_id="mobley_2501588", charge_model="am1bcc")
    err_am1 = abs(res_am1.delta_g_hyd - expt)

    # AM1-BCC error must be dramatically lower than MMFF94 error
    assert err_am1 < 2.5, f"AM1-BCC error {err_am1:.2f} kcal/mol should be < 2.5 kcal/mol"
    assert err_mmff > 6.0, f"MMFF94 error {err_mmff:.2f} kcal/mol should demonstrate push-pull failure > 6.0 kcal/mol"
    assert err_am1 < err_mmff - 4.5, "AM1-BCC must reduce push-pull error by at least 4.5 kcal/mol"
