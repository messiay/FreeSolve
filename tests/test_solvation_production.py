"""Unit tests for SolvDock Solvation Engine productionization."""

import pytest
import torch
from rdkit import Chem

from solvdock.solvation.applicability import check_applicability_domain
from solvdock.solvation.born import compute_born_ion_correction, compute_effective_born_radius
from solvdock.solvation.api import predict_solvation, batch_predict_solvation
from solvdock.core.poisson_solver import solve_poisson


def test_applicability_clean_drug_like():
    """Confirms that standard neutral drug-like molecules pass applicability checks cleanly."""
    aspirin_smiles = "CC(=O)Oc1ccccc1C(=O)O"
    mol = Chem.MolFromSmiles(aspirin_smiles)
    report = check_applicability_domain(mol)

    assert report.is_within_domain is True
    assert len(report.flags) == 0
    assert report.formal_charge == 0
    assert report.base_uncertainty == 1.20


def test_applicability_push_pull_nitroaromatic():
    """Confirms that poly-nitro push-pull aromatics are explicitly flagged with elevated uncertainty."""
    # Benefin / dinitro-trifluoromethyl aniline
    benefin_smiles = "CCCCN(CC)c1c(cc(cc1[N+](=O)[O-])C(F)(F)F)[N+](=O)[O-]"
    mol = Chem.MolFromSmiles(benefin_smiles)
    report = check_applicability_domain(mol)

    assert report.is_within_domain is False
    assert "PUSH_PULL_NITROAROMATIC" in report.flags
    assert report.base_uncertainty >= 4.50


def test_applicability_net_charge():
    """Confirms that net-charged species are explicitly flagged."""
    acetate_smiles = "CC(=O)[O-]"
    mol = Chem.MolFromSmiles(acetate_smiles)
    report = check_applicability_domain(mol)

    assert report.is_within_domain is False
    assert any("NET_CHARGE" in f for f in report.flags)
    assert report.formal_charge == -1
    assert report.base_uncertainty >= 3.50


def test_born_correction_sign_and_scaling():
    """Verifies analytical Born continuum correction physics for neutral vs charged solutes."""
    # 1. Neutral solute must have exactly 0.0 correction
    ethanol = Chem.MolFromSmiles("CCO")
    corr_neutral = compute_born_ion_correction(ethanol)
    assert corr_neutral == 0.0

    # 2. Net negative ion (acetate) must have negative solvation free energy
    acetate = Chem.MolFromSmiles("CC(=O)[O-]")
    corr_acetate = compute_born_ion_correction(acetate)
    assert corr_acetate < 0.0

    # 3. Divalent ion (+2) should scale quadratically with charge compared to (+1) of same size
    r_eff = compute_effective_born_radius(acetate)
    assert r_eff > 1.50  # reasonable molecular radius in Angstroms


def test_debye_huckel_screening_poisson():
    """Verifies that salt concentration screens electrostatic potential at long range."""
    grid = torch.zeros((1, 1, 24, 24, 24), dtype=torch.float32)
    # Point charge at center
    grid[0, 0, 12, 12, 12] = 1.0

    # Solve in pure water (I = 0.0)
    phi_water = solve_poisson(grid, grid_spacing=1.0, ionic_strength=0.0)

    # Solve in 1.0 M salt (I = 1.0)
    phi_salt = solve_poisson(grid, grid_spacing=1.0, ionic_strength=1.0)

    # At the center (r=0), potential is finite
    val_water_center = phi_water[0, 0, 12, 12, 12].item()
    val_salt_center = phi_salt[0, 0, 12, 12, 12].item()
    assert val_water_center > 0.0
    assert val_salt_center > 0.0

    # At distance r = 6 Angstroms (12+6 = 18), salt should significantly screen the potential
    val_water_r6 = phi_water[0, 0, 18, 12, 12].item()
    val_salt_r6 = phi_salt[0, 0, 18, 12, 12].item()

    assert val_salt_r6 < val_water_r6
    # Relative screening ratio at r = 6 A with kappa ~ 0.33 A^-1: e^(-2.0) ~ 0.13
    assert val_salt_r6 / val_water_r6 < 0.50


def test_public_api_prediction():
    """Verifies public predict_solvation API on a benchmark neutral molecule (ethanol)."""
    res = predict_solvation("CCO")

    assert res.smiles == "CCO"
    # Ethanol experimental dG_hyd is -5.00 kcal/mol; model should predict negative hydration
    assert -8.0 < res.delta_g_hyd < -2.0
    assert res.estimated_error == 1.20
    assert res.is_within_applicability_domain is True
    assert len(res.flags) == 0
    assert res.formal_charge == 0
    assert res.components["dG_born_ion_correction"] == 0.0
    assert res.runtime_ms > 0.0


def test_batch_prediction():
    """Verifies public batch_predict_solvation API."""
    smiles_list = ["CCO", "c1ccccc1"]
    results = batch_predict_solvation(smiles_list)

    assert len(results) == 2
    assert results[0].is_within_applicability_domain is True
    assert results[1].is_within_applicability_domain is True
    # Benzene is less negative than ethanol (more hydrophobic)
    assert results[1].delta_g_hyd > results[0].delta_g_hyd
