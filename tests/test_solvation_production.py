"""Unit tests for SolvDock Solvation Engine productionization."""

import pytest
import torch
from rdkit import Chem

from solvdock.solvation.applicability import check_applicability_domain
from solvdock.solvation.born import compute_born_radius, compute_textbook_born_energy
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
    # Provenance check: must equal exact measured clean test set RMSE (2.72 kcal/mol)
    assert report.base_uncertainty == 2.72
    assert report.uncertainty_provenance == "measured_freesolv_clean_test_rmse"


def test_applicability_push_pull_nitroaromatic():
    """Confirms that poly-nitro push-pull aromatics are explicitly flagged with measured residual uncertainty."""
    benefin_smiles = "CCCCN(CC)c1c(cc(cc1[N+](=O)[O-])C(F)(F)F)[N+](=O)[O-]"
    mol = Chem.MolFromSmiles(benefin_smiles)
    report = check_applicability_domain(mol)

    assert report.is_within_domain is False
    assert "PUSH_PULL_NITROAROMATIC" in report.flags
    # Provenance check: must equal measured profluralin/benefin residual (~24.5 kcal/mol)
    assert report.base_uncertainty == 24.50
    assert report.uncertainty_provenance == "measured_push_pull_outlier_residual"


def test_applicability_net_charge():
    """Confirms that net-charged species are explicitly flagged with heuristic prior label."""
    acetate_smiles = "CC(=O)[O-]"
    mol = Chem.MolFromSmiles(acetate_smiles)
    report = check_applicability_domain(mol)

    assert report.is_within_domain is False
    assert any("NET_CHARGE" in f for f in report.flags)
    assert report.formal_charge == -1
    # Must be explicitly labeled as heuristic prior, not a false claim of calibrated measurement
    assert report.base_uncertainty == 15.00
    assert report.uncertainty_provenance == "heuristic_risk_prior_unsupported_net_charge"


def test_born_textbook_reference_values():
    """Verifies textbook Born formula against canonical physical benchmarks."""
    # 1. Neutral sphere Q = 0
    assert compute_textbook_born_energy(0.0, 2.0) == 0.0

    # 2. Textbook monovalent sphere (Q = 1, R = 2.0 A) in water (eps_r = 78.4):
    # Delta G_Born = - (332.0637 / 2) * (1 - 1/78.4) * (1 / 2.0) = -81.96 kcal/mol
    dG_r2 = compute_textbook_born_energy(1.0, 2.0, epsilon_r=78.4)
    assert abs(dG_r2 - (-81.96)) < 0.05

    # 3. Na+ ion (R = 1.68 A, Q = +1)
    dG_na = compute_textbook_born_energy(1.0, 1.68, epsilon_r=78.4)
    assert abs(dG_na - (-97.57)) < 0.05

    # 4. Cl- ion (R = 1.95 A, Q = -1)
    dG_cl = compute_textbook_born_energy(-1.0, 1.95, epsilon_r=78.4)
    assert abs(dG_cl - (-84.06)) < 0.05

    # 5. Divalent ion (+2) must scale exactly quadratically with charge (Q^2 = 4)
    dG_mg = compute_textbook_born_energy(2.0, 2.0, epsilon_r=78.4)
    assert abs(dG_mg - 4.0 * dG_r2) < 1e-4


def test_acetate_ion_physical_decomposition():
    """Confirms that acetate ion hydration lands in the experimental -77 to -80 kcal/mol range."""
    res = predict_solvation("CC(=O)[O-]")

    assert res.formal_charge == -1
    assert res.is_within_applicability_domain is False
    assert "NET_CHARGE_-1" in res.flags
    assert res.estimated_error == 15.00
    assert res.uncertainty_provenance == "heuristic_risk_prior_unsupported_net_charge"

    # Acetate experimental hydration free energy is -77 to -80 kcal/mol (Marcus 1985)
    assert -85.0 < res.delta_g_hyd < -72.0

    # Verify component breakdown
    born_val = res.components["dG_born_monopole"]
    neutral_val = res.components["dG_neutral_cavity_polar"]
    assert -76.0 < born_val < -68.0       # textbook Born for acetate R_eff ~ 2.2-2.4 A
    assert -10.0 < neutral_val < -2.0     # neutral acetic acid cavity/polar contribution
    assert abs(res.delta_g_hyd - (born_val + neutral_val)) < 1e-4


def test_debye_huckel_screening_poisson():
    """Verifies that salt concentration screens electrostatic potential at long range."""
    grid = torch.zeros((1, 1, 24, 24, 24), dtype=torch.float32)
    grid[0, 0, 12, 12, 12] = 1.0

    # Solve in pure water (I = 0.0)
    phi_water = solve_poisson(grid, grid_spacing=1.0, ionic_strength=0.0)

    # Solve in 1.0 M salt (I = 1.0)
    phi_salt = solve_poisson(grid, grid_spacing=1.0, ionic_strength=1.0)

    val_water_center = phi_water[0, 0, 12, 12, 12].item()
    val_salt_center = phi_salt[0, 0, 12, 12, 12].item()
    assert val_water_center > 0.0
    assert val_salt_center > 0.0

    # At r = 6 A (12+6 = 18), salt should significantly screen the potential
    val_water_r6 = phi_water[0, 0, 18, 12, 12].item()
    val_salt_r6 = phi_salt[0, 0, 18, 12, 12].item()

    assert val_salt_r6 < val_water_r6
    assert val_salt_r6 / val_water_r6 < 0.50


def test_public_api_prediction_neutral():
    """Verifies public predict_solvation API on a benchmark neutral molecule (ethanol)."""
    res = predict_solvation("CCO")

    assert res.smiles == "CCO"
    # Ethanol experimental dG_hyd is -5.00 kcal/mol; model should predict negative hydration
    assert -8.0 < res.delta_g_hyd < -2.0
    assert res.estimated_error == 2.72  # exact measured clean test RMSE
    assert res.uncertainty_provenance == "measured_freesolv_clean_test_rmse"
    assert res.is_within_applicability_domain is True
    assert len(res.flags) == 0
    assert res.formal_charge == 0
    assert res.components["dG_born_monopole"] == 0.0
    assert res.runtime_ms > 0.0
    assert res.charge_model_used == "mmff94"


def test_batch_prediction():
    """Verifies public batch_predict_solvation API."""
    smiles_list = ["CCO", "c1ccccc1"]
    results = batch_predict_solvation(smiles_list)

    assert len(results) == 2
    assert results[0].is_within_applicability_domain is True
    assert results[1].is_within_applicability_domain is True
    # Benzene is less negative than ethanol (more hydrophobic)
    assert results[1].delta_g_hyd > results[0].delta_g_hyd


def test_nine_ion_benchmark_accuracy_and_provenance():
    """Verifies 9-ion benchmark performance against Truhlar 2006 / Marcus 1991 experimental data."""
    import numpy as np

    ion_benchmark = [
        ("formate", "C(=O)[O-]", -83.5),
        ("acetate", "CC(=O)[O-]", -78.9),
        ("propionate", "CCC(=O)[O-]", -76.2),
        ("benzoate", "c1ccccc1C(=O)[O-]", -69.8),
        ("methylammonium", "C[NH3+]", -87.3),
        ("ethylammonium", "CC[NH3+]", -84.1),
        ("dimethylammonium", "C[NH2+]C", -76.7),
        ("trimethylammonium", "C[NH+](C)C", -66.5),
        ("phenolate", "c1ccccc1[O-]", -69.0),
    ]

    residuals = []
    for name, smiles, expt in ion_benchmark:
        res = predict_solvation(smiles)
        assert res.is_within_applicability_domain is False
        assert any("NET_CHARGE" in f for f in res.flags)
        assert res.estimated_error == 15.00
        assert res.uncertainty_provenance == "heuristic_risk_prior_unsupported_net_charge"
        residuals.append(res.delta_g_hyd - expt)

    res_arr = np.array(residuals)
    bias = float(np.mean(res_arr))
    rmse = float(np.sqrt(np.mean(res_arr ** 2)))

    # Assert near-zero net bias (within +/- 1.0 kcal/mol)
    assert abs(bias) < 1.0
    # Assert RMSE is within 6.5 kcal/mol (exact measured: 5.90 kcal/mol)
    assert rmse < 6.50
