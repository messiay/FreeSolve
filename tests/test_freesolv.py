"""Tests for FreeSolv data loader, provenance verification, uncertainty auditing, and standardization."""

import os
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from solvdock.data.freesolv_data import (
    DEFAULT_LOCAL_PATH,
    load_full_freesolv,
    get_freesolv_dict,
    verify_freesolv_id,
    get_freesolv_entry,
    get_freesolv_split,
)


def test_database_line_count_and_provenance():
    """Verifies that database.txt contains 646 total lines (not millions) and confirms provenance lookup."""
    assert os.path.exists(DEFAULT_LOCAL_PATH), "database.txt must exist locally."

    with open(DEFAULT_LOCAL_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # database.txt has 3 header lines + 642 molecule records = 645 lines
    assert len(lines) == 645, f"Expected 645 lines in database.txt, found {len(lines)}"

    # Grounded dictionary key lookups
    assert verify_freesolv_id("mobley_2501588"), "mobley_2501588 (profluralin) must exist in database"
    assert verify_freesolv_id("mobley_7829570"), "mobley_7829570 (benefin) must exist in database"
    assert verify_freesolv_id("mobley_1017962"), "mobley_1017962 (methyl hexanoate) must exist in database"

    # Phantom/fabricated IDs must be rejected
    assert not verify_freesolv_id("mobley_930807"), "Fabricated ID mobley_930807 must NOT exist"
    assert not verify_freesolv_id("mobley_6367357"), "Fabricated ID mobley_6367357 must NOT exist"

    with pytest.raises(KeyError):
        get_freesolv_entry("mobley_930807")


def test_default_uncertainty_auditing():
    """Verifies that synthetic 0.60 kcal/mol default uncertainties are accurately flagged."""
    # Line 4: methyl hexanoate has synthetic 0.60 default uncertainty
    entry_def = get_freesolv_entry("mobley_1017962")
    assert entry_def["is_default_sem"] is True
    assert entry_def["has_measured_sem"] is False
    assert abs(entry_def["expt_sem"] - 0.60) < 1e-4

    # Line 24: triethylphosphate has measured 0.20 kcal/mol experimental SEM
    entry_meas = get_freesolv_entry("mobley_1323538")
    assert entry_meas["is_default_sem"] is False
    assert entry_meas["has_measured_sem"] is True
    assert abs(entry_meas["expt_sem"] - 0.20) < 1e-4


def test_expt_vs_gaff_separation():
    """Verifies that experimental ground truth and Amber GAFF calculations are stored in separate fields."""
    entry = get_freesolv_entry("mobley_1017962")
    # Field 3: Expt = -2.49 kcal/mol
    assert abs(entry["expt"] - (-2.49)) < 1e-4
    assert abs(entry["expt_delta_g"] - (-2.49)) < 1e-4

    # Field 5: Mobley GAFF calc = -3.30 kcal/mol
    assert abs(entry["calc"] - (-3.30)) < 1e-4
    assert abs(entry["gaff_delta_g"] - (-3.30)) < 1e-4


def test_neutral_protonation_preservation():
    """Verifies that carboxylic acids in FreeSolv are strictly neutral, preserving non-ionized state."""
    entry = get_freesolv_entry("mobley_1527293")  # flurbiprofen
    assert entry["name"] == "(2S)-2-(3-fluoro-4-phenyl-phenyl)propanoic acid" or "flurbiprofen" in entry["notes"].lower()
    assert abs(entry["expt"] - (-8.42)) < 1e-4

    mol = Chem.MolFromSmiles(entry["smiles"])
    assert mol is not None
    formal_charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    assert formal_charge == 0, "Flurbiprofen must be neutral carboxylic acid in FreeSolv"


def test_nitro_group_standardization():
    """Verifies that pentavalent uncharged nitro SMILES are standardized to zwitterionic [N+](=O)[O-]."""
    entry = get_freesolv_entry("mobley_1396156")  # pentachloronitrobenzene
    assert "N(=O)=O" in entry["smiles"]  # Raw representation in database.txt
    assert "[N+]([O-])" in entry["smiles_standardized"] or "[N+](=O)[O-]" in entry["smiles_standardized"]

    # Test MMFF94 parameterization on standardized molecule
    mol = Chem.MolFromSmiles(entry["smiles_standardized"])
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    mmff_props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
    assert mmff_props is not None, "MMFF94 parameterization must succeed on standardized nitro molecule"


def test_frozen_split_reproducibility():
    """Verifies that train/test split is exactly 514 / 128 and 100% reproducible."""
    train1, test1 = get_freesolv_split(seed=42)
    train2, test2 = get_freesolv_split(seed=42)

    assert len(train1) == 514
    assert len(test1) == 128
    assert [m["id"] for m in test1] == [m["id"] for m in test2]


def test_push_pull_nitroaromatic_substructure_filter():
    """Verifies that chemical substructure filter accurately identifies push-pull nitroaromatics."""
    from solvdock.benchmarks.freesolv_eval import is_poly_nitro_or_poly_halo_aromatic

    # Push-pull nitroaromatics (must be excluded from clean subset)
    mol_profluralin = Chem.MolFromSmiles(get_freesolv_entry("mobley_2501588")["smiles"])
    assert is_poly_nitro_or_poly_halo_aromatic(mol_profluralin, "mobley_2501588") is True

    mol_benefin = Chem.MolFromSmiles(get_freesolv_entry("mobley_7829570")["smiles"])
    assert is_poly_nitro_or_poly_halo_aromatic(mol_benefin, "mobley_7829570") is True

    mol_pentachloro = Chem.MolFromSmiles(get_freesolv_entry("mobley_1396156")["smiles"])
    assert is_poly_nitro_or_poly_halo_aromatic(mol_pentachloro, "mobley_1396156") is True

    # Simple mono-nitroaromatics (must NOT be excluded)
    mol_3nitrophenol = Chem.MolFromSmiles(get_freesolv_entry("mobley_7176290")["smiles"])
    assert is_poly_nitro_or_poly_halo_aromatic(mol_3nitrophenol, "mobley_7176290") is False

    mol_nitrotoluene = Chem.MolFromSmiles(get_freesolv_entry("mobley_7298388")["smiles"])
    assert is_poly_nitro_or_poly_halo_aromatic(mol_nitrotoluene, "mobley_7298388") is False
