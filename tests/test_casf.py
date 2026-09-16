"""Tests for CASF-2016 Core Set data loading, manifest integrity, and structure caching."""

import os
from collections import Counter
import pytest
from rdkit import Chem
from Bio.PDB import PDBParser

from solvdock.data.casf_loader import (
    load_casf_coreset_manifest,
    load_all_casf_complexes,
)


def test_casf_coreset_manifest_completeness():
    """Validates that CoreSet.dat contains exactly 285 complexes across 57 clusters."""
    manifest = load_casf_coreset_manifest("data/casf2016/CoreSet.dat")
    assert len(manifest) == 285, f"Expected 285 complexes in CoreSet.dat, found {len(manifest)}"

    # Check that all 57 target clusters have exactly 5 complexes each
    target_counts = Counter(entry["target_id"] for entry in manifest)
    assert len(target_counts) == 57, f"Expected 57 target clusters, found {len(target_counts)}"
    for target_id, count in target_counts.items():
        assert count == 5, f"Target cluster {target_id} has {count} complexes (expected 5)"

    # Check experimental affinity bounds (pKd typically 2.0 to 12.0)
    for entry in manifest:
        pkd = entry["pkd"]
        assert 1.5 <= pkd <= 13.0, f"PDB {entry['pdb_id']} has unusual pKd: {pkd}"
        assert entry["resl"] > 0.0, f"PDB {entry['pdb_id']} missing resolution"
        assert entry["year"] >= 1980, f"PDB {entry['pdb_id']} invalid year: {entry['year']}"


def test_casf_complexes_cached_and_loadable():
    """Validates that all 285 complexes are cached locally on disk and load correctly."""
    complexes = load_all_casf_complexes()
    assert len(complexes) == 285, f"Expected 285 cached complexes, found {len(complexes)}"

    # Spot check representative complexes
    parser = PDBParser(QUIET=True)
    for test_id in ["4llx", "1e66", "1a30", "2c3i", "5dwr"]:
        match = next((c for c in complexes if c["pdb_id"] == test_id), None)
        assert match is not None, f"Complex {test_id} not found in loaded complexes"

        pocket_path = match["pocket_path"]
        ligand_path = match["ligand_path"]
        assert os.path.exists(pocket_path), f"Pocket file missing for {test_id}: {pocket_path}"
        assert os.path.exists(ligand_path), f"Ligand file missing for {test_id}: {ligand_path}"

        # Check pocket structure
        struct = parser.get_structure(test_id, pocket_path)
        residues = [r for r in struct.get_residues() if r.id[0] == " "]
        assert len(residues) >= 10, f"Pocket {test_id} has only {len(residues)} residues"

        # Check ligand molecule
        suppl = Chem.SDMolSupplier(ligand_path, removeHs=False)
        mol = suppl[0] if len(suppl) > 0 else None
        assert mol is not None, f"Failed loading ligand SDF for {test_id}"
        assert mol.GetNumAtoms() >= 4, f"Ligand {test_id} has too few atoms ({mol.GetNumAtoms()})"
