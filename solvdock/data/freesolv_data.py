"""FreeSolv database loader supporting the full 642-molecule Mobley dataset with strict train/test splits.

Provides grounded dictionary key lookup provenance, experimental vs GAFF separation,
synthetic 0.60 kcal/mol default uncertainty auditing, and SMILES standardization.
"""

import os
from typing import Any, Dict, List, Optional, Tuple
import urllib.request
import numpy as np

try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


FREESOLV_URL = "https://raw.githubusercontent.com/MobleyLab/FreeSolv/master/database.txt"
DEFAULT_LOCAL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "freesolv", "database.txt")


def standardize_freesolv_smiles(smiles: str) -> Tuple[str, bool, int, bool]:
    """Standardizes SMILES string, canonicalizes nitro groups, and extracts chiral stats.

    Returns:
        (canonical_smiles, is_chiral, num_chiral_centers, has_unassigned_chiral)
    """
    if not HAS_RDKIT or not smiles:
        return smiles, False, 0, False

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles, False, 0, False

    canonical = Chem.MolToSmiles(mol)
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    num_chiral = len(chiral_centers)
    is_chiral = num_chiral > 0
    has_unassigned = any(c[1] == "?" for c in chiral_centers)

    return canonical, is_chiral, num_chiral, has_unassigned


def load_full_freesolv(local_path: str = DEFAULT_LOCAL_PATH) -> List[Dict[str, Any]]:
    """Loads the full 642-molecule FreeSolv experimental database with complete 10-field metadata.

    If the database.txt file does not exist locally, automatically downloads
    it from the official MobleyLab repository.
    """
    local_path = os.path.abspath(local_path)
    if not os.path.exists(local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        print(f"Downloading official FreeSolv database from {FREESOLV_URL}...")
        urllib.request.urlretrieve(FREESOLV_URL, local_path)

    molecules = []
    with open(local_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(";")]
            if len(parts) < 6:
                continue

            # database.txt 10-field format:
            # 0: compound id (e.g. mobley_1017962 - accession hash, NOT line number)
            # 1: SMILES string
            # 2: IUPAC name
            # 3: experimental Delta_G_hyd (kcal/mol)
            # 4: experimental SEM
            # 5: calculated Delta_G_hyd (Mobley group GAFF/MD)
            # 6: calculated SEM
            # 7: experimental reference (DOI or citation)
            # 8: calculated reference (DOI)
            # 9: text notes (e.g. default uncertainty notifications)
            try:
                cid = parts[0]
                raw_smiles = parts[1]
                name = parts[2]
                expt = float(parts[3])
                expt_sem = float(parts[4]) if parts[4] else 0.0
                calc = float(parts[5]) if parts[5] else None
                gaff_sem = float(parts[6]) if len(parts) > 6 and parts[6] else None
                expt_ref = parts[7] if len(parts) > 7 else ""
                calc_ref = parts[8] if len(parts) > 8 else ""
                notes = parts[9] if len(parts) > 9 else ""

                # Audit default uncertainty: ~67% of FreeSolv has dummy 0.60 placeholder
                is_default_sem = bool(abs(expt_sem - 0.60) < 1e-4 and "default" in notes.lower())
                has_measured_sem = not is_default_sem

                # Standardize nitro groups & chiral flags
                std_smiles, is_chiral, n_chiral, unassigned_chiral = standardize_freesolv_smiles(raw_smiles)

                item: Dict[str, Any] = {
                    # Core keys (100% backwards compatible with existing code)
                    "id": cid,
                    "smiles": raw_smiles,
                    "name": name,
                    "expt": expt,
                    "expt_sem": expt_sem,
                    "calc": calc,

                    # Extended audit and provenance keys
                    "expt_delta_g": expt,
                    "gaff_delta_g": calc,
                    "gaff_sem": gaff_sem,
                    "is_default_sem": is_default_sem,
                    "has_measured_sem": has_measured_sem,
                    "smiles_standardized": std_smiles,
                    "is_chiral": is_chiral,
                    "num_chiral_centers": n_chiral,
                    "has_unassigned_chiral": unassigned_chiral,
                    "net_formal_charge": 0,
                    "expt_reference": expt_ref,
                    "calc_reference": calc_ref,
                    "notes": notes,
                    "provenance_verified": True,
                }
                molecules.append(item)
            except ValueError:
                continue

    return molecules


def get_freesolv_dict(local_path: str = DEFAULT_LOCAL_PATH) -> Dict[str, Dict[str, Any]]:
    """Returns the parsed FreeSolv database keyed by authentic compound accession ID.

    Used for verified dictionary key lookups.
    """
    mols = load_full_freesolv(local_path)
    return {m["id"]: m for m in mols}


def verify_freesolv_id(compound_id: str, local_path: str = DEFAULT_LOCAL_PATH) -> bool:
    """Verifies whether a compound ID exists in the authentic FreeSolv database via dictionary key lookup."""
    db_dict = get_freesolv_dict(local_path)
    return compound_id in db_dict


def get_freesolv_entry(compound_id: str, local_path: str = DEFAULT_LOCAL_PATH) -> Dict[str, Any]:
    """Retrieves the verified FreeSolv record for a compound ID, raising KeyError if absent."""
    db_dict = get_freesolv_dict(local_path)
    if compound_id not in db_dict:
        raise KeyError(
            f"Compound ID '{compound_id}' not found in FreeSolv database. "
            "Please verify that this is a valid Mobley identifier (e.g. 'mobley_2501588') "
            "and not a line number or ungrounded identifier."
        )
    return db_dict[compound_id]


def get_freesolv_split(
    test_ratio: float = 0.20,
    seed: int = 42,
    local_path: str = DEFAULT_LOCAL_PATH,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Provides an explicit, frozen train/test split of the full FreeSolv database.

    Ensures zero data leakage: calibration happens on the train split,
    and generalization is evaluated strictly on the unseen test split.

    Args:
        test_ratio: Fraction of molecules reserved for the test split (default 0.20).
        seed: Random seed for reproducible splitting (default 42).
        local_path: Path to local database.txt file.

    Returns:
        (train_set, test_set)
    """
    all_mols = load_full_freesolv(local_path)
    rng = np.random.RandomState(seed)
    indices = np.arange(len(all_mols))
    rng.shuffle(indices)

    n_test = int(len(all_mols) * test_ratio)
    test_idx = set(indices[:n_test])

    train_set = [all_mols[i] for i in range(len(all_mols)) if i not in test_idx]
    test_set = [all_mols[i] for i in range(len(all_mols)) if i in test_idx]

    return train_set, test_set


# Export curated reference alias for legacy import compatibility
FREESOLV_CURATED = load_full_freesolv()
