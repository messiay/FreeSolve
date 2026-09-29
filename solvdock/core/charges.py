"""Unified atomic partial charge engine supporting MMFF94, Gasteiger, AM1-BCC, and user-preserved charges."""

import json
import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem


_FREESOLV_AM1BCC_CACHE: Optional[Dict[str, List[float]]] = None


def get_freesolv_am1bcc_cache() -> Dict[str, List[float]]:
    """Loads the precomputed authentic FreeSolv AM1-BCC charges cache."""
    global _FREESOLV_AM1BCC_CACHE
    if _FREESOLV_AM1BCC_CACHE is None:
        path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "freesolv", "freesolv_am1bcc.json")
        path = os.path.normpath(path)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                _FREESOLV_AM1BCC_CACHE = json.load(f)
        else:
            _FREESOLV_AM1BCC_CACHE = {}
    return _FREESOLV_AM1BCC_CACHE


def has_existing_charges(mol: Chem.Mol) -> bool:
    """Checks if the molecule already has explicit partial charges assigned to all atoms."""
    for atom in mol.GetAtoms():
        has_prop = any(
            atom.HasProp(prop)
            for prop in ("_AM1BCCCharge", "_PartialCharge", "_MMFFCharge", "_GasteigerCharge", "PartialCharge")
        )
        if not has_prop:
            return False
    return True


def get_partial_charges(
    mol: Chem.Mol,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """Extracts partial charges from an RDKit Mol into a PyTorch tensor.

    Prioritizes charges in the order:
    1. `_AM1BCCCharge` (quantum/semi-empirical)
    2. `_PartialCharge` (explicit user-supplied or loaded from mol2/sdf)
    3. `_MMFFCharge`
    4. `_GasteigerCharge`
    5. Fallback 0.0

    Guards strictly against NaN or Inf values by zeroing them.
    """
    num_atoms = mol.GetNumAtoms()
    charges = torch.zeros((num_atoms,), dtype=torch.float32, device=device)

    for i, atom in enumerate(mol.GetAtoms()):
        q = None
        for prop in ("_AM1BCCCharge", "_PartialCharge", "_MMFFCharge", "_GasteigerCharge", "PartialCharge"):
            if atom.HasProp(prop):
                try:
                    val = float(atom.GetProp(prop))
                    if np.isnan(val) or np.isinf(val):
                        val = 0.0
                    q = val
                    break
                except (ValueError, KeyError):
                    continue
        charges[i] = q if q is not None else 0.0

    return charges


def assign_charges(
    mol: Chem.Mol,
    scheme: str = "mmff94",
    preserve_existing: bool = True,
    compound_id: Optional[str] = None,
    am1bcc_cache: Optional[Dict[str, List[float]]] = None,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[torch.Tensor, str]:
    """Assigns partial charges to an RDKit Mol according to the requested scheme.

    Args:
        mol: RDKit Mol object (will be annotated in-place with atom properties).
        scheme: Charge scheme ('preserve', 'mmff94', 'gasteiger', 'am1bcc').
        preserve_existing: If True and mol already has full charges, respects them.
        compound_id: Optional identifier for cache lookup (e.g. FreeSolv 'mobley_XXXX').
        am1bcc_cache: Optional dictionary mapping compound_id to list of charges.
        device: PyTorch device for returned tensor.

    Returns:
        Tuple of (charges_tensor, scheme_used).
    """
    scheme = scheme.lower()
    num_atoms = mol.GetNumAtoms()

    # 1. Check if existing charges should be preserved
    if preserve_existing and scheme == "preserve" and has_existing_charges(mol):
        return get_partial_charges(mol, device=device), "preserved"

    # 2. AM1-BCC scheme
    if scheme == "am1bcc":
        cache = am1bcc_cache if am1bcc_cache is not None else get_freesolv_am1bcc_cache()
        cid = compound_id
        if cid is None and mol.HasProp("_FreeSolvID"):
            cid = mol.GetProp("_FreeSolvID")
        if cid is None and mol.HasProp("compound_id"):
            cid = mol.GetProp("compound_id")

        if cid is not None and cid in cache:
            q_list = cache[cid]
            if len(q_list) == num_atoms:
                for i, atom in enumerate(mol.GetAtoms()):
                    q_val = float(q_list[i])
                    atom.SetProp("_AM1BCCCharge", str(q_val))
                    atom.SetProp("_PartialCharge", str(q_val))
                return torch.tensor(q_list, dtype=torch.float32, device=device), "am1bcc"

        # If molecule already has _AM1BCCCharge properties on all atoms
        if all(a.HasProp("_AM1BCCCharge") for a in mol.GetAtoms()):
            return get_partial_charges(mol, device=device), "am1bcc"

        # Fallback to MMFF94 if AM1-BCC unavailable
        q_tensor, sub_scheme = assign_charges(mol, scheme="mmff94", preserve_existing=False, device=device)
        return q_tensor, f"am1bcc_fallback_{sub_scheme}"

    # 3. MMFF94 scheme
    if scheme == "mmff94":
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
        if mmff_props is not None:
            charges = torch.zeros((num_atoms,), dtype=torch.float32, device=device)
            for i, atom in enumerate(mol.GetAtoms()):
                q = float(mmff_props.GetMMFFPartialCharge(i))
                if np.isnan(q) or np.isinf(q):
                    q = 0.0
                atom.SetProp("_MMFFCharge", str(q))
                atom.SetProp("_PartialCharge", str(q))
                charges[i] = q
            return charges, "mmff94"
        else:
            # Fallback to Gasteiger if MMFF cannot parameterize
            q_tensor, _ = assign_charges(mol, scheme="gasteiger", preserve_existing=False, device=device)
            return q_tensor, "gasteiger_fallback"

    # 4. Gasteiger scheme
    if scheme == "gasteiger":
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            pass
        AllChem.ComputeGasteigerCharges(mol)
        charges = torch.zeros((num_atoms,), dtype=torch.float32, device=device)
        for i, atom in enumerate(mol.GetAtoms()):
            try:
                q = float(atom.GetProp("_GasteigerCharge"))
                if np.isnan(q) or np.isinf(q):
                    q = 0.0
            except KeyError:
                q = 0.0
            atom.SetProp("_GasteigerCharge", str(q))
            atom.SetProp("_PartialCharge", str(q))
            charges[i] = q
        return charges, "gasteiger"

    raise ValueError(f"Unknown charge scheme: '{scheme}'. Choose from 'preserve', 'mmff94', 'gasteiger', 'am1bcc'.")
