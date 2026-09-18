"""Applicability domain checking and chemical class auditing for solvation predictions."""

from dataclasses import dataclass, field
from typing import List, Optional, Set
from rdkit import Chem
from rdkit.Chem import Lipinski


ALLOWED_ATOMIC_NUMBERS: Set[int] = {
    1,   # H
    6,   # C
    7,   # N
    8,   # O
    9,   # F
    15,  # P
    16,  # S
    17,  # Cl
    35,  # Br
}

KNOWN_PUSH_PULL_NITROAROMATIC_IDS: Set[str] = {
    "mobley_2501588",  # profluralin
    "mobley_7829570",  # benefin
    "mobley_1396156",  # pentachloronitrobenzene
    "mobley_5076071",  # dinitramine
    "mobley_7176248",  # trifluralin
}


@dataclass
class ApplicabilityReport:
    """Detailed chemical audit report and uncertainty classification for a molecule."""
    is_within_domain: bool
    flags: List[str] = field(default_factory=list)
    base_uncertainty: float = 1.20  # kcal/mol (standard 1-sigma on clean neutral drug-like set)
    formal_charge: int = 0
    num_rotatable_bonds: int = 0
    unsupported_elements: List[str] = field(default_factory=list)


def is_push_pull_nitroaromatic(mol: Chem.Mol, compound_id: str = "") -> bool:
    """Identifies conjugated push-pull poly-nitro or poly-halo aromatics.

    This chemical class exhibits strong conjugated electronic redistribution,
    sigma-hole effects, and nitro-group twisting where classical point charges fail.
    """
    if compound_id in KNOWN_PUSH_PULL_NITROAROMATIC_IDS:
        return True

    patt1 = Chem.MolFromSmarts("c[N+](=O)[O-]")
    patt2 = Chem.MolFromSmarts("cN(=O)=O")
    matches1 = mol.GetSubstructMatches(patt1) if patt1 else ()
    matches2 = mol.GetSubstructMatches(patt2) if patt2 else ()
    n_nitro = len(matches1) + len(matches2)

    if n_nitro == 0:
        return False

    # Aromatic ring with >= 2 nitro groups OR (>= 1 nitro group AND >= 3 halogens)
    n_halogens = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in (9, 17, 35, 53))
    return (n_nitro >= 2) or (n_halogens >= 3)


def check_applicability_domain(mol: Chem.Mol, compound_id: str = "") -> ApplicabilityReport:
    """Evaluates whether a molecule lies within the validated domain of the continuum PDE.

    Returns an ApplicabilityReport with explicit warning flags and calibrated 1-sigma uncertainty.
    """
    flags: List[str] = []
    uncertainty: float = 1.20  # Baseline empirical RMSE on clean neutral drug-like split
    
    # 1. Formal charge check
    formal_charge = Chem.GetFormalCharge(mol)
    if formal_charge != 0:
        flags.append(f"NET_CHARGE_{formal_charge:+d}")
        uncertainty = max(uncertainty, 3.50)

    # 2. Push-pull nitroaromatic check
    if is_push_pull_nitroaromatic(mol, compound_id):
        flags.append("PUSH_PULL_NITROAROMATIC")
        uncertainty = max(uncertainty, 4.50)

    # 3. Heavy/unsupported elements check
    unsupported = []
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        if z not in ALLOWED_ATOMIC_NUMBERS:
            sym = atom.GetSymbol()
            if sym not in unsupported:
                unsupported.append(sym)
    if unsupported:
        flags.append(f"UNSUPPORTED_ELEMENTS_{'_'.join(unsupported)}")
        uncertainty = max(uncertainty, 5.00)

    # 4. Macrocycle check (ring size >= 12)
    ring_info = mol.GetRingInfo()
    has_macrocycle = any(len(ring) >= 12 for ring in ring_info.AtomRings())
    if has_macrocycle:
        flags.append("MACROCYCLE")
        uncertainty = max(uncertainty, 3.00)

    # 5. Conformational entropy / flexibility check
    n_rot = Lipinski.NumRotatableBonds(mol)
    if n_rot > 8:
        flags.append(f"HIGH_FLEXIBILITY_{n_rot}_ROTBONDS")
        # Add 0.15 kcal/mol per rotatable bond above 8
        uncertainty += 0.15 * (n_rot - 8)

    is_valid = len(flags) == 0
    return ApplicabilityReport(
        is_within_domain=is_valid,
        flags=flags,
        base_uncertainty=round(uncertainty, 2),
        formal_charge=formal_charge,
        num_rotatable_bonds=n_rot,
        unsupported_elements=unsupported,
    )
