"""Analytical Born continuum electrostatic correction for net-charged species."""

import math
from typing import Optional
from rdkit import Chem
from rdkit.Chem import AllChem


COULOMB_CONSTANT_KCAL = 332.0637  # e^2 / (4 * pi * eps0) in kcal * A / mol


def compute_effective_born_radius(mol: Chem.Mol, probe_radius: float = 0.10) -> float:
    """Computes effective Born radius R_eff from molecular vdW volume.

    R_eff = (3 * V_vdW / (4 * pi))^(1/3) + probe_radius
    """
    try:
        # If conformer exists, compute vdW volume
        if mol.GetNumConformers() > 0:
            vol = AllChem.ComputeMolVolume(mol)
        else:
            # Fallback: estimate from sum of atomic vdW spheres with 0.6 overlap packing fraction
            vdw_radii = {
                1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47,
                15: 1.80, 16: 1.80, 17: 1.75, 35: 1.85, 53: 1.98,
            }
            vol = sum((4.0 / 3.0) * math.pi * (vdw_radii.get(a.GetAtomicNum(), 1.70) ** 3)
                      for a in mol.GetAtoms()) * 0.65
    except Exception:
        vol = 100.0  # Fallback typical small molecule volume in A^3

    r_eff = ((3.0 * vol) / (4.0 * math.pi)) ** (1.0 / 3.0) + probe_radius
    return max(r_eff, 1.20)  # Lower bound: at least single atom radius


def compute_born_ion_correction(
    mol: Chem.Mol,
    epsilon_r: float = 78.4,
    grid_dimension_angstrom: float = 24.0,
) -> float:
    """Computes the analytical Born solvation free energy correction for net-charged solutes.

    For neutral solutes (Q = 0), this returns 0.0 kcal/mol identically.

    For net-charged solutes (Q = +/- 1, 2, ...):
    1. Evaluates Born self-energy:
       Delta G_Born = - (Q^2 * e^2 / (8 * pi * eps0 * R_eff)) * (1 - 1 / eps_r)
    2. Subtracts finite-grid boundary truncation error so the continuum grid
       does not double-count or unphysically truncate long-range ion reaction fields:
       Delta G_boundary = - (Q^2 * e^2 / (8 * pi * eps0 * R_box)) * (1 - 1 / eps_r)

    The net correction Delta G_corr = Delta G_Born - Delta G_box.

    Returns:
        correction_kcal: Energy contribution in kcal/mol (negative for net ions, 0.0 for neutrals).
    """
    formal_charge = Chem.GetFormalCharge(mol)
    if formal_charge == 0:
        return 0.0

    q = float(formal_charge)
    r_eff = compute_effective_born_radius(mol)
    r_box = max(grid_dimension_angstrom / 2.0, r_eff + 1.0)

    dielectric_factor = 1.0 - (1.0 / float(epsilon_r))
    born_factor = (COULOMB_CONSTANT_KCAL / 2.0) * (q ** 2) * dielectric_factor

    # Long-range ion reaction field outside grid boundary
    delta_g_outside = - born_factor / r_box

    # Combined analytical Born contribution for net ion
    # Total Born is - born_factor / r_eff.
    # The grid captures the field inside r_box; the analytical correction
    # accounts for the missing exterior dielectric response outside the finite domain.
    return float(delta_g_outside)
