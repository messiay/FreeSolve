"""Analytical Born continuum electrostatic model for ions and net-charged species."""

import math
from typing import Optional, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem

# Coulomb constant in kcal * A / (mol * e^2)
# 1 / (4 * pi * eps_0) = 332.0637085 kcal * A / mol
COULOMB_CONSTANT_KCAL = 332.0637


def compute_born_radius(mol: Chem.Mol, offset_angstrom: float = 0.0) -> float:
    """Computes effective Born radius R_eff (in Angstroms) from molecular vdW volume.

    R_eff = (3 * V_vdW / (4 * pi))^(1/3) + offset_angstrom
    """
    try:
        if mol.GetNumConformers() > 0:
            vol = AllChem.ComputeMolVolume(mol)
        else:
            vdw_radii = {
                1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47,
                11: 1.02, 12: 0.72, 15: 1.80, 16: 1.80, 17: 1.75,
                19: 1.38, 20: 1.00, 35: 1.85, 53: 1.98,
            }
            vol = sum((4.0 / 3.0) * math.pi * (vdw_radii.get(a.GetAtomicNum(), 1.70) ** 3)
                      for a in mol.GetAtoms()) * 0.65
    except Exception:
        vol = 50.0

    r_eff = ((3.0 * vol) / (4.0 * math.pi)) ** (1.0 / 3.0) + offset_angstrom
    return max(r_eff, 1.0)


def compute_textbook_born_energy(
    net_charge: float,
    born_radius_angstrom: float,
    epsilon_r: float = 78.4,
) -> float:
    """Computes textbook Born electrostatic hydration free energy (kcal/mol).

    Delta G_Born = - (Q^2 * e^2 / (8 * pi * eps_0 * R)) * (1 - 1 / eps_r)
                 = - (332.0637 / 2) * (Q^2 / R) * (1 - 1 / eps_r)
                 = - 163.914 * (Q^2 / R)  [for eps_r = 78.4]

    Reference values (T = 298.15 K, eps_r = 78.4):
        - Single sphere R = 2.0 A, Q = 1:  Delta G_Born = -81.96 kcal/mol
        - Na+ (R = 1.68 A, Q = +1):         Delta G_Born = -97.57 kcal/mol
        - Cl- (R = 1.95 A, Q = -1):         Delta G_Born = -84.06 kcal/mol
        - Acetate (R = 2.366 A, Q = -1):    Delta G_Born = -69.28 kcal/mol
    """
    if abs(net_charge) < 1e-6:
        return 0.0

    dielectric_factor = 1.0 - (1.0 / float(epsilon_r))
    born_prefactor = (COULOMB_CONSTANT_KCAL / 2.0) * dielectric_factor  # 163.914 kcal * A / mol
    return - float(born_prefactor * (net_charge ** 2) / float(born_radius_angstrom))


def compute_ion_solvation_estimate(
    mol: Chem.Mol,
    epsilon_r: float = 78.4,
) -> Tuple[float, float, float]:
    """Evaluates the Born electrostatic solvation energy for a net-charged solute.

    Returns:
        (delta_g_born, born_radius, net_charge)
    """
    q = float(Chem.GetFormalCharge(mol))
    if abs(q) < 1e-6:
        return 0.0, 0.0, 0.0

    r_eff = compute_born_radius(mol)
    dG_born = compute_textbook_born_energy(q, r_eff, epsilon_r=epsilon_r)
    return dG_born, r_eff, q
