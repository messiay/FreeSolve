"""PoseBusters physical plausibility benchmark for ligand poses."""

import argparse
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from solvdock.pipeline.refiner import SolvDockRefiner


# Novel scaffold test molecules (unusual rings, spiro systems, macrocyclic elements)
NOVEL_SCAFFOLD_MOLECULES = [
    ("spiro_adamantane", "O=C1CC2(CC3CC(C2)CC1C3)N4CCOCC4"),
    ("bridged_bicyclic", "CC12CCC(CC1)C(=O)N2c3ccccc3"),
    ("novel_oxadiazole", "c1ccc(cc1)c2nc(no2)C3CC3"),
    ("bicyclo_octane", "O=C(O)C12CCC(CC1)(CC2)C(=O)N"),
    ("macrocycle_mimic", "C1CCCCCCCC(=O)NCC1"),
    ("spiro_piperidine", "O=C1Nc2ccccc2C13CCNCC3"),
    ("indane_sulfonamide", "NS(=O)(=O)c1ccc2CC(O)Cc2c1"),
    ("fluorinated_heterocycle", "FC(F)(F)c1nc(cs1)c2cccnc2"),
]

STANDARD_SCAFFOLD_MOLECULES = [
    ("benzimidazole", "c1ccc2[nH]cnc2c1"),
    ("quinazolinedione", "O=C1NC(=O)c2ccccc2N1"),
    ("phenyl_piperazine", "c1ccccc1N2CCNCC2"),
    ("biphenyl_carboxylic", "OC(=O)c1ccc(cc1)c2ccccc2"),
    ("sulfonamide_aniline", "Nc1ccc(cc1)S(=O)(=O)Nc2ccccc2"),
    ("coumarin_amide", "O=C(Nc1ccccc1)c2ccc3c(c2)cco3"),
]


def check_steric_clashes(mol: Chem.Mol, clash_tolerance: float = 0.70) -> bool:
    """Passes if no non-bonded atom pairs are closer than tolerance * sum(vdW radii)."""
    conf = mol.GetConformer()
    num_atoms = mol.GetNumAtoms()

    vdw = {
        1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47,
        15: 1.80, 16: 1.80, 17: 1.75, 35: 1.85,
    }

    pts = [conf.GetAtomPosition(i) for i in range(num_atoms)]

    # Collect 1-2, 1-3, and 1-4 bonded pairs to exclude from clash checks
    bonded = set()
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonded.add((min(i, j), max(i, j)))

    for i in range(num_atoms):
        zi = mol.GetAtomWithIdx(i).GetAtomicNum()
        ri = vdw.get(zi, 1.70)
        for j in range(i + 1, num_atoms):
            if (i, j) in bonded:
                continue
            # Also skip 1-3 pairs
            a_i = mol.GetAtomWithIdx(i)
            if any(nbr.GetIdx() == j for nbr in a_i.GetNeighbors()):
                continue

            zj = mol.GetAtomWithIdx(j).GetAtomicNum()
            rj = vdw.get(zj, 1.70)
            threshold = clash_tolerance * (ri + rj)

            pi, pj = pts[i], pts[j]
            dist = np.sqrt((pi.x - pj.x)**2 + (pi.y - pj.y)**2 + (pi.z - pj.z)**2)
            if dist < threshold:
                return False
    return True


def check_bond_lengths(mol: Chem.Mol, max_dev: float = 0.20) -> bool:
    """Passes if all bond lengths are within max_dev of standard equilibrium lengths."""
    conf = mol.GetConformer()
    # Standard single bond ~1.50 A, double ~1.34 A, aromatic ~1.39 A, C-H ~1.09 A
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        pi, pj = conf.GetAtomPosition(i), conf.GetAtomPosition(j)
        dist = np.sqrt((pi.x - pj.x)**2 + (pi.y - pj.y)**2 + (pi.z - pj.z)**2)
        if dist < 0.80 or dist > 2.10:
            return False
    return True


def check_aromatic_planarity(mol: Chem.Mol, max_rmsd: float = 0.15) -> bool:
    """Passes if aromatic rings do not deviate significantly from planarity."""
    conf = mol.GetConformer()
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if len(ring) in (5, 6):
            # Check if all atoms in ring are aromatic
            if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
                pts = np.array([[conf.GetAtomPosition(idx).x,
                                 conf.GetAtomPosition(idx).y,
                                 conf.GetAtomPosition(idx).z] for idx in ring])
                # Fit best-plane via SVD
                center = pts.mean(axis=0)
                centered = pts - center
                _, s, vh = np.linalg.svd(centered)
                normal = vh[2]
                dist_to_plane = np.abs(np.dot(centered, normal))
                plane_rmsd = np.sqrt(np.mean(dist_to_plane ** 2))
                if plane_rmsd > max_rmsd:
                    return False
    return True


def evaluate_posebusters(
    novel_scaffold_subset: bool = False,
    device: str = "cpu",
    constants_path: str = "configs/calibrated_constants.yaml",
    checkpoint_path: str = "checkpoints/residual_mlp.pt",
):
    """Evaluates physical plausibility according to the PoseBusters benchmark."""
    subset_name = "Novel-Scaffold Subset" if novel_scaffold_subset else "Standard Set"
    print("=" * 68)
    print(f"POSEBUSTERS PHYSICAL PLAUSIBILITY BENCHMARK ({subset_name})")
    print("=" * 68)

    refiner = SolvDockRefiner(
        constants_path=constants_path,
        checkpoint_path=checkpoint_path,
        device=device,
        strict=True,
    )

    mol_list = NOVEL_SCAFFOLD_MOLECULES if novel_scaffold_subset else STANDARD_SCAFFOLD_MOLECULES

    raw_clash_pass = 0
    refined_clash_pass = 0
    raw_bond_pass = 0
    refined_bond_pass = 0
    raw_planarity_pass = 0
    refined_planarity_pass = 0
    total = len(mol_list)

    for name, smiles in mol_list:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        status = AllChem.EmbedMolecule(mol, randomSeed=42)
        if status != 0 or mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)
        AllChem.UFFOptimizeMolecule(mol)
        AllChem.ComputeGasteigerCharges(mol)

        # Raw pose evaluation
        if check_steric_clashes(mol):
            raw_clash_pass += 1
        if check_bond_lengths(mol):
            raw_bond_pass += 1
        if check_aromatic_planarity(mol):
            raw_planarity_pass += 1

        # Refine pose
        res = refiner.refine_pose(mol, max_steps=15, lr=0.03)
        ref_mol = res["mol"]

        if check_steric_clashes(ref_mol):
            refined_clash_pass += 1
        if check_bond_lengths(ref_mol):
            refined_bond_pass += 1
        if check_aromatic_planarity(ref_mol):
            refined_planarity_pass += 1

    print(f"Results for {total} molecules in {subset_name}:")
    print("-" * 68)
    print(f"{'Metric':<30} | {'Raw Input (%)':<16} | {'SolvDock Refined (%)':<18}")
    print("-" * 68)
    print(f"{'Steric Clash Test':<30} | {raw_clash_pass/total*100:<16.1f} | {refined_clash_pass/total*100:<18.1f}")
    print(f"{'Bond Length Validity':<30} | {raw_bond_pass/total*100:<16.1f} | {refined_bond_pass/total*100:<18.1f}")
    print(f"{'Aromatic Planarity Test':<30} | {raw_planarity_pass/total*100:<16.1f} | {refined_planarity_pass/total*100:<18.1f}")
    print("=" * 68)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PoseBusters plausibility evaluation.")
    parser.add_argument("--novel_scaffold_subset", action="store_true", help="Run on novel-scaffold subset.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--constants", type=str, default="configs/calibrated_constants.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/residual_mlp.pt")
    args = parser.parse_args()

    evaluate_posebusters(
        novel_scaffold_subset=args.novel_scaffold_subset,
        device=args.device,
        constants_path=args.constants,
        checkpoint_path=args.checkpoint,
    )
