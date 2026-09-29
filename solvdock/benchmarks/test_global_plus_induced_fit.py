"""End-to-End Simulation: Global Basin-Hopping Monte Carlo Docking + Induced-Fit Sidechain Refinement."""

import time
import os
import torch
from rdkit import Chem
from solvdock.pipeline.refiner import SolvDockRefiner
from solvdock.pipeline.flexible_refiner import FlexibleRefiner
from solvdock.benchmarks.flexible_docking_benchmark import compute_rmsd


def main():
    cid = '2v00'
    lig_path = f'data/casf2016_core/{cid}_ligand.sdf'
    poc_path = f'data/casf2016_core/{cid}_pocket.pdb'

    print("=" * 80)
    print(f"=== BLIND GLOBAL DOCKING + INDUCED-FIT PIPELINE SIMULATION: {cid} ===")
    print("=" * 80)

    # 1. Load ground truth crystal structure
    cryst_lig = Chem.SDMolSupplier(lig_path, removeHs=False)[0]
    n_lig = cryst_lig.GetNumAtoms()
    conf_cryst = cryst_lig.GetConformer()
    cryst_coords = torch.tensor(
        [[conf_cryst.GetAtomPosition(i).x, conf_cryst.GetAtomPosition(i).y, conf_cryst.GetAtomPosition(i).z] for i in range(n_lig)],
        dtype=torch.float32,
    )

    # 2. Stage 1: Global Basin-Hopping Docking
    refiner = SolvDockRefiner(strict=False, disable_residual_mlp=True)

    print("\n[Stage 1] Running Global Basin-Hopping Monte Carlo Docking (8 trials)...")
    t0 = time.time()
    res_global = refiner.dock_global(
        ligand_input=lig_path,
        pocket_input=poc_path,
        n_trials=8,
        local_steps=6,
        seed=42,
        charge_model='gasteiger',
    )
    t_global = time.time() - t0

    conf_global = res_global.best_mol.GetConformer()
    coords_global = torch.tensor(
        [[conf_global.GetAtomPosition(i).x, conf_global.GetAtomPosition(i).y, conf_global.GetAtomPosition(i).z] for i in range(n_lig)],
        dtype=torch.float32,
    )
    rmsd_stage1 = compute_rmsd(coords_global, cryst_coords)
    print(f"  Stage 1 Completed in {t_global:.2f}s")
    print(f"  Best Pose Energy: {res_global.best_delta_G_bind:.2f} kcal/mol")
    print(f"  Stage 1 Ligand RMSD to Crystal: {rmsd_stage1:.3f} A")
    print(f"  Distinct Docking Modes Found: {len(res_global.top_modes)}")

    # 3. Stage 2: Induced-Fit Sidechain Relaxation
    print("\n[Stage 2] Unlocking Pocket Side-Chains for Induced-Fit Refinement...")
    flex_refiner = FlexibleRefiner(potential=refiner.potential, k_backbone=50.0)
    poc_mol = Chem.MolFromPDBFile(poc_path, removeHs=False)
    bb_mask = flex_refiner.identify_backbone_mask(poc_mol)
    n_poc = poc_mol.GetNumAtoms()
    n_bb = int(bb_mask.sum().item())
    n_sc = n_poc - n_bb
    print(f"  Pocket atoms: {n_poc} ({n_bb} backbone held rigid, {n_sc} flexible side-chain atoms)")

    t1 = time.time()
    res_flex = flex_refiner.refine_induced_fit(
        ligand_mol=res_global.best_mol,
        pocket_mol=poc_mol,
        backbone_mask=bb_mask,
        freeze_backbone=True,
        max_steps=40,
        lr=0.05,
    )
    t_flex = time.time() - t1

    rmsd_stage2 = compute_rmsd(res_flex["final_ligand_coords"], cryst_coords)
    print(f"  Stage 2 Completed in {t_flex:.2f}s")
    print(f"  Direct Energy: {res_flex['initial_direct_energy']:.2f} -> {res_flex['final_direct_energy']:.2f} kcal/mol (Delta = {res_flex['final_direct_energy'] - res_flex['initial_direct_energy']:.2f})")
    print(f"  Side-Chain Induced RMSD: {res_flex['sidechain_rmsd']:.3f} A (Max local displacement: {res_flex['max_sidechain_displacement']:.3f} A)")
    print(f"  Backbone RMSD: {res_flex['backbone_rmsd']:.4f} A (Scaffold 100% preserved)")
    print(f"  Final Ligand RMSD to Crystal: {rmsd_stage2:.3f} A")

    # 4. Save Final Complex PDB
    out_dir = "data/docking"
    os.makedirs(out_dir, exist_ok=True)
    out_pdb = os.path.join(out_dir, f"{cid}_global_induced_fit_final.pdb")

    # Update ligand and pocket conformers
    final_lig_mol = Chem.Mol(res_global.best_mol)
    conf_lig_f = final_lig_mol.GetConformer()
    for i in range(n_lig):
        p = res_flex["final_ligand_coords"][i].numpy()
        conf_lig_f.SetAtomPosition(i, Chem.rdGeometry.Point3D(float(p[0]), float(p[1]), float(p[2])))

    final_poc_mol = Chem.Mol(poc_mol)
    conf_poc_f = final_poc_mol.GetConformer()
    for i in range(n_poc):
        p = res_flex["final_pocket_coords"][i].numpy()
        conf_poc_f.SetAtomPosition(i, Chem.rdGeometry.Point3D(float(p[0]), float(p[1]), float(p[2])))

    # Write combined PDB
    combined = Chem.CombineMols(final_poc_mol, final_lig_mol)
    with Chem.PDBWriter(out_pdb) as writer:
        writer.write(combined)
    print(f"\n[Artifact Saved] Final Docked Complex written to {out_pdb}")
    print("=" * 80)


if __name__ == "__main__":
    main()
