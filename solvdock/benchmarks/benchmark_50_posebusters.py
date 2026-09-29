"""Official 50-Target PoseBusters Benchmark for SolvDock.

Evaluates SolvDock on 50 real co-crystal complexes from the official
PoseBusters Benchmark dataset (Buttenschoen et al., Chemical Science 2024).
"""

import json
import os
import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Geometry import Point3D
import posebusters

import solvdock
from solvdock.core.kinematics import axis_angle_to_matrix


def create_perturbed_ai_pose(mol: Chem.Mol, seed: int = 42) -> Chem.Mol:
    """Simulates an unminimized generative AI pose with pocket shift and rotation."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    pert_mol = Chem.Mol(mol)
    conf = pert_mol.GetConformer()
    coords = torch.tensor(conf.GetPositions(), dtype=torch.float32)

    # Rigid translation: 1.1 - 1.4 A shift
    t_dir = torch.randn(3)
    t_dir = t_dir / torch.norm(t_dir)
    trans = t_dir * (1.15 + 0.25 * (seed % 3))

    # Rigid rotation: 6 - 9 degrees
    angle = 0.11 + 0.03 * (seed % 3)
    axis = torch.randn(3)
    axis = axis / torch.norm(axis)
    R = axis_angle_to_matrix(axis * angle)

    com = coords.mean(dim=0, keepdim=True)
    pert_coords = (coords - com) @ R.T + com + trans

    pert_np = pert_coords.numpy()
    for i in range(pert_mol.GetNumAtoms()):
        conf.SetAtomPosition(i, Point3D(float(pert_np[i, 0]), float(pert_np[i, 1]), float(pert_np[i, 2])))

    return pert_mol


def run_50_benchmark(data_dir: str = "data/posebusters_benchmark_50", n_targets: int = 50):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory {data_dir} not found. Please extract targets first.")

    target_folders = sorted(os.listdir(data_dir))[:n_targets]
    print("=" * 90)
    print(f"SOLVDOCK OFFICIAL POSEBUSTERS BENCHMARK (N = {len(target_folders)} Targets)")
    print("Evaluating physical clash relief and torsional refinement using official posebusters==0.6.5")
    print("=" * 90)

    buster = posebusters.PoseBusters(config="dock")
    results = []

    before_pb_valid_count = 0
    after_pb_valid_count = 0
    before_clash_pass_count = 0
    after_clash_pass_count = 0
    total_time = 0.0

    print(f"{'Target':<10} | {'Atoms':<6} | {'Clash (Pre->Post)':<18} | {'RMSD':<7} | {'Retain':<7} | {'Time':<6} | {'Status'}")
    print("-" * 90)

    os.makedirs("data/posebusters_temp", exist_ok=True)

    for idx, tid in enumerate(target_folders):
        folder = os.path.join(data_dir, tid)
        prot_file = os.path.join(folder, f"{tid}_protein.pdb")
        lig_file = os.path.join(folder, f"{tid}_ligand.sdf")

        if not os.path.exists(prot_file) or not os.path.exists(lig_file):
            continue

        try:
            m_cryst = Chem.SDMolSupplier(lig_file, removeHs=False)[0]
            if m_cryst is None:
                continue

            n_atoms = m_cryst.GetNumAtoms()
            cryst_com = np.mean(m_cryst.GetConformer().GetPositions(), axis=0)

            # 1. Create unminimized pose simulating generative AI outputs
            m_raw = create_perturbed_ai_pose(m_cryst, seed=200 + idx)
            raw_sdf = f"data/posebusters_temp/{tid}_raw.sdf"
            w = Chem.SDWriter(raw_sdf)
            w.write(m_raw)
            w.close()

            # 2. Evaluate Raw Pose with PoseBusters
            df_raw = buster.bust(raw_sdf, lig_file, prot_file)
            raw_clash_pass = bool(df_raw["minimum_distance_to_protein"].values[0])
            raw_valid = bool(df_raw["volume_overlap_with_protein"].values[0] and raw_clash_pass)

            # 3. Relax with SolvDock
            t0 = time.time()
            res = solvdock.relax(prot_file, raw_sdf, max_steps=15, lr=0.035)
            dt = time.time() - t0
            total_time += dt

            # 4. Save relaxed ligand
            solv_sdf = f"data/posebusters_temp/{tid}_solvdock.sdf"
            w = Chem.SDWriter(solv_sdf)
            w.write(res.ligand_mol)
            w.close()

            # 5. Evaluate Relaxed Pose with PoseBusters
            df_solv = buster.bust(solv_sdf, lig_file, prot_file)
            solv_clash_pass = bool(df_solv["minimum_distance_to_protein"].values[0])
            solv_overlap_pass = bool(df_solv["volume_overlap_with_protein"].values[0])
            solv_bond_pass = bool(df_solv["bond_lengths"].values[0])
            solv_angle_pass = bool(df_solv["bond_angles"].values[0])
            solv_valid = bool(solv_overlap_pass and solv_clash_pass and solv_bond_pass and solv_angle_pass)

            relaxed_com = np.mean(res.ligand_mol.GetConformer().GetPositions(), axis=0)
            pocket_dist = float(np.linalg.norm(relaxed_com - cryst_com))

            if raw_valid:
                before_pb_valid_count += 1
            if solv_valid:
                after_pb_valid_count += 1
            if raw_clash_pass:
                before_clash_pass_count += 1
            if solv_clash_pass:
                after_clash_pass_count += 1

            status = "VALID" if solv_valid else ("RESCUED" if (not raw_clash_pass and solv_clash_pass) else "STABLE")

            print(f"{tid:<10} | {n_atoms:<6} | {str(raw_clash_pass):<7} -> {str(solv_clash_pass):<8} | {res.rmsd_to_input:.2f}A  | {pocket_dist:.2f}A  | {dt:.2f}s | {status}")

            results.append({
                "target_id": tid,
                "n_atoms": n_atoms,
                "raw_clash_pass": raw_clash_pass,
                "solv_clash_pass": solv_clash_pass,
                "raw_valid": raw_valid,
                "solv_valid": solv_valid,
                "rmsd_to_input": float(res.rmsd_to_input),
                "pocket_distance": float(pocket_dist),
                "runtime_seconds": float(dt),
            })

            # Cleanup temp files
            if os.path.exists(raw_sdf):
                os.remove(raw_sdf)
            if os.path.exists(solv_sdf):
                os.remove(solv_sdf)

        except Exception as e:
            print(f"{tid:<10} | Error: {e}")

    n_done = len(results)
    avg_t = total_time / max(n_done, 1)

    print("=" * 90)
    print("50-TARGET POSEBUSTERS BENCHMARK SUMMARY")
    print("=" * 90)
    print(f"Total Targets Evaluated:         {n_done}")
    print(f"Steric Clash Pass Rate (Before): {before_clash_pass_count}/{n_done} ({before_clash_pass_count/n_done*100:.1f}%)")
    print(f"Steric Clash Pass Rate (After):  {after_clash_pass_count}/{n_done} ({after_clash_pass_count/n_done*100:.1f}%)")
    print(f"Absolute Clash Relief Gain:      +{(after_clash_pass_count - before_clash_pass_count)/n_done*100:.1f}%")
    print(f"Overall PB-Valid Rate (Before):  {before_pb_valid_count}/{n_done} ({before_pb_valid_count/n_done*100:.1f}%)")
    print(f"Overall PB-Valid Rate (After):   {after_pb_valid_count}/{n_done} ({after_pb_valid_count/n_done*100:.1f}%)")
    print(f"Average Runtime per Target:      {avg_t:.2f} seconds (Total: {total_time:.1f}s)")
    print("=" * 90)

    # Save results
    os.makedirs("data/benchmarks", exist_ok=True)
    json_path = "data/benchmarks/posebusters_50_results.json"
    with open(json_path, "w") as f:
        json.dump({
            "n_targets": n_done,
            "before_clash_pass_rate": before_clash_pass_count / n_done,
            "after_clash_pass_rate": after_clash_pass_count / n_done,
            "before_valid_rate": before_pb_valid_count / n_done,
            "after_valid_rate": after_pb_valid_count / n_done,
            "average_runtime_seconds": avg_t,
            "targets": results,
        }, f, indent=2)

    df_out = pd.DataFrame(results)
    csv_path = "data/benchmarks/posebusters_50_summary.csv"
    df_out.to_csv(csv_path, index=False)

    md_path = "data/benchmarks/posebusters_50_summary.md"
    with open(md_path, "w") as f:
        f.write("# 50-Target PoseBusters Benchmark Summary\n\n")
        f.write(f"- **Total Targets**: {n_done}\n")
        f.write(f"- **Clash Pass Rate Before**: {before_clash_pass_count}/{n_done} ({before_clash_pass_count/n_done*100:.1f}%)\n")
        f.write(f"- **Clash Pass Rate After**: {after_clash_pass_count}/{n_done} ({after_clash_pass_count/n_done*100:.1f}%)\n")
        f.write(f"- **Absolute Gain**: +{(after_clash_pass_count - before_clash_pass_count)/n_done*100:.1f}%\n")
        f.write(f"- **Average Time**: {avg_t:.2f}s per target\n\n")
        f.write("## Target Results\n\n")
        f.write(df_out.to_markdown(index=False))

    print(f"Saved results to {json_path}, {csv_path}, and {md_path}")


if __name__ == "__main__":
    run_50_benchmark()
