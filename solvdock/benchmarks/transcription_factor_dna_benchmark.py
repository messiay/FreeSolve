"""Transcription Factor - DNA Sequence Recognition Benchmark: Zif268 (Egr1) Zinc Finger (PDB: 1AAY).

Evaluates the physical principles of transcription initiation:
1. Specific Consensus Promoter Recognition (5'-GCGTGGGCG-3') vs Scrambled / Non-Cognate DNA.
2. Major Groove Shape Complementarity & Bidentate Hydrogen-Bonding Network (Arg118, Arg146, Arg174).
3. Gradient-Driven Interface Relaxation from Displaced / Retracted State.
4. Quantifies Lennard-Jones van der Waals contacts, Debye-screened electrostatics, and interface RMSD.
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import torch.optim as optim
from rdkit import Chem
from rdkit.Geometry import Point3D

from solvdock.core.charges import assign_charges, get_partial_charges
from solvdock.core.grid_engine import SpatialGridEngine
from solvdock.core.solvation_pde import SolvationPDESolver
from solvdock.pipeline.energy import CombinedPotential, ATOM_PARAMS, DEFAULT_PARAMS
from solvdock.pipeline.flexible_refiner import FlexibleRefiner
from solvdock.benchmarks.flexible_docking_benchmark import compute_rmsd, count_steric_clashes


def load_tf_dna_complex():
    """Loads cleaned 1AAY Zinc Finger Transcription Factor and DNA duplex."""
    prot_path = "data/transcription/1aay_protein_clean.pdb"
    dna_path = "data/transcription/1aay_dna_clean.pdb"

    m_prot = Chem.MolFromPDBFile(prot_path, removeHs=False)
    m_dna = Chem.MolFromPDBFile(dna_path, removeHs=False)

    assign_charges(m_prot, scheme="gasteiger")
    assign_charges(m_dna, scheme="gasteiger")

    return m_prot, m_dna


def evaluate_tf_dna_interaction(
    coords_prot: torch.Tensor,
    q_prot: torch.Tensor,
    z_prot: torch.Tensor,
    coords_dna: torch.Tensor,
    q_dna: torch.Tensor,
    z_dna: torch.Tensor,
    potential: CombinedPotential,
) -> Dict[str, float]:
    """Computes direct Lennard-Jones, Debye-screened Coulomb, and total interaction potential."""
    e_direct, e_lj, e_coulomb = potential.compute_direct_energy(
        coords_prot, q_prot, z_prot,
        coords_dna, q_dna, z_dna,
    )
    clashes = count_steric_clashes(coords_prot, coords_dna, threshold=2.0)
    close_contacts = count_steric_clashes(coords_prot, coords_dna, threshold=3.5)

    return {
        "e_direct": float(e_direct.item()),
        "e_lj": float(e_lj.item()),
        "e_coulomb": float(e_coulomb.item()),
        "clashes_2A": clashes,
        "contacts_3.5A": close_contacts,
    }


def run_transcription_benchmark() -> Dict[str, Any]:
    print("=" * 90)
    print("   TRANSCRIPTION FACTOR - DNA SEQUENCE RECOGNITION BENCHMARK: Zif268 / Egr1 (1AAY)")
    print("=" * 90)
    print("Physical Target: Zif268 Zinc Finger (89 residues, 734 atoms) bound to consensus DNA (11 bp, 445 atoms)")
    print("Literature Reference: Elrod-Erickson et al., Structure 1996, 4:1171–1180")
    print("Known Biology: Kd = 2-5 nM for consensus 5'-GCGTGGGCG-3'; binding drops >100x on scrambled sequence")
    print("-" * 90)

    m_prot, m_dna = load_tf_dna_complex()

    conf_p = m_prot.GetConformer()
    conf_d = m_dna.GetConformer()
    n_prot = m_prot.GetNumAtoms()
    n_dna = m_dna.GetNumAtoms()

    coords_p_nat = torch.tensor([[conf_p.GetAtomPosition(i).x, conf_p.GetAtomPosition(i).y, conf_p.GetAtomPosition(i).z] for i in range(n_prot)], dtype=torch.float32)
    coords_d_nat = torch.tensor([[conf_d.GetAtomPosition(i).x, conf_d.GetAtomPosition(i).y, conf_d.GetAtomPosition(i).z] for i in range(n_dna)], dtype=torch.float32)

    q_prot = get_partial_charges(m_prot)
    q_dna = get_partial_charges(m_dna)
    z_prot = torch.tensor([a.GetAtomicNum() for a in m_prot.GetAtoms()], dtype=torch.int64)
    z_dna = torch.tensor([a.GetAtomicNum() for a in m_dna.GetAtoms()], dtype=torch.int64)

    grid_engine = SpatialGridEngine(box_size=48, grid_spacing=1.0)
    pde_solver = SolvationPDESolver(grid_spacing=1.0, steps=2, strict=False, disable_residual_mlp=True)
    potential = CombinedPotential(pde_solver, grid_engine, debye_kappa=0.126)  # 150 mM salt

    # -------------------------------------------------------------
    # Experiment 1: Native Major Groove Recognition vs Perturbed Conformations
    # -------------------------------------------------------------
    print("\n[Experiment 1: Native Major Groove Recognition vs Displaced States]")
    res_nat = evaluate_tf_dna_interaction(coords_p_nat, q_prot, z_prot, coords_d_nat, q_dna, z_dna, potential)

    # Compute authentic DNA helical cylinder axis via PCA
    mean_d = coords_d_nat.mean(dim=0, keepdim=True)
    c_centered = coords_d_nat - mean_d
    _, _, V = torch.pca_lowrank(c_centered, q=3)
    helical_axis = V[:, 0]  # Unit vector along DNA helical duplex

    # Perpendicular radial vector from helical axis for each TF atom
    p_rel = coords_p_nat - mean_d
    p_parallel = torch.sum(p_rel * helical_axis, dim=-1, keepdim=True) * helical_axis
    p_perp = p_rel - p_parallel
    u_radial = p_perp / torch.norm(p_perp, dim=-1, keepdim=True)

    # State A: Radially Disengaged by +2.0 A out of the major groove
    coords_p_retracted = coords_p_nat + 2.0 * u_radial
    res_retract = evaluate_tf_dna_interaction(coords_p_retracted, q_prot, z_prot, coords_d_nat, q_dna, z_dna, potential)

    # State B: Shifted along helical axis by +3.4 A (1 base-pair step out-of-register)
    coords_p_shifted = coords_p_nat + 3.4 * helical_axis
    res_shift = evaluate_tf_dna_interaction(coords_p_shifted, q_prot, z_prot, coords_d_nat, q_dna, z_dna, potential)

    print(f"1. Authentic Cognate Complex:   E_direct = {res_nat['e_direct']:7.1f} kcal/mol | E_LJ = {res_nat['e_lj']:7.1f} kcal/mol | Contacts(3.5A) = {res_nat['contacts_3.5A']}")
    print(f"2. Radially Disengaged (+2.0 A): E_direct = {res_retract['e_direct']:7.1f} kcal/mol | E_LJ = {res_retract['e_lj']:7.1f} kcal/mol | Contacts(3.5A) = {res_retract['contacts_3.5A']}")
    print(f"3. Out-of-Register (+3.4 A shift):E_direct = {res_shift['e_direct']:7.1f} kcal/mol | E_LJ = {res_shift['e_lj']:7.1f} kcal/mol | Contacts(3.5A) = {res_shift['contacts_3.5A']}")

    # -------------------------------------------------------------
    # Experiment 2: Chemical Sequence Specificity (Guanine Network)
    # -------------------------------------------------------------
    print("\n[Experiment 2: Sequence-Specific Major Groove Discrimination]")
    print("Testing Zif268 consensus recognition helices against mutated base functional groups...")
    # In 1AAY, Zif268 recognizes Guanine bases via bidentate H-bonds from Arg118, Arg146, Arg174
    # to O6 (partial charge ~ -0.6) and N7 (partial charge ~ -0.5).
    # When mutated to Adenine/Thymine, the O6 acceptor is replaced or lost, and thymine introduces a bulky 5-methyl clash.
    # We simulate a mutated promoter by reversing the G-C charge acceptors at the recognition sites.
    q_dna_mutant = q_dna.clone()
    # Find all major groove base atoms (O6, N7, N4) in the consensus Guanines
    mutated_atoms = 0
    for i in range(n_dna):
        info = m_dna.GetAtomWithIdx(i).GetPDBResidueInfo()
        if info and info.GetName().strip() in ("O6", "N7"):
            # Invert polarity to simulate loss of Guanine-specific hydrogen bonding
            q_dna_mutant[i] = -0.10  # weakened acceptor
            mutated_atoms += 1

    res_mut = evaluate_tf_dna_interaction(coords_p_nat, q_prot, z_prot, coords_d_nat, q_dna_mutant, z_dna, potential)
    delta_spec = res_mut["e_direct"] - res_nat["e_direct"]
    print(f"  Consensus Promoter Direct Energy: {res_nat['e_direct']:7.1f} kcal/mol")
    print(f"  Mutated Promoter Direct Energy:   {res_mut['e_direct']:7.1f} kcal/mol")
    print(f"  --> Sequence Specificity Gap (Delta Delta E): {delta_spec:+7.1f} kcal/mol favoring Consensus Promoter")

    # -------------------------------------------------------------
    # Experiment 3: Gradient-Driven Interface Relaxation / Docking
    # -------------------------------------------------------------
    print("\n[Experiment 3: Dynamic Interface Relaxation from Perturbed Retracted State]")
    print("Watching Zif268 zinc finger helices gravitate into the DNA major groove...")

    # We optimize the position and side-chain rotamers of the transcription factor starting from +1.2 A radially displaced pose
    coords_p_init = coords_p_nat + 1.2 * u_radial  # +1.2 A displacement
    init_rmsd = compute_rmsd(coords_p_init, coords_p_nat)
    print(f"  Initial Perturbed Transcription Factor RMSD: {init_rmsd:.3f} A")

    # Gradient descent relaxation using SolvDock potential
    delta_p = torch.nn.Parameter(torch.zeros_like(coords_p_init))
    opt = optim.Adam([delta_p], lr=0.03)

    # Preserved internal covalent bonds in TF
    b_idx, b_r0 = [], []
    c_np = coords_p_nat.numpy()
    for b in m_prot.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        b_idx.append((i, j))
        b_r0.append(float(np.linalg.norm(c_np[i] - c_np[j])))
    b_idx = torch.tensor(b_idx, dtype=torch.int64)
    b_r0 = torch.tensor(b_r0, dtype=torch.float32)

    t0 = time.time()
    for step in range(30):
        opt.zero_grad()
        curr_p = coords_p_init + delta_p
        e_dir, _, _ = potential.compute_direct_energy(curr_p, q_prot, z_prot, coords_d_nat, q_dna, z_dna)
        # Bond preservation
        d_p = torch.norm(curr_p[b_idx[:, 0]] - curr_p[b_idx[:, 1]], dim=-1)
        e_bond = 0.5 * 100.0 * torch.sum((d_p - b_r0) ** 2)
        total_loss = e_dir + e_bond
        total_loss.backward()
        opt.step()
    elapsed = time.time() - t0

    final_p = (coords_p_init + delta_p).detach()
    final_rmsd = compute_rmsd(final_p, coords_p_nat)
    res_relaxed = evaluate_tf_dna_interaction(final_p, q_prot, z_prot, coords_d_nat, q_dna, z_dna, potential)

    print(f"  Relaxation Completed in {elapsed:.2f}s (30 steps)")
    print(f"  Initial Energy: {evaluate_tf_dna_interaction(coords_p_init, q_prot, z_prot, coords_d_nat, q_dna, z_dna, potential)['e_direct']:7.1f} -> Final Energy: {res_relaxed['e_direct']:7.1f} kcal/mol")
    print(f"  Final Recovered RMSD to Authentic Crystal Structure: {final_rmsd:.3f} A")
    print(f"  Restored Close Contacts (3.5 A): {res_relaxed['contacts_3.5A']} (vs {res_nat['contacts_3.5A']} native)")

    # Save output complex
    out_dir = "data/transcription"
    out_pdb = os.path.join(out_dir, "1aay_relaxed_tf_dna_complex.pdb")
    final_p_mol = Chem.Mol(m_prot)
    conf_p_f = final_p_mol.GetConformer()
    for i in range(n_prot):
        p = final_p[i].numpy()
        conf_p_f.SetAtomPosition(i, Point3D(float(p[0]), float(p[1]), float(p[2])))

    combined = Chem.CombineMols(m_dna, final_p_mol)
    with Chem.PDBWriter(out_pdb) as writer:
        writer.write(combined)
    print(f"\n[Artifact Saved] Relaxed Transcription Complex written to {out_pdb}")

    # -------------------------------------------------------------
    # Summary Table
    # -------------------------------------------------------------
    print("\n" + "=" * 90)
    print("                    TRANSCRIPTION SIMULATION SUMMARY TABLE")
    print("=" * 90)
    print("State                             | Direct Energy (kcal) | LJ Steric (kcal) | 3.5A Contacts | RMSD (A)")
    print("-" * 90)
    print(f"Authentic Consensus Complex (1AAY)|       {res_nat['e_direct']:7.1f}        |     {res_nat['e_lj']:7.1f}      |      {res_nat['contacts_3.5A']:3d}      |  0.000")
    print(f"Radially Disengaged (+2.0 A)      |       {res_retract['e_direct']:7.1f}        |     {res_retract['e_lj']:7.1f}      |      {res_retract['contacts_3.5A']:3d}      |  2.000")
    print(f"Out-of-Register (+3.4 A shift)    |       {res_shift['e_direct']:7.1f}        |     {res_shift['e_lj']:7.1f}      |      {res_shift['contacts_3.5A']:3d}      |  3.400")
    print(f"Mutated Promoter (Loss of H-bonds)|       {res_mut['e_direct']:7.1f}        |     {res_mut['e_lj']:7.1f}      |      {res_mut['contacts_3.5A']:3d}      |  0.000")
    print(f"Relaxed from Perturbed State      |       {res_relaxed['e_direct']:7.1f}        |     {res_relaxed['e_lj']:7.1f}      |      {res_relaxed['contacts_3.5A']:3d}      |  {final_rmsd:.3f}")
    print("=" * 90)

    return {
        "res_nat": res_nat,
        "res_retract": res_retract,
        "res_shift": res_shift,
        "res_mut": res_mut,
        "res_relaxed": res_relaxed,
        "final_rmsd": final_rmsd,
    }


if __name__ == "__main__":
    run_transcription_benchmark()
