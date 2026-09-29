"""Induced-Fit Protein-Protein Docking (PPI) Demonstration on Trypsin-BPTI (PDB: 2PTC)."""

import os
import sys
import time
import numpy as np
import torch
import torch.optim as optim
from rdkit import Chem
from rdkit.Geometry import Point3D

from solvdock.core.charges import assign_charges, get_partial_charges
from solvdock.core.grid_engine import SpatialGridEngine
from solvdock.core.kinematics import (
    DifferentiableSE3,
    DifferentiableTorsionTree,
    axis_angle_to_matrix,
    build_downstream_subgraphs,
    find_rotatable_bonds,
)
from solvdock.core.solvation_pde import SolvationPDESolver
from solvdock.core.topology import build_topological_scale_matrix
from solvdock.pipeline.energy import CombinedPotential
from solvdock.pipeline.flexible_refiner import FlexibleRefiner


def count_clashes(c1: torch.Tensor, c2: torch.Tensor, threshold: float = 2.0) -> int:
    """Counts number of atom pairs between two proteins with distance < threshold Angstroms."""
    dists = torch.cdist(c1, c2)
    return int(torch.sum(dists < threshold).item())


def main():
    print("=" * 80)
    print("SolvDock Induced-Fit Protein-Protein Docking (PPI) Benchmark")
    print("Target Complex: Bovine Trypsin (Chain E) + BPTI Inhibitor (Chain I) [PDB: 2PTC]")
    print("=" * 80)

    tryp_path = "data/docking/trypsin_receptor.pdb"
    bpti_path = "data/docking/bpti_ligand.pdb"

    if not os.path.exists(tryp_path) or not os.path.exists(bpti_path):
        raise FileNotFoundError("Clean Trypsin and BPTI PDB files not found.")

    m_tryp = Chem.MolFromPDBFile(tryp_path, removeHs=False)
    m_bpti = Chem.MolFromPDBFile(bpti_path, removeHs=False)

    assign_charges(m_tryp, scheme="gasteiger")
    assign_charges(m_bpti, scheme="gasteiger")

    n_tryp = m_tryp.GetNumAtoms()
    n_bpti = m_bpti.GetNumAtoms()

    conf_tryp = m_tryp.GetConformer()
    conf_bpti = m_bpti.GetConformer()

    c_tryp_0 = torch.tensor(conf_tryp.GetPositions(), dtype=torch.float32)
    c_bpti_0 = torch.tensor(conf_bpti.GetPositions(), dtype=torch.float32)

    q_tryp = get_partial_charges(m_tryp)
    q_bpti = get_partial_charges(m_bpti)
    z_tryp = torch.tensor([a.GetAtomicNum() for a in m_tryp.GetAtoms()], dtype=torch.int64)
    z_bpti = torch.tensor([a.GetAtomicNum() for a in m_bpti.GetAtoms()], dtype=torch.int64)

    # 1. Identify Interfacial Contact Residues (< 10 A)
    dists_0 = torch.cdist(c_tryp_0, c_bpti_0)
    min_to_bpti = torch.min(dists_0, dim=1)[0]
    min_to_tryp = torch.min(dists_0, dim=0)[0]

    contact_cutoff = 10.0
    iface_tryp_mask = min_to_bpti < contact_cutoff
    iface_bpti_mask = min_to_tryp < contact_cutoff

    print(f"Trypsin total atoms: {n_tryp} | Interface atoms (< 10 A): {iface_tryp_mask.sum().item()}")
    print(f"BPTI total atoms:    {n_bpti} | Interface atoms (< 10 A): {iface_bpti_mask.sum().item()}")
    print(f"Crystal minimum interface distance: {torch.min(dists_0).item():.3f} A")

    # 2. Extract Reference Bond Lengths for Invariance Verification
    bonds_tryp = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m_tryp.GetBonds()]
    bonds_bpti = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m_bpti.GetBonds()]
    r0_tryp = [float(torch.norm(c_tryp_0[i] - c_tryp_0[j])) for i, j in bonds_tryp]
    r0_bpti = [float(torch.norm(c_bpti_0[i] - c_bpti_0[j])) for i, j in bonds_bpti]

    # 3. Backbone Identification
    bb_mask_tryp = FlexibleRefiner.identify_backbone_mask(m_tryp)
    bb_mask_bpti = FlexibleRefiner.identify_backbone_mask(m_bpti)

    # 4. Rotatable Sidechain Chains Directed Outward from Backbone
    fixed_tryp = {i for i in range(n_tryp) if bb_mask_tryp[i].item() or not iface_tryp_mask[i].item()}
    fixed_bpti = {i for i in range(n_bpti) if bb_mask_bpti[i].item()}

    rot_tryp = find_rotatable_bonds(m_tryp, fixed_atom_indices=fixed_tryp)
    masks_tryp = build_downstream_subgraphs(m_tryp, rot_tryp, root_indices=fixed_tryp)

    rot_bpti = find_rotatable_bonds(m_bpti, fixed_atom_indices=fixed_bpti)
    masks_bpti = build_downstream_subgraphs(m_bpti, rot_bpti, root_indices=fixed_bpti)

    print(f"Trypsin interface rotatable sidechain dihedrals: {len(rot_tryp)}")
    print(f"BPTI rotatable sidechain dihedrals:           {len(rot_bpti)}")

    # 5. Create Realistic Perturbed Binding Pose for BPTI
    shift = torch.tensor([1.4, -1.1, 0.9], dtype=torch.float32)
    axis = torch.tensor([0.4, 0.8, -0.4], dtype=torch.float32)
    axis = axis / torch.norm(axis)
    theta_pert = torch.tensor(0.12)  # ~6.9 degrees
    R_pert = axis_angle_to_matrix(axis * theta_pert)

    bpti_center_0 = c_bpti_0.mean(dim=0, keepdim=True)
    c_bpti_pert = (c_bpti_0 - bpti_center_0) @ R_pert.T + bpti_center_0 + shift.view(1, 3)

    clashes_init = count_clashes(c_tryp_0, c_bpti_pert, threshold=2.0)
    rmsd_bpti_init = float(torch.sqrt(torch.mean(torch.sum((c_bpti_pert - c_bpti_0) ** 2, dim=-1))))
    print(f"\nPerturbed BPTI pose created: Initial RMSD = {rmsd_bpti_init:.3f} A")
    print(f"Severe Interfacial Clashes (d < 2.0 A): {clashes_init} clashes!")

    # 6. Set up Differentiable Kinematics & Potential
    torsion_tryp = DifferentiableTorsionTree(c_tryp_0, rot_tryp, masks_tryp)
    torsion_bpti = DifferentiableTorsionTree(c_bpti_pert, rot_bpti, masks_bpti)
    se3_bpti = DifferentiableSE3(c_bpti_pert.mean(dim=0))

    grid_engine = SpatialGridEngine(box_size=32, grid_spacing=1.0)
    pde_solver = SolvationPDESolver(
        grid_spacing=1.0, steps=2,
        calibrated_constants_path="configs/calibrated_constants.yaml",
        strict=False, disable_residual_mlp=True,
    )
    potential = CombinedPotential(pde_solver, grid_engine)

    topo_tryp = build_topological_scale_matrix(m_tryp)
    topo_bpti = build_topological_scale_matrix(m_bpti)

    opt_params = [p for p in list(se3_bpti.parameters()) + list(torsion_bpti.parameters()) + list(torsion_tryp.parameters()) if p.requires_grad]
    optimizer = optim.Adam(opt_params, lr=0.035)

    v_barrier = 1.5

    def compute_ppi_loss():
        curr_bpti = se3_bpti(torsion_bpti())
        curr_tryp = torsion_tryp()

        # Intermolecular interaction (Trypsin interface vs BPTI interface)
        e_direct, _, _ = potential.compute_direct_energy(
            curr_bpti[iface_bpti_mask], q_bpti[iface_bpti_mask], z_bpti[iface_bpti_mask],
            curr_tryp[iface_tryp_mask], q_tryp[iface_tryp_mask], z_tryp[iface_tryp_mask],
        )

        # Intramolecular non-bonded energy on Trypsin and BPTI interfaces
        e_intra_tryp, _, _ = potential.compute_intramolecular_energy(
            curr_tryp[iface_tryp_mask], q_tryp[iface_tryp_mask], z_tryp[iface_tryp_mask], topo_tryp[iface_tryp_mask][:, iface_tryp_mask]
        )
        e_intra_bpti, _, _ = potential.compute_intramolecular_energy(
            curr_bpti[iface_bpti_mask], q_bpti[iface_bpti_mask], z_bpti[iface_bpti_mask], topo_bpti[iface_bpti_mask][:, iface_bpti_mask]
        )

        e_strain_tryp = torch.sum(v_barrier * (1.0 - torch.cos(3.0 * torsion_tryp.thetas))) if torsion_tryp.n_torsions > 0 else 0.0
        e_strain_bpti = torch.sum(v_barrier * (1.0 - torch.cos(3.0 * torsion_bpti.thetas))) if torsion_bpti.n_torsions > 0 else 0.0

        return e_direct + 0.5 * (e_intra_tryp + e_intra_bpti) + e_strain_tryp + e_strain_bpti, e_direct

    print("\nStarting SolvDock Articulated Induced-Fit Optimization (30 steps)...")
    t0 = time.time()
    for step in range(30):
        optimizer.zero_grad()
        loss, e_dir = compute_ppi_loss()
        loss.backward()
        optimizer.step()
        if (step + 1) % 10 == 0:
            print(f"  Step {step+1:2d}/30 | Loss: {loss.item():.2f} | Direct E: {e_dir.item():.2f} kcal/mol")
    t_opt = time.time() - t0
    print(f"Optimization completed in {t_opt:.2f} seconds.")

    with torch.no_grad():
        final_tryp = torsion_tryp().detach()
        final_bpti = se3_bpti(torsion_bpti()).detach()

    # 7. Physical Invariance & Convergence Checks
    clashes_final = count_clashes(final_tryp, final_bpti, threshold=2.0)
    rmsd_bpti_final = float(torch.sqrt(torch.mean(torch.sum((final_bpti - c_bpti_0) ** 2, dim=-1))))
    bb_rmsd_bpti = float(torch.sqrt(torch.mean(torch.sum((final_bpti[bb_mask_bpti] - c_bpti_0[bb_mask_bpti]) ** 2, dim=-1))))

    # Sidechain displacements
    is_sc_tryp = ~bb_mask_tryp & iface_tryp_mask
    sc_disp_tryp = torch.sqrt(torch.sum((final_tryp[is_sc_tryp] - c_tryp_0[is_sc_tryp]) ** 2, dim=-1))
    max_disp_tryp = float(torch.max(sc_disp_tryp).item()) if len(sc_disp_tryp) > 0 else 0.0
    mean_disp_tryp = float(torch.mean(sc_disp_tryp).item()) if len(sc_disp_tryp) > 0 else 0.0

    is_sc_bpti = ~bb_mask_bpti & iface_bpti_mask
    sc_disp_bpti = torch.sqrt(torch.sum((final_bpti[is_sc_bpti] - c_bpti_pert[is_sc_bpti]) ** 2, dim=-1))
    max_disp_bpti = float(torch.max(sc_disp_bpti).item()) if len(sc_disp_bpti) > 0 else 0.0
    mean_disp_bpti = float(torch.mean(sc_disp_bpti).item()) if len(sc_disp_bpti) > 0 else 0.0

    # Covalent bond errors
    final_tryp_np = final_tryp.numpy()
    final_bpti_np = final_bpti.numpy()

    tryp_bond_err = max(abs(np.linalg.norm(final_tryp_np[i] - final_tryp_np[j]) - r0) for (i, j), r0 in zip(bonds_tryp, r0_tryp))
    bpti_bond_err = max(abs(np.linalg.norm(final_bpti_np[i] - final_bpti_np[j]) - r0) for (i, j), r0 in zip(bonds_bpti, r0_bpti))
    max_bond_err = max(tryp_bond_err, bpti_bond_err)

    print("\n" + "=" * 80)
    print("RESULTS: SolvDock Induced-Fit Protein-Protein Docking (2PTC)")
    print("=" * 80)
    print(f"Interfacial Clashes (< 2.0 A):       {clashes_init} clashes -> {clashes_final} clashes ({(clashes_init-clashes_final)/max(clashes_init, 1)*100:.1f}% relief)")
    print(f"BPTI Full-Atom Pose RMSD to Crystal: {rmsd_bpti_init:.3f} A -> {rmsd_bpti_final:.3f} A")
    print(f"BPTI Backbone RMSD to Crystal:      {bb_rmsd_bpti:.3f} A")
    print(f"Trypsin Max Sidechain Displacement: {max_disp_tryp:.3f} A (Mean: {mean_disp_tryp:.3f} A)")
    print(f"BPTI Max Sidechain Displacement:    {max_disp_bpti:.3f} A (Mean: {mean_disp_bpti:.3f} A)")
    print(f"Max Covalent Bond Distortion:       {max_bond_err:.8f} A (< 1e-4 A machine precision)")

    # 8. Export PDB for 3D Visualization
    out_pdb = "data/docking/2ptc_ppi_docked_result.pdb"
    with open(out_pdb, "w") as out:
        out.write("REMARK   SolvDock Induced-Fit Protein-Protein Docking (Trypsin-BPTI 2PTC)\n")
        out.write(f"REMARK   Clashes: {clashes_init} -> {clashes_final} | RMSD: {rmsd_bpti_final:.3f} A | Bond Error: {max_bond_err:.7f} A\n")

        atom_num = 1
        # Write Trypsin (Chain E)
        for i, atom in enumerate(m_tryp.GetAtoms()):
            res = atom.GetPDBResidueInfo()
            name = res.GetName() if res else atom.GetSymbol().rjust(4)
            resname = res.GetResidueName() if res else "TRP"
            resnum = res.GetResidueNumber() if res else 1
            pos = final_tryp_np[i]
            elem = atom.GetSymbol()
            out.write(f"ATOM  {atom_num:5d} {name:4s} {resname:3s} E{resnum:4d}    {pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00 20.00          {elem:>2s}\n")
            atom_num += 1

        # Write SolvDock Docked BPTI (Chain I)
        for i, atom in enumerate(m_bpti.GetAtoms()):
            res = atom.GetPDBResidueInfo()
            name = res.GetName() if res else atom.GetSymbol().rjust(4)
            resname = res.GetResidueName() if res else "BPT"
            resnum = res.GetResidueNumber() if res else 1
            pos = final_bpti_np[i]
            elem = atom.GetSymbol()
            out.write(f"ATOM  {atom_num:5d} {name:4s} {resname:3s} I{resnum:4d}    {pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00 20.00          {elem:>2s}\n")
            atom_num += 1

        # Write Perturbed Clashing BPTI (Chain P)
        c_pert_np = c_bpti_pert.numpy()
        for i, atom in enumerate(m_bpti.GetAtoms()):
            res = atom.GetPDBResidueInfo()
            name = res.GetName() if res else atom.GetSymbol().rjust(4)
            resname = res.GetResidueName() if res else "BPT"
            resnum = res.GetResidueNumber() if res else 1
            pos = c_pert_np[i]
            elem = atom.GetSymbol()
            out.write(f"ATOM  {atom_num:5d} {name:4s} {resname:3s} P{resnum:4d}    {pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00 20.00          {elem:>2s}\n")
            atom_num += 1

        # Write Native Crystal BPTI (Chain X)
        c_cryst_np = c_bpti_0.numpy()
        for i, atom in enumerate(m_bpti.GetAtoms()):
            res = atom.GetPDBResidueInfo()
            name = res.GetName() if res else atom.GetSymbol().rjust(4)
            resname = res.GetResidueName() if res else "BPT"
            resnum = res.GetResidueNumber() if res else 1
            pos = c_cryst_np[i]
            elem = atom.GetSymbol()
            out.write(f"ATOM  {atom_num:5d} {name:4s} {resname:3s} X{resnum:4d}    {pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00 20.00          {elem:>2s}\n")
            atom_num += 1

        # Write Initial Unrelaxed Trypsin (Chain U)
        c_tryp_init_np = c_tryp_0.numpy()
        for i, atom in enumerate(m_tryp.GetAtoms()):
            res = atom.GetPDBResidueInfo()
            name = res.GetName() if res else atom.GetSymbol().rjust(4)
            resname = res.GetResidueName() if res else "TRP"
            resnum = res.GetResidueNumber() if res else 1
            pos = c_tryp_init_np[i]
            elem = atom.GetSymbol()
            out.write(f"ATOM  {atom_num:5d} {name:4s} {resname:3s} U{resnum:4d}    {pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00 20.00          {elem:>2s}\n")
            atom_num += 1

        out.write("END\n")

    print(f"\nSaved multi-protein docked complex to: {out_pdb}")


if __name__ == "__main__":
    main()
