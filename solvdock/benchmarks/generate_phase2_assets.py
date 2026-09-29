"""Generates publication-grade PDB structural assets for the SolvDock 3D Viewport."""

import os
import numpy as np
import torch
from rdkit import Chem

from solvdock.core.kinematics import (
    DifferentiableSE3,
    DifferentiableTorsionTree,
    build_downstream_subgraphs,
    find_rotatable_bonds,
)
from solvdock.pipeline.flexible_refiner import FlexibleRefiner


def generate_2v00_head_to_head_pdb(out_path: str):
    """Combines 2v00 pocket, AutoDock Vina pose, SolvDock pose, and Crystal ligand into one clean PDB."""
    poc_path = "data/casf2016_core/2v00_pocket.pdb"
    vina_path = "data/docking/2v00_vina_out.pdbqt"
    slv_path = "data/docking/2v00_global_induced_fit_final.pdb"
    xry_path = "data/casf2016_core/2v00_ligand.sdf"

    # 1. Read pocket lines
    with open(poc_path, "r") as f:
        pocket_lines = [line for line in f if line.startswith("ATOM") or line.startswith("HETATM")]

    # 2. Extract Vina Model 1 coordinates
    vina_coords = []
    vina_atoms = []
    with open(vina_path, "r") as f:
        in_model_1 = False
        for line in f:
            if line.startswith("MODEL 1"):
                in_model_1 = True
                continue
            if line.startswith("ENDMDL") and in_model_1:
                break
            if in_model_1 and (line.startswith("ATOM") or line.startswith("HETATM")):
                name = line[12:16].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                elem = line[76:78].strip() if len(line) >= 78 else name[0]
                if not elem:
                    elem = name[0]
                if elem != "H":  # Heavy atoms
                    vina_coords.append((x, y, z))
                    vina_atoms.append((name, elem))

    # 3. Extract SolvDock pose
    slv_coords = []
    slv_atoms = []
    with open(slv_path, "r") as f:
        for line in f:
            if line.startswith("HETATM") or (line.startswith("ATOM") and "LIG" in line):
                name = line[12:16].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                elem = line[76:78].strip() if len(line) >= 78 else name[0]
                if not elem:
                    elem = name[0]
                if elem != "H":
                    slv_coords.append((x, y, z))
                    slv_atoms.append((name, elem))

    # 4. Extract Crystal Native ligand from SDF
    m_xry = Chem.SDMolSupplier(xry_path, removeHs=True)[0]
    conf_x = m_xry.GetConformer()
    xry_coords = [conf_x.GetAtomPosition(i) for i in range(m_xry.GetNumAtoms())]
    xry_elems = [a.GetSymbol() for a in m_xry.GetAtoms()]

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as out:
        out.write("REMARK   SolvDock vs AutoDock Vina Head-to-Head Comparison (2v00)\n")
        # Write pocket
        for line in pocket_lines:
            out.write(line)

        # Write Crystal native (Chain X, Resname XRY)
        for idx, (pos, elem) in enumerate(zip(xry_coords, xry_elems), 1):
            out.write(f"HETATM{idx:5d} {elem+str(idx):^4s} XRY X   1    {pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}  1.00 20.00          {elem:>2s}\n")

        # Write SolvDock Pose (Chain S, Resname SLV)
        for idx, (p, (name, elem)) in enumerate(zip(slv_coords, slv_atoms), 1):
            out.write(f"HETATM{idx:5d} {elem+str(idx):^4s} SLV S   1    {p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}  1.00 20.00          {elem:>2s}\n")

        # Write Vina Pose (Chain V, Resname VNA)
        for idx, (p, (name, elem)) in enumerate(zip(vina_coords, vina_atoms), 1):
            out.write(f"HETATM{idx:5d} {elem+str(idx):^4s} VNA V   1    {p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}  1.00 20.00          {elem:>2s}\n")

        out.write("END\n")

    print(f"Wrote head-to-head comparison PDB to {out_path}")


def generate_torsional_trajectory_pdb(out_path: str):
    """Generates a smooth, authentic multi-frame trajectory using PyTorch torsional kinematics."""
    poc_path = "data/casf2016_core/2v00_pocket.pdb"
    lig_path = "data/casf2016_core/2v00_ligand.sdf"

    m_poc = Chem.MolFromPDBFile(poc_path, removeHs=False)
    m_lig = Chem.SDMolSupplier(lig_path, removeHs=False)[0]

    conf_p = m_poc.GetConformer()
    conf_l = m_lig.GetConformer()

    coords_p_0 = torch.tensor([[conf_p.GetAtomPosition(i).x, conf_p.GetAtomPosition(i).y, conf_p.GetAtomPosition(i).z]
                               for i in range(m_poc.GetNumAtoms())], dtype=torch.float32)
    coords_l_0 = torch.tensor([[conf_l.GetAtomPosition(i).x, conf_l.GetAtomPosition(i).y, conf_l.GetAtomPosition(i).z]
                               for i in range(m_lig.GetNumAtoms())], dtype=torch.float32)

    # Initial perturbation: shifted by 2.0 A outside pocket
    shift = torch.tensor([2.0, -1.5, 1.2])
    coords_l_start = coords_l_0 + shift

    refiner = FlexibleRefiner()
    bb_mask = refiner.identify_backbone_mask(m_poc)
    fixed_indices = {i for i in range(m_poc.GetNumAtoms()) if bb_mask[i].item()}

    rot_poc = find_rotatable_bonds(m_poc, fixed_atom_indices=fixed_indices)
    masks_poc = build_downstream_subgraphs(m_poc, rot_poc, root_indices=fixed_indices)
    torsion_poc = DifferentiableTorsionTree(coords_p_0, rot_poc, masks_poc)

    rot_lig = find_rotatable_bonds(m_lig)
    masks_lig = build_downstream_subgraphs(m_lig, rot_lig)
    torsion_lig = DifferentiableTorsionTree(coords_l_start, rot_lig, masks_lig)
    se3_lig = DifferentiableSE3(coords_l_start.mean(dim=0))

    # Run 25 steps of torsional refinement recording coordinates
    opt_params = [p for p in list(se3_lig.parameters()) + list(torsion_lig.parameters()) + list(torsion_poc.parameters())
                  if p.requires_grad and p.numel() > 0]
    optimizer = torch.optim.Adam(opt_params, lr=0.04)

    from solvdock.core.charges import assign_charges, get_partial_charges
    assign_charges(m_lig, scheme="gasteiger")
    assign_charges(m_poc, scheme="gasteiger")
    q_lig = get_partial_charges(m_lig)
    q_poc = get_partial_charges(m_poc)
    z_lig = torch.tensor([a.GetAtomicNum() for a in m_lig.GetAtoms()], dtype=torch.int64)
    z_poc = torch.tensor([a.GetAtomicNum() for a in m_poc.GetAtoms()], dtype=torch.int64)

    recorded_frames = []
    # Frame 0
    with torch.no_grad():
        c_l = se3_lig(torsion_lig()).numpy()
        c_p = torsion_poc().numpy()
        recorded_frames.append((c_p, c_l))

    for step in range(1, 25):
        optimizer.zero_grad()
        curr_l = se3_lig(torsion_lig())
        curr_p = torsion_poc()
        loss, _, _ = refiner.potential.compute_direct_energy(curr_l, q_lig, z_lig, curr_p, q_poc, z_poc)
        loss.backward()
        optimizer.step()

        if step % 2 == 0 or step == 24:
            with torch.no_grad():
                c_l = se3_lig(torsion_lig()).numpy()
                c_p = torsion_poc().numpy()
                recorded_frames.append((c_p, c_l))

    # Write multi-model trajectory
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as out:
        for frame_idx, (frame_poc, frame_lig) in enumerate(recorded_frames, 1):
            out.write(f"MODEL     {frame_idx:4d}\n")
            # Pocket atoms
            for i, atom in enumerate(m_poc.GetAtoms()):
                sym = atom.GetSymbol()
                info = atom.GetPDBResidueInfo()
                res_name = info.GetResidueName() if info else "POC"
                res_num = info.GetResidueNumber() if info else 1
                chain = info.GetChainId() if info else "A"
                name = info.GetName() if info else f"{sym}{i+1}"
                p = frame_poc[i]
                out.write(f"ATOM  {i+1:5d} {name:^4s} {res_name:3s} {chain:1s}{res_num:4d}    {p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}  1.00 20.00          {sym:>2s}\n")

            # Ligand atoms
            n_poc = m_poc.GetNumAtoms()
            for j, atom in enumerate(m_lig.GetAtoms()):
                sym = atom.GetSymbol()
                p = frame_lig[j]
                out.write(f"HETATM{n_poc+j+1:5d} {sym+str(j+1):^4s} LIG L   1    {p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}  1.00 20.00          {sym:>2s}\n")

            out.write("ENDMDL\n")

    print(f"Wrote {len(recorded_frames)} torsional kinematic frames to {out_path}")


def prepare_1aay_clean(out_path: str):
    """Copies the authentic intact 1AAY PDB with protein, DNA, and Zn ions."""
    src = "data/transcription/1aay.pdb"
    with open(src, "r") as f:
        lines = [line for line in f if line.startswith("ATOM") or line.startswith("HETATM") or line.startswith("TER")]

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        f.writelines(lines)
        f.write("END\n")
    print(f"Prepared clean 1AAY complex at {out_path}")


if __name__ == "__main__":
    generate_2v00_head_to_head_pdb("data/viz/2v00_head_to_head.pdb")
    generate_torsional_trajectory_pdb("data/viz/2v00_torsional_trajectory.pdb")
    prepare_1aay_clean("data/viz/1aay_clean_tf_dna.pdb")
