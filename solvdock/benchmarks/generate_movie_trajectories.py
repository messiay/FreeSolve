"""Generates smooth multi-model trajectory PDB files for PyMOL and 3D web viewers."""

import os
import numpy as np
import torch
from rdkit import Chem
from rdkit.Geometry import Point3D

from solvdock.core.charges import assign_charges, get_partial_charges
from solvdock.core.grid_engine import SpatialGridEngine
from solvdock.core.solvation_pde import SolvationPDESolver
from solvdock.pipeline.energy import CombinedPotential
from solvdock.pipeline.flexible_refiner import FlexibleRefiner


def write_multimodel_pdb(file_path, base_mol, frames_coords):
    """Writes a multi-model PDB movie trajectory."""
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    with open(file_path, "w") as f:
        for frame_idx, coords in enumerate(frames_coords, 1):
            f.write(f"MODEL     {frame_idx:4d}\n")
            conf = base_mol.GetConformer()
            for atom_idx in range(base_mol.GetNumAtoms()):
                atom = base_mol.GetAtomWithIdx(atom_idx)
                sym = atom.GetSymbol()
                res_info = atom.GetPDBResidueInfo()
                res_name = res_info.GetResidueName() if res_info else "UNK"
                res_num = res_info.GetResidueNumber() if res_info else 1
                chain = res_info.GetChainId() if res_info else "A"
                name = res_info.GetName() if res_info else f"{sym}{atom_idx+1}"
                p = coords[atom_idx]
                line = f"ATOM  {atom_idx+1:5d} {name:^4s} {res_name:3s} {chain:1s}{res_num:4d}    {p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}  1.00 20.00          {sym:>2s}\n"
                f.write(line)
            f.write("ENDMDL\n")
    print(f"Wrote {len(frames_coords)} frames to {file_path}")


def generate_docking_movie():
    """Generates 2v00 induced-fit docking movie trajectory."""
    cid = '2v00'
    lig_path = f'data/casf2016_core/{cid}_ligand.sdf'
    poc_path = f'data/casf2016_core/{cid}_pocket.pdb'

    m_lig = Chem.SDMolSupplier(lig_path, removeHs=False)[0]
    m_poc = Chem.MolFromPDBFile(poc_path, removeHs=False)
    assign_charges(m_lig, scheme="gasteiger")
    assign_charges(m_poc, scheme="gasteiger")

    refiner = FlexibleRefiner(k_backbone=50.0, k_bond=100.0)
    bb_mask = refiner.identify_backbone_mask(m_poc)

    # Shift ligand initially
    conf_l = m_lig.GetConformer()
    conf_p = m_poc.GetConformer()
    n_lig = m_lig.GetNumAtoms()
    n_poc = m_poc.GetNumAtoms()

    coords_l_0 = torch.tensor([[conf_l.GetAtomPosition(i).x, conf_l.GetAtomPosition(i).y, conf_l.GetAtomPosition(i).z] for i in range(n_lig)], dtype=torch.float32)
    coords_p_0 = torch.tensor([[conf_p.GetAtomPosition(i).x, conf_p.GetAtomPosition(i).y, conf_p.GetAtomPosition(i).z] for i in range(n_poc)], dtype=torch.float32)

    # Initial perturbation: shifted by (2.5, -2.0, 1.5) A outside pocket
    shift = torch.tensor([2.5, -2.0, 1.5])
    coords_l_pert = coords_l_0 + shift
    m_lig_pert = Chem.Mol(m_lig)
    for i in range(n_lig):
        m_lig_pert.GetConformer().SetAtomPosition(i, Point3D(float(coords_l_pert[i, 0]), float(coords_l_pert[i, 1]), float(coords_l_pert[i, 2])))

    # Run induced-fit recording snapshots every 2 steps
    combined_base = Chem.CombineMols(m_poc, m_lig)

    # Record 15 interpolated simulation frames
    res = refiner.refine_induced_fit(m_lig_pert, m_poc, backbone_mask=bb_mask, freeze_backbone=True, max_steps=30, lr=0.04)

    final_l = res["final_ligand_coords"].numpy()
    final_p = res["final_pocket_coords"].numpy()
    start_l = coords_l_pert.numpy()
    start_p = coords_p_0.numpy()

    frames = []
    n_frames = 16
    for step in range(n_frames):
        alpha = float(step) / (n_frames - 1)
        # Non-linear easing for smooth approach
        t = alpha * alpha * (3.0 - 2.0 * alpha)
        curr_l = (1.0 - t) * start_l + t * final_l
        curr_p = (1.0 - t) * start_p + t * final_p
        curr_comb = np.vstack([curr_p, curr_l])
        frames.append(curr_comb)

    write_multimodel_pdb("data/docking/2v00_induced_fit_movie.pdb", combined_base, frames)


def generate_transcription_movie():
    """Generates 1AAY transcription factor major groove docking movie."""
    m_prot = Chem.MolFromPDBFile("data/transcription/1aay_protein_clean.pdb", removeHs=False)
    m_dna = Chem.MolFromPDBFile("data/transcription/1aay_dna_clean.pdb", removeHs=False)

    conf_p = m_prot.GetConformer()
    conf_d = m_dna.GetConformer()
    c_p = np.array([[conf_p.GetAtomPosition(i).x, conf_p.GetAtomPosition(i).y, conf_p.GetAtomPosition(i).z] for i in range(m_prot.GetNumAtoms())])
    c_d = np.array([[conf_d.GetAtomPosition(i).x, conf_d.GetAtomPosition(i).y, conf_d.GetAtomPosition(i).z] for i in range(m_dna.GetNumAtoms())])

    # Load relaxed coords
    m_relaxed = Chem.MolFromPDBFile("data/transcription/1aay_relaxed_tf_dna_complex.pdb", sanitize=False, removeHs=False)
    conf_r = m_relaxed.GetConformer()
    c_relaxed = np.array([[conf_r.GetAtomPosition(i).x, conf_r.GetAtomPosition(i).y, conf_r.GetAtomPosition(i).z] for i in range(m_relaxed.GetNumAtoms())])
    c_p_final = c_relaxed[445:]

    # Radial displacement outward
    mean_d = c_d.mean(axis=0)
    c_centered = c_d - mean_d
    _, _, Vt = np.linalg.svd(c_centered, full_matrices=False)
    helical_axis = Vt[0]

    p_rel = c_p - mean_d
    p_parallel = np.sum(p_rel * helical_axis, axis=-1, keepdims=True) * helical_axis
    p_perp = p_rel - p_parallel
    u_radial = p_perp / np.linalg.norm(p_perp, axis=-1, keepdims=True)

    c_p_start = c_p + 3.0 * u_radial

    combined_base = Chem.CombineMols(m_dna, m_prot)
    frames = []
    n_frames = 16
    for step in range(n_frames):
        alpha = float(step) / (n_frames - 1)
        t = alpha * alpha * (3.0 - 2.0 * alpha)
        curr_p = (1.0 - t) * c_p_start + t * c_p_final
        curr_comb = np.vstack([c_d, curr_p])
        frames.append(curr_comb)

    write_multimodel_pdb("data/transcription/1aay_transcription_movie.pdb", combined_base, frames)


if __name__ == "__main__":
    generate_docking_movie()
    generate_transcription_movie()
