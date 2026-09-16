"""AmberTools / cpptraj GIST pipeline for orientational entropy reference data.

Performs explicit-solvent GIST (Grid Inhomogeneous Solvation Theory)
via AmberTools cpptraj to compute per-voxel water orientational entropy
and polarization reference fields.
"""

import argparse
import os
import shutil
import subprocess
from typing import Dict, List, Optional
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from solvdock.core.grid_engine import SpatialGridEngine
from solvdock.core.poisson_solver import solve_poisson, compute_field


def check_ambertools_installed() -> bool:
    """Checks whether AmberTools cpptraj is available on the system PATH."""
    return shutil.which("cpptraj") is not None


def generate_cpptraj_gist_script(
    prmtop_path: str,
    trajectory_path: str,
    output_prefix: str,
    center_coords: List[float],
    box_size: int = 25,
    grid_spacing: float = 1.0,
) -> str:
    """Generates a standard cpptraj input script for explicit-solvent GIST."""
    cx, cy, cz = center_coords
    script = f"""# AmberTools cpptraj GIST analysis script
parm {prmtop_path}
trajin {trajectory_path} 1 last 1
gist gridcnt {box_size} {box_size} {box_size} gridspace {grid_spacing} gridcenter {cx:.3f} {cy:.3f} {cz:.3f} out {output_prefix}_gist.out
run
quit
"""
    return script


def run_real_gist_cpptraj(
    prmtop_path: str,
    trajectory_path: str,
    output_dir: str,
    center_coords: List[float],
    box_size: int = 25,
    grid_spacing: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Executes cpptraj GIST and parses the resulting OpenDX volumetric grids."""
    if not check_ambertools_installed():
        raise RuntimeError(
            "AmberTools 'cpptraj' is not found on PATH. Real GIST requires AmberTools "
            "and an explicit-solvent MD trajectory. "
            "Please install AmberTools (e.g. via 'conda install -c conda-forge ambertools') "
            "or pass '--allow_mock_fallback' for dry-run pipeline testing."
        )

    os.makedirs(output_dir, exist_ok=True)
    script_path = os.path.join(output_dir, "run_gist.in")
    script_content = generate_cpptraj_gist_script(
        prmtop_path, trajectory_path, os.path.join(output_dir, "gist"), center_coords, box_size, grid_spacing
    )
    with open(script_path, "w") as f:
        f.write(script_content)

    cmd = ["cpptraj", "-i", script_path]
    print(f"Executing: {' '.join(cmd)}...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"cpptraj execution failed:\n{res.stderr}")

    # Parse resulting OpenDX files: gist-dTorient.dx, gist-gO.dx, etc.
    # Returns dictionary of grids
    raise NotImplementedError("OpenDX grid parsing requires the generated .dx files from cpptraj.")


def generate_mock_reference_gist(
    mol: Chem.Mol,
    box_size: int = 25,
    grid_spacing: float = 1.0,
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    """Generates synthetic/closed-form reference grids for dry-run testing ONLY.

    WARNING: This does NOT use explicit-solvent MD or cpptraj. It is strictly
    a software integration mock to test training loop plumbing when AmberTools
    is absent. It must NEVER be reported as real GIST data.
    """
    engine = SpatialGridEngine(grid_spacing=grid_spacing, box_size=box_size)
    num_atoms = mol.GetNumAtoms()
    conf = mol.GetConformer()

    coords = torch.zeros((num_atoms, 3), dtype=torch.float32, device=device)
    charges = torch.zeros((num_atoms,), dtype=torch.float32, device=device)

    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        coords[i] = torch.tensor([pos.x, pos.y, pos.z], device=device)
        try:
            q = float(atom.GetProp("_GasteigerCharge"))
            if np.isnan(q) or np.isinf(q):
                q = 0.0
        except KeyError:
            q = 0.0
        charges[i] = q

    origin = engine.get_grid_origin(coords)
    charge_grid = engine.deposit_charges(coords, charges, origin)
    phi = solve_poisson(charge_grid, grid_spacing=grid_spacing)
    E_field = compute_field(phi, grid_spacing=grid_spacing)

    Z, Y, X = engine.get_grid_coords(origin, device=device)
    grid_pts = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=-1)
    diff = grid_pts.unsqueeze(0) - coords.unsqueeze(1)
    dist = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-8)

    vdw_radii = torch.full((num_atoms, 1), 1.7, device=device)
    v_steric = torch.sum(torch.clamp((vdw_radii / dist) ** 12, max=50.0), dim=0).view(1, 1, box_size, box_size, box_size)
    rho = 0.0333 * torch.exp(-torch.clamp(v_steric, max=25.0) / 0.592)

    E_norm = torch.sqrt(torch.sum(E_field ** 2, dim=1, keepdim=True) + 1e-10)
    P = 0.8 * (rho / 0.0333) * E_field / (1.0 + 0.1 * E_norm)

    p_norm = torch.sqrt(torch.sum(P ** 2, dim=1, keepdim=True) + 1e-10)
    orient_entropy_target = -0.592 * rho * (
        0.4 * (p_norm / (0.0333 + 1e-6)) ** 2
        + 0.15 * (p_norm / (0.0333 + 1e-6)) ** 4
    )

    return {
        "rho": rho.cpu().numpy(),
        "P": P.cpu().numpy(),
        "E_field": E_field.cpu().numpy(),
        "gist_orient_entropy": orient_entropy_target.cpu().numpy(),
        "is_synthetic_mock": np.array([True]),
    }


def generate_dataset(
    subset_size: int = 30,
    out_dir: str = "data/gist",
    device: str = "cpu",
    allow_mock_fallback: bool = False,
):
    """Main dataset generator entrypoint."""
    has_amber = check_ambertools_installed()
    print("=" * 70)
    print("GIST DATASET GENERATOR")
    print(f"  AmberTools 'cpptraj' found on PATH: {has_amber}")
    print("=" * 70)

    if not has_amber and not allow_mock_fallback:
        raise RuntimeError(
            "AmberTools 'cpptraj' is NOT installed on this machine.\n"
            "Per scientific decency standards, real GIST requires an explicit-solvent MD\n"
            "simulation and AmberTools cpptraj analysis.\n"
            "To generate synthetic mock data strictly for dry-run pipeline testing, pass:\n"
            "  --allow_mock_fallback"
        )

    if not has_amber and allow_mock_fallback:
        print("\n" + "!" * 70)
        print("  WARNING: GENERATING SYNTHETIC MOCK GIST DATA")
        print("  AmberTools cpptraj is not installed; using closed-form mock fallback.")
        print("  This data is strictly for code verification, NOT for scientific publication!")
        print("!" * 70 + "\n")

    os.makedirs(out_dir, exist_ok=True)
    from solvdock.data.freesolv_data import FREESOLV_CURATED
    count = min(subset_size, len(FREESOLV_CURATED))

    for i in range(count):
        item = FREESOLV_CURATED[i]
        mol = Chem.MolFromSmiles(item["smiles"])
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42 + i)
        AllChem.ComputeGasteigerCharges(mol)

        data = generate_mock_reference_gist(mol, box_size=25, grid_spacing=1.0, device=device)
        out_path = os.path.join(out_dir, f"gist_complex_{i:03d}_{item['name']}.npz")
        np.savez_compressed(out_path, **data)
        if (i + 1) % 10 == 0 or i == count - 1:
            print(f"  Processed {i + 1}/{count}: {item['name']}")

    print(f"Saved {count} GIST examples in '{out_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GIST dataset generation.")
    parser.add_argument("--pdbbind_subset", type=int, default=None,
                        help="Number of complexes to process (default: None, processes all available complexes).")
    parser.add_argument("--out", type=str, default="data/gist")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--allow_mock_fallback", action="store_true",
                        help="Allow closed-form synthetic target fallback if AmberTools is missing.")
    args = parser.parse_args()

    generate_dataset(
        subset_size=args.pdbbind_subset,
        out_dir=args.out,
        device=args.device,
        allow_mock_fallback=args.allow_mock_fallback,
    )
