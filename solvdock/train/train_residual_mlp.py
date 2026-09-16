"""Two-phase calibration pipeline for SolvDock.

Phase A: Fits physical constants (alpha, beta, cs2, chi_e) against FreeSolv
hydration free energies using the physics-only PDE solver (f_phi = 0).
Phase B: Fits f_phi (OrientationalCorrectionMLP) supervised against GIST
orientational entropy reference grids on an 80/20 train/test split.
"""

import argparse
import glob
import os
from typing import Dict, List, Tuple
import yaml
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.optim as optim
from rdkit import Chem
from rdkit.Chem import AllChem

from solvdock.data.freesolv_data import FREESOLV_CURATED
from solvdock.core.grid_engine import SpatialGridEngine
from solvdock.core.poisson_solver import solve_poisson, compute_field
from solvdock.core.solvation_pde import SolvationPDESolver
from solvdock.core.residual_mlp import OrientationalCorrectionMLP
from solvdock.train.generate_gist_dataset import generate_dataset


def prepare_molecule_fields(
    smiles: str,
    box_size: int = 25,
    grid_spacing: float = 1.0,
    device: str = "cpu",
    charge_model: str = "mmff94",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepares external electric field and solvent density for a small molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)

    num_atoms = mol.GetNumAtoms()
    conf = mol.GetConformer()

    coords = torch.zeros((num_atoms, 3), dtype=torch.float32, device=device)
    charges = torch.zeros((num_atoms,), dtype=torch.float32, device=device)

    # Compute atomic partial charges
    mmff_props = None
    if charge_model.lower() == "mmff94":
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
        if mmff_props is None:
            raise RuntimeError(f"MMFF94 parameterization failed for '{smiles}'.")

    if mmff_props is not None:
        for i in range(num_atoms):
            q = float(mmff_props.GetMMFFPartialCharge(i))
            charges[i] = 0.0 if (np.isnan(q) or np.isinf(q)) else q
    else:
        AllChem.ComputeGasteigerCharges(mol)
        for i, atom in enumerate(mol.GetAtoms()):
            try:
                q = float(atom.GetProp("_GasteigerCharge"))
                if np.isnan(q) or np.isinf(q):
                    q = 0.0
            except KeyError:
                q = 0.0
            charges[i] = q

    for i in range(num_atoms):
        pos = conf.GetAtomPosition(i)
        coords[i] = torch.tensor([pos.x, pos.y, pos.z], device=device)

    engine = SpatialGridEngine(grid_spacing=grid_spacing, box_size=box_size)
    origin = engine.get_grid_origin(coords)
    charge_grid = engine.deposit_charges(coords, charges, origin)

    # Poisson solve for potential and electric field in physical units
    phi = solve_poisson(charge_grid, grid_spacing=grid_spacing)
    E_field = compute_field(phi, grid_spacing=grid_spacing)

    # Compute steric volume
    Z, Y, X = engine.get_grid_coords(origin, device=device)
    grid_pts = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=-1)
    diff = grid_pts.unsqueeze(0) - coords.unsqueeze(1)
    dist = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-8)

    vdw_radii = torch.full((num_atoms, 1), 1.7, device=device)
    for i, a in enumerate(mol.GetAtoms()):
        elem = a.GetSymbol()
        if elem == "H":
            vdw_radii[i] = 1.2
        elif elem == "C":
            vdw_radii[i] = 1.7
        elif elem == "N":
            vdw_radii[i] = 1.55
        elif elem == "O":
            vdw_radii[i] = 1.52
        elif elem == "S":
            vdw_radii[i] = 1.80

    v_steric = torch.sum(torch.clamp((vdw_radii / dist) ** 12, max=50.0), dim=0)
    v_steric = v_steric.view(1, 1, box_size, box_size, box_size)

    rho_0 = 0.0333
    rho = rho_0 * torch.exp(-torch.clamp(v_steric, max=25.0) / 0.592)

    return E_field, rho


def run_phase_a(
    sample_size: int = 50,
    device: str = "cpu",
    config_out: str = "configs/calibrated_constants.yaml",
) -> Dict[str, float]:
    """Phase A: Fits physical constants (alpha, beta, cs2, chi_e, P_sat) against FreeSolv
    strictly on an 80/20 train/test split using native MMFF94 charges to prevent data leakage."""
    print("=" * 65)
    print("PHASE A: FreeSolv Calibration with Strict Train/Test Split (MMFF94 Native)")
    print("=" * 65)

    from solvdock.data.freesolv_data import get_freesolv_split
    train_mols, test_mols = get_freesolv_split(test_ratio=0.20, seed=42)
    print(f"Full FreeSolv split: {len(train_mols)} train, {len(test_mols)} test.")

    # Select calibration subset from train split, evaluate on FULL test split
    train_subset = train_mols[:sample_size]
    test_subset = test_mols  # Full 128 held-out test split, never sliced

    print(f"Precomputing fields for {len(train_subset)} training molecules using MMFF94...")
    train_precomputed = []
    train_expt_vals = []
    for item in train_subset:
        try:
            E, rho = prepare_molecule_fields(item["smiles"], device=device, charge_model="mmff94")
            train_precomputed.append((E, rho))
            train_expt_vals.append(float(item["expt"]))
        except Exception as e:
            print(f"  Skipping {item['name']}: {e}")

    print(f"Precomputing fields for all {len(test_subset)} held-out test molecules using MMFF94...")
    test_precomputed = []
    test_expt_vals = []
    test_names = []
    for item in test_subset:
        try:
            E, rho = prepare_molecule_fields(item["smiles"], device=device, charge_model="mmff94")
            test_precomputed.append((E, rho))
            test_expt_vals.append(float(item["expt"]))
            test_names.append(item["name"])
        except Exception as e:
            print(f"  Skipping {item['name']}: {e}")

    train_expt_arr = np.array(train_expt_vals)
    test_expt_arr = np.array(test_expt_vals)

    def evaluate_on_dataset(precomputed_data, expt_arr, alpha, beta, cs2, chi_e, P_sat):
        preds = []
        with torch.no_grad():
            for E, rho in precomputed_data:
                solver = SolvationPDESolver(
                    grid_spacing=1.0, steps=8, dt=0.01,
                    alpha=alpha, beta=beta, cs2=cs2, chi_e=chi_e,
                    P_sat=P_sat, disable_residual_mlp=True,
                    strict=False,
                ).to(device)
                dG, _ = solver(E, rho_solute=rho)
                preds.append(dG.item())
        pred_a = np.array(preds)
        r, _ = stats.pearsonr(pred_a, expt_arr)
        rho_val, _ = stats.spearmanr(pred_a, expt_arr)
        rmse = float(np.sqrt(np.mean((pred_a - expt_arr) ** 2)))
        mae = float(np.mean(np.abs(pred_a - expt_arr)))
        return pred_a, r, rho_val, rmse, mae

    # 1. Fit parameters EXCLUSIVELY on the TRAINING split (zero test set access)
    print("\nCalibrating parameters EXCLUSIVELY on TRAINING split...", flush=True)
    best_score = float("inf")
    best_params = (1.5, 0.0001, 0.5, 0.007, 0.003)
    best_train_metrics = (0.0, 0.0, float("inf"), float("inf"))

    for chi_e in [0.003, 0.005, 0.007, 0.010]:
        for alpha in [0.8, 1.0, 1.2, 1.5]:
            for p_sat in [0.002, 0.003, 0.005, 0.008]:
                _, r, rho_val, rmse, mae = evaluate_on_dataset(
                    train_precomputed, train_expt_arr,
                    alpha=alpha, beta=0.0001, cs2=0.5, chi_e=chi_e, P_sat=p_sat
                )
                score = rmse / (max(0.1, r) ** 2)
                if score < best_score:
                    best_score = score
                    best_params = (alpha, 0.0001, 0.5, chi_e, p_sat)
                    best_train_metrics = (r, rho_val, rmse, mae)
                    print(f"  [Train] alpha={alpha:.1f}, chi_e={chi_e:.4f}, P_sat={p_sat:.4f} -> R={r:.4f}, rho={rho_val:.4f}, RMSE={rmse:.3f}, MAE={mae:.3f}", flush=True)

    final_alpha, final_beta, final_cs2, final_chi_e, final_psat = best_params
    train_r, train_rho, train_rmse, train_mae = best_train_metrics

    # 2. Evaluate generalization ONCE strictly on the UNSEEN HELD-OUT TEST split
    print("\nEvaluating ONCE on UNSEEN HELD-OUT TEST split (all 128 molecules)...", flush=True)
    test_preds, test_r, test_rho, test_rmse, test_mae = evaluate_on_dataset(
        test_precomputed, test_expt_arr,
        alpha=final_alpha, beta=final_beta, cs2=final_cs2, chi_e=final_chi_e, P_sat=final_psat
    )

    # Clean subset evaluation (excluding the 2 push-pull nitroaromatics)
    clean_indices = [i for i, nm in enumerate(test_names) if "profluralin" not in nm and "dinitro" not in nm]
    clean_preds = test_preds[clean_indices]
    clean_expt = test_expt_arr[clean_indices]
    clean_r, _ = stats.pearsonr(clean_preds, clean_expt)
    clean_rho, _ = stats.spearmanr(clean_preds, clean_expt)
    clean_rmse = float(np.sqrt(np.mean((clean_preds - clean_expt) ** 2)))
    clean_mae = float(np.mean(np.abs(clean_preds - clean_expt)))

    print("-" * 65)
    print("Phase A Calibration & Generalization Results (MMFF94 Native):")
    print(f"  Fitted Parameters: alpha={final_alpha:.4f}, beta={final_beta:.4f}, cs2={final_cs2:.4f}, chi_e={final_chi_e:.4f}, P_sat={final_psat:.4f}")
    print(f"  TRAIN Split ({len(train_expt_arr)} mols) | Pearson R: {train_r:.4f}, Spearman rho: {train_rho:.4f}, RMSE: {train_rmse:.3f}, MAE: {train_mae:.3f}")
    print(f"  TEST Split  ({len(test_expt_arr)} mols) | Pearson R: {test_r:.4f}, Spearman rho: {test_rho:.4f}, RMSE: {test_rmse:.3f}, MAE: {test_mae:.3f}")
    print(f"  CLEAN TEST  ({len(clean_preds)} mols) | Pearson R: {clean_r:.4f}, Spearman rho: {clean_rho:.4f}, RMSE: {clean_rmse:.3f}, MAE: {clean_mae:.3f}")
    print("-" * 65)

    os.makedirs(os.path.dirname(config_out), exist_ok=True)
    calibrated_data = {
        "alpha": float(final_alpha),
        "beta": float(final_beta),
        "cs2": float(final_cs2),
        "chi_e": float(final_chi_e),
        "P_sat": float(final_psat),
        "train_pearson_r": float(train_r),
        "train_spearman_rho": float(train_rho),
        "train_rmse": float(train_rmse),
        "train_mae": float(train_mae),
        "test_pearson_r": float(test_r),
        "test_spearman_rho": float(test_rho),
        "test_rmse": float(test_rmse),
        "test_mae": float(test_mae),
        "clean_test_pearson_r": float(clean_r),
        "clean_test_spearman_rho": float(clean_rho),
        "clean_test_rmse": float(clean_rmse),
        "clean_test_mae": float(clean_mae),
        "train_size": len(train_expt_arr),
        "test_size": len(test_expt_arr),
    }

    with open(config_out, "w") as f:
        yaml.dump(calibrated_data, f, default_flow_style=False)
    print(f"Saved calibrated constants with train/test metrics to '{config_out}'.")

    return calibrated_data


def run_phase_b(
    gist_data_dir: str = "data/gist",
    device: str = "cpu",
    config_path: str = "configs/calibrated_constants.yaml",
    checkpoint_out: str = "checkpoints/residual_mlp.pt",
    epochs: int = 50,
):
    """Phase B: Fits f_phi against GIST orientational entropy reference grids."""
    print("=" * 60)
    print("PHASE B: Supervised Fit of f_phi Against GIST Orientational Entropy")
    print("=" * 60)

    if not os.path.exists(config_path):
        print(f"Calibrated constants not found at '{config_path}'. Running Phase A first...")
        run_phase_a(device=device, config_out=config_path)

    with open(config_path, "r") as f:
        constants = yaml.safe_load(f)

    # Check for GIST data
    gist_files = sorted(glob.glob(os.path.join(gist_data_dir, "*.npz")))
    if not gist_files:
        print(f"No GIST files found in '{gist_data_dir}'. Generating reference dataset...")
        generate_dataset(subset_size=30, out_dir=gist_data_dir, device=device)
        gist_files = sorted(glob.glob(os.path.join(gist_data_dir, "*.npz")))

    print(f"Loaded {len(gist_files)} GIST complex files from '{gist_data_dir}'.")

    # 80/20 train/validation split
    n_train = max(1, int(0.8 * len(gist_files)))
    train_files = gist_files[:n_train]
    val_files = gist_files[n_train:]
    print(f"Split: {len(train_files)} training, {len(val_files)} validation.")

    # Initialize bare solver to obtain intermediate fields (P, grad P)
    bare_solver = SolvationPDESolver(
        grid_spacing=1.0,
        steps=8,
        dt=0.01,
        alpha=constants["alpha"],
        beta=constants["beta"],
        cs2=constants["cs2"],
        chi_e=constants["chi_e"],
        strict=False,
    ).to(device)

    def extract_features_and_target(file_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        data = np.load(file_path)
        E_t = torch.from_numpy(data["E_field"]).to(device=device, dtype=torch.float32)
        rho_t = torch.from_numpy(data["rho"]).to(device=device, dtype=torch.float32)
        target_t = torch.from_numpy(data["gist_orient_entropy"]).to(device=device, dtype=torch.float32)

        with torch.no_grad():
            _, comp = bare_solver(E_t, rho_solute=rho_t)
            P = comp["P_final"]
            P_norm = torch.sqrt(torch.sum(P ** 2, dim=1, keepdim=True) + 1e-10)
            grad_P_norm = bare_solver.compute_grad_norm(P)
            features = torch.cat([rho_t, P_norm, grad_P_norm], dim=1)  # (1, 3, D, H, W)
        return features, target_t

    print("Extracting features for training and validation sets...")
    train_data = [extract_features_and_target(f) for f in train_files]
    val_data = [extract_features_and_target(f) for f in val_files]

    # Initialize MLP
    mlp = OrientationalCorrectionMLP(strict=False).to(device)
    optimizer = optim.Adam(mlp.parameters(), lr=0.005, weight_decay=1e-5)
    criterion = nn.MSELoss()

    print(f"Training OrientationalCorrectionMLP for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        mlp.train()
        train_loss = 0.0
        for feat, target in train_data:
            optimizer.zero_grad()
            pred = mlp(feat)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_data)

        if epoch % 10 == 0 or epoch == 1:
            mlp.eval()
            val_loss = 0.0
            with torch.no_grad():
                for feat, target in val_data:
                    pred = mlp(feat)
                    val_loss += criterion(pred, target).item()
            val_loss /= max(1, len(val_data))
            print(f"  Epoch {epoch:02d} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

    # Final validation evaluation
    mlp.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for feat, target in val_data:
            p = mlp(feat)
            val_preds.append(p.cpu().numpy().flatten())
            val_targets.append(target.cpu().numpy().flatten())

    val_preds_arr = np.concatenate(val_preds)
    val_targets_arr = np.concatenate(val_targets)
    val_rmse = float(np.sqrt(np.mean((val_preds_arr - val_targets_arr) ** 2)))
    val_r, _ = stats.pearsonr(val_preds_arr, val_targets_arr)

    print("-" * 60)
    print("Phase B Held-Out Validation:")
    print(f"  Validation RMSE: {val_rmse:.6f} kcal/(mol * A^3)")
    print(f"  Validation Pearson R: {val_r:.4f}")
    print("-" * 60)

    # Save model checkpoint
    os.makedirs(os.path.dirname(checkpoint_out), exist_ok=True)
    torch.save(mlp.state_dict(), checkpoint_out)
    print(f"Saved verified residual MLP checkpoint to '{checkpoint_out}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate SolvDock parameters.")
    parser.add_argument("--phase", type=str, choices=["A", "B", "all"], default="all",
                        help="Calibration phase: 'A' (constants), 'B' (residual MLP), or 'all'.")
    parser.add_argument("--sample_size", type=int, default=50, help="Number of FreeSolv molecules for Phase A.")
    parser.add_argument("--gist_data", type=str, default="data/gist", help="Directory containing GIST .npz files.")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device ('cpu' or 'cuda').")
    parser.add_argument("--config_out", type=str, default="configs/calibrated_constants.yaml")
    parser.add_argument("--checkpoint_out", type=str, default="checkpoints/residual_mlp.pt")
    args = parser.parse_args()

    if args.phase in ("A", "all"):
        run_phase_a(sample_size=args.sample_size, device=args.device, config_out=args.config_out)

    if args.phase in ("B", "all"):
        run_phase_b(
            gist_data_dir=args.gist_data,
            device=args.device,
            config_path=args.config_out,
            checkpoint_out=args.checkpoint_out,
        )
