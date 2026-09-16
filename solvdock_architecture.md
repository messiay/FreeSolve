# SolvDock: Differentiable Mean-Field Solvation PDE + Torsion-Space Pose Optimizer

Corrected build spec. Same overall approach as before — physics-based solvation term,
small calibrated residual for the entropy term it's known to miss, gradient-based
pose refinement — with the bugs and missing pieces from the review fixed:
Poisson solve added, residual MLP now has an actual training pipeline instead of
running untrained, quaternion normalization fixed, rotatable-bond test case fixed,
and a real head-to-head benchmark against Vina/DeepRMSD+Vina added.

---

## 1. System Architecture Overview

```
                            [ Receptor PDB + Ligand SDF ]
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    ▼                                           ▼
         [ 1. Pocket Grid Engine ]                   [ 2. Molecular Topology ]
         • Pocket Bounding Box (~25³, Δx = 1.0Å)      • Rigid fragment DAG
         • Gaussian Charge Splatting (σ = 1.0Å)       • Rotatable bond tree (θ, χ)
         • Poisson Solve: ∇²Φ = -ρ_q/ε₀   (NEW)       • Local contact graph (r_cut=4.5Å)
         • E(r) = -∇Φ(r), from Φ not from ρ_q         
                    │                                           │
                    └─────────────────────┬─────────────────────┘
                                          ▼
                    [ 3. Solvation PDE Solver (calibrated) ]
                    • 3D Conv Laplacian ∇²P, Dirichlet BC at grid edge
                    • Ginzburg-Landau relaxation (5-10 unrolled steps, ρ frozen)
                    • α, β, c_s², χ_e FIT against FreeSolv, not hardcoded
                    • f_φ(ρ, ‖P‖, ‖∇P‖): PRETRAINED residual, not random init (NEW)
                    • Integrates to ΔG_solv(g, θ)
                                          │
                                          ▼
                    [ 4. Differentiable Pose Optimizer ]
                    • Rigid rotation via so(3) exponential map (NEW — was raw quaternion)
                    • Torsion updates via Rodrigues FK, DAG root→leaves
                    • Gradient descent (Adam/L-BFGS) on ∂(E_direct+ΔG_solv)/∂(θ,χ,r,T)
                                          │
                                          ▼
                    [ 5. Refined Pose (SDF/PDB) + ΔG Score
                        + CASF-2016 / FreeSolv / PoseBusters reports ]
```

**What changed vs. the previous version, and why:**

| Issue found | Fix |
|---|---|
| E-field computed by differentiating the charge grid directly (dimensionally wrong) | Explicit Poisson solve (FFT Green's-function convolution) between charge deposition and field gradient |
| `f_φ` used at inference with random initialization — an untrained correction is just noise | New `train/` module: pretrain on FreeSolv, then on a GIST-derived pocket dataset, *before* it's used in `pose_optimizer.py` |
| Ethanol test case has zero rotatable bonds under the stated definition, not one | Test case swapped to propanol; ethanol kept as the "zero rotatable bonds" negative test |
| Raw quaternion optimized by Adam drifts off the unit sphere, corrupting rotations | Rigid rotation reparameterized as an so(3) axis-angle vector through `matrix_exp`, valid by construction |
| No comparison against the actual competing baseline (DeepRMSD+Vina) | Added `casf2016_eval.py`, same benchmark/protocol DeepRMSD+Vina reported on |
| ρ silently frozen, dropping the density-polarization coupling term | Documented as an explicit, named simplification (see §6), not hidden |
| Laplacian boundary condition unspecified | Zero-Dirichlet at grid edges, stated explicitly in the solver |

---

## 2. File Structure & Module Tree

```
solvdock/
├── configs/
│   └── default.yaml
├── core/
│   ├── __init__.py
│   ├── topology.py          # RDKit/BioPython DAG decomposition & rotatable bond tree
│   ├── grid_engine.py       # Voxelization + Gaussian charge deposition
│   ├── poisson_solver.py    # NEW — FFT Green's-function solve for Φ, then E = -∇Φ
│   ├── solvation_pde.py     # Ginzburg-Landau PDE solver (ρ frozen, P relaxed)
│   ├── residual_mlp.py      # Per-voxel Δ-learning network (orientational entropy)
│   ├── kinematics.py        # Differentiable Rodrigues FK + so(3) rigid rotation
│   └── pose_optimizer.py    # RENAMED from torsion_gnca.py — gradient-descent refiner,
│                             # not a message-passing network; name now matches behavior
├── train/
│   ├── __init__.py
│   ├── generate_gist_dataset.py  # NEW — offline: run GIST on a PDBbind subset
│   └── train_residual_mlp.py     # NEW — supervised fit of f_φ against FreeSolv + GIST
├── pipeline/
│   ├── __init__.py
│   ├── energy.py             # Combined potential: E_direct (LJ + Coulomb) + ΔG_solv
│   └── refiner.py            # End-to-end PoseRefiner class and high-level API
├── benchmarks/
│   ├── freesolv_eval.py      # Bare solvation model vs. experimental ΔG_hyd
│   ├── casf2016_eval.py      # NEW — head-to-head vs. Vina and DeepRMSD+Vina
│   └── posebusters_eval.py   # Physical plausibility of refined poses
├── tests/
│   ├── test_grid.py
│   ├── test_poisson.py       # NEW
│   ├── test_pde.py
│   ├── test_kinematics.py
│   └── test_energy.py
├── pyproject.toml
└── requirements.txt
```

---

## 3. Module Specifications & Build Sequence

Feed these to your IDE agent (Cursor / Claude Code / Copilot Workspace) **in this exact
order** — each step depends on the last, and step 3 (training) must run and produce a
saved checkpoint before step 4 is allowed to load it.

### Step 1 — Topology, Grid, and Poisson Solve

**Target files:** `core/topology.py`, `core/grid_engine.py`, `core/poisson_solver.py`

**Objective:** decompose the molecule into a kinematic DAG, voxelize charges, and
solve for the actual electrostatic potential field before differentiating it.

```markdown
Generate three Python modules: `solvdock/core/topology.py`,
`solvdock/core/grid_engine.py`, and `solvdock/core/poisson_solver.py` using PyTorch,
RDKit, and NumPy.

### 1. `solvdock/core/topology.py`:
- Class `MolecularTopology`:
  - `__init__(self, mol: rdkit.Chem.Mol)`:
    - Sanitize molecule; compute Gasteiger charges if partial charges absent.
    - Identify rotatable bonds: single, non-ring bonds where BOTH endpoint heavy
      atoms have degree ≥ 2 (i.e., neither endpoint is a terminal heavy atom —
      rotating a bond to a terminal heavy atom only moves hydrogens).
    - Decompose into a DAG of rigid fragments rooted at the largest/most central
      ring system (or central atom if acyclic).
    - Build, for each rotatable bond e_ij, the downstream atom mask that rotates
      rigidly around axis u_ij = (x_j - x_i) / ||x_j - x_i||.
  - Return tensors: atom_coords (N,3), atomic_numbers (N,), partial_charges (N,),
    rotatable_bonds (K,2), and {bond_idx: downstream_mask} dict.

### 2. `solvdock/core/grid_engine.py`:
- Class `SpatialGridEngine`:
  - `__init__(self, grid_spacing: float = 1.0, padding: float = 6.0, box_size: int = 25)`
  - `deposit_charges(coords, charges, grid_origin) -> torch.Tensor`:
    - Gaussian-splat point charges onto a 3D grid, σ = 1.0 Å, shape
      (1, 1, D, H, W). Assert total charge is conserved to within 1e-3.

### 3. `solvdock/core/poisson_solver.py`:
- Function `solve_poisson(charge_grid: torch.Tensor, grid_spacing: float,
  epsilon_0: float = 1.0) -> torch.Tensor`:
  - Solve ∇²Φ = -ρ_q / ε₀ on the periodic/padded grid via FFT: transform ρ_q,
    divide by the discrete Laplacian eigenvalues in Fourier space (zeroing the
    DC/k=0 term to avoid division by zero), inverse-transform to get Φ(r).
  - Zero-pad the grid by at least one box-width before the FFT to approximate
    an open (non-periodic) boundary and avoid self-interaction through the
    periodic images; crop back to the original box size afterward.
- Function `compute_field(phi: torch.Tensor, grid_spacing: float) -> torch.Tensor`:
  - E(r) = -∇Φ(r) via 3D central differences (fixed, non-trainable conv kernel).
    Return shape (1, 3, D, H, W).

Unit tests under `if __name__ == '__main__':`:
1. Propanol decomposes into exactly 1 rotatable bond (the central C-C-C-O
   backbone bond has non-terminal heavy atoms on both sides).
2. Ethanol decomposes into exactly 0 rotatable bonds (both candidate bonds
   have a terminal heavy-atom endpoint) — this is a NEGATIVE test, don't skip it.
3. `solve_poisson` on a single point charge reproduces the analytic 1/r Coulomb
   potential within grid-discretization error at points far from the source.
4. Charge deposition conserves total net charge.
```

### Step 2 — Solvation PDE (physics-only, no untrained correction plugged in yet)

**Target files:** `core/solvation_pde.py`, `core/residual_mlp.py`

**Objective:** implement the relaxation loop and the correction network's
*architecture* — training happens in Step 3, before this network is trusted anywhere.

Equations (ρ held fixed at its initial Boltzmann estimate — see §6 for why):

$$\nabla^2 \mathbf P(\vec r) \approx \frac{1}{\Delta x^2}\sum_{n\in\mathcal N_6}\big(\mathbf P(\vec r_n)-\mathbf P(\vec r)\big), \quad \text{zero-Dirichlet at grid edges}$$

$$\mathbf P^{(t+1)} = \mathbf P^{(t)} + \eta\Big(c_s^2\nabla^2\mathbf P^{(t)} - \alpha\mathbf P^{(t)} - \beta\|\mathbf P^{(t)}\|^2\mathbf P^{(t)} + \chi_e\vec E_{\text{ext}}\Big)$$

$$\Delta G_{\text{solv}} = \int_\Omega\Big(\underbrace{-\tfrac12\vec P\cdot\vec E_{\text{ext}}}_{\text{enthalpy}} + \underbrace{k_BT\,\rho\ln(\rho/\rho_0)}_{\text{translational entropy}} + \underbrace{f_\phi(\rho,\|\vec P\|,\|\nabla\vec P\|)}_{\text{orientational entropy, calibrated in Step 3}}\Big)d^3\vec r$$

```markdown
Generate `solvdock/core/residual_mlp.py` and `solvdock/core/solvation_pde.py` in PyTorch.

### 1. `residual_mlp.py`:
- `OrientationalCorrectionMLP(nn.Module)`:
  - Input: 3 channels per voxel [ρ, ‖P‖, ‖∇P‖].
  - Linear(3,32) -> SiLU -> Linear(32,32) -> SiLU -> Linear(32,1). <2000 params.
  - Add a `load_pretrained(path)` classmethod — the forward pass should refuse
    to run (raise, with a clear message) if no checkpoint has been loaded and
    `strict=True` is set, so it can never silently be used untrained by accident.

### 2. `solvation_pde.py`:
- `SolvationPDESolver(nn.Module)`:
  - `__init__(self, grid_spacing=1.0, steps=8, dt=0.1, alpha=None, beta=None,
    cs2=None, chi_e=None, residual_mlp_path=None)`:
    - If alpha/beta/cs2/chi_e are None, load them from a calibrated constants
      file produced by Step 3's FreeSolv fit — do NOT default to placeholder
      values silently.
    - Register the fixed 7-point 3D Laplacian conv kernel (non-trainable),
      zero-Dirichlet padding.
    - Load `OrientationalCorrectionMLP` from `residual_mlp_path`; error if unset.
  - `forward(self, E_field, rho_solute) -> (delta_G_solv, final_state)`:
    - Initialize P^(0) = 0. ρ^(0) = ρ_0 · exp(-V_steric / k_B T), held fixed
      for the remainder of the rollout (documented simplification, see §6).
    - Unroll `self.steps` updates of the damped Ginzburg-Landau equation above.
    - Compute enthalpy, translational entropy, and f_φ terms per voxel; sum
      and multiply by Δx³ for the volume integral.
  - Full autograd differentiability required: ∂ΔG_solv/∂E_ext must flow through
    every unrolled step. Use gradient checkpointing across steps if memory-bound.
```

### Step 3 — Calibration (the step the previous version skipped entirely)

**Target files:** `train/generate_gist_dataset.py`, `train/train_residual_mlp.py`

**Objective:** produce (a) fitted physical constants (α, β, c_s², χ_e) and (b) a
trained `residual_mlp.py` checkpoint. Nothing downstream is trustworthy without this.

```markdown
Generate `solvdock/train/generate_gist_dataset.py` and
`solvdock/train/train_residual_mlp.py`.

### 1. `generate_gist_dataset.py` (offline, run once, expensive — needs AmberTools):
- For a list of ~20-50 PDBbind complexes: run a short explicit-solvent MD
  simulation, then GIST (via AmberTools `cpptraj`) to get per-voxel enthalpy
  and (translational + orientational) entropy on a grid matching this
  project's grid spacing.
- Save each complex's {ρ, P (approximated from GIST's dipole density output),
  GIST_orientational_entropy_grid} as a training example (.npz).
- This step does NOT need to run on every training iteration — it's a
  one-time dataset-generation cost, keep it decoupled from `train_residual_mlp.py`.

### 2. `train_residual_mlp.py`:
Two-phase calibration:

Phase A — fit physical constants (α, β, c_s², χ_e) against FreeSolv:
  - Load FreeSolv (experimental small-molecule hydration free energies).
  - No protein, no residual MLP yet, no rotatable bonds — the simplest
    possible test of the bare PDE solver.
  - Grid-search or gradient-fit α, β, c_s², χ_e to minimize RMSE between
    predicted ΔG_hyd (enthalpy + translational-entropy terms only,
    f_φ=0) and experimental values. Save to `configs/calibrated_constants.yaml`.
  - Report Pearson R, Spearman ρ, RMSE. If R < ~0.5 here, stop — do not
    proceed to Phase B until the bare physics reproduces known small-molecule
    hydration energies reasonably; a bad bare solver can't be fixed by the
    residual network.

Phase B — fit f_φ against the GIST dataset from generate_gist_dataset.py:
  - With constants frozen from Phase A, run the PDE solver on each GIST
    training complex to get its own (enthalpy, translational-entropy) terms.
  - Train f_φ with a supervised loss:
    L = MSE(f_φ(ρ,‖P‖,‖∇P‖), GIST_orientational_entropy_grid - solver_output)
  - i.e., f_φ is explicitly fit to close the specific, named gap (orientational
    entropy) between the mean-field solver and GIST's reference decomposition —
    not fit to blindly minimize a downstream docking score.
  - Held-out split: train on ~80% of complexes, report residual fit quality on
    the other 20% before this checkpoint is allowed into solvation_pde.py.
  - Save checkpoint to `checkpoints/residual_mlp.pt`.
```

### Step 4 — Differentiable Kinematics and Pose Optimizer

**Target files:** `core/kinematics.py`, `core/pose_optimizer.py`, `pipeline/energy.py`

```markdown
Generate `solvdock/core/kinematics.py`, `solvdock/pipeline/energy.py`, and
`solvdock/core/pose_optimizer.py` in PyTorch.

### 1. `kinematics.py`:
- `rotate_subgraph(coords, origin, axis, theta, mask)`: Rodrigues' formula,
  applied only to masked (downstream) atoms — unchanged from before.
- `apply_torsions(coords, thetas, topology)`: apply hierarchically, DAG
  root -> leaves.
- `apply_rigid_transform(coords, omega: torch.Tensor, T: torch.Tensor)`:
  - `omega` is a 3-vector (so(3) axis-angle), NOT a quaternion.
  - Build the rotation matrix via `torch.linalg.matrix_exp(skew(omega))`
    (skew-symmetric cross-product matrix of omega) — this is valid SO(3)
    by construction for any omega, no renormalization step needed, no drift
    off the rotation manifold during gradient descent.
  - Apply: x' = R @ x + T.

### 2. `pipeline/energy.py`:
- `CombinedPotential(nn.Module)`:
  - `__init__(self, pde_solver: SolvationPDESolver, grid_engine, poisson_solver)`
  - `compute_direct_energy(...)`: 12-6 LJ with soft-core clipping (r_min=0.8Å)
    + Coulombic electrostatics — unchanged.
  - `forward(...)`: E_direct + ΔG_solv (via grid_engine -> poisson_solver ->
    pde_solver, in that order — Poisson solve is now a required hop, not
    optional).

### 3. `pose_optimizer.py` (renamed from `torsion_gnca.py` — this is a
   gradient-descent refiner over scalar pose parameters using
   CombinedPotential as the loss, not a message-passing network; name reflects
   what it actually does):
- `PoseOptimizer.refine(initial_mol, pocket_mol, steps=20, lr=0.05)`:
  - Parameters: θ (K rotatable bonds, zeros, requires_grad), ω (so(3)
    axis-angle, zeros, requires_grad), T (3-vector, zeros, requires_grad).
  - Adam/L-BFGS loop: apply_torsions -> apply_rigid_transform -> CombinedPotential
    -> backward -> step. Early exit if ||∇E|| < 1e-4.
  - Write optimized coordinates back to an RDKit Mol; return Mol + energy
    component breakdown (E_LJ, E_Coulomb, ΔG_solv, ΔG_bind total).
```

### Step 5 — High-Level API and the Full Benchmark Suite

**Target files:** `pipeline/refiner.py`, `benchmarks/freesolv_eval.py`,
`benchmarks/casf2016_eval.py`, `benchmarks/posebusters_eval.py`

```markdown
Generate `solvdock/pipeline/refiner.py` and the three benchmark scripts.

### 1. `refiner.py`:
- `SolvDockRefiner`: factory class wiring together `SpatialGridEngine`,
  `PoissonSolver`, `SolvationPDESolver` (loaded with calibrated constants +
  trained residual MLP — refuse to instantiate otherwise), `MolecularTopology`,
  `PoseOptimizer`.
- `refine_pose(ligand_input, pocket_input, output_path=None, max_steps=20) ->
  Dict[str, Any]`: returns refined Mol, delta_G_bind, and full component breakdown.

### 2. `benchmarks/freesolv_eval.py`:
- Same as before: predicted vs. experimental ΔG_hyd on FreeSolv, no protein.
  Report Pearson R, Spearman ρ, RMSE, runtime/molecule. This should already be
  passing from Step 3's Phase A calibration — this script re-validates it as
  a standalone regression check.

### 3. `benchmarks/casf2016_eval.py` (NEW — the actual competitive comparison):
- Run `SolvDockRefiner.refine_pose()` on the CASF-2016 core-set redocking and
  cross-docking tasks (same protocol DeepRMSD+Vina used).
- Report: RMSD success rate at <2Å in each interval, Spearman correlation for
  scoring power, side-by-side against (a) plain AutoDock Vina scores and
  (b) published DeepRMSD+Vina numbers on the same benchmark.
- This is the number that tells you whether the physics-based solvation term
  actually improved on the fitted-MLP baseline — without it, nothing here is
  falsifiable against the thing it's meant to compete with.

### 4. `benchmarks/posebusters_eval.py`:
- Same as before: physical validity (clashes, bond geometry, planarity) of
  refined poses vs. raw Vina/DiffDock inputs, via the `posebusters` package.
  Additionally run against the PoseBusters *novel-scaffold* subset specifically,
  since that's the generalization claim this whole approach is supposed to test.
```

---

## 4. Verification and Benchmark Protocol

Run in this exact order — later steps depend on artifacts from earlier ones:

```bash
# 1. Unit tests — grid math, Poisson solve, kinematics, invariances
pytest tests/test_grid.py tests/test_poisson.py tests/test_pde.py \
       tests/test_kinematics.py -v

# 2. Phase A calibration — fit physical constants against FreeSolv
python -m solvdock.train.train_residual_mlp --phase A --sample_size 100 --device cuda

# 3. Generate GIST training set (offline, slow — needs AmberTools; run once)
python -m solvdock.train.generate_gist_dataset --pdbbind_subset 40 --out data/gist/

# 4. Phase B calibration — fit the residual MLP against GIST data
python -m solvdock.train.train_residual_mlp --phase B \
       --gist_data data/gist/ --device cuda

# 5. Re-validate FreeSolv standalone (should already pass post-calibration)
python -m solvdock.benchmarks.freesolv_eval --sample_size 100 --device cuda

# 6. Head-to-head vs. Vina and DeepRMSD+Vina on CASF-2016 — the number that matters
python -m solvdock.benchmarks.casf2016_eval --device cuda

# 7. Physical plausibility, including the novel-scaffold subset
python -m solvdock.benchmarks.posebusters_eval --input_dir ./raw_vina_outputs/ \
       --novel_scaffold_subset --device cuda

# 8. Single-complex smoke test
python -c "
from solvdock.pipeline.refiner import SolvDockRefiner
refiner = SolvDockRefiner(device='cuda')
result = refiner.refine_pose(
    ligand_input='tests/data/sample_ligand.sdf',
    pocket_input='tests/data/sample_pocket.pdb',
    output_path='refined_pose.sdf'
)
print('Delta_G_bind:', result['delta_G_bind'], 'kcal/mol')
print('Components:', result['components'])
"
```

Do not run step 6 (or trust any docking-quality number from step 8) before steps 2-4
have completed — a `SolvationPDESolver` instantiated without a calibrated-constants
file and a trained residual checkpoint should raise, not silently fall back to
placeholder values.

---

## 5. What This Is, Honestly

A physics-based (mean-field Ginzburg-Landau) solvation term, with one small, named,
supervised correction for a specific known gap (orientational entropy), plugged into
a standard gradient-based pose optimizer. Validated in three stages: does the bare
physics reproduce known hydration energies (FreeSolv), does the corrected version
track known interfacial solvation thermodynamics (GIST), and does the whole thing
actually do better than the existing fitted-MLP baseline on the same public
benchmark (CASF-2016) — plus whether it degrades more gracefully than that baseline
on scaffolds neither model has seen (PoseBusters novel subset). That last comparison
is the actual thesis of the project; everything else is groundwork for being able to
make that comparison honestly.

## 6. Known Simplifications (stated explicitly, not hidden)

- **ρ is frozen** after its initial Boltzmann estimate; only P is relaxed in the
  unrolled loop. This drops the density-polarization coupling term from the
  original continuous free-energy functional. Cheaper and simpler to train; if this
  is written up later, say so rather than implying full co-relaxation.
- **The Poisson solve assumes open boundary conditions via zero-padding**, not a
  fully accurate treatment of a finite simulation box — adequate for a ~25Å local
  pocket region, would need revisiting for larger systems.
- **f_φ is a small correction, not a full solvation model** — it's fit to close one
  specific, named gap (orientational entropy), not to absorb all model error. If
  CASF-2016 results are poor, check whether the gap is actually elsewhere (e.g., the
  direct-energy LJ/Coulomb term, or the frozen-ρ approximation) before assuming a
  bigger residual network would fix it.
