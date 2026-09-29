# FreeSolvE: Solvent-Mediated Differentiable Pose Refinement & Biophysical Inductive Bias for PyTorch

[![PyPI](https://img.shields.io/pypi/v/freesolve.svg)](https://pypi.org/project/freesolve/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PoseBusters](https://img.shields.io/badge/PoseBusters-Validated-teal.svg)](https://github.com/trident-bio/posebusters)
[![bioRxiv](https://img.shields.io/badge/Preprint-bioRxiv-red.svg)](MANUSCRIPT_FreeSolvE_Preprint.html)

**FreeSolvE** is a differentiable continuum solvation and articulated kinematics engine natively optimized for both **CPU and GPU (CUDA)**, designed to bridge statistical generative molecular models (DiffDock, AlphaFold3, NeuralPLexer, TANKBind) with physical biophysics at **ultra-high computational speed**.

---

## ⚡ Key Highlights & Benchmark Results

- **+50.0% Absolute Gain on Official PoseBusters Benchmark**: On 50 diverse co-crystal complexes from the official PoseBusters validation package, FreeSolvE rescues severely clashing generative poses, jumping from **2.0% to 52.0% validity** (exceeding raw DiffDock at 36.4%, TANKBind at 24.5%, and EquiBind at 0.3%).
- **Ultra-High Speed on Standard CPU (2.33s per Target)**: Completes full pocket clash rescue and energy minimization in an average of **2.33 seconds per target on a standard laptop CPU** (>18,000× faster than explicit-solvent molecular dynamics), requiring zero expensive GPU hardware for inference while supporting seamless CUDA GPU acceleration for deep learning training.
- **Sub-10ms FFT Continuum Solvation**: Evaluates the complete 3D Poisson electrostatic potential and SASA cavity field via Fourier convolution in **under 8 milliseconds**.
- **PyTorch Training Layer (`FreeSolvEPhysicsLoss`)**: Seamlessly plugs into PyTorch training loops with **< 15 ms overhead per mini-batch step**, eliminating generative pocket clashes from **5 down to 0 within 5 epochs**.
- **Zero-Setup Preparation**: Operates directly on standard PDB and SDF files without requiring brittle manual topology curation, charge parameterization, or missing hydrogen repair (avoiding OpenMM template failure crashes).
- **100% Invariant Covalent Chemistry**: Parameterizes flexibility strictly in internal torsion space ($\mathrm{SO}(3) \times \mathbb{R}^3 \times \mathbb{T}^k$) using forward kinematics, guaranteeing bond lengths and angles remain chemically valid by construction.

---

## 📊 Benchmark Figures

### 1. PoseBusters Physical Validity & Speed Comparison
![PoseBusters Validity and Speed](data/benchmarks/figures/fig_speed_validity.png)

### 2. High-Resolution 3D Clash Relief
![Clash Relief Visualization](data/benchmarks/figures/fig_clash_relief.png)

### 3. Side-by-Side PyTorch Training Dynamics (MSE vs. FreeSolvE Physics)
![AI Training Curves](data/benchmarks/figures/fig_ai_training.png)

---

## 🚀 Installation

Install directly from **PyPI**:

```bash
pip install freesolve
```

Or install in editable mode from source:

```bash
git clone https://github.com/messiay/FreeSolve.git
cd FreeSolve
pip install -e .
```

### Requirements
- Python ≥ 3.9
- PyTorch ≥ 2.0
- RDKit
- NumPy, SciPy, PyYAML

---

## 💻 Quickstart: Using FreeSolvE in PyTorch

### 1. Real-Time Physics Loss During Neural Network Training
Incorporate `FreeSolvEPhysicsLoss` directly into your PyTorch training loop to prevent generative models from producing unphysical steric overlaps:

```python
import torch
import torch.nn as nn
from freesolve import FreeSolvEPhysicsLoss

# 1. Instantiate the differentiable physics loss layer
loss_layer = FreeSolvEPhysicsLoss(
    salt_concentration=0.150,    # 0.150 M physiological saline
    clash_penalty_weight=2.0,     # Penalize severe interatomic overlap (< 2.0 A)
    vdw_weight=1.0,               # Soft-core Lennard-Jones
    elec_weight=1.0               # Dielectric-screened electrostatics
)

# 2. Your deep learning model predicts ligand Cartesian coordinates
pred_coords = model(pocket_features, ligand_features)  # Shape: (N, 3), requires_grad=True

# 3. Compute differentiable physics loss
phys_outputs = loss_layer(
    ligand_coords=pred_coords,
    ligand_charges=ligand_charges,      # Shape: (N,)
    ligand_elements=ligand_elements,    # Shape: (N,) atomic numbers
    pocket_coords=pocket_coords,        # Shape: (M, 3)
    pocket_charges=pocket_charges,      # Shape: (M,)
    pocket_elements=pocket_elements     # Shape: (M,) atomic numbers
)

physics_loss = phys_outputs["total_loss"]
clash_count = phys_outputs["clash_count"]

# 4. Total Loss = Coordinate MSE + Biophysical Inductive Bias
total_loss = mse_loss(pred_coords, true_coords) + 0.08 * physics_loss

# 5. Standard PyTorch backward pass
total_loss.backward()
optimizer.step()
```

---

### 2. Convenience Factory from RDKit Molecules
If you have RDKit molecules, you can automatically extract all partial charges, elements, and coordinates with one line:

```python
from rdkit import Chem
from freesolve import FreeSolvEPhysicsLoss

ligand_mol = Chem.SDMolSupplier("pose.sdf")[0]
pocket_mol = Chem.MolFromPDBFile("pocket.pdb")

# Extract tensors and loss layer automatically
loss_layer, tensors = FreeSolvEPhysicsLoss.from_molecules(ligand_mol, pocket_mol)

# Forward pass using your predicted coordinates
loss_dict = loss_layer(
    ligand_coords=pred_coords,
    **tensors
)
loss_dict["total_loss"].backward()
```

---

### 3. Fast 2-Second Pose Rescue & Induced-Fit Docking
Use FreeSolvE at inference time to instantly rescue clashing poses generated by DiffDock, AlphaFold3, or AutoDock Vina:

```python
from freesolve import dock, relax

# High-speed induced-fit pose rescue (average 2.3 seconds)
result = dock(
    receptor="pocket.pdb",
    ligand="initial_pose.sdf",
    mode="torsional",        # Optimizes internal dihedrals + rigid SE(3)
    max_steps=25,
    lr=0.03
)

print(result.summary())
# Output: DockingResult(clashes: 5 -> 0, loss: 48.2 -> -12.4, time: 2.15s)

# Save refined pose
result.save(ligand_path="rescued_pose.sdf", complex_path="rescued_complex.pdb")
```

---

## 📑 Manuscript & Citation

If you use FreeSolvE in your research, please cite our preprint:

- **Word Document**: [`MANUSCRIPT_FreeSolvE_Preprint.docx`](MANUSCRIPT_FreeSolvE_Preprint.docx)
- **Web & PDF Version**: [`MANUSCRIPT_FreeSolvE_Preprint.html`](MANUSCRIPT_FreeSolvE_Preprint.html)
- **Markdown Source**: [`MANUSCRIPT_FreeSolvE_Preprint.md`](MANUSCRIPT_FreeSolvE_Preprint.md)

```bibtex
@article{arjun2026freesolve,
  title={FreeSolvE: Solvent-Mediated Differentiable Pose Refinement and Ultra-Fast Biophysical Inductive Bias for Molecular Generation and Docking},
  author={Arjun and collaborators},
  journal={bioRxiv / ChemRxiv preprint},
  year={2026}
}
```

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for details.
