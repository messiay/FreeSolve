"""Generates a publication-quality figure comparing Model A vs. Model B training dynamics."""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load experiment results
json_path = "data/benchmarks/ai_training_experiment_results.json"
with open(json_path) as f:
    data = json.load(f)

hist = data["history"]
epochs = hist["epoch"]
clashes_A = hist["model_A_clashes"]
clashes_B = hist["model_B_clashes"]
mse_A = hist["model_A_mse"]
mse_B = hist["model_B_mse"]

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

# 1. Clash Reduction Plot
ax1.plot(epochs, clashes_A, color="#e74c3c", linewidth=2.5, marker="o", markersize=5, label="Model A (Pure Data / MSE)")
ax1.plot(epochs, clashes_B, color="#2ecc71", linewidth=2.5, marker="s", markersize=5, label="Model B (Data + SolvDock Physics)")
ax1.set_title("Intermolecular Steric Pocket Clashes", fontsize=14, fontweight="bold", pad=12)
ax1.set_xlabel("Training Epoch", fontsize=12)
ax1.set_ylabel("Severe Pocket Clashes (< 2.0 Å)", fontsize=12)
ax1.set_ylim(-0.5, max(max(clashes_A), max(clashes_B)) + 1.5)
ax1.axhline(0, color="gray", linestyle="--", alpha=0.6)
ax1.legend(frameon=True, fontsize=11, loc="upper right")
ax1.grid(True, linestyle="--", alpha=0.5)

# 2. MSE Loss Convergence Plot
ax2.plot(epochs, mse_A, color="#e74c3c", linewidth=2.2, linestyle="--", label="Model A Coordinate Error")
ax2.plot(epochs, mse_B, color="#2ecc71", linewidth=2.2, label="Model B Coordinate Error")
ax2.set_title("Coordinate Reconstruction Error", fontsize=14, fontweight="bold", pad=12)
ax2.set_xlabel("Training Epoch", fontsize=12)
ax2.set_ylabel("Mean Squared Error (Å²)", fontsize=12)
ax2.legend(frameon=True, fontsize=11, loc="upper right")
ax2.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
out_png = "data/benchmarks/ai_training_figure.png"
plt.savefig(out_png)
print(f"Publication figure saved to: {out_png}")
