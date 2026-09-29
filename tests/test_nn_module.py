"""Unit tests for SolvDockPhysicsLoss torch.nn.Module integration."""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from rdkit import Chem
from rdkit.Chem import AllChem

from solvdock.nn import SolvDockPhysicsLoss


class ToyGenerativePoseModel(nn.Module):
    """A toy neural network predicting Cartesian coordinate adjustments."""

    def __init__(self, n_atoms: int):
        super().__init__()
        # Takes a 16-dimensional pocket embedding and outputs (N, 3) coordinate shifts
        self.net = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, n_atoms * 3),
        )
        self.n_atoms = n_atoms

    def forward(self, embedding: torch.Tensor, base_coords: torch.Tensor) -> torch.Tensor:
        shifts = self.net(embedding).view(self.n_atoms, 3) * 0.1
        return base_coords + shifts


def test_solvdock_nn_layer_gradient_backprop():
    """Verifies that a neural network can train directly on SolvDock physics loss."""
    # 1. Create small test ligand and pocket
    lig_mol = Chem.AddHs(Chem.MolFromSmiles("CCN(CC)CC"))
    poc_mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    AllChem.EmbedMolecule(lig_mol, randomSeed=42)
    AllChem.EmbedMolecule(poc_mol, randomSeed=42)

    physics_layer, tensors = SolvDockPhysicsLoss.from_molecules(lig_mol, poc_mol)

    n_lig_atoms = lig_mol.GetNumAtoms()
    base_lig_coords = torch.tensor(lig_mol.GetConformer().GetPositions(), dtype=torch.float32)

    # Intentionally position ligand with strong initial clash (< 1.5 A)
    base_lig_coords = base_lig_coords + torch.tensor([0.8, 0.0, 0.0])

    model = ToyGenerativePoseModel(n_lig_atoms)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    pocket_emb = torch.randn(16)

    # 2. Measure initial physical loss
    with torch.no_grad():
        init_coords = model(pocket_emb, base_lig_coords)
        init_res = physics_layer(
            ligand_coords=init_coords,
            ligand_charges=tensors["ligand_charges"],
            ligand_elements=tensors["ligand_elements"],
            pocket_coords=tensors["pocket_coords"],
            pocket_charges=tensors["pocket_charges"],
            pocket_elements=tensors["pocket_elements"],
            ligand_topo_matrix=tensors["ligand_topo_matrix"],
        )
        init_loss = init_res["total_loss"].item()
        init_clashes = int(init_res["clash_count"].item())

    # 3. Train neural network for 15 steps purely through SolvDock physics loss
    loss_history = []
    for step in range(15):
        optimizer.zero_grad()
        predicted_coords = model(pocket_emb, base_lig_coords)
        res = physics_layer(
            ligand_coords=predicted_coords,
            ligand_charges=tensors["ligand_charges"],
            ligand_elements=tensors["ligand_elements"],
            pocket_coords=tensors["pocket_coords"],
            pocket_charges=tensors["pocket_charges"],
            pocket_elements=tensors["pocket_elements"],
            ligand_topo_matrix=tensors["ligand_topo_matrix"],
        )
        loss = res["total_loss"]
        loss.backward()

        # Verify gradients flowed into the neural network weights
        for p in model.parameters():
            assert p.grad is not None
            assert not torch.isnan(p.grad).any()
            assert not torch.isinf(p.grad).any()

        optimizer.step()
        loss_history.append(loss.item())

    final_loss = loss_history[-1]
    print(f"\n[Neural Net Physics Training] Initial Loss: {init_loss:.2f} -> Final Loss: {final_loss:.2f}")

    assert final_loss < init_loss, "Neural network failed to decrease physical energy"
    assert len(loss_history) == 15


if __name__ == "__main__":
    pytest.main(["-v", "tests/test_nn_module.py"])
