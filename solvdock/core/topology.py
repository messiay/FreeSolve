"""Molecular DAG decomposition and rotatable bond topology."""

from typing import Dict, List, Tuple
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem


class MolecularTopology:
    """Decomposes an RDKit Mol into a kinematic DAG and rotatable bond tree.

    Identifies rotatable bonds as single, non-ring bonds connecting two
    non-terminal heavy atoms (degree >= 2), determines the DAG root
    (central ring system or central heavy atom), and computes downstream
    rigid fragment masks for each rotatable bond.
    """

    def __init__(self, mol: Chem.Mol, charge_model: str = "mmff94"):
        if mol is None:
            raise ValueError("Input molecule cannot be None.")

        # Work on a cloned, sanitized copy
        self.mol = Chem.Mol(mol)
        Chem.SanitizeMol(self.mol)

        # Ensure 3D conformer exists
        if self.mol.GetNumConformers() == 0:
            status = AllChem.EmbedMolecule(self.mol, randomSeed=42)
            if status != 0:
                AllChem.EmbedMolecule(self.mol, useRandomCoords=True, randomSeed=42)
            AllChem.UFFOptimizeMolecule(self.mol)

        self.num_atoms = self.mol.GetNumAtoms()
        conf = self.mol.GetConformer()

        # Extract atom coords (N, 3), atomic numbers (N,), partial charges (N,)
        coords = np.zeros((self.num_atoms, 3), dtype=np.float32)
        atomic_nums = np.zeros((self.num_atoms,), dtype=np.int64)
        charges = np.zeros((self.num_atoms,), dtype=np.float32)

        for i in range(self.num_atoms):
            pos = conf.GetAtomPosition(i)
            coords[i] = [pos.x, pos.y, pos.z]
            atomic_nums[i] = self.mol.GetAtomWithIdx(i).GetAtomicNum()

        # Compute partial charges via unified charge engine
        from solvdock.core.charges import assign_charges
        charges_t, self.charge_scheme_used = assign_charges(self.mol, scheme=charge_model)

        self.atom_coords = torch.from_numpy(coords)
        self.atomic_numbers = torch.from_numpy(atomic_nums)
        self.partial_charges = charges_t.cpu()

        # Identify rotatable bonds
        # Definition: single, non-ring bonds where BOTH endpoint heavy atoms
        # have heavy degree >= 2 (neither endpoint is a terminal heavy atom).
        self.rotatable_bonds_list: List[Tuple[int, int]] = []
        for bond in self.mol.GetBonds():
            if bond.IsInRing():
                continue
            if bond.GetBondType() != Chem.BondType.SINGLE:
                continue

            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()

            # Skip bonds involving hydrogen
            if a1.GetAtomicNum() <= 1 or a2.GetAtomicNum() <= 1:
                continue

            # Check heavy degree (number of neighbor heavy atoms with Z > 1)
            deg1 = sum(1 for n in a1.GetNeighbors() if n.GetAtomicNum() > 1)
            deg2 = sum(1 for n in a2.GetNeighbors() if n.GetAtomicNum() > 1)

            if deg1 >= 2 and deg2 >= 2:
                self.rotatable_bonds_list.append((a1.GetIdx(), a2.GetIdx()))

        # Determine the root atom / core ring system
        root_idx = self._find_root_atom()
        self.root_idx = root_idx

        # Orient rotatable bonds away from the root and compute downstream masks
        self.rotatable_bonds, self.downstream_masks = self._build_kinematic_dag(root_idx)

    def _find_root_atom(self) -> int:
        """Find the root atom (central ring system or central heavy atom)."""
        ring_info = self.mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()

        heavy_indices = [a.GetIdx() for a in self.mol.GetAtoms() if a.GetAtomicNum() > 1]
        if not heavy_indices:
            return 0

        # Compute all-pairs shortest paths on the heavy atom graph
        dist_matrix = Chem.GetDistanceMatrix(self.mol)

        if atom_rings:
            # Group into connected ring systems
            ring_systems = []
            for ring in atom_rings:
                ring_set = set(ring)
                merged = False
                for existing in ring_systems:
                    if existing & ring_set:
                        existing.update(ring_set)
                        merged = True
                        break
                if not merged:
                    ring_systems.append(ring_set)

            # Pick the largest ring system
            ring_systems.sort(key=lambda s: len(s), reverse=True)
            core_system = list(ring_systems[0])

            # Within core system, pick atom with minimal mean distance to all atoms
            best_atom = core_system[0]
            best_dist = float("inf")
            for idx in core_system:
                mean_d = float(dist_matrix[idx].mean())
                if mean_d < best_dist:
                    best_dist = mean_d
                    best_atom = idx
            return best_atom
        else:
            # Acyclic: graph center with minimal maximum distance (eccentricity)
            best_atom = heavy_indices[0]
            best_ecc = float("inf")
            for idx in heavy_indices:
                ecc = float(dist_matrix[idx].max())
                if ecc < best_ecc:
                    best_ecc = ecc
                    best_atom = idx
            return best_atom

    def _build_kinematic_dag(
        self, root_idx: int
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """Orients rotatable bonds and builds downstream atom masks."""
        dist_matrix = Chem.GetDistanceMatrix(self.mol)
        oriented_bonds: List[Tuple[int, int]] = []
        downstream_masks: Dict[int, torch.Tensor] = {}

        for k, (u, v) in enumerate(self.rotatable_bonds_list):
            # Orient bond (parent -> child) such that parent is closer to root
            if dist_matrix[root_idx, u] > dist_matrix[root_idx, v]:
                parent, child = v, u
            else:
                parent, child = u, v

            oriented_bonds.append((parent, child))

            # Downstream atoms: reachable from `child` in molecular graph
            # without traversing the edge (child, parent)
            visited = set([parent])
            queue = [child]
            downstream = set()

            while queue:
                curr = queue.pop(0)
                if curr in downstream:
                    continue
                downstream.add(curr)
                atom = self.mol.GetAtomWithIdx(curr)
                for nbr in atom.GetNeighbors():
                    nbr_idx = nbr.GetIdx()
                    if nbr_idx not in visited and nbr_idx not in downstream:
                        queue.append(nbr_idx)

            mask = torch.zeros(self.num_atoms, dtype=torch.bool)
            for idx in downstream:
                mask[idx] = True
            downstream_masks[k] = mask

        if oriented_bonds:
            rot_tensor = torch.tensor(oriented_bonds, dtype=torch.int64)
        else:
            rot_tensor = torch.empty((0, 2), dtype=torch.int64)

        return rot_tensor, downstream_masks

    def get_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[int, torch.Tensor]]:
        """Return (atom_coords, atomic_numbers, partial_charges, rotatable_bonds, downstream_masks)."""
        return (
            self.atom_coords,
            self.atomic_numbers,
            self.partial_charges,
            self.rotatable_bonds,
            self.downstream_masks,
        )


if __name__ == "__main__":
    print("Testing MolecularTopology...")
    # Unit test 1: Propanol has exactly 1 rotatable bond
    propanol = Chem.MolFromSmiles("CCCO")
    propanol = Chem.AddHs(propanol)
    AllChem.EmbedMolecule(propanol, randomSeed=42)
    top_prop = MolecularTopology(propanol)
    num_rot_prop = top_prop.rotatable_bonds.shape[0]
    print(f"Propanol rotatable bonds: {num_rot_prop}")
    assert num_rot_prop == 1, f"Expected 1 rotatable bond for propanol, got {num_rot_prop}"

    # Unit test 2: Ethanol has exactly 0 rotatable bonds (negative test)
    ethanol = Chem.MolFromSmiles("CCO")
    ethanol = Chem.AddHs(ethanol)
    AllChem.EmbedMolecule(ethanol, randomSeed=42)
    top_eth = MolecularTopology(ethanol)
    num_rot_eth = top_eth.rotatable_bonds.shape[0]
    print(f"Ethanol rotatable bonds: {num_rot_eth}")
    assert num_rot_eth == 0, f"Expected 0 rotatable bonds for ethanol, got {num_rot_eth}"

    print("All topology unit tests passed successfully!")
