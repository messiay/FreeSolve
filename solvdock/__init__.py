"""SolvDock: Differentiable Mean-Field Solvation PDE + Torsion-Space Pose Optimizer."""

from dataclasses import dataclass
import os
from typing import Optional, Union
import numpy as np
import torch
from rdkit import Chem
from rdkit.Geometry import Point3D

from solvdock.nn import SolvDockPhysicsLoss
from solvdock.pipeline.flexible_refiner import FlexibleRefiner

__version__ = "0.1.0"
__all__ = [
    "dock",
    "relax",
    "DockingResult",
    "SolvDockPhysicsLoss",
    "FlexibleRefiner",
    "__version__",
]


@dataclass
class DockingResult:
    """Encapsulates the output of a SolvDock docking or relaxation run."""

    ligand_mol: Chem.Mol
    pocket_mol: Chem.Mol
    initial_loss: float
    final_loss: float
    initial_clashes: int
    final_clashes: int
    rmsd_to_input: float
    n_ligand_torsions: int
    n_sidechain_torsions: int
    runtime_seconds: float

    def save(self, ligand_path: Optional[str] = None, complex_path: Optional[str] = None) -> None:
        """Saves docked ligand to SDF and/or full complex to PDB."""
        if ligand_path:
            os.makedirs(os.path.dirname(os.path.abspath(ligand_path)), exist_ok=True)
            if ligand_path.endswith(".sdf"):
                writer = Chem.SDWriter(ligand_path)
                writer.write(self.ligand_mol)
                writer.close()
            elif ligand_path.endswith(".pdb"):
                Chem.MolToPDBFile(self.ligand_mol, ligand_path)

        if complex_path:
            os.makedirs(os.path.dirname(os.path.abspath(complex_path)), exist_ok=True)
            combined = Chem.CombineMols(self.pocket_mol, self.ligand_mol)
            Chem.MolToPDBFile(combined, complex_path)

    def summary(self) -> str:
        return (
            f"DockingResult(clashes: {self.initial_clashes} -> {self.final_clashes}, "
            f"loss: {self.initial_loss:.1f} -> {self.final_loss:.1f}, "
            f"rmsd: {self.rmsd_to_input:.3f} A, "
            f"torsions: lig={self.n_ligand_torsions}/sc={self.n_sidechain_torsions}, "
            f"time: {self.runtime_seconds:.2f}s)"
        )


def _load_mol(mol_or_path: Union[str, Chem.Mol], is_protein: bool = False) -> Chem.Mol:
    """Helper to parse a file path or return an existing Chem.Mol."""
    if isinstance(mol_or_path, Chem.Mol):
        return Chem.Mol(mol_or_path)
    if not isinstance(mol_or_path, str):
        raise TypeError(f"Expected str path or Chem.Mol, got {type(mol_or_path)}")

    if not os.path.exists(mol_or_path):
        raise FileNotFoundError(f"File not found: {mol_or_path}")

    if mol_or_path.endswith(".pdb"):
        mol = Chem.MolFromPDBFile(mol_or_path, removeHs=False)
    elif mol_or_path.endswith(".sdf"):
        mol = Chem.SDMolSupplier(mol_or_path, removeHs=False)[0]
    elif mol_or_path.endswith(".mol2"):
        mol = Chem.MolFromMol2File(mol_or_path, removeHs=False)
    else:
        raise ValueError(f"Unsupported file format: {mol_or_path}")

    if mol is None:
        raise ValueError(f"RDKit failed to parse molecule from: {mol_or_path}")
    return mol


def count_clashes(c1: torch.Tensor, c2: torch.Tensor, threshold: float = 2.0) -> int:
    """Counts number of atom pairs with Euclidean distance < threshold."""
    return int(torch.sum(torch.cdist(c1, c2) < threshold).item())


def extract_pocket(protein_mol: Chem.Mol, ligand_coords: np.ndarray, radius: float = 8.0) -> Tuple[Chem.Mol, List[int]]:
    """Extracts residues within radius A of the ligand, preserving full residue structures and indexing."""
    prot_coords = protein_mol.GetConformer().GetPositions()
    min_dists = np.min(np.linalg.norm(prot_coords[:, None, :] - ligand_coords[None, :, :], axis=-1), axis=1)
    pocket_res_keys = set()
    for idx in np.where(min_dists < radius)[0]:
        atom = protein_mol.GetAtomWithIdx(int(idx))
        info = atom.GetPDBResidueInfo()
        if info:
            pocket_res_keys.add((info.GetChainId(), info.GetResidueNumber(), info.GetInsertionCode()))

    if not pocket_res_keys:
        return protein_mol, list(range(protein_mol.GetNumAtoms()))

    pocket_atom_indices = [
        i for i in range(protein_mol.GetNumAtoms())
        if protein_mol.GetAtomWithIdx(i).GetPDBResidueInfo() and 
           (protein_mol.GetAtomWithIdx(i).GetPDBResidueInfo().GetChainId(),
            protein_mol.GetAtomWithIdx(i).GetPDBResidueInfo().GetResidueNumber(),
            protein_mol.GetAtomWithIdx(i).GetPDBResidueInfo().GetInsertionCode()) in pocket_res_keys
    ]

    keep_set = set(pocket_atom_indices)
    rw = Chem.RWMol(protein_mol)
    for i in reversed(range(protein_mol.GetNumAtoms())):
        if i not in keep_set:
            rw.RemoveAtom(i)
    return rw.GetMol(), pocket_atom_indices


def dock(
    receptor: Union[str, Chem.Mol],
    ligand: Union[str, Chem.Mol],
    mode: str = "torsional",
    max_steps: int = 25,
    lr: float = 0.03,
    device: str = "cpu",
    pocket_radius: float = 8.0,
) -> DockingResult:
    """Performs induced-fit docking of a ligand into a receptor binding pocket.

    Args:
        receptor: PDB file path or RDKit Mol of the pocket/receptor.
        ligand: SDF/PDB file path or RDKit Mol of the ligand.
        mode: "torsional" (internal dihedrals + SE(3)) or "cartesian"
        max_steps: Gradient descent steps (default: 25).
        lr: Optimizer learning rate (default: 0.03).
        device: "cpu" or "cuda".
        pocket_radius: Extraction radius in Angstroms if full protein is provided (default: 8.0).

    Returns:
        DockingResult containing refined coordinates, clash statistics, and save methods.
    """
    import time
    start_t = time.time()

    m_rec = _load_mol(receptor, is_protein=True)
    m_lig = _load_mol(ligand, is_protein=False)

    conf_rec = m_rec.GetConformer()
    conf_lig = m_lig.GetConformer()
    c_rec_0 = torch.tensor(conf_rec.GetPositions(), dtype=torch.float32)
    c_lig_0 = torch.tensor(conf_lig.GetPositions(), dtype=torch.float32)

    # Automatic pocket extraction for large whole-protein PDBs (>600 atoms)
    pocket_indices = None
    if m_rec.GetNumAtoms() > 600:
        pocket_mol, pocket_indices = extract_pocket(m_rec, conf_lig.GetPositions(), radius=pocket_radius)
        sim_rec = pocket_mol
    else:
        sim_rec = m_rec

    conf_sim = sim_rec.GetConformer()
    c_sim_0 = torch.tensor(conf_sim.GetPositions(), dtype=torch.float32)
    init_clashes = count_clashes(c_lig_0, c_sim_0, threshold=2.0)

    refiner = FlexibleRefiner(device=device)
    res = refiner.refine_induced_fit(m_lig, sim_rec, mode=mode, max_steps=max_steps, lr=lr)

    final_lig_np = res["final_ligand_coords"].cpu().numpy()
    final_poc_np = res["final_pocket_coords"].cpu().numpy()

    # Update RDKit conformers with relaxed coordinates
    for i in range(m_lig.GetNumAtoms()):
        conf_lig.SetAtomPosition(i, Point3D(float(final_lig_np[i, 0]), float(final_lig_np[i, 1]), float(final_lig_np[i, 2])))

    for i in range(sim_rec.GetNumAtoms()):
        conf_sim.SetAtomPosition(i, Point3D(float(final_poc_np[i, 0]), float(final_poc_np[i, 1]), float(final_poc_np[i, 2])))

    if pocket_indices is not None:
        for local_idx, global_idx in enumerate(pocket_indices):
            conf_rec.SetAtomPosition(global_idx, Point3D(float(final_poc_np[local_idx, 0]), float(final_poc_np[local_idx, 1]), float(final_poc_np[local_idx, 2])))

    c_lig_final = torch.tensor(final_lig_np, dtype=torch.float32)
    c_sim_final = torch.tensor(final_poc_np, dtype=torch.float32)

    final_clashes = count_clashes(c_lig_final, c_sim_final, threshold=2.0)
    rmsd_to_input = float(torch.sqrt(torch.mean(torch.sum((c_lig_final - c_lig_0) ** 2, dim=-1))).item())
    runtime = time.time() - start_t

    return DockingResult(
        ligand_mol=m_lig,
        pocket_mol=sim_rec,
        initial_loss=float(res["initial_loss"]),
        final_loss=float(res["final_loss"]),
        initial_clashes=init_clashes,
        final_clashes=final_clashes,
        rmsd_to_input=rmsd_to_input,
        n_ligand_torsions=int(res.get("n_ligand_torsions", 0)),
        n_sidechain_torsions=int(res.get("n_sidechain_torsions", 0)),
        runtime_seconds=runtime,
    )


def relax(
    receptor: Union[str, Chem.Mol],
    ligand: Union[str, Chem.Mol],
    max_steps: int = 15,
    lr: float = 0.035,
    device: str = "cpu",
) -> DockingResult:
    """Fast 1-second gradient relaxation of an existing complex."""
    return dock(receptor, ligand, mode="torsional", max_steps=max_steps, lr=lr, device=device)
