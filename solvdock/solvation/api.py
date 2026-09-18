"""Public Python API for SolvDock Solvation Engine."""

from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional, Union
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from solvdock.solvation.applicability import check_applicability_domain, ApplicabilityReport
from solvdock.solvation.born import compute_born_ion_correction
from solvdock.core.solvation_pde import SolvationPDESolver
from solvdock.core.grid_engine import SpatialGridEngine
from solvdock.core.poisson_solver import solve_poisson, compute_field


@dataclass
class SolvationResult:
    """Standardized prediction result from SolvDock Solvation Engine."""
    smiles: str
    delta_g_hyd: float
    estimated_error: float
    is_within_applicability_domain: bool
    flags: List[str]
    components: Dict[str, float]
    formal_charge: int
    runtime_ms: float
    charge_model_used: str


_DEFAULT_SOLVER: Optional[SolvationPDESolver] = None


def get_default_solver(
    config_path: str = "configs/solvation_v1_0.yaml",
    device: str = "cpu",
) -> SolvationPDESolver:
    """Returns a singleton SolvationPDESolver initialized with v1.0 parameters."""
    global _DEFAULT_SOLVER
    if _DEFAULT_SOLVER is None:
        _DEFAULT_SOLVER = SolvationPDESolver(
            grid_spacing=1.0,
            steps=8,
            dt=0.01,
            calibrated_constants_path=config_path,
            disable_residual_mlp=True,
            strict=False,
        ).to(device)
    return _DEFAULT_SOLVER


def standardize_smiles(smiles: str) -> str:
    """Normalizes uncharged nitro groups N(=O)=O to canonical zwitterions [N+](=O)[O-]."""
    if "N(=O)=O" in smiles or "O=N=O" in smiles or "N(=O)O" in smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            patt = Chem.MolFromSmarts("[$([NX3](=O)=O),$([NX3+](=O)[O-])]")
            if patt and mol.HasSubstructMatch(patt):
                return Chem.MolToSmiles(mol)
    return smiles


def predict_solvation(
    molecule: Union[str, Chem.Mol],
    compound_id: str = "",
    ionic_strength: float = 0.0,
    charge_model: str = "mmff94",
    box_size: int = 24,
    config_path: str = "configs/solvation_v1_0.yaml",
    device: str = "cpu",
    solver: Optional[SolvationPDESolver] = None,
) -> SolvationResult:
    """Predicts hydration free energy with applicability audit and uncertainty estimation.

    Args:
        molecule: Input SMILES string or RDKit Mol object.
        compound_id: Optional ID for compound provenance and known-class checks.
        ionic_strength: Salt concentration / ionic strength in Molar (default 0.0 for pure water).
        charge_model: Electrostatic partial charge model ('mmff94' or 'gasteiger').
        box_size: Grid box dimension in voxels (default 24x24x24 A).
        config_path: Path to frozen v1.0 constants file.
        device: PyTorch device ('cpu' or 'cuda').
        solver: Optional pre-instantiated SolvationPDESolver.

    Returns:
        SolvationResult dataclass with predicted value, error bar, flags, and energy components.
    """
    t0 = time.perf_counter()

    # 1. Parse and standardize molecule
    if isinstance(molecule, str):
        raw_smiles = molecule
        std_smiles = standardize_smiles(raw_smiles)
        mol = Chem.MolFromSmiles(std_smiles)
        if mol is None:
            # Fallback to raw smiles
            mol = Chem.MolFromSmiles(raw_smiles)
            std_smiles = raw_smiles
        if mol is None:
            raise ValueError(f"Could not parse SMILES: '{raw_smiles}'")
    else:
        mol = molecule
        std_smiles = Chem.MolToSmiles(mol)

    # 2. Applicability Domain and Chemistry Audit
    audit: ApplicabilityReport = check_applicability_domain(mol, compound_id=compound_id)

    # 3. Conformer generation (ETKDG + UFF)
    mol_with_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    res = AllChem.EmbedMolecule(mol_with_h, params)
    if res != 0:
        AllChem.EmbedMolecule(mol_with_h, randomSeed=42, useRandomCoords=True)
    try:
        AllChem.UFFOptimizeMolecule(mol_with_h, maxIters=200)
    except Exception:
        pass

    num_atoms = mol_with_h.GetNumAtoms()
    conf = mol_with_h.GetConformer()
    coords = torch.zeros((num_atoms, 3), dtype=torch.float32, device=device)
    charges = torch.zeros((num_atoms,), dtype=torch.float32, device=device)

    # 4. Partial charge assignment
    charge_model_used = charge_model.lower()
    mmff_props = None
    if charge_model_used == "mmff94":
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol_with_h, mmffVariant="MMFF94")

    if mmff_props is not None:
        for i in range(num_atoms):
            q = float(mmff_props.GetMMFFPartialCharge(i))
            charges[i] = 0.0 if (torch.isnan(torch.tensor(q)) or torch.isinf(torch.tensor(q))) else q
    else:
        # Fallback to Gasteiger
        charge_model_used = "gasteiger"
        AllChem.ComputeGasteigerCharges(mol_with_h)
        for i, atom in enumerate(mol_with_h.GetAtoms()):
            try:
                q = float(atom.GetProp("_GasteigerCharge"))
                charges[i] = 0.0 if (torch.isnan(torch.tensor(q)) or torch.isinf(torch.tensor(q))) else q
            except KeyError:
                charges[i] = 0.0

    for i in range(num_atoms):
        pos = conf.GetAtomPosition(i)
        coords[i] = torch.tensor([pos.x, pos.y, pos.z], device=device)

    # 5. Grid deposition and Poisson solve
    engine = SpatialGridEngine(grid_spacing=1.0, box_size=box_size)
    origin = engine.get_grid_origin(coords)
    charge_grid = engine.deposit_charges(coords, charges, origin)

    phi = solve_poisson(
        charge_grid,
        grid_spacing=1.0,
        ionic_strength=ionic_strength,
    )
    E_field = compute_field(phi, grid_spacing=1.0)

    # 6. Steric repulsion volume
    Z, Y, X = engine.get_grid_coords(origin, device=device)
    grid_pts = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=-1)
    diff = grid_pts.unsqueeze(0) - coords.unsqueeze(1)
    dist = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-8)

    vdw_radii = torch.full((num_atoms, 1), 1.7, device=device)
    for i, a in enumerate(mol_with_h.GetAtoms()):
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
        elif elem == "P":
            vdw_radii[i] = 1.80
        elif elem == "F":
            vdw_radii[i] = 1.47
        elif elem == "Cl":
            vdw_radii[i] = 1.75
        elif elem == "Br":
            vdw_radii[i] = 1.85

    v_steric = torch.sum(torch.clamp((vdw_radii / dist) ** 12, max=50.0), dim=0)
    v_steric = v_steric.view(1, 1, box_size, box_size, box_size)

    rho_0 = 0.0333
    rho = rho_0 * torch.exp(-torch.clamp(v_steric, max=25.0) / 0.592)

    # 7. Continuum PDE relaxation
    active_solver = solver or get_default_solver(config_path=config_path, device=device)
    with torch.no_grad():
        dG_continuum, _ = active_solver(E_field, rho_solute=rho)
    dG_continuum_val = float(dG_continuum.detach().cpu().item())

    # 8. Analytical Born correction for net-charged solutes
    dG_born_val = 0.0
    if audit.formal_charge != 0:
        dG_born_val = compute_born_ion_correction(
            mol_with_h,
            epsilon_r=78.4,
            grid_dimension_angstrom=float(box_size),
        )

    total_dG = dG_continuum_val + dG_born_val
    runtime_ms = (time.perf_counter() - t0) * 1000.0

    return SolvationResult(
        smiles=std_smiles,
        delta_g_hyd=round(total_dG, 3),
        estimated_error=audit.base_uncertainty,
        is_within_applicability_domain=audit.is_within_domain,
        flags=audit.flags,
        components={
            "dG_continuum_pde": round(dG_continuum_val, 3),
            "dG_born_ion_correction": round(dG_born_val, 3),
            "total_dG_hyd": round(total_dG, 3),
        },
        formal_charge=audit.formal_charge,
        runtime_ms=round(runtime_ms, 2),
        charge_model_used=charge_model_used,
    )


def batch_predict_solvation(
    molecules: List[Union[str, Chem.Mol]],
    ionic_strength: float = 0.0,
    charge_model: str = "mmff94",
    box_size: int = 24,
    config_path: str = "configs/solvation_v1_0.yaml",
    device: str = "cpu",
) -> List[SolvationResult]:
    """Evaluates a batch of molecules sequentially or on GPU."""
    solver = get_default_solver(config_path=config_path, device=device)
    results = []
    for mol in molecules:
        res = predict_solvation(
            mol,
            ionic_strength=ionic_strength,
            charge_model=charge_model,
            box_size=box_size,
            config_path=config_path,
            device=device,
            solver=solver,
        )
        results.append(res)
    return results
