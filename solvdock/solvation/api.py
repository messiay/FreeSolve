"""Public Python API for SolvDock Solvation Engine."""

from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional, Union
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from solvdock.solvation.applicability import check_applicability_domain, ApplicabilityReport
from solvdock.solvation.born import compute_born_radius, compute_textbook_born_energy
from solvdock.core.solvation_pde import SolvationPDESolver
from solvdock.core.grid_engine import SpatialGridEngine
from solvdock.core.poisson_solver import solve_poisson, compute_field


@dataclass
class SolvationResult:
    """Standardized prediction result from SolvDock Solvation Engine."""
    smiles: str
    delta_g_hyd: float
    estimated_error: float
    uncertainty_provenance: str
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


def get_neutralized_mol(mol: Chem.Mol) -> Chem.Mol:
    """Attempts standard charge neutralization (e.g. carboxylate -> acid, ammonium -> amine)."""
    transforms = [
        ('[O-]', 'O'),
        ('[N+;!H0]', 'N'),
        ('[S-]', 'S'),
        ('[n+;!H0]', 'n'),
    ]
    m = Chem.Mol(mol)
    for p, r in transforms:
        patt = Chem.MolFromSmarts(p)
        repl = Chem.MolFromSmiles(r)
        if patt and m.HasSubstructMatch(patt):
            try:
                m = Chem.ReplaceSubstructs(m, patt, repl, replaceAll=True)[0]
            except Exception:
                pass
    try:
        Chem.SanitizeMol(m)
    except Exception:
        pass
    return m


def _solve_neutral_continuum_pde(
    mol: Chem.Mol,
    active_solver: SolvationPDESolver,
    ionic_strength: float = 0.0,
    charge_model: str = "mmff94",
    compound_id: str = "",
    box_size: int = 24,
    device: str = "cpu",
) -> Tuple[float, str]:
    """Runs the continuum PDE on a neutral molecule."""
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

    from solvdock.core.charges import assign_charges
    charges, charge_model_used = assign_charges(
        mol_with_h, scheme=charge_model, compound_id=compound_id, device=device
    )

    for i in range(num_atoms):
        pos = conf.GetAtomPosition(i)
        coords[i] = torch.tensor([pos.x, pos.y, pos.z], device=device)

    engine = SpatialGridEngine(grid_spacing=1.0, box_size=box_size)
    origin = engine.get_grid_origin(coords)
    charge_grid = engine.deposit_charges(coords, charges, origin)

    phi = solve_poisson(
        charge_grid,
        grid_spacing=1.0,
        ionic_strength=ionic_strength,
    )
    E_field = compute_field(phi, grid_spacing=1.0)

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

    with torch.no_grad():
        dG_continuum, _ = active_solver(E_field, rho_solute=rho)
    dG_continuum_val = float(dG_continuum.detach().cpu().item())

    return dG_continuum_val, charge_model_used


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
            mol = Chem.MolFromSmiles(raw_smiles)
            std_smiles = raw_smiles
        if mol is None:
            raise ValueError(f"Could not parse SMILES: '{raw_smiles}'")
    else:
        mol = molecule
        std_smiles = Chem.MolToSmiles(mol)

    # 2. Applicability Domain and Chemistry Audit
    audit: ApplicabilityReport = check_applicability_domain(mol, compound_id=compound_id)
    flags = list(audit.flags)

    active_solver = solver or get_default_solver(config_path=config_path, device=device)

    # 3. Treatment based on net charge
    if audit.formal_charge == 0:
        # Standard neutral continuum PDE evaluation
        dG_continuum_val, charge_model_used = _solve_neutral_continuum_pde(
            mol,
            active_solver,
            ionic_strength=ionic_strength,
            charge_model=charge_model,
            compound_id=compound_id,
            box_size=box_size,
            device=device,
        )
        total_dG = dG_continuum_val
        components = {
            "dG_continuum_pde": round(dG_continuum_val, 3),
            "dG_born_monopole": 0.0,
            "total_dG_hyd": round(total_dG, 3),
        }
    else:
        # Net-charged species: evaluate textbook Born model on ion + neutral cavity reference
        q = float(audit.formal_charge)
        r_born = compute_born_radius(mol)
        dG_born = compute_textbook_born_energy(q, r_born, epsilon_r=78.4)

        # Evaluate neutral reference for dipolar / cavity component
        neutral_mol = get_neutralized_mol(mol)
        dG_neutral, charge_model_used = _solve_neutral_continuum_pde(
            neutral_mol,
            active_solver,
            ionic_strength=ionic_strength,
            charge_model=charge_model,
            box_size=box_size,
            device=device,
        )
        total_dG = dG_born + dG_neutral
        components = {
            "dG_born_monopole": round(dG_born, 3),
            "dG_neutral_cavity_polar": round(dG_neutral, 3),
            "total_dG_hyd": round(total_dG, 3),
            "born_radius_angstrom": round(r_born, 3),
        }

    # 4. Check if charge model fell back to Gasteiger
    if charge_model_used == "gasteiger" and charge_model.lower() == "mmff94":
        flags.append("CHARGE_MODEL_FALLBACK_GASTEIGER")

    runtime_ms = (time.perf_counter() - t0) * 1000.0

    return SolvationResult(
        smiles=std_smiles,
        delta_g_hyd=round(total_dG, 3),
        estimated_error=audit.base_uncertainty,
        uncertainty_provenance=audit.uncertainty_provenance,
        is_within_applicability_domain=audit.is_within_domain and ("CHARGE_MODEL_FALLBACK_GASTEIGER" not in flags),
        flags=flags,
        components=components,
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
