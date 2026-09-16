"""Combined potential energy: Direct non-bonded (LJ + Coulomb) + Solvation PDE free energy."""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

from solvdock.core.grid_engine import SpatialGridEngine
from solvdock.core.poisson_solver import solve_poisson, compute_field
from solvdock.core.solvation_pde import SolvationPDESolver


# Standard atom type parameters for Lennard-Jones (Lorentz-Berthelot rules)
# (sigma in Angstroms, epsilon in kcal/mol)
ATOM_PARAMS = {
    1: (2.40, 0.030),    # H
    6: (3.40, 0.100),    # C
    7: (3.25, 0.170),    # N
    8: (3.00, 0.200),    # O
    9: (2.95, 0.150),    # F
    15: (3.70, 0.200),   # P
    16: (3.60, 0.250),   # S
    17: (3.50, 0.250),   # Cl
    35: (3.70, 0.300),   # Br
    53: (4.00, 0.400),   # I
}
DEFAULT_PARAMS = (3.40, 0.100)


class CombinedPotential(nn.Module):
    """Combined scoring function and potential: E_direct (LJ + Coulomb) + ΔG_solv.

    Enforces the required pipeline hop:
    coords/charges -> SpatialGridEngine -> solve_poisson -> compute_field -> SolvationPDESolver.
    """

    def __init__(
        self,
        pde_solver: SolvationPDESolver,
        grid_engine: SpatialGridEngine,
        poisson_method: str = "greens_function",
        r_min: float = 0.8,
    ):
        super().__init__()
        self.pde_solver = pde_solver
        self.grid_engine = grid_engine
        self.poisson_method = str(poisson_method)
        self.r_min = float(r_min)

    def get_atom_params(
        self, atomic_numbers: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Maps atomic numbers to (sigma, epsilon) tensors."""
        sigmas = []
        epsilons = []
        for z in atomic_numbers.cpu().tolist():
            sig, eps = ATOM_PARAMS.get(int(z), DEFAULT_PARAMS)
            sigmas.append(sig)
            epsilons.append(eps)
        sig_tensor = torch.tensor(sigmas, dtype=torch.float32, device=device)
        eps_tensor = torch.tensor(epsilons, dtype=torch.float32, device=device)
        return sig_tensor, eps_tensor

    def compute_direct_energy(
        self,
        ligand_coords: torch.Tensor,
        ligand_charges: torch.Tensor,
        ligand_z: torch.Tensor,
        pocket_coords: torch.Tensor,
        pocket_charges: torch.Tensor,
        pocket_z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes 12-6 Lennard-Jones with soft-core clipping and Coulombic electrostatics.

        Returns (E_direct, E_LJ, E_Coulomb).
        """
        device = ligand_coords.device
        N_lig = ligand_coords.shape[0]
        N_poc = pocket_coords.shape[0]

        if N_poc == 0:
            zero = torch.zeros(1, device=device, dtype=ligand_coords.dtype).squeeze()
            return zero, zero, zero

        # Pairwise distance matrix (N_lig, N_poc)
        diff = ligand_coords.unsqueeze(1) - pocket_coords.unsqueeze(0)  # (N_lig, N_poc, 3)
        dist_sq = torch.sum(diff ** 2, dim=-1)  # (N_lig, N_poc)

        # Soft-core smoothing: r_eff = (r^6 + r_min^6)^(1/6)
        r_min6 = self.r_min ** 6
        r_eff6 = dist_sq ** 3 + r_min6
        r_eff = torch.pow(r_eff6, 1.0 / 6.0)

        # Lennard-Jones combining rules
        sig_lig, eps_lig = self.get_atom_params(ligand_z, device)
        sig_poc, eps_poc = self.get_atom_params(pocket_z, device)

        sigma_ij = 0.5 * (sig_lig.unsqueeze(1) + sig_poc.unsqueeze(0))
        epsilon_ij = torch.sqrt(eps_lig.unsqueeze(1) * eps_poc.unsqueeze(0))

        # 12-6 LJ: 4 * eps * [ (sig/r)^12 - (sig/r)^6 ]
        ratio6 = (sigma_ij ** 6) / r_eff6
        e_lj = torch.sum(4.0 * epsilon_ij * (ratio6 ** 2 - ratio6))

        # Coulomb: 332.0637 * (q_i * q_j) / (eps_eff * r)
        # Using distance-dependent dielectric eps_eff = 2.0 * r_eff for biological screening
        coulomb_const = 332.0637
        e_coulomb = torch.sum(
            coulomb_const * (ligand_charges.unsqueeze(1) * pocket_charges.unsqueeze(0))
            / (2.0 * (r_eff ** 2))
        )

        e_direct = e_lj + e_coulomb
        return e_direct, e_lj, e_coulomb

    def forward(
        self,
        ligand_coords: torch.Tensor,
        ligand_charges: torch.Tensor,
        ligand_z: torch.Tensor,
        pocket_coords: torch.Tensor,
        pocket_charges: torch.Tensor,
        pocket_z: torch.Tensor,
        grid_origin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Evaluates total free energy: E_direct + ΔG_solv.

        Pipeline path:
        1. Direct intermolecular energy (LJ + Coulomb).
        2. Splat combined complex charges onto 3D grid via SpatialGridEngine.
        3. FFT Poisson solve for electrostatic potential Φ.
        4. Central-difference gradient for external field E = -∇Φ.
        5. Unroll Ginzburg-Landau relaxation in SolvationPDESolver to get ΔG_solv.
        """
        # Step 1: Direct interaction energy
        e_direct, e_lj, e_coulomb = self.compute_direct_energy(
            ligand_coords, ligand_charges, ligand_z,
            pocket_coords, pocket_charges, pocket_z,
        )

        # Step 2: Combine atoms into complex
        all_coords = torch.cat([ligand_coords, pocket_coords], dim=0)
        all_charges = torch.cat([ligand_charges, pocket_charges], dim=0)

        # Step 3: Deposit charges onto 3D spatial grid
        if grid_origin is None:
            grid_origin = self.grid_engine.get_grid_origin(all_coords)

        charge_grid = self.grid_engine.deposit_charges(all_coords, all_charges, grid_origin)

        # Step 4: Poisson solve for potential Φ
        phi = solve_poisson(
            charge_grid,
            grid_spacing=self.grid_engine.grid_spacing,
            epsilon_0=1.0,
            method=self.poisson_method,
        )

        # Step 5: Electric field E = -∇Φ
        E_field = compute_field(phi, grid_spacing=self.grid_engine.grid_spacing)

        # Step 6: Solvation PDE Solver
        delta_G_solv, pde_comp = self.pde_solver(E_field)

        # Total energy: E_direct + ΔG_solv
        total_energy = e_direct + delta_G_solv

        components = {
            "total_energy": total_energy,
            "e_direct": e_direct,
            "e_lj": e_lj,
            "e_coulomb": e_coulomb,
            "delta_G_solv": delta_G_solv,
            "enthalpy": pde_comp["enthalpy"],
            "trans_entropy": pde_comp["trans_entropy"],
            "orient_entropy": pde_comp["orient_entropy"],
        }

        return total_energy, components

    def compute_binding_free_energy(
        self,
        ligand_coords: torch.Tensor,
        ligand_charges: torch.Tensor,
        ligand_z: torch.Tensor,
        pocket_coords: torch.Tensor,
        pocket_charges: torch.Tensor,
        pocket_z: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Computes thermodynamic binding free energy via the MM/PBSA cycle:
        Delta_G_bind = E_direct + Delta_G_solv(complex) - Delta_G_solv(pocket) - Delta_G_solv(ligand).
        """
        # Direct intermolecular energy
        e_direct, e_lj, e_coulomb = self.compute_direct_energy(
            ligand_coords, ligand_charges, ligand_z,
            pocket_coords, pocket_charges, pocket_z,
        )

        # 1. Solvation of complex
        all_coords = torch.cat([ligand_coords, pocket_coords], dim=0)
        all_charges = torch.cat([ligand_charges, pocket_charges], dim=0)
        orig_c = self.grid_engine.get_grid_origin(all_coords)
        grid_c = self.grid_engine.deposit_charges(all_coords, all_charges, orig_c)
        phi_c = solve_poisson(grid_c, grid_spacing=self.grid_engine.grid_spacing)
        E_c = compute_field(phi_c, grid_spacing=self.grid_engine.grid_spacing)
        dG_complex, _ = self.pde_solver(E_c)

        # 2. Solvation of isolated pocket
        orig_p = self.grid_engine.get_grid_origin(pocket_coords)
        grid_p = self.grid_engine.deposit_charges(pocket_coords, pocket_charges, orig_p)
        phi_p = solve_poisson(grid_p, grid_spacing=self.grid_engine.grid_spacing)
        E_p = compute_field(phi_p, grid_spacing=self.grid_engine.grid_spacing)
        dG_pocket, _ = self.pde_solver(E_p)

        # 3. Solvation of isolated ligand
        orig_l = self.grid_engine.get_grid_origin(ligand_coords)
        grid_l = self.grid_engine.deposit_charges(ligand_coords, ligand_charges, orig_l)
        phi_l = solve_poisson(grid_l, grid_spacing=self.grid_engine.grid_spacing)
        E_l = compute_field(phi_l, grid_spacing=self.grid_engine.grid_spacing)
        dG_ligand, _ = self.pde_solver(E_l)

        # Net desolvation
        ddG_solv = dG_complex - dG_pocket - dG_ligand
        delta_G_bind = e_direct + ddG_solv

        components = {
            "delta_G_bind": delta_G_bind,
            "e_direct": e_direct,
            "e_lj": e_lj,
            "e_coulomb": e_coulomb,
            "ddG_solv": ddG_solv,
            "dG_complex": dG_complex,
            "dG_pocket": dG_pocket,
            "dG_ligand": dG_ligand,
        }
        return delta_G_bind, components
