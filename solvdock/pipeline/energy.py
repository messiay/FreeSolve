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

    def compute_solvation(
        self,
        coords: torch.Tensor,
        charges: torch.Tensor,
        grid_origin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Computes solvation free energy for atomic coordinates and partial charges."""
        if coords.shape[0] == 0:
            zero = torch.zeros(1, device=coords.device, dtype=coords.dtype).squeeze()
            return zero, {"enthalpy": zero, "trans_entropy": zero, "orient_entropy": zero}

        if grid_origin is None:
            grid_origin = self.grid_engine.get_grid_origin(coords)

        charge_grid = self.grid_engine.deposit_charges(coords, charges, grid_origin)
        phi = solve_poisson(
            charge_grid,
            grid_spacing=self.grid_engine.grid_spacing,
            epsilon_0=1.0,
            method=self.poisson_method,
        )
        E_field = compute_field(phi, grid_spacing=self.grid_engine.grid_spacing)
        return self.pde_solver(E_field)

    def forward(
        self,
        ligand_coords: torch.Tensor,
        ligand_charges: torch.Tensor,
        ligand_z: torch.Tensor,
        pocket_coords: torch.Tensor,
        pocket_charges: torch.Tensor,
        pocket_z: torch.Tensor,
        grid_origin: Optional[torch.Tensor] = None,
        dG_pocket: Optional[torch.Tensor] = None,
        dG_ligand: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Evaluates thermodynamic binding free energy via the 3-state MM/PBSA cycle:
        Delta_G_bind = E_direct + Delta_G_solv(complex) - Delta_G_solv(pocket) - Delta_G_solv(ligand).
        """
        # Step 1: Direct interaction energy
        e_direct, e_lj, e_coulomb = self.compute_direct_energy(
            ligand_coords, ligand_charges, ligand_z,
            pocket_coords, pocket_charges, pocket_z,
        )

        # Step 2: Solvation of combined complex
        if pocket_coords.shape[0] > 0:
            all_coords = torch.cat([ligand_coords, pocket_coords], dim=0)
            all_charges = torch.cat([ligand_charges, pocket_charges], dim=0)
            dG_complex, pde_comp_c = self.compute_solvation(all_coords, all_charges, grid_origin=grid_origin)
        else:
            dG_complex, pde_comp_c = self.compute_solvation(ligand_coords, ligand_charges, grid_origin=grid_origin)

        # Step 3: Solvation of isolated pocket (can be precomputed outside optimization loop)
        if dG_pocket is None:
            if pocket_coords.shape[0] > 0:
                dG_pocket, _ = self.compute_solvation(pocket_coords, pocket_charges)
            else:
                dG_pocket = torch.zeros(1, device=ligand_coords.device, dtype=ligand_coords.dtype).squeeze()

        # Step 4: Solvation of isolated ligand
        if dG_ligand is None:
            dG_ligand, _ = self.compute_solvation(ligand_coords, ligand_charges)

        # Step 5: Net desolvation and binding free energy
        if pocket_coords.shape[0] > 0:
            ddG_solv = dG_complex - dG_pocket - dG_ligand
            total_energy = e_direct + ddG_solv
        else:
            ddG_solv = dG_complex
            total_energy = dG_complex

        components = {
            "total_energy": total_energy,
            "delta_G_bind": total_energy,
            "e_direct": e_direct,
            "e_lj": e_lj,
            "e_coulomb": e_coulomb,
            "ddG_solv": ddG_solv,
            "delta_G_solv": ddG_solv,
            "dG_complex": dG_complex,
            "dG_pocket": dG_pocket,
            "dG_ligand": dG_ligand,
            "enthalpy": pde_comp_c["enthalpy"],
            "trans_entropy": pde_comp_c["trans_entropy"],
            "orient_entropy": pde_comp_c["orient_entropy"],
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
        grid_origin: Optional[torch.Tensor] = None,
        dG_pocket: Optional[torch.Tensor] = None,
        dG_ligand: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Convenience alias for computing binding free energy via the 3-state thermodynamic cycle."""
        return self.forward(
            ligand_coords, ligand_charges, ligand_z,
            pocket_coords, pocket_charges, pocket_z,
            grid_origin=grid_origin,
            dG_pocket=dG_pocket, dG_ligand=dG_ligand,
        )
