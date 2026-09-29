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
        gamma_rot: float = 0.50,
        max_pair_repulsion: float = 25.0,
        debye_kappa: float = 0.0,
    ):
        super().__init__()
        self.pde_solver = pde_solver
        self.grid_engine = grid_engine
        self.poisson_method = str(poisson_method)
        self.r_min = float(r_min)
        self.gamma_rot = float(gamma_rot)
        self.max_pair_repulsion = float(max_pair_repulsion) if max_pair_repulsion is not None else None
        self.debye_kappa = float(debye_kappa)

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
        """Computes 12-6 Lennard-Jones with soft-core clipping, pair repulsion cap, and Coulombic electrostatics.

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
        pair_lj = 4.0 * epsilon_ij * (ratio6 ** 2 - ratio6)
        if self.max_pair_repulsion is not None:
            # Smooth soft-cap: pair_lj > E_cap transitions smoothly via log1p
            # This maintains continuous outward repulsive gradients instead of killing gradients with hard clamp
            e_cap = self.max_pair_repulsion
            d_e = pair_lj - e_cap
            pair_lj = torch.where(d_e > 0, e_cap + 20.0 * torch.log1p(d_e / 20.0), pair_lj)
        e_lj = torch.sum(pair_lj)

        # Coulomb: 332.0637 * (q_i * q_j) / (eps_eff * r)
        # Using distance-dependent dielectric eps_eff = 2.0 * r_eff for biological screening
        coulomb_const = 332.0637
        pair_coulomb = (
            coulomb_const * (ligand_charges.unsqueeze(1) * pocket_charges.unsqueeze(0))
            / (2.0 * (r_eff ** 2))
        )
        if self.debye_kappa > 0.0:
            pair_coulomb = pair_coulomb * torch.exp(-self.debye_kappa * r_eff)
        e_coulomb = torch.sum(pair_coulomb)

        e_direct = e_lj + e_coulomb
        return e_direct, e_lj, e_coulomb

    def compute_intramolecular_energy(
        self,
        coords: torch.Tensor,
        charges: torch.Tensor,
        atomic_numbers: torch.Tensor,
        topo_scale_matrix: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes intramolecular non-bonded energy (LJ + Coulomb) respecting topological exclusions.

        Returns (E_intra, E_lj, E_coulomb).
        """
        device = coords.device
        N = coords.shape[0]
        if N <= 1:
            zero = torch.zeros(1, device=device, dtype=coords.dtype).squeeze()
            return zero, zero, zero

        diff = coords.unsqueeze(1) - coords.unsqueeze(0)
        dist_sq = torch.sum(diff ** 2, dim=-1)
        r_min6 = self.r_min ** 6
        r_eff6 = dist_sq ** 3 + r_min6
        r_eff = torch.pow(r_eff6, 1.0 / 6.0)

        sig, eps = self.get_atom_params(atomic_numbers, device)
        sigma_ij = 0.5 * (sig.unsqueeze(1) + sig.unsqueeze(0))
        epsilon_ij = torch.sqrt(eps.unsqueeze(1) * eps.unsqueeze(0))

        ratio6 = (sigma_ij ** 6) / r_eff6
        pair_lj = 4.0 * epsilon_ij * (ratio6 ** 2 - ratio6)
        if self.max_pair_repulsion is not None:
            # Smooth soft-cap: maintains continuous outward repulsive gradients
            e_cap = self.max_pair_repulsion
            d_e = pair_lj - e_cap
            pair_lj = torch.where(d_e > 0, e_cap + 20.0 * torch.log1p(d_e / 20.0), pair_lj)

        coulomb_const = 332.0637
        pair_coulomb = (
            coulomb_const * (charges.unsqueeze(1) * charges.unsqueeze(0))
            / (2.0 * (r_eff ** 2))
        )
        if self.debye_kappa > 0.0:
            pair_coulomb = pair_coulomb * torch.exp(-self.debye_kappa * r_eff)

        if topo_scale_matrix is not None:
            scale = topo_scale_matrix.to(device=device, dtype=pair_lj.dtype)
        else:
            scale = torch.ones((N, N), device=device, dtype=pair_lj.dtype)
            scale.fill_diagonal_(0.0)

        # 0.5 prefactor because pairwise matrix has double counted pairs (i, j) and (j, i)
        e_lj = 0.5 * torch.sum(pair_lj * scale)
        e_coulomb = 0.5 * torch.sum(pair_coulomb * scale)
        e_intra = e_lj + e_coulomb
        return e_intra, e_lj, e_coulomb

    def compute_solvent_density(
        self,
        coords: torch.Tensor,
        atomic_numbers: torch.Tensor,
        grid_origin: torch.Tensor,
    ) -> torch.Tensor:
        """Computes Boltzmann excluded-volume solvent density field rho(r)."""
        device = coords.device
        dtype = coords.dtype
        N = coords.shape[0]
        D = H = W = self.grid_engine.box_size
        if N == 0:
            return torch.full((1, 1, D, H, W), 0.0333, device=device, dtype=dtype)

        vdw_map = {1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47, 15: 1.80, 16: 1.80, 17: 1.75, 35: 1.85, 53: 1.98}
        vdw_list = [vdw_map.get(int(z), 1.70) for z in atomic_numbers.cpu().tolist()]
        vdw = torch.tensor(vdw_list, dtype=dtype, device=device).unsqueeze(1)

        Z, Y, X = self.grid_engine.get_grid_coords(grid_origin, device=device)
        grid_pts = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=-1)

        v_steric = torch.zeros(grid_pts.shape[0], device=device, dtype=dtype)
        chunk_size = 200
        for start in range(0, N, chunk_size):
            c_chunk = coords[start : start + chunk_size]
            v_chunk = vdw[start : start + chunk_size]
            diff = grid_pts.unsqueeze(0) - c_chunk.unsqueeze(1)
            dist = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-8)
            v_steric = v_steric + torch.sum(torch.clamp((v_chunk / dist) ** 12, max=50.0), dim=0)

        v_steric = v_steric.view(1, 1, D, H, W)
        rho_0 = 0.0333
        rho = rho_0 * torch.exp(-torch.clamp(v_steric, max=25.0) / 0.592)
        return rho

    def compute_solvation(
        self,
        coords: torch.Tensor,
        charges: torch.Tensor,
        atomic_numbers: Optional[torch.Tensor] = None,
        grid_origin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Computes solvation free energy for atomic coordinates, charges, and solvent cavity."""
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

        rho = None
        if atomic_numbers is not None and atomic_numbers.shape[0] == coords.shape[0]:
            rho = self.compute_solvent_density(coords, atomic_numbers, grid_origin)

        return self.pde_solver(E_field, rho_solute=rho)

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
        num_rotatable_bonds: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Evaluates thermodynamic binding free energy via the 3-state MM/PBSA cycle:
        Delta_G_bind = E_direct + Delta_G_solv(complex) - Delta_G_solv(pocket) - Delta_G_solv(ligand) + Delta_G_rot.
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
            all_z = torch.cat([ligand_z, pocket_z], dim=0) if (ligand_z is not None and pocket_z is not None) else None
            dG_complex, pde_comp_c = self.compute_solvation(
                all_coords, all_charges, atomic_numbers=all_z, grid_origin=grid_origin
            )
        else:
            dG_complex, pde_comp_c = self.compute_solvation(
                ligand_coords, ligand_charges, atomic_numbers=ligand_z, grid_origin=grid_origin
            )

        # Step 3: Solvation of isolated pocket (can be precomputed outside optimization loop)
        if dG_pocket is None:
            if pocket_coords.shape[0] > 0:
                dG_pocket, _ = self.compute_solvation(pocket_coords, pocket_charges, atomic_numbers=pocket_z)
            else:
                dG_pocket = torch.zeros(1, device=ligand_coords.device, dtype=ligand_coords.dtype).squeeze()

        # Step 4: Solvation of isolated ligand
        if dG_ligand is None:
            dG_ligand, _ = self.compute_solvation(ligand_coords, ligand_charges, atomic_numbers=ligand_z)

        # Step 5: Net desolvation, conformational entropy, and binding free energy
        if pocket_coords.shape[0] > 0:
            ddG_solv = dG_complex - dG_pocket - dG_ligand
            total_energy = e_direct + ddG_solv
            if num_rotatable_bonds is not None and num_rotatable_bonds > 0:
                dG_rot = torch.tensor(
                    self.gamma_rot * float(num_rotatable_bonds),
                    dtype=ligand_coords.dtype,
                    device=ligand_coords.device,
                )
                total_energy = total_energy + dG_rot
            else:
                dG_rot = torch.zeros(1, dtype=ligand_coords.dtype, device=ligand_coords.device).squeeze()
        else:
            ddG_solv = dG_complex
            total_energy = dG_complex
            dG_rot = torch.zeros(1, dtype=ligand_coords.dtype, device=ligand_coords.device).squeeze()

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
            "delta_G_rot": dG_rot,
            "gamma_rot": torch.tensor(self.gamma_rot, dtype=ligand_coords.dtype, device=ligand_coords.device),
            "num_rotatable_bonds": torch.tensor(float(num_rotatable_bonds or 0), dtype=ligand_coords.dtype, device=ligand_coords.device),
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
        num_rotatable_bonds: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Convenience alias for computing binding free energy via the 3-state thermodynamic cycle."""
        return self.forward(
            ligand_coords, ligand_charges, ligand_z,
            pocket_coords, pocket_charges, pocket_z,
            grid_origin=grid_origin,
            dG_pocket=dG_pocket, dG_ligand=dG_ligand,
            num_rotatable_bonds=num_rotatable_bonds,
        )
