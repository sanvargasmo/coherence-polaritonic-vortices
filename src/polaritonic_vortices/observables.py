from __future__ import annotations

import numpy as np

from .basis import in_triangle
from .dynamics import SimulationResult
from .reduced_density import cavity_density_matrix, exciton_density_matrix


def _angular_momentum_from_density(rho: np.ndarray, cutoff: int, basis_index) -> complex:
    """Dimensionless Lz expression used in the original notebooks."""
    total = 0.0 + 0.0j
    for n in range(cutoff + 1):
        for m in range(cutoff - n + 1):
            diagonal_index = basis_index(n, m)
            term1 = 0.0 + 0.0j
            if n >= 1 and in_triangle(cutoff, n - 1, m + 1):
                term1 = rho[basis_index(n - 1, m + 1), diagonal_index]
            term2 = 0.0 + 0.0j
            if m >= 1 and in_triangle(cutoff, n + 1, m - 1):
                term2 = rho[basis_index(n + 1, m - 1), diagonal_index]
            total += np.sqrt(n * (m + 1)) * term1 - np.sqrt(m * (n + 1)) * term2
    return -1j * total


def cavity_lz(result: SimulationResult, time: float) -> complex:
    rho = cavity_density_matrix(result, time)
    return _angular_momentum_from_density(rho, result.basis.cavity_cutoff, result.basis.cavity_index)


def exciton_lz(result: SimulationResult, time: float) -> complex:
    rho = exciton_density_matrix(result, time)
    return _angular_momentum_from_density(rho, result.basis.exciton_cutoff, result.basis.exciton_index)


def total_lz(result: SimulationResult, time: float) -> complex:
    return cavity_lz(result, time) + exciton_lz(result, time)


def characteristic_frequency(result: SimulationResult) -> float:
    """g1 = sqrt(Omega^2 + (DeltaOmega/2)^2) from the notebooks."""
    omega = result.physical.omega_ps_inv
    delta = result.physical.delta_omega_ps_inv
    return float(np.sqrt(omega**2 + (delta / 2.0) ** 2))


def rescaled_to_physical_time(result: SimulationResult, scaled_time) -> np.ndarray:
    g1 = characteristic_frequency(result)
    return np.asarray(scaled_time, dtype=float) * (2.0 * np.pi) / g1
