from __future__ import annotations

import numpy as np

from .dynamics import SimulationResult


def cavity_density_matrix(result: SimulationResult, time: float) -> np.ndarray:
    """Reduced cavity density matrix."""
    y = result.coefficient_matrix_at(time)
    return y @ y.conj().T


def exciton_density_matrix(result: SimulationResult, time: float) -> np.ndarray:
    """Reduced exciton density matrix."""
    y = result.coefficient_matrix_at(time)
    return y.T @ y.conj()


def density_trace(rho: np.ndarray) -> float:
    return float(np.trace(rho).real)


def purity(rho: np.ndarray, normalize: bool = False) -> float:
    """Return Tr(rho^2)."""
    matrix = np.asarray(rho, dtype=np.complex128)
    if normalize:
        tr = np.trace(matrix)
        if abs(tr) == 0:
            raise ValueError("Cannot normalize a density matrix with zero trace")
        matrix = matrix / tr
    return float(np.trace(matrix @ matrix).real)


def linear_entropy(rho: np.ndarray, normalize: bool = False) -> float:
    """Linear entropy 1 - Tr(rho^2)."""
    return 1.0 - purity(rho, normalize=normalize)
