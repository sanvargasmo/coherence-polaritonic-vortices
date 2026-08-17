from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import coo_matrix, csr_matrix

from .basis import ProductBasis, in_triangle
from .initial_state import InitialStateBuilder
from .parameters import PhysicalParameters, SimulationConfig


def build_coupling_matrices(
    basis: ProductBasis, physical: PhysicalParameters
) -> tuple[csr_matrix, csr_matrix]:
    """Build the sparse A_- and A_+ matrices from the original model."""
    rows_m, cols_m, data_m = [], [], []
    rows_p, cols_p, data_p = [], [], []
    omega = physical.omega_ps_inv
    nc, nx = basis.cavity_cutoff, basis.exciton_cutoff

    for na, ma in basis.cavity_pairs():
        for nb, mb in basis.exciton_pairs():
            dst = basis.state_index(na, ma, nb, mb)
            if na >= 1 and in_triangle(nc, na - 1, ma) and in_triangle(nx, nb + 1, mb):
                src = basis.state_index(na - 1, ma, nb + 1, mb)
                rows_m.append(dst); cols_m.append(src); data_m.append(omega * np.sqrt(na * (nb + 1)))
            if nb >= 1 and in_triangle(nc, na + 1, ma) and in_triangle(nx, nb - 1, mb):
                src = basis.state_index(na + 1, ma, nb - 1, mb)
                rows_p.append(dst); cols_p.append(src); data_p.append(omega * np.sqrt(nb * (na + 1)))
            if ma >= 1 and in_triangle(nc, na, ma - 1) and in_triangle(nx, nb, mb + 1):
                src = basis.state_index(na, ma - 1, nb, mb + 1)
                rows_m.append(dst); cols_m.append(src); data_m.append(omega * np.sqrt(ma * (mb + 1)))
            if mb >= 1 and in_triangle(nc, na, ma + 1) and in_triangle(nx, nb, mb - 1):
                src = basis.state_index(na, ma + 1, nb, mb - 1)
                rows_p.append(dst); cols_p.append(src); data_p.append(omega * np.sqrt(mb * (ma + 1)))

    shape = (basis.dimension, basis.dimension)
    a_minus = coo_matrix((np.asarray(data_m), (rows_m, cols_m)), shape=shape).tocsr()
    a_plus = coo_matrix((np.asarray(data_p), (rows_p, cols_p)), shape=shape).tocsr()
    return a_minus, a_plus


@dataclass
class SimulationResult:
    basis: ProductBasis
    config: SimulationConfig
    physical: PhysicalParameters
    times: np.ndarray
    states: np.ndarray
    initial_state: np.ndarray
    a_minus: csr_matrix
    a_plus: csr_matrix
    solver_success: bool
    solver_message: str
    dense_solution: Callable[[float], np.ndarray] | None = field(default=None, repr=False)

    @property
    def initial_norm(self) -> float:
        return float(np.vdot(self.initial_state, self.initial_state).real)

    def state_at(self, time: float) -> np.ndarray:
        """Return the state at an arbitrary time using the solver interpolant.

        The stable research notebooks used ``solve_ivp(..., dense_output=True)``
        and evaluated ``sol.sol(t)``. Keeping the dense interpolant avoids the
        appreciable phase error that linear interpolation can introduce for
        rapidly oscillating complex amplitudes.
        """
        t = float(time)
        if t <= self.times[0]:
            return self.states[:, 0]
        if t >= self.times[-1]:
            return self.states[:, -1]
        if self.dense_solution is None:
            raise RuntimeError("Dense solver output is unavailable")
        return np.asarray(self.dense_solution(t), dtype=np.complex128).reshape(-1)

    def coefficient_matrix_at(self, time: float) -> np.ndarray:
        return self.state_at(time).reshape(self.basis.n_cavity, self.basis.n_exciton)


def simulate(
    config: SimulationConfig | None = None,
    physical: PhysicalParameters | None = None,
) -> SimulationResult:
    config = config or SimulationConfig()
    physical = physical or PhysicalParameters()
    basis = ProductBasis(config.cavity_cutoff, config.exciton_cutoff)
    initial_state = InitialStateBuilder(basis, physical, config).build()
    a_minus, a_plus = build_coupling_matrices(basis, physical)
    delta_omega = physical.delta_omega_ps_inv

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        phase_plus = np.exp(1j * delta_omega * t)
        phase_minus = np.exp(-1j * delta_omega * t)
        return -1j * (phase_minus * (a_minus @ y) + phase_plus * (a_plus @ y))

    solution = solve_ivp(
        rhs,
        (0.0, config.t_max),
        initial_state,
        t_eval=config.times,
        rtol=config.rtol,
        atol=config.atol,
        method=config.method,
        dense_output=True,
    )
    return SimulationResult(
        basis=basis,
        config=config,
        physical=physical,
        times=np.asarray(solution.t, dtype=float),
        states=np.asarray(solution.y, dtype=np.complex128),
        initial_state=initial_state,
        a_minus=a_minus,
        a_plus=a_plus,
        solver_success=bool(solution.success),
        solver_message=str(solution.message),
        dense_solution=solution.sol,
    )
