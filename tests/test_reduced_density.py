import numpy as np

from polaritonic_vortices import (
    PhysicalParameters,
    ProductBasis,
    SimulationConfig,
    density_trace,
    linear_entropy,
)
from polaritonic_vortices.initial_state import InitialStateBuilder


def test_reduced_density_traces_equal_global_norm():
    config = SimulationConfig(
        cavity_cutoff=4,
        exciton_cutoff=4,
        mean_photon_number=0.5,
    )
    basis = ProductBasis(4, 4)
    state = InitialStateBuilder(basis, PhysicalParameters(), config).build()
    y = state.reshape(basis.n_cavity, basis.n_exciton)
    rho_c = y @ y.conj().T
    rho_x = y.T @ y.conj()
    norm = np.vdot(state, state).real
    assert np.isclose(density_trace(rho_c), norm)
    assert np.isclose(density_trace(rho_x), norm)


def test_linear_entropy_name_matches_formula():
    rho = np.diag([0.5, 0.5]).astype(complex)
    assert np.isclose(linear_entropy(rho), 0.5)
