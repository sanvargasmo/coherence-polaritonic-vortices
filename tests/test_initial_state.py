import numpy as np

from polaritonic_vortices import PhysicalParameters, ProductBasis, SimulationConfig
from polaritonic_vortices.initial_state import InitialStateBuilder
from polaritonic_vortices.dynamics import build_coupling_matrices


def _initial_rho_c00(nbar, cutoff):
    config = SimulationConfig(
        cavity_cutoff=cutoff,
        exciton_cutoff=cutoff,
        mean_photon_number=nbar,
    )
    basis = ProductBasis(cutoff, cutoff)
    state = InitialStateBuilder(basis, PhysicalParameters(), config).build()
    y = state.reshape(basis.n_cavity, basis.n_exciton)
    rho_c = y @ y.conj().T
    return state, rho_c


def test_alpha_encodes_mean_photon_number():
    for nbar in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        config = SimulationConfig(mean_photon_number=nbar)
        assert np.isclose(abs(config.alpha) ** 2, nbar)


def test_small_basis_matches_original_notebook_initial_value():
    # small.ipynb: cutoff=1, alpha=0, rho_CS(0,0;0,0,0)=0.5901639344262293
    state, rho_c = _initial_rho_c00(0.0, 1)
    assert np.isclose(rho_c[0, 0].real, 0.5901639344262293, atol=1e-14)


def test_n05_matches_original_notebook_initial_value():
    # vortex n05.ipynb: cutoff=8, alpha=0.5+0.5j -> nbar=0.5
    state, rho_c = _initial_rho_c00(0.5, 8)
    assert np.isclose(rho_c[0, 0].real, 0.2617573370675023, rtol=1e-12)


def test_sparse_matrix_sizes_match_original_notebooks():
    physical = PhysicalParameters()
    b8 = ProductBasis(8, 8)
    a_m, a_p = build_coupling_matrices(b8, physical)
    assert a_m.nnz + a_p.nnz == 5184
    b10 = ProductBasis(10, 10)
    a_m, a_p = build_coupling_matrices(b10, physical)
    assert a_m.nnz + a_p.nnz == 12100


def test_truncation_is_not_silently_renormalized():
    config = SimulationConfig(
        cavity_cutoff=8,
        exciton_cutoff=8,
        mean_photon_number=1.25,
        renormalize_initial_state=False,
    )
    basis = ProductBasis(8, 8)
    state = InitialStateBuilder(basis, PhysicalParameters(), config).build()
    assert np.vdot(state, state).real < 1.0


def test_historical_rounded_alpha_cases_match_saved_notebook_values():
    # n075 notebook used alpha = 0.612372 + 0.612372j.
    historical_nbar_075 = abs(complex(0.612372, 0.612372)) ** 2
    _, rho_c_075 = _initial_rho_c00(historical_nbar_075, 8)
    assert np.isclose(rho_c_075[0, 0].real, 0.16059565826288721, rtol=1e-12)

    # n125/n150 notebooks used alpha = 0.790569 + 0.790569j.
    historical_nbar_125 = abs(complex(0.790569, 0.790569)) ** 2
    _, rho_c_125 = _initial_rho_c00(historical_nbar_125, 8)
    assert np.isclose(rho_c_125[0, 0].real, 0.056581383660053684, rtol=1e-12)
