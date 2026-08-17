import numpy as np

from polaritonic_vortices import (
    PhysicalParameters,
    ProductBasis,
    SimulationConfig,
    build_coupling_matrices,
    simulate,
    cavity_density_matrix,
    exciton_density_matrix,
    density_trace,
    purity,
)


def test_coupling_matrices_are_adjoint_pairs():
    basis = ProductBasis(5, 5)
    a_minus, a_plus = build_coupling_matrices(basis, PhysicalParameters())
    difference = a_plus - a_minus.getH()
    assert difference.nnz == 0


def test_solver_preserves_global_norm_to_numerical_tolerance():
    config = SimulationConfig(
        cavity_cutoff=4,
        exciton_cutoff=4,
        mean_photon_number=0.5,
        t_max=5.0,
        n_times=101,
    )
    result = simulate(config)
    assert result.solver_success
    norms = np.sum(np.abs(result.states) ** 2, axis=0)
    assert np.max(np.abs(norms - norms[0])) < 2e-8


def test_dense_state_at_matches_independent_saved_grid_solution():
    # 0.55 is not on the coarse 0.1 grid but is on the fine 0.01 grid.
    coarse = simulate(
        SimulationConfig(
            cavity_cutoff=3,
            exciton_cutoff=3,
            mean_photon_number=0.5,
            t_max=1.0,
            n_times=11,
        )
    )
    fine = simulate(
        SimulationConfig(
            cavity_cutoff=3,
            exciton_cutoff=3,
            mean_photon_number=0.5,
            t_max=1.0,
            n_times=101,
        )
    )
    dense_state = coarse.state_at(0.55)
    fine_state = fine.states[:, 55]
    relative_error = np.linalg.norm(dense_state - fine_state) / np.linalg.norm(fine_state)
    assert relative_error < 1e-10


def test_reduced_states_remain_consistent_at_non_grid_time():
    result = simulate(
        SimulationConfig(
            cavity_cutoff=3,
            exciton_cutoff=3,
            mean_photon_number=0.75,
            t_max=1.0,
            n_times=11,
        )
    )
    t = 0.537
    state = result.state_at(t)
    global_norm = float(np.vdot(state, state).real)
    rho_c = cavity_density_matrix(result, t)
    rho_x = exciton_density_matrix(result, t)

    assert np.allclose(rho_c, rho_c.conj().T, atol=1e-12)
    assert np.allclose(rho_x, rho_x.conj().T, atol=1e-12)
    assert np.isclose(density_trace(rho_c), global_norm, atol=1e-11)
    assert np.isclose(density_trace(rho_x), global_norm, atol=1e-11)
    assert np.isclose(purity(rho_c), purity(rho_x), atol=1e-11)


def test_time_rescaling_matches_saved_n05_notebook_output():
    from polaritonic_vortices.observables import rescaled_to_physical_time

    result = simulate(
        SimulationConfig(
            cavity_cutoff=1,
            exciton_cutoff=1,
            mean_photon_number=0.5,
            t_max=1.0,
            n_times=3,
        )
    )
    scaled = np.array([0.0, 1.75, 3.5, 5.37425, 7.0])
    expected = np.array([0.0, 2.53915721, 5.07831442, 7.79775179, 10.15662883])
    assert np.allclose(rescaled_to_physical_time(result, scaled), expected, atol=5e-9)
