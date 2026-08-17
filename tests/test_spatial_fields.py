import numpy as np
from numpy.polynomial.hermite import hermgauss

from polaritonic_vortices import (
    SimulationConfig,
    simulate,
    cavity_density_matrix,
    exciton_density_matrix,
    density_trace,
    cavity_spatial_density,
    exciton_spatial_density,
    joint_wavefunction_same_coordinate,
    cavity_transition_field_x,
    cavity_transition_field_y,
    exciton_transition_field_x,
    exciton_transition_field_y,
    cavity_spatial_density_grid,
    exciton_spatial_density_grid,
    joint_wavefunction_grid_same_coordinate,
    transition_field_grid,
)
from polaritonic_vortices.trajectories import quadratic_density_coefficients


def _small_reference_result():
    return simulate(
        SimulationConfig(
            cavity_cutoff=1,
            exciton_cutoff=1,
            mean_photon_number=0.0,
            t_max=1.0,
            n_times=21,
        )
    )


def test_joint_wavefunction_matches_original_small_notebook_values():
    result = _small_reference_result()
    references = [
        ((0.0, 0.0, 0.0), -0.00030056802367518586j),
        ((5.0, -3.0, 0.0), 5.060513981826196e-06 - 0.0002941423751936472j),
        ((10.0, 4.0, 0.2), 1.963740065848562e-05 - 0.0002582560438091816j),
    ]
    for (x, y, t), expected in references:
        got = joint_wavefunction_same_coordinate(result, x, y, t)
        assert np.allclose(got, expected, rtol=1e-13, atol=1e-18)


def test_transition_fields_match_original_small_notebook_values():
    result = _small_reference_result()
    x, y, t = 5.0, -3.0, 0.0
    assert np.allclose(
        cavity_transition_field_x(result, x, y, t),
        8.321510002644281e-05 + 1.0830335123238596e-05j,
        rtol=1e-13,
        atol=1e-18,
    )
    assert np.allclose(
        cavity_transition_field_y(result, x, y, t),
        -4.167893254925982e-05 - 2.6107240933439838e-06j,
        rtol=1e-13,
        atol=1e-18,
    )
    assert np.allclose(
        exciton_transition_field_x(result, x, y, t),
        7.941971454007314e-05 - 8.146592308609618e-06j,
        rtol=1e-13,
        atol=1e-18,
    )
    assert np.allclose(
        exciton_transition_field_y(result, x, y, t),
        -5.7619551592012305e-05 + 4.220969782121371e-06j,
        rtol=1e-13,
        atol=1e-18,
    )


def test_transition_fields_match_original_at_nonzero_time():
    result = _small_reference_result()
    x, y, t = 10.0, 4.0, 0.2
    expected = [
        0.00012107945667708319 + 1.3660067963633978e-05j,
        4.612837072162619e-05 - 1.7956091293554097e-06j,
        0.00017062954138776832 - 2.4951191956381893e-05j,
        8.34006998088885e-05 + 7.656429586720911e-06j,
    ]
    got = [
        cavity_transition_field_x(result, x, y, t),
        cavity_transition_field_y(result, x, y, t),
        exciton_transition_field_x(result, x, y, t),
        exciton_transition_field_y(result, x, y, t),
    ]
    assert np.allclose(got, expected, rtol=1e-13, atol=1e-18)


def test_spatial_density_at_origin_matches_trajectory_taylor_constant():
    result = simulate(
        SimulationConfig(
            cavity_cutoff=2,
            exciton_cutoff=2,
            mean_photon_number=0.5,
            t_max=0.5,
            n_times=11,
        )
    )
    t = 0.31
    width = result.physical.oscillator_width

    rho_c = cavity_density_matrix(result, t)
    coeff_c = quadratic_density_coefficients(
        rho_c, result.basis.cavity_cutoff, result.basis.cavity_index, width
    )
    rho_x = exciton_density_matrix(result, t)
    coeff_x = quadratic_density_coefficients(
        rho_x, result.basis.exciton_cutoff, result.basis.exciton_index, width
    )

    assert np.isclose(cavity_spatial_density(result, 0.0, 0.0, t), coeff_c["a"].real, atol=1e-16)
    assert np.isclose(exciton_spatial_density(result, 0.0, 0.0, t), coeff_x["a"].real, atol=1e-16)


def test_reduced_spatial_density_integrates_to_reduced_trace():
    result = simulate(
        SimulationConfig(
            cavity_cutoff=2,
            exciton_cutoff=2,
            mean_photon_number=0.5,
            t_max=0.5,
            n_times=11,
        )
    )
    t = 0.31
    width = result.physical.oscillator_width
    nodes, weights = hermgauss(8)
    xs = width * nodes
    ys = width * nodes
    undo_weight = np.exp(nodes[:, None] ** 2 + nodes[None, :] ** 2)
    quadrature_weights = weights[:, None] * weights[None, :]

    cavity_grid = cavity_spatial_density_grid(result, t, xs, ys)
    exciton_grid = exciton_spatial_density_grid(result, t, xs, ys)
    cavity_integral = width**2 * np.sum(quadrature_weights * cavity_grid * undo_weight)
    exciton_integral = width**2 * np.sum(quadrature_weights * exciton_grid * undo_weight)

    assert np.isclose(cavity_integral, density_trace(cavity_density_matrix(result, t)), atol=2e-14)
    assert np.isclose(exciton_integral, density_trace(exciton_density_matrix(result, t)), atol=2e-14)


def test_reduced_spatial_densities_are_nonnegative_within_roundoff():
    result = simulate(
        SimulationConfig(
            cavity_cutoff=3,
            exciton_cutoff=3,
            mean_photon_number=0.75,
            t_max=0.7,
            n_times=15,
        )
    )
    xs = np.linspace(-30.0, 30.0, 9)
    ys = np.linspace(-30.0, 30.0, 8)
    for t in [0.0, 0.237, 0.7]:
        assert np.min(cavity_spatial_density_grid(result, t, xs, ys)) > -1e-15
        assert np.min(exciton_spatial_density_grid(result, t, xs, ys)) > -1e-15


def test_vectorized_grids_match_point_evaluators():
    result = _small_reference_result()
    xs = np.array([-5.0, 0.0, 7.0])
    ys = np.array([-4.0, 3.0])
    t = 0.2

    psi_grid = joint_wavefunction_grid_same_coordinate(result, t, xs, ys)
    cavity_grid = cavity_spatial_density_grid(result, t, xs, ys)
    tx_grid = transition_field_grid(result, t, xs, ys, subsystem="cavity", axis="x")

    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            assert np.allclose(
                psi_grid[j, i], joint_wavefunction_same_coordinate(result, x, y, t), atol=1e-18
            )
            assert np.isclose(
                cavity_grid[j, i], cavity_spatial_density(result, x, y, t), atol=1e-18
            )
            assert np.allclose(
                tx_grid[j, i], cavity_transition_field_x(result, x, y, t), atol=1e-18
            )


def test_spatial_fields_match_original_copy_small_large_basis_at_t0():
    nbar = 2.0 * 0.74162**2
    result = simulate(
        SimulationConfig(
            cavity_cutoff=8,
            exciton_cutoff=8,
            mean_photon_number=nbar,
            t_max=0.01,
            n_times=2,
        )
    )
    x, y, t = 5.0, -3.0, 0.0
    expected = [
        -1.6887464856093505e-05 + 0.00011206102985094105j,
        -4.320796615892901e-05 + 4.631314354532843e-05j,
        -1.057573110865559e-05 + 1.7391772518944428e-06j,
        -1.4321784089868924e-05 + 1.0367439293169702e-05j,
        -2.766524712055581e-05 + 1.3500132563655829e-05j,
    ]
    got = [
        cavity_transition_field_x(result, x, y, t),
        cavity_transition_field_y(result, x, y, t),
        exciton_transition_field_x(result, x, y, t),
        exciton_transition_field_y(result, x, y, t),
        joint_wavefunction_same_coordinate(result, x, y, t),
    ]
    assert np.allclose(got, expected, rtol=2e-13, atol=1e-18)
