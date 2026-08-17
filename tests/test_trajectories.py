import numpy as np
from scipy.special import eval_hermite

from polaritonic_vortices import ProductBasis
from polaritonic_vortices.trajectories import (
    cavity_trajectory_point,
    critical_point_from_quadratic,
    exciton_trajectory_point,
    hermite_values_at_zero,
    oscillator_normalization_table,
    quadratic_density_coefficients,
)


def test_hermite_values_and_derivatives_at_zero():
    h0, h1, h2 = hermite_values_at_zero(6)
    assert np.allclose(h0, [1, 0, -2, 0, 12, 0, -120])
    assert np.allclose(h1, [0, 2, 0, -12, 0, 120, 0])
    assert np.allclose(h2, [0, 0, 8, 0, -96, 0, 1440])


def test_quadratic_coefficients_match_direct_spatial_finite_differences():
    basis = ProductBasis(2, 2)
    rng = np.random.default_rng(17)
    a = rng.normal(size=(basis.n_cavity, basis.n_cavity)) + 1j * rng.normal(
        size=(basis.n_cavity, basis.n_cavity)
    )
    rho = a @ a.conj().T
    width = 25.0
    coeffs = quadratic_density_coefficients(
        rho, 2, basis.cavity_index, width
    )
    cc = oscillator_normalization_table(2, width)
    pairs = list(basis.cavity_pairs())

    def direct_degaussianized_density(x, y):
        ux, uy = x / width, y / width
        total = 0.0 + 0.0j
        for n, m in pairs:
            i = basis.cavity_index(n, m)
            for np_, mp in pairs:
                j = basis.cavity_index(np_, mp)
                total += (
                    cc[n, m]
                    * cc[np_, mp]
                    * rho[i, j]
                    * eval_hermite(n, ux)
                    * eval_hermite(np_, ux)
                    * eval_hermite(m, uy)
                    * eval_hermite(mp, uy)
                )
        return total

    h = 1e-2
    f00 = direct_degaussianized_density(0.0, 0.0)
    b_fd = (
        direct_degaussianized_density(h, 0.0)
        - direct_degaussianized_density(-h, 0.0)
    ) / (2.0 * h)
    c_fd = (
        direct_degaussianized_density(0.0, h)
        - direct_degaussianized_density(0.0, -h)
    ) / (2.0 * h)
    d_fd = (
        direct_degaussianized_density(h, 0.0)
        - 2.0 * f00
        + direct_degaussianized_density(-h, 0.0)
    ) / (2.0 * h * h)
    e_fd = (
        direct_degaussianized_density(0.0, h)
        - 2.0 * f00
        + direct_degaussianized_density(0.0, -h)
    ) / (2.0 * h * h)
    g_fd = (
        direct_degaussianized_density(h, h)
        - direct_degaussianized_density(h, -h)
        - direct_degaussianized_density(-h, h)
        + direct_degaussianized_density(-h, -h)
    ) / (4.0 * h * h)

    assert np.isclose(coeffs["a"], f00, atol=1e-14)
    assert np.isclose(coeffs["b"], b_fd, rtol=3e-7, atol=2e-12)
    assert np.isclose(coeffs["c"], c_fd, rtol=3e-7, atol=2e-12)
    assert np.isclose(coeffs["d"], d_fd, rtol=3e-7, atol=2e-12)
    assert np.isclose(coeffs["e"], e_fd, rtol=3e-7, atol=2e-12)
    assert np.isclose(coeffs["g"], g_fd, rtol=3e-7, atol=2e-12)


def test_critical_point_solves_quadratic_gradient():
    coeffs = {
        "a": 3.0,
        "b": 0.7,
        "c": -0.4,
        "d": 1.2,
        "e": 0.8,
        "g": 0.25,
    }
    x, y = critical_point_from_quadratic(coeffs)
    grad_x = coeffs["b"] + 2 * coeffs["d"] * x + coeffs["g"] * y
    grad_y = coeffs["c"] + coeffs["g"] * x + 2 * coeffs["e"] * y
    assert abs(grad_x) < 1e-14
    assert abs(grad_y) < 1e-14


def test_n05_trajectory_matches_original_reference_points(n05_short_result):
    # Values independently evaluated with the original notebook's exact
    # Hermite/Taylor trajectory implementation.
    expected = {
        0.0: (
            (0.20857306844076862, 0.21135899104023712),
            (-0.2120196723121518, -0.2209916366429713),
        ),
        0.3: (
            (0.3001300369687745, -0.5749097077382170),
            (-0.21501923541114923, -0.021547293275482104),
        ),
        1.0: (
            (-0.1402180652040131, 0.4518842351619486),
            (0.10887748716775399, -0.1601149815708273),
        ),
    }
    for t, (expected_c, expected_x) in expected.items():
        actual_c = cavity_trajectory_point(n05_short_result, t)
        actual_x = exciton_trajectory_point(n05_short_result, t)
        assert np.allclose(actual_c, expected_c, atol=8e-13)
        assert np.allclose(actual_x, expected_x, atol=8e-13)
