import numpy as np

from polaritonic_vortices import (
    cavity_density_matrix,
    exciton_density_matrix,
    linear_entropy,
)
from polaritonic_vortices.observables import cavity_lz, exciton_lz, total_lz


def test_linear_entropy_matches_original_trrho_schro_formula(n05_short_result):
    # Original notebooks define trrhoSchro = sum rho_ij rho_ji = Tr(rho^2).
    for t in (0.0, 0.3, 1.0):
        rho_c = cavity_density_matrix(n05_short_result, t)
        direct = 1.0 - np.real(np.sum(rho_c * rho_c.T))
        assert np.isclose(linear_entropy(rho_c), direct, atol=2e-13)


def test_lz_matches_original_n05_reference_values(n05_short_result):
    # Independently evaluated from the original notebook's rho_CS/rho_XS loops.
    expected = {
        0.0: (0.7691437840311992, 0.5479058042156637),
        0.3: (0.4446382150107357, 0.8724113726876690),
        1.0: (0.4109242166925929, 0.9061253698082511),
    }
    for t, (expected_c, expected_x) in expected.items():
        actual_c = cavity_lz(n05_short_result, t)
        actual_x = exciton_lz(n05_short_result, t)
        assert np.isclose(actual_c.real, expected_c, atol=5e-13)
        assert np.isclose(actual_x.real, expected_x, atol=5e-13)
        assert abs(actual_c.imag) < 2e-13
        assert abs(actual_x.imag) < 2e-13


def test_total_lz_is_conserved_in_n05_short_run(n05_short_result):
    values = np.array(
        [total_lz(n05_short_result, t).real for t in np.linspace(0.0, 1.0, 11)]
    )
    assert np.max(np.abs(values - values[0])) < 3e-9


def test_cavity_and_exciton_reduced_states_have_same_linear_entropy(n05_short_result):
    # For the same unnormalized global pure state, both reductions have the same purity.
    for t in (0.0, 0.37, 1.0):
        rho_c = cavity_density_matrix(n05_short_result, t)
        rho_x = exciton_density_matrix(n05_short_result, t)
        assert np.isclose(linear_entropy(rho_c), linear_entropy(rho_x), atol=2e-12)
