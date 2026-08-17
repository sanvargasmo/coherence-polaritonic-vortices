import numpy as np

from polaritonic_vortices import SimulationConfig, simulate
from polaritonic_vortices.observables import cavity_lz, exciton_lz, total_lz


def test_lz_is_real_within_numerical_precision():
    result = simulate(
        SimulationConfig(
            cavity_cutoff=2,
            exciton_cutoff=2,
            mean_photon_number=0.5,
            t_max=0.5,
            n_times=11,
        )
    )
    for t in [0.0, 0.13, 0.37, 0.5]:
        assert abs(cavity_lz(result, t).imag) < 1e-10
        assert abs(exciton_lz(result, t).imag) < 1e-10


def test_total_lz_equals_sum_of_subsystems():
    result = simulate(
        SimulationConfig(
            cavity_cutoff=2,
            exciton_cutoff=2,
            mean_photon_number=0.5,
            t_max=0.5,
            n_times=11,
        )
    )
    for t in [0.0, 0.21, 0.5]:
        assert np.allclose(total_lz(result, t), cavity_lz(result, t) + exciton_lz(result, t))


def test_total_lz_conserved_in_reference_dynamics():
    result = simulate(
        SimulationConfig(
            cavity_cutoff=2,
            exciton_cutoff=2,
            mean_photon_number=0.5,
            t_max=1.0,
            n_times=21,
        )
    )
    values = np.array([total_lz(result, t).real for t in result.times])
    assert np.max(np.abs(values - values[0])) < 1e-7
