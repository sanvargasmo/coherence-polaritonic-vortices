import pytest

from polaritonic_vortices import SimulationConfig, simulate


@pytest.fixture(scope="session")
def n05_short_result():
    """Short nbar=0.5 run matching the stable vortex n05 notebook."""
    return simulate(
        SimulationConfig(
            cavity_cutoff=8,
            exciton_cutoff=8,
            mean_photon_number=0.5,
            t_max=1.0,
            n_times=21,
        )
    )
