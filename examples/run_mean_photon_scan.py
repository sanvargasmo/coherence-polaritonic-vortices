"""Example: coherent-pump scan over the original mean-photon-number values."""

from polaritonic_vortices import (
    SimulationConfig,
    cavity_density_matrix,
    density_trace,
    exciton_density_matrix,
    linear_entropy,
    simulate,
)

MEAN_PHOTON_NUMBERS = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00]

for nbar in MEAN_PHOTON_NUMBERS:
    config = SimulationConfig(mean_photon_number=nbar)
    result = simulate(config)
    rho_c0 = cavity_density_matrix(result, 0.0)
    rho_x0 = exciton_density_matrix(result, 0.0)
    print(
        f"nbar={nbar:>4.2f}  "
        f"|alpha|^2={abs(config.alpha)**2:.6f}  "
        f"initial_norm={result.initial_norm:.8f}  "
        f"Tr(rho_C)={density_trace(rho_c0):.8f}  "
        f"Tr(rho_X)={density_trace(rho_x0):.8f}  "
        f"S_L(C)={linear_entropy(rho_c0):.8f}"
    )
