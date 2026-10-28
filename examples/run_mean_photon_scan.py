"""Scan the coherent-background preparation parameter, not a pump rate.

The historical filename is retained for compatibility. |alpha|^2 is not the
mean occupation of the state after multiplication by the vortex factor.
"""
from polaritonic_vortices import ExactCoherentEvolution, SimulationConfig

MEAN_PHOTON_NUMBERS = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00]

for background_alpha2 in MEAN_PHOTON_NUMBERS:
    result = ExactCoherentEvolution(SimulationConfig(mean_photon_number=background_alpha2))
    print(f"background |alpha|^2={background_alpha2:.2f} "
          f"initial S_L={result.entropy(0):.3e} "
          f"initial total Lz/hbar={result.angular_momentum(0,'cavity')+result.angular_momentum(0,'exciton'):.8f}")
