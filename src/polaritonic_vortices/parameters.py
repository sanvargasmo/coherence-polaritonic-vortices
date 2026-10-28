from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PhysicalParameters:
    """Physical parameters: energies in meV, time in ps, coordinates in um.

    The effective-energy difference follows Eq. (2) of PRB 110, 155433.
    """

    detuning_mev: float = -1.5
    rabi_energy_mev: float = 2.75
    hbar_si: float = 1.054e-34
    hbar_mev_s: float = 6.5821e-16 * 1e3
    electron_mass_rel: float = 0.18
    hole_mass_rel: float = 0.067
    electron_mass_kg: float = 9.11e-31
    exciton_energy_mev: float = 1.519e3
    speed_of_light_m_s: float = 299_792_458.0
    refractive_index_squared: float = 10.9
    oscillator_width: float = 25.0
    displacement_scale: float = 1.2

    @property
    def exciton_mass_kg(self) -> float:
        return (self.electron_mass_rel + self.hole_mass_rel) * self.electron_mass_kg

    @property
    def cavity_energy_mev(self) -> float:
        return self.exciton_energy_mev + self.detuning_mev

    @property
    def cavity_mass_kg(self) -> float:
        nref = np.sqrt(self.refractive_index_squared)
        return (
            (nref / self.speed_of_light_m_s) ** 2
            * self.cavity_energy_mev
            * (1.6e-19 / 1000.0)
        )

    @property
    def omega_ps_inv(self) -> float:
        omega_s_inv = self.rabi_energy_mev / self.hbar_mev_s
        return omega_s_inv / 1e12

    @property
    def exciton_energy_internal(self) -> float:
        return (
            (self.exciton_energy_mev / self.hbar_mev_s)
            * 1e-12
            * self.hbar_si
        )

    @property
    def cavity_energy_internal(self) -> float:
        return (
            (self.cavity_energy_mev / self.hbar_mev_s)
            * 1e-12
            * self.hbar_si
        )

    @property
    def cavity_x(self) -> float:
        return self.displacement_scale * self.oscillator_width / np.sqrt(2.0)

    @property
    def cavity_y(self) -> float:
        return self.cavity_x

    @property
    def exciton_x(self) -> float:
        return -self.cavity_x

    @property
    def exciton_y(self) -> float:
        return -self.cavity_y

    @property
    def delta_omega_ps_inv(self) -> float:
        """(Ecal_C-Ecal_X)/hbar in ps^-1, with Ecal_k from paper Eq. (2).

        oscillator_width is in micrometres; masses and hbar_si are SI.
        The kinetic correction has the minus sign printed in the paper.
        """
        width_m = self.oscillator_width * 1e-6
        kinetic_ps_inv = (
            self.hbar_si / (4 * width_m**2)
            * (1 / self.cavity_mass_kg - 1 / self.exciton_mass_kg) * 1e-12
        )
        return self.detuning_mev / self.hbar_mev_s * 1e-12 - kinetic_ps_inv


@dataclass(frozen=True)
class SimulationConfig:
    """Numerical settings for one simulation."""

    cavity_cutoff: int = 8
    exciton_cutoff: int = 8
    mean_photon_number: float = 0.5
    coherent_phase: float = np.pi / 4
    t_max: float = 100.0
    n_times: int = 1000
    rtol: float = 1e-9
    atol: float = 1e-12
    method: str = "RK45"
    use_bare_vortex_initial_state: bool = False
    renormalize_initial_state: bool = True

    @property
    def alpha(self) -> complex:
        """Coherent-state amplitude with |alpha|^2 = mean_photon_number."""
        if self.mean_photon_number < 0:
            raise ValueError("mean_photon_number must be non-negative")
        return np.sqrt(self.mean_photon_number) * np.exp(1j * self.coherent_phase)

    @property
    def times(self) -> np.ndarray:
        return np.linspace(0.0, self.t_max, self.n_times)
