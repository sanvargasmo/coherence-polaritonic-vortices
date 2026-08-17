from __future__ import annotations

from math import factorial
import numpy as np

from .basis import ProductBasis
from .parameters import PhysicalParameters, SimulationConfig


class InitialStateBuilder:
    """Initial cavity-exciton state used in the original notebooks."""

    def __init__(self, basis: ProductBasis, physical: PhysicalParameters, config: SimulationConfig) -> None:
        self.basis = basis
        self.physical = physical
        self.config = config
        max_cutoff = max(basis.cavity_cutoff, basis.exciton_cutoff)
        self.factorials = np.array([factorial(k) for k in range(max_cutoff + 2)], dtype=float)

    def oscillator_normalization(self, n: int, m: int) -> float:
        if n < 0 or m < 0:
            return 0.0
        w = self.physical.oscillator_width
        return 1.0 / (w * np.sqrt(np.pi * (2.0 ** (n + m)) * self.factorials[n] * self.factorials[m]))

    def coherent_coefficient(self, n: int, m: int, alpha_x: complex, alpha_y: complex) -> complex:
        if n < 0 or m < 0:
            return 0.0 + 0.0j
        return (
            np.exp(-(abs(alpha_x) ** 2 + abs(alpha_y) ** 2) / 2)
            * alpha_x**n * alpha_y**m
            / np.sqrt(self.factorials[n] * self.factorials[m])
        )

    def cavity_vortex_coefficient(self, n: int, m: int, alpha_x: complex, alpha_y: complex) -> complex:
        p = self.physical
        w, x_c, y_c = p.oscillator_width, p.cavity_x, p.cavity_y
        sqrt2 = np.sqrt(2.0)
        denominator = np.sqrt(w**2 + (x_c - sqrt2 * w * alpha_x.real) ** 2 + (y_c - sqrt2 * w * alpha_y.real) ** 2)
        c = self.coherent_coefficient
        cc = self.oscillator_normalization
        t1 = c(n + 1, m, alpha_x, alpha_y) * (cc(n + 1, m) / cc(n, m)) * w * (n + 1)
        t2 = 1j * c(n, m + 1, alpha_x, alpha_y) * (cc(n, m + 1) / cc(n, m)) * w * (m + 1)
        t3 = c(n - 1, m, alpha_x, alpha_y) * (cc(n - 1, m) / cc(n, m)) * (w / 2.0) * (1 if n > 0 else 0)
        t4 = 1j * c(n, m - 1, alpha_x, alpha_y) * (cc(n, m - 1) / cc(n, m)) * (w / 2.0) * (1 if m > 0 else 0)
        t5 = -(x_c + 1j * y_c) * c(n, m, alpha_x, alpha_y)
        return (t1 + t2 + t3 + t4 + t5) / denominator

    def exciton_vortex_coefficient(self, n: int, m: int, alpha_x: complex, alpha_y: complex) -> complex:
        p = self.physical
        w, x_x, y_x = p.oscillator_width, p.exciton_x, p.exciton_y
        sqrt2 = np.sqrt(2.0)
        denominator = np.sqrt(w**2 + (x_x - sqrt2 * w * alpha_x.real) ** 2 + (y_x - sqrt2 * w * alpha_y.real) ** 2)
        c = self.coherent_coefficient
        cc = self.oscillator_normalization
        t1 = c(n + 1, m, alpha_x, alpha_y) * (cc(n + 1, m) / cc(n, m)) * w * (n + 1)
        t2 = 1j * c(n, m + 1, alpha_x, alpha_y) * (cc(n, m + 1) / cc(n, m)) * w * (m + 1)
        t3 = c(n - 1, m, alpha_x, alpha_y) * (cc(n - 1, m) / cc(n, m)) * (w / 2.0) * (1 if n > 1 else 0)
        t4 = 1j * c(n, m - 1, alpha_x, alpha_y) * (cc(n, m - 1) / cc(n, m)) * (w / 2.0) * (1 if m > 1 else 0)
        t5 = 0.0 + 0.0j
        if n == 1:
            t5 += c(n - 1, m, alpha_x, alpha_y) * (cc(n - 1, m) / cc(n, m)) * (w / 2.0)
        if m == 1:
            t5 += 1j * c(n, m - 1, alpha_x, alpha_y) * (cc(n, m - 1) / cc(n, m)) * (w / 2.0)
        t6 = -(x_x + 1j * y_x) * c(n, m, alpha_x, alpha_y)
        return (t1 + t2 + t3 + t4 + t5 + t6) / denominator

    def bare_cavity_vortex_coefficient(self, n: int, m: int) -> complex:
        p = self.physical
        w, x_c, y_c = p.oscillator_width, p.cavity_x, p.cavity_y
        den = np.sqrt(w**2 + x_c**2 + y_c**2)
        return (
            (1 if n == 0 and m == 0 else 0) * (-(x_c + 1j * y_c)) / den
            + (1 if n == 0 and m == 1 else 0) * (1j * w * np.sqrt(2.0)) / (2 * den)
            + (1 if n == 1 and m == 0 else 0) * (w * np.sqrt(2.0)) / (2 * den)
        )

    def bare_exciton_vortex_coefficient(self, n: int, m: int) -> complex:
        p = self.physical
        w, x_x, y_x = p.oscillator_width, p.exciton_x, p.exciton_y
        den = np.sqrt(w**2 + x_x**2 + y_x**2)
        return (
            (1 if n == 0 and m == 0 else 0) * (-(x_x + 1j * y_x)) / den
            + (1 if n == 0 and m == 1 else 0) * (1j * w * np.sqrt(2.0)) / (2 * den)
            + (1 if n == 1 and m == 0 else 0) * (w * np.sqrt(2.0)) / (2 * den)
        )

    def build(self) -> np.ndarray:
        alpha = self.config.alpha
        state = np.zeros(self.basis.dimension, dtype=np.complex128)
        for na, ma in self.basis.cavity_pairs():
            for nb, mb in self.basis.exciton_pairs():
                idx = self.basis.state_index(na, ma, nb, mb)
                if self.config.use_bare_vortex_initial_state:
                    value = self.bare_cavity_vortex_coefficient(na, ma) * self.bare_exciton_vortex_coefficient(nb, mb)
                else:
                    value = self.cavity_vortex_coefficient(na, ma, alpha, alpha) * self.exciton_vortex_coefficient(nb, mb, alpha, alpha)
                state[idx] = value
        if self.config.renormalize_initial_state:
            norm = np.linalg.norm(state)
            if norm == 0:
                raise ValueError("Cannot normalize a zero initial state")
            state = state / norm
        return state
