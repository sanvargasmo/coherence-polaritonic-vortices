from __future__ import annotations

from dataclasses import dataclass


def triangular_count(cutoff: int) -> int:
    if cutoff < 0:
        raise ValueError("cutoff must be non-negative")
    return (cutoff + 1) * (cutoff + 2) // 2


def in_triangle(cutoff: int, n: int, m: int) -> bool:
    return n >= 0 and m >= 0 and n + m <= cutoff


def triangular_index(cutoff: int, n: int, m: int) -> int:
    """Row-by-row index for pairs (n,m) satisfying n+m <= cutoff."""
    if not in_triangle(cutoff, n, m):
        raise IndexError(f"({n}, {m}) is outside cutoff {cutoff}")
    return (n * (2 * cutoff + 3 - n)) // 2 + m


@dataclass(frozen=True)
class ProductBasis:
    """Triangular 2D oscillator basis for cavity and exciton subsystems."""

    cavity_cutoff: int
    exciton_cutoff: int

    @property
    def n_cavity(self) -> int:
        return triangular_count(self.cavity_cutoff)

    @property
    def n_exciton(self) -> int:
        return triangular_count(self.exciton_cutoff)

    @property
    def dimension(self) -> int:
        return self.n_cavity * self.n_exciton

    def cavity_index(self, n: int, m: int) -> int:
        return triangular_index(self.cavity_cutoff, n, m)

    def exciton_index(self, n: int, m: int) -> int:
        return triangular_index(self.exciton_cutoff, n, m)

    def state_index(self, na: int, ma: int, nb: int, mb: int) -> int:
        return self.cavity_index(na, ma) * self.n_exciton + self.exciton_index(nb, mb)

    def cavity_pairs(self):
        for n in range(self.cavity_cutoff + 1):
            for m in range(self.cavity_cutoff - n + 1):
                yield n, m

    def exciton_pairs(self):
        for n in range(self.exciton_cutoff + 1):
            for m in range(self.exciton_cutoff - n + 1):
                yield n, m
