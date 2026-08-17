from __future__ import annotations

from math import factorial
import numpy as np

from .reduced_density import cavity_density_matrix, exciton_density_matrix


def hermite_values_at_zero(cutoff: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return H_n(0), H'_n(0), and H''_n(0) for physicists' Hermite polynomials."""
    h0 = np.zeros(cutoff + 1, dtype=float)
    h1 = np.zeros(cutoff + 1, dtype=float)
    h2 = np.zeros(cutoff + 1, dtype=float)

    for n in range(cutoff + 1):
        if n % 2 == 0:
            k = n // 2
            h0[n] = ((-1.0) ** k) * factorial(2 * k) / factorial(k)
            if n >= 2:
                h2[n] = 4.0 * n * (n - 1) * h0[n - 2]
        else:
            h1[n] = 2.0 * n * h0[n - 1]

    return h0, h1, h2


def oscillator_normalization_table(cutoff: int, width: float) -> np.ndarray:
    """Normalization c_{nm} used by the oscillator basis in the original notebooks."""
    table = np.zeros((cutoff + 1, cutoff + 1), dtype=float)
    for n in range(cutoff + 1):
        for m in range(cutoff - n + 1):
            table[n, m] = 1.0 / (
                width
                * np.sqrt(
                    np.pi
                    * (2.0 ** (n + m))
                    * factorial(n)
                    * factorial(m)
                )
            )
    return table


def quadratic_density_coefficients(
    rho: np.ndarray,
    cutoff: int,
    basis_index,
    width: float,
) -> dict[str, complex]:
    """Quadratic Taylor coefficients used by the original trajectory method.

    The notebooks remove the common Gaussian factor and expand

        F(x,y) = exp((x^2+y^2)/w^2) * rho(x,y)

    about the origin as

        F ~= a + b*x + c*y + d*x^2 + e*y^2 + g*x*y.

    This function evaluates those coefficients analytically from the reduced
    density matrix in the triangular oscillator basis.
    """
    matrix = np.asarray(rho, dtype=np.complex128)
    h0, h1, h2 = hermite_values_at_zero(cutoff)
    cc = oscillator_normalization_table(cutoff, width)
    inv_w = 1.0 / width
    inv_w2 = inv_w * inv_w

    a = b = c = d = e = g = 0.0 + 0.0j
    pairs = [
        (n, m)
        for n in range(cutoff + 1)
        for m in range(cutoff - n + 1)
    ]

    for n, m in pairs:
        i = basis_index(n, m)
        for np_, mp in pairs:
            j = basis_index(np_, mp)
            r = matrix[i, j]
            if r == 0:
                continue

            prefactor = cc[n, m] * cc[np_, mp] * r

            a += prefactor * (h0[n] * h0[np_]) * (h0[m] * h0[mp])
            b += (
                prefactor
                * inv_w
                * (h1[n] * h0[np_] + h0[n] * h1[np_])
                * (h0[m] * h0[mp])
            )
            c += (
                prefactor
                * inv_w
                * (h0[n] * h0[np_])
                * (h1[m] * h0[mp] + h0[m] * h1[mp])
            )
            d += (
                prefactor
                * (0.5 * inv_w2)
                * (h2[n] * h0[np_] + 2.0 * h1[n] * h1[np_] + h0[n] * h2[np_])
                * (h0[m] * h0[mp])
            )
            e += (
                prefactor
                * (0.5 * inv_w2)
                * (h0[n] * h0[np_])
                * (h2[m] * h0[mp] + 2.0 * h1[m] * h1[mp] + h0[m] * h2[mp])
            )
            g += (
                prefactor
                * inv_w2
                * (h1[n] * h0[np_] + h0[n] * h1[np_])
                * (h1[m] * h0[mp] + h0[m] * h1[mp])
            )

    return {"a": a, "b": b, "c": c, "d": d, "e": e, "g": g}


def critical_point_from_quadratic(
    coefficients: dict[str, complex],
) -> tuple[complex, complex]:
    """Stationary point of a+b*x+c*y+d*x^2+e*y^2+g*x*y."""
    b = coefficients["b"]
    c = coefficients["c"]
    d = coefficients["d"]
    e = coefficients["e"]
    g = coefficients["g"]

    determinant = 4.0 * d * e - g * g
    if abs(determinant) < np.finfo(float).eps:
        nan = np.nan + 0.0j
        return nan, nan

    x = (c * g - 2.0 * b * e) / determinant
    y = (b * g - 2.0 * c * d) / determinant
    return x, y


def cavity_trajectory_point(
    result,
    time: float,
    *,
    dimensionless: bool = True,
) -> tuple[float, float]:
    """Cavity vortex-core point from the original quadratic-density method."""
    rho = cavity_density_matrix(result, time)
    coeffs = quadratic_density_coefficients(
        rho,
        result.basis.cavity_cutoff,
        result.basis.cavity_index,
        result.physical.oscillator_width,
    )
    x, y = critical_point_from_quadratic(coeffs)
    scale = result.physical.oscillator_width if dimensionless else 1.0
    return float(np.real(x) / scale), float(np.real(y) / scale)


def exciton_trajectory_point(
    result,
    time: float,
    *,
    dimensionless: bool = True,
) -> tuple[float, float]:
    """Exciton vortex-core point from the original quadratic-density method."""
    rho = exciton_density_matrix(result, time)
    coeffs = quadratic_density_coefficients(
        rho,
        result.basis.exciton_cutoff,
        result.basis.exciton_index,
        result.physical.oscillator_width,
    )
    x, y = critical_point_from_quadratic(coeffs)
    scale = result.physical.oscillator_width if dimensionless else 1.0
    return float(np.real(x) / scale), float(np.real(y) / scale)


def trajectory_curve_like_original(
    result,
    subsystem: str,
    duration: float = 20.0,
    fps: int = 30,
    radius: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a vortex-core trajectory exactly like the notebook plots.

    The research notebooks use ``n_frames = int(fps * duration)`` and
    ``np.linspace(0, duration, n_frames, endpoint=False)``. A point is kept
    only when it is finite and lies inside the circle ``x^2+y^2 <= radius^2``
    in dimensionless x/w, y/w coordinates. When a point leaves that circle,
    NaNs are inserted so Matplotlib breaks the line, matching the static plot
    and animation logic in the original notebooks.
    """
    if duration <= 0:
        raise ValueError("duration must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if radius <= 0:
        raise ValueError("radius must be positive")
    if duration > result.times[-1]:
        raise ValueError("duration exceeds the simulated time interval")

    if subsystem == "cavity":
        point_function = cavity_trajectory_point
    elif subsystem == "exciton":
        point_function = exciton_trajectory_point
    else:
        raise ValueError("subsystem must be 'cavity' or 'exciton'")

    n_frames = int(fps * duration)
    sample_times = np.linspace(0.0, duration, n_frames, endpoint=False)
    xs: list[float] = []
    ys: list[float] = []
    r2 = radius * radius

    for time in sample_times:
        x, y = point_function(result, float(time), dimensionless=True)
        inside = np.isfinite(x) and np.isfinite(y) and (x * x + y * y <= r2)
        if inside:
            xs.append(x)
            ys.append(y)
        elif xs and ys:
            xs.append(np.nan)
            ys.append(np.nan)

    return sample_times, np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
