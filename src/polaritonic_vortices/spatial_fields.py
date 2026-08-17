from __future__ import annotations

from math import factorial
import numpy as np
from scipy.special import eval_hermite

from .reduced_density import cavity_density_matrix, exciton_density_matrix


def gaussian_factor(x: float, y: float, width: float) -> float:
    """Common Gaussian factor exp[-(x^2+y^2)/w^2] used in density kernels."""
    return float(np.exp(-(x * x + y * y) / (width * width)))


def oscillator_normalization(n: int, m: int, width: float) -> float:
    """Normalization of the 2D Cartesian harmonic-oscillator basis."""
    if n < 0 or m < 0:
        raise ValueError("Oscillator indices must be non-negative")
    return 1.0 / (
        width
        * np.sqrt(np.pi * (2.0 ** (n + m)) * factorial(n) * factorial(m))
    )


def oscillator_basis_value(
    n: int,
    m: int,
    x: float,
    y: float,
    width: float,
) -> float:
    """Real-space oscillator wavefunction phi_nm(x,y).

    This is the normalized basis used implicitly in the original notebooks:

        phi_nm = c_nm H_n(x/w) H_m(y/w)
                 exp[-(x^2+y^2)/(2 w^2)].
    """
    u = x / width
    v = y / width
    return float(
        oscillator_normalization(n, m, width)
        * eval_hermite(n, u)
        * eval_hermite(m, v)
        * np.exp(-0.5 * (u * u + v * v))
    )


def _basis_vector(pairs, x: float, y: float, width: float) -> np.ndarray:
    u = x / width
    v = y / width
    gaussian_half = np.exp(-0.5 * (u * u + v * v))
    return np.asarray(
        [
            oscillator_normalization(n, m, width)
            * eval_hermite(n, u)
            * eval_hermite(m, v)
            * gaussian_half
            for n, m in pairs
        ],
        dtype=float,
    )


def cavity_basis_vector(result, x: float, y: float) -> np.ndarray:
    return _basis_vector(
        list(result.basis.cavity_pairs()), x, y, result.physical.oscillator_width
    )


def exciton_basis_vector(result, x: float, y: float) -> np.ndarray:
    return _basis_vector(
        list(result.basis.exciton_pairs()), x, y, result.physical.oscillator_width
    )


def spatial_density_from_reduced_matrix(
    rho: np.ndarray,
    basis_vector: np.ndarray,
) -> float:
    """Evaluate rho(r,r) = sum_ij rho_ij phi_i(r) phi_j*(r)."""
    phi = np.asarray(basis_vector, dtype=np.complex128)
    matrix = np.asarray(rho, dtype=np.complex128)
    value = phi @ matrix @ phi.conj()
    return float(np.real_if_close(value, tol=1000).real)


def cavity_spatial_density(result, x: float, y: float, time: float) -> float:
    rho = cavity_density_matrix(result, time)
    phi = cavity_basis_vector(result, x, y)
    return spatial_density_from_reduced_matrix(rho, phi)


def exciton_spatial_density(result, x: float, y: float, time: float) -> float:
    rho = exciton_density_matrix(result, time)
    phi = exciton_basis_vector(result, x, y)
    return spatial_density_from_reduced_matrix(rho, phi)


def joint_wavefunction_same_coordinate(
    result,
    x: float,
    y: float,
    time: float,
) -> complex:
    """Reproduce the original PsiTTG(x,y,t) construction.

    The cavity and exciton coordinates are both evaluated at the same spatial
    point (x,y), exactly as in ``Copia de small.ipynb``. This is kept as a
    distinct quantity from the reduced cavity/exciton spatial densities.
    """
    cavity_phi = cavity_basis_vector(result, x, y)
    exciton_phi = exciton_basis_vector(result, x, y)
    coefficients = result.coefficient_matrix_at(time)
    return complex(cavity_phi @ coefficients @ exciton_phi)


def _raised_basis_vector(
    pairs,
    x: float,
    y: float,
    width: float,
    cutoff: int,
    axis: str,
) -> np.ndarray:
    """Vector multiplying the second density-matrix index in aTx/aTy/bTx/bTy.

    The original notebook sums terms such as

        sqrt(n') rho[(n,m),(n'-1,m')] phi_nm phi_n'm'.

    Therefore, for a reduced-basis index k=(n,m), the second factor is
    sqrt(n+1) phi_(n+1,m), not the action of a truncated annihilation matrix.
    The raised oscillator pair may lie just outside the triangular many-body
    cutoff while each Cartesian index remains within the notebook's loop
    bounds. Preserving that detail is necessary to reproduce the original
    spatial transition fields exactly.
    """
    if axis not in {"x", "y"}:
        raise ValueError("axis must be 'x' or 'y'")

    values = []
    for n, m in pairs:
        if axis == "x":
            if n >= cutoff:
                values.append(0.0)
            else:
                values.append(
                    np.sqrt(n + 1)
                    * oscillator_basis_value(n + 1, m, x, y, width)
                )
        else:
            if m >= cutoff:
                values.append(0.0)
            else:
                values.append(
                    np.sqrt(m + 1)
                    * oscillator_basis_value(n, m + 1, x, y, width)
                )
    return np.asarray(values, dtype=float)


def transition_field_from_reduced_matrix(
    rho: np.ndarray,
    basis_vector: np.ndarray,
    raised_basis_vector: np.ndarray,
) -> complex:
    """Evaluate the complex kernel used as aTx/aTy/bTx/bTy in the notebooks."""
    phi = np.asarray(basis_vector, dtype=np.complex128)
    raised = np.asarray(raised_basis_vector, dtype=np.complex128)
    matrix = np.asarray(rho, dtype=np.complex128)
    return complex(phi @ matrix @ raised)


def cavity_transition_field(
    result,
    x: float,
    y: float,
    time: float,
    axis: str,
) -> complex:
    rho = cavity_density_matrix(result, time)
    pairs = list(result.basis.cavity_pairs())
    width = result.physical.oscillator_width
    phi = _basis_vector(pairs, x, y, width)
    raised = _raised_basis_vector(
        pairs, x, y, width, result.basis.cavity_cutoff, axis
    )
    return transition_field_from_reduced_matrix(rho, phi, raised)


def exciton_transition_field(
    result,
    x: float,
    y: float,
    time: float,
    axis: str,
) -> complex:
    rho = exciton_density_matrix(result, time)
    pairs = list(result.basis.exciton_pairs())
    width = result.physical.oscillator_width
    phi = _basis_vector(pairs, x, y, width)
    raised = _raised_basis_vector(
        pairs, x, y, width, result.basis.exciton_cutoff, axis
    )
    return transition_field_from_reduced_matrix(rho, phi, raised)


def cavity_transition_field_x(result, x: float, y: float, time: float) -> complex:
    """Clean equivalent of the original ``aTx`` function."""
    return cavity_transition_field(result, x, y, time, axis="x")


def cavity_transition_field_y(result, x: float, y: float, time: float) -> complex:
    """Clean equivalent of the original ``aTy`` function."""
    return cavity_transition_field(result, x, y, time, axis="y")


def exciton_transition_field_x(result, x: float, y: float, time: float) -> complex:
    """Clean equivalent of the original ``bTx`` function."""
    return exciton_transition_field(result, x, y, time, axis="x")


def exciton_transition_field_y(result, x: float, y: float, time: float) -> complex:
    """Clean equivalent of the original ``bTy`` function."""
    return exciton_transition_field(result, x, y, time, axis="y")


def evaluate_field_grid(
    field,
    result,
    time: float,
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> np.ndarray:
    """Evaluate any scalar/complex field on a Cartesian grid."""
    xs = np.asarray(x_values, dtype=float)
    ys = np.asarray(y_values, dtype=float)
    out = np.empty((ys.size, xs.size), dtype=np.complex128)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            out[j, i] = field(result, float(x), float(y), float(time))
    return out


def _basis_grid(pairs, x_values: np.ndarray, y_values: np.ndarray, width: float) -> np.ndarray:
    """Vectorized basis values with shape (ny, nx, n_basis)."""
    xs = np.asarray(x_values, dtype=float)
    ys = np.asarray(y_values, dtype=float)
    u = xs / width
    v = ys / width
    gaussian_half = np.exp(-0.5 * (v[:, None] ** 2 + u[None, :] ** 2))
    grid = np.empty((ys.size, xs.size, len(pairs)), dtype=float)
    for k, (n, m) in enumerate(pairs):
        grid[:, :, k] = (
            oscillator_normalization(n, m, width)
            * eval_hermite(m, v)[:, None]
            * eval_hermite(n, u)[None, :]
            * gaussian_half
        )
    return grid


def _raised_basis_grid(
    pairs,
    x_values: np.ndarray,
    y_values: np.ndarray,
    width: float,
    cutoff: int,
    axis: str,
) -> np.ndarray:
    xs = np.asarray(x_values, dtype=float)
    ys = np.asarray(y_values, dtype=float)
    u = xs / width
    v = ys / width
    gaussian_half = np.exp(-0.5 * (v[:, None] ** 2 + u[None, :] ** 2))
    grid = np.zeros((ys.size, xs.size, len(pairs)), dtype=float)

    for k, (n, m) in enumerate(pairs):
        if axis == "x" and n < cutoff:
            grid[:, :, k] = (
                np.sqrt(n + 1)
                * oscillator_normalization(n + 1, m, width)
                * eval_hermite(m, v)[:, None]
                * eval_hermite(n + 1, u)[None, :]
                * gaussian_half
            )
        elif axis == "y" and m < cutoff:
            grid[:, :, k] = (
                np.sqrt(m + 1)
                * oscillator_normalization(n, m + 1, width)
                * eval_hermite(m + 1, v)[:, None]
                * eval_hermite(n, u)[None, :]
                * gaussian_half
            )
    return grid


def cavity_spatial_density_grid(
    result,
    time: float,
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> np.ndarray:
    rho = cavity_density_matrix(result, time)
    pairs = list(result.basis.cavity_pairs())
    phi = _basis_grid(pairs, x_values, y_values, result.physical.oscillator_width)
    values = np.einsum("...i,ij,...j->...", phi, rho, phi.conj(), optimize=True)
    return np.real_if_close(values, tol=1000).real


def exciton_spatial_density_grid(
    result,
    time: float,
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> np.ndarray:
    rho = exciton_density_matrix(result, time)
    pairs = list(result.basis.exciton_pairs())
    phi = _basis_grid(pairs, x_values, y_values, result.physical.oscillator_width)
    values = np.einsum("...i,ij,...j->...", phi, rho, phi.conj(), optimize=True)
    return np.real_if_close(values, tol=1000).real


def joint_wavefunction_grid_same_coordinate(
    result,
    time: float,
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> np.ndarray:
    width = result.physical.oscillator_width
    cavity_phi = _basis_grid(list(result.basis.cavity_pairs()), x_values, y_values, width)
    exciton_phi = _basis_grid(list(result.basis.exciton_pairs()), x_values, y_values, width)
    coefficients = result.coefficient_matrix_at(time)
    return np.einsum(
        "...i,ij,...j->...", cavity_phi, coefficients, exciton_phi, optimize=True
    )


def transition_field_grid(
    result,
    time: float,
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    subsystem: str,
    axis: str,
) -> np.ndarray:
    if subsystem == "cavity":
        rho = cavity_density_matrix(result, time)
        pairs = list(result.basis.cavity_pairs())
        cutoff = result.basis.cavity_cutoff
    elif subsystem == "exciton":
        rho = exciton_density_matrix(result, time)
        pairs = list(result.basis.exciton_pairs())
        cutoff = result.basis.exciton_cutoff
    else:
        raise ValueError("subsystem must be 'cavity' or 'exciton'")

    width = result.physical.oscillator_width
    phi = _basis_grid(pairs, x_values, y_values, width)
    raised = _raised_basis_grid(pairs, x_values, y_values, width, cutoff, axis)
    return np.einsum("...i,ij,...j->...", phi, rho, raised, optimize=True)
