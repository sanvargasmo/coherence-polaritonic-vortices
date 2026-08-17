"""Generate reference figures for the validated nbar = 0.5 case.

The figures are written to ``results/figures`` and are intended to be shown
in the repository README. The simulation uses the validated package code; no
results are hard-coded.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from polaritonic_vortices import SimulationConfig, simulate
from polaritonic_vortices.observables import cavity_lz, exciton_lz
from polaritonic_vortices.reduced_density import cavity_density_matrix, linear_entropy
from polaritonic_vortices.spatial_fields import (
    cavity_spatial_density_grid,
    exciton_spatial_density_grid,
)
from polaritonic_vortices.trajectories import (
    quadratic_density_coefficients,
    critical_point_from_quadratic,
)
from polaritonic_vortices.reduced_density import exciton_density_matrix


OUTPUT_DIR = Path("results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def conditioned_critical_points(result, subsystem: str):
    """Return well-conditioned critical-point locations for visualization.

    The quadratic critical-point formula becomes numerically unstable when
    ``|4*d*e - g**2|`` is very small. For the presentation figure only, the
    lowest 15% of determinant magnitudes and the largest 10% of radii are
    omitted. The underlying unfiltered trajectory functions remain available
    in the package and are used by the regression tests.
    """
    if subsystem == "cavity":
        density = cavity_density_matrix
        cutoff = result.basis.cavity_cutoff
        index = result.basis.cavity_index
    elif subsystem == "exciton":
        density = exciton_density_matrix
        cutoff = result.basis.exciton_cutoff
        index = result.basis.exciton_index
    else:
        raise ValueError("subsystem must be 'cavity' or 'exciton'")

    width = result.physical.oscillator_width
    xs, ys, determinants = [], [], []

    for time in result.times:
        coeffs = quadratic_density_coefficients(
            density(result, float(time)), cutoff, index, width
        )
        determinant = 4.0 * coeffs["d"] * coeffs["e"] - coeffs["g"] ** 2
        x, y = critical_point_from_quadratic(coeffs)
        xs.append(float(np.real(x) / width))
        ys.append(float(np.real(y) / width))
        determinants.append(abs(determinant))

    xs = np.asarray(xs)
    ys = np.asarray(ys)
    determinants = np.asarray(determinants)
    radii = np.hypot(xs, ys)

    valid = (
        determinants >= np.quantile(determinants, 0.15)
    ) & (
        radii <= np.quantile(radii, 0.90)
    )
    return xs[valid], ys[valid], result.times[valid]


def main() -> None:
    config = SimulationConfig(
        cavity_cutoff=8,
        exciton_cutoff=8,
        mean_photon_number=0.5,
        t_max=10.0,
        n_times=201,
    )
    result = simulate(config)
    if not result.solver_success:
        raise RuntimeError(result.solver_message)

    times = result.times
    lz_cavity = np.asarray([cavity_lz(result, float(t)).real for t in times])
    lz_exciton = np.asarray([exciton_lz(result, float(t)).real for t in times])
    lz_total = lz_cavity + lz_exciton
    entropy = np.asarray(
        [linear_entropy(cavity_density_matrix(result, float(t))) for t in times]
    )

    plt.figure(figsize=(8, 5))
    plt.plot(times, lz_cavity, label="Cavity")
    plt.plot(times, lz_exciton, label="Exciton")
    plt.plot(times, lz_total, label="Total")
    plt.xlabel("Time")
    plt.ylabel("Lz")
    plt.title("Angular momentum vs time (n̄ = 0.5)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "lz_vs_time.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(times, entropy)
    plt.xlabel("Time")
    plt.ylabel("Linear entropy")
    plt.title("Linear entropy vs time (n̄ = 0.5)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "linear_entropy_vs_time.png", dpi=180)
    plt.close()

    x_c, y_c, t_c = conditioned_critical_points(result, "cavity")
    x_x, y_x, t_x = conditioned_critical_points(result, "exciton")
    plt.figure(figsize=(7, 6))
    plt.scatter(x_c, y_c, c=t_c, cmap="Blues", s=28, alpha=0.85, label="Cavity")
    plt.scatter(x_x, y_x, c=t_x, cmap="Oranges", s=28, alpha=0.85, marker="s", label="Exciton")
    plt.xlabel("x / w")
    plt.ylabel("y / w")
    plt.title("Conditioned critical-point locations (n̄ = 0.5)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "critical_points_conditioned.png", dpi=180)
    plt.close()

    coordinates = np.linspace(-60.0, 60.0, 121)
    cavity_density = cavity_spatial_density_grid(
        result, 0.0, coordinates, coordinates
    )
    exciton_density = exciton_spatial_density_grid(
        result, 0.0, coordinates, coordinates
    )

    plt.figure(figsize=(6, 5))
    image = plt.imshow(
        cavity_density,
        extent=[coordinates[0], coordinates[-1], coordinates[0], coordinates[-1]],
        origin="lower",
        aspect="equal",
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Cavity spatial density at t = 0 (n̄ = 0.5)")
    plt.colorbar(image)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cavity_density_t0.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 5))
    image = plt.imshow(
        exciton_density,
        extent=[coordinates[0], coordinates[-1], coordinates[0], coordinates[-1]],
        origin="lower",
        aspect="equal",
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Exciton spatial density at t = 0 (n̄ = 0.5)")
    plt.colorbar(image)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "exciton_density_t0.png", dpi=180)
    plt.close()

    print(f"Generated figures in {OUTPUT_DIR}")
    print(f"Initial norm = {result.initial_norm:.12f}")
    print(f"Total Lz variation = {np.ptp(lz_total):.3e}")


if __name__ == "__main__":
    main()
