"""Generate reference figures for the validated nbar = 0.5 case.

The figures are written to ``results/figures`` and are intended to be shown
in the repository README. The simulation uses the validated package code; no
results are hard-coded.

This script is also executed automatically by GitHub Actions. The generated
PNG files are intentionally tracked so they render directly in the README.
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
from polaritonic_vortices.trajectories import trajectory_curve_like_original


OUTPUT_DIR = Path("results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _last_finite_point(xs: np.ndarray, ys: np.ndarray):
    mask = np.isfinite(xs) & np.isfinite(ys)
    if not np.any(mask):
        return None
    return xs[mask][-1], ys[mask][-1]


def main() -> None:
    # The original trajectory plots use a duration of 20 at 30 fps. The
    # simulation therefore covers that full interval, while retaining the
    # validated nbar=0.5 and cutoff=8 reference setup.
    config = SimulationConfig(
        cavity_cutoff=8,
        exciton_cutoff=8,
        mean_photon_number=0.5,
        t_max=20.0,
        n_times=401,
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

    # Reproduce the trajectory plotting logic used in the original notebooks:
    # 30 fps, duration=20, radius R=3, NaNs to break the line outside R.
    _, cavity_x, cavity_y = trajectory_curve_like_original(
        result, "cavity", duration=20.0, fps=30, radius=3.0
    )
    _, exciton_x, exciton_y = trajectory_curve_like_original(
        result, "exciton", duration=20.0, fps=30, radius=3.0
    )

    figure, axes = plt.subplots(1, 2, figsize=(8.4, 4.0))
    trajectory_specs = [
        (axes[0], cavity_x, cavity_y, "red", "darkred", "Cavity vortex core"),
        (axes[1], exciton_x, exciton_y, "blue", "navy", "Exciton vortex core"),
    ]
    for axis, xs, ys, line_color, point_color, title in trajectory_specs:
        axis.plot(xs, ys, color=line_color, lw=1.5)
        last = _last_finite_point(xs, ys)
        if last is not None:
            axis.scatter([last[0]], [last[1]], s=40, color=point_color)
        axis.set_xlim(-3.0, 3.0)
        axis.set_ylim(-3.0, 3.0)
        axis.set_aspect("equal")
        axis.set_xlabel("x/w")
        axis.set_ylabel("y/w")
        axis.set_title(title)
    figure.suptitle("Vortex-core trajectories (n̄ = 0.5)")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "vortex_core_trajectories.png", dpi=180)
    plt.close(figure)

    # A combined view is useful for direct cavity/exciton comparison while
    # preserving exactly the same sampled points as the original-style plots.
    plt.figure(figsize=(6.5, 6.0))
    plt.plot(cavity_x, cavity_y, color="red", lw=1.4, label="Cavity")
    plt.plot(exciton_x, exciton_y, color="blue", lw=1.4, label="Exciton")
    plt.xlim(-3.0, 3.0)
    plt.ylim(-3.0, 3.0)
    plt.gca().set_aspect("equal")
    plt.xlabel("x/w")
    plt.ylabel("y/w")
    plt.title("Cavity and exciton vortex-core trajectories (n̄ = 0.5)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "vortex_core_trajectories_combined.png", dpi=180)
    plt.close()

    # Remove the earlier presentation-only figure based on percentile
    # conditioning. It did not reproduce the visualization used in the source
    # notebooks and should not remain as a primary repository result.
    obsolete = OUTPUT_DIR / "critical_points_conditioned.png"
    if obsolete.exists():
        obsolete.unlink()

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
