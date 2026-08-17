"""Generate tracked reference figures for the repository README.

Most validation figures use the established nbar = 0.5 reference case. The
vortex-core trajectory figure intentionally follows the conference-poster
presentation and compares the low-excitation cases nbar = 0 and nbar = 0.005.

The figures are written to ``results/figures`` and are generated from the
package code; no numerical trajectories are hard-coded. GitHub Actions runs
this script and commits the PNG files so they render directly in the README.
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


def _trajectory_result(mean_photon_number: float):
    """Low-excitation simulation used for the poster-style trajectory panels.

    The two trajectory cases use identical model parameters and differ only in
    ``mean_photon_number``.  A cutoff of 1 is sufficient for this deliberately
    low-excitation comparison and matches the small-basis trajectory regime
    used in the exploratory notebooks.
    """
    result = simulate(
        SimulationConfig(
            cavity_cutoff=1,
            exciton_cutoff=1,
            mean_photon_number=mean_photon_number,
            t_max=20.0,
            n_times=401,
        )
    )
    if not result.solver_success:
        raise RuntimeError(result.solver_message)
    return result


def _trajectory_curves(result):
    """Return cavity/exciton curves with the original notebook sampling logic."""
    _, cavity_x, cavity_y = trajectory_curve_like_original(
        result, "cavity", duration=20.0, fps=30, radius=3.0
    )
    _, exciton_x, exciton_y = trajectory_curve_like_original(
        result, "exciton", duration=20.0, fps=30, radius=3.0
    )
    return cavity_x, cavity_y, exciton_x, exciton_y


def _style_trajectory_axis(axis, title: str):
    axis.set_xlim(-3.0, 3.0)
    axis.set_ylim(-3.0, 3.0)
    axis.set_aspect("equal")
    axis.set_xlabel("x/w")
    axis.set_ylabel("y/w")
    axis.set_title(title)


def main() -> None:
    # ------------------------------------------------------------------
    # Validated reference case used for Lz, entropy, and spatial densities.
    # ------------------------------------------------------------------
    reference = simulate(
        SimulationConfig(
            cavity_cutoff=8,
            exciton_cutoff=8,
            mean_photon_number=0.5,
            t_max=10.0,
            n_times=201,
        )
    )
    if not reference.solver_success:
        raise RuntimeError(reference.solver_message)

    times = reference.times
    lz_cavity = np.asarray([cavity_lz(reference, float(t)).real for t in times])
    lz_exciton = np.asarray([exciton_lz(reference, float(t)).real for t in times])
    lz_total = lz_cavity + lz_exciton
    entropy = np.asarray(
        [linear_entropy(cavity_density_matrix(reference, float(t))) for t in times]
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

    # ------------------------------------------------------------------
    # Poster-style vortex-core comparison: nbar = 0 versus nbar = 0.005.
    # These two calculations use the same parameters; only nbar changes.
    # With coherent_phase = pi/4, nbar=0.005 corresponds to
    # alpha_x = alpha_y = 0.05 + 0.05 i in the original coefficient notation.
    # ------------------------------------------------------------------
    trajectory_cases = [
        (0.0, _trajectory_result(0.0)),
        (0.005, _trajectory_result(0.005)),
    ]

    curves = {}
    for nbar, result in trajectory_cases:
        curves[nbar] = _trajectory_curves(result)

    # Four-panel comparison following the layout of the conference poster:
    # Exciton/Cavity for nbar=0, then Exciton/Cavity for nbar=0.005.
    figure, axes = plt.subplots(2, 2, figsize=(8.6, 8.2), sharex=True, sharey=True)
    panel_specs = [
        (axes[0, 0], curves[0.0][2], curves[0.0][3], "Exciton", "navy"),
        (axes[0, 1], curves[0.0][0], curves[0.0][1], "Cavity", "crimson"),
        (axes[1, 0], curves[0.005][2], curves[0.005][3], "Exciton", "navy"),
        (axes[1, 1], curves[0.005][0], curves[0.005][1], "Cavity", "crimson"),
    ]
    for axis, xs, ys, title, line_color in panel_specs:
        axis.plot(xs, ys, color=line_color, lw=1.35)
        last = _last_finite_point(xs, ys)
        if last is not None:
            axis.scatter([last[0]], [last[1]], s=22, color=line_color)
        _style_trajectory_axis(axis, title)

    axes[0, 0].text(
        0.5,
        -0.20,
        r"$\bar n = 0$",
        transform=axes[0, 0].transAxes,
        ha="center",
        va="top",
        fontsize=11,
    )
    axes[0, 1].text(
        0.5,
        -0.20,
        r"$\bar n = 0$",
        transform=axes[0, 1].transAxes,
        ha="center",
        va="top",
        fontsize=11,
    )
    axes[1, 0].text(
        0.5,
        -0.20,
        r"$\bar n = 0.005$",
        transform=axes[1, 0].transAxes,
        ha="center",
        va="top",
        fontsize=11,
    )
    axes[1, 1].text(
        0.5,
        -0.20,
        r"$\bar n = 0.005$",
        transform=axes[1, 1].transAxes,
        ha="center",
        va="top",
        fontsize=11,
    )
    figure.suptitle("Vortex-core trajectories in the low-excitation regime")
    figure.subplots_adjust(hspace=0.42, wspace=0.22, top=0.92, bottom=0.09)
    figure.savefig(OUTPUT_DIR / "vortex_core_trajectories.png", dpi=220)
    plt.close(figure)

    # Keep separate side-by-side figures as inspectable repository results.
    for nbar, slug in [(0.0, "nbar0"), (0.005, "nbar0005")]:
        cavity_x, cavity_y, exciton_x, exciton_y = curves[nbar]
        figure, axes = plt.subplots(1, 2, figsize=(8.4, 4.0))
        specs = [
            (axes[0], exciton_x, exciton_y, "Exciton", "navy"),
            (axes[1], cavity_x, cavity_y, "Cavity", "crimson"),
        ]
        for axis, xs, ys, title, color in specs:
            axis.plot(xs, ys, color=color, lw=1.35)
            _style_trajectory_axis(axis, title)
        figure.suptitle(rf"Vortex-core trajectories ($\bar n = {nbar:g}$)")
        figure.tight_layout()
        figure.savefig(OUTPUT_DIR / f"vortex_core_trajectories_{slug}.png", dpi=180)
        plt.close(figure)

    # Delete figures from earlier trajectory presentation variants so that the
    # tracked output directory contains only the current intended results.
    for obsolete_name in (
        "critical_points_conditioned.png",
        "vortex_core_trajectories_combined.png",
    ):
        obsolete = OUTPUT_DIR / obsolete_name
        if obsolete.exists():
            obsolete.unlink()

    coordinates = np.linspace(-60.0, 60.0, 121)
    cavity_density = cavity_spatial_density_grid(
        reference, 0.0, coordinates, coordinates
    )
    exciton_density = exciton_spatial_density_grid(
        reference, 0.0, coordinates, coordinates
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
    print(f"Reference initial norm = {reference.initial_norm:.12f}")
    print(f"Reference total Lz variation = {np.ptp(lz_total):.3e}")
    print("Trajectory comparison = nbar 0 versus 0.005")


if __name__ == "__main__":
    main()
