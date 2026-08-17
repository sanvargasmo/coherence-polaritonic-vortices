"""Numerical tools for the polaritonic-vortex research project."""

from .basis import ProductBasis, in_triangle, triangular_count, triangular_index
from .dynamics import SimulationResult, build_coupling_matrices, simulate
from .parameters import PhysicalParameters, SimulationConfig
from .reduced_density import (
    cavity_density_matrix,
    density_trace,
    exciton_density_matrix,
    linear_entropy,
    purity,
)
from .observables import cavity_lz, exciton_lz, total_lz
from .trajectories import (
    cavity_trajectory_point,
    exciton_trajectory_point,
    trajectory_curve_like_original,
)
from .spatial_fields import (
    cavity_spatial_density,
    exciton_spatial_density,
    joint_wavefunction_same_coordinate,
    cavity_transition_field_x,
    cavity_transition_field_y,
    exciton_transition_field_x,
    exciton_transition_field_y,
    cavity_spatial_density_grid,
    exciton_spatial_density_grid,
    joint_wavefunction_grid_same_coordinate,
    transition_field_grid,
)

__all__ = [
    "PhysicalParameters",
    "SimulationConfig",
    "ProductBasis",
    "SimulationResult",
    "simulate",
    "build_coupling_matrices",
    "triangular_count",
    "triangular_index",
    "in_triangle",
    "cavity_density_matrix",
    "exciton_density_matrix",
    "density_trace",
    "purity",
    "linear_entropy",
    "cavity_lz",
    "exciton_lz",
    "total_lz",
    "cavity_trajectory_point",
    "exciton_trajectory_point",
    "trajectory_curve_like_original",
    "cavity_spatial_density",
    "exciton_spatial_density",
    "joint_wavefunction_same_coordinate",
    "cavity_transition_field_x",
    "cavity_transition_field_y",
    "exciton_transition_field_x",
    "exciton_transition_field_y",
    "cavity_spatial_density_grid",
    "exciton_spatial_density_grid",
    "joint_wavefunction_grid_same_coordinate",
    "transition_field_grid",
]
