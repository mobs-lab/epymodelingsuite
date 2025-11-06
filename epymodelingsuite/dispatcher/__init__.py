"""Dispatcher module for building, running, and generating outputs from epidemic models."""

from .builder import (
    build_basemodel,
    build_calibration,
    build_sampling,
    calculate_compartment_initial_conditions,
    dispatch_builder,
    dist_func_dict,
    register_builder,
)
from .output import (
    dispatch_output_generator,
    generate_calibration_outputs,
    generate_simulation_outputs,
    register_output_generator,
)
from .runner import (
    dispatch_runner,
    run_calibration,
    run_calibration_with_projection,
    run_simulation,
)

__all__ = [
    # Builder functions and utilities
    "build_basemodel",
    "build_calibration",
    "build_sampling",
    "calculate_compartment_initial_conditions",
    "dispatch_builder",
    "dist_func_dict",
    "register_builder",
    # Runner functions
    "dispatch_runner",
    "run_calibration",
    "run_calibration_with_projection",
    "run_simulation",
    # Output functions
    "dispatch_output_generator",
    "generate_calibration_outputs",
    "generate_simulation_outputs",
    "register_output_generator",
]
