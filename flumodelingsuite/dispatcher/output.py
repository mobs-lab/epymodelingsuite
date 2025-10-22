"""Output generation functions for formatting and saving results."""

import logging

from ..schema.dispatcher import CalibrationOutput, SimulationOutput
from ..schema.output import OutputConfig

logger = logging.getLogger(__name__)


# ===== Output Generator Registry and Functions =====


OUTPUT_GENERATOR_REGISTRY = {}


def register_output_generator(kind_set):
    """Decorator for output generation dispatch."""

    def deco(fn):
        OUTPUT_GENERATOR_REGISTRY[frozenset(kind_set)] = fn
        return fn

    return deco


@register_output_generator({"simulation", "outputs"})
def generate_simulation_outputs(*, simulation: list[SimulationOutput], outputs: OutputConfig, **_) -> None:
    """
    Generate and save outputs from simulation results.

    Parameters
    ----------
    simulation : list[SimulationOutput]
        List of simulation results with metadata.
    outputs : OutputConfig
        Configuration for output generation (formats, paths, etc.).

    Notes
    -----
    This function is called for its side effects (writing files).
    """
    logger.info("OUTPUT GENERATOR: dispatched for simulation")


@register_output_generator({"calibration", "outputs"})
def generate_calibration_outputs(*, calibration: list[CalibrationOutput], outputs: OutputConfig, **_) -> None:
    """
    Generate and save outputs from calibration results.

    Parameters
    ----------
    calibration : list[CalibrationOutput]
        List of calibration results with metadata.
    outputs : OutputConfig
        Configuration for output generation (formats, paths, etc.).

    Notes
    -----
    This function is called for its side effects (writing files).
    """
    logger.info("OUTPUT GENERATOR: dispatched for calibration")


def dispatch_output_generator(**configs) -> None:
    """Dispatch output generator functions. Write outputs to file, called for effect."""
    kinds = frozenset(k for k, v in configs.items() if v is not None)
    return OUTPUT_GENERATOR_REGISTRY[kinds](**configs)
