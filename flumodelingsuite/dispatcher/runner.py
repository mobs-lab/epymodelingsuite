"""Runner functions for executing simulations and calibrations."""

import logging

import numpy as np

from ..schema.dispatcher import BuilderOutput, CalibrationOutput, SimulationOutput

logger = logging.getLogger(__name__)


# ===== Runner Functions =====


def run_simulation(configs: BuilderOutput) -> SimulationOutput:
    """
    Run a simulation using EpiModel.run_simulations.

    Parameters
    ----------
    configs : BuilderOutput
        BuilderOutput containing model and simulation parameters.

    Returns
    -------
    SimulationOutput
        Results of the simulation with metadata.

    Raises
    ------
    RuntimeError
        If simulation fails.
    """
    logger.info("RUNNER: running simulation.")
    try:
        results = configs.model.run_simulations(**dict(configs.simulation))
        logger.info("RUNNER: completed simulation.")
        return SimulationOutput(primary_id=configs.primary_id, seed=configs.seed, results=results)
    except Exception as e:
        raise RuntimeError(f"Error during simulation: {e}")


def run_calibration(configs: BuilderOutput) -> CalibrationOutput:
    """
    Run a calibration using ABCSampler.calibrate.

    Parameters
    ----------
    configs : BuilderOutput
        BuilderOutput containing calibrator and calibration parameters.

    Returns
    -------
    CalibrationOutput
        Results of the calibration with metadata.

    Raises
    ------
    RuntimeError
        If calibration fails.
    """
    logger.info("RUNNER: running calibration.")
    try:
        results = configs.calibrator.calibrate(strategy=configs.calibration.name, **configs.calibration.options)
        logger.info("RUNNER: completed calibration.")
        return CalibrationOutput(primary_id=configs.primary_id, seed=configs.seed, results=results)
    except Exception as e:
        raise RuntimeError(f"Error during calibration: {e}")


def run_calibration_with_projection(configs: BuilderOutput) -> CalibrationOutput:
    """
    Run a calibration followed by projection using ABCSampler.

    Parameters
    ----------
    configs : BuilderOutput
        BuilderOutput containing calibrator, calibration, and projection parameters.

    Returns
    -------
    CalibrationOutput
        Results of the projection with metadata (includes calibration results).

    Raises
    ------
    RuntimeError
        If calibration or projection fails.
    """
    logger.info("RUNNER: running calibration and projection.")
    try:
        calibration_results = configs.calibrator.calibrate(
            strategy=configs.calibration.name, **configs.calibration.options
        )
        projection_results = configs.calibrator.run_projections(
            parameters={
                "projection": True,
                "end_date": configs.projection.end_date,
                "generation": configs.projection.generation_number,
                "epimodel": configs.model,
            },
            iterations=configs.projection.n_trajectories,
        )
        logger.info("RUNNER: completed calibration and projection.")
        return CalibrationOutput(primary_id=configs.primary_id, seed=configs.seed, results=projection_results)
    except Exception as e:
        raise RuntimeError(f"Error during calibration/projection: {e}")


def dispatch_runner(configs: BuilderOutput) -> SimulationOutput | CalibrationOutput:
    """
    Dispatch simulation/calibration/projection using a BuilderOutput and return the results.

    Parameters
    ----------
        configs: a single BuilderOutput created by dispatch_builder()

    Returns
    -------
        An object containing metadata and results of simulation/calibration/projection.

    Raises
    ------
        RuntimeError if simulation/calibration/projection fails.
        AssertionError if configs are invalid.
    """
    np.random.seed(configs.seed)

    # Handle simulation
    if configs.simulation:
        logger.info("RUNNER: dispatched for simulation.")
        return run_simulation(configs)

    # Handle calibration
    elif configs.calibration and not configs.projection:
        logger.info("RUNNER: dispatched for calibration.")
        return run_calibration(configs)

    # Handle calibration and projection
    elif configs.calibration and configs.projection:
        logger.info("RUNNER: dispatched for calibration and projection.")
        return run_calibration_with_projection(configs)

    # Error
    else:
        raise AssertionError(
            "Runner called without simulation or calibration specs. Verify that your BuilderOutputs are valid."
        )
