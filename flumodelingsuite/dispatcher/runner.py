"""Runner functions for executing simulations and calibrations."""

import logging
import time

from ..schema.dispatcher import BuilderOutput, CalibrationOutput, SimulationOutput
from ..telemetry import ExecutionTelemetry

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
    start_time = time.time()

    try:
        results = configs.model.run_simulations(**dict(configs.simulation))
        duration = time.time() - start_time
        logger.info("RUNNER: completed simulation.")

        output = SimulationOutput(
            primary_id=configs.primary_id,
            seed=configs.seed,
            delta_t=configs.delta_t,
            population=configs.model.population.name,
            results=results,
        )

        # Track metrics if summary is available in context
        telemetry = ExecutionTelemetry.get_current()
        if telemetry:
            telemetry.capture_simulation(output, duration)

        return output
    except Exception as e:
        # Create output with None results for error tracking
        output = SimulationOutput(
            primary_id=configs.primary_id,
            seed=configs.seed,
            delta_t=configs.delta_t,
            population=configs.model.population.name,
            results=None,  # type: ignore
        )
        telemetry = ExecutionTelemetry.get_current()
        if telemetry:
            duration = time.time() - start_time
            telemetry.capture_simulation(output, duration, error=str(e))
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
    start_time = time.time()

    try:
        results = configs.calibrator.calibrate(strategy=configs.calibration.name, **configs.calibration.options)
        duration = time.time() - start_time
        logger.info("RUNNER: completed calibration.")

        output = CalibrationOutput(
            primary_id=configs.primary_id,
            seed=configs.seed,
            delta_t=configs.delta_t,
            population=configs.model.population.name,
            results=results,
        )

        # Track metrics if telemetry is available in context
        telemetry = ExecutionTelemetry.get_current()
        if telemetry:
            telemetry.capture_calibration(output, duration, calibration_strategy=configs.calibration)

        return output
    except Exception as e:
        # Create output with None results for error tracking
        output = CalibrationOutput(
            primary_id=configs.primary_id,
            seed=configs.seed,
            delta_t=configs.delta_t,
            population=configs.model.population.name,
            results=None,  # type: ignore
        )
        telemetry = ExecutionTelemetry.get_current()
        if telemetry:
            duration = time.time() - start_time
            telemetry.capture_calibration(output, duration, error=str(e), calibration_strategy=configs.calibration)
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
    population = configs.calibrator.parameters["epimodel"].population.name

    # Calibration phase
    calibration_start = time.time()
    try:
        calibration_results = configs.calibrator.calibrate(
            strategy=configs.calibration.name, **configs.calibration.options
        )
        calibration_duration = time.time() - calibration_start
        logger.info("RUNNER: completed calibration.")
    except Exception as e:
        # Create output with None results for error tracking
        output = CalibrationOutput(
            primary_id=configs.primary_id,
            seed=configs.seed,
            delta_t=configs.delta_t,
            population=population,
            results=None,  # type: ignore
        )
        telemetry = ExecutionTelemetry.get_current()
        if telemetry:
            calibration_duration = time.time() - calibration_start
            telemetry.capture_calibration(
                output, calibration_duration, error=f"Calibration error: {e}", calibration_strategy=configs.calibration
            )
        raise RuntimeError(f"Error during calibration: {e}")

    # Projection phase
    projection_start = time.time()
    try:
        projection_results = configs.calibrator.run_projections(
            parameters={
                "projection": True,
                "end_date": configs.projection.end_date,
                "generation": configs.projection.generation_number,
                "epimodel": configs.model,
            },
            iterations=configs.projection.n_trajectories,
        )
        projection_duration = time.time() - projection_start
        logger.info("RUNNER: completed calibration and projection.")

        output = CalibrationOutput(
            primary_id=configs.primary_id,
            seed=configs.seed,
            delta_t=configs.delta_t,
            population=configs.model.population.name,
            results=projection_results,
        )

        # Track metrics if telemetry is available in context
        telemetry = ExecutionTelemetry.get_current()
        if telemetry:
            telemetry.capture_projection(
                output,
                calibration_duration,
                projection_duration,
                configs.projection.n_trajectories,
                calibration_strategy=configs.calibration,
            )

        return output

    # If projection fails, return calibration results
    except Exception as e:
        projection_duration = time.time() - projection_start
        logger.warning(
            f"RUNNER: projection failed for model with primary_id={configs.primary_id}, returning calibration results.\nError message: {e}"
        )

        output = CalibrationOutput(
            primary_id=configs.primary_id,
            seed=configs.seed,
            delta_t=configs.delta_t,
            population=configs.model.population.name,
            results=calibration_results,
        )

        # Track metrics even if projection failed
        telemetry = ExecutionTelemetry.get_current()
        if telemetry:
            telemetry.capture_projection(
                output,
                calibration_duration,
                projection_duration,
                configs.projection.n_trajectories,
                error=f"Projection error: {e}",
                calibration_strategy=configs.calibration,
            )

        return output


def dispatch_runner(configs: BuilderOutput) -> SimulationOutput | CalibrationOutput:
    """
    Dispatch simulation/calibration/projection using a BuilderOutput and return the results.

    Parameters
    ----------
    configs : BuilderOutput
        A single BuilderOutput created by dispatch_builder()

    Returns
    -------
    SimulationOutput | CalibrationOutput
        An object containing metadata and results of simulation/calibration/projection.

    Raises
    ------
    RuntimeError
        If simulation/calibration/projection fails.
    AssertionError
        If configs are invalid.
    """
    # Get telemetry from context
    telemetry = ExecutionTelemetry.get_current()

    # Set as current context (for nested calls)
    ExecutionTelemetry.set_current(telemetry)

    try:
        # Enter runner stage
        if telemetry:
            telemetry.enter_runner()

        # Handle simulation
        if configs.simulation:
            logger.info("RUNNER: dispatched for simulation.")
            result = run_simulation(configs)
        # Handle calibration
        elif configs.calibration and not configs.projection:
            logger.info("RUNNER: dispatched for calibration.")
            result = run_calibration(configs)
        # Handle calibration and projection
        elif configs.calibration and configs.projection:
            logger.info("RUNNER: dispatched for calibration and projection.")
            result = run_calibration_with_projection(configs)
        # Error
        else:
            raise AssertionError(
                "Runner called without simulation or calibration specs. Verify that your BuilderOutputs are valid."
            )

        # Exit runner stage
        if telemetry:
            telemetry.exit_runner()

        return result
    finally:
        # Clear context when done
        ExecutionTelemetry.set_current(None)
