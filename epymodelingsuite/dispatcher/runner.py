"""Runner functions for executing simulations and calibrations."""

import logging
import time

import numpy as np

from ..schema.dispatcher import BuilderOutput, CalibrationOutput, SimulationOutput
from ..telemetry import ExecutionTelemetry

logger = logging.getLogger(__name__)


# ===== Runner Functions =====


def run_simulation(configs: BuilderOutput, rng: np.random.Generator | None = None) -> SimulationOutput:
    """
    Run a simulation using EpiModel.run_simulations.

    Parameters
    ----------
    configs : BuilderOutput
        BuilderOutput containing model and simulation parameters.
    rng : np.random.Generator | None, optional
        Random number generator for reproducible simulations.
        If None, creates a new RNG from configs.seed.

    Returns
    -------
    SimulationOutput
        Results of the simulation with metadata.

    Raises
    ------
    RuntimeError
        If simulation fails.
    """
    population = configs.model.population.name
    logger.info(f"RUNNER: Running simulation for {population}", extra={"stage": "runner", "population": population})
    start_time = time.time()

    # Create RNG if not provided
    if rng is None:
        rng = np.random.default_rng(configs.seed)

    try:
        results = configs.model.run_simulations(**dict(configs.simulation), rng=rng)
        duration = time.time() - start_time
        logger.info(
            f"RUNNER: Completed simulation for {population} in {duration:.2f}s",
            extra={"stage": "runner", "population": population, "duration_s": duration},
        )

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


def run_calibration(configs: BuilderOutput, rng: np.random.Generator | None = None) -> CalibrationOutput:
    """
    Run a calibration using ABCSampler.calibrate.

    Parameters
    ----------
    configs : BuilderOutput
        BuilderOutput containing calibrator and calibration parameters.
    rng : np.random.Generator | None, optional
        Random number generator for reproducible calibration.
        If None, creates a new RNG from configs.seed.
        The RNG is passed to the builder, which creates simulate_wrapper with it.

    Returns
    -------
    CalibrationOutput
        Results of the calibration with metadata.

    Raises
    ------
    RuntimeError
        If calibration fails.
    """
    population = configs.model.population.name
    logger.info(f"RUNNER: Running calibration for {population}", extra={"stage": "runner", "population": population})
    start_time = time.time()

    try:
        results = configs.calibrator.calibrate(strategy=configs.calibration.name, **configs.calibration.options)
        duration = time.time() - start_time
        logger.info(
            f"RUNNER: Completed calibration for {population} in {duration:.2f}s",
            extra={"stage": "runner", "population": population, "duration_s": duration},
        )

        output = CalibrationOutput(
            primary_id=configs.primary_id,
            seed=configs.seed,
            delta_t=configs.delta_t,
            population=configs.model.population.name,
            start_date_reference=configs.start_date_reference,
            results=results,
            calibration_strategy=configs.calibration,
        )

        # Track metrics if telemetry is available in context
        telemetry = ExecutionTelemetry.get_current()
        if telemetry:
            telemetry.capture_calibration(output, duration, builder_output=configs)

        return output
    except Exception as e:
        # Create output with None results for error tracking
        output = CalibrationOutput(
            primary_id=configs.primary_id,
            seed=configs.seed,
            delta_t=configs.delta_t,
            population=configs.model.population.name,
            start_date_reference=configs.start_date_reference,
            results=None,  # type: ignore
            calibration_strategy=configs.calibration,
        )
        telemetry = ExecutionTelemetry.get_current()
        if telemetry:
            duration = time.time() - start_time
            telemetry.capture_calibration(output, duration, error=str(e), builder_output=configs)
        raise RuntimeError(f"Error during calibration: {e}")


def run_calibration_with_projection(
    configs: BuilderOutput, rng: np.random.Generator | None = None
) -> CalibrationOutput:
    """
    Run a calibration followed by projection using ABCSampler.

    Parameters
    ----------
    configs : BuilderOutput
        BuilderOutput containing calibrator, calibration, and projection parameters.
    rng : np.random.Generator | None, optional
        Random number generator for reproducible calibration and projection.
        If None, creates a new RNG from configs.seed.
        The RNG is passed to the builder, which creates simulate_wrapper with it.

    Returns
    -------
    CalibrationOutput
        Results of the projection with metadata (includes calibration results).

    Raises
    ------
    RuntimeError
        If calibration or projection fails.
    """
    population = configs.calibrator.parameters["epimodel"].population.name
    logger.info(
        f"RUNNER: Running calibration and projection for {population}",
        extra={"stage": "runner", "population": population},
    )

    # Calibration phase
    calibration_start = time.time()
    try:
        calibration_results = configs.calibrator.calibrate(
            strategy=configs.calibration.name, **configs.calibration.options
        )
        calibration_duration = time.time() - calibration_start
        logger.info(
            f"RUNNER: Completed calibration phase for {population} in {calibration_duration:.2f}s",
            extra={"stage": "runner", "population": population, "duration_s": calibration_duration},
        )
    except Exception as e:
        # Create output with None results for error tracking
        output = CalibrationOutput(
            primary_id=configs.primary_id,
            seed=configs.seed,
            delta_t=configs.delta_t,
            population=population,
            start_date_reference=configs.start_date_reference,
            results=None,  # type: ignore
            calibration_strategy=configs.calibration,
        )
        telemetry = ExecutionTelemetry.get_current()
        if telemetry:
            calibration_duration = time.time() - calibration_start
            telemetry.capture_calibration(
                output, calibration_duration, error=f"Calibration error: {e}", builder_output=configs
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
        logger.info(
            f"RUNNER: Completed calibration and projection for {population} (calibration: {calibration_duration:.2f}s, projection: {projection_duration:.2f}s)",
            extra={
                "stage": "runner",
                "population": population,
                "calibration_s": calibration_duration,
                "projection_s": projection_duration,
            },
        )

        output = CalibrationOutput(
            primary_id=configs.primary_id,
            seed=configs.seed,
            delta_t=configs.delta_t,
            population=configs.model.population.name,
            start_date_reference=configs.start_date_reference,
            results=projection_results,
            calibration_strategy=configs.calibration,
        )

        # Track metrics if telemetry is available in context
        telemetry = ExecutionTelemetry.get_current()
        if telemetry:
            telemetry.capture_projection(
                output,
                calibration_duration,
                projection_duration,
                configs.projection.n_trajectories,
                builder_output=configs,
            )

        return output

    # If projection fails, return calibration results
    except Exception as e:
        projection_duration = time.time() - projection_start
        logger.warning(
            f"RUNNER: Projection failed for {population} (primary_id={configs.primary_id}), returning calibration results. Error: {e}",
            extra={"stage": "runner", "population": population, "primary_id": configs.primary_id, "error": str(e)},
        )

        output = CalibrationOutput(
            primary_id=configs.primary_id,
            seed=configs.seed,
            delta_t=configs.delta_t,
            population=configs.model.population.name,
            start_date_reference=configs.start_date_reference,
            results=calibration_results,
            calibration_strategy=configs.calibration,
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
                builder_output=configs,
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
    # Create RNG from seed
    rng = np.random.default_rng(configs.seed)

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
            logger.info("RUNNER: Dispatched for simulation", extra={"stage": "runner"})
            result = run_simulation(configs, rng=rng)
        # Handle calibration
        elif configs.calibration and not configs.projection:
            logger.info("RUNNER: Dispatched for calibration", extra={"stage": "runner"})
            result = run_calibration(configs, rng=rng)
        # Handle calibration and projection
        elif configs.calibration and configs.projection:
            logger.info("RUNNER: Dispatched for calibration with projection", extra={"stage": "runner"})
            result = run_calibration_with_projection(configs, rng=rng)
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
