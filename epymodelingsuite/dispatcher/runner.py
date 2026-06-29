"""Runner functions for executing simulations and calibrations."""

import logging
import time

import numpy as np
import pandas as pd
from epydemix.calibration import ae, mae, mape, rmse, wmape
from ..filtering import anchor_projections_on_data

from ..schema.dispatcher import BuilderOutput, CalibrationOutput, SimulationOutput
from ..telemetry import ExecutionTelemetry

logger = logging.getLogger(__name__)

# Distance function dictionary matching builder
dist_func_dict = {
    "rmse": rmse,
    "wmape": wmape,
    "ae": ae,
    "mae": mae,
    "mape": mape,
}

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
    logger.info("RUNNER: running simulation.")
    start_time = time.time()

    # Create RNG if not provided
    if rng is None:
        rng = np.random.default_rng(configs.seed)

    try:
        results = configs.model.run_simulations(**dict(configs.simulation), rng=rng)
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
        logger.info("RUNNER: completed calibration and projection.")

        # Apply anchoring/filtering if specified
        if configs.anchoring is not None:
            try:
                # Load surveillance data
                surveillance_data = pd.read_csv(configs.anchoring.observed_data_path)
                
                # Get distance function
                distance_function = dist_func_dict[configs.anchoring.distance_function]
                
                # Create temporary CalibrationOutput for filtering
                temp_output = CalibrationOutput(
                    primary_id=configs.primary_id,
                    seed=configs.seed,
                    delta_t=configs.delta_t,
                    population=configs.model.population.name,
                    start_date_reference=configs.start_date_reference,
                    results=projection_results,
                )
                
                # Apply filtering
                original_count = len(projection_results.projections["baseline"])
                filtered_projections, filtered_projection_parameters = anchor_projections_on_data(
                    runner_output=temp_output,
                    surveillance_data=surveillance_data,
                    top_fraction=configs.anchoring.top_fraction,
                    anchor_start_date=configs.anchoring.anchor_start_date,
                    anchor_end_date=configs.anchoring.anchor_end_date,
                    distance_function=distance_function,
                    surveillance_location_col=configs.anchoring.observed_location_column,
                    surveillance_target_col=configs.anchoring.observed_value_column,
                    surveillance_date_col=configs.anchoring.observed_date_column,
                    simulation_target=configs.anchoring.simulation_target,
                )
                
                # Update projection results with filtered data
                projection_results.projections["baseline"] = filtered_projections
                projection_results.projection_parameters["baseline"] = filtered_projection_parameters
                
                logger.info(
                    f"RUNNER: applied anchoring filter, {len(filtered_projections)} trajectories remain "
                    f"out of {original_count}"
                )
            except Exception as e:
                logger.warning(
                    f"RUNNER: anchoring filter failed for model with primary_id={configs.primary_id}, "
                    f"returning unfiltered results.\nError message: {e}"
                )
                # Continue with unfiltered results


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
            f"RUNNER: projection failed for model with primary_id={configs.primary_id}, returning calibration results.\nError message: {e}"
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
            logger.info("RUNNER: dispatched for simulation.")
            result = run_simulation(configs, rng=rng)
        # Handle calibration
        elif configs.calibration and not configs.projection:
            logger.info("RUNNER: dispatched for calibration.")
            result = run_calibration(configs, rng=rng)
        # Handle calibration and projection
        elif configs.calibration and configs.projection:
            logger.info("RUNNER: dispatched for calibration and projection.")
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
