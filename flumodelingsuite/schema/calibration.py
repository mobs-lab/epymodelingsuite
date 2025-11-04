import logging
from datetime import date, timedelta
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from ..utils import parse_timedelta, validate_iso3166
from .common import DateParameter, Distribution, Meta

from epydemix.calibration import ABCSampler, CalibrationResults

logger = logging.getLogger(__name__)

def reproduce_trajectory(
    calibrator: ABCSampler,
    calibration_results: CalibrationResults,
    particle_index: int,
    generation: int,
) -> dict:
    """
    Reproduce a specific trajectory using the stored simulate_wrapper.

    Parameters
    ----------
    calibrator : ABCSampler
        The ABCSampler instance that contains the simulation function.
    calibration_results : CalibrationResults
        Calibration results containing trajectories and parameters.
    generation : int
        Generation number to reproduce from.
    particle_index : int
        Index of the particle/trajectory to reproduce.

    Returns
    -------
    dict
        Reproduced simulation results.
    """
    # Validate generation index
    num_generations = len(calibration_results.posterior_distributions)
    if generation < 0 or generation >= num_generations:
        errMsg = f"Invalid generation {generation}. Must be between 0 and {num_generations - 1}."
        raise ValueError(errMsg)

    # Extract the parameters for this particle
    params_df = calibration_results.posterior_distributions[generation]

    # Validate particle index
    num_particles = len(params_df)
    if particle_index < 0 or particle_index >= num_particles:
        errMsg = f"Invalid particle_index {particle_index}. Must be between 0 and {num_particles - 1}."
        raise ValueError(errMsg)
    
    params = params_df.iloc[particle_index].to_dict()

    # Get fixed parameters from the calibrator
    fixed_params = calibrator.parameters.copy()

    # Merge sampled parameters with fixed parameters
    all_params = {**fixed_params, **params}

    # Extract the random state from the original trajectory
    original_trajectory = calibration_results.selected_trajectories[generation][particle_index]
    random_state = original_trajectory["random_state"]

    # Add random state to params
    all_params["random_state"] = random_state
    all_params["projection"] = True

    # Call the stored simulate_wrapper
    result = calibrator.simulation_function(all_params)

    return result


def reproduce_trajectories_in_generation(
    calibrator: ABCSampler,
    calibration_results: CalibrationResults,
    generation: int | None = None,
) -> list[dict]:
    """
    Reproduce all trajectories in a specified generation.

    Parameters
    ----------
    calibrator : ABCSampler
        The ABCSampler instance that contains the simulation function.
    calibration_results : CalibrationResults
        Calibration results containing trajectories and parameters.
    generation : int, optional
        Generation number to reproduce. Defaults to the last generation.

    Returns
    -------
    list of dict
        List of reproduced simulation results for each particle in the generation.
    """
    if generation is None:
        generation = len(calibration_results.selected_trajectories) - 1

    results = []
    for particle_index in range(len(calibration_results.selected_trajectories[generation])):
        reproduced = reproduce_trajectory(
            calibrator=calibrator,
            calibration_results=calibration_results,
            particle_index=particle_index,
            generation=generation,
        )
        results.append(reproduced)

    return results


def reproduce_all_trajectories(
    calibrator: ABCSampler,
    calibration_results: CalibrationResults,
) -> dict[int, list[dict]]:
    """
    Reproduce all trajectories across all generations.

    Parameters
    ----------
    calibrator : ABCSampler
        The ABCSampler instance that contains the simulation function.
    calibration_results : CalibrationResults
        Calibration results containing trajectories and parameters.

    Returns
    -------
    dict
        Dictionary mapping generation numbers to lists of reproduced trajectories.
    """
    results = {}
    for generation in range(len(calibration_results.selected_trajectories)):
        results[generation] = reproduce_trajectories_in_generation(
            calibrator=calibrator,
            calibration_results=calibration_results,
            generation=generation,
        )

    return results

# ----------------------------------------
# Schema models
# ----------------------------------------


class CalibrationStrategy(BaseModel):
    """Calibration strategy configuration."""

    class CalibrationStrategyEnum(str, Enum):
        """Types of calibration strategies."""

        SMC = "SMC"
        rejection = "rejection"
        top_fraction = "top_fraction"

    name: str | CalibrationStrategyEnum = Field(
        description="Name of calibration strategy for epydemix.calibration.abc module (e.g., 'SMC', 'rejection', 'top_fraction')",
    )
    options: dict[str, Any] = Field(
        default_factory=dict, description="Strategy-specific arguments for calibrate() function"
    )

    @field_validator("options")
    @classmethod
    def parse_max_time(cls, v: dict[str, Any]) -> dict[str, Any]:
        """
        Parse max_time option from string to timedelta if present.

        If the options dict contains a 'max_time' key with a string value,
        it will be converted to a datetime.timedelta using parse_timedelta().
        If max_time is already a timedelta, it will be kept as-is.

        Supported formats include pandas Timedelta strings (e.g., '30m', '2H', '1h30m')
        and frequency aliases (e.g., 'W', '2D'). See pandas documentation for details:
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases

        Parameters
        ----------
        v : dict[str, Any]
            The options dictionary to validate.

        Returns
        -------
        dict[str, Any]
            The options dictionary with max_time converted to timedelta if applicable.

        Raises
        ------
        ValueError
            If max_time string cannot be parsed into a valid timedelta.

        Examples
        --------
        >>> strategy = CalibrationStrategy(name="SMC", options={"max_time": "4h"})
        >>> strategy.options["max_time"]
        datetime.timedelta(seconds=14400)
        """
        if "max_time" in v and isinstance(v["max_time"], str):
            try:
                v["max_time"] = parse_timedelta(v["max_time"])
            except ValueError as e:
                msg = f"Invalid max_time value: {e}"
                raise ValueError(msg) from e
        return v


class ProjectionSpec(BaseModel):
    """Specification for projection after calibration."""

    n_trajectories: int = Field("Number of trajectories to simulate from posterior after calibration")
    generation_number: int | None = Field(
        default=None, description="SMC generation number from which to draw parameter sets for projection"
    )


class ComparisonSpec(BaseModel):
    """Specification for comparing observed and simulated data."""

    observed_value_column: str = Field(description="Name of column containing observed values in observed data CSV")
    observed_date_column: str = Field(description="Name of column containing target dates in observed data CSV")
    simulation: list[str] = Field(description="List of transition names to sum for comparison (e.g. I_to_R)")


class CalibrationParameter(BaseModel):
    """Parameter specification for calibration."""

    prior: Distribution = Field(description="Prior distribution for parameter calibration")


class FittingWindow(BaseModel):
    """Specification for the time window used in calibration fitting."""

    start_date: date = Field(description="Start date of fitting window.")
    end_date: date = Field(description="End date of fitting window.")

    @model_validator(mode="after")
    def validate_date_order(self: "FittingWindow") -> "FittingWindow":
        """Ensure end_date is after start_date."""
        # Note: DateParameter can be a string date or have a prior distribution
        # Only validate if both are actual date strings
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")
        return self


class CalibrationConfiguration(BaseModel):
    """Calibration configuration section."""

    strategy: CalibrationStrategy = Field(description="Calibration strategy configuration")

    # Sampler options, passed directly when initializing ABCSampler
    distance_function: str = Field("rmse", description="Distance function for comparing data")
    observed_data_path: str = Field(description="Path to observed data CSV file")
    comparison: list[ComparisonSpec] = Field(description="Specifications for data comparison")

    # What we calibrate for
    start_date: DateParameter | None = Field(None, description="Start date parameter specification")
    parameters: dict[str, CalibrationParameter] | None = Field(
        None, description="Parameter specifications for calibration"
    )
    compartments: dict[str, CalibrationParameter] | None = Field(
        None, description="Initial conditions specifications for calibration"
    )
    fitting_window: FittingWindow = Field(description="Time window for calibration fitting")

    projection: ProjectionSpec | None = Field(None, description="Specification for projection")

    @model_validator(mode="after")
    def check_calibration_consistency(self: "CalibrationConfiguration") -> "CalibrationConfiguration":
        """Validate calibration configuration consistency."""
        # Ensure we have at least one comparison specification
        if not self.comparison:
            msg = "At least one comparison specification is required"
            raise ValueError(msg)

        # Ensure all comparison specs have non-empty simulation lists
        for comp in self.comparison:
            if not comp.simulation:
                msg = f"Comparison for '{comp.observed}' must specify at least one simulation transition"
                raise ValueError(msg)

        assert self.start_date or self.parameters or self.compartments, (
            "Calibration requires at least one of start_date, parameters, or compartments"
        )
        return self

    @model_validator(mode="after")
    def validate_sampled_start_date_against_fitting_window(
        self: "CalibrationConfiguration",
    ) -> "CalibrationConfiguration":
        """Check that sampled start_date does not exceed fitting window end_date."""
        # Only validate if start_date with prior is specified
        if not self.start_date or not self.start_date.prior:
            return self

        prior = self.start_date.prior
        reference_date = self.start_date.reference_date
        fitting_window_end = self.fitting_window.end_date

        # Determine max offset based on distribution type
        max_offset = None

        if prior.name == "randint":
            # randint(low, high) samples integers in [low, high), so max is high - 1
            if len(prior.args) >= 2:
                max_offset = int(prior.args[1]) - 1
        elif prior.name == "uniform":
            # uniform(loc, scale) samples in [loc, loc+scale]
            if len(prior.args) >= 2:
                max_offset = int(prior.args[0] + prior.args[1])
        else:
            # Unsupported distribution - skip validation with warning
            warn_msg = (
                f"Start date uses distribution '{prior.name}' which does not get validated for not exceeding fitting window."
                "Only 'randint' and 'uniform' distributions are validated against fitting window. "
                "Please ensure your start_date prior is compatible with the fitting window manually."
            )
            logger.warning(warn_msg)
            return self

        # If we couldn't extract max_offset, warn and skip
        if max_offset is None:
            warn_msg = f"Could not determine max offset from start_date prior distribution '{prior.name}' with args {prior.args}. Skipping validation."
            logger.warning(warn_msg)
            return self

        # Calculate maximum possible start date
        max_start_date = reference_date + timedelta(days=max_offset)

        # Raise error if max start date exceeds fitting window end
        if max_start_date > fitting_window_end:
            msg = (
                "Sampled start_date could extend beyond fitting window. "
                f"Reference date: {reference_date}, Max offset: {max_offset} days, "
                f"Max possible start date: {max_start_date}, "
                f"Fitting window end: {fitting_window_end}. "
                "Consider reducing the prior upper bound or extending the fitting window end date."
            )
            raise ValueError(msg)

        return self


class CalibrationModelset(BaseModel):
    """Modelset configuration for calibration."""

    meta: Meta | None = Field(None, description="General metadata.")
    population_names: list[str] = Field(description="List of population names")
    calibration: CalibrationConfiguration = Field(description="Calibration configuration")

    @field_validator("population_names")
    @classmethod
    def validate_populations(cls, v):
        """Validate each population name in the list."""
        validated_populations = []
        for population in v:
            if population == "all":
                validated_populations.append(population)
            else:
                validated_populations.append(validate_iso3166(population))
        return validated_populations


class CalibrationConfig(BaseModel):
    """Root configuration model for calibration."""

    modelset: CalibrationModelset = Field(description="Modelset configuration")


def validate_calibration(config: dict) -> CalibrationConfig:
    """
    Validate the given configuration against the schema.

    Parameters
    ----------
    config: dict
        The configuration dictionary to validate.

    Returns
    -------
    CalibrationConfig
        The validated configuration.
    """
    try:
        root = CalibrationConfig(**config)
        logger.info("Configuration validated successfully.")
    except Exception as e:
        msg = f"Configuration validation error: {e}"
        raise ValueError(msg) from e
    return root
