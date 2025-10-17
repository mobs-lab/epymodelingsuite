import logging
from datetime import date
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from ..utils import validate_iso3166
from .common_validators import DateParameter, Distribution, Meta

logger = logging.getLogger(__name__)

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


class ProjectionSpec(BaseModel):
    """Specification for projection after calibration."""
    n_trajectories: int = Field(
        "Number of trajectories to simulate from posterior after calibration"
    )
    generation_number: int | None = Field(
        default=None,
        description="SMC generation number from which to draw parameter sets for projection"
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

        assert self.start_date or self.parameters or self.initial_conditions, (
            "Calibration requires at least one of start_date, parameters, or compartments"
        )
        return self


class CalibrationModelset(BaseModel):
    """Modelset configuration for calibration."""

    meta: Meta | None = Field(None, description="Metadata")
    population_names: list[str] = Field(description="List of population names")
    calibration: CalibrationConfiguration = Field(description="Calibration configuration")

    @field_validator("population_names")
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
