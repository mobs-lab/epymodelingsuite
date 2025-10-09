import logging
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from .common_validators import DateParameter, Distribution, Meta
from .utils import validate_iso3166

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
        ...,
        description="Name of calibration strategy for epydemix.calibration.abc module (e.g., 'SMC', 'rejection', 'top_fraction')",
    )
    options: dict[str, Any] = Field(
        default_factory=dict, description="Strategy-specific arguments for calibrate() function"
    )


class ComparisonSpec(BaseModel):
    """Specification for comparing observed and simulated data."""

    observed: str = Field(..., description="Name of column containing observed values in observed data CSV")
    obs_date: str = Field(..., description="Name of column containing target dates in observed data CSV")
    simulation: list[str] = Field(..., description="List of transition names to sum for comparison (e.g. I_to_R)")


class CalibrationParameter(BaseModel):
    """Parameter specification for calibration."""

    prior: Distribution = Field(..., description="Prior distribution for parameter calibration")
    
class FittingWindow(BaseModel):
    """Specification for the time window used in calibration fitting."""
    
    window_start: str = Field(
        ..., 
        description="Start date of fitting window (ISO format: YYYY-MM-DD)"
    )
    window_end: str = Field(
        ..., 
        description="End date of fitting window (ISO format: YYYY-MM-DD)"
    )
    
    @model_validator(mode="after")
    def validate_date_order(cls, m: "FittingWindow") -> "FittingWindow":
        """Ensure end_date is after start_date."""
        # Note: DateParameter can be a string date or have a prior distribution
        # Only validate if both are actual date strings
        if isinstance(m.window_start, str) and isinstance(m.window_end, str):
            from datetime import datetime
            start = datetime.fromisoformat(m.window_start)
            end = datetime.fromisoformat(m.window_end)
            if end <= start:
                raise ValueError("end_date must be after start_date")
        return m
    
class CalibrationConfiguration(BaseModel):
    """Calibration configuration section."""

    strategy: CalibrationStrategy = Field(..., description="Calibration strategy configuration")

    # Sampler options, passed directly when initializing ABCSampler
    distance_function: str = Field("rmse", description="Distance function for comparing data")
    observed_data_path: str = Field(..., description="Path to observed data CSV file")
    comparison: list[ComparisonSpec] = Field(..., description="Specifications for data comparison")

    # What we calibrate for
    start_date: DateParameter | None = Field(None, description="Start date parameter specification")
    parameters: dict[str, CalibrationParameter] | None = Field(
        None, description="Parameter specifications for calibration"
    )
    compartments: dict[str, CalibrationParameter] | None = Field(
        None, description="Initial conditions specifications for calibration"
    )
    fitting_window: FittingWindow = Field(
        ...,
        description="Time window for calibration fitting"
    )

    @model_validator(mode="after")
    def check_calibration_consistency(cls, m: "CalibrationConfiguration") -> "CalibrationConfiguration":
        """Validate calibration configuration consistency."""
        # Ensure we have at least one comparison specification
        if not m.comparison:
            msg = "At least one comparison specification is required"
            raise ValueError(msg)

        # Ensure all comparison specs have non-empty simulation lists
        for comp in m.comparison:
            if not comp.simulation:
                msg = f"Comparison for '{comp.observed}' must specify at least one simulation transition"
                raise ValueError(msg)

        assert m.start_date or m.parameters or m.initial_conditions, (
            "Calibration requires at least one of start_date, parameters, or compartments"
        )
        return m


class CalibrationModelset(BaseModel):
    """Modelset configuration for calibration."""

    meta: Meta | None = Field(None, description="Metadata")
    population_names: list[str] = Field(..., description="List of population names")
    calibration: CalibrationConfiguration = Field(..., description="Calibration configuration")

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

    modelset: CalibrationModelset = Field(..., description="Modelset configuration")


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
