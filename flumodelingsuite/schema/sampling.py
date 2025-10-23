import logging

from ..utils import validate_iso3166

logger = logging.getLogger(__name__)

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from .common import DateParameter, Distribution, Meta

# ----------------------------------------
# Schema models
# ----------------------------------------


class CompartmentSamplingRange(BaseModel):
    """Range specification for compartment sampling."""

    min: float = Field(description="Minimum value for compartment")
    max: float = Field(description="Maximum value for compartment")

    @field_validator("max")
    def check_max_greater_than_min(cls, v: float, info: Any) -> float:
        """Ensure max is greater than min."""
        min_val = info.data.get("min")
        if min_val is not None and v <= min_val:
            raise ValueError("max must be greater than min")
        return v


class Parameter(BaseModel):
    """Parameter specification for sampling."""

    distribution: Distribution | None = Field(None, description="Distribution for parameter sampling")
    values: list[float] | None = Field(None, description="List of discrete values for grid sampling")

    @model_validator(mode="after")
    def check_parameter_specification(self: "Parameter") -> "Parameter":
        """Ensure parameter has either distribution or values specified."""
        if self.distribution is None and self.values is None:
            raise ValueError("Parameter must have either 'distribution' or 'values' specified")
        if self.distribution is not None and self.values is not None:
            raise ValueError("Parameter cannot have both 'distribution' and 'values' specified")
        return self


class Sampler(BaseModel):
    """Sampler configuration."""

    class StrategyEnum(str, Enum):
        """Sampling strategies."""

        grid = "grid"
        LHS = "LHS"

    strategy: StrategyEnum = Field(description="Sampling strategy")
    n_samples: int | None = Field(None, description="Number of samples (required for LHS and montecarlo)")
    parameters: list[str] = Field(description="List of parameters to sample")
    compartments: list[str] | None = Field(None, description="List of compartments to sample (for LHS/montecarlo)")

    @model_validator(mode="after")
    def check_sampler_requirements(self: "Sampler") -> "Sampler":
        """Validate sampler configuration based on strategy."""
        if self.strategy in ["LHS"] and self.n_samples is None:
            raise ValueError(f"{self.strategy} strategy requires 'n_samples' to be specified")
        if self.strategy == "grid" and self.n_samples is not None:
            raise ValueError("grid strategy does not use 'n_samples'")
        return self


class SamplingConfiguration(BaseModel):
    """Sampling configuration section."""

    samplers: list[Sampler] = Field(description="List of samplers")
    compartments: dict[str, CompartmentSamplingRange] | None = Field(
        None, description="Compartment ranges for sampling"
    )
    parameters: dict[str, Parameter] = Field(description="Parameter specifications")
    start_date: DateParameter | None = Field(None, description="Start date parameter specification")

    @model_validator(mode="after")
    def check_sampling_consistency(self: "SamplingConfiguration") -> "SamplingConfiguration":
        """Validate consistency between samplers and parameter/compartment definitions."""
        all_sampled_params = set()
        all_sampled_compartments = set()

        for sampler in self.samplers:
            all_sampled_params.update(sampler.parameters)
            if sampler.compartments:
                all_sampled_compartments.update(sampler.compartments)

        # Check that all sampled parameters are defined
        for param in all_sampled_params:
            if param == "start_date":
                if self.start_date is None:
                    raise ValueError(f"Parameter '{param}' is sampled but not defined in sampling configuration")
            elif param not in self.parameters:
                raise ValueError(f"Parameter '{param}' is sampled but not defined in parameters section")

        # Check that all sampled compartments are defined
        for compartment in all_sampled_compartments:
            if self.compartments is None or compartment not in self.compartments:
                raise ValueError(f"Compartment '{compartment}' is sampled but not defined in compartments section")

        return self


class SamplingModelset(BaseModel):
    """Modelset configuration for sampling."""

    meta: Meta | None = Field(None, description="General metadata")
    population_names: list[str] = Field(description="List of population names")
    sampling: SamplingConfiguration = Field(description="Sampling configuration")

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


class SamplingConfig(BaseModel):
    """Root configuration model."""

    modelset: SamplingModelset = Field(description="Modelset configuration")


def validate_sampling(config: dict) -> SamplingConfig:
    """
    Validate the given configuration against the schema.

    Parameters
    ----------
    config: dict
        The configuration dictionary to validate.

    Returns
    -------
    SamplingConfig
        The validated configuration.
    """
    try:
        root = SamplingConfig(**config)
        logger.info("Configuration validated successfully.")
    except Exception as e:
        raise ValueError(f"Configuration validation error: {e}")
    return root
