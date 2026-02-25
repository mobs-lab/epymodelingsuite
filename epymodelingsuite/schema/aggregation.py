import logging
from enum import Enum

from pydantic import BaseModel, Field

from .common import Meta

logger = logging.getLogger(__name__)

# ----------------------------------------
# Schema models
# ----------------------------------------


class SourceExperiment(BaseModel):
    """
    Source experiment specification for aggregation.

    Experiment identifiers for integration with an external pipeline, e.g. epycloud.
    Used to locate sets of CalibrationOutput objects for aggregation.
    """

    strain: str = Field(description="")
    experiment: str = Field(description="")
    trajectory_file: str = Field(description="")
    target_column: str = Field(default="hospitalizations", description="")
    date_column: str = Field(default="date", description="")
    location_column: str = Field(default="population", description="")
    sim_id: str = Field(default="sim_id", description="")
    run_id: str = Field(default="latest", description="Run ID, default 'latest'.")  # Not used yet
    weight: float = Field(default=1.0, description="")  # For weighted aggregation (future)


class SamplingStrategyEnum(str, Enum):
    """
    Strategy for sampling multistrain results.
    """

    random = "random"


class AggregationStrategyEnum(str, Enum):
    """
    Strategy for aggregating multistrain results.
    """

    sum = "sum"
    weighted_sum = "weighted_sum"
    bootstrap = "bootstrap"
    correlated = "correlated"


class SamplingConfiguration(BaseModel):
    """Configuration for sampling mappings of individual trajectories across experiments."""

    method: SamplingStrategyEnum = Field(description="Strategy for sampling multistrain results.")
    n_samples: int = Field(description="Number of trajectory mapping samples to take")


class AggregationOutputConfiguration(BaseModel):
    """"""

    base_fname: str | None = Field(default=None, description="")
    raw_trajectories: bool = Field(default=True, description="")
    aggregated_trajectories: bool = Field(default=True, description="")


class AggregationConfiguration(BaseModel):
    """Configuration for aggregating multiple experiment results."""

    meta: Meta | None = Field(None, description="General metadata.")
    bucket: str = Field(description="")
    random_seed: int | None = Field(None, description="Random seed for reproducibility")
    sources: list[SourceExperiment] = Field(description="Identifiers for stage C trajectory outputs to aggregate.")
    sampling: SamplingConfiguration = Field(
        description="Configuration for sampling mappings of individual trajectories across experiments."
    )
    aggregate_method: AggregationStrategyEnum = Field(description="Strategy for aggregating multistrain results.")
    outputs: AggregationOutputConfiguration = Field(description="")


class AggregationConfig(BaseModel):
    """Root schema for aggregation configuration YAML files."""

    aggregation: AggregationConfiguration


def validate_aggregation(config: dict) -> AggregationConfig:
    """
    Validate the given configuration against the schema.

    Parameters
    ----------
    config: dict
        The configuration dictionary to validate.

    Returns
    -------
    AggregationConfig
        The validated configuration.
    """
    try:
        root = AggregationConfig(**config)
        logger.info("Configuration validated successfully.")
    except Exception as e:
        raise ValueError(f"Configuration validation error: {e}")
    return root
