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

    strain: str = Field(description="Label for strain.")
    experiment: str = Field(description="Name of experiment in cloud bucket.")
    run_id: str = Field(default="latest", description="Run ID, set to a value from the google cloud bucket, or 'any' (fails if more than one matching trajectory file exists), or 'latest' (default).")
    trajectory_file: str = Field(description="Name of trajectory file from experiment.")
    target_column: str = Field(default="hospitalizations", description="Name of column in trajectory file with target values.")
    date_column: str = Field(default="date", description="Name of column in trajectory file with date.")
    location_column: str = Field(default="population", description="Name of column in trajectory file with location/population.")
    sim_id: str = Field(default="sim_id", description="Name of column in trajectory file with simulation ID.")


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


class BaselineStrategyEnum(str, Enum):
    """
    Strategy for post-aggregation baseline noise addition.
    """

    negative_binomial = "negative_binomial"
    

class SamplingConfiguration(BaseModel):
    """Configuration for sampling mappings of individual trajectories across experiments."""

    method: SamplingStrategyEnum = Field(description="Strategy for sampling multistrain results.")
    n_samples: int = Field(description="Number of trajectory mapping samples to take.")


class BaselineConfiguration(BaseModel):
    """Configuration for adding post-hoc baseline noise."""
    
    method: BaselineStrategyEnum = Field(description="Strategy for post-aggregation baseline addition.")
    observed_means: str = Field(description="Filename to look for in data module containing baseline averages.")
    dispersion_values: int | list[int] = Field(description="Value(s) to use for parameter modifying dispersion/variance of baseline noise.")


class AggregationConfiguration(BaseModel):
    """Configuration for aggregating multiple experiment results."""

    meta: Meta | None = Field(None, description="General metadata.")
    bucket: str = Field(description="Path to bucket on cloud e.g. 'gs://gs_mobs_jessica/pipeline/flu'")
    random_seed: int | None = Field(None, description="Random seed for reproducibility")
    submission_week: str | int = Field(description="Epiweek of submission in CDC format, i.e. 'YYYYww'")
    sources: list[SourceExperiment] = Field(description="Identifiers for stage C trajectory outputs to aggregate.")
    sampling: SamplingConfiguration = Field(
        description="Configuration for sampling mappings of individual trajectories across experiments."
    )
    aggregate_method: AggregationStrategyEnum = Field(description="Strategy for aggregating multistrain results.")
    baseline: BaselineConfiguration | None = Field(None, description="Configuration for adding post-hoc baseline noise.")


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
