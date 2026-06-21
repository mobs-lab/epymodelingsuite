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

    exp_id: str = Field(description="Experiment ID.")
    run_id: str = Field("latest", description="Run ID, default 'latest'.")
    weight: float = Field(1.0, description="")  # For weighted aggregation (future)


class AggregationStrategyEnum(str, Enum):
    """
    Strategy for aggregating multistrain results.
    """

    sum = "sum"
    weighted_sum = "weighted_sum"
    bootstrap = "bootstrap"
    correlated = "correlated"


class AggregationConfiguration(BaseModel):
    """Configuration for aggregating multiple experiment results."""

    meta: Meta | None = Field(None, description="General metadata.")
    sources: list[SourceExperiment] = Field(
        description="Identifiers for locating CalibrationOutput objects for aggregation."
    )
    method: AggregationStrategyEnum = Field(description="Strategy for aggregating multistrain results.")
    sampling: str = Field(
        "random", description="Strategy for sampling groups of individual trajectories for aggregation."
    )
    compartments: list[str] | bool = Field(
        False,
        description="Aggregate results for compartments. Set `True` to get all compartments, or provide a list of identifiers (e.g. 'I_total') to select compartments.",
    )
    transitions: list[str] | bool = Field(
        False,
        description="Aggregate results for transitions. Set `True` to get all transitions, or provide a list of identifiers (e.g. 'I_to_R_total') to select transitions.",
    )
    method_options: dict = Field(
        default_factory=dict, description="Method-specific options (e.g., n_samples for bootstrap)."
    )
    output_config: str = Field(description="Filename for output configuration.")


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
