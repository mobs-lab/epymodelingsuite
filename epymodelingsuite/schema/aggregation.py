import logging

from pydantic import BaseModel, Field

from .common import Meta

logger = logging.getLogger(__name__)

# ----------------------------------------
# Schema models
# ----------------------------------------


class SourceExperiment(BaseModel):
    """Source experiment specification for aggregation."""

    exp_id: str
    run_id: str = "latest"
    weight: float = 1.0  # For weighted aggregation (future)


class AggregationConfiguration(BaseModel):
    """Configuration for aggregating multiple experiment results."""

    meta: Meta | None = Field(None, description="General metadata.")
    sources: list[SourceExperiment]
    method: str = (
        "sum"  # "sum", "bootstrap", "correlated", etc. Not fully sure at the moment but probably something like this.
    )
    compartments: list[str] | str = "all"
    transitions: list[str] | str = "all"
    method_options: dict = {}  # Method-specific options (e.g., n_samples for bootstrap)
    output_config: str


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
