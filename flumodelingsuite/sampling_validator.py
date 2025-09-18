import logging

logger = logging.getLogger(__name__)

from typing import Any

from pydantic import BaseModel, Field

# ----------------------------------------
# Schema models
# ----------------------------------------


class Sampling(BaseModel):
    """Sampling Configuration"""

    population_names: list[str] | None = Field(None, description="Population configuration")


class SamplingConfig(BaseModel):
    modelset: Sampling
    metadata: dict[str, Any] | None = None


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
        The validated Model.
    """
    try:
        root = SamplingConfig(**config)
        logger.info("Configuration validated successfully.")
    except Exception as e:
        raise ValueError(f"Configuration validation error: {e}")
    return root
