import logging

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class OutputConfiguration(BaseModel):
    """Output configuration."""


class OutputConfig(BaseModel):
    """Root configuration model."""

    output_specs: OutputConfiguration = Field(..., description="Modelset configuration")


def validate_output(config: dict) -> OutputConfig:
    """
    Validate the given configuration against the schema.

    Parameters
    ----------
    config: dict
        The configuration dictionary to validate.

    Returns
    -------
    OutputConfig
        The validated configuration.
    """
    try:
        root = OutputConfig(**config)
        logger.info("Configuration validated successfully.")
    except Exception as e:
        raise ValueError(f"Configuration validation error: {e}")
    return root
