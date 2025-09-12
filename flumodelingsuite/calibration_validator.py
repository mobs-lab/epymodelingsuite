import logging

logger = logging.getLogger(__name__)

from typing import Any

from pydantic import BaseModel, Field

# ----------------------------------------
# Schema models
# ----------------------------------------


class Calibration(BaseModel):
    """Calibration Configuration"""

    population_names: list[str] | None = Field(None, description="Population configuration")


class CalibrationConfig(BaseModel):
    modelset: Calibration
    metadata: dict[str, Any] | None = None


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
        The validated Model.
    """
    try:
        root = CalibrationConfig(**config)
        logger.info("Configuration validated successfully.")
    except Exception as e:
        raise ValueError(f"Configuration validation error: {e}")
    return root
