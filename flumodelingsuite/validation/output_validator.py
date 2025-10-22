import logging
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

from .common_validators import Meta

logger = logging.getLogger(__name__)


class QuantilesOutput(BaseModel):
    """Specifications for quantile outputs."""

    class QuantileFormatEnum(str, Enum):
        """Types of quantile output formats."""

        default = "default"
        flusightforecast = "flusightforecast"
        flusmh = "flusmh"
        covid19forecast = "covid19forecast"

    selections: list[float] | None = Field(
        [0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975],
        description="Desired quantiles expressed as floats, default [0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975].",
    )
    data_format: str | QuantileFormatEnum | None = Field(
        "default", description="Create quantile outputs in the default format or a supported HubVerse format."
    )
    rate_trends: bool | None = Field(
        False,
        description="Option to add rate-trend forecasts to quantile outputs if generating a format for which rate-trend categories are defined e.g. flusightforecast format.",
    )

    @model_validator(mode="after")
    def check_rate_trend(self):
        """Ensure rate-trend categories are available with the selected format."""
        available_formats = ["flusightforecast"]
        if self.data_format not in available_formats and self.rate_trends:
            logger.warning(
                f"Rate-trend forecasts are not defined for selected format {self.data_format}, available formats are {available_formats}"
            )

    @field_validator("selections")
    def check_selections(cls, v):
        """Ensure quantiles are in (0, 1)."""
        if not all([0.0 < q < 1.0 for q in v]):
            raise ValueError("Received quantile not in (0, 1).")


class TrajectoriesOutput(BaseModel):
    """Specifications for trajectory outputs."""

    compartments: list[str] | None = Field(
        None, description="Filter trajectories for selected compartments, default is all compartments."
    )
    resample_freq: str | None = Field(
        None, description="Resample trajectories to a new frequency, e.g. 'D' or 'W-SAT'."
    )


class PosteriorsOutput(BaseModel):
    """Specifications for posterior outputs."""


class ModelMetaOutput(BaseModel):
    """Specifications for parameter tracking / run metadata outputs."""


class OutputConfiguration(BaseModel):
    """Output configuration."""

    meta: Meta | None = Field(None, description="General metadata.")
    quantiles: QuantilesOutput | None = Field(None, description="Specifications for quantile outputs.")
    trajectories: TrajectoriesOutput | None = Field(None, description="Specifications for trajectory outputs.")
    posteriors: PosteriorsOutput | None = Field(None, description="Specifications for posterior outputs.")
    model_meta: ModelMetaOutput | None = Field(
        None, description="Specifications for parameter tracking / model metadata outputs."
    )


class OutputConfig(BaseModel):
    """Root configuration model."""

    outputs: OutputConfiguration = Field(description="Modelset configuration")


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
