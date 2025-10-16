import logging

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

    path: str = Field(description="Destination filepath for quantile outputs.")
    selections: list[float] = Field("Desired quantiles expressed as floats.")
    data_format: str | QuantileFormatEnum = Field(
        "default", description="Create quantile outputs in the default format or a supported HubVerse format."
    )
    rate_trends: bool = Field(
        False,
        description="Option to add rate-trend forecasts to quantile outputs if generating a format for which rate-trend categories are defined e.g. flusightforecast format.",
    )

    @model_validator(mode="after")
    def check_rate_trend(self):
        """Ensure rate-trend categories are available with the selected format."""
        available_formats = ["flusightforecast"]
        if self.data_format not in available_formats and self.rate_trends:
            raise ValueError

    @field_validator("selections")
    def check_selections(cls, v):
        """Ensure quantiles are in (0,1)."""
        if not all([0.0 < q < 1.0 for q in v]):
            raise ValueError("Received quantile not in (0, 1).")


class TrajectoriesOutput(BaseModel):
    """Specifications for trajectory outputs."""

    path: str = Field(description="Destination filepath for trajectory outputs.")


class PosteriorsOutput(BaseModel):
    """Specifications for posterior outputs."""

    path: str = Field(description="Destination filepath for posterior outputs.")


class RunMetaOutput(BaseModel):
    """Specifications for parameter tracking / run metadata outputs."""

    path: str = Field(description="Destination filepath for parameter tracking / run metadata outputs.")


class LogsOutput(BaseModel):
    """Specifications for logger outputs."""

    path: str = Field(description="Destination filepath for logger outputs.")


class OutputConfiguration(BaseModel):
    """Output configuration."""

    meta: Meta | None = Field(None, description="General metadata.")
    quantiles: QuantilesOutput | None = Field(None, description="Specifications for quantile outputs.")
    trajectories: TrajectoriesOutput | None = Field(None, description="Specifications for trajectory outputs.")
    posteriors: PosteriorsOutput | None = Field(None, description="Specifications for posterior outputs.")
    run_meta: RunMetaOutput | None = Field(
        None, description="Specifications for parameter tracking / run metadata outputs."
    )
    logs: LogsOutput | None = Field(None, description="Specifications for logger outputs.")


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
