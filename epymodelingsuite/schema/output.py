import logging
from datetime import date
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from .common import Meta

logger = logging.getLogger(__name__)


class TabularOutputTypeEnum(str, Enum):
    """
    Types of output objects for tabular data.
    """

    CSVBytes = "CSVBytes"
    DataFrame = "DataFrame"
    Parquet = "Parquet"


class FigureOutputTypeEnum(str, Enum):
    """
    Types of output objects for figures.
    """

    MPLFigure = "MPLFigure"
    PNG = "PNG"


def get_default_tabular_output() -> list[TabularOutputTypeEnum]:
    """
    Return list containing DataFrame as default tabular output.
    """
    return [TabularOutputTypeEnum.DataFrame]


def get_default_figure_output() -> list[FigureOutputTypeEnum]:
    """
    Return list containing MPLFigure as default figure output.
    """
    return [FigureOutputTypeEnum.MPLFigure]


class OutputObject(BaseModel):
    """
    Representation for objects returned as final outputs. For internal use in dispatcher.
    """

    output_type: TabularOutputTypeEnum | FigureOutputTypeEnum = Field(description="Type of output object.")
    name: str = Field(
        description="Filename with extension for saving object (when applicable), or name without extension."
    )
    data: Any = Field(description="Actual data, such as Bytes, pd.DataFrame, mpl.Figure, etc.")


class FluScenariosOutput(BaseModel):
    """Specifications for quantile outputs in Flu Scenario Modeling Hub format."""


class Covid19ForecastOutput(BaseModel):
    """Specifications for quantile outputs in Covid 19 Forecast Hub format."""


def get_flusight_quantiles() -> list[float]:
    """
    Return an array containing the quantiles needed for FluSight submissions.
    The set of quantiles is defined at https://github.com/cdcepi/FluSight-forecast-hub/tree/main/model-output#quantile-output
    """
    import numpy as np

    # This has floating point errors
    quantiles = np.append(np.append([0.01, 0.025], np.arange(0.05, 0.95 + 0.05, 0.05)), [0.975, 0.99])
    return [round(_, 2) for _ in quantiles]
    # 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, and 0.99


class FlusightRateTrends(BaseModel):
    """Specifications for FluSight rate-trends."""

    observed_data_path: str = Field(description="Path to observed data CSV file")
    observed_value_column: str = Field(description="Name of column containing observed values in observed data CSV")
    observed_date_column: str = Field(description="Name of column containing target dates in observed data CSV")
    observed_location_column: str = Field(description="Name of column containing location in observed data CSV")


class FlusightForecastOutput(BaseModel):
    """Specifications for outputs in flusight forecast hub format."""

    reference_date: date = Field(
        description="'YYYY-MM-DD' date to treat as reference date when creating horizons and target dates for submission file."
    )
    rate_trends: FlusightRateTrends | None = Field(
        None,
        description="Add rate-trend forecasts to submission file.",
    )


class QuantilesOutput(BaseModel):
    """Specifications for quantile outputs."""

    selections: list[float] | None = Field(
        default_factory=get_flusight_quantiles,
        description="Desired quantiles expressed as floats.",
        validate_default=True,
    )
    compartments: list[str] | bool = Field(
        False,
        description="Return projection quantiles for compartments. Set `True` to get all compartments, or provide a list of identifiers (e.g. 'I_total') to select compartments.",
    )
    transitions: list[str] | bool = Field(
        False,
        description="Return projection quantiles for transitions. Set `True` to get all transitions, or provide a list of identifiers (e.g. 'I_to_R_total') to select transitions.",
    )
    calibration: list[int] | bool = Field(
        False,
        description="Return quantiles from calibration. Only calibration comparison target is available. Set `True` to get last generation, or provide a list of integers to select generations.",
    )

    @field_validator("selections")
    @classmethod
    def check_selections(cls, v):
        """Ensure quantiles are in (0, 1)."""
        if not all([0.0 < q < 1.0 for q in v]):
            raise ValueError("Received quantile not in (0, 1).")
        return v


class TrajectoriesOutput(BaseModel):
    """Specifications for trajectory outputs."""

    compartments: list[str] | bool = Field(
        False,
        description="Return projection trajectories for compartments. Set `True` to get all compartments, or provide a list of identifiers (e.g. 'I_total') to select compartments.",
    )
    transitions: list[str] | bool = Field(
        False,
        description="Return projection trajectories for transitions. Set `True` to get all transitions, or provide a list of identifiers (e.g. 'I_to_R_total') to select transitions.",
    )
    calibration: list[int] | bool = Field(
        False,
        description="Return trajectories from calibration. Only calibration comparison target is available. Set `True` to get last generation, or provide a list of integers to select generations.",
    )


class PosteriorsOutput(BaseModel):
    """Specifications for posterior outputs."""

    generations: list[int] | None = Field(None, description="Generations of SMC to get posteriors for")


class ModelMetaOutput(BaseModel):
    """Specifications for parameter tracking / run metadata outputs."""

    projection_parameters: bool = Field(
        False,
        description="Whether to record projection parameters (calibration parameters always recorded in calibration workflow).",
    )


class OutputConfiguration(BaseModel):
    """Output configuration."""

    meta: Meta | None = Field(None, description="General metadata.")

    tabular_output_types: list[TabularOutputTypeEnum] | None = Field(
        default_factory=get_default_tabular_output,
        description="Output formats to create for all requested tabular outputs.",
    )
    figure_output_types: list[FigureOutputTypeEnum] | None = Field(
        default_factory=get_default_figure_output,
        description="Output formats to create for all requested figure outputs.",
    )

    # Tabular outputs
    quantiles: QuantilesOutput | None = Field(None, description="Specifications for default format quantile outputs.")
    trajectories: TrajectoriesOutput | None = Field(
        None, description="Specifications for default format trajectory outputs."
    )
    posteriors: PosteriorsOutput | bool = Field(False, description="Specifications for posterior outputs.")

    flusight_format: FlusightForecastOutput | None = Field(
        None, description="Specifications for outputs in FluSight Forecast Hub format."
    )
    covid19_format: Covid19ForecastOutput | None = Field(
        None, description="Specifications for outputs in Covid 19 Forecast Hub format."
    )
    flusmh_format: FluScenariosOutput | None = Field(
        None, description="Specifications for outputs in Flu Scenario Modeling Hub format."
    )

    model_meta: ModelMetaOutput = Field(
        default_factory=ModelMetaOutput, description="Specifications for parameter tracking / model metadata outputs."
    )

    @model_validator(mode="after")
    def check_formats(self):
        """Ensure output format selections are compatible."""
        hub_formats = [self.flusight_format, self.covid19_format, self.flusmh_format]

        if len([1 for _ in hub_formats if bool(_)]) > 1:
            raise ValueError("Received specifications for more than one hub format.")

        return self


class OutputConfig(BaseModel):
    """Root configuration model."""

    output: OutputConfiguration = Field(description="Output configuration")


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
