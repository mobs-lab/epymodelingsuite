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
    PDF = "PDF"
    SVG = "SVG"


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
    def check_selections(cls, v: list[float]) -> list[float]:
        """Ensure quantiles are in (0, 1)."""
        if not all(0.0 < q < 1.0 for q in v):
            msg = "Received quantile not in (0, 1)."
            raise ValueError(msg)
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


class PosteriorsOutput(BaseModel):
    """Specifications for posterior outputs."""

    generations: list[int] | None = Field(None, description="Generations of SMC to get posteriors for")


class ModelMetaOutput(BaseModel):
    """Specifications for parameter tracking / run metadata outputs."""

    projection_parameters: bool = Field(
        False,
        description="Whether to record projection parameters (calibration parameters always recorded in calibration workflow).",
    )


class PosteriorPlotConfig(BaseModel):
    """Configuration for posterior distribution plots."""

    single: bool | list[str] = Field(
        False,
        description="Create single plot per location. True for all locations, or list of specific locations.",
    )
    grid: bool = Field(False, description="Create grid plot with all locations.")
    bins: int = Field(30, description="Number of histogram bins.")


class QuantilesOutputTypeEnum(str, Enum):
    """Types of quantile plot outputs."""

    FILTERED = "filtered"
    FULL = "full"
    SIDE_BY_SIDE = "side_by_side"


class SideBySidePanelConfig(BaseModel):
    """Configuration for a single panel in side-by-side plot."""

    surveillance_points: int | None = Field(
        None, description="Number of most recent surveillance points to display. None = all points."
    )
    surveillance_start_date: str | None = Field(
        None, description="Filter surveillance to show only points >= this date (YYYY-MM-DD). Overrides surveillance_points if both set."
    )


class QuantilesOutputConfig(BaseModel):
    """Configuration for a single quantile plot output."""

    type: QuantilesOutputTypeEnum = Field(description="Type of output to generate.")

    # Display options (for all output types)
    show_calibration: bool = Field(False, description="Show calibration period quantiles.")
    show_projection: bool = Field(True, description="Show projection period quantiles.")
    show_surveillance: bool = Field(True, description="Overlay surveillance observations.")
    show_fitting_window_line: bool = Field(True, description="Show fitting window vertical lines.")

    # Filter settings (only for FILTERED and FULL types)
    surveillance_points: int | None = Field(
        None, description="Number of most recent surveillance points to display. None = all points. Not used for SIDE_BY_SIDE."
    )
    surveillance_start_date: str | None = Field(
        None, description="Filter surveillance to show only points >= this date (YYYY-MM-DD). Overrides surveillance_points if both set. Not used for SIDE_BY_SIDE."
    )
    horizon_max: int | None = Field(None, description="Override base horizon_max. None = use base config value.")

    # Panel settings (only for SIDE_BY_SIDE type)
    full_panel: SideBySidePanelConfig | None = Field(
        None, description="Configuration for full (left) panel in side-by-side plot."
    )
    filtered_panel: SideBySidePanelConfig | None = Field(
        None, description="Configuration for filtered (right) panel in side-by-side plot."
    )

    # Layout settings (only for SIDE_BY_SIDE type)
    spacing: float = Field(0.3, description="For side-by-side: horizontal spacing between panels.")
    figsize: tuple[float, float] | None = Field(None, description="For side-by-side: figure size (width, height).")


class QuantilesGridConfig(BaseModel):
    """Configuration for quantiles grid plot."""

    enabled: bool = Field(False, description="Create grid plot.")
    panels_per_row: int = Field(4, description="Number of panels per row in grid.")


class QuantilesCalibrationConfig(BaseModel):
    """Configuration for calibration period visualization styling."""

    color: str = Field("C0", description="Color for calibration ribbons.")


class QuantilesProjectionConfig(BaseModel):
    """Configuration for projection period visualization styling."""

    color: str = Field("C1", description="Color for projection ribbons.")


class QuantilesSurveillanceConfig(BaseModel):
    """Configuration for surveillance data source."""

    data_path: str | None = Field(None, description="Path to surveillance data file.")
    value_column: str | None = Field(None, description="Column containing observed values.")
    date_column: str | None = Field(None, description="Column containing dates.")
    location_column: str | None = Field(None, description="Column containing location identifiers.")


class QuantilesPlotConfig(BaseModel):
    """Configuration for quantile ribbon plots."""

    single: bool | list[str] = Field(
        False,
        description="Create single plot per location. True for all locations, or list of specific locations.",
    )
    grid: QuantilesGridConfig = Field(
        default_factory=QuantilesGridConfig,
        description="Grid plot configuration.",
    )

    # Output configuration
    outputs: list[QuantilesOutputConfig] = Field(
        default_factory=lambda: [
            QuantilesOutputConfig(
                type=QuantilesOutputTypeEnum.FILTERED,
                surveillance_points=8,
                show_calibration=False,
                show_projection=True,
                show_surveillance=True,
                show_fitting_window_line=True,
            ),
            QuantilesOutputConfig(
                type=QuantilesOutputTypeEnum.FULL,
                surveillance_points=None,
                show_calibration=False,
                show_projection=True,
                show_surveillance=True,
                show_fitting_window_line=True,
            ),
            QuantilesOutputConfig(
                type=QuantilesOutputTypeEnum.SIDE_BY_SIDE,
                show_calibration=False,
                show_projection=True,
                show_surveillance=True,
                show_fitting_window_line=True,
                columns=4,
                spacing=0.3,
            ),
        ],
        description="List of output configurations to generate.",
    )

    # Shared settings
    quantiles: list[float] = Field(
        [0.025, 0.25, 0.5, 0.75, 0.975],
        description="Quantile levels for ribbons (95% CrI + IQR + median).",
    )
    horizon_max: int | None = Field(
        3, description="Base maximum forecast horizon (weeks ahead). Can be overridden per output."
    )

    # Shared styling (data source and colors)
    calibration: QuantilesCalibrationConfig = Field(
        default_factory=QuantilesCalibrationConfig,
        description="Calibration period styling.",
    )
    projection: QuantilesProjectionConfig = Field(
        default_factory=QuantilesProjectionConfig,
        description="Projection period styling.",
    )
    surveillance: QuantilesSurveillanceConfig = Field(
        ...,
        description="Surveillance data source configuration.",
    )

    @field_validator("quantiles")
    @classmethod
    def check_quantiles_valid(cls, v: list[float]) -> list[float]:
        """Ensure quantiles are in [0, 1]."""
        for q in v:
            if not 0.0 <= q <= 1.0:
                msg = f"Quantile {q} must be between 0 and 1"
                raise ValueError(msg)
        return v


class PlotsConfig(BaseModel):
    """Configuration for visualization plots."""

    figure_output_types: list[FigureOutputTypeEnum] = Field(
        default_factory=get_default_figure_output,
        description="Output formats to create for all requested figure outputs.",
    )
    dpi: int | None = Field(150, description="DPI for raster formats.")

    reference_date: date = Field(description="Forecast reference date for vertical line.")

    posterior: PosteriorPlotConfig = Field(
        default_factory=PosteriorPlotConfig,
        description="Posterior distribution plot settings.",
    )
    quantiles: QuantilesPlotConfig = Field(
        default_factory=QuantilesPlotConfig,
        description="Quantile ribbon plot settings.",
    )


class OutputConfiguration(BaseModel):
    """Output configuration."""

    meta: Meta | None = Field(None, description="General metadata.")

    tabular_output_types: list[TabularOutputTypeEnum] | None = Field(
        default_factory=get_default_tabular_output,
        description="Output formats to create for all requested tabular outputs.",
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

    # Plots
    plots: PlotsConfig | None = Field(
        None,
        description="Visualization plot settings. Requires modelset config for inference.",
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
