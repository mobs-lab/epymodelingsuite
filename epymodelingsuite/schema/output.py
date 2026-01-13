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
    Return a list containing the quantiles needed for FluSight submissions.

    The set of quantiles is defined at https://github.com/cdcepi/FluSight-forecast-hub/tree/main/model-output#quantile-output
    """
    return [
        0.01,
        0.025,
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
        0.975,
        0.99,
    ]


def get_quantile_ribbon_default() -> list[float]:
    """Return a list containing default quantiles for plotting ribbons."""
    return [0.025, 0.25, 0.5, 0.75, 0.975]


class ObservedValuesConfig(BaseModel):
    """Specifications for selecting observed values."""

    data_path: str = Field(description="Path to observed data CSV file")
    value_column: str = Field(description="Name of column containing observed values in observed data CSV")
    date_column: str = Field(description="Name of column containing target dates in observed data CSV")
    location_column: str = Field(description="Name of column containing location in observed data CSV")
    location_format: str = Field(
        default="ISO",
        description="Format of location identifiers in observed data. Options: ISO, FIPS, abbreviation, name, epydemix_population, metrocast_location_id",
    )

    @field_validator("location_format")
    @classmethod
    def validate_location_format(cls, v: str) -> str:
        """Ensure location_format is a valid option."""
        valid_formats = {"ISO", "FIPS", "abbreviation", "name", "epydemix_population", "metrocast_location_id"}
        if v not in valid_formats:
            msg = f"location_format must be one of {valid_formats}, got '{v}'"
            raise ValueError(msg)
        return v


class PropEDStrategyEnum(str, Enum):
    """
    Strategy for generating prop ED forecasts.

    surveillance_window - fit rescaling factor using observed hospitalizations against observed ED visits over a fitting window
    calibration_window - fit rescaling factor using median calibration quantile against observed ED visits over a fitting window anchored at the end of the calibration fitting window
    transition - use a transition directly (e.g., ed_prop) computed in UDF, no rescaling needed
    """

    surveillance_window = "surveillance_window"
    calibration_window = "calibration_window"
    transition = "transition"


# TODO: Consider renaming 'prop_ed' key to a more generic name (e.g., 'ed_visits', 'ed_target')
# Metrocast uses "Flu ED visits pct" which is a percentage, not a proportion.
class FlusightPropED(BaseModel):
    """Specifications for generating ED-related target forecasts."""

    target: str = Field(
        default="wk inc flu prop ed visits",
        description="Target name for the submission file (e.g., 'wk inc flu prop ed visits', 'Flu ED visits pct').",
    )
    strategy: PropEDStrategyEnum = Field(description="Strategy for generating ED forecasts.")
    transition_name: str | None = Field(
        None,
        description="Name of transition to use for prop ED forecasts (required iff using 'transition' strategy).",
    )
    ed_source: str | None = Field(
        None,
        description="Reference name for 'wk inc flu prop ed visits' surveillance data (required for rescaling strategies).",
    )
    hosp_source: str | None = Field(
        None,
        description="Name of hospitalization surveillance source from output.options.surveillance (iff using 'surveillance_window' strategy).",
    )
    fit_start: date | None = Field(
        None, description="Start date for rescaling factor fitting window (iff using 'surveillance_window' strategy)."
    )
    fit_end: date | None = Field(
        None, description="End date for rescaling factor fitting window (iff using 'surveillance_window' strategy)."
    )
    num_fit_weeks: int | None = Field(
        None,
        description="Number of weeks for rescaling factor fitting window, extending back from the end of the calibration fitting window (iff using 'calibration_window' strategy).",
    )

    # TODO: validators
    @model_validator(mode="after")
    def check_strategy_args(self):
        """Ensure appropriate arguments are provided for each strategy."""
        match self.strategy:
            case PropEDStrategyEnum.surveillance_window:
                # Ensure ed_source present
                if not self.ed_source:
                    msg = "Prop ED strategy 'surveillance_window' requires field 'ed_source'"
                    raise ValueError(msg)
                # Ensure source present
                if not self.hosp_source:
                    msg = "Prop ED strategy 'surveillance_window' requires field 'hosp_source'"
                    raise ValueError(msg)
                # Ensure window specified and valid
                if not (bool(self.fit_start) and bool(self.fit_end)):
                    msg = "Prop ED strategy 'surveillance_window' requires fields 'fit_start' and 'fit_end'"
                    raise ValueError(msg)
                if not self.fit_start <= self.fit_end:
                    msg = f"Received fit_start: {self.fit_start} > fit_end: {self.fit_end}"
                    raise ValueError(msg)
                # Warn unused fields
                if self.num_fit_weeks is not None:
                    msg = "Received unused 'num_fit_weeks' with prop ED strategy 'surveillance_window'; ignoring."
                    logger.warning(msg)
                if self.transition_name is not None:
                    msg = "Received unused 'transition_name' with prop ED strategy 'surveillance_window'; ignoring."
                    logger.warning(msg)
            case PropEDStrategyEnum.calibration_window:
                # Ensure ed_source present
                if not self.ed_source:
                    msg = "Prop ED strategy 'calibration_window' requires field 'ed_source'"
                    raise ValueError(msg)
                # Ensure window specified
                if self.num_fit_weeks is None:
                    msg = "Prop ED strategy 'calibration_window' requires field 'num_fit_weeks'"
                    raise ValueError(msg)
                # Warn unused fields
                if bool(self.fit_start) or bool(self.fit_end):
                    msg = (
                        "Received unused 'fit_start' or 'fit_end' with prop ED strategy 'calibration_window'; ignoring."
                    )
                    logger.warning(msg)
                if self.hosp_source:
                    msg = "Received unused 'hosp_source' with prop ED strategy 'calibration_window'; ignoring."
                    logger.warning(msg)
                if self.transition_name is not None:
                    msg = "Received unused 'transition_name' with prop ED strategy 'calibration_window'; ignoring."
                    logger.warning(msg)
            case PropEDStrategyEnum.transition:
                # Ensure transition_name present
                if not self.transition_name:
                    msg = "Prop ED strategy 'transition' requires field 'transition_name'"
                    raise ValueError(msg)
                # Warn unused fields
                if self.ed_source is not None:
                    msg = "Received unused 'ed_source' with prop ED strategy 'transition'; ignoring."
                    logger.warning(msg)
                if self.hosp_source is not None:
                    msg = "Received unused 'hosp_source' with prop ED strategy 'transition'; ignoring."
                    logger.warning(msg)
                if bool(self.fit_start) or bool(self.fit_end):
                    msg = "Received unused 'fit_start' or 'fit_end' with prop ED strategy 'transition'; ignoring."
                    logger.warning(msg)
                if self.num_fit_weeks is not None:
                    msg = "Received unused 'num_fit_weeks' with prop ED strategy 'transition'; ignoring."
                    logger.warning(msg)
            case _:
                msg = f"Received invalid/unimplemented strategy {_}"
                raise ValueError(msg)
        return self


class FlusightHospitalizations(BaseModel):
    """Specifications for generating hospitalizations forecasts."""

    target: str = Field(
        default="wk inc flu hosp",
        description="Target name for the submission file (e.g., 'wk inc flu hosp').",
    )


class FlusightForecastOutput(BaseModel):
    """Specifications for outputs in flusight forecast hub format."""

    reference_date: date = Field(
        description="'YYYY-MM-DD' date to treat as reference date when creating horizons and target dates for submission file."
    )
    hospitalizations: FlusightHospitalizations | None = Field(
        default_factory=FlusightHospitalizations,
        description="Add 'wk inc flu hosp' quantile forecasts and rate-trend forecasts to submission file. Set to null to disable all hospitalization outputs, omit to enable.",
    )
    rate_trends_source: str | None = Field(
        None,
        description="Name of surveillance source from output.options.surveillance to use for rate-trend forecasts.",
    )
    prop_ed: FlusightPropED | None = Field(
        None,
        description="Add 'wk inc flu prop ed visits' target to submission file.",
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
        None,
        description="Filter surveillance to show only points >= this date (YYYY-MM-DD). Overrides surveillance_points if both set.",
    )
    xlabel_interval: str | None = Field(
        None,
        description="X-axis label interval as pandas offset string (e.g., 'W-SAT', '2W-SAT', 'MS'). None = auto (matplotlib default).",
    )


class QuantilesOutputConfig(BaseModel):
    """Configuration for a single quantile plot output."""

    type: QuantilesOutputTypeEnum = Field(description="Type of output to generate.")

    # Display options (for all output types)
    show_calibration: bool = Field(False, description="Show calibration period quantiles.")
    show_projection: bool = Field(True, description="Show projection period quantiles.")
    show_surveillance: bool = Field(True, description="Overlay surveillance observations.")
    show_fitting_window_line: bool = Field(True, description="Show fitting window vertical lines.")
    surveillance_source: str | None = Field(
        None,
        description="Name of surveillance source from surveillance dict. If None and surveillance dict has one entry, use that entry.",
    )

    # Filter settings (only for FILTERED and FULL types)
    surveillance_points: int | None = Field(
        None,
        description="Number of most recent surveillance points to display. None = all points. Not used for SIDE_BY_SIDE.",
    )
    surveillance_start_date: str | None = Field(
        None,
        description="Filter surveillance to show only points >= this date (YYYY-MM-DD). Overrides surveillance_points if both set. Not used for SIDE_BY_SIDE.",
    )
    horizon_max: int | None = Field(None, description="Override base horizon_max. None = use base config value.")
    xlabel_interval: str | None = Field(
        None,
        description="X-axis label interval as pandas offset string (e.g., 'W-SAT', '2W-SAT', 'MS'). None = auto. For SIDE_BY_SIDE, use panel configs instead.",
    )

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


class QuantilesPlotConfig(BaseModel):
    """Configuration for quantile ribbon plots."""

    single: list[str] | bool = Field(
        False,
        description="Create single plot per location (default disabled). Set true for all locations, or provide list of specific locations.",
    )
    grid: QuantilesGridConfig | bool = Field(
        default_factory=QuantilesGridConfig,
        description="Grid plot with all locations (default enabled). Set true to use default options, or set options in subfields.",
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
                spacing=0.3,
            ),
        ],
        description="List of output configurations to generate.",
    )

    # Shared settings
    quantiles: list[float] = Field(
        default_factory=get_quantile_ribbon_default,
        description="Quantile levels for ribbons (default enabled: 95% CrI + IQR + median).",
        validate_default=True,
    )
    horizon_max: int | None = Field(
        3, description="Base maximum forecast horizon (weeks ahead). Can be overridden per output."
    )

    # Shared styling (data source and colors)
    calibration: QuantilesCalibrationConfig | bool = Field(
        default_factory=QuantilesCalibrationConfig,
        description="Calibration period quantile ribbons (default enabled). Set true to use default options, or set options in subfields.",
    )
    projection: QuantilesProjectionConfig | bool = Field(
        default_factory=QuantilesProjectionConfig,
        description="Projection period quantile ribbons (default enabled). Set true to use default options, or set options in subfields.",
    )

    value_column: str = Field(
        "hospitalizations",
        description="Column name for projection quantiles to plot. Common values: 'hospitalizations', 'ed_signal', 'value'. Must match a transition name in output.quantiles.transitions.",
    )
    ylabel: str | None = Field(
        default=None,
        description="Y-axis label for quantile plots (e.g., 'Hospitalizations'). If None, no label is shown.",
    )

    @field_validator("grid")
    @classmethod
    def validate_grid(cls, v: QuantilesGridConfig | bool) -> QuantilesGridConfig | bool:
        """If passed True, use default factory."""
        if v is True:
            return QuantilesGridConfig()
        return v

    @field_validator("calibration")
    @classmethod
    def validate_calibration(cls, v: QuantilesCalibrationConfig | bool) -> QuantilesCalibrationConfig | bool:
        """If passed True, use default factory."""
        if v is True:
            return QuantilesCalibrationConfig()
        return v

    @field_validator("projection")
    @classmethod
    def validate_projection(cls, v: QuantilesProjectionConfig | bool) -> QuantilesProjectionConfig | bool:
        """If passed True, use default factory."""
        if v is True:
            return QuantilesProjectionConfig()
        return v

    @field_validator("quantiles")
    @classmethod
    def check_quantiles_valid(cls, v: list[float]) -> list[float]:
        """Ensure quantiles are in [0, 1]."""
        for q in v:
            if not 0.0 <= q <= 1.0:
                msg = f"Quantile {q} must be between 0 and 1"
                raise ValueError(msg)
        return v

    @field_validator("single")
    @classmethod
    def validate_single_plot_locations(cls, v: list[str]):
        """Validate each population name in the list."""
        if isinstance(v, bool):
            return v
        validated_populations = [validate_iso3166(population) for population in v]
        return validated_populations


class CategoricalPlotConfig(BaseModel):
    """Configuration for categorical forecast probability plots.

    Creates a vertical stack of 4 subplots (one per horizon 0-3) showing
    stacked bar charts of categorical forecast probabilities for rate-change trends.
    All panels share the same x-axis with labels shown only on the bottom panel.

    To enable categorical plots, include this section in your config.
    To disable, omit the section or set categorical: null.
    """

    categories: list[str] = Field(
        default_factory=lambda: ["large_decrease", "decrease", "stable", "increase", "large_increase"],
        description="Category names in display order (bottom to top in stacked bars).",
    )

    colors: list[str] = Field(
        default_factory=lambda: ["#476a6f", "#519e8a", "#b7c3f3", "#dd7596", "#cf1259"],
        description="Colors for categories (must match length of categories list).",
    )

    horizons: list[int] = Field(
        default_factory=lambda: [0, 1, 2, 3],
        description="Forecast horizons to display (0-3 for FluSight).",
    )

    figsize: tuple[float, float] | None = Field(
        None,
        description="Figure size (width, height). If None, defaults to (10, 4.5 * n_horizons).",
    )

    @field_validator("colors")
    @classmethod
    def validate_colors_match_categories(cls, v: list[str], info) -> list[str]:
        """Ensure colors list matches categories list length."""
        if "categories" in info.data and len(v) != len(info.data["categories"]):
            msg = f"colors list length ({len(v)}) must match categories list length ({len(info.data['categories'])})"
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
    categorical: CategoricalPlotConfig | None = Field(
        None,
        description="Categorical forecast probability plot settings (rate-change trends). Set to null or omit to disable.",
    )


class OutputOptions(BaseModel):
    """Shared options for output configuration."""

    surveillance: dict[str, ObservedValuesConfig] | None = Field(
        None,
        description="Named surveillance data sources. Keys are source names, values are ObservedValuesConfig.",
    )


class OutputConfiguration(BaseModel):
    """Output configuration."""

    meta: Meta | None = Field(None, description="General metadata.")

    tabular_output_types: list[TabularOutputTypeEnum] | None = Field(
        default_factory=get_default_tabular_output,
        description="Output formats to create for all requested tabular outputs.",
    )

    # Shared options
    options: OutputOptions | None = Field(
        None,
        description="Shared options for outputs (e.g., surveillance data sources).",
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

    @model_validator(mode="after")
    def validate_surveillance_references(self):
        """Ensure all surveillance_source references point to valid sources."""
        # If no surveillance sources defined, check if any are referenced
        if not self.options or not self.options.surveillance:
            errors = []

            # Check flusight_format
            if self.flusight_format:
                if self.flusight_format.rate_trends_source:
                    errors.append(
                        f"flusight_format.rate_trends_source='{self.flusight_format.rate_trends_source}' "
                        "but no surveillance sources defined in output.options.surveillance"
                    )
                if self.flusight_format.prop_ed and self.flusight_format.prop_ed.strategy != "transition":
                    errors.append(
                        f"flusight_format.prop_ed with strategy '{self.flusight_format.prop_ed.strategy}' "
                        "but no surveillance sources defined in output.options.surveillance"
                    )

            # Check plots.quantiles.outputs
            if self.plots and self.plots.quantiles:
                for i, output in enumerate(self.plots.quantiles.outputs):
                    if output.surveillance_source:
                        errors.append(
                            f"plots.quantiles.outputs[{i}].surveillance_source='{output.surveillance_source}' "
                            "but no surveillance sources defined in output.options.surveillance"
                        )

            if errors:
                msg = "Surveillance source references without defined sources:\n" + "\n".join(
                    f"  - {e}" for e in errors
                )
                raise ValueError(msg)

            return self

        # Surveillance sources defined - validate references exist
        available_sources = set(self.options.surveillance.keys())
        errors = []

        # Check flusight_format
        if self.flusight_format:
            if (
                self.flusight_format.rate_trends_source
                and self.flusight_format.rate_trends_source not in available_sources
            ):
                errors.append(
                    f"flusight_format.rate_trends_source='{self.flusight_format.rate_trends_source}' "
                    f"not found in surveillance sources: {available_sources}"
                )
            if self.flusight_format.prop_ed:
                if (
                    self.flusight_format.prop_ed.ed_source
                    and self.flusight_format.prop_ed.ed_source not in available_sources
                ):
                    errors.append(
                        f"flusight_format.prop_ed.ed_source='{self.flusight_format.prop_ed.ed_source}' "
                        f"not found in surveillance sources: {available_sources}"
                    )
                if (
                    self.flusight_format.prop_ed.hosp_source
                    and self.flusight_format.prop_ed.hosp_source not in available_sources
                ):
                    errors.append(
                        f"flusight_format.prop_ed.hosp_source='{self.flusight_format.prop_ed.hosp_source}' "
                        f"not found in surveillance sources: {available_sources}"
                    )

        # Check plots.quantiles.outputs
        if self.plots and self.plots.quantiles:
            for i, output in enumerate(self.plots.quantiles.outputs):
                if output.surveillance_source and output.surveillance_source not in available_sources:
                    errors.append(
                        f"plots.quantiles.outputs[{i}].surveillance_source='{output.surveillance_source}' "
                        f"not found in surveillance sources: {available_sources}"
                    )

        if errors:
            msg = "Invalid surveillance source references:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ValueError(msg)

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
