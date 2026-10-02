"""Plot generation orchestration for calibration/projection outputs.

This module contains high-level functions that orchestrate the creation of visualization outputs
by loading data, collecting quantiles/posteriors, calling visualization primitives, and packaging
outputs as OutputObject instances.
"""

import logging
import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Literal

import matplotlib.pyplot as plt
import pandas as pd

from ..builders.utils import get_data_in_location
from ..schema.dispatcher import CalibrationOutput
from ..schema.output import (
    ObservedValuesConfig,
    OutputObject,
    PlotsConfig,
    QuantilesOutputConfig,
    QuantilesOutputTypeEnum,
    QuantilesPlotConfig,
)
from ..utils import convert_location_name_format
from ..utils.quantiles import get_calibration_quantiles, get_projection_quantiles
from .core import (
    figure_to_output_object,
    format_location_name,
    plot_calibration_projection,
    plot_calibration_projection_grid,
    plot_calibration_projection_sidebyside,
    plot_categorical_stacked_bars_multihorizon,
    plot_posterior_histogram,
    plot_posterior_histogram_grid,
    sort_locations_by_state,
)

logger = logging.getLogger(__name__)

# Columns that are metadata/identifiers and should be excluded when extracting parameter names
POSTERIOR_METADATA_COLUMNS = {"sim_id", "location", "population", "primary_id", "seed"}


def _check_incomplete_generations(calibration: CalibrationOutput) -> str | None:
    """Return a note string if fewer generations completed than requested.

    ABC-SMC from epydemix can return fewer generations than requested if it reaches the stopping criterion (e.g., max_time) before completing all generations. It discards the incomplete generation and returns the previously completed generation as the final result (CalibrationResults).

    This function checks if the number of completed generations in the results is fewer than the number requested in the calibration strategy, and if so, returns a note string to indicate this. If all generations completed or if the necessary information is unavailable, it returns None.

    Parameters
    ----------
    calibration : CalibrationOutput
        Calibration output with optional calibration_strategy and results.

    Returns
    -------
    str or None
        A note string if calibration completed fewer generations than requested,
        or None if all generations completed or info is unavailable.
    """
    # Check number of generations requested
    strategy = getattr(calibration, "calibration_strategy", None)
    requested = strategy.options.get("num_generations") if strategy else None
    if requested is None or calibration.results is None:
        return None

    # Check number of completed generations
    posterior_dists = getattr(calibration.results, "posterior_distributions", None)
    if posterior_dists is None:
        return None
    completed = len(posterior_dists)

    # Compare
    if completed < requested:
        return f"Completed {completed} of {requested} requested generations"
    return None


def _format_plot_notes(notes: list[str]) -> tuple[str, str]:
    """Return (title_suffix, footnote_text) from a list of notes.

    Parameters
    ----------
    notes : list of str
        List of note strings to format.

    Returns
    -------
    tuple of (str, str)
        (title_suffix, footnote_text). Empty strings if no notes.
    """
    if not notes:
        return ("", "")
    return ("*", "* " + "; ".join(notes))


def _add_footnote(fig: plt.Figure, footnote: str) -> None:
    """Add footnote text to bottom of figure if non-empty.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to add footnote to.
    footnote : str
        Footnote text. If empty, no action is taken.
    """
    if footnote:
        fig.subplots_adjust(bottom=fig.subplotpars.bottom + 0.03)
        fig.text(0.5, 0.01, footnote, ha="center", fontsize=8, style="italic")


def _fetch_quantiles_for_location(
    calibration: CalibrationOutput,
    plots_config: PlotsConfig,
    needs_calibration: bool,
    needs_projection: bool,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Fetch calibration and projection quantiles for a single location.

    Parameters
    ----------
    calibration : CalibrationOutput
        Calibration results for one location.
    plots_config : PlotsConfig
        Plot configuration with quantile levels.
    needs_calibration : bool
        Whether calibration quantiles are requested.
    needs_projection : bool
        Whether projection quantiles are requested.

    Returns
    -------
    tuple
        (cal_quant, proj_quant) where either can be None if not requested or retrieval fails.
    """
    cal_quant = None
    if needs_calibration:
        try:
            cal_trajs = calibration.results.get_selected_trajectories()
            cal_dates = cal_trajs[0].get("date") if cal_trajs else None
            cal_quant = get_calibration_quantiles(
                calibration.results,
                quantiles=plots_config.quantiles.quantiles,
                dates=cal_dates,
                variables=["data"],
                ignore_nan=True,
            )
        except (ValueError, AttributeError, TypeError, IndexError) as e:
            logger.warning("Failed to get calibration quantiles for %s: %s", calibration.population, e)

    proj_quant = None
    if needs_projection:
        try:
            proj_sims = calibration.results.projections.get("baseline", [])
            proj_dates = proj_sims[0].get("date") if proj_sims else None
            proj_quant = get_projection_quantiles(
                calibration.results,
                quantiles=plots_config.quantiles.quantiles,
                dates=proj_dates,
                variables=[plots_config.quantiles.value_column],
                ignore_nan=True,
            )
        except (ValueError, AttributeError, TypeError, IndexError) as e:
            logger.warning("Failed to get projection quantiles for %s: %s", calibration.population, e)

    return cal_quant, proj_quant


def _select_surveillance(
    surveillance: pd.DataFrame,
    location: str,
    surveillance_config: ObservedValuesConfig,
) -> pd.DataFrame | None:
    """
    Select one location's surveillance rows as a date/value frame sorted by date.

    Parameters
    ----------
    surveillance : pd.DataFrame
        Raw surveillance data for all locations.
    location : str
        Location (population name) to select.
    surveillance_config : ObservedValuesConfig
        Configuration for the surveillance source (location, date and value columns).

    Returns
    -------
    pd.DataFrame or None
        Frame with ``date`` and ``value`` columns, or None if the location has no rows.
    """
    location_rows = get_data_in_location(
        surveillance,
        location,
        surveillance_config.location_column,
        surveillance_config.location_format,
    )
    if location_rows.empty:
        return None
    # Select columns before renaming to avoid duplicates if source data already has 'value' column
    return (
        location_rows[[surveillance_config.date_column, surveillance_config.value_column]]
        .rename(columns={surveillance_config.date_column: "date", surveillance_config.value_column: "value"})
        .sort_values("date")
    )


def _rename_value_column(df: pd.DataFrame | None, old_name: str) -> pd.DataFrame | None:
    """
    Select the specified value column and rename it to 'value', keeping only metadata columns.

    This function filters the DataFrame to keep only metadata columns (date, quantile, population)
    plus the specified value column, then renames that column to 'value'.

    NOTE: The rename to 'value' is necessary because plot_calibration_projection() uses a single
    value_col parameter for both calibration and projection quantiles. Since calibration uses
    'data' and projection uses configurable columns (e.g., 'ed_signal'), we need a common name.

    The column selection is critical to avoid duplicate column names. CalibrationResults.get_projection_quantiles()
    returns 269 columns including a pre-existing 'value' column (from epydemix) plus all transitions
    (ed_signal, hospitalizations, etc.) and compartments. Simply renaming without filtering would
    create two 'value' columns, causing pivot() to fail with "Data must be 1-dimensional, got ndarray
    of shape (N, 2)".

    Parameters
    ----------
    df : pd.DataFrame or None
        DataFrame containing quantile data with multiple value columns.
    old_name : str
        Column name to select and rename to 'value'.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with only metadata columns plus the selected value column renamed to 'value',
        or None if input is None.

    Raises
    ------
    ValueError
        If df is not None and old_name is not found in columns.
    """
    if df is None:
        return None

    if old_name in df.columns:
        # Select only metadata columns plus the specified value column
        # This excludes the pre-existing 'value' column and all other transitions/compartments
        metadata_cols = ["date", "quantile", "population"]
        cols_to_keep = [c for c in metadata_cols if c in df.columns] + [old_name]
        return df[cols_to_keep].rename(columns={old_name: "value"})

    # Column not found - provide helpful error message
    metadata_cols_for_error = {"date", "location", "quantile", "population"}
    available_value_cols = [c for c in df.columns if c not in metadata_cols_for_error]

    msg = (
        f"Column '{old_name}' not found in quantiles DataFrame. "
        f"Available value columns: {available_value_cols}. "
        f"Check output.plots.quantiles.value_column in your config matches "
        f"a transition in output.quantiles.transitions."
    )
    raise ValueError(msg)


def _clip_to_start(df: pd.DataFrame | None, start: date) -> pd.DataFrame | None:
    """
    Keep the rows of ``df`` dated on or after ``start``.

    Parameters
    ----------
    df : pd.DataFrame or None
        DataFrame with a "date" column (e.g. calibration or projection quantiles).
    start : date
        First date to keep.

    Returns
    -------
    pd.DataFrame or None
        Rows of ``df`` where date >= ``start`` (possibly empty), or None if ``df`` is None.
    """
    if df is None:
        return None
    return df[pd.to_datetime(df["date"]).dt.date >= start]


def _clip_surveillance(
    surv: pd.DataFrame | None,
    *,
    surveillance_start_date: str | date | None = None,
    surveillance_points: int | None = None,
    reference_date: date | None = None,
) -> pd.DataFrame | None:
    """
    Filter surveillance data by a start date or by keeping the last N points.

    When both parameters are provided, ``surveillance_start_date`` takes precedence.

    Parameters
    ----------
    surv : pd.DataFrame or None
        Surveillance DataFrame with a "date" column.
    surveillance_start_date : str, date or None
        If given, keep only rows with date >= this value (parsed via ``pd.to_datetime``).
    surveillance_points : int or None
        If given (and ``surveillance_start_date`` is None), keep the last N rows
        *before* ``reference_date``, plus all rows after it.
    reference_date : date or None
        The reference date used to split before/after when ``surveillance_points``
        is specified. Required when ``surveillance_points`` is not None.

    Returns
    -------
    pd.DataFrame or None
        Filtered surveillance data, or the input unchanged if no filter applies.
    """
    if surv is None or surv.empty:
        return surv
    if surveillance_start_date is not None:
        start = pd.to_datetime(surveillance_start_date).date()
        dates = pd.to_datetime(surv["date"]).dt.date
        return surv[dates >= start]
    if surveillance_points is not None:
        surv = surv.sort_values("date")
        dates = pd.to_datetime(surv["date"]).dt.date
        if reference_date is not None:
            before = surv[dates <= reference_date].tail(surveillance_points)
            after = surv[dates > reference_date]
            return pd.concat([before, after])
        return surv.tail(surveillance_points)
    return surv


def _clip_to_horizon(
    proj: pd.DataFrame | None,
    horizon_max: int | None,
    reference_date: date,
) -> pd.DataFrame | None:
    """
    Clip projection quantiles to a maximum forecast horizon.

    Keeps only rows whose date is at most ``reference_date + horizon_max`` weeks.

    Parameters
    ----------
    proj : pd.DataFrame or None
        Projection quantiles DataFrame with a "date" column.
    horizon_max : int or None
        Maximum number of weeks past ``reference_date`` to keep. If None, no clipping is applied.
    reference_date : date
        The reference date from which the horizon is measured.

    Returns
    -------
    pd.DataFrame or None
        Clipped projection data, or ``proj`` unchanged when ``horizon_max`` is None.
    """
    if proj is None or horizon_max is None:
        return proj
    dates = pd.to_datetime(proj["date"]).dt.date
    end = reference_date + timedelta(weeks=horizon_max)
    return proj[dates.values <= end]


@dataclass(frozen=True)
class ResolvedPanelSettings:
    """Final settings for one panel of an output, after all overrides are resolved.

    A panel here is the full or filtered part of an output, matching ``full_panel`` / ``filtered_panel``
    in the schema. In grid plots the same settings apply to every location's panel.
    """

    view: Literal["full", "filtered"]
    surveillance_source: str | None  # resolved name; None = no surveillance
    surveillance_points: int | None
    surveillance_start_date: date | None
    horizon_max: int | None  # effective value; None = unlimited
    xlabel_interval: str | None
    show_calibration: bool
    show_projection: bool
    show_surveillance: bool
    show_fitting_window_line: bool
    calibration_color: str
    projection_color: str


def resolve_panel_settings(
    quantiles: QuantilesPlotConfig,
    output: QuantilesOutputConfig,
    loaded_source_names: list[str],
) -> list[ResolvedPanelSettings]:
    """
    Resolve the layered plot settings (base -> output -> panel) for one output.

    Precedence is panel > output > base > default, where ``None`` means "inherit from the next level".
    ``surveillance_points`` and ``surveillance_start_date`` are inherited as a pair: if a panel sets either,
    neither is taken from the output.

    Parameters
    ----------
    quantiles : QuantilesPlotConfig
        Base quantile plot configuration.
    output : QuantilesOutputConfig
        The output being plotted.
    loaded_source_names : list of str
        Names of the surveillance sources that were loaded successfully.

    Returns
    -------
    list of ResolvedPanelSettings
        One entry for filtered/full outputs, ``[full, filtered]`` for side_by_side.
    """
    if output.surveillance_source in loaded_source_names:
        surveillance_source = output.surveillance_source
    elif len(loaded_source_names) == 1:
        surveillance_source = loaded_source_names[0]
    else:
        surveillance_source = None

    horizon_max = output.horizon_max if output.horizon_max is not None else quantiles.horizon_max

    if output.type == QuantilesOutputTypeEnum.SIDE_BY_SIDE:
        views = [("full", output.full_panel), ("filtered", output.filtered_panel)]
    else:
        views = [(output.type.value, None)]

    settings = []
    for view, panel in views:
        panel_sets_limits = panel is not None and (
            panel.surveillance_points is not None or panel.surveillance_start_date is not None
        )
        limits = panel if panel_sets_limits else output
        xlabel_interval = output.xlabel_interval
        if panel is not None and panel.xlabel_interval is not None:
            xlabel_interval = panel.xlabel_interval
        start_date = limits.surveillance_start_date
        settings.append(
            ResolvedPanelSettings(
                view=view,
                surveillance_source=surveillance_source,
                surveillance_points=limits.surveillance_points,
                surveillance_start_date=None if start_date is None else pd.to_datetime(start_date).date(),
                horizon_max=horizon_max,
                xlabel_interval=xlabel_interval,
                show_calibration=output.show_calibration,
                show_projection=output.show_projection,
                show_surveillance=output.show_surveillance,
                show_fitting_window_line=output.show_fitting_window_line,
                calibration_color=quantiles.calibration.color,
                projection_color=quantiles.projection.color,
            )
        )
    return settings


@dataclass(frozen=True)
class LocationPlotData:
    """Quantiles, surveillance and notes for one location, collected once before plotting."""

    location: str
    calibration_quantiles: pd.DataFrame | None  # value column renamed to "value"
    projection_quantiles: pd.DataFrame | None  # untrimmed, value column renamed to "value"
    surveillance: dict[str, pd.DataFrame]  # source name -> this location's rows as date/value
    fitting_window_start: date | None  # from untrimmed calibration data, incl. the median-only fallback
    fitting_window_end: date | None
    title_suffix: str
    footnote: str


@dataclass(frozen=True)
class PanelPlotData:
    """Data for one location and one panel. ``None`` = layer not drawn; an empty frame draws nothing."""

    calibration: pd.DataFrame | None
    projection: pd.DataFrame | None
    surveillance: pd.DataFrame | None


def prepare_panel_plot_data(
    location_plot_data: LocationPlotData,
    panel_settings: ResolvedPanelSettings,
    reference_date: date,
) -> PanelPlotData:
    """
    Build the frames for one location and one panel.

    Steps, in order:

    1. Surveillance from the resolved source, only if the panel shows it.
    2. Surveillance range: ``surveillance_start_date`` if set, else the last ``surveillance_points`` rows on or
       before ``reference_date`` (all later rows are kept). With neither, the full view keeps everything and the
       filtered view starts at the first projection date if this panel shows the projection, else at the first
       calibration date if it shows the calibration.
    3. Projection is clipped to ``reference_date + horizon_max`` weeks.
    4. Filtered view: calibration and projection are clipped to the first visible surveillance date. If no
       surveillance is left visible, they are clipped to where it would have started instead
       (``surveillance_start_date``, or the start above without limits).
    5. Hidden layers are returned as ``None``. Frames that become empty stay empty.

    Parameters
    ----------
    location_plot_data : LocationPlotData
        Collected data for the location.
    panel_settings : ResolvedPanelSettings
        Resolved settings for the panel.
    reference_date : date
        Forecast reference date.

    Returns
    -------
    PanelPlotData
        Frames to draw.
    """
    settings = panel_settings
    calibration = location_plot_data.calibration_quantiles
    projection = location_plot_data.projection_quantiles

    surveillance = None
    if settings.show_surveillance and settings.surveillance_source is not None:
        surveillance = location_plot_data.surveillance.get(settings.surveillance_source)

    filtered_start = None  # None = the filtered view is not clipped
    if surveillance is not None:
        if settings.surveillance_start_date is not None or settings.surveillance_points is not None:
            surveillance = _clip_surveillance(
                surveillance,
                surveillance_start_date=settings.surveillance_start_date,
                surveillance_points=settings.surveillance_points,
                reference_date=reference_date,
            )
            filtered_start = settings.surveillance_start_date
        elif settings.view == "filtered":
            # Only this panel's shown layers count, so other outputs loading the projection do not move the start
            shown = [
                frame
                for frame, show in ((projection, settings.show_projection), (calibration, settings.show_calibration))
                if show and frame is not None
            ]
            if shown:
                filtered_start = pd.to_datetime(shown[0]["date"]).min().date()
                surveillance = _clip_to_start(surveillance, filtered_start)
        if not surveillance.empty:
            filtered_start = pd.to_datetime(surveillance["date"]).min().date()

    projection = _clip_to_horizon(projection, settings.horizon_max, reference_date)

    if settings.view == "filtered" and filtered_start is not None:
        calibration = _clip_to_start(calibration, filtered_start)
        projection = _clip_to_start(projection, filtered_start)

    return PanelPlotData(
        calibration=calibration if settings.show_calibration else None,
        projection=projection if settings.show_projection else None,
        surveillance=surveillance,
    )


def _load_surveillance_sources(
    surveillance_sources: dict[str, ObservedValuesConfig] | None,
    outputs: list[QuantilesOutputConfig],
) -> dict[str, dict[str, Any]]:
    """
    Load surveillance CSV files for all configured sources.

    Skips loading entirely if no output requires surveillance data or if no sources are configured.

    Parameters
    ----------
    surveillance_sources : dict[str, ObservedValuesConfig] or None
        Mapping of source name to its configuration (including ``data_path``).
    outputs : list[QuantilesOutputConfig]
        Output configurations; loading is skipped if none have ``show_surveillance`` enabled.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping of source name to ``{"data": pd.DataFrame, "config": ObservedValuesConfig}``.
        Empty dict if no sources are needed or available.
    """
    if not any(o.show_surveillance for o in outputs) or not surveillance_sources:
        return {}
    result: dict[str, dict[str, Any]] = {}
    for name, config in surveillance_sources.items():
        try:
            result[name] = {"data": pd.read_csv(config.data_path), "config": config}
        except Exception as e:
            logger.warning("Failed to load surveillance data from %s: %s", config.data_path, e)
    return result


def _compute_fitting_window(
    calibration: CalibrationOutput,
    cal_quant: pd.DataFrame | None,
) -> tuple[date | None, date | None]:
    """
    Compute the fitting window date range from quantile dates or the first trajectory.

    Uses the provided ``cal_quant`` if available; otherwise reads trajectory dates
    without stacking samples or computing their median.

    Parameters
    ----------
    calibration : CalibrationOutput
        Calibration output for one location, used as fallback source for dates.
    cal_quant : pd.DataFrame or None
        Pre-fetched calibration quantiles. If None, trajectory dates are used.

    Returns
    -------
    tuple[date or None, date or None]
        (start, end) of the fitting window, or (None, None) if dates are unavailable.
    """
    if cal_quant is not None and "date" in cal_quant.columns:
        dates = pd.to_datetime(cal_quant["date"]).dt.date
        return dates.min(), dates.max()
    if cal_quant is None:
        try:
            cal_trajs = calibration.results.get_selected_trajectories()
            cal_dates = cal_trajs[0].get("date") if cal_trajs else None
            if cal_dates is not None and len(cal_dates) > 0:
                dates = pd.to_datetime(cal_dates).date
                return dates.min(), dates.max()
        except (ValueError, AttributeError, TypeError, IndexError) as e:
            logger.warning(
                "Failed to read fitting window dates for %s: %s",
                calibration.population,
                e,
            )

    return None, None


def _package_figure_outputs(
    fig: plt.Figure,
    name: str,
    plots_config: PlotsConfig,
) -> list[OutputObject]:
    """
    Convert a matplotlib figure to OutputObject instances for each configured format.

    Parameters
    ----------
    fig : plt.Figure
        The matplotlib figure to serialize.
    name : str
        Base name for the output (used in filenames).
    plots_config : PlotsConfig
        Configuration providing ``figure_output_types`` (e.g. png, svg) and ``dpi``.

    Returns
    -------
    list[OutputObject]
        One OutputObject per configured figure output type.
    """
    return [
        figure_to_output_object(fig, name, output_type, plots_config.dpi)
        for output_type in plots_config.figure_output_types
    ]


def _collect_location_plot_data(
    calibration: CalibrationOutput,
    plots_config: PlotsConfig,
    surveillance_data: dict[str, dict[str, Any]],
    *,
    needs_calibration: bool,
    needs_projection: bool,
    needs_fitting_window: bool,
) -> LocationPlotData:
    """
    Collect quantiles, surveillance, fitting window and notes for one location.

    Parameters
    ----------
    calibration : CalibrationOutput
        Calibration results for one location.
    plots_config : PlotsConfig
        Plot configuration (quantile levels and projection value column).
    surveillance_data : dict[str, dict[str, Any]]
        Loaded surveillance sources from ``_load_surveillance_sources``.
    needs_calibration, needs_projection, needs_fitting_window : bool
        Whether any output needs calibration quantiles, projection quantiles or the fitting window.

    Returns
    -------
    LocationPlotData
        Collected data for the location.
    """
    location = calibration.population
    notes = []
    if note := _check_incomplete_generations(calibration):
        notes.append(note)
    title_suffix, footnote = _format_plot_notes(notes)

    calibration_quantiles, projection_quantiles = _fetch_quantiles_for_location(
        calibration, plots_config, needs_calibration, needs_projection
    )

    fitting_window_start, fitting_window_end = None, None
    if needs_fitting_window:
        fitting_window_start, fitting_window_end = _compute_fitting_window(calibration, calibration_quantiles)

    surveillance = {}
    for source_name, source in surveillance_data.items():
        try:
            location_surveillance = _select_surveillance(source["data"], location, source["config"])
        except Exception as e:
            logger.warning(
                "Failed to prepare surveillance '%s' for location %s: %s", source_name, location, e, exc_info=True
            )
            continue
        if location_surveillance is not None:
            surveillance[source_name] = location_surveillance

    return LocationPlotData(
        location=location,
        calibration_quantiles=_rename_value_column(calibration_quantiles, "data"),
        projection_quantiles=_rename_value_column(projection_quantiles, plots_config.quantiles.value_column),
        surveillance=surveillance,
        fitting_window_start=fitting_window_start,
        fitting_window_end=fitting_window_end,
        title_suffix=title_suffix,
        footnote=footnote,
    )


def _plot_location_panels(
    location_plot_data: LocationPlotData,
    panel_settings: list[ResolvedPanelSettings],
    panels: list[PanelPlotData],
    output_config: QuantilesOutputConfig,
    *,
    title: str,
    ylabel: str | None,
    axes: tuple[plt.Axes, plt.Axes] | None = None,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Draw one location's panels: a single plot for filtered/full, a (full, filtered) pair for side_by_side.

    Parameters
    ----------
    location_plot_data : LocationPlotData
        Collected data for the location (fitting window).
    panel_settings : list of ResolvedPanelSettings
        Resolved settings, one per panel.
    panels : list of PanelPlotData
        Prepared frames, one per panel.
    output_config : QuantilesOutputConfig
        Output configuration (side_by_side figsize and spacing).
    title : str
        Panel title.
    ylabel : str or None
        Y-axis label for the (left) panel.
    axes : tuple of plt.Axes, optional
        Existing (full, filtered) axes for side_by_side panels in a grid.

    Returns
    -------
    tuple
        (fig, axes) with one axis per panel.
    """
    settings = panel_settings[0]
    show_fitting_window = settings.show_fitting_window_line
    common = {
        "value_col": "value",
        "calibration_color": settings.calibration_color,
        "projection_color": settings.projection_color,
        "fitting_window_start": location_plot_data.fitting_window_start if show_fitting_window else None,
        "fitting_window_end": location_plot_data.fitting_window_end if show_fitting_window else None,
        "title": title,
        "ylabel": ylabel,
    }
    if len(panels) == 1:
        (panel,) = panels
        fig, ax = plot_calibration_projection(
            calibration_quantiles=panel.calibration,
            projection_quantiles=panel.projection,
            df_surveillance=panel.surveillance,
            xlabel_interval=settings.xlabel_interval,
            **common,
        )
        return fig, [ax]

    full, filtered = panels
    ax_full, ax_filtered = axes if axes is not None else (None, None)
    fig, (ax_full, ax_filtered) = plot_calibration_projection_sidebyside(
        calibration_quantiles=full.calibration,
        calibration_quantiles_filtered=filtered.calibration,
        projection_quantiles_full=full.projection,
        projection_quantiles_filtered=filtered.projection,
        surveillance_full=full.surveillance,
        surveillance_filtered=filtered.surveillance,
        xlabel_interval_full=panel_settings[0].xlabel_interval,
        xlabel_interval_filtered=panel_settings[1].xlabel_interval,
        figsize=output_config.figsize,
        spacing=output_config.spacing,
        ax_full=ax_full,
        ax_filtered=ax_filtered,
        **common,
    )
    return fig, [ax_full, ax_filtered]


def get_locations_to_plot(calibrations: list[CalibrationOutput], single_config: bool | list[str]) -> set[str]:
    """
    Get set of locations to plot based on config.

    Parameters
    ----------
    calibrations : list[CalibrationOutput]
        List of calibration outputs
    single_config : bool or list of str
        If True, return all locations. If list, return those specific locations. Entries may be in any
        location format (e.g. ISO "US-CA" or epydemix "United_States__California").

    Returns
    -------
    set of str
        Set of location names to plot, as they appear in `calibration.population`.
    """
    if single_config is True:
        return {cal.population for cal in calibrations}
    if isinstance(single_config, list):
        requested = {_to_epydemix_population_name(location) for location in single_config}
        return {cal.population for cal in calibrations if _to_epydemix_population_name(cal.population) in requested}
    return set()


def _quantile_plot_needs(outputs: list[QuantilesOutputConfig]) -> dict[str, bool]:
    """Which data any output needs, as keyword arguments for ``_collect_location_plot_data``."""
    return {
        "needs_calibration": any(output.show_calibration for output in outputs),
        "needs_projection": any(output.show_projection for output in outputs),
        "needs_fitting_window": any(output.show_fitting_window_line for output in outputs),
    }


def _to_epydemix_population_name(location: str) -> str:
    """Convert a location in any known format to its epydemix population name, or return it unchanged."""
    try:
        return convert_location_name_format(location, "epydemix_population")
    except (AssertionError, ValueError):
        return location


def generate_single_quantile_plots(
    calibrations: list[CalibrationOutput],
    plots_config: PlotsConfig,
    out_dict: dict[str, list[OutputObject]],
    surveillance_sources: dict[str, ObservedValuesConfig] | None = None,
) -> None:
    """
    Generate individual quantile plots for each location.

    Creates separate calibration/projection quantile plots for each location specified in the plots
    configuration. Each plot can optionally include calibration quantiles, projection quantiles,
    surveillance data, and fitting window lines.

    Parameters
    ----------
    calibrations : list[CalibrationOutput]
        List of calibration outputs containing results for each location. Failed projections are expected
        to be filtered already (as done by `generate_calibration_outputs()`).
    plots_config : PlotsConfig
        Configuration object specifying plot settings (quantiles, colors, surveillance data, etc.)
    out_dict : dict[str, list[OutputObject]]
        Dictionary to store generated plot outputs. Modified in-place by adding entries with keys
        like "quantiles_{location}_{type}" mapping to lists of OutputObject instances.
    surveillance_sources : dict[str, ObservedValuesConfig] or None, optional
        Surveillance sources that outputs can select with ``surveillance_source``.

    Returns
    -------
    None
        Modifies out_dict in-place by adding quantile plot outputs.
    """
    if not plots_config.quantiles.single:
        return

    locations = get_locations_to_plot(calibrations, plots_config.quantiles.single)
    logger.info("Generating single-location quantile plots for %d locations", len(locations))

    quantiles_config = plots_config.quantiles
    surveillance_data = _load_surveillance_sources(surveillance_sources, quantiles_config.outputs)
    output_settings = [
        (output_config, resolve_panel_settings(quantiles_config, output_config, list(surveillance_data)))
        for output_config in quantiles_config.outputs
    ]

    for calibration in calibrations:
        if calibration.population not in locations:
            continue
        location = calibration.population

        try:
            location_plot_data = _collect_location_plot_data(
                calibration, plots_config, surveillance_data, **_quantile_plot_needs(quantiles_config.outputs)
            )
        except Exception as e:
            logger.warning("Failed to process location %s for quantile plots: %s", location, e, exc_info=True)
            continue

        for output_config, panel_settings in output_settings:
            output_name = f"quantiles_{location}_{output_config.type.value}"
            logger.info("    Creating %s plot for %s", output_config.type.value, location)
            try:
                panels = [
                    prepare_panel_plot_data(location_plot_data, settings, plots_config.reference_date)
                    for settings in panel_settings
                ]
                fig, axes = _plot_location_panels(
                    location_plot_data,
                    panel_settings,
                    panels,
                    output_config,
                    title=format_location_name(location),
                    ylabel=quantiles_config.ylabel,
                )

                # Add generation notice to plot title and footnote
                if location_plot_data.title_suffix:
                    axes[0].set_title(axes[0].get_title() + location_plot_data.title_suffix)
                    _add_footnote(fig, location_plot_data.footnote)

                out_dict[output_name] = _package_figure_outputs(fig, output_name, plots_config)
                plt.close(fig)
            except Exception as e:
                logger.warning(
                    "Failed to create %s quantile plot for %s: %s",
                    output_config.type.value,
                    location,
                    e,
                    exc_info=True,
                )


def _plot_quantile_grid(
    location_plot_data: dict[str, LocationPlotData],
    settings: ResolvedPanelSettings,
    panels: dict[str, PanelPlotData],
    plots_config: PlotsConfig,
) -> plt.Figure:
    """Draw a filtered or full grid with one panel per location."""

    def by_location(layer: str) -> dict[str, pd.DataFrame] | None:
        frames = {location: getattr(panel, layer) for location, panel in panels.items()}
        return {location: frame for location, frame in frames.items() if frame is not None} or None

    fitting_window_starts = fitting_window_ends = None
    if settings.show_fitting_window_line:
        with_window = {
            location: data for location, data in location_plot_data.items() if data.fitting_window_start is not None
        }
        fitting_window_starts = {location: data.fitting_window_start for location, data in with_window.items()}
        fitting_window_ends = {location: data.fitting_window_end for location, data in with_window.items()}

    fig, axes = plot_calibration_projection_grid(
        location_calibration_quantiles=by_location("calibration"),
        location_projection_quantiles=by_location("projection"),
        value_col="value",
        calibration_color=settings.calibration_color,
        projection_color=settings.projection_color,
        location_surveillance=by_location("surveillance"),
        location_fitting_window_starts=fitting_window_starts,
        location_fitting_window_ends=fitting_window_ends,
        panels_per_row=plots_config.quantiles.grid.panels_per_row,
        ylabel=plots_config.quantiles.ylabel,
        xlabel_interval=settings.xlabel_interval,
        suptitle=plots_config.quantiles.suptitle,
    )

    # Add generation notice to grid panel titles
    grid_footnotes = set()
    for ax in axes.flat:
        title = ax.get_title()
        if not title:
            continue
        for location, data in location_plot_data.items():
            if data.title_suffix and title == format_location_name(location):
                ax.set_title(title + data.title_suffix)
                grid_footnotes.add(data.footnote)
                break
    if grid_footnotes:
        _add_footnote(fig, "; ".join(sorted(grid_footnotes)))
    return fig


def _plot_quantile_sidebyside_grid(
    location_plot_data: dict[str, LocationPlotData],
    panel_settings: list[ResolvedPanelSettings],
    panels: dict[str, list[PanelPlotData]],
    plots_config: PlotsConfig,
    output_config: QuantilesOutputConfig,
) -> plt.Figure:
    """Draw a side_by_side grid where each location gets a (full, filtered) pair of panels."""
    locations = sort_locations_by_state(
        location
        for location, location_panels in panels.items()
        if any(panel.calibration is not None or panel.projection is not None for panel in location_panels)
    )
    if not locations:
        msg = "No locations to plot"
        raise ValueError(msg)

    n_locations = len(locations)
    ncols = plots_config.quantiles.grid.panels_per_row
    pairs_per_row = ncols // 2  # Each location needs 2 panels
    nrows = math.ceil(n_locations / pairs_per_row)

    figsize = output_config.figsize or (4 * ncols, 3.6 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for pair_index, location in enumerate(locations):
        row = pair_index // pairs_per_row
        col_start = (pair_index % pairs_per_row) * 2  # 0, 2, 4, ...
        ax_full, ax_filtered = axes[row, col_start], axes[row, col_start + 1]

        data = location_plot_data[location]
        _plot_location_panels(
            data,
            panel_settings,
            panels[location],
            output_config,
            title=format_location_name(location) + data.title_suffix,
            ylabel=plots_config.quantiles.ylabel if col_start == 0 else None,
            axes=(ax_full, ax_filtered),
        )

        # Only show legend on the full panel of the leftmost pair
        legends = [ax_filtered.get_legend()]
        if col_start != 0:
            legends.append(ax_full.get_legend())
        for legend in legends:
            if legend is not None:
                legend.remove()

    # Remove unused axes
    for index in range(n_locations * 2, nrows * ncols):
        axes[index // ncols, index % ncols].axis("off")

    if plots_config.quantiles.suptitle:
        fig.suptitle(plots_config.quantiles.suptitle)

    plt.tight_layout()

    # Add generation notice footnote
    footnotes = {location_plot_data[location].footnote for location in locations} - {""}
    if footnotes:
        _add_footnote(fig, "; ".join(sorted(footnotes)))
    return fig


GRID_OUTPUT_NAMES = {
    QuantilesOutputTypeEnum.FILTERED: "quantiles_grid_filtered",
    QuantilesOutputTypeEnum.FULL: "quantiles_grid_full",
    QuantilesOutputTypeEnum.SIDE_BY_SIDE: "quantiles_grid_sidebyside",
}


def generate_quantile_grid_plot(
    calibrations: list[CalibrationOutput],
    plots_config: PlotsConfig,
    out_dict: dict[str, list[OutputObject]],
    surveillance_sources: dict[str, ObservedValuesConfig] | None = None,
) -> None:
    """
    Generate multi-location quantile grid plot.

    Creates one figure per configured output with a panel (or a full/filtered pair of panels for side_by_side)
    per location. Each panel can optionally include calibration quantiles, projection quantiles,
    surveillance data, and fitting window lines.

    Parameters
    ----------
    calibrations : list[CalibrationOutput]
        List of calibration outputs containing results for each location. Failed projections are expected
        to be filtered already (as done by `generate_calibration_outputs()`).
    plots_config : PlotsConfig
        Configuration object specifying plot settings (quantiles, colors, surveillance data,
        panels per row, etc.)
    out_dict : dict[str, list[OutputObject]]
        Dictionary to store generated plot outputs. Modified in-place by adding entries with keys
        "quantiles_grid_filtered", "quantiles_grid_full" and "quantiles_grid_sidebyside".
    surveillance_sources : dict[str, ObservedValuesConfig] or None, optional
        Surveillance sources that outputs can select with ``surveillance_source``.

    Returns
    -------
    None
        Modifies out_dict in-place by adding quantile grid plot output.
    """
    grid_config = plots_config.quantiles.grid
    if grid_config is False or not grid_config.enabled:
        return

    logger.info("Generating grid quantile plots for %d locations", len(calibrations))

    quantiles_config = plots_config.quantiles
    surveillance_data = _load_surveillance_sources(surveillance_sources, quantiles_config.outputs)

    location_plot_data: dict[str, LocationPlotData] = {}
    for calibration in calibrations:
        try:
            data = _collect_location_plot_data(
                calibration, plots_config, surveillance_data, **_quantile_plot_needs(quantiles_config.outputs)
            )
        except Exception as e:
            logger.warning(
                "Failed to process location %s for grid quantile plots: %s", calibration.population, e, exc_info=True
            )
            continue
        if data.calibration_quantiles is not None or data.projection_quantiles is not None:
            location_plot_data[data.location] = data

    if not location_plot_data:
        return

    for output_config in quantiles_config.outputs:
        output_type_name = output_config.type.value
        logger.info("    Creating %s grid plot", output_type_name)
        try:
            panel_settings = resolve_panel_settings(quantiles_config, output_config, list(surveillance_data))
            panels = {
                location: [
                    prepare_panel_plot_data(data, settings, plots_config.reference_date) for settings in panel_settings
                ]
                for location, data in location_plot_data.items()
            }
            if output_config.type == QuantilesOutputTypeEnum.SIDE_BY_SIDE:
                fig = _plot_quantile_sidebyside_grid(
                    location_plot_data, panel_settings, panels, plots_config, output_config
                )
            else:
                single_panels = {location: location_panels[0] for location, location_panels in panels.items()}
                fig = _plot_quantile_grid(location_plot_data, panel_settings[0], single_panels, plots_config)

            output_name = GRID_OUTPUT_NAMES[output_config.type]
            out_dict[output_name] = _package_figure_outputs(fig, output_name, plots_config)
            plt.close(fig)
        except Exception as e:
            logger.warning("Failed to create %s quantile grid plot: %s", output_type_name, e, exc_info=True)


def generate_single_location_posterior_plots(
    calibrations: list[CalibrationOutput],
    plots_config: PlotsConfig,
    out_dict: dict[str, list[OutputObject]],
    start_date_reference: str | None = None,
) -> None:
    """
    Generate individual posterior histogram plots for each location.

    Creates a single figure per location with subplots showing histograms for all calibrated
    parameters. Each subplot shows the posterior distribution of one parameter from the
    calibration results.

    Parameters
    ----------
    calibrations : list[CalibrationOutput]
        List of calibration outputs containing posterior distributions for each location
    plots_config : PlotsConfig
        Configuration object specifying plot settings (bins, format, dpi, etc.)
    out_dict : dict[str, list[OutputObject]]
        Dictionary to store generated plot outputs. Modified in-place by adding entries with keys
        like "posterior_{location}" mapping to lists of OutputObject instances.
    start_date_reference : str or None, optional
        Reference date for converting start_date parameter offsets to actual dates.
        If None, start_date parameters will be plotted as integer offsets.

    Returns
    -------
    None
        Modifies out_dict in-place by adding posterior plot outputs.
    """
    if not plots_config.posterior.single:
        return

    locations = get_locations_to_plot(calibrations, plots_config.posterior.single)
    logger.info("Generating single-location posterior plots for %d locations", len(locations))

    # Generate plots for each location
    for calibration in calibrations:
        if calibration.population not in locations:
            continue

        location = calibration.population
        logger.info("    Creating posterior plot for %s", location)

        try:
            posterior_df = calibration.results.get_posterior_distribution()

            # Get parameters to plot (exclude metadata columns)
            params = [col for col in posterior_df.columns if col not in POSTERIOR_METADATA_COLUMNS]

            if not params:
                logger.warning("No parameters to plot for %s", location)
                continue

            # Create subplots - arrange in grid
            n_params = len(params)
            n_cols = min(3, n_params)  # Max 3 columns
            n_rows = math.ceil(n_params / n_cols)

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            if n_params == 1:
                axes = [axes]
            else:
                axes = axes.flatten() if n_params > 1 else [axes]

            # Plot each parameter
            for idx, param in enumerate(params):
                try:
                    plot_posterior_histogram(
                        df_posterior=posterior_df,
                        parameter=param,
                        bins=plots_config.posterior.bins,
                        ax=axes[idx],
                        start_date_reference=start_date_reference,
                    )
                    axes[idx].set_title(param)
                except Exception as e:
                    logger.warning(
                        "Failed to create posterior histogram for %s - %s: %s", location, param, e, exc_info=True
                    )
                    axes[idx].set_visible(False)

            # Hide unused subplots
            for idx in range(n_params, len(axes)):
                axes[idx].set_visible(False)

            # Add generation notice to posterior suptitle
            notes = []
            if note := _check_incomplete_generations(calibration):
                notes.append(note)
            title_suffix, footnote = _format_plot_notes(notes)

            fig.suptitle(f"Posterior Distributions - {location}{title_suffix}", fontsize=14, y=0.995)
            fig.tight_layout()
            if footnote:
                _add_footnote(fig, footnote)

            # Package output
            out_dict[f"posterior_{location}"] = _package_figure_outputs(fig, f"posterior_{location}", plots_config)
            plt.close(fig)

        except (ValueError, AttributeError) as e:
            logger.warning("Failed to get posterior distribution for %s: %s", location, e)


def generate_posterior_grid_plot(
    calibrations: list[CalibrationOutput],
    plots_config: PlotsConfig,
    out_dict: dict[str, list[OutputObject]],
    start_date_reference: str | None = None,
) -> None:
    """
    Generate multi-location posterior histogram grid plot.

    Creates a single figure with multiple panels showing posterior distributions for all parameters
    across all locations in a grid layout. Each row represents a different location, and each column
    represents a different calibrated parameter.

    Parameters
    ----------
    calibrations : list[CalibrationOutput]
        List of calibration outputs containing posterior distributions for each location
    plots_config : PlotsConfig
        Configuration object specifying plot settings (bins, format, dpi, etc.)
    out_dict : dict[str, list[OutputObject]]
        Dictionary to store generated plot outputs. Modified in-place by adding an entry with key
        "posterior_grid" mapping to a list containing the grid plot OutputObject.
    start_date_reference : str or None, optional
        Reference date for converting start_date parameter offsets to actual dates.
        If None, start_date parameters will be plotted as integer offsets.

    Returns
    -------
    None
        Modifies out_dict in-place by adding posterior grid plot output.
    """
    if not plots_config.posterior.grid:
        return

    logger.info("Generating grid posterior plot for %d locations", len(calibrations))

    location_posteriors = {}
    all_params = set()
    location_notes: dict[str, tuple[str, str]] = {}  # loc -> (title_suffix, footnote)

    # Collect posterior distributions for each location
    for calibration in calibrations:
        loc = calibration.population

        # Check for incomplete generations
        notes = []
        if note := _check_incomplete_generations(calibration):
            notes.append(note)
        location_notes[loc] = _format_plot_notes(notes)

        try:
            posterior_df = calibration.results.get_posterior_distribution()
            if not posterior_df.empty:
                location_posteriors[loc] = posterior_df
                all_params.update(col for col in posterior_df.columns if col not in POSTERIOR_METADATA_COLUMNS)
            else:
                logger.warning("Skipping posterior plot for %s: posterior data is empty", loc)
        except (ValueError, AttributeError) as e:
            logger.warning("Failed to get posterior distribution for %s: %s", loc, e)

    if location_posteriors and all_params:
        parameters = sorted(all_params)

        try:
            fig, axes = plot_posterior_histogram_grid(
                location_posteriors=location_posteriors,
                parameters=parameters,
                bins=plots_config.posterior.bins,
                start_date_reference=start_date_reference,
            )

            # Add generation notice to posterior grid y-labels (first column)
            grid_footnotes = set()
            for ax_row in axes:
                ax_first = ax_row[0] if hasattr(ax_row, "__getitem__") else ax_row
                ylabel = ax_first.get_ylabel()
                if not ylabel:
                    continue
                for loc, (suffix, fn) in location_notes.items():
                    if suffix and ylabel == loc:
                        ax_first.set_ylabel(loc + suffix)
                        grid_footnotes.add(fn)
                        break
            if grid_footnotes:
                _add_footnote(fig, "; ".join(sorted(grid_footnotes)))

            # Package output
            out_dict["posterior_grid"] = _package_figure_outputs(fig, "posterior_grid", plots_config)
            plt.close(fig)
        except Exception as e:
            logger.warning("Failed to create posterior grid plot: %s", e, exc_info=True)


def generate_categorical_plots(
    plots_config: PlotsConfig,
    out_dict: dict[str, list[OutputObject]],
    hub_format_data: pd.DataFrame | None = None,
) -> None:
    """
    Generate categorical forecast probability plots and add to output dictionary.

    Parameters
    ----------
    plots_config : PlotsConfig
        Contains categorical plot configuration.
    out_dict : dict[str, list[OutputObject]]
        Modified in-place to add output objects under "categorical_rate_trends" key.
    hub_format_data : pd.DataFrame or None, optional
        Pre-generated FluSight format data containing rate-trend forecasts.
        Required for categorical plots. If None or empty, plots are skipped with a warning.

    Notes
    -----
    - Requires FluSight format data with rate-trend categorical forecasts (target: "wk flu hosp rate change")
    - Data should be generated by `make_rate_trends_flusightforecast()` in dispatcher/output.py
    - Gracefully skips if rate-trend data is not available
    - Thresholds are configured in `flusight_format.rate_trends.thresholds`

    Examples
    --------
    >>> # Categorical plots depend on FluSight rate-trend data
    >>> hub_format_output = pd.concat([...])  # From flusight_format outputs
    >>> generate_categorical_plots(plots_config, out_dict, hub_format_output)
    >>> # out_dict["categorical_rate_trends"] now contains the plot OutputObjects

    """
    # Early return if categorical plots are disabled
    if not plots_config.categorical:
        return

    # Early return with warning if hub format data is not available
    if hub_format_data is None or hub_format_data.empty:
        logger.warning(
            "Categorical plots enabled but no FluSight format data available. "
            "Ensure flusight_format.rate_trends is configured."
        )
        return

    # Filter hub format data for rate-trend categorical forecasts
    df_rate_trends = hub_format_data[
        (hub_format_data["target"] == "wk flu hosp rate change") & (hub_format_data["output_type"] == "pmf")
    ]

    # Early return with warning if filtered data is empty
    if df_rate_trends.empty:
        logger.warning(
            "No rate-trend categorical forecasts found in FluSight format data. "
            "Check flusight_format.rate_trends configuration."
        )
        return

    # Convert location codes to readable names
    df_rate_trends = df_rate_trends.copy()
    df_rate_trends["location"] = df_rate_trends["location"].apply(format_location_name)

    # Default category labels (prettier display names)
    default_category_labels = {
        "large_decrease": "Large Decrease",
        "decrease": "Decrease",
        "stable": "Stable",
        "increase": "Increase",
        "large_increase": "Large Increase",
    }

    # Try to create the categorical plot
    try:
        fig, axes = plot_categorical_stacked_bars_multihorizon(
            df_categorical=df_rate_trends,
            categories=plots_config.categorical.categories,
            colors=plots_config.categorical.colors,
            horizons=plots_config.categorical.horizons,
            figsize=plots_config.categorical.figsize,
            reference_date=plots_config.reference_date,
            category_labels=default_category_labels,
        )

        # Package output
        out_dict["categorical_rate_trends"] = _package_figure_outputs(fig, "categorical_rate_trends", plots_config)
        plt.close(fig)

        logger.info("Generated categorical rate-trend plot with %d horizons", len(plots_config.categorical.horizons))

    except Exception as e:
        logger.error("Failed to create categorical rate-trend plot: %s", e, exc_info=True)
