"""Prepare quantile plot panels without drawing or modifying calibration results.

Resolve shared/variant/panel settings once, collect each location once, and select
and clip the frames once per variant. Single and grid renderers share these panels.
"""

import logging
from copy import copy
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Literal

import pandas as pd

from ..builders.utils import get_data_in_location
from ..schema.dispatcher import CalibrationOutput
from ..schema.output import (
    ObservedValuesConfig,
    PlotsConfig,
    QuantilesOutputConfig,
    QuantilesOutputTypeEnum,
    QuantilesPlotConfig,
)

logger = logging.getLogger(__name__)


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
            cal_quant = calibration.results.get_calibration_quantiles(
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
            # Quantiles only need successful trajectories. Keep the caller's results,
            # parameter alignment and failure counts untouched; share the trajectory arrays.
            projection_results = copy(calibration.results)
            projection_results.projections = {
                scenario: [trajectory for trajectory in trajectories if trajectory]
                for scenario, trajectories in calibration.results.projections.items()
            }
            proj_sims = projection_results.projections.get("baseline", [])
            proj_dates = proj_sims[0].get("date") if proj_sims else None
            proj_quant = projection_results.get_projection_quantiles(
                quantiles=plots_config.quantiles.quantiles,
                dates=proj_dates,
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

    NOTE: The rename to 'value' is necessary because plot_quantile_panel() uses a single
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
    """Final settings for one panel of a plot variant, after all overrides are resolved.

    A panel here is the full or filtered view within a plot variant, matching ``full_panel`` / ``filtered_panel``
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
    plot_config: QuantilesPlotConfig,
    variant_config: QuantilesOutputConfig,
    loaded_source_names: list[str],
) -> list[ResolvedPanelSettings]:
    """
    Resolve plot settings (shared plot -> variant -> panel) for one plot variant.

    Precedence is panel > variant > shared plot > default, where ``None`` means "inherit from the next level".
    ``surveillance_points`` and ``surveillance_start_date`` are inherited as a pair: if a panel sets either,
    neither is taken from the variant.

    Parameters
    ----------
    plot_config : QuantilesPlotConfig
        Shared quantile plot configuration from ``output.plots.quantiles``.
    variant_config : QuantilesOutputConfig
        One plot variant from ``output.plots.quantiles.outputs[]``.
    loaded_source_names : list of str
        Names of the surveillance sources that were loaded successfully.

    Returns
    -------
    list of ResolvedPanelSettings
        One entry for filtered/full variants, ``[full, filtered]`` for side_by_side.
    """
    if variant_config.surveillance_source in loaded_source_names:
        surveillance_source = variant_config.surveillance_source
    elif len(loaded_source_names) == 1:
        surveillance_source = loaded_source_names[0]
    else:
        surveillance_source = None

    horizon_max = variant_config.horizon_max if variant_config.horizon_max is not None else plot_config.horizon_max

    if variant_config.type == QuantilesOutputTypeEnum.SIDE_BY_SIDE:
        views = [("full", variant_config.full_panel), ("filtered", variant_config.filtered_panel)]
    else:
        views = [(variant_config.type.value, None)]

    settings = []
    for view, panel in views:
        panel_sets_limits = panel is not None and (
            panel.surveillance_points is not None or panel.surveillance_start_date is not None
        )
        limits = panel if panel_sets_limits else variant_config
        xlabel_interval = variant_config.xlabel_interval
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
                show_calibration=variant_config.show_calibration,
                show_projection=variant_config.show_projection,
                show_surveillance=variant_config.show_surveillance,
                show_fitting_window_line=variant_config.show_fitting_window_line,
                calibration_color=plot_config.calibration.color,
                projection_color=plot_config.projection.color,
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
    """Ready-to-draw frames and styling for one panel; no inheritance or visibility rules remain.

    ``None`` means a layer is not drawn; an empty frame also draws nothing.
    Renderers treat the frames as read-only so single and grid can share them.
    """

    location: str
    calibration_quantiles: pd.DataFrame | None
    projection_quantiles: pd.DataFrame | None
    surveillance: pd.DataFrame | None
    calibration_color: str
    projection_color: str
    xlabel_interval: str | None
    fitting_window_start: date | None
    fitting_window_end: date | None
    title_suffix: str
    footnote: str


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
        location=location_plot_data.location,
        calibration_quantiles=calibration if settings.show_calibration else None,
        projection_quantiles=projection if settings.show_projection else None,
        surveillance=surveillance,
        calibration_color=settings.calibration_color,
        projection_color=settings.projection_color,
        xlabel_interval=settings.xlabel_interval,
        fitting_window_start=location_plot_data.fitting_window_start if settings.show_fitting_window_line else None,
        fitting_window_end=location_plot_data.fitting_window_end if settings.show_fitting_window_line else None,
        title_suffix=location_plot_data.title_suffix,
        footnote=location_plot_data.footnote,
    )


def _load_surveillance_sources(
    surveillance_sources: dict[str, ObservedValuesConfig] | None,
    variant_configs: list[QuantilesOutputConfig],
) -> dict[str, dict[str, Any]]:
    """
    Load surveillance CSV files for all configured sources.

    Skips loading entirely if no plot variant requires surveillance data or if no sources are configured.

    Parameters
    ----------
    surveillance_sources : dict[str, ObservedValuesConfig] or None
        Mapping of source name to its configuration (including ``data_path``).
    variant_configs : list[QuantilesOutputConfig]
        Plot variant configurations; loading is skipped if none have ``show_surveillance`` enabled.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping of source name to ``{"data": pd.DataFrame, "config": ObservedValuesConfig}``.
        Empty dict if no sources are needed or available.
    """
    if not any(variant_config.show_surveillance for variant_config in variant_configs) or not surveillance_sources:
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
    Compute the fitting window date range from calibration quantiles.

    Uses the provided ``cal_quant`` if available; otherwise falls back to fetching a minimal set of calibration quantiles (median only) from the calibration results.

    Parameters
    ----------
    calibration : CalibrationOutput
        Calibration output for one location, used as fallback source for quantile data.
    cal_quant : pd.DataFrame or None
        Pre-fetched calibration quantiles. If None, the function attempts to fetch them.

    Returns
    -------
    tuple[date or None, date or None]
        (start, end) of the fitting window, or (None, None) if quantiles are unavailable.
    """
    cal_quant_for_fitting = cal_quant
    if cal_quant_for_fitting is None:
        try:
            cal_trajs = calibration.results.get_selected_trajectories()
            cal_dates = cal_trajs[0].get("date") if cal_trajs else None
            cal_quant_for_fitting = calibration.results.get_calibration_quantiles(
                quantiles=[0.5],
                dates=cal_dates,
                variables=["data"],
                ignore_nan=True,
            )
        except (ValueError, AttributeError, TypeError, IndexError) as e:
            logger.warning(
                "Failed to get calibration quantiles for fitting window for %s: %s",
                calibration.population,
                e,
            )

    if cal_quant_for_fitting is not None and "date" in cal_quant_for_fitting.columns:
        dates = pd.to_datetime(cal_quant_for_fitting["date"]).dt.date
        return dates.min(), dates.max()
    return None, None


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
        Whether any plot variant needs calibration quantiles, projection quantiles or the fitting window.

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


def _get_plot_data_requirements(variant_configs: list[QuantilesOutputConfig]) -> dict[str, bool]:
    """Return data requirements across plot variants, as keyword arguments for ``_collect_location_plot_data``."""
    return {
        "needs_calibration": any(variant_config.show_calibration for variant_config in variant_configs),
        "needs_projection": any(variant_config.show_projection for variant_config in variant_configs),
        "needs_fitting_window": any(variant_config.show_fitting_window_line for variant_config in variant_configs),
    }


def prepare_quantile_plots(
    calibrations: list[CalibrationOutput],
    plots_config: PlotsConfig,
    surveillance_sources: dict[str, ObservedValuesConfig] | None = None,
) -> dict[QuantilesOutputTypeEnum, dict[str, list[PanelPlotData]]]:
    """Prepare each variant's panels by location, once for all requested layouts.

    The caller selects locations before preparation. Source loading, setting
    resolution and quantile collection are shared by single and grid rendering.
    A failed location/variant is logged and omitted without dropping other plots.
    Input calibration results, projection parameters and failure counts are unchanged.
    """
    variant_configs = plots_config.quantiles.outputs
    if not calibrations or not variant_configs:
        return {}

    surveillance_data = _load_surveillance_sources(surveillance_sources, variant_configs)
    data_requirements = _get_plot_data_requirements(variant_configs)
    settings_by_variant = {
        variant_config.type: resolve_panel_settings(plots_config.quantiles, variant_config, list(surveillance_data))
        for variant_config in variant_configs
    }
    prepared = {variant_type: {} for variant_type in settings_by_variant}
    for calibration in calibrations:
        try:
            location_data = _collect_location_plot_data(
                calibration, plots_config, surveillance_data, **data_requirements
            )
        except Exception as e:
            logger.warning("Failed to prepare quantile data for %s: %s", calibration.population, e, exc_info=True)
            continue
        for variant_type, panel_settings in settings_by_variant.items():
            try:
                panel_data = [
                    prepare_panel_plot_data(location_data, settings, plots_config.reference_date)
                    for settings in panel_settings
                ]
                if any(
                    panel.calibration_quantiles is not None or panel.projection_quantiles is not None
                    for panel in panel_data
                ):
                    prepared[variant_type][location_data.location] = panel_data
            except Exception as e:
                logger.warning(
                    "Failed to prepare %s panels for %s: %s",
                    variant_type.value,
                    calibration.population,
                    e,
                    exc_info=True,
                )
    return prepared
