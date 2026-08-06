"""Plot generation orchestration for calibration/projection outputs.

This module contains high-level functions that orchestrate the creation of visualization outputs
by loading data, collecting quantiles/posteriors, calling visualization primitives, and packaging
outputs as OutputObject instances.
"""

import logging
import math
from datetime import date, timedelta
from typing import Any

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
)
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
            proj_sims = calibration.results.projections.get("baseline", [])
            proj_dates = proj_sims[0].get("date") if proj_sims else None
            proj_quant = calibration.results.get_projection_quantiles(
                quantiles=plots_config.quantiles.quantiles,
                dates=proj_dates,
                ignore_nan=True,
            )
        except (ValueError, AttributeError, TypeError, IndexError) as e:
            logger.warning("Failed to get projection quantiles for %s: %s", calibration.population, e)

    return cal_quant, proj_quant


def _prepare_surveillance_for_location(
    surveillance: pd.DataFrame | None,
    location: str,
    proj_quant: pd.DataFrame | None,
    cal_quant: pd.DataFrame | None,
    surveillance_config: ObservedValuesConfig,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, date | None]:
    """
    Build full and filtered surveillance dataframes for a location.

    Parameters
    ----------
    surveillance : pd.DataFrame or None
        Raw surveillance data for all locations.
    location : str
        Location identifier to filter surveillance.
    proj_quant : pd.DataFrame or None
        Projection quantiles for the location (preferred to infer timespan start).
    cal_quant : pd.DataFrame or None
        Calibration quantiles for the location (fallback to infer timespan start).
    surveillance_config : ObservedValuesConfig
        Configuration for surveillance data source.

    Returns
    -------
    tuple
        (df_surv_full, df_surv_filtered, surveillance_start_date) where start date is the first filtered point.
    """
    df_surv_full = None
    df_surv_filtered = None
    surveillance_start_date = None

    if surveillance is None:
        return df_surv_full, df_surv_filtered, surveillance_start_date

    surv = get_data_in_location(
        surveillance,
        location,
        surveillance_config.location_column,
        surveillance_config.location_format,
    )

    # Full surveillance (no filtering)
    # Select columns before renaming to avoid duplicates if source data already has 'value' column
    if not surv.empty:
        df_surv_full = surv[[surveillance_config.date_column, surveillance_config.value_column]].rename(
            columns={
                surveillance_config.date_column: "date",
                surveillance_config.value_column: "value",
            }
        )

    # Filtered surveillance
    surv_filtered = surv.copy()

    if not surv_filtered.empty:
        quantiles_for_timespan = proj_quant if proj_quant is not None else cal_quant
        if quantiles_for_timespan is not None and "date" in quantiles_for_timespan.columns:
            timespan_start = pd.to_datetime(quantiles_for_timespan["date"]).min().date()
            surv_filtered = surv_filtered[
                pd.to_datetime(surv_filtered[surveillance_config.date_column]).dt.date >= timespan_start
            ]

    if not surv_filtered.empty:
        # Select columns before renaming to avoid duplicates if source data already has 'value' column
        df_surv_filtered = surv_filtered[[surveillance_config.date_column, surveillance_config.value_column]].rename(
            columns={
                surveillance_config.date_column: "date",
                surveillance_config.value_column: "value",
            }
        )

        if not df_surv_filtered.empty:
            surveillance_start_date = pd.to_datetime(df_surv_filtered["date"]).min().date()

    return df_surv_full, df_surv_filtered, surveillance_start_date


def _prepare_projection_quantiles(
    proj_quant: pd.DataFrame | None,
    surveillance_start_date: date | None,
    plots_config: PlotsConfig,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Create full and filtered projection quantiles for a location.

    Parameters
    ----------
    proj_quant : pd.DataFrame or None
        Projection quantiles for the location.
    surveillance_start_date : date or None
        First surveillance date used to cut projections for the filtered version.
    plots_config : PlotsConfig
        Plot configuration providing reference_date and base horizon_max.

    Returns
    -------
    tuple
        (proj_quant_full, proj_quant_filtered) using base horizon_max; filtered also cuts before surveillance start.
    """
    proj_quant_full = None
    proj_quant_filtered = None

    if proj_quant is None:
        return proj_quant_full, proj_quant_filtered

    # Full version: horizon_max only
    proj_quant_full = proj_quant.copy()
    if plots_config.quantiles.horizon_max is not None:
        proj_dates_full = pd.to_datetime(proj_quant_full["date"]).dt.date
        horizon_end_full = plots_config.reference_date + timedelta(weeks=plots_config.quantiles.horizon_max)
        proj_quant_full = proj_quant_full[proj_dates_full.values <= horizon_end_full]

    # Filtered version: surveillance start + horizon_max
    proj_quant_filtered = proj_quant.copy()
    proj_dates = pd.to_datetime(proj_quant_filtered["date"]).dt.date
    if surveillance_start_date is not None:
        proj_start = proj_dates.min()
        if proj_start < surveillance_start_date:
            proj_quant_filtered = proj_quant_filtered[proj_dates.values >= surveillance_start_date]

    if plots_config.quantiles.horizon_max is not None:
        proj_dates = pd.to_datetime(proj_quant_filtered["date"]).dt.date
        horizon_end = plots_config.reference_date + timedelta(weeks=plots_config.quantiles.horizon_max)
        proj_quant_filtered = proj_quant_filtered[proj_dates.values <= horizon_end]

    return proj_quant_full, proj_quant_filtered


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


def _clip_to_surveillance_start(
    df: pd.DataFrame | None,
    surv: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """
    Clip a dataframe to start at the earliest date in surveillance data. Used to align calibration/projection quantile ribbons with the visible surveillance range.

    Parameters
    ----------
    df : pd.DataFrame or None
        DataFrame with a "date" column (e.g. calibration or projection quantiles).
    surv : pd.DataFrame or None
        Surveillance DataFrame with a "date" column whose minimum date defines the clip boundary.

    Returns
    -------
    pd.DataFrame or None
        Rows of ``df`` where date >= earliest surveillance date, or None if the result is empty.
        Returns ``df`` unchanged when ``surv`` is None or empty.
    """
    if df is None or surv is None or surv.empty:
        return df
    start = pd.to_datetime(surv["date"]).dt.date.min()
    clipped = df[pd.to_datetime(df["date"]).dt.date >= start]
    return clipped if not clipped.empty else None


def _clip_surveillance(
    surv: pd.DataFrame | None,
    *,
    surveillance_start_date: str | None = None,
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
    surveillance_start_date : str or None
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
    Returns a copy to avoid mutating the caller's DataFrame.

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
    proj = proj.copy()
    dates = pd.to_datetime(proj["date"]).dt.date
    end = reference_date + timedelta(weeks=horizon_max)
    return proj[dates.values <= end]


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


def _create_quantile_plot(
    location: str,
    cal_quant: pd.DataFrame | None,
    proj_quant: pd.DataFrame | None,
    df_surv: pd.DataFrame | None,
    fitting_window_start: date | None,
    fitting_window_end: date | None,
    plots_config: PlotsConfig,
    value_col: str,
    output_config: QuantilesOutputConfig,
) -> tuple:
    """
    Create a quantile plot (filtered or full) for a location.

    Parameters
    ----------
    location : str
        Location name
    cal_quant : pd.DataFrame or None
        Calibration quantiles
    proj_quant : pd.DataFrame or None
        Projection quantiles
    df_surv : pd.DataFrame or None
        Surveillance data
    fitting_window_start : date or None
        Start of fitting window
    fitting_window_end : date or None
        End of fitting window
    plots_config : PlotsConfig
        Plot configuration
    value_col : str
        Name of value column
    output_config : QuantilesOutputConfig
        Output configuration with show flags

    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes
    """
    return plot_calibration_projection(
        calibration_quantiles=cal_quant if output_config.show_calibration else None,
        projection_quantiles=proj_quant if output_config.show_projection else None,
        value_col=value_col,
        calibration_color=plots_config.quantiles.calibration.color,
        projection_color=plots_config.quantiles.projection.color,
        df_surveillance=df_surv if output_config.show_surveillance else None,
        fitting_window_start=fitting_window_start if output_config.show_fitting_window_line else None,
        fitting_window_end=fitting_window_end if output_config.show_fitting_window_line else None,
        title=format_location_name(location),
        xlabel_interval=output_config.xlabel_interval,
        ylabel=plots_config.quantiles.ylabel,
    )


def _create_sidebyside_plot(
    location: str,
    cal_quant: pd.DataFrame | None,
    proj_quant_full: pd.DataFrame | None,
    proj_quant_filtered: pd.DataFrame | None,
    df_surv_full: pd.DataFrame | None,
    df_surv_filtered: pd.DataFrame | None,
    fitting_window_start: date | None,
    fitting_window_end: date | None,
    plots_config: PlotsConfig,
    value_col: str,
    output_config: QuantilesOutputConfig,
    cal_quant_filtered: pd.DataFrame | None = None,
) -> tuple:
    """
    Create side-by-side (full | filtered) quantile plot for a location.

    Parameters
    ----------
    location : str
        Location name
    cal_quant : pd.DataFrame or None
        Calibration quantiles for full (left) panel
    proj_quant_full : pd.DataFrame or None
        Projection quantiles for left panel (full version)
    proj_quant_filtered : pd.DataFrame or None
        Projection quantiles for right panel (filtered version)
    df_surv_full : pd.DataFrame or None
        Surveillance data for left panel (full version)
    df_surv_filtered : pd.DataFrame or None
        Surveillance data for right panel (filtered version)
    fitting_window_start : date or None
        Start of fitting window
    fitting_window_end : date or None
        End of fitting window
    plots_config : PlotsConfig
        Plot configuration
    value_col : str
        Name of value column
    output_config : QuantilesOutputConfig
        Output configuration with show flags
    cal_quant_filtered : pd.DataFrame or None, optional
        Calibration quantiles for filtered (right) panel. If None, uses cal_quant.

    Returns
    -------
    tuple
        (fig, (ax_full, ax_filtered)) matplotlib figure and tuple of axes
    """
    # Get xlabel_interval for each panel from panel configs
    xlabel_interval_full = output_config.full_panel.xlabel_interval if output_config.full_panel else None
    xlabel_interval_filtered = output_config.filtered_panel.xlabel_interval if output_config.filtered_panel else None

    return plot_calibration_projection_sidebyside(
        calibration_quantiles=cal_quant if output_config.show_calibration else None,
        calibration_quantiles_filtered=cal_quant_filtered if output_config.show_calibration else None,
        projection_quantiles_full=proj_quant_full if output_config.show_projection else None,
        projection_quantiles_filtered=proj_quant_filtered if output_config.show_projection else None,
        surveillance_full=df_surv_full if output_config.show_surveillance else None,
        surveillance_filtered=df_surv_filtered if output_config.show_surveillance else None,
        value_col=value_col,
        calibration_color=plots_config.quantiles.calibration.color,
        projection_color=plots_config.quantiles.projection.color,
        fitting_window_start=fitting_window_start if output_config.show_fitting_window_line else None,
        fitting_window_end=fitting_window_end if output_config.show_fitting_window_line else None,
        title=format_location_name(location),
        figsize=output_config.figsize,
        spacing=output_config.spacing,
        ylabel=plots_config.quantiles.ylabel,
        xlabel_interval_full=xlabel_interval_full,
        xlabel_interval_filtered=xlabel_interval_filtered,
    )


def get_locations_to_plot(calibrations: list[CalibrationOutput], single_config: bool | list[str]) -> set[str]:
    """
    Get set of locations to plot based on config.

    Parameters
    ----------
    calibrations : list[CalibrationOutput]
        List of calibration outputs
    single_config : bool or list of str
        If True, return all locations. If list, return those specific locations.

    Returns
    -------
    set of str
        Set of location names to plot
    """
    if single_config is True:
        return {cal.population for cal in calibrations}
    if isinstance(single_config, list):
        return set(single_config)
    return set()


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
    surveillance data, and a reference date line.

    Parameters
    ----------
    calibrations : list[CalibrationOutput]
        List of calibration outputs containing results for each location
    plots_config : PlotsConfig
        Configuration object specifying plot settings (quantiles, colors, surveillance data, etc.)
    out_dict : dict[str, list[OutputObject]]
        Dictionary to store generated plot outputs. Modified in-place by adding entries with keys
        like "quantiles_{location}" mapping to lists of OutputObject instances.

    Returns
    -------
    None
        Modifies out_dict in-place by adding quantile plot outputs.
    """
    from ..dispatcher.output import filter_failed_projections

    if not plots_config.quantiles.single:
        return

    locations = get_locations_to_plot(calibrations, plots_config.quantiles.single)
    logger.info("Generating single-location quantile plots for %d locations", len(locations))

    # Load surveillance data once before loop if any output needs it
    surveillance_data = _load_surveillance_sources(surveillance_sources, plots_config.quantiles.outputs)

    needs_calibration = any(output.show_calibration for output in plots_config.quantiles.outputs)
    needs_projection = any(output.show_projection for output in plots_config.quantiles.outputs)
    needs_fitting_window = any(output.show_fitting_window_line for output in plots_config.quantiles.outputs)

    # Generate plots for each location
    for calibration in calibrations:
        # Skip locations not in config
        if calibration.population not in locations:
            continue
        location = calibration.population

        try:
            # Check for incomplete generations
            notes = []
            if note := _check_incomplete_generations(calibration):
                notes.append(note)
            title_suffix, footnote = _format_plot_notes(notes)

            # Filter failed calibration trajectories and projections
            # calibration.results = filter_failed_calibration_trajectories(calibration.results)
            calibration.results = filter_failed_projections(calibration.results)

            cal_quant, proj_quant = _fetch_quantiles_for_location(
                calibration, plots_config, needs_calibration, needs_projection
            )

            # Calculate fitting window start and end from calibration quantiles
            fitting_window_start = None
            fitting_window_end = None
            if needs_fitting_window:
                fitting_window_start, fitting_window_end = _compute_fitting_window(calibration, cal_quant)

            # TODO: Determine which surveillance source to use (logic will be added with per-output processing)
            # For now, use first available source if any
            surveillance = None
            surveillance_config = None
            if surveillance_data:
                first_source = next(iter(surveillance_data.values()))
                surveillance = first_source["data"]
                surveillance_config = first_source["config"]

            df_surv_full, df_surv_filtered, surveillance_start_date = (
                _prepare_surveillance_for_location(surveillance, location, proj_quant, cal_quant, surveillance_config)
                if surveillance_config
                else (None, None, None)
            )

            proj_quant_full, proj_quant_filtered = _prepare_projection_quantiles(
                proj_quant, surveillance_start_date, plots_config
            )

            # Rename columns to have consistent naming for plotting
            cal_quant = _rename_value_column(cal_quant, "data")
            proj_quant_filtered = _rename_value_column(proj_quant_filtered, plots_config.quantiles.value_column)
            proj_quant_full = _rename_value_column(proj_quant_full, plots_config.quantiles.value_column)

            value_col = "value"

            # Create plots for each configured output
            for output_config in plots_config.quantiles.outputs:
                output_name = f"quantiles_{location}_{output_config.type.value}"
                logger.info("    Creating %s plot for %s", output_config.type.value, location)

                try:
                    # Determine which data to use and apply per-output filtering
                    if output_config.type == QuantilesOutputTypeEnum.FILTERED:
                        proj_to_use = proj_quant_filtered
                        surv_to_use = df_surv_filtered
                    elif output_config.type == QuantilesOutputTypeEnum.FULL:
                        proj_to_use = proj_quant_full
                        surv_to_use = df_surv_full
                    elif output_config.type == QuantilesOutputTypeEnum.SIDE_BY_SIDE:
                        # Side-by-side handled separately below
                        proj_to_use = None
                        surv_to_use = None
                    else:
                        logger.warning("Unknown output type %s for %s", output_config.type, location)
                        continue

                    # Apply per-output surveillance filtering if needed (for filtered/full types)
                    surv_to_use = _clip_surveillance(
                        surv_to_use,
                        surveillance_start_date=output_config.surveillance_start_date,
                        surveillance_points=output_config.surveillance_points,
                        reference_date=plots_config.reference_date,
                    )

                    # For filtered plots, clip projection and calibration quantiles to
                    # the visible surveillance start so the ribbons match the zoomed view
                    cal_to_use = cal_quant
                    if output_config.type == QuantilesOutputTypeEnum.FILTERED:
                        proj_to_use = _clip_to_surveillance_start(proj_to_use, surv_to_use)
                        cal_to_use = _clip_to_surveillance_start(cal_to_use, surv_to_use)

                    # Apply per-output horizon_max if specified (overrides base config)
                    proj_to_use = _clip_to_horizon(proj_to_use, output_config.horizon_max, plots_config.reference_date)

                    # Create the plot
                    if output_config.type == QuantilesOutputTypeEnum.FILTERED:
                        fig, ax = _create_quantile_plot(
                            location,
                            cal_to_use,
                            proj_to_use,
                            surv_to_use,
                            fitting_window_start,
                            fitting_window_end,
                            plots_config,
                            value_col,
                            output_config,
                        )
                    elif output_config.type == QuantilesOutputTypeEnum.FULL:
                        fig, ax = _create_quantile_plot(
                            location,
                            cal_quant,
                            proj_to_use,
                            surv_to_use,
                            fitting_window_start,
                            fitting_window_end,
                            plots_config,
                            value_col,
                            output_config,
                        )
                    elif output_config.type == QuantilesOutputTypeEnum.SIDE_BY_SIDE:
                        # Clip calibration and projection quantiles for the filtered panel
                        cal_quant_for_filtered = _clip_to_surveillance_start(cal_quant, df_surv_filtered)
                        proj_quant_for_filtered = _clip_to_surveillance_start(proj_quant_filtered, df_surv_filtered)

                        fig, (ax_full, ax_filtered) = _create_sidebyside_plot(
                            location,
                            cal_quant,
                            proj_quant_full,
                            proj_quant_for_filtered,
                            df_surv_full,
                            df_surv_filtered,
                            fitting_window_start,
                            fitting_window_end,
                            plots_config,
                            value_col,
                            output_config,
                            cal_quant_filtered=cal_quant_for_filtered,
                        )
                    else:
                        logger.warning("Unknown output type %s for %s", output_config.type, location)
                        continue

                    # Add generation notice to plot titles and footnote
                    if title_suffix:
                        if output_config.type == QuantilesOutputTypeEnum.SIDE_BY_SIDE:
                            current_title = ax_full.get_title()
                            if current_title:
                                ax_full.set_title(current_title + title_suffix)
                        else:
                            current_title = ax.get_title()
                            if current_title:
                                ax.set_title(current_title + title_suffix)
                        _add_footnote(fig, footnote)

                    # Package output
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
        except Exception as e:
            logger.warning(
                "Failed to process location %s for quantile plots: %s",
                location,
                e,
                exc_info=True,
            )


def generate_quantile_grid_plot(
    calibrations: list[CalibrationOutput],
    plots_config: PlotsConfig,
    out_dict: dict[str, list[OutputObject]],
    surveillance_sources: dict[str, ObservedValuesConfig] | None = None,
) -> None:
    """
    Generate multi-location quantile grid plot.

    Creates a single figure with multiple panels showing calibration/projection quantiles for all
    locations in a grid layout. Each panel can optionally include calibration quantiles, projection
    quantiles, surveillance data, and a reference date line.

    Parameters
    ----------
    calibrations : list[CalibrationOutput]
        List of calibration outputs containing results for each location
    plots_config : PlotsConfig
        Configuration object specifying plot settings (quantiles, colors, surveillance data,
        panels per row, etc.)
    out_dict : dict[str, list[OutputObject]]
        Dictionary to store generated plot outputs. Modified in-place by adding an entry with key
        "quantiles_grid" mapping to a list containing the grid plot OutputObject.

    Returns
    -------
    None
        Modifies out_dict in-place by adding quantile grid plot output.
    """
    from ..dispatcher.output import filter_failed_projections

    if not plots_config.quantiles.grid:
        return

    logger.info("Generating grid quantile plots for %d locations", len(calibrations))

    # Load surveillance data once before loop if any output needs it
    surveillance_data = _load_surveillance_sources(surveillance_sources, plots_config.quantiles.outputs)

    needs_calibration = any(output.show_calibration for output in plots_config.quantiles.outputs)
    needs_projection = any(output.show_projection for output in plots_config.quantiles.outputs)
    needs_fitting_window = any(output.show_fitting_window_line for output in plots_config.quantiles.outputs)

    # Collect quantiles and surveillance data for each location
    location_cal_quants = {}
    location_proj_quants_raw = {}
    location_fitting_window_starts = {}
    location_fitting_window_ends = {}
    location_notes: dict[str, tuple[str, str]] = {}  # loc -> (title_suffix, footnote)

    # Collect quantiles for each location
    for calibration in calibrations:
        loc = calibration.population

        try:
            # Check for incomplete generations
            notes = []
            if note := _check_incomplete_generations(calibration):
                notes.append(note)
            location_notes[loc] = _format_plot_notes(notes)

            # Filter failed calibration trajectories and projections
            # calibration.results = filter_failed_calibration_trajectories(calibration.results)
            calibration.results = filter_failed_projections(calibration.results)

            cal_quant, proj_quant_raw = _fetch_quantiles_for_location(
                calibration, plots_config, needs_calibration, needs_projection
            )
            if cal_quant is not None:
                location_cal_quants[loc] = cal_quant
            if proj_quant_raw is not None:
                location_proj_quants_raw[loc] = proj_quant_raw

            # Calculate fitting window start and end from calibration quantiles
            if needs_fitting_window:
                fitting_window_start, fitting_window_end = _compute_fitting_window(
                    calibration, location_cal_quants.get(loc)
                )
                if fitting_window_start is not None:
                    location_fitting_window_starts[loc] = fitting_window_start
                    location_fitting_window_ends[loc] = fitting_window_end
        except Exception as e:
            logger.warning(
                "Failed to process location %s for grid quantile plots: %s",
                loc,
                e,
                exc_info=True,
            )

    # Go over collected data, plot grid, and add to output dict
    if location_cal_quants or location_proj_quants_raw:
        # Rename columns to have consistent naming for plotting
        # TODO: Calibration uses "data", projection uses "hospitalizations" - make this configurable
        for loc in location_cal_quants:
            location_cal_quants[loc] = _rename_value_column(location_cal_quants[loc], "data")

        value_col = "value"

        # Loop over configured outputs and generate plots
        for output_config in plots_config.quantiles.outputs:
            output_type_name = output_config.type.value  # "filtered", "full", or "side_by_side"

            # Determine which surveillance source to use for this output
            surveillance_source_name = output_config.surveillance_source
            if surveillance_source_name is None and surveillance_data and len(surveillance_data) == 1:
                # If only one source available and none specified, use it
                surveillance_source_name = next(iter(surveillance_data.keys()))

            # Prepare surveillance data for this output's specified source
            location_surveillance_for_output = {}
            surveillance_start_dates = {}
            if (
                output_config.show_surveillance
                and surveillance_source_name
                and surveillance_source_name in surveillance_data
            ):
                source = surveillance_data[surveillance_source_name]
                surveillance_df = source["data"]
                surveillance_config = source["config"]
                logger.info(
                    f"Loading surveillance from source '{surveillance_source_name}' for {output_type_name} output"
                )
                logger.debug(f"Surveillance data shape: {surveillance_df.shape}")

                for loc in location_proj_quants_raw:
                    try:
                        cal_quant = location_cal_quants.get(loc)
                        proj_quant_raw = location_proj_quants_raw.get(loc)

                        logger.debug(f"Preparing surveillance for location: {loc}")
                        df_surv_full, df_surv_filtered, surveillance_start_date = _prepare_surveillance_for_location(
                            surveillance_df, loc, proj_quant_raw, cal_quant, surveillance_config
                        )
                        logger.debug(
                            f"Location {loc}: surv_full={df_surv_full.shape if df_surv_full is not None else None}, surv_filtered={df_surv_filtered.shape if df_surv_filtered is not None else None}"
                        )

                        # Use the appropriate surveillance data based on output type
                        if output_type_name == "filtered":
                            if df_surv_filtered is not None:
                                location_surveillance_for_output[loc] = df_surv_filtered
                        elif output_type_name == "full":
                            if df_surv_full is not None:
                                location_surveillance_for_output[loc] = df_surv_full

                        if surveillance_start_date is not None:
                            surveillance_start_dates[loc] = surveillance_start_date
                    except Exception as e:
                        logger.warning(
                            "Failed to prepare surveillance data for location %s in grid plot: %s",
                            loc,
                            e,
                            exc_info=True,
                        )

            # Prepare projection quantiles for this output (using surveillance start dates if available)
            location_proj_quants_for_output = {}
            for loc, proj_quant_raw in location_proj_quants_raw.items():
                surveillance_start_date = surveillance_start_dates.get(loc)
                proj_quant_full, proj_quant_filtered = _prepare_projection_quantiles(
                    proj_quant_raw, surveillance_start_date, plots_config
                )

                # Use the appropriate projection data based on output type
                if output_type_name == "filtered":
                    if proj_quant_filtered is not None:
                        location_proj_quants_for_output[loc] = _rename_value_column(
                            proj_quant_filtered, plots_config.quantiles.value_column
                        )
                elif output_type_name == "full":
                    if proj_quant_full is not None:
                        location_proj_quants_for_output[loc] = _rename_value_column(
                            proj_quant_full, plots_config.quantiles.value_column
                        )

            # Set data to use for this output
            if output_type_name in ["filtered", "full"]:
                proj_quants_to_use = location_proj_quants_for_output
                surv_data_to_use = location_surveillance_for_output
            elif output_type_name == "side_by_side":
                # Side-by-side uses both, handled separately below
                proj_quants_to_use = None
                surv_data_to_use = None
            else:
                logger.warning(f"Unknown output type: {output_type_name}")
                continue

            # Apply per-output surveillance filtering if needed
            if surv_data_to_use:
                surv_data_to_use = {
                    loc: _clip_surveillance(
                        surv_df,
                        surveillance_start_date=output_config.surveillance_start_date,
                        surveillance_points=output_config.surveillance_points,
                        reference_date=plots_config.reference_date,
                    )
                    for loc, surv_df in surv_data_to_use.items()
                }

            # For filtered plots, clip projection and calibration quantiles to
            # the visible surveillance start so the ribbons match the zoomed view
            cal_quants_to_use = location_cal_quants
            if output_type_name == "filtered" and surv_data_to_use:
                cal_quants_clipped = {}
                proj_quants_clipped = {}
                for loc in set(list(cal_quants_to_use or []) + list(proj_quants_to_use or [])):
                    surv_df = surv_data_to_use.get(loc)
                    if loc in (cal_quants_to_use or {}):
                        clipped = _clip_to_surveillance_start(cal_quants_to_use[loc], surv_df)
                        if clipped is not None:
                            cal_quants_clipped[loc] = clipped
                    if loc in (proj_quants_to_use or {}):
                        clipped = _clip_to_surveillance_start(proj_quants_to_use[loc], surv_df)
                        if clipped is not None:
                            proj_quants_clipped[loc] = clipped
                if cal_quants_to_use:
                    cal_quants_to_use = cal_quants_clipped
                if proj_quants_to_use:
                    proj_quants_to_use = proj_quants_clipped

            # Apply per-output horizon_max if specified (overrides base config)
            if proj_quants_to_use and output_config.horizon_max is not None:
                proj_quants_to_use = {
                    loc: _clip_to_horizon(df, output_config.horizon_max, plots_config.reference_date)
                    for loc, df in proj_quants_to_use.items()
                }

            # Generate grid plot based on output type
            if output_type_name in ["filtered", "full"]:
                logger.info("    Creating %s grid plot", output_type_name)
                try:
                    xlim = None
                    if output_config.xlim_start is not None or output_config.xlim_end is not None:
                        xlim = (
                            pd.Timestamp(output_config.xlim_start) if output_config.xlim_start else None,
                            pd.Timestamp(output_config.xlim_end) if output_config.xlim_end else None,
                        )

                    fig, axes = plot_calibration_projection_grid(
                        location_calibration_quantiles=(
                            cal_quants_to_use if output_config.show_calibration and cal_quants_to_use else None
                        ),
                        location_projection_quantiles=(
                            proj_quants_to_use if output_config.show_projection and proj_quants_to_use else None
                        ),
                        value_col=value_col,
                        calibration_color=plots_config.quantiles.calibration.color,
                        projection_color=plots_config.quantiles.projection.color,
                        location_surveillance=(
                            surv_data_to_use if output_config.show_surveillance and surv_data_to_use else None
                        ),
                        location_fitting_window_starts=(
                            location_fitting_window_starts if output_config.show_fitting_window_line else None
                        ),
                        location_fitting_window_ends=(
                            location_fitting_window_ends if output_config.show_fitting_window_line else None
                        ),
                        panels_per_row=plots_config.quantiles.grid.panels_per_row,
                        ylabel=plots_config.quantiles.ylabel,
                        xlabel_interval=output_config.xlabel_interval,
                        suptitle=plots_config.quantiles.suptitle,
                        xlim=xlim,
                    )

                    # Add generation notice to grid panel titles
                    grid_footnotes = set()
                    for ax in axes.flat:
                        title = ax.get_title()
                        if not title:
                            continue
                        for loc, (suffix, fn) in location_notes.items():
                            if suffix and title == format_location_name(loc):
                                ax.set_title(title + suffix)
                                grid_footnotes.add(fn)
                                break
                    if grid_footnotes:
                        _add_footnote(fig, "; ".join(sorted(grid_footnotes)))

                    # Package output
                    out_dict[f"quantiles_grid_{output_type_name}"] = _package_figure_outputs(
                        fig, f"quantiles_grid_{output_type_name}", plots_config
                    )
                    plt.close(fig)
                except Exception as e:
                    logger.warning("Failed to create %s quantile grid plot: %s", output_type_name, e, exc_info=True)

            elif output_type_name == "side_by_side":
                # Side-by-side grid plot
                # Grid layout: each location gets 2 panels (full + filtered)
                logger.info("    Creating side_by_side grid plot")
                try:
                    # Prepare surveillance and projection data for both full and filtered panels
                    location_surveillance_full_sbs = {}
                    location_surveillance_filtered_sbs = {}
                    location_proj_quants_full_sbs = {}
                    location_proj_quants_filtered_sbs = {}
                    surveillance_start_dates_sbs = {}

                    if surveillance_source_name and surveillance_source_name in surveillance_data:
                        source = surveillance_data[surveillance_source_name]
                        surveillance_df = source["data"]
                        surveillance_config = source["config"]

                        for loc in location_proj_quants_raw:
                            try:
                                cal_quant = location_cal_quants.get(loc)
                                proj_quant_raw = location_proj_quants_raw.get(loc)

                                df_surv_full, df_surv_filtered, surveillance_start_date = (
                                    _prepare_surveillance_for_location(
                                        surveillance_df, loc, proj_quant_raw, cal_quant, surveillance_config
                                    )
                                )

                                if df_surv_full is not None:
                                    location_surveillance_full_sbs[loc] = df_surv_full
                                if df_surv_filtered is not None:
                                    location_surveillance_filtered_sbs[loc] = df_surv_filtered
                                if surveillance_start_date is not None:
                                    surveillance_start_dates_sbs[loc] = surveillance_start_date
                            except Exception as e:
                                logger.warning(
                                    "Failed to prepare surveillance data for location %s in side-by-side grid plot: %s",
                                    loc,
                                    e,
                                    exc_info=True,
                                )

                    # Prepare projection quantiles
                    for loc, proj_quant_raw in location_proj_quants_raw.items():
                        try:
                            surveillance_start_date = surveillance_start_dates_sbs.get(loc)
                            proj_quant_full, proj_quant_filtered = _prepare_projection_quantiles(
                                proj_quant_raw, surveillance_start_date, plots_config
                            )
                            if proj_quant_full is not None:
                                location_proj_quants_full_sbs[loc] = _rename_value_column(
                                    proj_quant_full, plots_config.quantiles.value_column
                                )
                            if proj_quant_filtered is not None:
                                location_proj_quants_filtered_sbs[loc] = _rename_value_column(
                                    proj_quant_filtered, plots_config.quantiles.value_column
                                )
                        except Exception as e:
                            logger.warning(
                                "Failed to prepare projection quantiles for location %s in side-by-side grid plot: %s",
                                loc,
                                e,
                                exc_info=True,
                            )

                    # Get all locations
                    locations = set()
                    if location_cal_quants:
                        locations.update(location_cal_quants.keys())
                    if location_proj_quants_full_sbs:
                        locations.update(location_proj_quants_full_sbs.keys())
                    if location_proj_quants_filtered_sbs:
                        locations.update(location_proj_quants_filtered_sbs.keys())
                    locations = sort_locations_by_state(locations)

                    if locations:
                        n_locations = len(locations)
                        panels_per_row = plots_config.quantiles.grid.panels_per_row
                        pairs_per_row = panels_per_row // 2  # Each location needs 2 panels
                        nrows = math.ceil(n_locations / pairs_per_row)
                        ncols = panels_per_row

                        figsize = output_config.figsize or (4 * ncols, 3.6 * nrows)
                        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

                        for i, location in enumerate(locations):
                            pair_idx = i  # Which location-pair (0, 1, 2, ...)
                            row = pair_idx // pairs_per_row
                            col_start = (pair_idx % pairs_per_row) * 2  # 0, 2, 4, ...

                            ax_full = axes[row, col_start]  # Left panel of pair
                            ax_filtered = axes[row, col_start + 1]  # Right panel of pair

                            cal_quant = (
                                location_cal_quants.get(location)
                                if output_config.show_calibration and location_cal_quants
                                else None
                            )
                            proj_quant_full = (
                                location_proj_quants_full_sbs.get(location)
                                if output_config.show_projection and location_proj_quants_full_sbs
                                else None
                            )
                            proj_quant_filtered = (
                                location_proj_quants_filtered_sbs.get(location)
                                if output_config.show_projection and location_proj_quants_filtered_sbs
                                else None
                            )

                            surv_full = None
                            if (
                                output_config.show_surveillance
                                and location_surveillance_full_sbs
                                and location in location_surveillance_full_sbs
                            ):
                                surv_full = location_surveillance_full_sbs[location].copy()
                                if output_config.full_panel:
                                    surv_full = _clip_surveillance(
                                        surv_full,
                                        surveillance_start_date=output_config.full_panel.surveillance_start_date,
                                        surveillance_points=output_config.full_panel.surveillance_points,
                                        reference_date=plots_config.reference_date,
                                    )

                            surv_filtered = None
                            if (
                                output_config.show_surveillance
                                and location_surveillance_filtered_sbs
                                and location in location_surveillance_filtered_sbs
                            ):
                                surv_filtered = location_surveillance_filtered_sbs[location].copy()
                                if output_config.filtered_panel:
                                    surv_filtered = _clip_surveillance(
                                        surv_filtered,
                                        surveillance_start_date=output_config.filtered_panel.surveillance_start_date,
                                        surveillance_points=output_config.filtered_panel.surveillance_points,
                                        reference_date=plots_config.reference_date,
                                    )

                            # Clip filtered panel quantiles to visible surveillance start
                            cal_quant_filtered = _clip_to_surveillance_start(cal_quant, surv_filtered)
                            proj_quant_filtered = _clip_to_surveillance_start(proj_quant_filtered, surv_filtered)

                            fitting_window_start = (
                                location_fitting_window_starts.get(location)
                                if output_config.show_fitting_window_line and location_fitting_window_starts
                                else None
                            )
                            fitting_window_end = (
                                location_fitting_window_ends.get(location)
                                if output_config.show_fitting_window_line and location_fitting_window_ends
                                else None
                            )

                            # Use core helper to draw both panels on provided axes
                            # Get xlabel_interval for each panel from panel configs
                            x_interval_full = (
                                output_config.full_panel.xlabel_interval if output_config.full_panel else None
                            )
                            x_interval_filtered = (
                                output_config.filtered_panel.xlabel_interval if output_config.filtered_panel else None
                            )

                            sbs_title_suffix = location_notes.get(location, ("", ""))[0]
                            plot_calibration_projection_sidebyside(
                                calibration_quantiles=cal_quant,
                                calibration_quantiles_filtered=cal_quant_filtered,
                                projection_quantiles_full=proj_quant_full,
                                projection_quantiles_filtered=proj_quant_filtered,
                                surveillance_full=surv_full,
                                surveillance_filtered=surv_filtered,
                                value_col=value_col,
                                calibration_color=plots_config.quantiles.calibration.color,
                                projection_color=plots_config.quantiles.projection.color,
                                fitting_window_start=fitting_window_start,
                                fitting_window_end=fitting_window_end,
                                title=format_location_name(location) + sbs_title_suffix,
                                ax_full=ax_full,
                                ax_filtered=ax_filtered,
                                ylabel=plots_config.quantiles.ylabel if col_start == 0 else None,
                                xlabel_interval_full=x_interval_full,
                                xlabel_interval_filtered=x_interval_filtered,
                            )

                            # Hide legends except for leftmost column
                            # Only show legend on the left panel (col_start == 0)
                            if col_start != 0:
                                legend = ax_full.get_legend()
                                if legend is not None:
                                    legend.remove()
                            # Always hide legend on right panel (filtered)
                            legend = ax_filtered.get_legend()
                            if legend is not None:
                                legend.remove()

                        # Remove unused axes
                        for idx in range(n_locations * 2, nrows * ncols):
                            r = idx // ncols
                            c = idx % ncols
                            axes[r, c].axis("off")

                        if plots_config.quantiles.suptitle:
                            fig.suptitle(plots_config.quantiles.suptitle)

                        plt.tight_layout()

                        # Add generation notice footnote for side-by-side grid
                        sbs_footnotes = {fn for loc in locations for _, fn in [location_notes.get(loc, ("", ""))] if fn}
                        if sbs_footnotes:
                            _add_footnote(fig, "; ".join(sorted(sbs_footnotes)))

                        # Package output
                        out_dict["quantiles_grid_sidebyside"] = _package_figure_outputs(
                            fig, "quantiles_grid_sidebyside", plots_config
                        )
                        plt.close(fig)
                except Exception as e:
                    logger.warning("Failed to create sidebyside quantile grid plot: %s", e, exc_info=True)


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
