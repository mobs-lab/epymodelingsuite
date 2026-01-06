"""Plot generation orchestration for calibration/projection outputs.

This module contains high-level functions that orchestrate the creation of visualization outputs
by loading data, collecting quantiles/posteriors, calling visualization primitives, and packaging
outputs as OutputObject instances.
"""

import logging
import math
from datetime import date, timedelta

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
    _format_location_name,
    figure_to_output_object,
    plot_calibration_projection,
    plot_calibration_projection_grid,
    plot_calibration_projection_sidebyside,
    plot_posterior_histogram,
    plot_posterior_histogram_grid,
)

logger = logging.getLogger(__name__)


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
            cal_quant = calibration.results.get_calibration_quantiles(
                quantiles=plots_config.quantiles.quantiles,
                variables=["date", "data"],
            )
        except (ValueError, AttributeError, TypeError, IndexError) as e:
            logger.warning("Failed to get calibration quantiles for %s: %s", calibration.population, e)

    proj_quant = None
    if needs_projection:
        try:
            proj_quant = calibration.results.get_projection_quantiles(
                quantiles=plots_config.quantiles.quantiles,
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

    surv = get_data_in_location(surveillance, location, surveillance_config.location_column)

    # Full surveillance (no filtering)
    if not surv.empty:
        df_surv_full = surv.rename(
            columns={
                surveillance_config.date_column: "date",
                surveillance_config.value_column: "value",
            }
        )[["date", "value"]]

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
        df_surv_filtered = surv_filtered.rename(
            columns={
                surveillance_config.date_column: "date",
                surveillance_config.value_column: "value",
            }
        )[["date", "value"]]

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
    Return a copy with `old_name` renamed to `value` if the column exists.

    Parameters
    ----------
    df : pd.DataFrame or None
        DataFrame containing quantile data.
    old_name : str
        Column name to rename to 'value'.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with renamed column, or None if input is None.

    Raises
    ------
    ValueError
        If df is not None and old_name is not found in columns.
    """
    if df is None:
        return None

    if old_name in df.columns:
        return df.rename(columns={old_name: "value"})

    # Column not found - provide helpful error message
    metadata_cols = {"date", "location", "quantile"}
    available_value_cols = [c for c in df.columns if c not in metadata_cols]

    msg = (
        f"Column '{old_name}' not found in quantiles DataFrame. "
        f"Available value columns: {available_value_cols}. "
        f"Check output.plots.quantiles.value_column in your config matches "
        f"a transition in output.quantiles.transitions."
    )
    raise ValueError(msg)


def _create_filtered_plot(
    location: str,
    cal_quant: pd.DataFrame | None,
    proj_quant: pd.DataFrame | None,
    df_surv: pd.DataFrame | None,
    fitting_window_start: pd.Timestamp | None,
    fitting_window_end: pd.Timestamp | None,
    plots_config: PlotsConfig,
    value_col: str,
    output_config: QuantilesOutputConfig,
) -> tuple:
    """
    Create filtered quantile plot for a location.

    Parameters
    ----------
    location : str
        Location name
    cal_quant : pd.DataFrame or None
        Calibration quantiles
    proj_quant : pd.DataFrame or None
        Projection quantiles (filtered version)
    df_surv : pd.DataFrame or None
        Surveillance data (filtered version)
    fitting_window_start : pd.Timestamp or None
        Start of fitting window
    fitting_window_end : pd.Timestamp or None
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
        title=_format_location_name(location),
    )


def _create_full_plot(
    location: str,
    cal_quant: pd.DataFrame | None,
    proj_quant: pd.DataFrame | None,
    df_surv: pd.DataFrame | None,
    fitting_window_start: pd.Timestamp | None,
    fitting_window_end: pd.Timestamp | None,
    plots_config: PlotsConfig,
    value_col: str,
    output_config: QuantilesOutputConfig,
) -> tuple:
    """
    Create full quantile plot for a location.

    Parameters
    ----------
    location : str
        Location name
    cal_quant : pd.DataFrame or None
        Calibration quantiles
    proj_quant : pd.DataFrame or None
        Projection quantiles (full version)
    df_surv : pd.DataFrame or None
        Surveillance data (full version)
    fitting_window_start : pd.Timestamp or None
        Start of fitting window
    fitting_window_end : pd.Timestamp or None
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
        title=_format_location_name(location),
    )


def _create_sidebyside_plot(
    location: str,
    cal_quant: pd.DataFrame | None,
    proj_quant_full: pd.DataFrame | None,
    proj_quant_filtered: pd.DataFrame | None,
    df_surv_full: pd.DataFrame | None,
    df_surv_filtered: pd.DataFrame | None,
    fitting_window_start: pd.Timestamp | None,
    fitting_window_end: pd.Timestamp | None,
    plots_config: PlotsConfig,
    value_col: str,
    output_config: QuantilesOutputConfig,
) -> tuple:
    """
    Create side-by-side (full | filtered) quantile plot for a location.

    Parameters
    ----------
    location : str
        Location name
    cal_quant : pd.DataFrame or None
        Calibration quantiles (shown in both panels)
    proj_quant_full : pd.DataFrame or None
        Projection quantiles for left panel (full version)
    proj_quant_filtered : pd.DataFrame or None
        Projection quantiles for right panel (filtered version)
    df_surv_full : pd.DataFrame or None
        Surveillance data for left panel (full version)
    df_surv_filtered : pd.DataFrame or None
        Surveillance data for right panel (filtered version)
    fitting_window_start : pd.Timestamp or None
        Start of fitting window
    fitting_window_end : pd.Timestamp or None
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
        (fig, (ax_full, ax_filtered)) matplotlib figure and tuple of axes
    """
    return plot_calibration_projection_sidebyside(
        calibration_quantiles=cal_quant if output_config.show_calibration else None,
        projection_quantiles_full=proj_quant_full if output_config.show_projection else None,
        projection_quantiles_filtered=proj_quant_filtered if output_config.show_projection else None,
        surveillance_full=df_surv_full if output_config.show_surveillance else None,
        surveillance_filtered=df_surv_filtered if output_config.show_surveillance else None,
        value_col=value_col,
        calibration_color=plots_config.quantiles.calibration.color,
        projection_color=plots_config.quantiles.projection.color,
        fitting_window_start=fitting_window_start if output_config.show_fitting_window_line else None,
        fitting_window_end=fitting_window_end if output_config.show_fitting_window_line else None,
        title=_format_location_name(location),
        figsize=output_config.figsize,
        spacing=output_config.spacing,
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
    surveillance_data = {}
    needs_surveillance = any(output.show_surveillance for output in plots_config.quantiles.outputs)
    if needs_surveillance and surveillance_sources:
        for source_name, source_config in surveillance_sources.items():
            try:
                surveillance_data[source_name] = {
                    "data": pd.read_csv(source_config.data_path),
                    "config": source_config,
                }
            except Exception as e:
                logger.warning("Failed to load surveillance data from %s: %s", source_config.data_path, e)

    needs_calibration = any(output.show_calibration for output in plots_config.quantiles.outputs)
    needs_projection = any(output.show_projection for output in plots_config.quantiles.outputs)
    needs_fitting_window = any(output.show_fitting_window_line for output in plots_config.quantiles.outputs)

    # Generate plots for each location
    for calibration in calibrations:
        # Skip locations not in config
        if calibration.population not in locations:
            continue
        location = calibration.population

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
            cal_quant_for_fitting = cal_quant
            if cal_quant_for_fitting is None:
                try:
                    cal_quant_for_fitting = calibration.results.get_calibration_quantiles(
                        quantiles=[0.5],  # Only need one quantile to get dates
                        variables=["date", "data"],
                    )
                except (ValueError, AttributeError, TypeError) as e:
                    logger.warning(
                        "Failed to get calibration quantiles for fitting window for %s: %s",
                        location,
                        e,
                    )

            if cal_quant_for_fitting is not None and "date" in cal_quant_for_fitting.columns:
                dates = pd.to_datetime(cal_quant_for_fitting["date"]).dt.date
                fitting_window_start = dates.min()
                fitting_window_end = dates.max()

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
                if surv_to_use is not None:
                    # Date-based filtering takes precedence over points-based filtering
                    if output_config.surveillance_start_date is not None:
                        start_date = pd.to_datetime(output_config.surveillance_start_date).date()
                        surv_dates = pd.to_datetime(surv_to_use["date"]).dt.date
                        surv_to_use = surv_to_use[surv_dates >= start_date]
                    elif output_config.surveillance_points is not None:
                        surv_to_use = surv_to_use.sort_values("date").tail(output_config.surveillance_points)

                # Apply per-output horizon_max if specified (overrides base config)
                if proj_to_use is not None and output_config.horizon_max is not None:
                    proj_dates = pd.to_datetime(proj_to_use["date"]).dt.date
                    horizon_end = plots_config.reference_date + timedelta(weeks=output_config.horizon_max)
                    proj_to_use = proj_to_use[proj_dates.values <= horizon_end]

                # Create the plot
                if output_config.type == QuantilesOutputTypeEnum.FILTERED:
                    fig, ax = _create_filtered_plot(
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
                elif output_config.type == QuantilesOutputTypeEnum.FULL:
                    fig, ax = _create_full_plot(
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
                    fig, (ax_full, ax_filtered) = _create_sidebyside_plot(
                        location,
                        cal_quant,
                        proj_quant_full,
                        proj_quant_filtered,
                        df_surv_full,
                        df_surv_filtered,
                        fitting_window_start,
                        fitting_window_end,
                        plots_config,
                        value_col,
                        output_config,
                    )
                else:
                    logger.warning("Unknown output type %s for %s", output_config.type, location)
                    continue

                # Package output
                output_objs = []
                for figure_output_type in plots_config.figure_output_types:
                    output_objs.append(figure_to_output_object(fig, output_name, figure_output_type, plots_config.dpi))
                out_dict[output_name] = output_objs
                plt.close(fig)
            except Exception as e:
                logger.warning("Failed to create %s quantile plot for %s: %s", output_config.type.value, location, e)


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
    surveillance_data = {}
    needs_surveillance = any(output.show_surveillance for output in plots_config.quantiles.outputs)
    if needs_surveillance and surveillance_sources:
        for source_name, source_config in surveillance_sources.items():
            try:
                surveillance_data[source_name] = {
                    "data": pd.read_csv(source_config.data_path),
                    "config": source_config,
                }
            except Exception as e:
                logger.warning("Failed to load surveillance data from %s: %s", source_config.data_path, e)

    needs_calibration = any(output.show_calibration for output in plots_config.quantiles.outputs)
    needs_projection = any(output.show_projection for output in plots_config.quantiles.outputs)
    needs_fitting_window = any(output.show_fitting_window_line for output in plots_config.quantiles.outputs)

    # Collect quantiles and surveillance data for each location
    location_cal_quants = {}
    location_proj_quants_raw = {}
    location_fitting_window_starts = {}
    location_fitting_window_ends = {}

    # Collect quantiles for each location
    for calibration in calibrations:
        loc = calibration.population

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
            # Fetch calibration quantiles for fitting window calculation even if not displaying them
            cal_quant_for_fitting = location_cal_quants.get(loc)
            if cal_quant_for_fitting is None:
                try:
                    cal_quant_for_fitting = calibration.results.get_calibration_quantiles(
                        quantiles=[0.5],  # Only need one quantile to get dates
                        variables=["date", "data"],
                    )
                except (ValueError, AttributeError, TypeError, IndexError) as e:
                    logger.warning(
                        "Failed to get calibration quantiles for fitting window for %s: %s",
                        loc,
                        e,
                    )

            if cal_quant_for_fitting is not None and "date" in cal_quant_for_fitting.columns:
                dates = pd.to_datetime(cal_quant_for_fitting["date"]).dt.date
                location_fitting_window_starts[loc] = dates.min()
                location_fitting_window_ends[loc] = dates.max()

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
                # Date-based filtering takes precedence over points-based filtering
                if output_config.surveillance_start_date is not None:
                    surv_data_filtered_by_output = {}
                    start_date = pd.to_datetime(output_config.surveillance_start_date).date()
                    for loc, surv_df in surv_data_to_use.items():
                        surv_df_copy = surv_df.copy()
                        surv_dates = pd.to_datetime(surv_df_copy["date"]).dt.date
                        surv_data_filtered_by_output[loc] = surv_df_copy[surv_dates >= start_date]
                    surv_data_to_use = surv_data_filtered_by_output
                elif output_config.surveillance_points is not None:
                    surv_data_filtered_by_output = {}
                    for loc, surv_df in surv_data_to_use.items():
                        surv_data_filtered_by_output[loc] = surv_df.sort_values("date").tail(
                            output_config.surveillance_points
                        )
                    surv_data_to_use = surv_data_filtered_by_output

            # Apply per-output horizon_max if specified (overrides base config)
            if proj_quants_to_use and output_config.horizon_max is not None:
                proj_quants_horizon_filtered = {}
                for loc, proj_df in proj_quants_to_use.items():
                    proj_df_copy = proj_df.copy()
                    proj_dates = pd.to_datetime(proj_df_copy["date"]).dt.date
                    horizon_end = plots_config.reference_date + timedelta(weeks=output_config.horizon_max)
                    proj_quants_horizon_filtered[loc] = proj_df_copy[proj_dates.values <= horizon_end]
                proj_quants_to_use = proj_quants_horizon_filtered

            # Generate grid plot based on output type
            if output_type_name in ["filtered", "full"]:
                logger.info("    Creating %s grid plot", output_type_name)
                try:
                    fig, axes = plot_calibration_projection_grid(
                        location_calibration_quantiles=(
                            location_cal_quants if output_config.show_calibration and location_cal_quants else None
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
                    )

                    # Package output
                    output_objs = []
                    for fig_output_type in plots_config.figure_output_types:
                        output_objs.append(
                            figure_to_output_object(
                                fig, f"quantiles_grid_{output_type_name}", fig_output_type, plots_config.dpi
                            )
                        )
                    out_dict[f"quantiles_grid_{output_type_name}"] = output_objs
                    plt.close(fig)
                except Exception as e:
                    logger.warning(f"Failed to create {output_type_name} quantile grid plot: %s", e)

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

                    # Prepare projection quantiles
                    for loc, proj_quant_raw in location_proj_quants_raw.items():
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

                    # Get all locations
                    locations = set()
                    if location_cal_quants:
                        locations.update(location_cal_quants.keys())
                    if location_proj_quants_full_sbs:
                        locations.update(location_proj_quants_full_sbs.keys())
                    if location_proj_quants_filtered_sbs:
                        locations.update(location_proj_quants_filtered_sbs.keys())
                    locations = sorted(locations)

                    if locations:
                        n_locations = len(locations)
                        panels_per_row = plots_config.quantiles.grid.panels_per_row
                        pairs_per_row = panels_per_row // 2  # Each location needs 2 panels
                        nrows = math.ceil(n_locations / pairs_per_row)
                        ncols = panels_per_row

                        figsize = output_config.figsize if output_config.figsize else (4 * ncols, 3.6 * nrows)
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
                                    if output_config.full_panel.surveillance_start_date is not None:
                                        start_date = pd.to_datetime(
                                            output_config.full_panel.surveillance_start_date
                                        ).date()
                                        surv_dates = pd.to_datetime(surv_full["date"]).dt.date
                                        surv_full = surv_full[surv_dates >= start_date]
                                    elif output_config.full_panel.surveillance_points is not None:
                                        surv_full = surv_full.sort_values("date").tail(
                                            output_config.full_panel.surveillance_points
                                        )

                            surv_filtered = None
                            if (
                                output_config.show_surveillance
                                and location_surveillance_filtered_sbs
                                and location in location_surveillance_filtered_sbs
                            ):
                                surv_filtered = location_surveillance_filtered_sbs[location].copy()
                                if output_config.filtered_panel:
                                    if output_config.filtered_panel.surveillance_start_date is not None:
                                        start_date = pd.to_datetime(
                                            output_config.filtered_panel.surveillance_start_date
                                        ).date()
                                        surv_dates = pd.to_datetime(surv_filtered["date"]).dt.date
                                        surv_filtered = surv_filtered[surv_dates >= start_date]
                                    elif output_config.filtered_panel.surveillance_points is not None:
                                        surv_filtered = surv_filtered.sort_values("date").tail(
                                            output_config.filtered_panel.surveillance_points
                                        )
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
                            plot_calibration_projection_sidebyside(
                                calibration_quantiles=cal_quant,
                                projection_quantiles_full=proj_quant_full,
                                projection_quantiles_filtered=proj_quant_filtered,
                                surveillance_full=surv_full,
                                surveillance_filtered=surv_filtered,
                                value_col=value_col,
                                calibration_color=plots_config.quantiles.calibration.color,
                                projection_color=plots_config.quantiles.projection.color,
                                fitting_window_start=fitting_window_start,
                                fitting_window_end=fitting_window_end,
                                title=_format_location_name(location),
                                ax_full=ax_full,
                                ax_filtered=ax_filtered,
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

                        plt.tight_layout()

                        # Package output
                        output_objs = []
                        for fig_output_type in plots_config.figure_output_types:
                            output_objs.append(
                                figure_to_output_object(
                                    fig, "quantiles_grid_sidebyside", fig_output_type, plots_config.dpi
                                )
                            )
                        out_dict["quantiles_grid_sidebyside"] = output_objs
                        plt.close(fig)
                except Exception as e:
                    logger.warning(f"Failed to create sidebyside quantile grid plot: {e}")


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
            params = [col for col in posterior_df.columns if col not in ["sim_id", "location"]]

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
                    logger.warning("Failed to create posterior histogram for %s - %s: %s", location, param, e)
                    axes[idx].set_visible(False)

            # Hide unused subplots
            for idx in range(n_params, len(axes)):
                axes[idx].set_visible(False)

            fig.suptitle(f"Posterior Distributions - {location}", fontsize=14, y=0.995)
            fig.tight_layout()

            # Package output
            output_objs = []
            for output_type in plots_config.figure_output_types:
                output_objs.append(figure_to_output_object(fig, f"posterior_{location}", output_type, plots_config.dpi))
            out_dict[f"posterior_{location}"] = output_objs
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

    # Collect posterior distributions for each location
    for calibration in calibrations:
        loc = calibration.population
        try:
            posterior_df = calibration.results.get_posterior_distribution()
            location_posteriors[loc] = posterior_df
            all_params.update(col for col in posterior_df.columns if col not in ["sim_id", "location"])
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

            # Package output
            output_objs = []
            for output_type in plots_config.figure_output_types:
                output_objs.append(figure_to_output_object(fig, "posterior_grid", output_type, plots_config.dpi))
            out_dict["posterior_grid"] = output_objs
            plt.close(fig)
        except Exception as e:
            logger.warning("Failed to create posterior grid plot: %s", e)
