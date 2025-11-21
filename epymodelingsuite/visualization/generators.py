"""Plot generation orchestration for calibration/projection outputs.

This module contains high-level functions that orchestrate the creation of visualization outputs
by loading data, collecting quantiles/posteriors, calling visualization primitives, and packaging
outputs as OutputObject instances.
"""

import logging
import math

import matplotlib.pyplot as plt
import pandas as pd

from ..builders.utils import get_data_in_location
from ..schema.dispatcher import CalibrationOutput
from ..schema.output import OutputObject, PlotsConfig
from .core import (
    _format_location_name,
    figure_to_output_object,
    plot_calibration_projection,
    plot_calibration_projection_grid,
    plot_posterior_histogram,
    plot_posterior_histogram_grid,
)

logger = logging.getLogger(__name__)


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

    # Load surveillance data once before loop
    surveillance = None
    if plots_config.quantiles.surveillance:
        try:
            surveillance = pd.read_csv(plots_config.quantiles.surveillance.source.data_path)
        except Exception as e:
            logger.warning(
                "Failed to load surveillance data from %s: %s",
                plots_config.quantiles.surveillance.source.data_path,
                e,
            )

    # Generate plots for each location
    for calibration in calibrations:
        # Skip locations not in config
        if calibration.population not in locations:
            continue
        location = calibration.population

        # Filter failed calibration trajectories and projections
        # calibration.results = filter_failed_calibration_trajectories(calibration.results)
        calibration.results = filter_failed_projections(calibration.results)

        # Get pre-computed quantiles

        # Calibration quantiles
        cal_quant = None
        if plots_config.quantiles.calibration:
            try:
                cal_quant = calibration.results.get_calibration_quantiles(
                    quantiles=plots_config.quantiles.quantiles,
                    variables=["date", "data"],
                )
            except (ValueError, AttributeError, TypeError) as e:
                logger.warning(
                    "Failed to get calibration quantiles for %s: %s",
                    location,
                    e,
                )

        # Calculate fitting window start and end from calibration quantiles
        fitting_window_start = None
        fitting_window_end = None
        if plots_config.quantiles.fitting_window_line.show:
            # Fetch calibration quantiles for fitting window calculation even if not displaying them
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

        # Projection quantiles
        # TODO: "hospitalizations" column is hardcoded in projection quantiles
        proj_quant = None
        if plots_config.quantiles.projection:
            try:
                proj_quant = calibration.results.get_projection_quantiles(
                    quantiles=plots_config.quantiles.quantiles,
                )
            except (ValueError, AttributeError) as e:
                logger.warning(
                    "Failed to get projection quantiles for %s: %s",
                    location,
                    e,
                )

        # Prepare surveillance data for this location
        df_surv_full = None  # Full surveillance data (no filtering)
        df_surv_filtered = None  # Filtered surveillance data
        surveillance_start_date = None
        if surveillance is not None:
            surv = get_data_in_location(surveillance, location, plots_config.quantiles.surveillance.source.location_column)

            # Create full surveillance dataframe (no filtering)
            if not surv.empty:
                df_surv_full = surv.rename(
                    columns={
                        plots_config.quantiles.surveillance.source.date_column: "date",
                        plots_config.quantiles.surveillance.source.value_column: "value",
                    }
                )[["date", "value"]]

            # Create filtered surveillance dataframe
            surv_filtered = surv.copy()

            # Filter surveillance data to start from timespan start (inferred from projection or calibration quantiles)
            if not surv_filtered.empty:
                # Use projection quantiles to get full timespan, fall back to calibration if not available
                quantiles_for_timespan = proj_quant if proj_quant is not None else cal_quant
                if quantiles_for_timespan is not None and "date" in quantiles_for_timespan.columns:
                    # Infer timespan start from the earliest date in quantiles
                    timespan_start = pd.to_datetime(quantiles_for_timespan["date"]).min().date()
                    # Filter surveillance data to start from timespan start
                    surv_filtered = surv_filtered[
                        pd.to_datetime(surv_filtered[plots_config.quantiles.surveillance.source.date_column]).dt.date
                        >= timespan_start
                    ]

                df_surv_filtered = surv_filtered.rename(
                    columns={
                        plots_config.quantiles.surveillance.source.date_column: "date",
                        plots_config.quantiles.surveillance.source.value_column: "value",
                    }
                )[["date", "value"]]

                # Filter to most recent N points if surveillance_points is set
                if plots_config.quantiles.surveillance.surveillance_points is not None:
                    df_surv_filtered = df_surv_filtered.sort_values("date").tail(
                        plots_config.quantiles.surveillance.surveillance_points
                    )

                # Get first surveillance date for projection filtering
                if not df_surv_filtered.empty:
                    surveillance_start_date = pd.to_datetime(df_surv_filtered["date"]).min().date()

        # Prepare projection quantiles - create both filtered and full versions
        proj_quant_filtered = None
        proj_quant_full = None

        if proj_quant is not None:
            from datetime import timedelta

            # Full version: apply horizon_max only (no surveillance_start_date filtering)
            proj_quant_full = proj_quant.copy()
            if plots_config.quantiles.horizon_max is not None:
                proj_dates_full = pd.to_datetime(proj_quant_full["date"]).dt.date
                horizon_end_full = plots_config.reference_date + timedelta(weeks=plots_config.quantiles.horizon_max)
                proj_quant_full = proj_quant_full[proj_dates_full.values <= horizon_end_full]

            # Filtered version: apply standard filtering
            proj_quant_filtered = proj_quant.copy()
            proj_dates = pd.to_datetime(proj_quant_filtered["date"]).dt.date

            # Start at first surveillance point if there's projection data before it
            if surveillance_start_date is not None:
                proj_start = proj_dates.min()
                # Only filter if projection starts before first surveillance point
                if proj_start < surveillance_start_date:
                    proj_quant_filtered = proj_quant_filtered[proj_dates.values >= surveillance_start_date]

            # End at horizon_max if specified
            if plots_config.quantiles.horizon_max is not None:
                horizon_end = plots_config.reference_date + timedelta(weeks=plots_config.quantiles.horizon_max)
                proj_dates = pd.to_datetime(proj_quant_filtered["date"]).dt.date
                proj_quant_filtered = proj_quant_filtered[proj_dates.values <= horizon_end]

        # Rename columns to have consistent naming for plotting
        # TODO: Calibration uses "data", projection uses "hospitalizations" - make this configurable
        if cal_quant is not None and "data" in cal_quant.columns:
            cal_quant = cal_quant.rename(columns={"data": "value"})
        if proj_quant_filtered is not None and "hospitalizations" in proj_quant_filtered.columns:
            proj_quant_filtered = proj_quant_filtered.rename(columns={"hospitalizations": "value"})
        if proj_quant_full is not None and "hospitalizations" in proj_quant_full.columns:
            proj_quant_full = proj_quant_full.rename(columns={"hospitalizations": "value"})

        value_col = "value"

        # Create filtered plot
        try:
            fig, ax = plot_calibration_projection(
                calibration_quantiles=cal_quant,
                projection_quantiles=proj_quant_filtered,
                value_col=value_col,
                calibration_color=plots_config.quantiles.calibration.color,
                projection_color=plots_config.quantiles.projection.color,
                df_surveillance=df_surv_filtered,
                fitting_window_start=(
                    fitting_window_start if plots_config.quantiles.fitting_window_line.show else None
                ),
                fitting_window_end=(fitting_window_end if plots_config.quantiles.fitting_window_line.show else None),
                title=_format_location_name(location),
            )

            # Package output
            output_objs = []
            for output_type in plots_config.figure_output_types:
                output_objs.append(
                    figure_to_output_object(fig, f"quantiles_{location}_filtered", output_type, plots_config.dpi)
                )
            out_dict[f"quantiles_{location}_filtered"] = output_objs
            plt.close(fig)
        except Exception as e:
            logger.warning("Failed to create filtered quantile plot for %s: %s", location, e)

        # Create full plot
        try:
            fig, ax = plot_calibration_projection(
                calibration_quantiles=cal_quant,
                projection_quantiles=proj_quant_full,
                value_col=value_col,
                calibration_color=plots_config.quantiles.calibration.color,
                projection_color=plots_config.quantiles.projection.color,
                df_surveillance=df_surv_full,
                fitting_window_start=(
                    fitting_window_start if plots_config.quantiles.fitting_window_line.show else None
                ),
                fitting_window_end=(fitting_window_end if plots_config.quantiles.fitting_window_line.show else None),
                title=_format_location_name(location),
            )

            # Package output
            output_objs = []
            for output_type in plots_config.figure_output_types:
                output_objs.append(
                    figure_to_output_object(fig, f"quantiles_{location}_full", output_type, plots_config.dpi)
                )
            out_dict[f"quantiles_{location}_full"] = output_objs
            plt.close(fig)
        except Exception as e:
            logger.warning("Failed to create full quantile plot for %s: %s", location, e)


def generate_quantile_grid_plot(
    calibrations: list[CalibrationOutput],
    plots_config: PlotsConfig,
    out_dict: dict[str, list[OutputObject]],
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

    # Load surveillance data once before loop
    surveillance = None
    if plots_config.quantiles.show_surveillance:
        try:
            surveillance = pd.read_csv(plots_config.quantiles.surveillance.source.data_path)
        except Exception as e:
            logger.warning(
                "Failed to load surveillance data from %s: %s",
                plots_config.quantiles.surveillance.source.data_path,
                e,
            )

    # Collect quantiles and surveillance data for each location
    location_cal_quants = {}
    location_proj_quants_filtered = {}
    location_proj_quants_full = {}
    location_surveillance_filtered = {}
    location_surveillance_full = {}
    location_fitting_window_starts = {}
    location_fitting_window_ends = {}

    # Collect quantiles for each location
    for calibration in calibrations:
        loc = calibration.population

        # Filter failed calibration trajectories and projections
        # calibration.results = filter_failed_calibration_trajectories(calibration.results)
        calibration.results = filter_failed_projections(calibration.results)

        # Get pre-computed quantiles

        # Calibration quantiles
        if plots_config.quantiles.calibration:
            try:
                location_cal_quants[loc] = calibration.results.get_calibration_quantiles(
                    quantiles=plots_config.quantiles.quantiles,
                    variables=["date", "data"],
                )
            except (ValueError, AttributeError, TypeError, IndexError) as e:
                logger.warning(
                    "Failed to get calibration quantiles for %s: %s",
                    loc,
                    e,
                )

        # Calculate fitting window start and end from calibration quantiles
        if plots_config.quantiles.fitting_window_line.show:
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

        # Projection quantiles
        # TODO: "hospitalizations" column is hardcoded in projection quantiles
        proj_quant_raw = None
        if plots_config.quantiles.projection:
            try:
                proj_quant_raw = calibration.results.get_projection_quantiles(
                    quantiles=plots_config.quantiles.quantiles,
                )
            except (ValueError, AttributeError, TypeError, IndexError) as e:
                logger.warning(
                    "Failed to get projection quantiles for %s: %s",
                    loc,
                    e,
                )

        # Prepare surveillance data for this location
        surveillance_start_date = None
        if surveillance is not None:
            surv = get_data_in_location(surveillance, loc, plots_config.quantiles.surveillance.source.location_column)

            # Create full surveillance dataframe (no filtering)
            if not surv.empty:
                location_surveillance_full[loc] = surv.rename(
                    columns={
                        plots_config.quantiles.surveillance.source.date_column: "date",
                        plots_config.quantiles.surveillance.source.value_column: "value",
                    }
                )[["date", "value"]]

            # Create filtered surveillance dataframe
            surv_filtered = surv.copy()

            # Filter surveillance data to start from timespan start (inferred from projection or calibration quantiles)
            if not surv_filtered.empty:
                # Use projection quantiles to get full timespan, fall back to calibration if not available
                quantiles_for_timespan = proj_quant_raw if proj_quant_raw is not None else location_cal_quants.get(loc)

                if quantiles_for_timespan is not None and "date" in quantiles_for_timespan.columns:
                    # Infer timespan start from the earliest date in quantiles
                    timespan_start = pd.to_datetime(quantiles_for_timespan["date"]).min().date()
                    # Filter surveillance data to start from timespan start
                    surv_filtered = surv_filtered[
                        pd.to_datetime(surv_filtered[plots_config.quantiles.surveillance.source.date_column]).dt.date
                        >= timespan_start
                    ]

            if not surv_filtered.empty:
                location_surveillance_filtered[loc] = surv_filtered.rename(
                    columns={
                        plots_config.quantiles.surveillance.source.date_column: "date",
                        plots_config.quantiles.surveillance.source.value_column: "value",
                    }
                )[["date", "value"]]

                # Filter to most recent N points if surveillance_points is set
                if plots_config.quantiles.surveillance.surveillance_points is not None:
                    location_surveillance_filtered[loc] = (
                        location_surveillance_filtered[loc]
                        .sort_values("date")
                        .tail(plots_config.quantiles.surveillance.surveillance_points)
                    )

                # Get first surveillance date for projection filtering
                if not location_surveillance_filtered[loc].empty:
                    surveillance_start_date = pd.to_datetime(location_surveillance_filtered[loc]["date"]).min().date()

        # Prepare projection quantiles - create both filtered and full versions
        if proj_quant_raw is not None:
            from datetime import timedelta

            # Full version: apply horizon_max only (no surveillance_start_date filtering)
            location_proj_quants_full[loc] = proj_quant_raw.copy()
            if plots_config.quantiles.horizon_max is not None:
                proj_dates_full = pd.to_datetime(location_proj_quants_full[loc]["date"]).dt.date
                horizon_end_full = plots_config.reference_date + timedelta(weeks=plots_config.quantiles.horizon_max)
                location_proj_quants_full[loc] = location_proj_quants_full[loc][
                    proj_dates_full.values <= horizon_end_full
                ]

            # Filtered version: apply standard filtering
            location_proj_quants_filtered[loc] = proj_quant_raw.copy()
            proj_dates = pd.to_datetime(location_proj_quants_filtered[loc]["date"]).dt.date

            # Start at first surveillance point if there's projection data before it
            if surveillance_start_date is not None:
                proj_start = proj_dates.min()
                # Only filter if projection starts before first surveillance point
                if proj_start < surveillance_start_date:
                    location_proj_quants_filtered[loc] = location_proj_quants_filtered[loc][
                        proj_dates.values >= surveillance_start_date
                    ]

            # End at horizon_max if specified
            if plots_config.quantiles.horizon_max is not None:
                horizon_end = plots_config.reference_date + timedelta(weeks=plots_config.quantiles.horizon_max)
                proj_dates = pd.to_datetime(location_proj_quants_filtered[loc]["date"]).dt.date
                location_proj_quants_filtered[loc] = location_proj_quants_filtered[loc][
                    proj_dates.values <= horizon_end
                ]

    # Go over collected data, plot grid, and add to output dict
    if location_cal_quants or location_proj_quants_filtered or location_proj_quants_full:
        # Rename columns to have consistent naming for plotting
        # TODO: Calibration uses "data", projection uses "hospitalizations" - make this configurable
        for loc in location_cal_quants:
            if "data" in location_cal_quants[loc].columns:
                location_cal_quants[loc] = location_cal_quants[loc].rename(columns={"data": "value"})
        for loc in location_proj_quants_filtered:
            if "hospitalizations" in location_proj_quants_filtered[loc].columns:
                location_proj_quants_filtered[loc] = location_proj_quants_filtered[loc].rename(
                    columns={"hospitalizations": "value"}
                )
        for loc in location_proj_quants_full:
            if "hospitalizations" in location_proj_quants_full[loc].columns:
                location_proj_quants_full[loc] = location_proj_quants_full[loc].rename(
                    columns={"hospitalizations": "value"}
                )

        value_col = "value"

        # Create filtered grid plot
        try:
            fig, axes = plot_calibration_projection_grid(
                location_calibration_quantiles=location_cal_quants if location_cal_quants else None,
                location_projection_quantiles=location_proj_quants_filtered if location_proj_quants_filtered else None,
                value_col=value_col,
                calibration_color=plots_config.quantiles.calibration.color,
                projection_color=plots_config.quantiles.projection.color,
                location_surveillance=location_surveillance_filtered if location_surveillance_filtered else None,
                location_fitting_window_starts=(
                    location_fitting_window_starts if plots_config.quantiles.fitting_window_line.show else None
                ),
                location_fitting_window_ends=(
                    location_fitting_window_ends if plots_config.quantiles.fitting_window_line.show else None
                ),
                panels_per_row=plots_config.quantiles.grid.panels_per_row,
            )

            # Package output
            output_objs = []
            for output_type in plots_config.figure_output_types:
                output_objs.append(
                    figure_to_output_object(fig, "quantiles_grid_filtered", output_type, plots_config.dpi)
                )
            out_dict["quantiles_grid_filtered"] = output_objs
            plt.close(fig)
        except Exception as e:
            logger.warning("Failed to create filtered quantile grid plot: %s", e)

        # Create full grid plot
        try:
            fig, axes = plot_calibration_projection_grid(
                location_calibration_quantiles=location_cal_quants if location_cal_quants else None,
                location_projection_quantiles=location_proj_quants_full if location_proj_quants_full else None,
                value_col=value_col,
                calibration_color=plots_config.quantiles.calibration.color,
                projection_color=plots_config.quantiles.projection.color,
                location_surveillance=location_surveillance_full if location_surveillance_full else None,
                location_fitting_window_starts=(
                    location_fitting_window_starts if plots_config.quantiles.fitting_window_line.show else None
                ),
                location_fitting_window_ends=(
                    location_fitting_window_ends if plots_config.quantiles.fitting_window_line.show else None
                ),
                panels_per_row=plots_config.quantiles.grid.panels_per_row,
            )

            # Package output
            output_objs = []
            for output_type in plots_config.figure_output_types:
                output_objs.append(figure_to_output_object(fig, "quantiles_grid_full", output_type, plots_config.dpi))
            out_dict["quantiles_grid_full"] = output_objs
            plt.close(fig)
        except Exception as e:
            logger.warning("Failed to create full quantile grid plot: %s", e)


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

    # Generate plots for each location
    for calibration in calibrations:
        if calibration.population not in locations:
            continue

        location = calibration.population

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
