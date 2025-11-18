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
    if plots_config.quantiles.surveillance.show and plots_config.quantiles.surveillance.data_path:
        try:
            surveillance = pd.read_csv(plots_config.quantiles.surveillance.data_path)
        except Exception as e:
            logger.warning(
                "Failed to load surveillance data from %s: %s", plots_config.quantiles.surveillance.data_path, e
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
        if plots_config.quantiles.calibration.show:
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

        # Projection quantiles
        # TODO: "hospitalizations" column is hardcoded in projection quantiles
        proj_quant = None
        if plots_config.quantiles.projection.show:
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

        # Filter surveillance data for this location
        df_surv = None
        if surveillance is not None and plots_config.quantiles.surveillance.location_column:
            surv = get_data_in_location(surveillance, location, plots_config.quantiles.surveillance.location_column)
            if (
                not surv.empty
                and plots_config.quantiles.surveillance.date_column
                and plots_config.quantiles.surveillance.value_column
            ):
                df_surv = surv.rename(
                    columns={
                        plots_config.quantiles.surveillance.date_column: "date",
                        plots_config.quantiles.surveillance.value_column: "value",
                    }
                )[["date", "value"]]

        # Rename columns to have consistent naming for plotting
        # TODO: Calibration uses "data", projection uses "hospitalizations" - make this configurable
        if cal_quant is not None and "data" in cal_quant.columns:
            cal_quant = cal_quant.rename(columns={"data": "value"})
        if proj_quant is not None and "hospitalizations" in proj_quant.columns:
            proj_quant = proj_quant.rename(columns={"hospitalizations": "value"})

        value_col = "value"

        # Plot and add it to output dict
        try:
            fig, ax = plot_calibration_projection(
                calibration_quantiles=cal_quant,
                projection_quantiles=proj_quant,
                value_col=value_col,
                calibration_color=plots_config.quantiles.calibration.color,
                projection_color=plots_config.quantiles.projection.color,
                df_surveillance=df_surv,
                reference_date=(plots_config.reference_date if plots_config.quantiles.reference_line.show else None),
                title=location,
            )

            # Package output
            output_obj = figure_to_output_object(fig, f"quantiles_{location}", plots_config.format, plots_config.dpi)
            out_dict[f"quantiles_{location}"] = [output_obj]
            plt.close(fig)
        except Exception as e:
            logger.warning("Failed to create quantile plot for %s: %s", location, e)


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

    if not plots_config.quantiles.grid.enabled:
        return

    # Load surveillance data once before loop
    surveillance = None
    if plots_config.quantiles.surveillance.show and plots_config.quantiles.surveillance.data_path:
        try:
            surveillance = pd.read_csv(plots_config.quantiles.surveillance.data_path)
        except Exception as e:
            logger.warning(
                "Failed to load surveillance data from %s: %s", plots_config.quantiles.surveillance.data_path, e
            )

    # Collect quantiles and surveillance data for each location
    location_cal_quants = {}
    location_proj_quants = {}
    location_surveillance = {}

    # Collect quantiles for each location
    for calibration in calibrations:
        loc = calibration.population

        # Filter failed calibration trajectories and projections
        # calibration.results = filter_failed_calibration_trajectories(calibration.results)
        calibration.results = filter_failed_projections(calibration.results)

        # Get pre-computed quantiles

        # Calibration quantiles
        if plots_config.quantiles.calibration.show:
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

        # Projection quantiles
        # TODO: "hospitalizations" column is hardcoded in projection quantiles
        if plots_config.quantiles.projection.show:
            try:
                location_proj_quants[loc] = calibration.results.get_projection_quantiles(
                    quantiles=plots_config.quantiles.quantiles,
                )
            except (ValueError, AttributeError, TypeError, IndexError) as e:
                logger.warning(
                    "Failed to get projection quantiles for %s: %s",
                    loc,
                    e,
                )

        # Filter surveillance data for this location
        if surveillance is not None and plots_config.quantiles.surveillance.location_column:
            surv = get_data_in_location(surveillance, loc, plots_config.quantiles.surveillance.location_column)
            if (
                not surv.empty
                and plots_config.quantiles.surveillance.date_column
                and plots_config.quantiles.surveillance.value_column
            ):
                location_surveillance[loc] = surv.rename(
                    columns={
                        plots_config.quantiles.surveillance.date_column: "date",
                        plots_config.quantiles.surveillance.value_column: "value",
                    }
                )[["date", "value"]]

    # Go over collected data, plot grid, and add to output dict
    if location_cal_quants or location_proj_quants:
        # Rename columns to have consistent naming for plotting
        # TODO: Calibration uses "data", projection uses "hospitalizations" - make this configurable
        for loc in location_cal_quants:
            if "data" in location_cal_quants[loc].columns:
                location_cal_quants[loc] = location_cal_quants[loc].rename(columns={"data": "value"})
        for loc in location_proj_quants:
            if "hospitalizations" in location_proj_quants[loc].columns:
                location_proj_quants[loc] = location_proj_quants[loc].rename(columns={"hospitalizations": "value"})

        value_col = "value"

        try:
            fig, axes = plot_calibration_projection_grid(
                location_calibration_quantiles=location_cal_quants if location_cal_quants else None,
                location_projection_quantiles=location_proj_quants if location_proj_quants else None,
                value_col=value_col,
                calibration_color=plots_config.quantiles.calibration.color,
                projection_color=plots_config.quantiles.projection.color,
                location_surveillance=location_surveillance if location_surveillance else None,
                reference_date=(plots_config.reference_date if plots_config.quantiles.reference_line.show else None),
                panels_per_row=plots_config.quantiles.grid.panels_per_row,
            )

            output_obj = figure_to_output_object(fig, "quantiles_grid", plots_config.format, plots_config.dpi)
            out_dict["quantiles_grid"] = [output_obj]
            plt.close(fig)
        except Exception as e:
            logger.warning("Failed to create quantile grid plot: %s", e)


def generate_single_location_posterior_plots(
    calibrations: list[CalibrationOutput],
    plots_config: PlotsConfig,
    out_dict: dict[str, list[OutputObject]],
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

            output_obj = figure_to_output_object(fig, f"posterior_{location}", plots_config.format, plots_config.dpi)
            out_dict[f"posterior_{location}"] = [output_obj]
            plt.close(fig)

        except (ValueError, AttributeError) as e:
            logger.warning("Failed to get posterior distribution for %s: %s", location, e)


def generate_posterior_grid_plot(
    calibrations: list[CalibrationOutput],
    plots_config: PlotsConfig,
    out_dict: dict[str, list[OutputObject]],
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
            )

            output_obj = figure_to_output_object(fig, "posterior_grid", plots_config.format, plots_config.dpi)
            out_dict["posterior_grid"] = [output_obj]
            plt.close(fig)
        except Exception as e:
            logger.warning("Failed to create posterior grid plot: %s", e)
