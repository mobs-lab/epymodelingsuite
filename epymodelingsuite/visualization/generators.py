"""Plot generation orchestration for calibration/projection outputs.

This module contains high-level functions that orchestrate the creation of visualization outputs
by loading data, collecting quantiles/posteriors, calling visualization primitives, and packaging
outputs as OutputObject instances.

Quantile plots have three separate choices: location layout (single/grid), plot
variant (full/filtered/side_by_side), and data layers (calibration/projection/surveillance).
preparation.py resolves the settings and builds the PanelPlotData objects once.
Both layouts draw those same prepared frames with plot_quantile_panel.
"""

import logging
import math

import matplotlib.pyplot as plt
import pandas as pd

from ..schema.dispatcher import CalibrationOutput
from ..schema.output import (
    ObservedValuesConfig,
    OutputObject,
    PlotsConfig,
    QuantilesOutputConfig,
    QuantilesOutputTypeEnum,
    QuantilesPlotConfig,
)
from .core import (
    figure_to_output_object,
    format_location_name,
    plot_categorical_stacked_bars_multihorizon,
    plot_posterior_histogram,
    plot_posterior_histogram_grid,
    plot_quantile_panel,
    sort_locations_by_state,
)
from .preparation import (
    PanelPlotData,
    _check_incomplete_generations,
    _format_plot_notes,
    prepare_quantile_plots,
)

logger = logging.getLogger(__name__)

# Columns that are metadata/identifiers and should be excluded when extracting parameter names
POSTERIOR_METADATA_COLUMNS = {"sim_id", "location", "population", "primary_id", "seed"}


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


def _draw_quantile_panel(panel_data: PanelPlotData, ax: plt.Axes, ylabel: str | None) -> None:
    """Draw prepared frames and styling; all selection and filtering is already complete."""
    plot_quantile_panel(
        calibration_quantiles=panel_data.calibration_quantiles,
        projection_quantiles=panel_data.projection_quantiles,
        df_surveillance=panel_data.surveillance,
        value_col="value",
        calibration_color=panel_data.calibration_color,
        projection_color=panel_data.projection_color,
        fitting_window_start=panel_data.fitting_window_start,
        fitting_window_end=panel_data.fitting_window_end,
        xlabel_interval=panel_data.xlabel_interval,
        title=format_location_name(panel_data.location) + panel_data.title_suffix,
        ylabel=ylabel,
        ax=ax,
    )


def _plot_location_quantiles(
    panel_data: list[PanelPlotData],
    variant_config: QuantilesOutputConfig,
    ylabel: str | None,
) -> plt.Figure:
    """Arrange one location's prepared panels in one figure."""
    side_by_side = len(panel_data) == 2
    fig, axes = plt.subplots(
        1,
        len(panel_data),
        squeeze=False,
        figsize=(variant_config.figsize or (16, 6)) if side_by_side else None,
        gridspec_kw={"wspace": variant_config.spacing} if side_by_side else None,
    )
    try:
        for index, panel in enumerate(panel_data):
            _draw_quantile_panel(panel, axes[0, index], ylabel if index == 0 else None)
        _add_footnote(fig, panel_data[0].footnote)
    except Exception:
        plt.close(fig)
        raise
    return fig


def _plot_quantiles_grid(
    panel_data_by_location: dict[str, list[PanelPlotData]],
    variant_config: QuantilesOutputConfig,
    plot_config: QuantilesPlotConfig,
) -> plt.Figure:
    """Arrange prepared panels by location; a side-by-side pair stays in the same row."""
    locations = sort_locations_by_state(panel_data_by_location)
    panels_per_location = len(panel_data_by_location[locations[0]])
    ncols = plot_config.grid.panels_per_row
    n_panels = len(locations) * panels_per_location
    nrows = math.ceil(n_panels / ncols)
    figsize = variant_config.figsize if panels_per_location == 2 else None
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (4 * ncols, 3.6 * nrows), squeeze=False)
    try:
        for location_index, location in enumerate(locations):
            for panel_index, panel in enumerate(panel_data_by_location[location]):
                index = location_index * panels_per_location + panel_index
                ax = axes.flat[index]
                leftmost = index % ncols == 0
                _draw_quantile_panel(panel, ax, plot_config.ylabel if leftmost else None)
                if not leftmost and (legend := ax.get_legend()) is not None:
                    legend.remove()
        for ax in axes.flat[n_panels:]:
            ax.axis("off")
        if plot_config.suptitle:
            fig.suptitle(plot_config.suptitle)
        fig.tight_layout()
        footnotes = {panels[0].footnote for panels in panel_data_by_location.values()} - {""}
        if footnotes:
            _add_footnote(fig, "; ".join(sorted(footnotes)))
    except Exception:
        plt.close(fig)
        raise
    return fig


GRID_OUTPUT_NAMES = {
    QuantilesOutputTypeEnum.FILTERED: "quantiles_grid_filtered",
    QuantilesOutputTypeEnum.FULL: "quantiles_grid_full",
    QuantilesOutputTypeEnum.SIDE_BY_SIDE: "quantiles_grid_sidebyside",
}


def generate_quantile_plots(
    calibrations: list[CalibrationOutput],
    plots_config: PlotsConfig,
    out_dict: dict[str, list[OutputObject]],
    surveillance_sources: dict[str, ObservedValuesConfig] | None = None,
) -> None:
    """Prepare quantile panels once, then render the requested single and grid figures.

    The input calibration results are read-only. Both layouts use the same prepared
    frames, styles and notes. Data preparation and drawing failures are isolated so
    other locations/variants can still be generated.
    """
    plot_config = plots_config.quantiles
    single_locations = get_locations_to_plot(calibrations, plot_config.single)
    if not single_locations and not plot_config.grid.enabled:
        return
    selected_calibrations = [
        calibration
        for calibration in calibrations
        if plot_config.grid.enabled or calibration.population in single_locations
    ]
    panels_by_variant = prepare_quantile_plots(selected_calibrations, plots_config, surveillance_sources)
    for variant_config in plot_config.outputs:
        panel_data_by_location = panels_by_variant.get(variant_config.type, {})
        for location, panel_data in panel_data_by_location.items():
            if location not in single_locations:
                continue
            output_name = f"quantiles_{location}_{variant_config.type.value}"
            fig = None
            try:
                fig = _plot_location_quantiles(panel_data, variant_config, plot_config.ylabel)
                out_dict[output_name] = _package_figure_outputs(fig, output_name, plots_config)
            except Exception as e:
                logger.warning("Failed to create %s: %s", output_name, e, exc_info=True)
            finally:
                if fig is not None:
                    plt.close(fig)

        if plot_config.grid.enabled and panel_data_by_location:
            output_name = GRID_OUTPUT_NAMES[variant_config.type]
            fig = None
            try:
                fig = _plot_quantiles_grid(panel_data_by_location, variant_config, plot_config)
                out_dict[output_name] = _package_figure_outputs(fig, output_name, plots_config)
            except Exception as e:
                logger.warning("Failed to create %s: %s", output_name, e, exc_info=True)
            finally:
                if fig is not None:
                    plt.close(fig)


def generate_single_quantile_plots(
    calibrations: list[CalibrationOutput],
    plots_config: PlotsConfig,
    out_dict: dict[str, list[OutputObject]],
    surveillance_sources: dict[str, ObservedValuesConfig] | None = None,
) -> None:
    """Generate only single-location figures; use generate_quantile_plots for both layouts."""
    plot_config = plots_config.quantiles.model_copy(
        update={"grid": plots_config.quantiles.grid.model_copy(update={"enabled": False})}
    )
    generate_quantile_plots(
        calibrations, plots_config.model_copy(update={"quantiles": plot_config}), out_dict, surveillance_sources
    )


def generate_quantile_grid_plot(
    calibrations: list[CalibrationOutput],
    plots_config: PlotsConfig,
    out_dict: dict[str, list[OutputObject]],
    surveillance_sources: dict[str, ObservedValuesConfig] | None = None,
) -> None:
    """Generate only grid figures; use generate_quantile_plots for both layouts."""
    plot_config = plots_config.quantiles.model_copy(update={"single": False})
    generate_quantile_plots(
        calibrations, plots_config.model_copy(update={"quantiles": plot_config}), out_dict, surveillance_sources
    )


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
