"""Visualization functions for calibration/projection outputs."""

import io
from datetime import datetime
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..schema.output import FigureOutputTypeEnum, OutputObject
from ..utils.location import convert_location_name_format, get_metrocast_locations, get_parent_region

# Constants
MEDIAN_QUANTILE = 0.5


# == Helper functions ==


def _get_location_sort_key(location: str) -> tuple[str, str]:
    """
    Get sort key for a location (state, location_name).

    For metrocast locations, returns (state_abbreviation, location_name).
    For ISO locations, returns (state_code, location_name).
    For unknown locations, returns ("ZZZ", location) to sort them last.

    Parameters
    ----------
    location : str
        Location name (e.g., "denver", "US-MA").

    Returns
    -------
    tuple[str, str]
        (state_key, location_name) for sorting.
    """
    try:
        # Try to get parent state abbreviation
        state_abbrev = get_parent_region(location, output_format="abbreviation")
        return (state_abbrev, location)
    except (ValueError, KeyError):
        # For ISO locations or unknown, extract state code or sort last
        if "-" in location:
            # ISO format like "US-MA" -> use "MA" as state key
            return (location.split("-")[-1], location)
        return ("ZZZ", location)


def sort_locations_by_state(locations: list[str] | set[str]) -> list[str]:
    """
    Sort locations by state, then alphabetically within each state.

    Parameters
    ----------
    locations : list or set of str
        Location names to sort.

    Returns
    -------
    list of str
        Sorted location names.
    """
    return sorted(locations, key=_get_location_sort_key)


def format_location_name(location: str) -> str:
    """
    Convert location name from epydemix format to clean readable name.

    Parameters
    ----------
    location : str
        Location name in any format (e.g., "United_States_Texas", "United_States").

    Returns
    -------
    str
        Clean location name (e.g., "Texas", "United States").
    """
    try:
        # Convert from epydemix format to name format
        return convert_location_name_format(location, "name")
    except (AssertionError, KeyError, IndexError):
        # If conversion fails, return original name with underscores replaced
        return location.replace("_", " ")


# == Base plotting functions ==


def plot_quantiles(  # noqa: PLR0913
    df_quantiles: pd.DataFrame,
    value_col: str,
    date_col: str = "date",
    quantile_col: str = "quantile",
    color: str = "C0",
    title: str | None = None,
    ax: plt.Axes | None = None,
    marker: str | None = None,
    zorder: float = 2,
) -> tuple[plt.Figure | None, plt.Axes]:
    """
    Plot quantile ribbons from quantile DataFrame.

    This function accepts a DataFrame with quantiles (typically from CalibrationResults.get_projection_quantiles()) and plots quantile ribbons with graduated transparency.

    Parameters
    ----------
    df_quantiles : pd.DataFrame
        DataFrame with columns [date_col, quantile_col, value_col].
        Example:
            date        quantile  hospitalizations
            2024-01-01  0.025     95.2
            2024-01-01  0.500     102.5
            2024-01-01  0.975     110.8
    value_col : str
        Name of column containing values to plot (e.g., "hospitalizations").
    date_col : str, optional
        Name of date column, by default "date".
    quantile_col : str, optional
        Name of quantile column, by default "quantile".
    color : str, optional
        Color for ribbons and center line, by default "C0".
    title : str | None, optional
        Plot title, by default None.
    ax : plt.Axes | None, optional
        Matplotlib axes to plot on. If None, creates new figure, by default None.
    marker : str | None, optional
        Marker style for median line points (e.g., 'o', 's', '^'). If None, no markers, by default None.
    zorder : float, optional
        Z-order for layering (higher values are on top), by default 2.

    Returns
    -------
    tuple[plt.Figure | None, plt.Axes]
        Figure (None if ax was provided) and axes object.

    Raises
    ------
    ValueError
        If required columns are missing or data is empty.

    Examples
    --------
    >>> # From CalibrationResults
    >>> quan_df = calibration.results.get_projection_quantiles(
    ...     quantiles=[0.025, 0.5, 0.975],
    ...     variables=["date", "quantile", "hospitalizations"]
    ... )
    >>> fig, ax = plot_quantiles(
    ...     df_quantiles=quan_df,
    ...     value_col="hospitalizations",
    ...     title="Hospitalization Projections"
    ... )
    >>> fig.savefig("projection.png")
    >>> plt.close(fig)

    """
    # Input validation
    if df_quantiles.empty:
        msg = "Quantile data is empty"
        raise ValueError(msg)

    required_cols = [date_col, quantile_col, value_col]
    missing = set(required_cols) - set(df_quantiles.columns)
    if missing:
        msg = f"Missing columns: {missing}"
        raise ValueError(msg)

    # Pivot DataFrame to get quantiles as columns
    df_quantile = df_quantiles.pivot(index=date_col, columns=quantile_col, values=value_col)  # noqa: PD010

    # Ensure dates are datetime
    df_quantile.index = pd.to_datetime(df_quantile.index)

    # Create figure if no axes provided
    fig = None
    if ax is None:
        fig, ax = plt.subplots()

    # Plot ribbon between lowest and highest quantiles
    quantiles = sorted(df_quantile.columns.tolist())
    ax.fill_between(
        df_quantile.index,
        df_quantile[quantiles[0]].values,
        df_quantile[quantiles[-1]].values,
        color=color,
        alpha=0.3,
        linewidth=0,
        zorder=zorder,
    )

    # Create legend handles for the ribbons
    legend_handles = []
    legend_labels = []

    # Add 95% CrI to legend
    legend_handles.append(mpatches.Patch(color=color, alpha=0.3))
    legend_labels.append("95% CrI")

    # Plot IQR ribbon (25th-75th percentile) on top if available
    if 0.25 in df_quantile.columns and 0.75 in df_quantile.columns:
        ax.fill_between(
            df_quantile.index,
            df_quantile[0.25].values,
            df_quantile[0.75].values,
            color=color,
            alpha=0.5,
            linewidth=0,
            zorder=zorder + 0.1,
        )
        # Add IQR to legend
        legend_handles.append(mpatches.Patch(color=color, alpha=0.5))
        legend_labels.append("IQR")

    # Plot median line if median quantile exists
    if MEDIAN_QUANTILE in df_quantile.columns:
        if marker is not None:
            ax.plot(
                df_quantile.index,
                df_quantile[MEDIAN_QUANTILE].values,
                color=color,
                linewidth=1.5,
                marker=marker,
                markersize=4,
                zorder=zorder + 0.2,
            )
        else:
            ax.plot(
                df_quantile.index,
                df_quantile[MEDIAN_QUANTILE].values,
                color=color,
                linewidth=1.5,
                zorder=zorder + 0.2,
            )

    # Format axes
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
        tick.set_ha("right")

    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(visible=True, linestyle="--", alpha=0.3, linewidth=0.5)

    # Add legend
    if legend_handles:
        ax.legend(legend_handles, legend_labels, loc="upper left", fontsize=8)

    return fig, ax


def plot_quantiles_grid(  # noqa: PLR0913
    location_quantiles: dict[str, pd.DataFrame],
    value_col: str,
    date_col: str = "date",
    quantile_col: str = "quantile",
    color: str = "C0",
    panels_per_row: int = 4,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Create grid of quantile plots for multiple locations.

    This function creates a grid layout with one panel per location, each
    showing quantile ribbons from pre-computed quantile data.

    Parameters
    ----------
    location_quantiles : dict[str, pd.DataFrame]
        Mapping of location names to their quantile DataFrames.
        Each DataFrame should have columns [date_col, quantile_col, value_col].
    value_col : str
        Name of column containing values to plot.
    date_col : str, optional
        Name of date column, by default "date".
    quantile_col : str, optional
        Name of quantile column, by default "quantile".
    color : str, optional
        Color for ribbons and center line, by default "C0".
    panels_per_row : int, optional
        Number of panels per row, by default 4.
    figsize : tuple[float, float] | None, optional
        Figure size. If None, auto-calculated as (4*ncols, 3.6*nrows).

    Returns
    -------
    tuple[plt.Figure, np.ndarray]
        Figure and 2D array of axes objects.

    Examples
    --------
    >>> # Get quantile data for multiple locations
    >>> location_quantiles = {}
    >>> for calibration in calibrations:
    ...     loc = calibration.population
    ...     location_quantiles[loc] = calibration.results.get_projection_quantiles(
    ...         quantiles=[0.025, 0.5, 0.975],
    ...         variables=["date", "quantile", "hospitalizations"]
    ...     )
    >>> fig, axes = plot_quantiles_grid(
    ...     location_quantiles=location_quantiles,
    ...     value_col="hospitalizations",
    ... )
    >>> fig.savefig("all_locations_quantiles.png")
    >>> plt.close(fig)

    """
    locations = list(location_quantiles.keys())
    n = len(locations)
    ncols = panels_per_row
    nrows = int(np.ceil(n / ncols))

    if figsize is None:
        figsize = (4 * ncols, 3.6 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for idx, location in enumerate(locations):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]

        plot_quantiles(
            df_quantiles=location_quantiles[location],
            value_col=value_col,
            date_col=date_col,
            quantile_col=quantile_col,
            color=color,
            title=format_location_name(location),
            ax=ax,
        )

        # Hide legend for non-leftmost columns (c != 0)
        if c != 0:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

    # Remove unused axes
    for idx in range(len(locations), nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r, c].axis("off")

    plt.tight_layout()

    return fig, axes


def plot_calibration_projection(  # noqa: PLR0913
    calibration_quantiles: pd.DataFrame | None = None,
    projection_quantiles: pd.DataFrame | None = None,
    value_col: str = "hospitalizations",
    date_col: str = "date",
    quantile_col: str = "quantile",
    calibration_color: str = "C0",
    projection_color: str = "C1",
    df_surveillance: pd.DataFrame | None = None,
    surveillance_date_col: str = "date",
    surveillance_value_col: str = "value",
    surveillance_size: float = 16.0,
    fitting_window_start: str | pd.Timestamp | datetime | None = None,
    fitting_window_end: str | pd.Timestamp | datetime | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
    weekly_x_labels: bool = False,
) -> tuple[plt.Figure | None, plt.Axes]:
    """
    Plot calibration and projection quantiles on top of each other for a single location.

    This function combines calibration and projection quantile ribbons on the same axes,
    optionally adding surveillance data as scatter points and a reference date line.

    Parameters
    ----------
    calibration_quantiles : pd.DataFrame | None, optional
        Pre-computed quantiles for calibration period. If None, not plotted.
    projection_quantiles : pd.DataFrame | None, optional
        Pre-computed quantiles for projection period. If None, not plotted.
    value_col : str, optional
        Name of column containing values to plot, by default "hospitalizations".
    date_col : str, optional
        Name of date column, by default "date".
    quantile_col : str, optional
        Name of quantile column, by default "quantile".
    calibration_color : str, optional
        Color for calibration ribbons, by default "C0".
    projection_color : str, optional
        Color for projection ribbons, by default "C1".
    df_surveillance : pd.DataFrame | None, optional
        Surveillance data to overlay as scatter points, by default None.
    surveillance_date_col : str, optional
        Date column in surveillance data, by default "date".
    surveillance_value_col : str, optional
        Value column in surveillance data, by default "value".
    surveillance_size : float, optional
        Marker size for surveillance points, by default 30.0.
    fitting_window_start : str | pd.Timestamp | datetime | None, optional
        If provided, draw vertical line at start of calibration/fitting window,
        by default None.
    fitting_window_end : str | pd.Timestamp | datetime | None, optional
        If provided, draw vertical line at end of calibration/fitting window,
        by default None.
    title : str | None, optional
        Plot title, by default None.
    ax : plt.Axes | None, optional
        Matplotlib axes to plot on. If None, creates new figure, by default None.

    Returns
    -------
    tuple[plt.Figure | None, plt.Axes]
        Figure (None if ax was provided) and axes object.

    Raises
    ------
    ValueError
        If both calibration_quantiles and projection_quantiles are None.

    Examples
    --------
    >>> # Get quantile data from CalibrationResults
    >>> cal_quant = results.get_calibration_quantiles(
    ...     quantiles=[0.025, 0.5, 0.975],
    ...     variables=["date", "quantile", "hospitalizations"]
    ... )
    >>> proj_quant = results.get_projection_quantiles(
    ...     quantiles=[0.025, 0.5, 0.975],
    ...     variables=["date", "quantile", "hospitalizations"]
    ... )
    >>> # Filter surveillance data for location
    >>> surv = surveillance_df[surveillance_df["location"] == "US-CA"]
    >>> fig, ax = plot_calibration_projection(
    ...     calibration_quantiles=cal_quant,
    ...     projection_quantiles=proj_quant,
    ...     value_col="hospitalizations",
    ...     df_surveillance=surv,
    ...     reference_date="2024-11-30",
    ...     title="US-CA"
    ... )
    >>> fig.savefig("forecast.png")
    >>> plt.close(fig)

    """
    if calibration_quantiles is None and projection_quantiles is None:
        msg = "At least one of calibration_quantiles or projection_quantiles must be provided"
        raise ValueError(msg)

    # Create figure if no axes provided
    fig = None
    if ax is None:
        fig, ax = plt.subplots()

    # Plot fitting window lines FIRST (lowest zorder - most behind)
    if fitting_window_start is not None:
        start_date = pd.to_datetime(fitting_window_start)
        ax.axvline(
            start_date, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, label="Fitting window start", zorder=1
        )

    if fitting_window_end is not None:
        end_date = pd.to_datetime(fitting_window_end)
        ax.axvline(
            end_date, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, label="Fitting window end", zorder=1
        )

    # Plot projection and calibration quantiles with controlled zorder
    # We'll collect legend info manually since we need to combine calibration + projection
    projection_legend_info = []
    calibration_legend_info = []

    if projection_quantiles is not None:
        plot_quantiles(
            df_quantiles=projection_quantiles,
            value_col=value_col,
            date_col=date_col,
            quantile_col=quantile_col,
            color=projection_color,
            ax=ax,
            marker="o",  # Add markers to projection quantiles
            zorder=2,  # Projection behind calibration
        )
        # Remove the legend added by plot_quantiles (we'll recreate it with both cal+proj)
        legend = ax.get_legend()
        if legend is not None:
            # Store legend info before removing
            projection_legend_info = [
                (h, l) for h, l in zip(legend.legend_handles, [t.get_text() for t in legend.get_texts()])
            ]
            legend.remove()

    # Plot calibration
    if calibration_quantiles is not None:
        plot_quantiles(
            df_quantiles=calibration_quantiles,
            value_col=value_col,
            date_col=date_col,
            quantile_col=quantile_col,
            color=calibration_color,
            ax=ax,
            zorder=3,  # Calibration on top of projection
        )
        # Remove the legend added by plot_quantiles
        legend = ax.get_legend()
        if legend is not None:
            # Store legend info before removing
            calibration_legend_info = [
                (h, l) for h, l in zip(legend.legend_handles, [t.get_text() for t in legend.get_texts()])
            ]
            legend.remove()

    # Add surveillance overlay (on top with high zorder)
    if df_surveillance is not None:
        plot_surveillance_scatter(
            df_surveillance=df_surveillance,
            date_col=surveillance_date_col,
            value_col=surveillance_value_col,
            size=surveillance_size,
            ax=ax,
            zorder=10,  # Surveillance on top of everything
        )

    if title is not None:
        ax.set_title(title)

    # Recreate combined legend with both projection and calibration
    all_handles = []
    all_labels = []

    # Add projection legend entries (no prefix)
    for handle, label in projection_legend_info:
        all_handles.append(handle)
        all_labels.append(label)

    # Add calibration legend entries with prefix
    for handle, label in calibration_legend_info:
        all_handles.append(handle)
        all_labels.append(f"Cal. {label}")

    if all_handles:
        ax.legend(all_handles, all_labels, loc="upper left", fontsize=8)

    # Apply weekly x-axis labels if requested
    if weekly_x_labels:
        from matplotlib.dates import DateFormatter, WeekdayLocator

        ax.xaxis.set_major_locator(WeekdayLocator(byweekday=5))  # Saturday = 5 (epiweek ending)
        ax.xaxis.set_major_formatter(DateFormatter("%m/%d"))
        ax.tick_params(axis="x", rotation=45)
        plt.setp(ax.xaxis.get_majorticklabels(), ha="right")

    return fig, ax


def plot_calibration_projection_sidebyside(  # noqa: PLR0913
    calibration_quantiles: pd.DataFrame | None = None,
    projection_quantiles_full: pd.DataFrame | None = None,
    projection_quantiles_filtered: pd.DataFrame | None = None,
    surveillance_full: pd.DataFrame | None = None,
    surveillance_filtered: pd.DataFrame | None = None,
    value_col: str = "hospitalizations",
    date_col: str = "date",
    quantile_col: str = "quantile",
    calibration_color: str = "C0",
    projection_color: str = "C1",
    surveillance_date_col: str = "date",
    surveillance_value_col: str = "value",
    surveillance_size: float = 16.0,
    fitting_window_start: str | pd.Timestamp | datetime | None = None,
    fitting_window_end: str | pd.Timestamp | datetime | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    spacing: float = 0.3,
    ax_full: plt.Axes | None = None,
    ax_filtered: plt.Axes | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """
    Create side-by-side quantile plots: [Full Range | Filtered].

    Left panel shows full surveillance range with projection filtered only by horizon_max.
    Right panel shows limited surveillance points with projection filtered by both
    surveillance start and horizon_max.

    Parameters
    ----------
    calibration_quantiles : pd.DataFrame | None, optional
        Pre-computed quantiles for calibration period (shown in both panels).
    projection_quantiles_full : pd.DataFrame | None, optional
        Projection quantiles for left panel (full range, horizon_max only).
    projection_quantiles_filtered : pd.DataFrame | None, optional
        Projection quantiles for right panel (filtered by surveillance + horizon).
    surveillance_full : pd.DataFrame | None, optional
        Full surveillance data for left panel.
    surveillance_filtered : pd.DataFrame | None, optional
        Filtered surveillance data for right panel.
    value_col : str, optional
        Name of column containing values to plot, by default "hospitalizations".
    date_col : str, optional
        Name of date column, by default "date".
    quantile_col : str, optional
        Name of quantile column, by default "quantile".
    calibration_color : str, optional
        Color for calibration ribbons, by default "C0".
    projection_color : str, optional
        Color for projection ribbons, by default "C1".
    surveillance_date_col : str, optional
        Date column in surveillance data, by default "date".
    surveillance_value_col : str, optional
        Value column in surveillance data, by default "value".
    surveillance_size : float, optional
        Marker size for surveillance points, by default 30.0.
    fitting_window_start : str | pd.Timestamp | datetime | None, optional
        If provided, draw vertical line at start of calibration/fitting window,
        by default None.
    fitting_window_end : str | pd.Timestamp | datetime | None, optional
        If provided, draw vertical line at end of calibration/fitting window,
        by default None.
    title : str | None, optional
        Base plot title (will be suffixed with panel type), by default None.
    figsize : tuple[float, float] | None, optional
        Figure size. If None, defaults to (16, 6).
    spacing : float, optional
        Horizontal spacing between subplots, by default 0.3.
    ax_full : plt.Axes | None, optional
        Optional axes for the full panel. If provided, a new figure is not created.
    ax_filtered : plt.Axes | None, optional
        Optional axes for the filtered panel. If provided, a new figure is not created.

    Returns
    -------
    tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]
        Figure and tuple of (ax_full, ax_filtered) axes objects.

    Examples
    --------
    >>> # Prepare data with two versions
    >>> cal_quant = results.get_calibration_quantiles([0.025, 0.5, 0.975])
    >>> proj_quant_full = results.get_projection_quantiles([0.025, 0.5, 0.975])
    >>> # Filter projection for right panel
    >>> proj_quant_filtered = proj_quant_full[proj_quant_full["date"] >= surveillance_start]
    >>> fig, (ax_full, ax_filtered) = plot_calibration_projection_sidebyside(
    ...     calibration_quantiles=cal_quant,
    ...     projection_quantiles_full=proj_quant_full,
    ...     projection_quantiles_filtered=proj_quant_filtered,
    ...     surveillance_full=surv_full,
    ...     surveillance_filtered=surv_filtered,
    ...     value_col="hospitalizations",
    ...     title="US-CA"
    ... )
    >>> fig.savefig("sidebyside_forecast.png")
    >>> plt.close(fig)

    """
    fig_provided = ax_full is not None and ax_filtered is not None

    if not fig_provided:
        if figsize is None:
            figsize = (16, 6)
        fig, (ax_full, ax_filtered) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"wspace": spacing})
    else:
        fig = ax_full.figure

    # Left panel: Full range
    plot_calibration_projection(
        calibration_quantiles=calibration_quantiles,
        projection_quantiles=projection_quantiles_full,
        value_col=value_col,
        date_col=date_col,
        quantile_col=quantile_col,
        calibration_color=calibration_color,
        projection_color=projection_color,
        df_surveillance=surveillance_full,
        surveillance_date_col=surveillance_date_col,
        surveillance_value_col=surveillance_value_col,
        surveillance_size=surveillance_size,
        fitting_window_start=fitting_window_start,
        fitting_window_end=fitting_window_end,
        title=title,
        ax=ax_full,
    )

    # Right panel: Filtered
    plot_calibration_projection(
        calibration_quantiles=calibration_quantiles,
        projection_quantiles=projection_quantiles_filtered,
        value_col=value_col,
        date_col=date_col,
        quantile_col=quantile_col,
        calibration_color=calibration_color,
        projection_color=projection_color,
        df_surveillance=surveillance_filtered,
        surveillance_date_col=surveillance_date_col,
        surveillance_value_col=surveillance_value_col,
        surveillance_size=surveillance_size,
        fitting_window_start=fitting_window_start,
        fitting_window_end=fitting_window_end,
        title=title,
        ax=ax_filtered,
    )

    return fig, (ax_full, ax_filtered)


def plot_calibration_projection_grid(  # noqa: PLR0913
    location_calibration_quantiles: dict[str, pd.DataFrame] | None = None,
    location_projection_quantiles: dict[str, pd.DataFrame] | None = None,
    value_col: str = "hospitalizations",
    date_col: str = "date",
    quantile_col: str = "quantile",
    calibration_color: str = "C0",
    projection_color: str = "C1",
    location_surveillance: dict[str, pd.DataFrame] | None = None,
    surveillance_date_col: str = "date",
    surveillance_value_col: str = "value",
    surveillance_size: float = 16.0,
    location_fitting_window_starts: dict[str, datetime] | None = None,
    location_fitting_window_ends: dict[str, datetime] | None = None,
    panels_per_row: int = 4,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Create multipanel grid of calibration and projection quantile plots.

    This function creates a grid layout with one panel per location, each showing
    calibration and projection quantile ribbons with optional overlays.

    Parameters
    ----------
    location_calibration_quantiles : dict[str, pd.DataFrame] | None, optional
        Mapping of location names to their calibration quantile DataFrames.
    location_projection_quantiles : dict[str, pd.DataFrame] | None, optional
        Mapping of location names to their projection quantile DataFrames.
    value_col : str, optional
        Name of column containing values to plot, by default "hospitalizations".
    date_col : str, optional
        Name of date column, by default "date".
    quantile_col : str, optional
        Name of quantile column, by default "quantile".
    calibration_color : str, optional
        Color for calibration ribbons, by default "C0".
    projection_color : str, optional
        Color for projection ribbons, by default "C1".
    location_surveillance : dict[str, pd.DataFrame] | None, optional
        Mapping of location names to surveillance DataFrames, by default None.
    surveillance_date_col : str, optional
        Date column in surveillance data, by default "date".
    surveillance_value_col : str, optional
        Value column in surveillance data, by default "value".
    surveillance_size : float, optional
        Marker size for surveillance points, by default 30.0.
    location_fitting_window_starts : dict[str, datetime] | None, optional
        Dictionary mapping location names to fitting window start dates for vertical lines,
        by default None.
    location_fitting_window_ends : dict[str, datetime] | None, optional
        Dictionary mapping location names to fitting window end dates for vertical lines,
        by default None.
    panels_per_row : int, optional
        Number of panels per row, by default 4.
    figsize : tuple[float, float] | None, optional
        Figure size. If None, auto-calculated as (4*ncols, 3.6*nrows).

    Returns
    -------
    tuple[plt.Figure, np.ndarray]
        Figure and 2D array of axes objects.

    Raises
    ------
    ValueError
        If no locations are provided.

    Examples
    --------
    >>> # Prepare quantile data for multiple locations
    >>> cal_quants = {}
    >>> proj_quants = {}
    >>> surv = {}
    >>> for calibration in calibrations:
    ...     loc = calibration.population
    ...     cal_quants[loc] = calibration.results.get_calibration_quantiles(
    ...         quantiles=[0.025, 0.5, 0.975],
    ...         variables=["date", "quantile", "hospitalizations"]
    ...     )
    ...     proj_quants[loc] = calibration.results.get_projection_quantiles(
    ...         quantiles=[0.025, 0.5, 0.975],
    ...         variables=["date", "quantile", "hospitalizations"]
    ...     )
    ...     surv[loc] = surveillance_df[surveillance_df["location"] == loc]
    >>> fig, axes = plot_calibration_projection_grid(
    ...     location_calibration_quantiles=cal_quants,
    ...     location_projection_quantiles=proj_quants,
    ...     location_surveillance=surv,
    ...     value_col="hospitalizations",
    ...     reference_date="2024-11-30",
    ... )
    >>> fig.savefig("all_forecasts.png")
    >>> plt.close(fig)

    """
    # Get all locations
    locations = set()
    if location_calibration_quantiles is not None:
        locations.update(location_calibration_quantiles.keys())
    if location_projection_quantiles is not None:
        locations.update(location_projection_quantiles.keys())
    locations = sort_locations_by_state(locations)

    if not locations:
        msg = "No locations provided"
        raise ValueError(msg)

    n = len(locations)
    ncols = panels_per_row
    nrows = int(np.ceil(n / ncols))

    if figsize is None:
        figsize = (4 * ncols, 3.6 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for idx, location in enumerate(locations):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]

        cal_quant = location_calibration_quantiles.get(location) if location_calibration_quantiles else None
        proj_quant = location_projection_quantiles.get(location) if location_projection_quantiles else None
        surv_df = location_surveillance.get(location) if location_surveillance else None
        fitting_window_start = location_fitting_window_starts.get(location) if location_fitting_window_starts else None
        fitting_window_end = location_fitting_window_ends.get(location) if location_fitting_window_ends else None

        plot_calibration_projection(
            calibration_quantiles=cal_quant,
            projection_quantiles=proj_quant,
            value_col=value_col,
            date_col=date_col,
            quantile_col=quantile_col,
            calibration_color=calibration_color,
            projection_color=projection_color,
            df_surveillance=surv_df,
            surveillance_date_col=surveillance_date_col,
            surveillance_value_col=surveillance_value_col,
            surveillance_size=surveillance_size,
            fitting_window_start=fitting_window_start,
            fitting_window_end=fitting_window_end,
            title=format_location_name(location),
            ax=ax,
        )

        # Hide legend for non-leftmost columns (c != 0)
        if c != 0:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

    # Remove unused axes
    for idx in range(len(locations), nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r, c].axis("off")

    plt.tight_layout()

    return fig, axes


def plot_surveillance_scatter(  # noqa: PLR0913
    df_surveillance: pd.DataFrame,
    date_col: str = "date",
    value_col: str = "value",
    color: str = "black",
    size: float = 20.0,
    title: str | None = None,
    ax: plt.Axes | None = None,
    zorder: float = 10,
) -> tuple[plt.Figure | None, plt.Axes]:
    """
    Plot surveillance data as scatter points.

    Parameters
    ----------
    df_surveillance : pd.DataFrame
        Surveillance data with columns [date_col, value_col].
        Should be pre-filtered for a single location.
    date_col : str
        Column containing dates.
    value_col : str
        Column containing observed values.
    color : str
        Marker color.
    size : float
        Marker size.
    title : str, optional
        Plot title.
    ax : plt.Axes, optional
        Axes to plot into. If None, creates new figure and axes.
    zorder : float, optional
        Z-order for layering (higher values are on top), by default 10.

    Returns
    -------
    fig : plt.Figure or None
        Figure (None if ax was provided).
    ax : plt.Axes
        Axes with scatter points.

    Raises
    ------
    ValueError
        If DataFrame is empty or missing required columns.

    Examples
    --------
    >>> # Pre-filter surveillance data for location
    >>> location_surv = surveillance_df[surveillance_df["location"] == "US-CA"]
    >>> fig, ax = plot_surveillance_scatter(
    ...     df_surveillance=location_surv,
    ...     date_col="target_end_date",
    ...     value_col="value",
    ...     size=30,
    ... )

    """
    if df_surveillance.empty:
        msg = "Surveillance data is empty"
        raise ValueError(msg)

    # Check required columns
    missing = {date_col, value_col} - set(df_surveillance.columns)
    if missing:
        msg = f"Missing columns: {missing}"
        raise ValueError(msg)

    df_surveillance = df_surveillance.copy()
    df_surveillance[date_col] = pd.to_datetime(df_surveillance[date_col])
    df_surveillance = df_surveillance.sort_values(date_col)

    # Create figure if no axes provided
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    ax.scatter(df_surveillance[date_col], df_surveillance[value_col], color=color, s=size, zorder=zorder)
    ax.set_xlabel("")

    if title is not None:
        ax.set_title(title)

    # Rotate x-axis labels
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
        tick.set_ha("right")

    return fig, ax


def plot_posterior_histogram(  # noqa: PLR0913
    df_posterior: pd.DataFrame,
    parameter: str,
    bins: int = 25,
    color: str | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
    start_date_reference: str | None = None,
    **kwargs: Any,
) -> tuple[plt.Figure | None, plt.Axes]:
    """
    Plot histogram of posterior samples for a single parameter.

    Parameters
    ----------
    df_posterior : pd.DataFrame
        Posterior samples with column for the parameter.
        Should be pre-filtered for a single location if needed.
    parameter : str
        Column name of parameter to plot.
    bins : int
        Number of histogram bins.
    color : str, optional
        Histogram color.
    title : str, optional
        Plot title.
    ax : plt.Axes, optional
        Axes to plot into.
    start_date_reference : str or None, optional
        If parameter == "start_date", samples are treated as integer day offsets
        from this reference date and converted to actual dates.
    **kwargs
        Additional arguments passed to matplotlib hist.

    Returns
    -------
    fig : plt.Figure or None
        Figure (None if ax was provided).
    ax : plt.Axes
        Axes with histogram.

    Raises
    ------
    ValueError
        If DataFrame is empty or missing the parameter column.

    Examples
    --------
    >>> posterior_df = calibration.results.get_posterior_distribution()
    >>> fig, ax = plot_posterior_histogram(
    ...     df_posterior=posterior_df,
    ...     parameter="R0",
    ...     bins=30,
    ...     title="R0",
    ... )

    """
    if df_posterior.empty:
        msg = "Posterior data is empty"
        raise ValueError(msg)

    if parameter not in df_posterior.columns:
        msg = f"Parameter '{parameter}' not in DataFrame columns"
        raise ValueError(msg)

    data = df_posterior[parameter].copy()

    # Handle start_date offset -> actual datetime
    labels_as_dates = False
    if parameter == "start_date" and start_date_reference is not None:
        ref_date = pd.to_datetime(start_date_reference)
        date_series = ref_date + pd.to_timedelta(data, unit="D")
        data_numeric = date_series.map(datetime.toordinal)
        labels_as_dates = True
        plotting_data = data_numeric
    else:
        plotting_data = data

    # Create figure if no axes provided
    fig = None
    if ax is None:
        fig, ax = plt.subplots()

    # Plot histogram
    hist_kwargs = {"bins": bins, "density": True, "alpha": 0.7, "edgecolor": "black"}
    if color is not None:
        hist_kwargs["color"] = color
    hist_kwargs.update(kwargs)

    ax.hist(plotting_data.dropna(), **hist_kwargs)

    # Fix date tick labels if needed
    if labels_as_dates:
        xticks = ax.get_xticks()
        ax.set_xticks(xticks)
        labels = [datetime.fromordinal(int(v)).strftime("%Y-%m-%d") for v in xticks]
        ax.set_xticklabels(labels, rotation=45, ha="right")

    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(parameter)

    ax.set_ylabel("Density")

    return fig, ax


def plot_posterior_histogram_grid(
    location_posteriors: dict[str, pd.DataFrame],
    parameters: list[str],
    bins: int = 25,
    figsize: tuple[float, float] | None = None,
    start_date_reference: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Create grid of posterior histograms (rows=locations, cols=parameters).

    Parameters
    ----------
    location_posteriors : dict[str, pd.DataFrame]
        Mapping of location names to their posterior DataFrames.
    parameters : list of str
        Parameter names to plot (one per column).
    bins : int
        Number of histogram bins.
    figsize : tuple, optional
        Figure size.
    start_date_reference : str or None, optional
        Reference date for converting start_date offsets.

    Returns
    -------
    fig : plt.Figure
        Figure with grid.
    axes : np.ndarray of plt.Axes
        Array of axes (shape: n_locations x n_parameters).

    Examples
    --------
    >>> location_posteriors = {
    ...     "US-CA": cal_ca.results.get_posterior_distribution(),
    ...     "US-TX": cal_tx.results.get_posterior_distribution(),
    ... }
    >>> fig, axes = plot_posterior_histogram_grid(
    ...     location_posteriors=location_posteriors,
    ...     parameters=["R0", "beta", "start_date"],
    ...     bins=30,
    ... )
    >>> fig.savefig("posteriors_grid.png")

    """
    locations = list(location_posteriors.keys())
    n_rows = len(locations)
    n_cols = len(parameters)

    if figsize is None:
        figsize = (3.5 * n_cols, 3 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    for i, location in enumerate(locations):
        df_post = location_posteriors[location]
        for j, param in enumerate(parameters):
            ax = axes[i, j]

            plot_posterior_histogram(
                df_posterior=df_post,
                parameter=param,
                bins=bins,
                ax=ax,
                start_date_reference=start_date_reference,
            )

            ax.set_xlabel("")

            # Only first column keeps the location label
            if j == 0:
                ax.set_ylabel(location)
            else:
                ax.set_ylabel("")

            # Only first row gets titles
            if i == 0:
                ax.set_title(param)
            else:
                ax.set_title("")

    plt.tight_layout()

    return fig, axes


def plot_categorical_stacked_bars(
    df_categorical: pd.DataFrame,
    categories: list[str],
    colors: list[str],
    location_col: str = "location",
    category_col: str = "output_type_id",
    value_col: str = "value",
    title: str | None = None,
    ax: plt.Axes | None = None,
    category_labels: dict[str, str] | None = None,
) -> tuple[plt.Figure | None, plt.Axes]:
    """
    Plot stacked bar chart of categorical forecast probabilities for a single horizon.

    Parameters
    ----------
    df_categorical : pd.DataFrame
        DataFrame containing categorical forecast data.
    categories : list of str
        Category names in display order (bottom to top in stacked bars).
    colors : list of str
        Colors for categories (must match length of categories list).
    location_col : str, default "location"
        Column name for location identifiers.
    category_col : str, default "output_type_id"
        Column name for category identifiers.
    value_col : str, default "value"
        Column name for probability values.
    title : str or None, optional
        Plot title.
    ax : plt.Axes or None, optional
        Axes to plot on. If None, creates new figure and axes.
    category_labels : dict[str, str] or None, optional
        Mapping from category values to display labels for legend.
        If None, uses category values as-is.

    Returns
    -------
    fig : plt.Figure or None
        Figure (None if ax was provided).
    ax : plt.Axes
        Axes with the plot.

    Raises
    ------
    ValueError
        If data is empty, required columns are missing, or colors/categories lists
        have different lengths.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "location": ["US-CA", "US-CA", "US-TX", "US-TX"],
    ...     "output_type_id": ["increase", "stable", "increase", "stable"],
    ...     "value": [0.3, 0.7, 0.4, 0.6],
    ... })
    >>> fig, ax = plot_categorical_stacked_bars(
    ...     df, categories=["stable", "increase"], colors=["blue", "red"]
    ... )

    """
    # Validate inputs
    if df_categorical.empty:
        msg = "Cannot create categorical plot: data is empty"
        raise ValueError(msg)

    required_cols = [location_col, category_col, value_col]
    missing_cols = [col for col in required_cols if col not in df_categorical.columns]
    if missing_cols:
        msg = f"Required columns missing from data: {missing_cols}"
        raise ValueError(msg)

    if len(categories) != len(colors):
        msg = f"categories list length ({len(categories)}) must match colors list length ({len(colors)})"
        raise ValueError(msg)

    # Create figure if no axes provided
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))

    # Get sorted locations list with United States first, then by state
    locations = sort_locations_by_state(df_categorical[location_col].unique())
    if "United States" in locations:
        locations.remove("United States")
        locations = ["United States"] + locations
    n_locations = len(locations)

    # Build stacked bars by iterating through categories
    bottoms = np.zeros(n_locations)
    bar_handles = []

    for category, color in zip(categories, colors):
        # Extract probability values for each location
        values = []
        for location in locations:
            mask = (df_categorical[location_col] == location) & (df_categorical[category_col] == category)
            matching_rows = df_categorical[mask]

            if not matching_rows.empty:
                values.append(matching_rows[value_col].values[0])
            else:
                # Handle missing data (default to 0.0)
                values.append(0.0)

        # Get display label for legend
        display_label = category_labels.get(category, category) if category_labels else category

        # Create bar segment
        bars = ax.bar(range(n_locations), values, bottom=bottoms, color=color, label=display_label, width=0.8)
        bar_handles.append(bars)

        # Update bottom for next category
        bottoms += np.array(values)

    # Format axes
    ax.set_xticks(range(n_locations))
    ax.set_xticklabels([format_location_name(loc) for loc in locations], rotation=90)
    ax.set_ylabel("")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    if title:
        ax.set_title(title)

    return fig, ax


def plot_categorical_stacked_bars_multihorizon(
    df_categorical: pd.DataFrame,
    categories: list[str],
    colors: list[str],
    horizons: list[int],
    location_col: str = "location",
    horizon_col: str = "horizon",
    category_col: str = "output_type_id",
    value_col: str = "value",
    figsize: tuple[float, float] | None = None,
    reference_date: datetime | None = None,
    category_labels: dict[str, str] | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Create vertical stack of categorical stacked bars across horizons.

    Creates a nx1 grid (n = len(horizons)) with shared x-axis.
    X-axis labels are shown only on the bottom panel.

    Parameters
    ----------
    df_categorical : pd.DataFrame
        DataFrame containing categorical forecast data with horizon column.
    categories : list of str
        Category names in display order (bottom to top in stacked bars).
    colors : list of str
        Colors for categories (must match length of categories list).
    horizons : list of int
        Forecast horizons to display (e.g., [0, 1, 2, 3] for FluSight).
    location_col : str, default "location"
        Column name for location identifiers.
    horizon_col : str, default "horizon"
        Column name for horizon identifiers.
    category_col : str, default "output_type_id"
        Column name for category identifiers.
    value_col : str, default "value"
        Column name for probability values.
    figsize : tuple of float or None, optional
        Figure size (width, height). If None, defaults to (10, 3 * n_horizons).
    reference_date : datetime or None, optional
        Reference date for calculating target_end_date in panel titles.
    category_labels : dict[str, str] or None, optional
        Mapping from category values to display labels for legend.
        If None, uses category values as-is.

    Returns
    -------
    fig : plt.Figure
        Figure with stacked panels.
    axes : np.ndarray
        Array of axes (shape: n_horizons x 1).

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "location": ["US-CA", "US-TX"] * 8,
    ...     "horizon": [0]*4 + [1]*4 + [2]*4 + [3]*4,
    ...     "output_type_id": ["stable", "increase"] * 8,
    ...     "value": [0.6, 0.4] * 8,
    ... })
    >>> fig, axes = plot_categorical_stacked_bars_multihorizon(
    ...     df, categories=["stable", "increase"], colors=["blue", "red"],
    ...     horizons=[0, 1, 2, 3]
    ... )

    """
    # Set grid layout
    nrows = len(horizons)
    ncols = 1

    # Default figsize
    if figsize is None:
        figsize = (10, 3 * nrows)

    # Create subplots with shared x-axis
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, sharex=True)

    # Set overall figure title
    fig.suptitle("Rate-change trend forecasts", fontsize=14, y=0.995)

    # For each horizon
    for idx, horizon in enumerate(horizons):
        ax = axes[idx, 0]

        # Filter data for this horizon
        df_horizon = df_categorical[df_categorical[horizon_col] == horizon]

        # Generate panel title
        week_ahead = horizon + 2  # FluSight convention

        # Extract target_end_date from data if available, otherwise calculate from reference_date
        target_end_date_str = ""
        if "target_end_date" in df_horizon.columns:
            target_dates = df_horizon["target_end_date"].unique()
            if len(target_dates) > 0:
                # Convert to pandas Timestamp and format as date only
                target_date = pd.Timestamp(target_dates[0])
                target_end_date_str = f" ({target_date.strftime('%Y-%m-%d')})"
        elif reference_date is not None:
            # Calculate target_end_date from reference_date (assuming weekly forecasts)
            from datetime import timedelta

            target_date = reference_date + timedelta(weeks=week_ahead)
            target_end_date_str = f" ({target_date.strftime('%Y-%m-%d')})"

        panel_title = f"{week_ahead} week ahead{target_end_date_str}"

        # Call single-horizon plot function
        _, ax = plot_categorical_stacked_bars(
            df_categorical=df_horizon,
            categories=categories,
            colors=colors,
            location_col=location_col,
            category_col=category_col,
            value_col=value_col,
            title=panel_title,
            ax=ax,
            category_labels=category_labels,
        )

        # Remove individual panel legend (will create shared legend later)
        if ax.get_legend() is not None:
            ax.get_legend().remove()

        # Hide x-axis labels for all panels except the last one (bottom panel)
        if idx < nrows - 1:
            ax.set_xticklabels([])

    # Create shared legend outside all panels
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.08, 0.5))

    # Apply tight_layout to make room for legend
    plt.tight_layout(rect=[0, 0, 0.91, 0.99])

    return fig, axes


# == Utilities ==


def figure_to_output_object(
    fig: plt.Figure,
    name: str,
    output_type: str | FigureOutputTypeEnum = FigureOutputTypeEnum.MPLFigure,
    dpi: int = 150,
) -> OutputObject:
    """
    Convert matplotlib figure to OutputObject for dispatcher output.

    Parameters
    ----------
    fig : plt.Figure
        Matplotlib figure to convert.
    name : str
        Base name for the output (without extension).
    output_type : str | FigureOutputTypeEnum
        Output type: "MPLFigure", "PNG", "PDF", or "SVG". Default "MPLFigure".
    dpi : int
        DPI for raster formats.

    Returns
    -------
    OutputObject
        Output object containing figure as bytes.

    Examples
    --------
    >>> fig, ax = plot_trajectory_quantiles(...)
    >>> output_obj = figure_to_output_object(fig, "quantiles_US-CA", "png", 150)
    >>> # Dispatcher can then save this to disk

    """
    # Convert types
    if isinstance(output_type, str):
        output_type = FigureOutputTypeEnum(output_type)

    # Return bare figure if requested
    if output_type == FigureOutputTypeEnum.MPLFigure:
        return OutputObject(
            output_type=output_type,
            name=f"{name}.{output_type.name.lower()}",
            data=fig,
        )

    # Otherwise, return image binary in requested format
    buffer = io.BytesIO()
    fig.savefig(buffer, format=output_type.name.lower(), bbox_inches="tight", dpi=dpi)
    buffer.seek(0)

    return OutputObject(
        output_type=output_type,
        name=f"{name}.{output_type.name.lower()}",
        data=buffer.getvalue(),
    )
