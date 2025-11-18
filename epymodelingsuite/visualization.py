"""Visualization functions for calibration/projection outputs."""

import io
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .schema.output import FigureOutputTypeEnum, OutputObject

# Constants
MEDIAN_QUANTILE = 0.5

# == Base plotting functions ==


def plot_quantiles(  # noqa: PLR0913
    df_quantiles: pd.DataFrame,
    value_col: str,
    date_col: str = "date",
    quantile_col: str = "quantile",
    color: str = "C0",
    title: str | None = None,
    ax: plt.Axes | None = None,
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

    Returns
    -------
    tuple[plt.Figure | None, plt.Axes]
        Figure (None if ax was provided) and axes objects.

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
    )

    # Plot median line if median quantile exists
    if MEDIAN_QUANTILE in df_quantile.columns:
        ax.plot(df_quantile.index, df_quantile[MEDIAN_QUANTILE].values, color=color, linewidth=1.5)

    # Format axes
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
        tick.set_ha("right")

    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(visible=True, linestyle="--", alpha=0.3, linewidth=0.5)

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
            title=location,
            ax=ax,
        )

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
    surveillance_size: float = 30.0,
    reference_date: str | pd.Timestamp | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
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
    reference_date : str | pd.Timestamp | None, optional
        Date for vertical reference line (e.g., calibration/projection boundary),
        by default None.
    title : str | None, optional
        Plot title, by default None.
    ax : plt.Axes | None, optional
        Matplotlib axes to plot on. If None, creates new figure, by default None.

    Returns
    -------
    tuple[plt.Figure | None, plt.Axes]
        Figure (None if ax was provided) and axes objects.

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

    # Plot projection first (so calibration overlays)
    if projection_quantiles is not None:
        plot_quantiles(
            df_quantiles=projection_quantiles,
            value_col=value_col,
            date_col=date_col,
            quantile_col=quantile_col,
            color=projection_color,
            ax=ax,
        )

    # Plot calibration
    if calibration_quantiles is not None:
        plot_quantiles(
            df_quantiles=calibration_quantiles,
            value_col=value_col,
            date_col=date_col,
            quantile_col=quantile_col,
            color=calibration_color,
            ax=ax,
        )

    # Add surveillance overlay
    if df_surveillance is not None:
        plot_surveillance_scatter(
            df_surveillance=df_surveillance,
            date_col=surveillance_date_col,
            value_col=surveillance_value_col,
            size=surveillance_size,
            ax=ax,
        )

    # Add reference date line
    if reference_date is not None:
        ref_date = pd.to_datetime(reference_date)
        ax.axvline(ref_date, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)

    if title is not None:
        ax.set_title(title)

    return fig, ax


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
    surveillance_size: float = 30.0,
    reference_date: str | pd.Timestamp | None = None,
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
    reference_date : str | pd.Timestamp | None, optional
        Date for vertical reference line, by default None.
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
    locations = sorted(locations)

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
            reference_date=reference_date,
            title=location,
            ax=ax,
        )

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

    ax.scatter(df_surveillance[date_col], df_surveillance[value_col], color=color, s=size)
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


# == Utilities ==


def figure_to_output_object(
    fig: plt.Figure,
    name: str,
    output_format: str = "png",
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
    output_format : str
        Output format: "png", "pdf", or "svg".
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
    buffer = io.BytesIO()
    fig.savefig(buffer, format=output_format, bbox_inches="tight", dpi=dpi)
    buffer.seek(0)

    format_to_enum = {
        "png": FigureOutputTypeEnum.PNG,
        "pdf": FigureOutputTypeEnum.PDF,
        "svg": FigureOutputTypeEnum.SVG,
    }
    output_type = format_to_enum.get(output_format.lower(), FigureOutputTypeEnum.PNG)

    return OutputObject(
        output_type=output_type,
        name=f"{name}.{output_format}",
        data=buffer.getvalue(),
    )
