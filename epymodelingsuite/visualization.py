from epydemix.visualization import plot_posterior_distribution
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Iterable
from .utils import convert_location_name_format


def _compute_date_quantiles(
    series: pd.Series,
    quantiles: Iterable[float],
) -> pd.Series:
    """
    Compute quantiles for a 1D numeric Series and return a Series indexed by the
    quantile values (as floats). Uses numpy.quantile for consistent results.
    """
    qs = np.asarray(quantiles, dtype=float)
    # numpy.quantile will raise on empty arrays; handle gracefully
    if series.size == 0:
        vals = np.full(len(qs), np.nan)
    else:
        vals = np.quantile(series.values, qs)
    return pd.Series(vals, index=qs)


def plot_calibration_projection_quantiles(
    state: str,
    calibration_settings: dict,
    projection_settings: dict,
    surveillance_settings: dict | None = None,
    forecast_reference_date: str | pd.Timestamp | None = None,
    ax: plt.Axes | None = None,
):
    """
    Plot time-series quantiles for one state with both calibration and projection data.

    Parameters
    ----------
    state : str
        State to plot (ISO format)
    calibration_settings : dict
        Dictionary containing quantile settings for calibration data:
            - df : pd.DataFrame with [population, date_col, sim_id] and value columns
            - date_col : str, column containing the date
            - value_cols : str or list[str], column(s) to plot quantiles of
            - quantiles : list[float], ordered list of quantiles to plot
            - color : str, base color for quantile shading and line
    projection_settings : dict
        Dictionary containing quantile settings for projection data (same structure as calibration_settings)
    surveillance_settings : dict, optional
        Dictionary containing:
            - df : pd.DataFrame with surveillance data
            - date_col : str, column containing dates
            - location_col : str, column containing geographic identifiers
            - target_col : str, column containing values to plot
            - size : float, marker size for scatter plot
    forecast_reference_date : str, pd.Timestamp, or None, optional
        If provided, adds a vertical dashed line at this date.
    ax : matplotlib.axes.Axes, optional
        Axis to plot into. If None, a new axis is created.

    Returns
    -------
    matplotlib.axes.Axes
    """
    
    if ax is None:
        _, ax = plt.subplots()

    # Plot projection quantiles (no surveillance or reference date yet) 
    plot_state_quantiles(
        state=state,
        quantile_settings=projection_settings,
        surveillance_settings=None,
        forecast_reference_date=None,
        ax=ax,
    )

    # Plot calibration quantiles (now add surveillance and reference date)
    plot_state_quantiles(
        state=state,
        quantile_settings=calibration_settings,
        surveillance_settings=surveillance_settings,
        forecast_reference_date=forecast_reference_date,
        ax=ax,
    )

    return ax

def plot_calibration_projection_quantiles_grid(
    states: list[str],
    calibration_settings: dict,
    projection_settings: dict,
    surveillance_settings: dict | None = None,
    forecast_reference_date: str | pd.Timestamp | None = None,
    panels_per_row: int = 4,
    figsize: tuple[int, int] | None = None,
    outfile: str | None = None,
):
    """
    Create a multipanel grid of time-series quantile plots with both calibration and projection data.

    Parameters
    ----------
    states : list[str]
        List of states (ISO format).
    calibration_settings : dict
        Dictionary containing quantile settings for calibration data:
            - df : pd.DataFrame with [population, date_col, sim_id] and value columns
            - date_col : str, column containing the date
            - value_cols : str or list[str], column(s) to plot quantiles of
            - quantiles : list[float], ordered list of quantiles to plot
            - color : str, base color for quantile shading and line
    projection_settings : dict
        Dictionary containing quantile settings for projection data (same structure as calibration_settings)
    surveillance_settings : dict, optional
        Dictionary containing:
            - df : pd.DataFrame with surveillance data
            - date_col : str, column containing dates
            - location_col : str, column containing geographic identifiers
            - target_col : str, column containing values to plot
            - size : float, marker size for scatter plot
    forecast_reference_date : str, pd.Timestamp, or None, optional
        If provided, adds a vertical dashed line at this date on all plots.
    panels_per_row : int, optional
        Number of panels per row in the grid.
    figsize : (int, int), optional
        Size of the full figure.
    outfile : str or None, optional
        If provided, save the figure to this path.

    Returns
    -------
    (Figure, ndarray of Axes)
    """

    n = len(states)
    ncols = panels_per_row
    nrows = int(np.ceil(n / ncols))

    # auto figsize if None
    if figsize is None:
        figsize = (4 * ncols, 3.6 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for idx, state in enumerate(states):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]

        plot_calibration_projection_quantiles(
            state=state,
            calibration_settings=calibration_settings,
            projection_settings=projection_settings,
            surveillance_settings=surveillance_settings,
            forecast_reference_date=forecast_reference_date,
            ax=ax,
        )

    # Remove unused axes
    for idx in range(len(states), nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r, c].axis("off")

    plt.tight_layout()

    if outfile is not None:
        fig.savefig(outfile, bbox_inches="tight")

    return fig, axes


def plot_state_quantiles(
    state: str,
    quantile_settings: dict,
    surveillance_settings: dict | None = None,
    forecast_reference_date: str | pd.Timestamp | None = None,
    ax: plt.Axes | None = None,
):
    """
    Plot time-series quantiles for one state.

    Parameters
    ----------
    state : str
        State to plot (ISO format)
    quantile_settings : dict
        Dictionary containing:
            - df : pd.DataFrame with [population, date_col, sim_id] and value columns
            - date_col : str, column containing the date
            - value_cols : str or list[str], column(s) to plot quantiles of
            - quantiles : list[float], ordered list of quantiles to plot
            - color : str, base color for quantile shading and line
    surveillance_settings : dict, optional
        Dictionary containing:
            - df : pd.DataFrame with surveillance data
            - date_col : str, column containing dates
            - location_col : str, column containing geographic identifiers
            - target_col : str, column containing values to plot
            - size : float, marker size for scatter plot
    forecast_reference_date : str, pd.Timestamp, or None, optional
        If provided, adds a vertical dashed line at this date.
    ax : matplotlib.axes.Axes, optional
        Axis to plot into. If None, a new axis is created.

    Returns
    -------
    matplotlib.axes.Axes
    """
    
    if ax is None:
        _, ax = plt.subplots()

    # Unpack quantile settings
    df = quantile_settings['df'].copy()
    date_col = quantile_settings['date_col']
    value_cols = quantile_settings['value_cols']
    quantiles = quantile_settings['quantiles']
    color = quantile_settings.get('color', 'C0')

    unique_populations = df["population"].unique()
    conversion_dict = {state: convert_location_name_format(state, output_format="ISO") 
                    for state in unique_populations}
    df["population"] = df["population"].map(conversion_dict)

    if isinstance(value_cols, str):
        value_cols = [value_cols]

    # filter to state
    sdf = df[df["population"] == state].copy()
    if sdf.empty:
        ax.text(0.5, 0.5, f"No data for {state}", ha="center", va="center")
        ax.set_title(state)
        return ax

    # ensure date column is datetime-like for good plotting / grouping
    sdf[date_col] = pd.to_datetime(sdf[date_col])

    # 1) Sum value_cols per (date, sim_id) -> single numeric value per sim per date
    summed = (
        sdf.groupby([date_col, "sim_id"])[value_cols]
        .sum()              # sums across rows for (date,sim_id) and across columns per next step
        .sum(axis=1)        # sum across the multiple value_cols -> a Series indexed by (date, sim_id)
        .rename("value")
        .reset_index()
    )

    # 2) For each date, compute the requested quantiles of that summed value across sim_id
    qlist = sorted([float(q) for q in quantiles])
    # groupby date and apply robust quantile computation which always returns floats as column labels
    qdf = (
        summed.groupby(date_col)["value"]
        .apply(lambda s: _compute_date_quantiles(s, qlist))
        .unstack(level=-1)   # now each column is a float quantile value
        .sort_index()
    )

    # Defensive: ensure columns are floats and sorted
    try:
        qdf.columns = qdf.columns.astype(float)
    except Exception:
        # if this fails, coerce via list(map(float,...))
        qdf.columns = [float(c) for c in qdf.columns]

    qdf = qdf.reindex(columns=qlist)  # reorder columns to sorted quantiles

    # Determine banding: pair lowest with highest, etc.
    n = len(qlist)
    mid = n // 2
    has_center = (n % 2 == 1)
    center_q = qlist[mid] if has_center else None

    # Plot outer-to-inner so inner patches overpaint outer ones (better visibility)
    for k in range(mid):
        low = qlist[k]
        high = qlist[-(k+1)]
        # safety: if any are all-NaN, skip
        y1 = qdf[low].values
        y2 = qdf[high].values
        ax.fill_between(
            qdf.index,
            y1,
            y2,
            color=color,
            alpha=0.25 * (1 + 0.6 * (1 - k / max(1, mid-1))),  # slightly stronger inner bands
            linewidth=0,
            step=None,
        )

    # center line (if odd)
    if has_center:
        ax.plot(qdf.index, qdf[center_q].values, color=color, linewidth=1.5)

    # rotate x-ticks 45 degrees
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
        tick.set_ha("right")
            
    # tidy
    ax.set_title(state)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)

    # Add surveillance data if provided
    if surveillance_settings is not None:
        plot_surveillance_timeseries(
            df=surveillance_settings['df'],
            location_value=state,
            location_col=surveillance_settings['location_col'],
            date_col=surveillance_settings['date_col'],
            target_col=surveillance_settings['target_col'],
            size=surveillance_settings['size'],
            color="black",
            ax=ax,
        )

    # Add forecast reference date line if provided
    if forecast_reference_date is not None:
        ref_date = pd.to_datetime(forecast_reference_date)
        ax.axvline(ref_date, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

    return ax


def plot_state_quantiles_grid(
    states: list[str],
    quantile_settings: dict,
    surveillance_settings: dict | None = None,
    forecast_reference_date: str | pd.Timestamp | None = None,
    panels_per_row: int = 4,
    figsize: tuple[int, int] | None = None,
    outfile: str | None = None,
):
    """
    Create a multipanel grid of time-series quantile plots.

    Parameters
    ----------
    states : list[str]
        List of states (ISO format)
    quantile_settings : dict
        Dictionary containing:
            - df : pd.DataFrame with [population, date_col, sim_id] and value columns
            - date_col : str, column containing the date
            - value_cols : str or list[str], column(s) to plot quantiles of
            - quantiles : list[float], ordered list of quantiles to plot
            - color : str, base color for quantile shading and line
    surveillance_settings : dict, optional
        Dictionary containing:
            - df : pd.DataFrame with surveillance data
            - date_col : str, column containing dates
            - location_col : str, column containing geographic identifiers
            - target_col : str, column containing values to plot
            - size : float, marker size for scatter plot
    forecast_reference_date : str, pd.Timestamp, or None, optional
        If provided, adds a vertical dashed line at this date on all plots.
    panels_per_row : int, optional
        Number of panels per row in the grid.
    figsize : (int, int), optional
        Size of the full figure.
    outfile : str or None, optional
        If provided, save the figure to this path.

    Returns
    -------
    (Figure, ndarray of Axes)
    """

    n = len(states)
    ncols = panels_per_row
    nrows = int(np.ceil(n / ncols))

    # auto figsize if None
    if figsize is None:
        figsize = (4 * ncols, 3.6 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for idx, state in enumerate(states):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]

        plot_state_quantiles(
            state=state,
            quantile_settings=quantile_settings,
            surveillance_settings=surveillance_settings,
            forecast_reference_date=forecast_reference_date,
            ax=ax,
        )

    # Remove unused axes
    for idx in range(len(states), nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r, c].axis("off")

    plt.tight_layout()

    if outfile is not None:
        fig.savefig(outfile, bbox_inches="tight")

    return fig, axes


def plot_surveillance_timeseries(
    df: pd.DataFrame,
    location_value: str,
    location_col: str,
    date_col: str,
    target_col: str,
    size: float,
    color: str = "black",
    ax: plt.Axes | None = None,
    title: str | None = None,
):
    """
    Plot a time series for a specific location_value from a surveillance dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Surveillance dataframe. Must include location_col, date_col, and target_col.
    location_value : str
        The geographic region to filter.
    location_col : str
        Column name for geographic identifiers.
    date_col : str
        Column name for dates.
    target_col : str
        Column name for values to plot.
    size : float
        Marker size for scatter plot.
    color : str, optional
        Color of the markers.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. Creates a new figure if None.
    title : str, optional
        Plot title. If None, uses location_value.

    Returns
    -------
    matplotlib.axes.Axes
    """
    # Filter
    sdf = df[df[location_col] == location_value].copy()
    if sdf.empty:
        raise ValueError(f"No data found for {location_col}='{location_value}'")

    # Ensure dates are datetime
    sdf[date_col] = pd.to_datetime(sdf[date_col])

    # Sort by date
    sdf = sdf.sort_values(date_col)

    # Create axis if needed
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    ax.scatter(sdf[date_col], sdf[target_col], color=color, s=size)
    ax.set_xlabel("")
    ax.set_title(title if title is not None else location_value)

    # Rotate x-axis labels
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
        tick.set_ha("right")

    return ax

def plot_posterior(
    posterior: pd.DataFrame,
    state: str,
    parameter: str,
    ax: plt.Axes | None = None,
    start_date_reference: str | None = None,
    **kwargs,
) -> plt.Axes:
    """
    Plot the posterior distribution for one state-parameter pair.

    Parameters
    ----------
    posterior : pd.DataFrame
        Posterior samples containing a 'state' column and the parameter column.
    state : str
        State whose samples should be plotted.
    parameter : str
        Name of the parameter to plot.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on. If None, a new axis is created.
    start_date_reference : str or None, optional
        If parameter == "start_date", samples are treated as integer day offsets
        from this reference date and converted to actual dates.
    **kwargs :
        Additional arguments passed to `plot_posterior_distribution` from epydemix.

    Returns
    -------
    matplotlib.axes.Axes
        The axis containing the posterior plot.
    """
    # Filter the state
    data = posterior.loc[posterior["state"] == state, parameter].copy()

    # Handle start_date offset → actual datetime
    labels_as_dates = False
    if parameter == "start_date" and start_date_reference is not None:
        ref_date = pd.to_datetime(start_date_reference)
        date_series = ref_date + pd.to_timedelta(data, unit="D")
        data_numeric = date_series.map(datetime.toordinal)
        labels_as_dates = True
        plotting_df = pd.DataFrame({parameter: data_numeric})
    else:
        plotting_df = pd.DataFrame({parameter: data})

    # Create axis if needed
    if ax is None:
        _, ax = plt.subplots()

    # Use the new unified plotting function
    ax = plot_posterior_distribution(
        posterior=plotting_df,
        parameter=parameter,
        ax=ax,
        **kwargs,
    )

    # Fix date tick labels if needed
    if labels_as_dates:
        xticks = ax.get_xticks()

        # Force these ticks to be fixed before assigning labels
        ax.set_xticks(xticks)

        labels = [
            datetime.fromordinal(int(v)).strftime("%Y-%m-%d")
            for v in xticks
        ]
        ax.set_xticklabels(labels, rotation=45, ha="right")

    ax.set_title(parameter)
    return ax

def plot_state_posterior_grid(
    posterior: pd.DataFrame,
    states: list[str],
    parameters: list[str],
    rows_per_state: int | list[int] | np.ndarray,
    bins: int = 25,
    figsize: tuple[int, int] = (12, 8),
    start_date_reference: str | None = None,
    outfile: str | None = None,
    **kwargs,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot a grid of posterior distributions with rows = states and columns = parameters.

    Parameters
    ----------
    posterior : pd.DataFrame
        Posterior samples. Length must match the total rows implied by rows_per_state.
    states : list[str]
        States to plot, one per grid row.
    parameters : list[str]
        Parameters to plot, one per grid column.
    rows_per_state : int or list-like
        - If int: each state is assigned the same number of posterior samples.
        - If list/array: must match len(states); entries give samples per state.
    bins : int, optional
        Histogram bins passed to the posterior plotting function.
    figsize : (int, int), optional
        Figure size.
    start_date_reference : str or None, optional
        Reference date for converting start_date offsets.
    outfile : str or None, optional
        If provided, the figure is saved to this path.
    **kwargs :
        Additional arguments passed to `plot_posterior_distribution`.

    Returns
    -------
    (Figure, ndarray of Axes)
        The figure and the axes grid.
    """


    # --- Normalize rows_per_state ---
    if isinstance(rows_per_state, int):
        counts = [rows_per_state] * len(states)
    else:
        # convert tuples, pandas Series, numpy arrays to list
        counts = list(rows_per_state)
        if len(counts) != len(states):
            raise ValueError(
                f"rows_per_state must have length {len(states)}, got {len(counts)}"
            )

    # --- Validate total sample count ---
    expected_total = sum(counts)
    if len(posterior) != expected_total:
        raise ValueError(
            f"Row count mismatch: expected {expected_total}, got {len(posterior)}"
        )

    # --- Assign states to rows ---
    posterior = posterior.copy()
    posterior["state"] = np.concatenate([
        np.repeat(state, n) for state, n in zip(states, counts)
    ])

    # --- Figure layout ---
    n_rows = len(states)
    n_cols = len(parameters)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    for i, state in enumerate(states):
        for j, param in enumerate(parameters):

            ax = axes[i, j]

            plot_posterior(
                posterior=posterior,
                state=state,
                parameter=param,
                ax=ax,
                start_date_reference=start_date_reference,
                bins=bins,
                **kwargs,
            )

            # Remove x-labels
            ax.set_xlabel("")

            # Only first column keeps the state label
            if j == 0:
                ax.set_ylabel(state)
            else:
                ax.set_ylabel("")

            # Only first row gets titles
            if i == 0:
                ax.set_title(param)
            else:
                ax.set_title("")

    plt.tight_layout()

    if outfile is not None:
        fig.savefig(outfile, bbox_inches="tight")

    return fig, axes
