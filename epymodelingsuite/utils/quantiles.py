"""Batched quantiles with epydemix's output layout and interpolation defaults."""

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
from epydemix.calibration import CalibrationResults


def compute_quantiles(
    trajectories: dict[str, np.ndarray],
    dates: Sequence | np.ndarray | None,
    quantiles: Sequence[float],
    *,
    ignore_nan: bool = False,
) -> pd.DataFrame:
    """Compute quantiles across draws without modifying the input arrays.

    Parameters
    ----------
    trajectories : dict[str, ndarray]
        Variable arrays of shape (draws, dates); nonnumeric arrays are ignored.
    dates : sequence or ndarray or None, optional
        Labels for the time axis. None uses integer timestep labels.
    quantiles : sequence of float
        Quantile levels in [0, 1], preserving the requested order and duplicates.
    ignore_nan : bool, optional
        If True, exclude NaNs and warn when a time point has more than 50% NaNs.

    Returns
    -------
    pd.DataFrame
        Quantile-major rows with date, quantile and numeric variable columns.

    Raises
    ------
    ValueError
        A numeric array is not two-dimensional or NumPy rejects the quantile levels.

    Notes
    -----
    float16/float32 retain scalar quantile calls to preserve NumPy rounding.
    With missing dates, an empty trajectory mapping retains epydemix's IndexError.
    """
    if dates is None:
        dates = np.arange(list(trajectories.values())[0].shape[1])  # noqa: RUF015 -- preserve epydemix's IndexError
    # Quantile-major rows: every date for the first level, then every date for the next level.
    data = {
        "date": [d for _ in quantiles for d in dates],
        "quantile": [q for q in quantiles for _ in dates],
    }
    quantile_func = np.nanquantile if ignore_nan else np.quantile
    for name, values in trajectories.items():
        if not np.issubdtype(values.dtype, np.number):
            continue
        if values.ndim != 2:  # noqa: PLR2004 -- draws and dates
            msg = f"Variable '{name}' must have shape (draws, dates), got {values.shape}."
            raise ValueError(msg)
        if ignore_nan:
            max_nan_prop = np.max(np.isnan(values).mean(axis=0))
            if max_nan_prop > 0.5:  # noqa: PLR2004 -- epydemix's warning threshold
                warnings.warn(
                    f"Variable '{name}' has time points with up to {max_nan_prop:.1%} NaN values. "
                    "Quantiles at these time points may be unreliable due to small sample size.",
                    stacklevel=2,
                )
        is_low_precision_float = values.dtype.kind == "f" and values.dtype.itemsize < np.dtype(np.float64).itemsize
        if is_low_precision_float:
            # NumPy's scalar and vector q interpolate low-precision floats
            # differently; retain scalar calls until NumPy gives identical results.
            data[name] = [value for q in quantiles for value in quantile_func(values, q, axis=0)]
        else:
            data[name] = list(quantile_func(values, quantiles, axis=0).reshape(-1))
    return pd.DataFrame(data)


def get_calibration_quantiles(  # noqa: PLR0913 -- match the epydemix public API
    results: CalibrationResults,
    *,
    dates: Sequence | np.ndarray | None = None,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
    generation: int | None = None,
    variables: list[str] | None = None,
    ignore_nan: bool = False,
) -> pd.DataFrame:
    """Compute calibration quantiles using epydemix's public trajectory API.

    Parameters
    ----------
    results : CalibrationResults
        Source results; their identity distinguishes shared computation groups.
    dates : sequence or ndarray or None, optional
        Labels for the time axis. None uses integer timestep labels.
    quantiles : sequence of float
        Quantile levels in [0, 1], preserving the requested order and duplicates.
    generation : int or None, optional
        Calibration generation to select. None uses epydemix's default selection.
    variables : list of str or None, optional
        Variables to stack. None or an empty list selects all variables through epydemix.
    ignore_nan : bool, optional
        If True, exclude NaNs and warn when a time point has more than 50% NaNs.

    Returns
    -------
    pd.DataFrame
        Quantile-major rows with date, quantile and numeric variable columns.

    Notes
    -----
    Trajectory selection and validation errors propagate from epydemix.
    """
    trajectories = results.get_calibration_trajectories(generation, variables=variables)
    return compute_quantiles(trajectories, dates, quantiles, ignore_nan=ignore_nan)


def get_projection_quantiles(  # noqa: PLR0913 -- match the epydemix public API
    results: CalibrationResults,
    *,
    dates: Sequence | np.ndarray | None = None,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
    scenario_id: str = "baseline",
    variables: list[str] | None = None,
    ignore_nan: bool = False,
) -> pd.DataFrame:
    """Compute projection quantiles using epydemix's public trajectory API.

    Parameters
    ----------
    results : CalibrationResults
        Source results; their identity distinguishes shared computation groups.
    dates : sequence or ndarray or None, optional
        Labels for the time axis. None uses integer timestep labels.
    quantiles : sequence of float
        Quantile levels in [0, 1], preserving the requested order and duplicates.
    scenario_id : str, optional
        Projection scenario key, defaulting to baseline.
    variables : list of str or None, optional
        Variables to stack. None or an empty list selects all variables through epydemix.
    ignore_nan : bool, optional
        If True, exclude NaNs and warn when a time point has more than 50% NaNs.

    Returns
    -------
    pd.DataFrame
        Quantile-major rows with date, quantile and numeric variable columns.

    Notes
    -----
    Scenario selection and validation errors propagate from epydemix.
    """
    trajectories = results.get_projection_trajectories(scenario_id, variables=variables)
    return compute_quantiles(trajectories, dates, quantiles, ignore_nan=ignore_nan)
