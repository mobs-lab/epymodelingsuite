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
    """Batch levels per numeric variable, preserving values and input arrays.

    Arrays have shape (draws, dates). Rows are quantile-major, as in epydemix.
    Non-numeric metadata is ignored. Low-precision floats retain scalar calls
    to preserve NumPy's rounding. With no supplied dates, retain epydemix's
    integer timestep labels and its error for an empty trajectory mapping.
    """
    if dates is None:
        dates = np.arange(list(trajectories.values())[0].shape[1])  # noqa: RUF015 -- preserve epydemix's IndexError
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
        if values.dtype.kind == "f" and values.dtype.itemsize < np.dtype(np.float64).itemsize:
            # ponytail: NumPy's scalar and vector q interpolate low-precision floats
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
    """Use the public trajectory API, retaining generation and variable selection."""
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
    """Use the public trajectory API, retaining scenario and variable selection."""
    trajectories = results.get_projection_trajectories(scenario_id, variables=variables)
    return compute_quantiles(trajectories, dates, quantiles, ignore_nan=ignore_nan)
