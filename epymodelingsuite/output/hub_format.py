"""Shared value formatting for hub forecast targets."""

import numpy as np
import pandas as pd


def normalize_target_values(values: np.ndarray | pd.Series, target: str) -> np.ndarray | pd.Series:
    """Clip and round forecast values according to the target's units.

    Parameters
    ----------
    values : np.ndarray or pd.Series
        Numeric forecast values, including missing values. Arrays may have any
        shape; Series retain their index and name. The input is not modified.
    target : str
        ``wk inc flu hosp`` rounds non-negative counts to integers;
        ``wk inc flu prop ed visits`` clips proportions to [0, 1];
        ``Flu ED visits pct`` and ``ILI ED visits pct`` clip percentages to
        [0, 100]. Values for other targets are returned unchanged. No units
        are converted.

    Returns
    -------
    np.ndarray or pd.Series
        New values of the same container type and shape, with missing values
        preserved. Dtype conversion for file output is handled separately.

    Notes
    -----
    NumPy rint, like pandas round, rounds exact .5 ties to the nearest even
    integer (10.5 -> 10, 11.5 -> 12).
    """
    upper_bounds = {
        "wk inc flu hosp": None,
        "wk inc flu prop ed visits": 1,
        "Flu ED visits pct": 100,
        "ILI ED visits pct": 100,
    }
    if target not in upper_bounds:
        return values.copy()
    normalized = np.clip(values, 0, upper_bounds[target])
    return np.rint(normalized) if target == "wk inc flu hosp" else normalized
