"""Distance function utilities."""

import copy
from collections.abc import Callable

import numpy as np
from epydemix.calibration import ae, mae, mape, rmse, wmape
from epydemix.calibration.metrics import validate_data


def wrmse(data: dict, simulation: dict):
    """
    RMSE weighted to favor recent data points.

    Uses time-dependent weighting factor w(t)=1/((t_n+1)-t)
    where t_n is the greatest timestep and t is in [1, t_n]

    Parameters
    ----------
    data: dict
        Dictionary containing the observed data with a key "data" pointing to an array of observations
    simulation: dict
        Dictionary containing the simulated data with a key "data" pointing to an array of simulated values.

    Returns
    -------
        float: The weighted RMSE value indicating the weighted average magnitude of the error between the observed and simulated data.
    """
    observed, simulated = validate_data(data, simulation)
    t_n = len(observed)
    w_t = np.array([1 / ((t_n + 1) - t) for t in range(1, t_n + 1)])

    return np.sqrt(np.nanmean(w_t * ((observed - simulated) ** 2))) / np.sum(w_t)


dist_func_dict = {
    "rmse": rmse,
    "wmape": wmape,
    "ae": ae,
    "mae": mae,
    "mape": mape,
    "wrmse": wrmse,
}


def count_nans_at_start(arr: np.ndarray) -> int:
    """
    Count the number of NaN values prepended to the input array, if any.

    Parameters
    ----------
    arr: np.ndarray
        The array to count NaNs from.

    Returns
    -------
    int
        The number of NaNs found prepended to the array.
    """
    # Handle empty array case
    if arr.size == 0:
        return 0

    # Create a boolean mask for NaN values
    mask = np.isnan(arr)

    # If all values are NaN, return len(arr)
    if mask.all():
        return len(arr)

    # If no values are NaN, return 0
    if not mask.any():
        return 0

    # Find the index of the first non-NaN value
    first_non_nan_index = np.argmax(~mask)
    return first_non_nan_index


def dist_func_date_alignment_wrapper(dist_func: Callable) -> Callable:
    """
    Create a wrapped version of an epydemix distance function which truncates observed and simulated
    trajectories to exclude prepended NaNs before applying the distance calculation.

    Parameters
    ----------
    dist_func: Callable
        An epydemix distance function.

    Returns
    -------
    Callable
        The wrapped distance function for handling prepended NaNs.
    """

    def wrapped_dist_func(data: dict, simulation: dict) -> float | np.ndarray:
        """
        A wrapped epydemix distance function which truncates observed and simulated
        trajectories to exclude prepended NaNs before applying the distance calculation.

        Parameters
        ----------
        data: dict
            A dictionary containing the observed data with a key "data" pointing to an array of observations.
        simulation: dict
            A dictionary containing the simulated data with a key "data" pointing to an array of simulated values.

        Returns
        -------
        float | np.ndarray
            The result of calling dist_func on the truncated data, which is np.ndarray if
            dist_func is Absolute Error and float otherwise.
        """
        observed = np.array(data["data"])
        simulated = np.array(simulation["data"])

        num_nans = count_nans_at_start(simulated)

        trunc_data = copy.deepcopy(data)
        trunc_simulation = copy.deepcopy(simulation)
        trunc_data["data"] = observed[num_nans:]
        trunc_simulation["data"] = simulated[num_nans:]

        return dist_func(trunc_data, trunc_simulation)

    return wrapped_dist_func


def multi_comparison_distance_wrapper(dist_func: Callable, keys: tuple[str, ...]) -> Callable:
    """Unwrap observed series from ABCSampler and trim them to a common start."""

    def wrapped_dist_func(data: dict, simulation: dict) -> float:
        simulated = {key: np.array(simulation[key]) for key in keys}
        start = max(count_nans_at_start(values) for values in simulated.values())
        return dist_func(
            {key: np.array(data["data"][key])[start:] for key in keys},
            {key: values[start:] for key, values in simulated.items()},
        )

    return wrapped_dist_func
