"""Distance function utilities."""

import numpy as np
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
