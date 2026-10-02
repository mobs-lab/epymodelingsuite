"""Distance function utilities."""

import numpy as np
from epydemix.calibration.metrics import validate_data


def wrmse(data: dict, simulation: dict) -> float:
    """
    RMSE weighted to favor recent data points.

    Uses time-dependent weighting factor w(t)=1/((t_n+1)-t)
    where t_n is the greatest timestep and t is in [1, t_n].
    Computes sqrt(sum(w * error**2) / sum(w)) over points where both observed and simulated values are not NaN.

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

    # Normalize by the weights of the points that have both an observation and a prediction
    valid = ~(np.isnan(observed) | np.isnan(simulated))
    w_t = w_t[valid]
    squared_error = (observed[valid] - simulated[valid]) ** 2

    return float(np.sqrt(np.sum(w_t * squared_error) / np.sum(w_t)))
