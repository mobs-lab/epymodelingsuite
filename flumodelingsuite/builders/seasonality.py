"""Seasonality building functions for EpiModel instances."""

import logging

import numpy as np
from epydemix.model import EpiModel

from ..schema.basemodel import Seasonality, Timespan
from ..seasonality import get_seasonal_transmission_balcan

logger = logging.getLogger(__name__)


def add_seasonality_from_config(model: EpiModel, seasonality: Seasonality, timespan: Timespan) -> EpiModel:
    """
    Add seasonally varying transmission rate to the EpiModel.

    Parameters
    ----------
        model: The EpiModel instance to apply seasonality to.
        seasonality: Seasonality configuration object.
        timespan: Timespan configuration object with simulation dates.

    Returns
    -------
        EpiModel instance with seasonal transmission applied.
    """
    # Parameter must already be defined
    try:
        previous_value = model.get_parameter(seasonality.target_parameter)
    except KeyError:
        raise ValueError(f"Attempted to apply seasonality to undefined parameter {seasonality.target_parameter}")

    # Calculate rescaling factor with requested method
    if seasonality.method == "balcan":
        # Minimum transmission date is optional
        if seasonality.seasonality_min_date is not None:
            date_tmin = seasonality.seasonality_min_date
        else:
            date_tmin = None
        # Do the calculation
        dates, st = get_seasonal_transmission_balcan(
            date_start=timespan.start_date,
            date_stop=timespan.end_date,
            date_tmax=seasonality.seasonality_max_date,
            date_tmin=date_tmin,
            val_min=seasonality.min_value,
            val_max=seasonality.max_value,
            delta_t=timespan.delta_t,
        )
    else:
        raise ValueError(f"Undefined seasonality method recieved: {seasonality.method}")

    # Handle possibilities for previous parameter value (expressions should already be evaluated at parameter definition)
    T = len(st)
    N = model.population.num_groups
    # If existing parameter is constant, transform to array of size (T, 1) with time-varying values
    # If existing parameter is time-varying (array of size (T, 1)), do piecewise multiplication
    if (not hasattr(previous_value, "__len__")) or previous_value.shape == (T,):
        new_value = np.array(st) * np.array(previous_value)
    # If existing parameter is age-varying (array of size (1, N)), transform to array of size (T, N) with time-varying and age-varying values
    # If existing parameter is time-varying and age-varying (array of size (T, N)), do piecewise for each age group
    elif previous_value.shape == (T, N) or previous_value.shape == (1, N):
        new_value = np.zeros((T, N))
        for i in range(N):
            new_value[:, i] = np.array(st) * np.array(previous_value[:, i])
    # Uncertain how this will work for priors
    else:
        raise ValueError(
            f"Cannot apply seasonality to existing parameter {seasonality.target_parameter} = {previous_value}"
        )

    # Overwrite parameter with new seasonal values
    try:
        model.add_parameter(seasonality.target_parameter, new_value)
        logger.info(f"Added seasonality to parameter {seasonality.target_parameter}")
    except Exception as e:
        raise ValueError(f"Error adding parameters to model: {e}")

    return model
