"""Seasonality building functions for EpiModel instances."""

import logging
from typing import Any

import numpy as np
from epydemix.model import EpiModel

from ..schema.basemodel import Seasonality, Timespan
from ..seasonality import (
    get_seasonal_transmission_balcan,
    get_seasonal_transmission_data_driven,
    load_seasonality_data,
)

logger = logging.getLogger(__name__)


def resolve_seasonality_location_from_population(
    population_name: str,
    location_format: str = "ISO",
) -> str:
    """
    Map an epydemix population name to a seasonality data CSV location identifier.

    ISO state populations use their ISO code (e.g. US-CA). Metrocast locations
    inherit the parent state's seasonality series, matching contact-matrix resolution.
    """
    from ..utils.location import convert_location_name_format, get_parent_region, parse_population_name

    location_name, location_type = parse_population_name(population_name)
    if location_type == "iso":
        iso = convert_location_name_format(location_name, "ISO")
    else:
        iso = get_parent_region(location_name, output_format="ISO", granularity="state")

    if location_format == "ISO":
        return iso
    return convert_location_name_format(iso, location_format, input_format="ISO")


def _resolve_seasonality_coeff(
    param_name: str,
    model: EpiModel,
    param_overrides: dict[str, Any] | None,
) -> float:
    """Resolve a scalar seasonality coefficient from overrides or model parameters."""
    if param_overrides is not None and param_name in param_overrides:
        value = param_overrides[param_name]
    else:
        try:
            value = model.get_parameter(param_name)
        except KeyError as e:
            raise ValueError(
                f"Data-driven seasonality requires coefficient {param_name!r} as a model parameter "
                f"or in param_overrides"
            ) from e

    if hasattr(value, "__len__") and not isinstance(value, str):
        raise ValueError(
            f"Data-driven seasonality coefficient {param_name!r} must be scalar, got shape "
            f"{getattr(value, 'shape', len(value))}"
        )
    return float(value)


def add_seasonality_from_config(
    model: EpiModel,
    seasonality: Seasonality,
    timespan: Timespan,
    param_overrides: dict[str, Any] | None = None,
) -> EpiModel:
    """
    Add seasonally varying transmission rate to the EpiModel.

    Parameters
    ----------
        model: The EpiModel instance to apply seasonality to.
        seasonality: Seasonality configuration object.
        timespan: Timespan configuration object with simulation dates.
        param_overrides: Optional sampled parameter values (e.g. calibrated b1, b2, b3).

    Returns
    -------
        The same EpiModel instance with seasonal transmission applied (modified in-place).
    """
    # Parameter must already be defined
    try:
        previous_value = model.get_parameter(seasonality.target_parameter)
    except KeyError:
        raise ValueError(f"Attempted to apply seasonality to undefined parameter {seasonality.target_parameter}")

    # Calculate rescaling factor with requested method
    if seasonality.method == Seasonality.SeasonalityMethodEnum.balcan:
        if seasonality.seasonality_min_date is not None:
            date_tmin = seasonality.seasonality_min_date
        else:
            date_tmin = None
        dates, st = get_seasonal_transmission_balcan(
            date_start=timespan.start_date,
            date_stop=timespan.end_date,
            date_tmax=seasonality.seasonality_max_date,
            date_tmin=date_tmin,
            val_min=seasonality.min_value,
            val_max=seasonality.max_value,
            delta_t=timespan.delta_t,
        )
    elif seasonality.method in (
        Seasonality.SeasonalityMethodEnum.data_driven,
        Seasonality.SeasonalityMethodEnum.humidity_only,
        Seasonality.SeasonalityMethodEnum.temperature_only,
    ):
        humidity_only = seasonality.method == Seasonality.SeasonalityMethodEnum.humidity_only
        temperature_only = seasonality.method == Seasonality.SeasonalityMethodEnum.temperature_only
        if humidity_only:
            b1, b3, b4 = 1.0, 0.0, 0.0
        elif temperature_only:
            b1, b3, b4 = 0.0, 1.0, 0.0
        else:
            b1 = _resolve_seasonality_coeff(seasonality.b1_param, model, param_overrides)
            b3 = _resolve_seasonality_coeff(seasonality.b3_param, model, param_overrides)
            b4 = (
                _resolve_seasonality_coeff(seasonality.b4_param, model, param_overrides)
                if seasonality.mobility_column
                else 0.0
            )
        s_min = _resolve_seasonality_coeff(seasonality.s_min_param, model, param_overrides)
        seasonality_location = resolve_seasonality_location_from_population(
            model.population.name,
            location_format=seasonality.location_format,
        )
        seasonality_data = load_seasonality_data(
            seasonality_data_path=seasonality.seasonality_data_path,
            date_column=seasonality.date_column,
            temp_column=seasonality.temp_column,
            rh_column=seasonality.rh_column,
            mobility_column=seasonality.mobility_column,
            location=seasonality_location,
            location_column=seasonality.location_column,
        )
        logger.info(
            "Data-driven seasonality for population %s using location %s (method=%s)",
            model.population.name,
            seasonality_location,
            seasonality.method.value,
        )
        dates, st = get_seasonal_transmission_data_driven(
            date_start=timespan.start_date,
            date_stop=timespan.end_date,
            seasonality_data=seasonality_data,
            b1=b1,
            b3=b3,
            s_min=s_min,
            rh_optimum=seasonality.rh_optimum,
            delta_t=timespan.delta_t,
            b4=b4,
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
