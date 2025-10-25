"""Model orchestration functions for creating collections, setting up schedules and interventions."""

import copy
import datetime as dt
import logging

import numpy as np
import pandas as pd
from epydemix import simulate
from epydemix.model import EpiModel

from ..schema.basemodel import BaseEpiModel, Parameter, Timespan
from ..schema.calibration import CalibrationConfig
from ..school_closures import make_school_closure_dict
from ..utils import get_location_codebook, make_dummy_population
from ..vaccinations import reaggregate_vaccines, scenario_to_epydemix
from .base import (
    add_model_compartments_from_config,
    add_model_parameters_from_config,
    add_model_transitions_from_config,
    calculate_parameters_from_config,
    set_population_from_config,
)
from .interventions import (
    add_contact_matrix_interventions_from_config,
    add_parameter_interventions_from_config,
    add_school_closure_intervention_from_config,
)
from .seasonality import add_seasonality_from_config
from .vaccination import add_vaccination_schedules_from_config

logger = logging.getLogger(__name__)


def create_model_collection(
    basemodel: BaseEpiModel,
    population_names: list[str] | None,
) -> tuple[list[EpiModel], list[str]]:
    """
    Create a collection of EpiModels with dummy population setup.

    This function creates an initial EpiModel with a dummy population, adds
    compartments, transitions, and parameters, then creates copies for each
    specified population location.

    Parameters
    ----------
    basemodel : BaseEpiModel
        The base model configuration containing compartments, transitions,
        parameters, and population settings.
    population_names : list[str] | None
        List of population names to create models for. Can contain "all" to
        expand to all locations in the codebook. If None, uses the single
        population from basemodel.

    Returns
    -------
    tuple[list[EpiModel], list[str]]
        - List of configured EpiModel instances (one per population)
        - List of resolved population names (expanded if "all" was specified)

    Examples
    --------
    >>> models, pop_names = create_model_collection(basemodel, ["California", "Texas"])
    >>> len(models)
    2
    >>> models, pop_names = create_model_collection(basemodel, ["all"])
    >>> len(models)
    51  # All US states
    """
    models = []
    init_model = EpiModel()

    # Set the model name if provided in the config
    if basemodel.name is not None:
        init_model.name = basemodel.name

    logger.info("BUILDER: setting up EpiModels...")

    # Add dummy population with age structure (required for static age-structured parameters)
    dummy_pop = make_dummy_population(basemodel)
    init_model.set_population(dummy_pop)

    # All models will share compartments, transitions, and non-sampled/calculated parameters
    add_model_compartments_from_config(init_model, basemodel.compartments)
    add_model_transitions_from_config(init_model, basemodel.transitions)
    add_model_parameters_from_config(init_model, basemodel.parameters)

    # Create models with populations set
    if population_names:
        if "all" in population_names:
            resolved_names = get_location_codebook()["location_name_epydemix"]
        else:
            resolved_names = population_names
        for name in resolved_names:
            m = copy.deepcopy(init_model)
            set_population_from_config(m, name, basemodel.population.age_groups)
            models.append(m)
    else:
        set_population_from_config(init_model, basemodel.population.name, basemodel.population.age_groups)
        models.append(init_model)
        resolved_names = [basemodel.population.name]

    return models, resolved_names


def setup_vaccination_schedules(
    basemodel: BaseEpiModel,
    models: list[EpiModel],
    sampled_start_timespan: Timespan | None,
    population_names: list[str],
) -> tuple[list[EpiModel], dict | None]:
    """
    Set up vaccination schedules, handling start_date sampling if present.

    This function handles two scenarios:
    1. If start_date is sampled (sampled_start_timespan exists): Creates an earliest
       vaccination schedule that will later be reaggregated for each sampled start_date
    2. If start_date is not sampled: Adds vaccination schedules to all models

    Parameters
    ----------
    basemodel : BaseEpiModel
        The base model configuration containing vaccination settings.
    models : list[EpiModel]
        List of EpiModel instances to add vaccination schedules to
        (only used if start_date is not sampled).
    sampled_start_timespan : Timespan | None
        The earliest timespan when start_date is sampled, or None if not sampled.
    population_names : list[str]
        List of population/location names (states) to create vaccination schedules for.

    Returns
    -------
    tuple[list[EpiModel], dict | None]
        - List of EpiModel instances (with vaccination applied if start_date not sampled)
        - The earliest vaccination schedule (for later reaggregation) if start_date is sampled,
          otherwise None.

    Notes
    -----
    Vaccination schedules are sensitive to location and start_date but not to model parameters.
    When start_date is sampled, we precalculate the schedule from the earliest possible date,
    then reaggregate it later for each specific sampled start_date value.
    """
    if not basemodel.vaccination:
        return models, None

    # If start_date is sampled, precalculate schedule with earliest start for later reaggregation
    if sampled_start_timespan:
        earliest_vax = scenario_to_epydemix(
            input_filepath=basemodel.vaccination.scenario_data_path,
            start_date=sampled_start_timespan.start_date,
            end_date=sampled_start_timespan.end_date,
            target_age_groups=basemodel.population.age_groups,
            delta_t=sampled_start_timespan.delta_t,
            states=population_names,
        )
        return models, earliest_vax

    # If start_date not sampled, add vaccination to models now
    for model in models:
        add_vaccination_schedules_from_config(model, basemodel.transitions, basemodel.vaccination, basemodel.timespan)
    return models, None


def setup_interventions(
    models: list[EpiModel],
    basemodel: BaseEpiModel,
    intervention_types: list[str],
    sampled_start_timespan: Timespan | None,
) -> list[EpiModel]:
    """
    Set up interventions that depend on location but not model parameters.

    This function handles interventions that are sensitive to location/population
    but not to model parameters, allowing them to be applied before the models
    are further duplicated with different parameter sets.

    Currently handles:
    - School closure interventions (date-dependent)
    - Contact matrix interventions (location-dependent)

    Parameters
    ----------
    models : list[EpiModel]
        List of EpiModel instances to add interventions to.
    basemodel : BaseEpiModel
        The base model configuration containing intervention settings and timespan.
    intervention_types : list[str]
        List of intervention type strings extracted from basemodel.interventions.
    sampled_start_timespan : Timespan | None
        The earliest timespan when start_date is sampled, or None if not sampled.
        Used to determine the date range for school closures.

    Returns
    -------
    list[EpiModel]
        List of EpiModel instances with interventions applied.

    Notes
    -----
    These interventions are applied early in the model building process because they:
    1. Depend on location/population (different for each model in the collection)
    2. Do NOT depend on sampled/calibrated parameters
    3. Can be applied once before models are further duplicated with parameter variations
    """
    if not basemodel.interventions:
        return models

    # Determine the effective timespan for date-dependent interventions
    effective_timespan = sampled_start_timespan if sampled_start_timespan else basemodel.timespan

    for model in models:
        # School closure
        if "school_closure" in intervention_types:
            closure_dict = make_school_closure_dict(
                range(effective_timespan.start_date.year, effective_timespan.end_date.year + 1)
            )
            add_school_closure_intervention_from_config(model, basemodel.interventions, closure_dict)

        # Contact matrix
        if "contact_matrix" in intervention_types:
            add_contact_matrix_interventions_from_config(model, basemodel.interventions)

    return models


def make_simulate_wrapper(
    basemodel: BaseEpiModel,
    calibration: CalibrationConfig,
    data_state: pd.DataFrame,
    intervention_types: list[str],
    sampled_start_timespan: Timespan | None = None,
    earliest_vax: dict | None = None,
) -> callable:
    """
    Create a simulate_wrapper function for ABCSampler calibration.
    simulate_wrapper takes param dictionary and runs a simulation/projection.

    Parameters
    ----------
    basemodel : BaseEpiModel
        Base model configuration with compartments, parameters, interventions, etc.
    calibration : CalibrationConfig
        Calibration settings including comparison targets, priors, and fitting window.
    data_state : pd.DataFrame
        Observed data for this specific location/model (already filtered to location).
    intervention_types : list[str]
        List of intervention types to apply (e.g., ["parameter", "school_closure"]).
    sampled_start_timespan : Timespan | None, optional
        Earliest timespan if start_date is being sampled/calibrated.
        If provided, enables start_date sampling and vaccination reaggregation.
    earliest_vax : dict | None, optional
        Pre-calculated vaccination schedule from earliest start_date.
        Required when sampled_start_timespan is provided and vaccinations are used.

    Returns
    -------
    callable
        A simulate_wrapper function, which takes params:dict and returns dict.
        This wrapper is passed to ABCSampler and will be called during calibration.

    Notes
    -----
    The wrapper function expects params dict to contain:
    - "epimodel": The model to simulate (passed via fixed_parameters)
    - "end_date": Simulation end date
    - "projection": Boolean indicating calibration vs projection mode
    - Calibrated parameter values (e.g., "beta", "initial_infected")
    - Optional "start_date" (offset in days, if start_date is calibrated)
    - Optional "seasonality_min" (if seasonality is calibrated)

    The wrapper returns:
    - For calibration (projection=False): {"data": np.ndarray} of simulated values
      at observation times, aligned with observed data dates
    - For projection (projection=True): {"dates": list, "transitions": dict,
      "compartments": dict} with full simulation results

    """

    def simulate_wrapper(params: dict) -> dict:
        # Extract model from params
        wrapper_model = params["epimodel"]
        m = copy.deepcopy(wrapper_model)

        # Accommodate for sampled start_date
        if sampled_start_timespan:
            start_date = sampled_start_timespan.start_date + dt.timedelta(days=params["start_date"])
        else:
            start_date = basemodel.timespan.start_date

        timespan = Timespan(start_date=start_date, end_date=params["end_date"], delta_t=basemodel.timespan.delta_t)

        # Sampled/calculated parameters
        new_params = {
            k: Parameter(type="scalar", value=v)
            for k, v in params.items()
            if k in basemodel.parameters and basemodel.parameters[k].type == "calibrated"
        }
        if new_params:
            add_model_parameters_from_config(m, new_params)
        if "calculated" in [param_args.type.value for _, param_args in basemodel.parameters.items()]:
            calculate_parameters_from_config(m, basemodel.parameters)

        # Vaccination (if start_date is sampled)
        if basemodel.vaccination and sampled_start_timespan:
            reaggregated_vax = reaggregate_vaccines(earliest_vax, timespan.start_date)
            add_vaccination_schedules_from_config(
                m, basemodel.transitions, basemodel.vaccination, timespan, use_schedule=reaggregated_vax
            )

        # Seasonality (this must occur before parameter interventions to preserve parameter overrides)
        if basemodel.seasonality:
            if "seasonality_min" in params:
                basemodel.seasonality.min_value = params["seasonality_min"]
            add_seasonality_from_config(m, basemodel.seasonality, timespan)

        # Parameter interventions
        if basemodel.interventions and "parameter" in intervention_types:
            add_parameter_interventions_from_config(m, basemodel.interventions, timespan)

        # Initial conditions
        from .base import calculate_compartment_initial_conditions

        compartment_init = calculate_compartment_initial_conditions(
            compartments=basemodel.compartments,
            population_array=m.population.Nk,
            sampled_compartments=params,
        )

        # Collect settings
        sim_params = {
            "epimodel": m,
            "initial_conditions_dict": compartment_init,
            "start_date": timespan.start_date,
            "end_date": params["end_date"],
            "resample_frequency": basemodel.simulation.resample_frequency,
        }

        # Run simulation
        try:
            results = simulate(**sim_params)
        except (ValueError, RuntimeError, KeyError) as e:
            failed_params = params.copy()
            failed_params.pop("epimodel", None)
            logger.warning("Simulation failed with parameters %s: %s", failed_params, e)

            if params["projection"]:
                return {}

            # Calibration mode: return zero-filled array
            data_dates = list(pd.to_datetime(data_state.target_end_date.values))
            return {"data": np.full(len(data_dates), 0)}

        # Format output based on mode
        if params["projection"]:
            # Projection: return full results
            return {
                "dates": results.dates,
                "transitions": results.transitions,
                "compartments": results.compartments,
            }

        # Calibration: align to observed dates
        trajectory_dates = results.dates
        data_dates = list(pd.to_datetime(data_state.target_end_date.values))

        mask = [date in data_dates for date in trajectory_dates]

        total_hosp = sum(results.transitions[key] for key in calibration.comparison[0].simulation)

        total_hosp = total_hosp[mask]

        if len(total_hosp) < len(data_dates):
            pad_len = len(data_dates) - len(total_hosp)
            total_hosp = np.pad(total_hosp, (pad_len, 0), constant_values=0)

        return {"data": total_hosp}

    return simulate_wrapper
