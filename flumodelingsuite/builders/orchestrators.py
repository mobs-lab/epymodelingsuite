"""Model orchestration functions for creating collections, setting up schedules and interventions."""

import copy
import datetime as dt
import logging
from collections.abc import Callable
from typing import Any

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
    calculate_compartment_initial_conditions,
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


def flatten_simulation_results(results: Any) -> dict:
    """
    Flatten simulation results for projection mode.

    Flattens results structure into a single dict with dates, transitions,
    and compartments at the top level.

    Parameters
    ----------
    results : SimulationResults
        Output from epydemix.simulate().

    Returns
    -------
    dict
        Flattened results with "dates" key and all transitions/compartments.
    """
    output = {"dates": results.dates}
    output.update(results.transitions)
    output.update(results.compartments)
    return output


def align_simulation_to_observed_dates(
    results: Any,
    comparison_transitions: list[str],
    data_dates: list,
) -> np.ndarray:
    """
    Align simulation output to observation dates and aggregate specified transitions.

    Parameters
    ----------
    results : SimulationResults
        Output from epydemix.simulate() with dates and transitions.
    comparison_transitions : list[str]
        List of transition keys to sum (e.g., ["Hosp_vax", "Hosp_unvax"]).
    data_dates : list
        List of observation dates to align to.

    Returns
    -------
    np.ndarray
        Aggregated simulation values aligned to observation dates.
        Pads with zeros if simulation starts after observations.
    """
    trajectory_dates = results.dates

    mask = np.array([date in data_dates for date in trajectory_dates])

    # Sum transitions specified in calibration config (e.g., Hospitalization = Hosp_vax + Hosp_unvax)
    comparison_transition_arrays = [results.transitions[key] for key in comparison_transitions]
    simulated_data = sum(comparison_transition_arrays)

    simulated_data = simulated_data[mask]

    # Pad with zeros at beginning if sampled start_date is later than earliest observation date
    if len(simulated_data) < len(data_dates):
        pad_len = len(data_dates) - len(simulated_data)
        simulated_data = np.pad(simulated_data, (pad_len, 0), constant_values=0)

    return simulated_data


def apply_seasonality_with_sampled_min(
    model: EpiModel,
    basemodel: BaseEpiModel,
    timespan: Timespan,
    params: dict,
) -> None:
    """
    Apply seasonality configuration, using sampled min_value if provided.

    Creates a deep copy of seasonality config to avoid mutating shared state,
    optionally overrides min_value if it's being calibrated, then applies to model.

    Parameters
    ----------
    model : EpiModel
        Model to add seasonality to.
    basemodel : BaseEpiModel
        Base configuration with seasonality settings.
    timespan : Timespan
        Simulation timespan.
    params : dict
        Simulation parameters, may contain "seasonality_min" if being calibrated.
    """
    if not basemodel.seasonality:
        return

    # Use copy to avoid mutating shared basemodel
    seasonality_config = copy.deepcopy(basemodel.seasonality)

    # Override min_value if sampled/calibrated
    if "seasonality_min" in params:
        seasonality_config.min_value = params["seasonality_min"]

    add_seasonality_from_config(model, seasonality_config, timespan)


def apply_vaccination_for_sampled_start(
    model: EpiModel,
    basemodel: BaseEpiModel,
    timespan: Timespan,
    earliest_vax: dict | None,
    sampled_start_timespan: Timespan | None,
) -> None:
    """
    Apply vaccination schedules, reaggregating if start_date is sampled.

    Only applies vaccination when start_date is sampled. When start_date is sampled,
    the vaccination schedule must be reaggregated to match the actual simulation
    start date.

    Parameters
    ----------
    model : EpiModel
        Model to add vaccination to.
    basemodel : BaseEpiModel
        Base configuration with vaccination settings.
    timespan : Timespan
        Actual simulation timespan (may differ from basemodel if start_date sampled).
    earliest_vax : dict | None
        Pre-calculated earliest vaccination schedule for reaggregation.
    sampled_start_timespan : Timespan | None
        If not None, indicates start_date is sampled and vaccination needs reaggregation.
    """
    if not basemodel.vaccination:
        return

    if sampled_start_timespan is None:
        # No start_date sampling, vaccination already applied in setup
        return

    # Start_date is sampled, need to reaggregate
    reaggregated_vax = reaggregate_vaccines(earliest_vax, timespan.start_date)
    add_vaccination_schedules_from_config(
        model, basemodel.transitions, basemodel.vaccination, timespan, use_schedule=reaggregated_vax
    )


def apply_calibrated_parameters(
    model: EpiModel,
    params: dict,
    parameter_config: dict[str, Parameter],
) -> None:
    """
    Apply calibrated and calculated parameters to model.

    Modifies model in-place by adding calibrated parameter values
    and recalculating derived parameters.

    Parameters
    ----------
    model : EpiModel
        Model to modify.
    params : dict
        Dictionary containing calibrated parameter values from ABC sampler.
    parameter_config : dict[str, Parameter]
        Parameter configuration from basemodel.
    """
    # Extract calibrated parameters
    calibrated_params = {
        k: Parameter(type="scalar", value=v)
        for k, v in params.items()
        if k in parameter_config and parameter_config[k].type == "calibrated"
    }

    if calibrated_params:
        add_model_parameters_from_config(model, calibrated_params)

    # Recalculate derived parameters if any exist
    has_calculated = any(param.type.value == "calculated" for param in parameter_config.values())
    if has_calculated:
        calculate_parameters_from_config(model, parameter_config)


def compute_simulation_start_date(
    params: dict,
    basemodel_timespan: Timespan,
    sampled_start_timespan: Timespan | None,
) -> dt.date:
    """
    Calculate simulation start date based on sampling configuration.

    Handles three cases:
    1. No sampling: use fixed start date from basemodel
    2. Projection mode: use earliest start for consistent trajectory lengths
    3. Calibration mode: use sampled offset from earliest start

    Parameters
    ----------
    params : dict
        Simulation parameters. If start_date sampling is enabled,
        params["start_date"] should be an integer offset in days.
    basemodel_timespan : Timespan
        Default timespan from base model configuration.
    sampled_start_timespan : Timespan | None
        Earliest timespan for start_date sampling, or None if not sampled.

    Returns
    -------
    dt.date
        Calculated start date for simulation.
    """
    # Case 1: No sampling - use fixed start date
    if sampled_start_timespan is None:
        return basemodel_timespan.start_date

    # Case 2: Projection mode - use earliest start for consistent trajectory lengths
    is_projection_mode = params.get("projection", False)
    if is_projection_mode:
        return sampled_start_timespan.start_date

    # Case 3: Calibration mode - use sampled offset
    offset_days = params["start_date"]
    return sampled_start_timespan.start_date + dt.timedelta(days=offset_days)


def make_simulate_wrapper(
    basemodel: BaseEpiModel,
    calibration: CalibrationConfig,
    data_state: pd.DataFrame,
    intervention_types: list[str],
    sampled_start_timespan: Timespan | None = None,
    earliest_vax: dict | None = None,
) -> Callable[[dict], dict]:
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
        # 1. Extract model from params
        wrapper_model = params["epimodel"]
        model = copy.deepcopy(wrapper_model)

        # 2. Calculate start date
        start_date = compute_simulation_start_date(
            params=params,
            basemodel_timespan=basemodel.timespan,
            sampled_start_timespan=sampled_start_timespan,
        )
        timespan = Timespan(start_date=start_date, end_date=params["end_date"], delta_t=basemodel.timespan.delta_t)

        # 3. Apply calibrated parameters
        apply_calibrated_parameters(model=model, params=params, parameter_config=basemodel.parameters)

        # 4. Apply vaccination (reaggregating if start_date is sampled)
        apply_vaccination_for_sampled_start(
            model=model,
            basemodel=basemodel,
            timespan=timespan,
            earliest_vax=earliest_vax,
            sampled_start_timespan=sampled_start_timespan,
        )

        # 5. Apply seasonality (this must occur before parameter interventions to preserve parameter overrides)
        apply_seasonality_with_sampled_min(model=model, basemodel=basemodel, timespan=timespan, params=params)

        # 6. Add parameter interventions
        if basemodel.interventions and "parameter" in intervention_types:
            add_parameter_interventions_from_config(
                model=model, interventions=basemodel.interventions, timespan=timespan
            )

        # 7. Calculate compartment initial conditions
        compartment_init = calculate_compartment_initial_conditions(
            compartments=basemodel.compartments,
            population_array=model.population.Nk,
            sampled_compartments=params,
        )

        # 8. Collect settings for simulation
        sim_params = {
            "epimodel": model,
            "initial_conditions_dict": compartment_init,
            "start_date": timespan.start_date,
            "end_date": params["end_date"],
            "resample_frequency": basemodel.simulation.resample_frequency,
        }

        # 9. Extract observed dates for calibration (before simulation to avoid duplication)
        if not params["projection"]:
            date_column = calibration.comparison[0].observed_date_column
            data_dates = list(pd.to_datetime(data_state[date_column].values))

        # 10. Run simulation
        try:
            results = simulate(**sim_params)
        except (ValueError, RuntimeError, KeyError) as e:
            failed_params = params.copy()
            failed_params.pop("epimodel", None)
            logger.warning("Simulation failed with parameters %s: %s", failed_params, e)

            # Handle simulation failure
            # Projection mode: return empty dict
            if params["projection"]:
                return {}

            # Calibration mode: return zero-filled array
            return {"data": np.full(len(data_dates), 0)}

        # 11. Format output based on mode
        # Projection: return full results flattened
        if params["projection"]:
            return flatten_simulation_results(results=results)

        # Calibration: align to observed dates
        aligned_data = align_simulation_to_observed_dates(
            results=results,
            comparison_transitions=calibration.comparison[0].simulation,
            data_dates=data_dates,
        )
        return {"data": aligned_data}

    return simulate_wrapper
