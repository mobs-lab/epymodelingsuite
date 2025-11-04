"""Model orchestration functions for creating collections, setting up schedules and interventions."""

import copy
import datetime as dt
import logging
from collections.abc import Callable
from typing import Any, TypedDict

import numpy as np
import pandas as pd
from epydemix import simulate
from epydemix.model import EpiModel
from numpy.random import Generator

from ..schema.basemodel import BaseEpiModel, Parameter, Timespan
from ..schema.calibration import CalibrationConfig, ComparisonSpec
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


def pad_array_with_zeros(
    array: np.ndarray,
    pad_length: int,
) -> np.ndarray:
    """
    Pad a numpy array with zeros at the beginning.

    Parameters
    ----------
    array : np.ndarray
            Array to pad.
    pad_length : int
            Number of zeros to add at the beginning.

    Returns
    -------
    np.ndarray
            Padded array.
    """
    if pad_length <= 0:
        return array

    return np.pad(array, (pad_length, 0), constant_values=0)


def pad_trajectory_arrays(
    arrays_dict: dict[str, np.ndarray],
    pad_length: int,
) -> dict[str, np.ndarray]:
    """
    Pad all arrays in a dictionary with zeros at the beginning.

    Parameters
    ----------
    arrays_dict : dict[str, np.ndarray]
            Dictionary of arrays to pad.
    pad_length : int
            Number of zeros to add at the beginning of each array.

    Returns
    -------
    dict[str, np.ndarray]
            Dictionary with all arrays padded.
    """
    if pad_length <= 0:
        return arrays_dict

    return {key: pad_array_with_zeros(value, pad_length) for key, value in arrays_dict.items()}


def calculate_padding_for_date_alignment(
    actual_dates: np.ndarray | list,
    target_dates: np.ndarray | list,
) -> int:
    """
    Calculate padding needed to align actual dates to target date grid.

    Determines how many padding steps are needed at the beginning when
    actual dates start later than target dates.

    Parameters
    ----------
    actual_dates : np.ndarray | list
            Dates from simulation results.
    target_dates : np.ndarray | list
            Target date grid to align to (e.g., from reference start or observed data).

    Returns
    -------
    int
            Number of padding steps needed (0 if no padding required).

    Examples
    --------
    >>> actual = [date(2024, 1, 10), date(2024, 1, 11)]
    >>> target = [date(2024, 1, 5), date(2024, 1, 6), ..., date(2024, 1, 11)]
    >>> calculate_padding_for_date_alignment(actual, target)
    5  # Need 5 zeros for dates Jan 5-9
    """
    # Find which target dates are covered by actual dates
    mask = np.isin(target_dates, actual_dates)
    covered_count = mask.sum()

    # Padding needed = total target length - covered dates
    pad_len = len(target_dates) - covered_count

    return max(0, pad_len)  # Ensure non-negative


def flatten_simulation_results(results: Any) -> dict:
    """
    Flatten simulation results structure for projection output.

    Flattens nested results structure into a single dict with dates, transitions,
    and compartments at the top level.

    Parameters
    ----------
    results : SimulationResults
            Output from epydemix.simulate().

    Returns
    -------
    dict
            Flattened results with "date" key and all transition/compartment keys at top level.
    """
    output = {"date": results.dates}
    output.update(results.transitions)
    output.update(results.compartments)
    return output


def get_aggregated_comparison_transition(
    results: Any,
    comparison_transitions: list[str],
) -> np.ndarray:
    """
    Aggregate specified transitions for comparison with observed data.

    Parameters
    ----------
    results : SimulationResults
        Output from epydemix.simulate().
    comparison_transitions : list[str]
        List of transition names to sum (e.g., ["Hosp_vax", "Hosp_unvax"]).

    Returns
    -------
    np.ndarray
        Aggregated transition array (sum of specified transitions).
    """
    transition_arrays = [results.transitions[key] for key in comparison_transitions]
    return sum(transition_arrays)


def format_projection_trajectories(
    results: Any,
    reference_start_date: dt.date | None = None,
    actual_start_date: dt.date | None = None,
    target_end_date: dt.date | None = None,
    resample_frequency: str | None = None,
    comparison_specs: list[ComparisonSpec] | None = None,
) -> dict:
    """
    Format simulation results for projection mode (flatten + pad for stacking).

    Flattens structure and pads trajectories to a consistent target length
    when start_date is sampled. This ensures all trajectories have the same length
    for np.stack() in epydemix's get_projection_trajectories().

    Parameters
    ----------
    results : SimulationResults
            Output from epydemix.simulate().
    reference_start_date : date | None, optional
            The earliest possible start date (used when start_date is sampled).
            If provided, trajectories will be padded to align with this date.
    actual_start_date : date | None, optional
            The actual start date of this specific trajectory.
            Required if reference_start_date is provided.
    target_end_date : date | None, optional
            The target end date for all trajectories (projection end date).
            Used to calculate consistent target length for all trajectories.
    resample_frequency : str | None, optional
            Resampling frequency (e.g., "W-SAT", "D").
            Required if padding is needed to compute the target length.
    comparison_specs : list[ComparisonSpec] | None, optional
            List of comparison specifications for aggregating transitions.
            Each spec defines which transitions to sum and the output column name.

    Returns
    -------
    dict
            Dictionary with "date" and all transition/compartment arrays.
            Arrays are padded with zeros at the beginning to match target length.
            If comparison_specs provided, also includes aggregated transition arrays
            with keys based on observed_value_column (e.g., "total_hosp").
    """
    # Flatten results structure
    output = flatten_simulation_results(results)

    # Add aggregated comparison transitions
    # The key would be based on observed_value_column
    if comparison_specs is not None:
        for spec in comparison_specs:
            aggregated = get_aggregated_comparison_transition(results, spec.simulation)
            output[spec.observed_value_column] = aggregated

    # If no padding needed, return as-is
    if reference_start_date is None or actual_start_date is None or target_end_date is None:
        return output

    # Calculate target length from reference start to target end
    target_date_range = pd.date_range(
        start=reference_start_date,
        end=target_end_date,
        freq=resample_frequency if resample_frequency else "D",
    )
    target_length = len(target_date_range)

    # Calculate padding needed to reach target length
    current_length = len(results.dates)
    pad_len = target_length - current_length

    if pad_len < 0:
        logger.warning(
            "Trajectory length (%d) exceeds target length (%d). "
            "This may indicate inconsistent resampling or end dates.",
            current_length,
            target_length,
        )
        return output

    if pad_len > 0:
        # Pad all arrays except 'date'
        arrays_to_pad = {k: v for k, v in output.items() if k != "date" and isinstance(v, np.ndarray)}
        padded_arrays = pad_trajectory_arrays(arrays_to_pad, pad_len)
        output.update(padded_arrays)

        # Generate exactly pad_len dates to prepend
        # Use periods parameter to ensure we get exactly the right number
        prepend_dates = pd.date_range(
            start=reference_start_date,
            periods=pad_len,
            freq=resample_frequency if resample_frequency else "D",
        )

        # Convert to Timestamp and concatenate with existing dates
        prepend_dates = [pd.Timestamp(d) for d in prepend_dates]
        results_dates_list = list(results.dates)
        output["date"] = prepend_dates + results_dates_list

    return output


def format_calibration_data(
    results: Any,
    comparison_transitions: list[str],
    data_dates: list,
    random_state: dict[str, Any],
) -> dict[str, Any]:
    """
    Format simulation results for calibration mode (aggregate + filter + pad).

    Processes simulation results through three steps:
    1. Aggregates specified transitions (e.g., sum Hosp_vax + Hosp_unvax)
    2. Filters to observed dates only
    3. Pads to align with full observation date grid

    Parameters
    ----------
    results : SimulationResults
            Output from epydemix.simulate() with dates and transitions.
    comparison_transitions : list[str]
            List of transition keys to sum (e.g., ["Hosp_vax", "Hosp_unvax"]).
    data_dates : list
            List of observation dates from observed data.

    Returns
    -------
    dict[str, Any]
            Dictionary containing:
            - "data": np.ndarray of aggregated simulation values aligned to observation dates.
                Padded with zeros at the beginning if simulation starts after first observation.
            - "date": list of observation dates from observed data.
    """
    # Step 1: Aggregate specified transitions
    # Match simulation outputs to observed data granularity.
    # e.g. observed "hospitalizations" = sum(Home_sev_to_Hosp_total + Home_sev_vax_to_Hosp_vax_total)
    aggregated_data = get_aggregated_comparison_transition(results, comparison_transitions)

    # Step 2: Filter to observed dates
    # Simulation may run at different frequency (daily) than observations (weekly).
    # Extract only the dates that match the observed data for comparison.
    mask = np.isin(results.dates, data_dates)
    filtered_data = aggregated_data[mask]

    # Step 3: Pad to align with full observation grid
    # When start_date is sampled, simulation may start later than first observation.
    # Pad with zeros at beginning to align arrays for distance calculation.
    pad_len = calculate_padding_for_date_alignment(results.dates, data_dates)
    aligned_data = pad_array_with_zeros(filtered_data, pad_len)

    return {"data": aligned_data, "date": data_dates, "random_state": random_state}


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
    reference_start_date: dt.date | None,
) -> dt.date:
    """
    Calculate simulation start date based on sampling configuration.

    Handles two cases:
    1. No sampling: use fixed start date from basemodel
    2. Sampling enabled: use sampled offset from reference start

    Parameters
    ----------
    params : dict
            Simulation parameters. If start_date sampling is enabled,
            params["start_date"] should be an integer offset in days.
    basemodel_timespan : Timespan
            Default timespan from base model configuration.
    reference_start_date : date | None
            Reference start date for start_date sampling, or None if not sampled.
            This is the earliest possible start date when sampling is enabled.

    Returns
    -------
    dt.date
            Calculated start date for simulation.
    """
    # Case 1: No sampling - use fixed start date
    if reference_start_date is None:
        return basemodel_timespan.start_date

    # Case 2: Sampling enabled - use sampled offset from reference
    offset_days = params["start_date"]
    return reference_start_date + dt.timedelta(days=offset_days)


class SimulateWrapperParams(TypedDict, total=False):
    """
    Type definition for simulate_wrapper parameters.

    Note: Uses total=False because params may contain additional calibrated
    parameter values (e.g., "beta", "initial_infected") that are determined
    dynamically based on the calibration configuration.
    """

    epimodel: EpiModel
    end_date: dt.date
    projection: bool
    start_date: int  # Offset in days, present if start_date is calibrated
    seasonality_min: float  # Present if seasonality minimum is calibrated


def make_simulate_wrapper(
    basemodel: BaseEpiModel,
    calibration: CalibrationConfig,
    observed_data: pd.DataFrame,
    intervention_types: list[str],
    sampled_start_timespan: Timespan | None = None,
    earliest_vax: pd.DataFrame | None = None,
    rng: Generator | None = None,
) -> Callable[[dict], dict]:
    """
    Create a simulate_wrapper function for ABCSampler calibration.

    Parameters
    ----------
    basemodel : BaseEpiModel
            Base model configuration with compartments, parameters, interventions, etc.
    calibration : CalibrationConfig
            Calibration settings including comparison targets, priors, and fitting window.
    observed_data : pd.DataFrame
            Observed data for calibration. Should contain date column specified in
            calibration.comparison[0].observed_date_column. Must be filtered to a single
            location (no duplicate dates).
    intervention_types : list[str]
            List of intervention types to apply (e.g., ["parameter", "school_closure"]).
    sampled_start_timespan : Timespan | None, optional
            Earliest timespan if start_date is being sampled/calibrated.
            If provided, enables start_date sampling and vaccination reaggregation.
    earliest_vax : pd.DataFrame | None, optional
            Pre-calculated vaccination schedule from earliest start_date.
            Required when sampled_start_timespan is provided and vaccinations are used.
            DataFrame with columns: "dates", "location", and age group columns
            (e.g., "0-4", "5-17", "18-49", "50-64", "65+").
            Typically created by `setup_vaccination_schedules()` which calls
            `scenario_to_epydemix()` with the earliest start date.

    Returns
    -------
    callable
            A simulate_wrapper function that takes params dict and returns results dict.
            This wrapper is passed to ABCSampler and called during calibration/projection.

    """
    # Validate observed_data: check for duplicate dates (indicates mixed location data)
    date_column = calibration.comparison[0].observed_date_column
    observed_dates = pd.to_datetime(observed_data[date_column])
    if observed_dates.duplicated().any():
        duplicated_dates = observed_dates[observed_dates.duplicated()].unique()
        msg = (
            f"Duplicate dates found in observed_data: {duplicated_dates.tolist()}. "
            "This likely indicates data from multiple locations is mixed. "
            "observed_data must be filtered to a single location."
        )
        raise ValueError(msg)

    def simulate_wrapper(params: SimulateWrapperParams) -> dict[str, Any]:
        """
        Run a single simulation with the given parameters.

        This function is called by ABCSampler during calibration and projection.

        Parameters
        ----------
        params : SimulateWrapperParams (dict)
                Simulation parameters dictionary containing:

                - epimodel : EpiModel
                        The model to simulate (passed via fixed_parameters)
                - end_date : date
                        Simulation end date
                - projection : bool
                        Boolean indicating calibration vs projection mode
                - start_date : int, optional
                        Offset in days from reference date (if start_date is calibrated)
                - seasonality_min : float, optional
                        Minimum seasonality value (if seasonality is calibrated)
                - Additional calibrated parameter values (e.g., "beta", "initial_infected")

        Returns
        -------
        dict
                For calibration (projection=False):
                        {"data": np.ndarray} of simulated values at observation times,
                        aligned with observed data dates
                For projection (projection=True):
                        Flattened simulation results with keys:
                        - "date": list of simulation dates
                        - Individual transition keys (e.g., "S_to_I", "I_to_R")
                        - Individual compartment keys (e.g., "S", "I", "R")
                        All keys are at the top level (flattened, not nested)
        """
        # 1. Extract model from params
        wrapper_model = params["epimodel"]
        model = copy.deepcopy(wrapper_model)

        # 2. Calculate start date
        reference_start_date = sampled_start_timespan.start_date if sampled_start_timespan else None
        start_date = compute_simulation_start_date(
            params=params,
            basemodel_timespan=basemodel.timespan,
            reference_start_date=reference_start_date,
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
            params_dict=params,
        )

        # 8 Handle random state
        if "random_state" in params.keys():
            rng.bit_generator.state = params["random_state"]
        random_state = rng.bit_generator.state

        # 9. Collect settings for simulation
        sim_params = {
            "epimodel": model,
            "initial_conditions_dict": compartment_init,
            "start_date": timespan.start_date,
            "end_date": params["end_date"],
            "resample_frequency": basemodel.simulation.resample_frequency,
            "rng": rng,
        }

        # 10. Extract observed dates for calibration (before simulation to avoid duplication)
        if not params["projection"]:
            date_column = calibration.comparison[0].observed_date_column
            data_dates = list(pd.to_datetime(observed_data[date_column].values))

        # 11. Run simulation
        try:
            results = simulate(**sim_params)
        except (ValueError, RuntimeError, KeyError) as e:
            failed_params = params.copy()
            failed_params.pop("epimodel", None)
            logger.warning("Simulation failed with parameters %s: %s", failed_params, e)

            # Handle simulation failure
            # Projection: return empty dict
            if params["projection"]:
                return {}

            # Calibration: return zero-filled array
            return {"data": np.full(len(data_dates), 0)}

        # 12. Format output based on mode
        # Projection: return full trajectories (flattened + padded)
        if params["projection"]:
            return format_projection_trajectories(
                results=results,
                reference_start_date=reference_start_date,
                actual_start_date=start_date,
                target_end_date=params["end_date"],
                resample_frequency=basemodel.simulation.resample_frequency,
                comparison_specs=calibration.comparison,
            )

        # Calibration: return aggregated data (aligned to observed dates)
        return format_calibration_data(
            results=results,
            comparison_transitions=calibration.comparison[0].simulation,
            data_dates=data_dates,
            random_state=random_state,
        )

    return simulate_wrapper
