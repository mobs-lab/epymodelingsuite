import copy
import datetime as dt
import logging
from datetime import date

import numpy as np
import pandas as pd
from epydemix import simulate
from epydemix.calibration import ABCSampler, CalibrationResults, ae, mae, mape, rmse, wmape
from epydemix.model import EpiModel
from epydemix.model.simulation_results import SimulationResults
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config_loader import (
    _add_contact_matrix_interventions_from_config,
    _add_model_compartments_from_config,
    _add_model_parameters_from_config,
    _add_model_transitions_from_config,
    _add_parameter_interventions_from_config,
    _add_school_closure_intervention_from_config,
    _add_seasonality_from_config,
    _add_vaccination_schedules_from_config,
    _calculate_parameters_from_config,
    _set_population_from_config,
)
from .school_closures import make_school_closure_dict
from .utils import convert_location_name_format, get_location_codebook, make_dummy_population
from .vaccinations import reaggregate_vaccines, scenario_to_epydemix
from .validation.basemodel_validator import BaseEpiModel, BasemodelConfig, Parameter, Timespan
from .validation.calibration_validator import CalibrationConfig, CalibrationStrategy
from .validation.general_validator import validate_modelset_consistency
from .validation.output_validator import OutputConfig
from .validation.sampling_validator import SamplingConfig

logger = logging.getLogger(__name__)


# ===== Classes and Helpers =====


class SimulationArguments(BaseModel):
    """
    Arguments for a single call to epydemix.EpiModel.run_simulations
    Follows https://epydemix.readthedocs.io/en/stable/epydemix.model.html#epydemix.model.epimodel.EpiModel.run_simulations
    """

    start_date: date = Field(description="Start date of the simulation.")
    end_date: date = Field(description="End date of the simulation.")
    initial_conditions_dict: dict | None = Field(None, description="Initial conditions dictionary.")
    Nsim: int | None = Field(10, description="Number of simulation runs to perform for a single EpiModel.")
    dt: float | None = Field(1.0, description="Timestep for simulation, defaults to 1.0 = 1 day.")
    resample_frequency: str | None = Field(
        None,
        description="The frequency at which to resample the simulation results. Follows https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#period-aliases",
    )
    # NOTE: percentage_in_agents, resample_aggregation_compartments, resample_aggregation_transitions, and fill_method not used


class ProjectionArguments(BaseModel):
    """Projection arguments."""

    end_date: date = Field(description="End date of the projection.")
    n_trajectories: int = Field(description="Number of trajectories to simulate from posterior after calibration.")
    generation_number: int | None = Field(
        default=None, description="SMC generation number from which to draw parameter sets for projection."
    )


class BuilderOutput(BaseModel):
    """
    A bundle containing a single EpiModel or ABCSampler object paired with instructions for simulation/calibration/projection.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    primary_id: int = Field(
        description="Primary identifier of an EpiModel or ABCSampler object paired with instructions for simulation/calibration/projection."
    )
    seed: int | None = Field(None, description="Random seed.")
    model: EpiModel | None = Field(None, description="EpiModel object for simulation.")
    calibrator: ABCSampler | None = Field(None, description="ABCSampler object for calibration.")
    simulation: SimulationArguments | None = Field(
        None, description="Arguments for a single call to EpiModel.run_simulations"
    )
    calibration: CalibrationStrategy | None = Field(
        None, description="Arguments for a single call to ABCSampler.calibrate"
    )
    projection: ProjectionArguments | None = Field(None, description="Arguments for a single call to ABCSampler.run_projections")

    @model_validator(mode="after")
    def check_fields(self: "BuilderOutput") -> "BuilderOutput":
        """
        Ensure combination of fields is valid.
        """
        assert self.model or self.calibrator, "BuilderOutput must contain an EpiModel or ABCSampler."

        if self.simulation:
            assert not self.calibration and not self.projection, (
                "Simulation workflow cannot be combined with calibration/projection."
            )
            assert self.model, "Simulation workflow requires EpiModel but received only ABCSampler."

        elif self.calibration or self.projection:
            assert self.calibrator, "Calibration/projection workflow requires ABCSampler but received only EpiModel."

        return self


class SimulationOutput(BaseModel):
    """Results of a call to EpiModel.run_simulations() with tracking information."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    primary_id: int = Field(
        description="Primary identifier of an EpiModel object paired with instructions for simulation."
    )
    seed: int | None = Field(None, description="Random seed.")
    population: str = Field(description="Population name (epydemix).")
    results: SimulationResults = Field(description="Results of a call to EpiModel.run_simulations()")


class CalibrationOutput(BaseModel):
    """Results of a call to ABCSampler.calibrate() or ABCSampler.run_projections() with tracking information."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    primary_id: int = Field(
        description="Primary identifier of an ABCSampler object paired with instructions for calibration/projection."
    )
    seed: int | None = Field(None, description="Random seed.")
    population: str = Field(description="Population name (epydemix).")
    results: CalibrationResults = Field(
        description="Results of a call to ABCSampler.calibrate() or ABCSampler.run_projections()"
    )


def calculate_compartment_initial_conditions(
    compartments: list,
    population_array: np.ndarray,
    sampled_compartments: dict | None = None,
) -> dict | None:
    """
    Calculate initial conditions for compartments based on their initialization values.

    This function handles three types of compartment initialization:
    1. Counts (value >= 1): Distributed proportionally across age groups
    2. Proportions (value < 1): Applied directly to population by age group
    3. Default: Remaining population distributed per age group

    Parameters
    ----------
    compartments : list
        List of compartment objects with `id` and `init` attributes.
        `init` can be a numeric value (int/float) or the string "default".
    population_array : np.ndarray
        Population counts by age group obtained from EpiModel (model.population.Nk).
    sampled_compartments : dict | None, optional
        Dictionary of sampled/calibrated compartment values to override base configuration.
        Keys are compartment IDs, values are numeric initial conditions.

    Returns
    -------
    dict | None
        Dictionary mapping compartment IDs to initial condition arrays (age-structured)
        or None if no initial conditions are specified.

    Notes
    -----
    The remaining population for default compartments is calculated per age group
    to preserve exact population counts: remaining = population_array - sum_age_structured
    """
    compartment_init = {}
    compartment_ids = {c.id for c in compartments}

    # Initialize non-sampled compartments with counts
    # Distribute counts proportionally across age groups
    compartment_init.update(
        {
            compartment.id: compartment.init * population_array / sum(population_array)
            for compartment in compartments
            if isinstance(compartment.init, (int, float, np.int64, np.float64)) and compartment.init >= 1
        }
    )

    # Initialize non-sampled compartments with proportions
    compartment_init.update(
        {
            compartment.id: compartment.init * population_array
            for compartment in compartments
            if isinstance(compartment.init, (int, float, np.int64, np.float64)) and compartment.init < 1
        }
    )

    # Initialize sampled/calibrated compartments (if provided)
    if sampled_compartments:
        # Counts: Distribute proportionally
        compartment_init.update(
            {
                k: v * population_array / sum(population_array)
                for k, v in sampled_compartments.items()
                if k in compartment_ids and isinstance(v, (int, float, np.int64, np.float64)) and v >= 1
            }
        )
        # Proportions
        compartment_init.update(
            {
                k: v * population_array
                for k, v in sampled_compartments.items()
                if k in compartment_ids and isinstance(v, (int, float, np.int64, np.float64)) and v < 1
            }
        )

    # Initialize default compartments
    # Calculate remaining population per age group to preserve exact counts
    default_compartments = [compartment for compartment in compartments if compartment.init == "default"]
    if default_compartments:
        sum_age_structured = np.sum(
            [vals for vals in compartment_init.values() if isinstance(vals, np.ndarray)], axis=0
        )
        remaining = population_array - sum_age_structured
        compartment_init.update(
            {compartment.id: remaining / len(default_compartments) for compartment in default_compartments}
        )

    return compartment_init if compartment_init else None


def _create_model_collection(
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
    >>> models, pop_names = _create_model_collection(basemodel, ["California", "Texas"])
    >>> len(models)
    2
    >>> models, pop_names = _create_model_collection(basemodel, ["all"])
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
    _add_model_compartments_from_config(init_model, basemodel.compartments)
    _add_model_transitions_from_config(init_model, basemodel.transitions)
    _add_model_parameters_from_config(init_model, basemodel.parameters)

    # Create models with populations set
    if population_names:
        if "all" in population_names:
            resolved_names = get_location_codebook()["location_name_epydemix"]
        else:
            resolved_names = population_names
        for name in resolved_names:
            m = copy.deepcopy(init_model)
            _set_population_from_config(m, name, basemodel.population.age_groups)
            models.append(m)
    else:
        _set_population_from_config(init_model, basemodel.population.name, basemodel.population.age_groups)
        models.append(init_model)
        resolved_names = [basemodel.population.name]

    return models, resolved_names


def _setup_vaccination_schedules(
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
        _add_vaccination_schedules_from_config(model, basemodel.transitions, basemodel.vaccination, basemodel.timespan)
    return models, None


def _setup_interventions(
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
            _add_school_closure_intervention_from_config(model, basemodel.interventions, closure_dict)

        # Contact matrix
        if "contact_matrix" in intervention_types:
            _add_contact_matrix_interventions_from_config(model, basemodel.interventions)

    return models


def _make_simulate_wrapper(
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
            _add_model_parameters_from_config(m, new_params)
        if "calculated" in [param_args.type.value for _, param_args in basemodel.parameters.items()]:
            _calculate_parameters_from_config(m, basemodel.parameters)

        # Vaccination (if start_date is sampled)
        if basemodel.vaccination and sampled_start_timespan:
            reaggregated_vax = reaggregate_vaccines(earliest_vax, timespan.start_date)
            _add_vaccination_schedules_from_config(
                m, basemodel.transitions, basemodel.vaccination, timespan, use_schedule=reaggregated_vax
            )

        # Seasonality (this must occur before parameter interventions to preserve parameter overrides)
        if basemodel.seasonality:
            if "seasonality_min" in params:
                basemodel.seasonality.min_value = params["seasonality_min"]
            _add_seasonality_from_config(m, basemodel.seasonality, timespan)

        # Parameter interventions
        if basemodel.interventions and "parameter" in intervention_types:
            _add_parameter_interventions_from_config(m, basemodel.interventions, timespan)

        # Initial conditions
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
        if not params["projection"]:
            try:
                results = simulate(**sim_params)
                trajectory_dates = results.dates
                data_dates = list(pd.to_datetime(data_state.target_end_date.values))

                mask = [date in data_dates for date in trajectory_dates]

                total_hosp = sum(results.transitions[key] for key in calibration.comparison[0].simulation)

                total_hosp = total_hosp[mask]

                if len(total_hosp) < len(data_dates):
                    pad_len = len(data_dates) - len(total_hosp)
                    total_hosp = np.pad(total_hosp, (pad_len, 0), constant_values=0)

            except Exception as e:  # noqa: BLE001
                failed_params = params.copy()
                failed_params.pop("epimodel", None)
                logger.info("Simulation failed with parameters %s: %s", failed_params, e)
                data_dates = list(pd.to_datetime(data_state["target_end_date"].values))
                total_hosp = np.full(len(data_dates), 0)

            return {"data": total_hosp}

        # Run projection if params["projection"] is True
        try:
            results = simulate(**sim_params)
        except Exception as e:  # noqa: BLE001
            failed_params = params.copy()
            failed_params.pop("epimodel", None)
            logger.info("Projection failed with parameters %s: %s", failed_params, e)
            return {}
        else:
            # Return results from successful projection
            return {
                "dates": results.dates,
                "transitions": results.transitions,
                "compartments": results.compartments,
            }

    return simulate_wrapper


def _get_data_in_window(data: pd.DataFrame, calibration: CalibrationConfig) -> pd.DataFrame:
    """Get data within a specified time window."""
    window_start = calibration.fitting_window.start_date
    window_end = calibration.fitting_window.end_date

    date_col_name = calibration.comparison[0].observed_date_column
    date_col = pd.to_datetime(data[date_col_name]).dt.date

    mask = (date_col >= window_start) & (date_col <= window_end)
    return data.loc[mask]


def _get_data_in_location(data: pd.DataFrame, model: EpiModel) -> pd.DataFrame:
    """Get data for a specific location."""
    location_iso = convert_location_name_format(model.population.name, "ISO")
    # TODO: geo_value column name should be configurable.
    return data[data["geo_value"] == location_iso]


dist_func_dict = {
    "rmse": rmse,
    "wmape": wmape,
    "ae": ae,
    "mae": mae,
    "mape": mape,
}

# ===== Builders =====


BUILDER_REGISTRY = {}


def register_builder(kind_set):
    """Decorator for builder dispatch."""

    def deco(fn):
        BUILDER_REGISTRY[frozenset(kind_set)] = fn
        return fn

    return deco


@register_builder({"basemodel_config"})
def build_basemodel(*, basemodel_config: BasemodelConfig, **_) -> BuilderOutput:
    """
    Construct an EpiModel and arguments for simulation using a BasemodelConfig parsed from YAML.

    Parameters
    ----------
        basemodel: configuration parsed from YAML

    Returns
    -------
        BuilderOutput containing id, seed, EpiModel, and arguments for simulation.
    """
    logger.info("BUILDER: dispatched for single model.")

    # For compactness
    basemodel = basemodel_config.model

    # build a single EpiModel from basemodel config
    model = EpiModel()

    # Set the model name if provided in the config
    if basemodel.name is not None:
        model.name = basemodel.name

    logger.info("BUILDER: setting up single model...")

    # This workflow uses a single population
    _set_population_from_config(model, basemodel.population.name, basemodel.population.age_groups)

    # Compartments and transitions
    _add_model_compartments_from_config(model, basemodel.compartments)
    _add_model_transitions_from_config(model, basemodel.transitions)

    # Vaccination
    if basemodel.vaccination:
        _add_vaccination_schedules_from_config(model, basemodel.transitions, basemodel.vaccination, basemodel.timespan)

    # Parameters
    _add_model_parameters_from_config(model, basemodel.parameters)
    if "calculated" in [param_args.type.value for param, param_args in (basemodel.parameters).items()]:
        _calculate_parameters_from_config(model, basemodel.parameters)

    # Seasonality (this must occur before interventions to preserve parameter overrides)
    if basemodel.seasonality:
        _add_seasonality_from_config(model, basemodel.seasonality, basemodel.timespan)

    # Interventions
    if basemodel.interventions:
        intervention_types = [i.type for i in basemodel.interventions]

        # School closure
        if "school_closure" in intervention_types:
            closure_dict = make_school_closure_dict(
                range(basemodel.timespan.start_date.year, basemodel.timespan.end_date.year + 1)
            )
            _add_school_closure_intervention_from_config(model, basemodel.interventions, closure_dict)

        # Contact matrix
        if "contact_matrix" in intervention_types:
            _add_contact_matrix_interventions_from_config(model, basemodel.interventions)

        # Parameter
        if "parameter" in intervention_types:
            _add_parameter_interventions_from_config(model, basemodel.interventions, basemodel.timespan)

    # Initial conditions
    compartment_inits = calculate_compartment_initial_conditions(
        compartments=basemodel.compartments,
        population_array=model.population.Nk,
    )

    simulation_args = SimulationArguments(
        start_date=basemodel.timespan.start_date,
        end_date=basemodel.timespan.end_date,
        initial_conditions_dict=compartment_inits,
        Nsim=basemodel.simulation.n_sims,
        dt=basemodel.timespan.delta_t,
        resample_frequency=basemodel.simulation.resample_frequency,
    )

    logger.info("BUILDER: completed for single model.")

    return BuilderOutput(primary_id=0, seed=basemodel.random_seed, model=model, simulation=simulation_args)


@register_builder({"basemodel_config", "sampling_config"})
def build_sampling(
    *, basemodel_config: BasemodelConfig, sampling_config: SamplingConfig, **kwargs
) -> list[BuilderOutput]:
    """
    Construct a set of EpiModels and arguments for simulation using a BasemodelConfig and SamplingConfig parsed from YAMLs.

    Parameters
    ----------
        basemodel: configuration parsed from YAML
        sampling: configuration parsed from YAML

    Returns
    -------
        BuilderOutput containing id, seed, EpiModel, and arguments for simulation.
    """
    from .sample_generator import generate_samples

    logger.info("BUILDER: dispatched for sampling.")

    # Validate references between basemodel and sampling
    validate_modelset_consistency(basemodel_config, sampling_config)

    # For compactness
    basemodel = basemodel_config.model
    sampling = sampling_config.modelset

    # Build a collection of EpiModels
    models, population_names = _create_model_collection(basemodel, sampling.population_names)

    # Output of this is a list of dicts containing start_date, initial conditions, and parameter value
    # combinations where parameters is in the same format as basemodel.parameters
    sampled_vars = generate_samples(sampling_config, basemodel.random_seed)

    # Extract intervention types
    if basemodel.interventions:
        intervention_types = [i.type for i in basemodel.interventions]

    # If start_date is sampled, find earliest instance
    try:
        sampled_start_timespan = Timespan(
            start_date=sorted([varset["start_date"] for varset in sampled_vars])[0],
            end_date=basemodel.timespan.end_date,
            delta_t=basemodel.timespan.delta_t,
        )
    except KeyError:  # case where start_date is not sampled
        sampled_start_timespan = None

    # Vaccination is sensitive to location and start_date but not to model parameters.
    models, earliest_vax = _setup_vaccination_schedules(basemodel, models, sampled_start_timespan, population_names)

    # These interventions are sensitive to location but not to model parameters and can be applied
    # using the earliest start_date before further duplicating the models.
    models = _setup_interventions(models, basemodel, intervention_types, sampled_start_timespan)

    logger.info("BUILDER: using sampled values to modify EpiModels")

    # Create models with sampled/calculated parameters, apply vaccination and interventions
    simulation_args = []
    final_models = []
    for model in models:
        for varset in sampled_vars:
            m = copy.deepcopy(model)

            # Accomodate for sampled start_date
            start_date = varset.setdefault("start_date", basemodel.timespan.start_date)
            timespan = Timespan(
                start_date=start_date,
                end_date=basemodel.timespan.end_date,
                delta_t=basemodel.timespan.delta_t,
            )

            # Sampled/calculated parameters
            if "parameters" in varset.keys():
                parameters = {k: Parameter(type="scalar", value=v) for k, v in varset["parameters"].items()}
                _add_model_parameters_from_config(m, parameters)
            if "calculated" in [param_args.type.value for param, param_args in (basemodel.parameters).items()]:
                _calculate_parameters_from_config(m, basemodel.parameters)

            # Vaccination (if start_date is sampled)
            if basemodel.vaccination and sampled_start_timespan:
                reaggregated_vax = reaggregate_vaccines(earliest_vax, timespan.start_date)
                _add_vaccination_schedules_from_config(
                    m, basemodel.transitions, basemodel.vaccination, timespan, use_schedule=reaggregated_vax
                )

            # Seasonality (this must occur before parameter interventions to preserve parameter overrides)
            if basemodel.seasonality:
                _add_seasonality_from_config(m, basemodel.seasonality, timespan)

            # Parameter interventions
            if basemodel.interventions and "parameter" in intervention_types:
                _add_parameter_interventions_from_config(m, basemodel.interventions, timespan)

            # Initial conditions
            compartment_init = calculate_compartment_initial_conditions(
                compartments=basemodel.compartments,
                population_array=m.population.Nk,
                sampled_compartments=varset.get("compartments"),
            )

            sim_args = SimulationArguments(
                start_date=timespan.start_date,
                end_date=timespan.end_date,
                initial_conditions_dict=compartment_init,
                Nsim=basemodel.simulation.n_sims,
                dt=basemodel.timespan.delta_t,
                resample_frequency=basemodel.simulation.resample_frequency,
            )

            final_models.append(m)
            simulation_args.append(sim_args)

    # Ensure models and specifications align
    assert len(final_models) == len(simulation_args), (
        f"Mismatch: created {len(final_models)} EpiModels and {len(simulation_args)} simulation specifications."
    )

    logger.info("BUILDER: completed for sampling.")
    return [
        BuilderOutput(primary_id=i, seed=basemodel.random_seed, model=t[0], simulation=t[1])
        for i, t in enumerate(zip(final_models, simulation_args, strict=True))
    ]


@register_builder({"basemodel_config", "calibration_config"})
def build_calibration(
    *, basemodel_config: BasemodelConfig, calibration_config: CalibrationConfig, **_
) -> list[BuilderOutput]:
    """
    Construct a set of ABCSamplers and arguments for calibration/projection using a BasemodelConfig and CalibrationConfig parsed from YAML.

    Parameters
    ----------
        basemodel: configuration parsed from YAML
        calibration: configuration parsed from YAML

    Returns
    -------
        BuilderOutput containing id, seed, ABCSampler, and arguments for calibration and projection.
    """
    from .utils import distribution_to_scipy

    logger.info("BUILDER: dispatched for calibration.")

    # Validate references between basemodel and calibration
    validate_modelset_consistency(basemodel_config, calibration_config)

    # For compactness
    basemodel = basemodel_config.model
    modelset = calibration_config.modelset
    calibration = modelset.calibration

    # Build a collection of EpiModels
    models, population_names = _create_model_collection(basemodel, modelset.population_names)

    # Extract intervention types
    if basemodel.interventions:
        intervention_types = [i.type for i in basemodel.interventions]

    # If start_date is sampled, make earliest timespan
    if calibration.start_date:
        sampled_start_timespan = Timespan(
            start_date=calibration.start_date.reference_date,
            end_date=basemodel.timespan.end_date,
            delta_t=basemodel.timespan.delta_t,
        )
    else:  # case where start_date is not sampled
        sampled_start_timespan = None

    # Vaccination is sensitive to location and start_date but not to model parameters.
    models, earliest_vax = _setup_vaccination_schedules(basemodel, models, sampled_start_timespan, population_names)

    # These interventions are sensitive to location but not to model parameters and can be applied
    # using the earliest start_date before creating ABCSamplers.
    models = _setup_interventions(models, basemodel, intervention_types, sampled_start_timespan)

    logger.info("BUILDER: setting up ABCSamplers...")

    observed = pd.read_csv(calibration.observed_data_path)
    data_in_window = _get_data_in_window(observed, calibration)
    calibrators = []
    for model in models:
        data_state = _get_data_in_location(data_in_window, model)

        # Create simulate_wrapper
        simulate_wrapper = _make_simulate_wrapper(
            basemodel=basemodel,
            calibration=calibration,
            data_state=data_state,
            intervention_types=intervention_types,
            sampled_start_timespan=sampled_start_timespan,
            earliest_vax=earliest_vax,
        )

        # Parse priors into scipy functions
        priors = {}
        priors.update({k: distribution_to_scipy(v.prior) for k, v in calibration.parameters.items()})
        priors.update({k: distribution_to_scipy(v.prior) for k, v in calibration.compartments.items()})
        if sampled_start_timespan:
            priors["start_date"] = distribution_to_scipy(calibration.start_date.prior)

        fixed_parameters = {k: v for k, v in model.parameters.items() if v is not None}
        fixed_parameters.update(
            {"end_date": calibration.fitting_window.end_date, "projection": False, "epimodel": model}
        )

        # ABCSamplers are the main outputs
        abc_sampler = ABCSampler(
            simulation_function=simulate_wrapper,
            priors=priors,
            parameters=fixed_parameters,
            observed_data=data_state[calibration.comparison[0].observed_value_column].values,
            distance_function=dist_func_dict[calibration.distance_function],
        )

        calibrators.append(abc_sampler)

    if calibration.projection is not None:
        proj_options_dict = {
            "end_date": basemodel.timespan.end_date,
            "n_trajectories": calibration.projection.n_trajectories,
        }
        if calibration.projection.generation_number is not None:
            proj_options_dict["generation_number"] = calibration.projection.generation_number

        projection_options = ProjectionArguments(**proj_options_dict)
    else:
        projection_options = None

    # Ensure models and specifications align
    assert len(models) == len(calibrators), (
        f"Mismatch: created {len(models)} EpiModels and {len(calibrators)} ABCSamplers."
    )

    logger.info("BUILDER: completed calibration.")
    return [
        BuilderOutput(
            primary_id=i,
            seed=basemodel.random_seed,
            model=t[0],
            calibrator=t[1],
            calibration=calibration.strategy,
            projection=projection_options,
        )
        for i, t in enumerate(zip(models, calibrators, strict=True))
    ]


def dispatch_builder(**configs) -> BuilderOutput | list[BuilderOutput]:
    """
    Dispatch builder functions using the supplied configs parsed from YAML.

    Dispatch to build_basemodel if supplied configs: BasemodelConfig
    Dispatch to build_sampling if supplied configs: BasemodelConfig, SamplingConfig
    Dispatch to build_calibration if supplied configs: BasemodelConfig, CalibrationConfig
    """
    kinds = frozenset(k for k, v in configs.items() if v is not None)
    return BUILDER_REGISTRY[kinds](**configs)


# ===== Runner =====
# This just calls the epydemix methods and passes on the results, for ease of writing workflows on gcloud


def dispatch_runner(configs: BuilderOutput) -> SimulationOutput | CalibrationOutput:
    """
    Dispatch simulation/calibration/projection using a BuilderOutput and return the results.

    Parameters
    ----------
        configs: a single BuilderOutput created by dispatch_builder()

    Returns
    -------
        An object containing metadata and results of simulation/calibration/projection.

    Raises
    ------
        RuntimeError if simulation/calibration/projection fails.
        AssertionError if configs are invalid.
    """
    np.random.seed(configs.seed)

    # Handle simulation
    if configs.simulation:
        logger.info("RUNNER: dispatched for simulation.")
        try:
            results = configs.model.run_simulations(**dict(configs.simulation))
            logger.info("RUNNER: completed simulation.")
            return SimulationOutput(primary_id=configs.primary_id, seed=configs.seed, population=configs.model.population.name, results=results)
        except Exception as e:
            raise RuntimeError(f"Error during simulation: {e}")

    # Handle calibration
    elif configs.calibration and not configs.projection:
        logger.info("RUNNER: dispatched for calibration.")
        try:
            results = configs.calibrator.calibrate(strategy=configs.calibration.name, **configs.calibration.options)
            logger.info("RUNNER: completed calibration.")
            return CalibrationOutput(primary_id=configs.primary_id, seed=configs.seed, population=configs.model.population.name, results=results)
        except Exception as e:
            raise RuntimeError(f"Error during calibration: {e}")

    # Handle calibration and projection
    elif configs.calibration and configs.projection:
        logger.info("RUNNER: dispatched for calibration and projection.")
        try:
            calibration_results = configs.calibrator.calibrate(
                strategy=configs.calibration.name, **configs.calibration.options
            )
            projection_results = configs.calibrator.run_projections(
                parameters={
                    "projection": True,
                    "end_date": configs.projection.end_date,
                    "generation": configs.projection.generation_number,
                    "epimodel": configs.model,
                },
                iterations=configs.projection.n_trajectories,
            )
            logger.info("RUNNER: completed calibration.")
            return CalibrationOutput(primary_id=configs.primary_id, seed=configs.seed, population=configs.model.population.name, results=projection_results)
        except Exception as e:
            raise RuntimeError(f"Error during calibration/projection: {e}")
    # Error
    else:
        raise AssertionError(
            "Runner called without simulation or calibration specs. Verify that your BuilderOutputs are valid."
        )


# ===== Output Generators =====
# these will convert to modeling hub format, create rate trend forecasts, create metadata files, create files with trajectories, create files with posteriors, and optionally save CalibrationResults/SimulationResults directly (pickle?)


OUTPUT_GENERATOR_REGISTRY = {}


def register_output_generator(kind_set):
    """Decorator for output generation dispatch."""

    def deco(fn):
        OUTPUT_GENERATOR_REGISTRY[frozenset(kind_set)] = fn
        return fn

    return deco


@register_output_generator({"simulation", "outputs"})
def generate_simulation_outputs(*, simulation: list[SimulationOutput], outputs: OutputConfig, **_) -> dict:
    """"""
    
    logger.info("OUTPUT GENERATOR: dispatched for simulation")

    quantiles = pd.DataFrame()
    trajectories = pd.DataFrame()
    
    for model in simulation:

        # Quantiles
        if outputs.quantiles:
            quan_df = model.results.get_quantiles_compartments(outputs.quantiles.selections)
            quan_df.insert(0, "primary_id", model.primary_id)
            quan_df.insert(1, "seed", model.seed)
            quan_df.insert(2, "population", model.population)
            quantiles = pd.concat([quantiles, quan_df])

        # Trajectories
        if outputs.trajectories:
            for i, traj in enumerate(model.results.trajectories):
                traj_df = pd.DataFrame(traj.compartments)
                traj_df.insert(0, "primary_id", model.primary_id)
                traj_df.insert(1, "sim_id", i)
                traj_df.insert(2, "seed", model.seed)
                traj_df.insert(3, "population", model.population)
                trajectories = pd.concat([trajectories, traj_df])
            
    return {"quantiles.csv.gz": quantiles, "trajectories.csv.gz": trajectories}
        


@register_output_generator({"calibration", "outputs"})
def generate_calibration_outputs(*, calibration: list[CalibrationOutput], outputs: OutputConfig, **_) -> None:
    """"""
    logger.info("OUTPUT GENERATOR: dispatched for calibration")


def dispatch_output_generator(**configs) -> None:
    """Dispatch output generator functions. Write outputs to file, called for effect."""
    kinds = frozenset(k for k, v in configs.items() if v is not None)
    return OUTPUT_GENERATOR_REGISTRY[kinds](**configs)
