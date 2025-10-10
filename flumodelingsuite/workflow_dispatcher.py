import copy
import datetime as dt
import logging
from typing import NamedTuple

import numpy as np
import pandas as pd
from epydemix import simulate
from epydemix.calibration import ABCSampler, CalibrationResults, ae, mae, mape, rmse, wmape
from epydemix.model import EpiModel
from epydemix.model.simulation_results import SimulationResults
from numpy import float64, int64, ndarray
from numpy.random import seed

from .basemodel_validator import BasemodelConfig, Parameter, Timespan
from .calibration_validator import CalibrationConfig, CalibrationStrategy
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
from .general_validator import validate_modelset_consistency
from .sampling_validator import SamplingConfig
from .school_closures import make_school_closure_dict
from .utils import convert_location_name_format, get_location_codebook, make_dummy_population
from .vaccinations import reaggregate_vaccines, scenario_to_epydemix

logger = logging.getLogger(__name__)


# ===== Classes and Helpers =====


class SimulationArguments(NamedTuple):
    """Typed namedtuple for simulation arguments."""

    start_date: int
    end_date: int
    initial_conditions_dict: dict | None = None
    Nsim: int | None = None
    dt: float | None = 1.0
    resample_frequency: str | None = None


class ProjectionArguments(NamedTuple):
    """Typed namedtuple for projection arguments."""


class BuilderOutput(NamedTuple):
    """Typed namedtuple for builder outputs."""

    primary_id: int
    seed: int | None = None
    model: EpiModel | None = None
    calibrator: ABCSampler | None = None
    simulation: SimulationArguments | None = None
    calibration: CalibrationStrategy | None = None
    projection: ProjectionArguments | None = None


class SimulationOutput(NamedTuple):
    """Typed namedtuple for simulation outputs."""

    primary_id: int
    results: SimulationResults
    seed: int | None = None


# Typed namedtuple for calibration arguments
class CalibrationArguments(NamedTuple):
    pass


class CalibrationOutput(NamedTuple):
    """Typed namedtuple for calibration outputs."""

    primary_id: int
    results: CalibrationResults
    seed: int | None = None


def get_data_in_window(data: pd.DataFrame, calibration: CalibrationConfig) -> pd.DataFrame:
    """Get data within a specified time window"""
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
    def deco(fn):
        BUILDER_REGISTRY[frozenset(kind_set)] = fn
        return fn

    return deco


@register_builder({"basemodel"})
def build_basemodel(*, basemodel: BasemodelConfig, **_) -> BuilderOutput:
    """
    Workflow using only a basemodel.
    """
    logger.info("BUILDER: dispatched for single model.")

    # For compactness
    basemodel = basemodel.model

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
    if "calculated" in [p.type for p in basemodel.parameters]:
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
                range(start=basemodel.timespan.start_date.year, stop=basemodel.timespan.end_date.year + 1)
            )
            _add_school_closure_intervention_from_config(model, basemodel.interventions, closure_dict)

        # Contact matrix
        if "contact_matrix" in intervention_types:
            _add_contact_matrix_interventions_from_config(model, basemodel.interventions)

        # Parameter
        if "parameter" in intervention_types:
            _add_parameter_interventions_from_config(model, basemodel.interventions, basemodel.timespan)

    # Initial conditions
    compartment_inits = {}
    # Initialize compartments with counts
    compartment_inits.update(
        {
            compartment.id: compartment.init
            for compartment in basemodel.compartments
            if isinstance(compartment.init, (int, float, int64, float64)) and compartment.init >= 1
        }
    )
    # Initialize compartments with proportions
    compartment_inits.update(
        {
            compartment.id: compartment.init * model.population.Nk
            for compartment in basemodel.compartments
            if isinstance(compartment.init, (int, float, int64, float64)) and compartment.init < 1
        }
    )
    # Initialize default compartments
    default_compartments = [compartment for compartment in basemodel.compartments if compartment.init == "default"]
    sum_age_structured = sum([sum(vals) for vals in compartment_inits.values() if isinstance(vals, ndarray)])
    sum_no_age = sum([val for val in compartment_inits.values() if isinstance(val, (int, float, int64, float64))])
    remaining = sum(model.population.Nk) - sum_age_structured - sum_no_age
    compartment_inits.update(
        {compartment.id: remaining / len(default_compartments) for compartment in default_compartments}
    )

    if not compartment_inits:
        compartment_inits = None

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


@register_builder({"basemodel", "sampling"})
def build_sampling(*, basemodel: BasemodelConfig, sampling: SamplingConfig, **_) -> list[BuilderOutput]:
    """
    Sampling workflow.
    """
    from .sample_generator import generate_samples

    logger.info("BUILDER: dispatched for sampling.")

    # Validate references between basemodel and sampling
    validate_modelset_consistency(basemodel, sampling)

    # For compactness
    basemodel = basemodel.model
    sampling = sampling.modelset

    # Build a collection of EpiModels
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
    if sampling.population_names:
        if "all" in sampling.population_names:
            population_names = get_location_codebook()["location_name_epydemix"]
        else:
            population_names = sampling.population_names
        for name in population_names:
            m = copy.deepcopy(init_model)
            _set_population_from_config(m, name, basemodel.population.age_groups)
            models.append(m)
    else:
        _set_population_from_config(init_model, basemodel.population.name, basemodel.population.age_groups)
        models.append(init_model)

    # Output of this is a list of dicts containing start_date, initial conditions, and parameter value
    # combinations where parameters is in the same format as basemodel.parameters
    sampled_vars = generate_samples(sampling.sampling, basemodel.random_seed)

    # Extract intervention types
    if basemodel.interventions:
        intervention_types = [i.type for i in basemodel.interventions]

    # If start_date is sampled, find earliest instance
    try:
        earliest_timespan = Timespan(
            {
                "start_date": sorted([varset["start_date"] for varset in sampled_vars])[0],
                "end_date": basemodel.timespan.end_date,
                "delta_t": basemodel.timespan.delta_t,
            }
        )
    except KeyError:  # case where start_date is not sampled
        earliest_timespan = False

    # Vaccination is sensitive to location and start_date but not to model parameters.
    if basemodel.vaccination:
        # If start_date is sampled, precalculate schedule with earliest start for later reaggregation
        if earliest_timespan:
            if sampling.population_names:
                states = population_names
            else:
                states = [basemodel.population.name]
            earliest_vax = scenario_to_epydemix(
                input_filepath=basemodel.vaccination.scenario_data_path,
                start_date=earliest_timespan.start_date,
                end_date=earliest_timespan.end_date,
                target_age_groups=basemodel.population.age_groups,
                delta_t=earliest_timespan.delta_t,
                states=states,
            )

        # If start_date not sampled, add vaccination to models now
        else:
            for model in models:
                _add_vaccination_schedules_from_config(
                    model, basemodel.transitions, basemodel.vaccination, basemodel.timespan
                )

    # These interventions are sensitive to location but not to model parameters and can be applied
    # using the earliest start_date before further duplicating the models.
    if basemodel.interventions:
        for model in models:
            # School closure
            if "school_closure" in intervention_types:
                # If start_date is sampled, we can just use the earliest start date
                if earliest_timespan:
                    closure_dict = make_school_closure_dict(
                        range(start=earliest_timespan.start_date.year, stop=earliest_timespan.end_date.year + 1)
                    )
                else:
                    closure_dict = make_school_closure_dict(
                        range(
                            start=basemodel.timespan.start_date.year,
                            stop=basemodel.timespan.end_date.year + 1,
                        )
                    )
                _add_school_closure_intervention_from_config(model, basemodel.interventions, closure_dict)

            # Contact matrix
            if "contact_matrix" in intervention_types:
                _add_contact_matrix_interventions_from_config(model, basemodel.interventions)

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
                {
                    "start_date": start_date,
                    "end_date": basemodel.timespan.end_date,
                    "delta_t": basemodel.timespan.delta_t,
                }
            )

            # Sampled/calculated parameters
            if "parameters" in varset.keys():
                parameters = {k: Parameter({"type": "scalar", "value": v}) for k, v in varset["parameters"].items()}
                _add_model_parameters_from_config(m, parameters)
            if "calculated" in [p.type for p in basemodel.parameters]:
                _calculate_parameters_from_config(m, basemodel.parameters)

            # Vaccination (if start_date is sampled)
            if basemodel.vaccination and earliest_timespan:
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
            compartment_init = {}
            # Initialize non-sampled compartments with counts
            compartment_init.update(
                {
                    compartment.id: compartment.init
                    for compartment in basemodel.compartments
                    if isinstance(compartment.init, (int, float, int64, float64)) and compartment.init >= 1
                }
            )
            # Initialize non-sampled compartments with proportions
            compartment_init.update(
                {
                    compartment.id: compartment.init * m.population.Nk
                    for compartment in basemodel.compartments
                    if isinstance(compartment.init, (int, float, int64, float64)) and compartment.init < 1
                }
            )
            # Initialize sampled compartments
            if varset.get("compartments"):
                # Counts
                compartment_init.update(
                    {
                        k: v
                        for k, v in varset["compartments"].items()
                        if isinstance(v, (int, float, int64, float64)) and v >= 1
                    }
                )
                # Proportions
                compartment_init.update(
                    {
                        k: v * m.population.Nk
                        for k, v in varset["compartments"].items()
                        if isinstance(v, (int, float, int64, float64)) and v < 1
                    }
                )
            # Initialize default compartments
            default_compartments = [
                compartment for compartment in basemodel.compartments if compartment.init == "default"
            ]
            sum_age_structured = sum([sum(vals) for vals in compartment_init.values() if isinstance(vals, ndarray)])
            sum_no_age = sum(
                [val for val in compartment_init.values() if isinstance(val, (int, float, int64, float64))]
            )
            remaining = sum(m.population.Nk) - sum_age_structured - sum_no_age
            compartment_init.update(
                {compartment.id: remaining / len(default_compartments) for compartment in default_compartments}
            )

            if not compartment_init:
                compartment_init = None

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

    # Ensure models and inits align
    assert len(final_models) == len(simulation_args), (
        f"Mismatch: created {len(final_models)} models and {len(simulation_args)} simulation specifications."
    )

    logger.info("BUILDER: completed for sampling.")
    return [
        BuilderOutput(primary_id=i, seed=basemodel.random_seed, model=t[0], simulation=t[1])
        for i, t in enumerate(zip(final_models, simulation_args, strict=True))
    ]


@register_builder({"basemodel", "calibration"})
def build_calibration(*, basemodel: BasemodelConfig, calibration: CalibrationConfig, **_) -> list[BuilderOutput]:
    """
    Calibration workflow.
    """
    from .utils import distribution_to_scipy

    logger.info("BUILDER: dispatched for calibration.")

    # Validate references between basemodel and calibration
    validate_modelset_consistency(basemodel, calibration)

    # For compactness
    basemodel = basemodel.model
    modelset = calibration.modelset
    calibration = modelset.calibration

    # Build a collection of EpiModels
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
    if modelset.population_names:
        if "all" in modelset.population_names:
            population_names = get_location_codebook()["location_name_epydemix"]
        else:
            population_names = modelset.population_names
        for name in population_names:
            m = copy.deepcopy(init_model)
            _set_population_from_config(m, name, basemodel.population.age_groups)
            models.append(m)
    else:
        _set_population_from_config(init_model, basemodel.population.name, basemodel.population.age_groups)
        models.append(init_model)

    # Extract intervention types
    if basemodel.interventions:
        intervention_types = [i.type for i in basemodel.interventions]

    # If start_date is sampled, make earliest timespan
    if calibration.start_date:
        earliest_timespan = Timespan(
            start_date=calibration.start_date.reference_date,
            end_date=basemodel.timespan.end_date,
            delta_t=basemodel.timespan.delta_t,
        )
    else:  # case where start_date is not sampled
        earliest_timespan = False

    # Vaccination is sensitive to location and start_date but not to model parameters.
    if basemodel.vaccination:
        # If start_date is sampled, precalculate schedule with earliest start for later reaggregation
        if earliest_timespan:
            if modelset.population_names:
                states = population_names
            else:
                states = [basemodel.population.name]
            earliest_vax = scenario_to_epydemix(
                input_filepath=basemodel.vaccination.scenario_data_path,
                start_date=earliest_timespan.start_date,
                end_date=earliest_timespan.end_date,
                target_age_groups=basemodel.population.age_groups,
                delta_t=earliest_timespan.delta_t,
                states=states,
            )

        # If start_date not sampled, add vaccination to models now
        else:
            for model in models:
                _add_vaccination_schedules_from_config(
                    model, basemodel.transitions, basemodel.vaccination, basemodel.timespan
                )

    # These interventions are sensitive to location but not to model parameters and can be applied
    # using the earliest start_date before creating ABCSamplers.
    if basemodel.interventions:
        for model in models:
            # School closure
            if "school_closure" in intervention_types:
                # If start_date is sampled, we can just use the earliest start date
                if earliest_timespan:
                    closure_dict = make_school_closure_dict(
                        range(earliest_timespan.start_date.year, earliest_timespan.end_date.year + 1)
                    )
                else:
                    closure_dict = make_school_closure_dict(
                        range(
                            basemodel.timespan.start_date.year,
                            basemodel.timespan.end_date.year + 1,
                        )
                    )
                _add_school_closure_intervention_from_config(model, basemodel.interventions, closure_dict)

            # Contact matrix
            if "contact_matrix" in intervention_types:
                _add_contact_matrix_interventions_from_config(model, basemodel.interventions)

    logger.info("BUILDER: setting up ABCSamplers...")

    observed = pd.read_csv(calibration.observed_data_path)
    data_in_window = get_data_in_window(observed, calibration)
    calibrators = []
    for model in models:
        data_state = _get_data_in_location(data_in_window, model)

        # Create simulate wrapper
        def simulate_wrapper(params):
            seed(basemodel.random_seed)
            m = copy.deepcopy(model)

            # Accomodate for sampled start_date
            if earliest_timespan:
                start_date = earliest_timespan.start_date + dt.timedelta(days=params["start_date"])
            else:
                start_date = basemodel.timespan.start_date
            timespan = Timespan(
                start_date=start_date, end_date=basemodel.timespan.end_date, delta_t=basemodel.timespan.delta_t
            )

            # Sampled/calculated parameters
            new_params = {
                k: Parameter(type="scalar", value=v)
                for k, v in params.items()
                if k in basemodel.parameters.keys() and basemodel.parameters[k].type == "calibrated"
            }
            if new_params:
                _add_model_parameters_from_config(m, new_params)
            if "calculated" in [p.type for _, p in basemodel.parameters.items()]:
                _calculate_parameters_from_config(m, basemodel.parameters)

            # Vaccination (if start_date is sampled)
            if basemodel.vaccination and earliest_timespan:
                reaggregated_vax = reaggregate_vaccines(earliest_vax, timespan.start_date)
                _add_vaccination_schedules_from_config(
                    m, basemodel.transitions, basemodel.vaccination, timespan, use_schedule=reaggregated_vax
                )

            # Seasonality (this must occur before parameter interventions to preserve parameter overrides)
            if basemodel.seasonality:
                if "seasonality_min" in params.keys():
                    basemodel.seasonality.min_value = params["seasonality_min"]
                _add_seasonality_from_config(m, basemodel.seasonality, timespan)

            # Parameter interventions
            if basemodel.interventions and "parameter" in intervention_types:
                _add_parameter_interventions_from_config(m, basemodel.interventions, timespan)

            # Initial conditions
            compartment_init = {}
            # Initialize non-calibrated compartments with counts
            compartment_init.update(
                {
                    compartment.id: compartment.init
                    for compartment in basemodel.compartments
                    if isinstance(compartment.init, (int, float, int64, float64)) and compartment.init >= 1
                }
            )
            # Initialize non-calibrated compartments with proportions
            compartment_init.update(
                {
                    compartment.id: compartment.init * m.population.Nk
                    for compartment in basemodel.compartments
                    if isinstance(compartment.init, (int, float, int64, float64)) and compartment.init < 1
                }
            )
            # Initialize calibrated compartments
            compartment_ids = {c.id for c in basemodel.compartments}
            # Initialize calibrated compartments with proportions
            compartment_init.update(
                {compartment: v for compartment, v in params.items() if compartment in compartment_ids and v >= 1}
            )

            # Initialize calibrated compartments with proportions
            compartment_init.update(
                {
                    compartment: v * m.population.Nk
                    for compartment, v in params.items()
                    if compartment in compartment_ids and v < 1
                }
            )
            # Initialize default compartments
            default_compartments = [
                compartment for compartment in basemodel.compartments if compartment.init == "default"
            ]
            sum_age_structured = np.sum(
                [vals for vals in compartment_init.values() if isinstance(vals, ndarray)], axis=0
            )
            remaining = m.population.Nk - sum_age_structured
            compartment_init.update(
                {compartment.id: remaining / len(default_compartments) for compartment in default_compartments}
            )

            if not compartment_init:
                compartment_init = None

            # Collect settings
            sim_params = {
                "epimodel": m,
                "initial_conditions_dict": compartment_init,
                "start_date": timespan.start_date,
                "end_date": timespan.end_date,
                "resample_frequency": basemodel.simulation.resample_frequency,
            }

            # Run simulation
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

            except Exception as e:
                logger.info(f"Simulation failed with parameters {params}: {e}")
                data_dates = list(pd.to_datetime(data_state["target_end_date"].values))
                total_hosp = np.full(len(data_dates), 0)

            return {"data": total_hosp}

        # Parse priors into scipy functions
        priors = {}
        priors.update({k: distribution_to_scipy(v.prior) for k, v in calibration.parameters.items()})
        priors.update({k: distribution_to_scipy(v.prior) for k, v in calibration.compartments.items()})
        if earliest_timespan:
            priors["start_date"] = distribution_to_scipy(calibration.start_date.prior)

        # ABCSamplers are the main outputs
        abc_sampler = ABCSampler(
            simulation_function=simulate_wrapper,
            priors=priors,
            parameters={k: v for k, v in model.parameters.items() if v is not None},
            observed_data=data_state[calibration.comparison[0].observed_value_column].values,
            distance_function=dist_func_dict[calibration.distance_function],
        )

        calibrators.append(abc_sampler)

    logger.info("BUILDER: completed calibration.")
    return [
        BuilderOutput(primary_id=i, seed=basemodel.random_seed, calibrator=c, calibration=calibration.strategy)
        for i, c in enumerate(calibrators)
    ]


def dispatch_builder(**configs) -> BuilderOutput | list[BuilderOutput]:
    """"""
    kinds = frozenset(k for k, v in configs.items() if v is not None)
    return BUILDER_REGISTRY[kinds](**configs)


# ===== Runners =====
# This just calls the epydemix methods and passes on the results, for ease of writing workflows on gcloud


def dispatch_runner(configs: BuilderOutput) -> SimulationOutput | CalibrationOutput:
    """"""
    seed(configs.seed)

    # Validate configs
    assert configs.model or configs.calibrator, "Runner dispatched without an EpiModel or ABCSampler (requires one)."

    # Handle simulation
    if configs.simulation:
        logger.info("RUNNER: dispatched for simulation.")

        # Validate configs
        assert not configs.calibration and not configs.projection, (
            "Simulation cannot be performed with calibration/projection."
        )
        assert not configs.calibrator, "Simulation requires EpiModel but received ABCSampler."

        # Run simulations
        try:
            results = configs.model.run_simulations(*configs.simulation)
            logger.info("RUNNER: completed simulation.")
            return SimulationOutput(primary_id=configs.primary_id, results=results, seed=configs.seed)
        except Exception as e:
            raise RuntimeError(f"Error during simulation: {e}")

    # Handle calibration
    elif configs.calibration and not configs.projection:
        logger.info("RUNNER: dispatched for calibration.")

        # Validate configs
        assert not configs.model, "Calibration requires ABCSampler but received EpiModel."

        # Run calibration
        try:
            results = configs.calibrator.calibrate(strategy=configs.calibration.name, **configs.calibration.options)
            logger.info("RUNNER: completed calibration.")
            return CalibrationOutput(primary_id=configs.primary_id, results=results, seed=configs.seed)
        except Exception as e:
            raise ValueError(f"Error during calibration: {e}")

    # Handle calibration and projection
    elif configs.calibration and configs.projection:
        logger.info("RUNNER: dispatched for calibration and projection.")

        # Validate configs
        assert not configs.model, "Calibration requires ABCSampler but received EpiModel."

        # Run calibration and projection
        try:
            pass
        except Exception as e:
            raise ValueError(f"Error during calibration/projection: {e}")
    # Error
    else:
        raise ValueError(
            "Runner called without simulation or calibration specs. Verify that your BuilderOutputs are valid."
        )


# ===== Output Generators =====
# these will convert to modeling hub format, create rate trend forecasts, create metadata files, create files with trajectories, create files with posteriors, and optionally save CalibrationResults/SimulationResults directly (pickle?)


OUTPUT_GENERATOR_REGISTRY = {}


def register_output_generator(kind_set):
    def deco(fn):
        OUTPUT_GENERATOR_REGISTRY[frozenset(kind_set)] = fn
        return fn

    return deco


@register_output_generator({"simulation", "outputs"})
def generate_simulation_outputs(*, simulation: SimulationOutput, outputs: OutputConfig, **_) -> None:
    """"""
    logger.info("OUTPUT GENERATOR: dispatched for simulation")


@register_output_generator({"calibration_results", "outputs"})
def generate_calibration_outputs(*, calibration: CalibrationOutput, outputs: OutputConfig, **_) -> None:
    """"""
    logger.info("OUTPUT GENERATOR: dispatched for calibration")


def dispatch_output_generator(**configs) -> None:
    """"""
    kinds = frozenset(k for k, v in configs.items() if v is not None)
    return OUTPUT_GENERATOR_REGISTRY[kinds](**configs)
