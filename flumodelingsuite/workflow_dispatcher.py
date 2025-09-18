import copy

from epydemix.model import EpiModel
from numpy import float64, int64, ndarray

from .basemodel_validator import BasemodelConfig, Parameter, Timespan
from .calibration_validator import CalibrationConfig
from .config_loader import *
from .sampling_validator import SamplingConfig
from .school_closures import make_school_closure_dict
from .utils import get_location_codebook
from .vaccinations import reaggregate_vaccines, scenario_to_epydemix

# ===== Helpers =====


# Needed for school closures
def get_year(datestring: str) -> dt.date:
    """Extract year from a string (YYYY-MM-DD)"""
    date_format = "%Y-%m-%d"
    return dt.datetime.strptime(datestring, date_format).year


# ===== Workflows =====

WORKFLOW_REGISTRY = {}


def register(kind_set):
    def deco(fn):
        WORKFLOW_REGISTRY[frozenset(kind_set)] = fn
        return fn

    return deco


@register({"basemodel"})
def wf_base_only(*, basemodel: BasemodelConfig, **_):
    """
    Workflow using only a basemodel.
    """
    # For compactness
    basemodel = basemodel.model

    # build a single EpiModel from basemodel config
    model = EpiModel()

    # Set the model name if provided in the config
    if basemodel.name is not None:
        model.name = basemodel.name

    # This workflow uses a single population
    _set_population_from_config(model, basemodel.population.name, basemodel.population.age_groups)

    # Compartments and transitions
    _add_compartments_from_config(model, basemodel.compartments)
    _add_transitions_from_config(model, basemodel.transitions)

    # Vaccination
    if basemodel.vaccination:
        _add_vaccination_schedules_from_config(model, basemodel.transitions, basemodel.vaccination, basemodel.timespan)

    # Parameters
    _add_model_parameters_from_config(model, basemodel.parameters)
    if "calculated" in [p.type for p in basemodel.parameters]:
        _calculate_parameters_from_config(model, basemodel.parameters)

    # Interventions
    if basemodel.interventions:
        intervention_types = [i.type for i in basemodel.interventions]

        # School closure
        if "school_closure" in intervention_types:
            closure_dict = make_school_closure_dict(
                range(start=get_year(basemodel.timespan.start_date), stop=get_year(basemodel.timespan.end_date) + 1)
            )
            _add_school_closure_intervention_from_config(model, basemodel.interventions, closure_dict)

        # Contact matrix
        if "contact_matrix" in intervention_types:
            _add_contact_matrix_interventions_from_config(model, basemodel.interventions)

        # Parameter
        if "parameter" in intervention_types:
            _add_parameter_interventions_from_config(model, basemodel.interventions)

    # Seasonality
    if basemodel.seasonality:
        _add_seasonality_from_config(model, basemodel.seasonality, basemodel.timespan)

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

    return {"workflow": "base_only", "model": model, "compartment_inits": compartment_inits}


@register({"basemodel", "sampling"})
def wf_sampling(*, basemodel: BasemodelConfig, sampling: SamplingConfig, **_) -> dict:
    """
    Sampling workflow.
    """
    # Need validation of references between basemodel and sampling
    validate_sampling_basemodel(basemodel, sampling)

    # For compactness
    basemodel = basemodel.model
    sampling = sampling.modelset

    # Set the model name if provided in the config
    if basemodel.name is not None:
        model.name = basemodel.name

    # Build a collection of EpiModels
    models = []
    init_model = EpiModel()

    # All models will share compartments, transitions, and non-sampled/calculated parameters
    _add_compartments_from_config(init_model, basemodel.compartments)
    _add_transitions_from_config(init_model, basemodel.transitions)
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

    # output of this should be a list of dicts containing start_date, initial conditions, and parameter value
    # combinations where parameters is in the same format as basemodel.parameters
    sampled_vars = _sample_vars_from_config(modelset.sampling)

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
                        range(
                            start=get_year(earliest_timespan.start_date), stop=get_year(earliest_timespan.end_date) + 1
                        )
                    )
                else:
                    closure_dict = make_school_closure_dict(
                        range(
                            start=get_year(basemodel.timespan.start_date),
                            stop=get_year(basemodel.timespan.end_date) + 1,
                        )
                    )
                _add_school_closure_intervention_from_config(model, basemodel.interventions, closure_dict)

            # Contact matrix
            if "contact_matrix" in intervention_types:
                _add_contact_matrix_interventions_from_config(model, basemodel.interventions)

    # Create models with sampled/calculated parameters, apply vaccination and interventions
    compartment_inits = []
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
            if basemodel.vaccination:
                if earliest_timespan:
                    reaggregated_vax = reaggregate_vaccines(earliest_vax, timespan.start_date)
                    _add_vaccination_schedules_from_config(
                        m, basemodel.transitions, basemodel.vaccination, timespan, use_schedule=reaggregated_vax
                    )

            # Parameter interventions
            if basemodel.interventions:
                if "parameter" in intervention_types:
                    _add_parameter_interventions_from_config(m, basemodel.interventions)

            # Seasonality
            if basemodel.seasonality:
                _add_seasonality_from_config(m, basemodel.seasonality, timespan)

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

            final_models.append(m)
            compartment_inits.append(compartment_init)

    return {"workflow": "sampling", "models": final_models, "compartment_inits": compartment_inits}


@register({"basemodel", "calibration"})
def wf_base_calibration(*, basemodel: BasemodelConfig, calibration: CalibrationConfig, **_):
    models = []
    return {"workflow": "calibration", "models": models}


def dispatch_workflow(**configs):
    kinds = frozenset(k for k, v in configs.items() if v is not None)
    return WORKFLOW_REGISTRY[kinds](**configs)
