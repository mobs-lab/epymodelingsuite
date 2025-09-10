import copy

from epydemix.model import EpiModel
from .config_loader import *

WORKFLOW_REGISTRY = {}


def register(kind_set):
    def deco(fn):
        WORKFLOW_REGISTRY[frozenset(kind_set)] = fn
        return fn

    return deco


@register({"basemodel"})
def wf_base_only(*, basemodel: BasemodelConfig, **_):
    
    # for compactness
    basemodel = basemodel.model
    
    # build a single EpiModel from basemodel config
    model = EpiModel()

    # Set the model name if provided in the config
    if basemodel.name is not None:
        model.name = basemodel.name
        
    _set_population_from_config(model, basemodel.population.name, basemodel.population.age_groups)
    _add_compartments_from_config(model, basemodel.compartments)
    _add_transitions_from_config(model, basemodel.transitions)
    _add_vaccination_from_config(model, basemodel.transitions, basemodel.vaccination, basemodel.timespan)
    _add_parameters_from_config(model, basemodel.parameters)
    _calculate_parameters_from_config(model, basemodel.parameters)
    _add_school_closure_intervention_from_config(model, basemodel.interventions, basemodel.timespan)
    _add_contact_matrix_interventions_from_config(model, basemodel.interventions)
    _add_seasonality_from_config(model, basemodel.seasonality, basemodel.timespan)
    _add_parameter_interventions_from_config(model, basemodel.interventions)

    # TODO: initial conditions and simulations
    
    return {"workflow": "base_only", "model": model}


@register({"basemodel", "sampling"})
def wf_base_sampling(*, basemodel: BasemodelConfig, sampling: SamplingConfig, **_):
    from .utils import get_location_codebook
    
    # Need validation of references between basemodel and sampling
    validate_sampling_basemodel(basemodel, sampling)

    # For compactness
    basemodel = basemodel.model
    sampling = sampling.modelset
    
    # Set the model name if provided in the config
    if basemodel.name is not None:
        model.name = basemodel.name

    # Build a set of EpiModels
    models = []
    model = EpiModel()

    # All models will share compartments, transitions, and non-sampled/calculated parameters
    _add_compartments_from_config(model, basemodel.compartments)
    _add_transitions_from_config(model, basemodel.transitions)
    _add_parameters_from_config(model, basemodel.parameters)

    # Create models with populations set
    if sampling.population_names:
        if "all" in sampling.population_names:
            population_names = get_location_codebook()["location_name_epydemix"]
        else:
            population_names = sampling.population_names
        for name in population_names:
            m = copy.deepcopy(model)
            _set_population_from_config(m, name, basemodel.population.age_groups)
            models.append(m)
    else:
        _set_population_from_config(model, basemodel.population.name, basemodel.population.age_groups)
        models.append(model)

    # output of this should be a list of dicts containing start_date, initial conditions, and parameter value
    # combinations where parameters is in the same format as basemodel.parameters
    sampled_vars = _sample_vars_from_config(modelset.sampling)

    # Create models with sampled/calculated parameters, apply vaccination and interventions
    newmodels = []
    for model in models:
        for varset in sampled_vars:
            m = copy.deepcopy(model)
            timespan = {
                "start_date": varset.start_date,
                "end_date": basemodel.timespan.end_date,
                "delta_t": basemodel.timespan.delta_t,
            }
            _add_parameters_from_config(m, varset.parameters)
            _calculate_parameters_from_config(m, basemodel.parameters)
            _add_vaccination_from_config(m, basemodel.transitions, basemodel.vaccination, timespan)
            _add_school_closure_intervention_from_config(m, basemodel.interventions, timespan)
            _add_contact_matrix_interventions_from_config(m, basemodel.interventions)
            _add_seasonality_from_config(m, basemodel.seasonality, timespan)
            _add_parameter_interventions_from_config(m, basemodel.interventions)
            
            # TODO: initial conditions

            newmodels.append(m)
    models = newmodels

    # TODO: simulations

    return {"workflow": "sampling", "models": models}


@register({"basemodel", "calibration"})
def wf_base_calibration(*, basemodel: BasemodelConfig, calibration: CalibrationConfig, **_):
    models = []
    return {"workflow": "calibration", "models": models}


def dispatch_workflow(**configs):
    kinds = frozenset(k for k, v in configs.items() if v is not None)
    return WORKFLOW_REGISTRY[kinds](**configs)
