import copy

from epydemix.model import EpiModel

WORKFLOW_REGISTRY = {}


def register(kind_set):
    def deco(fn):
        WORKFLOW_REGISTRY[frozenset(kind_set)] = fn
        return fn

    return deco


@register({"basemodel"})
def wf_base_only(*, basemodel, **_):
    # build a single EpiModel from basemodel config
    model = EpiModel()
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
    return {"workflow": "base_only", "model": model}


@register({"basemodel", "modelset"})
def wf_base_modelset(*, basemodel, modelset, **_):
    # need validation of references between basemodel and modelset
    validate_config_pair(basemodel, modelset)

    models = []
    model = EpiModel()
    _add_compartments_from_config(model, basemodel.compartments)
    _add_transitions_from_config(model, basemodel.transitions)
    _add_parameters_from_config(model, basemodel.parameters)
    if modelset.population_names:
        if "all" in modelset.population_names:
            population_names = get_location_codebook()["location_name_epydemix"]
        else:
            population_names = modelset.population_names
        for name in population_names:
            m = copy.deepcopy(model)
            _set_population_from_config(m, name, basemodel.population.age_groups)
            models.append(m)
    else:
        _set_population_from_config(model, basemodel.population.name, basemodel.population.age_groups)
        models.append(m)

    # output of this should be a list of dicts containing start_date, initial conditions, and parameter value
    # combinations where parameters is in the same format as basemodel.parameters
    sampled_vars = _sample_vars_from_config(modelset.sampling)

    newmodels = []
    for model in models:
        for varset in sampled_vars:
            m = copy.deepcopy(model)
            timespan = {
                "start_date": varset.start_date,
                "end_date": basemodel.timespan.end_date,
                "delta_t": basemodel.timespan.delta_t,
            }
            _add_vaccination_from_config(m, basemodel.transitions, basemodel.vaccination, timespan)
            _add_school_closure_intervention_from_config(m, basemodel.interventions, timespan)
            _add_contact_matrix_interventions_from_config(m, basemodel.interventions)

    return {"workflow": "base_modelset", "models": models}


def dispatch_workflow(**configs):
    kinds = frozenset(k for k, v in configs.items() if v is not None)
    return WORKFLOW_REGISTRY[kinds](**configs)
