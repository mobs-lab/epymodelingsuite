"""Builder functions for constructing EpiModels and ABCSamplers from configuration."""

import copy
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from epydemix.calibration import ABCSampler, ae, mae, mape, rmse, wmape
from epydemix.model import EpiModel

from ..builders.base import (
    add_model_compartments_from_config,
    add_model_parameters_from_config,
    add_model_transitions_from_config,
    calculate_compartment_initial_conditions,
    calculate_parameters_from_config,
    set_population_from_config,
)
from ..builders.interventions import (
    add_contact_matrix_interventions_from_config,
    add_parameter_interventions_from_config,
    add_school_closure_intervention_from_config,
)
from ..builders.orchestrators import (
    create_model_collection,
    make_simulate_wrapper,
    setup_interventions,
    setup_vaccination_schedules,
)
from ..builders.seasonality import add_seasonality_from_config
from ..builders.utils import get_data_in_location, get_data_in_window
from ..builders.vaccination import add_vaccination_schedules_from_config
from ..schema.basemodel import BasemodelConfig, Parameter, Timespan
from ..schema.calibration import CalibrationConfig
from ..schema.dispatcher import BuilderOutput, ProjectionArguments, SimulationArguments
from ..schema.general import validate_cross_config_consistency
from ..schema.sampling import SamplingConfig
from ..school_closures import make_school_closure_dict
from ..telemetry import ExecutionTelemetry, extract_builder_metadata
from ..utils.config import get_workflow_type_from_configs
from ..vaccinations import reaggregate_vaccines

logger = logging.getLogger(__name__)


# ===== Helper Functions =====


dist_func_dict = {
    "rmse": rmse,
    "wmape": wmape,
    "ae": ae,
    "mae": mae,
    "mape": mape,
}


# ===== Builder Registry and Functions =====


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
    set_population_from_config(model, basemodel.population.name, basemodel.population.age_groups)

    # Compartments and transitions
    add_model_compartments_from_config(model, basemodel.compartments)
    add_model_transitions_from_config(model, basemodel.transitions)

    # Vaccination
    if basemodel.vaccination:
        add_vaccination_schedules_from_config(model, basemodel.transitions, basemodel.vaccination, basemodel.timespan)

    # Parameters
    add_model_parameters_from_config(model, basemodel.parameters)
    if "calculated" in [param_args.type.value for param, param_args in (basemodel.parameters).items()]:
        calculate_parameters_from_config(model, basemodel.parameters)

    # Seasonality (this must occur before interventions to preserve parameter overrides)
    if basemodel.seasonality:
        add_seasonality_from_config(model, basemodel.seasonality, basemodel.timespan)

    # Interventions
    if basemodel.interventions:
        intervention_types = [i.type for i in basemodel.interventions]

        # School closure
        if "school_closure" in intervention_types:
            closure_dict = make_school_closure_dict(
                range(basemodel.timespan.start_date.year, basemodel.timespan.end_date.year + 1)
            )
            add_school_closure_intervention_from_config(model, basemodel.interventions, closure_dict)

        # Contact matrix
        if "contact_matrix" in intervention_types:
            add_contact_matrix_interventions_from_config(model, basemodel.interventions)

        # Parameter
        if "parameter" in intervention_types:
            add_parameter_interventions_from_config(model, basemodel.interventions, basemodel.timespan)

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

    return BuilderOutput(
        primary_id=0,
        seed=basemodel.random_seed,
        delta_t=basemodel.timespan.delta_t,
        model=model,
        simulation=simulation_args,
    )


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
    from ..sample_generator import generate_samples

    logger.info("BUILDER: dispatched for sampling.")

    # Validate references between basemodel and sampling
    validate_cross_config_consistency(basemodel_config, sampling_config)

    # For compactness
    basemodel = basemodel_config.model
    sampling = sampling_config.modelset

    # Build a collection of EpiModels
    models, population_names = create_model_collection(basemodel, sampling.population_names)

    # Output of this is a list of dicts containing start_date, initial conditions, and parameter value
    # combinations where parameters is in the same format as basemodel.parameters.
    # Create empty structure when only using modelset for multiple populations.
    if sampling.sampling == "populations":
        sampled_vars = [{}]
    else:
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
    models, earliest_vax = setup_vaccination_schedules(basemodel, models, sampled_start_timespan, population_names)

    # These interventions are sensitive to location but not to model parameters and can be applied
    # using the earliest start_date before further duplicating the models.
    models = setup_interventions(models, basemodel, intervention_types, sampled_start_timespan)

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
                add_model_parameters_from_config(m, parameters)
            if "calculated" in [param_args.type.value for param, param_args in (basemodel.parameters).items()]:
                calculate_parameters_from_config(m, basemodel.parameters)

            # Vaccination (if start_date is sampled)
            if basemodel.vaccination and sampled_start_timespan:
                reaggregated_vax = reaggregate_vaccines(earliest_vax, timespan.start_date)
                add_vaccination_schedules_from_config(
                    m, basemodel.transitions, basemodel.vaccination, timespan, use_schedule=reaggregated_vax
                )

            # Seasonality (this must occur before parameter interventions to preserve parameter overrides)
            if basemodel.seasonality:
                add_seasonality_from_config(m, basemodel.seasonality, timespan)

            # Parameter interventions
            if basemodel.interventions and "parameter" in intervention_types:
                add_parameter_interventions_from_config(m, basemodel.interventions, timespan)

            # Initial conditions
            compartment_init = calculate_compartment_initial_conditions(
                compartments=basemodel.compartments,
                population_array=m.population.Nk,
                params_dict=varset.get("compartments"),
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
        BuilderOutput(
            primary_id=i, seed=basemodel.random_seed, delta_t=basemodel.timespan.delta_t, model=t[0], simulation=t[1]
        )
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
    from ..utils import distribution_to_scipy

    logger.info("BUILDER: dispatched for calibration.")

    # Validate references between basemodel and calibration
    validate_cross_config_consistency(basemodel_config, calibration_config)

    # For compactness
    basemodel = basemodel_config.model
    modelset = calibration_config.modelset
    calibration = modelset.calibration

    # Create random number generator
    rng = np.random.default_rng(basemodel.random_seed)

    # Build a collection of EpiModels
    models, population_names = create_model_collection(basemodel, modelset.population_names)

    # Extract intervention types
    intervention_types = []
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
    models, earliest_vax = setup_vaccination_schedules(basemodel, models, sampled_start_timespan, population_names)

    # These interventions are sensitive to location but not to model parameters and can be applied
    # using the earliest start_date before creating ABCSamplers.
    models = setup_interventions(models, basemodel, intervention_types, sampled_start_timespan)

    logger.info("BUILDER: setting up ABCSamplers...")

    observed_raw = pd.read_csv(calibration.observed_data_path)
    observed_in_window = get_data_in_window(observed_raw, calibration)
    calibrators = []
    for model in models:
        # TODO: Make location column name configurable instead of hardcoded "geo_value"
        # Should be added to ComparisonSpec schema (e.g., observed_location_column)
        observed_data = get_data_in_location(observed_in_window, model, "geo_value")
        vax_state = get_data_in_location(earliest_vax, model, "location") if earliest_vax is not None else None
        # Create simulate_wrapper
        simulate_wrapper = make_simulate_wrapper(
            basemodel=basemodel,
            calibration=calibration,
            observed_data=observed_data,
            intervention_types=intervention_types,
            sampled_start_timespan=sampled_start_timespan,
            earliest_vax=vax_state,
            rng=rng,
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
            observed_data=observed_data[calibration.comparison[0].observed_value_column].values,
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
            delta_t=basemodel.timespan.delta_t,
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

    Parameters
    ----------
    **configs
        Configuration objects (basemodel_config, sampling_config, calibration_config)

    Returns
    -------
    BuilderOutput | list[BuilderOutput]
        Builder outputs
    """
    # Determine workflow type (used for registry key and summary tracking)
    workflow_type = get_workflow_type_from_configs(configs)

    # Get telemetry from context
    telemetry = ExecutionTelemetry.get_current()

    # Set as current context (for nested calls)
    ExecutionTelemetry.set_current(telemetry)

    try:
        if telemetry:
            telemetry.enter_builder(workflow_type)

        # Dispatch to appropriate builder
        kinds = frozenset(k for k, v in configs.items() if v is not None)
        builder_outputs = BUILDER_REGISTRY[kinds](**configs)

        # Extract metadata and exit builder stage
        if telemetry:
            metadata = extract_builder_metadata(builder_outputs, configs)
            if metadata:
                telemetry.exit_builder(**metadata)

        return builder_outputs
    finally:
        # Clear context when done
        ExecutionTelemetry.set_current(None)
