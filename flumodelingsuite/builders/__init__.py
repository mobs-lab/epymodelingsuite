"""Builder modules for constructing EpiModel instances from configuration."""

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
from .orchestrators import (
    create_model_collection,
    make_simulate_wrapper,
    setup_interventions,
    setup_vaccination_schedules,
)
from .seasonality import add_seasonality_from_config
from .utils import get_data_in_location, get_data_in_window
from .vaccination import add_vaccination_schedules_from_config

# Keep old names with underscores for backward compatibility during transition
_set_population_from_config = set_population_from_config
_add_model_compartments_from_config = add_model_compartments_from_config
_add_model_transitions_from_config = add_model_transitions_from_config
_add_model_parameters_from_config = add_model_parameters_from_config
_calculate_parameters_from_config = calculate_parameters_from_config
_add_vaccination_schedules_from_config = add_vaccination_schedules_from_config
_add_school_closure_intervention_from_config = add_school_closure_intervention_from_config
_add_contact_matrix_interventions_from_config = add_contact_matrix_interventions_from_config
_add_parameter_interventions_from_config = add_parameter_interventions_from_config
_add_seasonality_from_config = add_seasonality_from_config
_create_model_collection = create_model_collection
_setup_vaccination_schedules = setup_vaccination_schedules
_setup_interventions = setup_interventions
_make_simulate_wrapper = make_simulate_wrapper
_get_data_in_window = get_data_in_window
_get_data_in_location = get_data_in_location

__all__ = [
    # Public API
    "set_population_from_config",
    "add_model_compartments_from_config",
    "add_model_transitions_from_config",
    "add_model_parameters_from_config",
    "calculate_parameters_from_config",
    "calculate_compartment_initial_conditions",
    "add_vaccination_schedules_from_config",
    "add_school_closure_intervention_from_config",
    "add_contact_matrix_interventions_from_config",
    "add_parameter_interventions_from_config",
    "add_seasonality_from_config",
    "create_model_collection",
    "setup_vaccination_schedules",
    "setup_interventions",
    "make_simulate_wrapper",
    "get_data_in_window",
    "get_data_in_location",
    # Backward compatibility (private names)
    "_set_population_from_config",
    "_add_model_compartments_from_config",
    "_add_model_transitions_from_config",
    "_add_model_parameters_from_config",
    "_calculate_parameters_from_config",
    "_add_vaccination_schedules_from_config",
    "_add_school_closure_intervention_from_config",
    "_add_contact_matrix_interventions_from_config",
    "_add_parameter_interventions_from_config",
    "_add_seasonality_from_config",
    "_create_model_collection",
    "_setup_vaccination_schedules",
    "_setup_interventions",
    "_make_simulate_wrapper",
    "_get_data_in_window",
    "_get_data_in_location",
]
