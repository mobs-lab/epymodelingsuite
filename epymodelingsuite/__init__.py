# epymodelingsuite/__init__.py
# ruff: noqa: E402

from .utils.logging import configure_logging, setup_logger

# Initialize logger with NullHandler (library best practice)
setup_logger()

from .config_loader import (
    load_basemodel_config_from_file,
    load_calibration_config_from_file,
    load_output_config_from_file,
    load_sampling_config_from_file,
)
from .school_closures import add_school_closure_interventions, make_school_closure_dict
from .seasonality import get_seasonal_transmission_balcan
from .vaccinations import (
    add_vaccination_schedule,
    make_vaccination_rate_function,
    scenario_to_epydemix,
    smh_data_to_epydemix,
)

__all__ = [
    "add_school_closure_interventions",
    "add_vaccination_schedule",
    "configure_logging",
    "get_seasonal_transmission_balcan",
    "load_basemodel_config_from_file",
    "load_calibration_config_from_file",
    "load_output_config_from_file",
    "load_sampling_config_from_file",
    "make_school_closure_dict",
    "make_vaccination_rate_function",
    "scenario_to_epydemix",
    "smh_data_to_epydemix",
]
