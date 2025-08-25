### config_loader.py
# Functions for loading and validating configuration files (defined in YAML format).

import ast
import logging
import operator
from typing import Any

import numpy as np
import scipy
from epydemix.model import EpiModel

from .config_validator import RootConfig, validate_config

logger = logging.getLogger(__name__)

# === Utility functions for evaluating expressions ===
# Allowed binary operators mapping
_allowed_operators = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

# Allowed unary operators mapping
_allowed_unary_operators = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Names of top-level modules we allow
_allowed_modules = {"np", "scipy"}


class SafeEvalVisitor(ast.NodeVisitor):
    """A NodeVisitor that only allows numeric, numpy, and scipy expressions."""

    def visit(self, node):
        t = type(node)
        # Permit only these node types
        if t in (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Constant,
            ast.Num,
            ast.Load,
            ast.Name,
            ast.Attribute,
            ast.Call,
        ):
            return super().visit(node)
        raise ValueError(f"Disallowed expression: {t.__name__}")

    def visit_BinOp(self, node):
        self.visit(node.left)
        self.visit(node.right)
        if type(node.op) not in _allowed_operators:
            raise ValueError(f"Operator {type(node.op).__name__} not allowed")

    def visit_UnaryOp(self, node):
        self.visit(node.operand)
        if type(node.op) not in _allowed_unary_operators:
            raise ValueError(f"Unary operator {type(node.op).__name__} not allowed")

    def visit_Constant(self, node):
        # Only allow numeric constants
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"Constant of type {type(node.value).__name__} not allowed")

    def visit_Num(self, node):
        # For Python <3.8
        if not isinstance(node.n, (int, float)):
            raise ValueError(f"Num of type {type(node.n).__name__} not allowed")

    def visit_Name(self, node):
        # Only allow top‐level names 'np' and 'scipy'
        if node.id not in _allowed_modules:
            raise ValueError(f"Name '{node.id}' is not allowed")

    def visit_Attribute(self, node):
        # Recursively ensure base is allowed module (np or scipy)
        if self._is_allowed_attr_chain(node):
            # visit the base value to enforce nested checks
            self.visit(node.value)
        else:
            raise ValueError(f"Attribute access '{ast.dump(node)}' not allowed")

    def _is_allowed_attr_chain(self, node):
        # Base case: node.value is Name in allowed_modules
        if isinstance(node.value, ast.Name) and node.value.id in _allowed_modules:
            return True
        # Recursive: node.value is another Attribute
        if isinstance(node.value, ast.Attribute):
            return self._is_allowed_attr_chain(node.value)
        return False

    def visit_Call(self, node):
        # Only allow calls of form (np.xxx(...)) or (scipy.xxx(...))
        if isinstance(node.func, ast.Attribute):
            # validate the attribute chain (np or scipy)
            self.visit(node.func)
            # validate all arguments
            for arg in node.args:
                self.visit(arg)
            for kw in node.keywords:
                self.visit(kw.value)
        else:
            raise ValueError("Function calls other than np.xxx or scipy.xxx are not allowed")


def _safe_eval(expr: str) -> Any:
    """
    Safely evaluate a numeric expression from a string, allowing literal numbers,
    basic arithmetic operators, and functions from numpy and scipy.

    Parameters
    ----------
    expr : str
            The expression to evaluate (e.g. "1/10" or "np.exp(-2) + 3 * np.sqrt(4)").

    Returns
    -------
    Any
            The result of evaluating the expression. Depending on the expression, this
            may be one of:
              - A Python numeric type: int, float, or complex.
              - A NumPy scalar (e.g. numpy.int64, numpy.float64).
              - A NumPy ndarray.
              - A SciPy sparse matrix (subclass of scipy.sparse.spmatrix).

    """
    # Parse into an AST
    tree = ast.parse(expr, mode="eval")

    # Validate AST nodes
    SafeEvalVisitor().visit(tree)

    # Compile and evaluate with restricted globals
    code = compile(tree, filename="<safe_eval>", mode="eval")
    return eval(code, {"__builtins__": None, "np": np, "scipy": scipy}, {})


# === Model setup functions ===
def _add_model_compartments_from_config(model: EpiModel, config: RootConfig) -> EpiModel:
    """
    Add compartments to the EpiModel instance from the configuration dictionary.

    Parameters
    ----------
            model (EpiModel): The EpiModel instance to which compartments will be added.
            config (dict): Configuration dictionary containing compartment definitions.

    Returns
    -------
            EpiModel: EpiModel instance with compartments added.
    """
    # Check that required attributes of model configuration are not None
    if config.model.compartments is None:
        return model

    # Add compartments to the model
    try:
        compartment_ids = [compartment.id for compartment in config.model.compartments]
        model.add_compartments(compartment_ids)
        logger.info(f"Added compartments: {compartment_ids}")
    except Exception as e:
        raise ValueError(f"Error adding compartments: {e}")

    return model


def _add_model_transitions_from_config(model: EpiModel, config: RootConfig) -> EpiModel:
    """
    Add transitions between compartments to the EpiModel instance from the configuration dictionary.

    Parameters
    ----------
            model (EpiModel): The EpiModel instance to which compartment transitions will be added.
            config (dict): Configuration dictionary containing compartment transitions.

    Returns
    -------
            EpiModel: EpiModel instance with compartment transitions added.
    """
    # Check that required attributes of model configuration are not None
    if config.model.transitions is None:
        return model

    # Add transitions to the model
    for transition in config.model.transitions:
        if transition.type == "mediated":
            try:
                model.add_transition(
                    transition.source,
                    transition.target,
                    params=(transition.mediators.rate, transition.mediators.source),
                    kind=transition.type,
                )
                logger.info(
                    f"Added mediated transition: {transition.source} -> {transition.target} (mediator: {transition.mediators.source}, rate: {transition.mediators.rate})"
                )
            except Exception as e:
                raise ValueError(f"Error adding mediated transition {transition}: {e}")
        elif transition.type == "spontaneous":
            try:
                model.add_transition(transition.source, transition.target, params=transition.rate, kind=transition.type)
                logger.info(
                    f"Added spontaneous transition: {transition.source} -> {transition.target} (rate: {transition.rate})"
                )
            except Exception as e:
                raise ValueError(f"Error adding spontaneous transition {transition}: {e}")

    return model


def _add_model_parameters_from_config(model: EpiModel, config: RootConfig) -> EpiModel | list[EpiModel]:
    """
    Add parameters to the EpiModel instance from the configuration dictionary.

    Parameters
    ----------
            model (EpiModel): The EpiModel instance to which parameters will be added.
            config (dict): Configuration dictionary containing model parameters.

    Returns
    -------
            EpiModel: EpiModel instance with parameters added.
            Or,
            list[EpiModel]: EpiModel instances with scanned parameters.
    """
    import copy
    from itertools import product

    from epydemix.utils import convert_to_2Darray

    # Check that required attributes of model configuration are not None
    if config.model.parameters is None:
        return model

    # Add parameters to the model
    parameters_dict = {}
    scan_dict = {}
    for key, data in config.model.parameters.items():
        if data.type == "scalar":
            parameters_dict[key] = _safe_eval(data.value)
        elif data.type == "age_varying":  # Ensure array matches population age structure
            if model.population.num_groups == len(data.values):
                parameters_dict[key] = convert_to_2Darray([_safe_eval(val) for val in data.values])
            else:
                raise ValueError(
                    f"Array values supplied for parameter {key} do not match model population age structure"
                )
        elif data.type == "calibrated":
            parameters_dict[key] = None
        elif data.type == "scan":
            scan_dict[key] = [_safe_eval(val) for val in data.values]

    try:
        model.add_parameter(parameters_dict=parameters_dict)
        logger.info(f"Added parameters: {list(parameters_dict.keys())}")

        if scan_dict:
            models = []
            keys, values = zip(*scan_dict.items(), strict=False)
            permutations = [dict(zip(keys, v, strict=False)) for v in product(*values)]
            for perm in permuations:
                mod = copy.deepcopy(model)
                mod.add_parameter(parameters_dict=perm)
                models.append(mod)

            logger.info(f"Created split models for scanned parameters {keys}")
            return models

        return model
    except Exception as e:
        raise ValueError(f"Error adding parameters to model: {e}")


def _add_seasonality_from_config(model: EpiModel, config: RootConfig) -> EpiModel:
    """
    Add seasonally varying transmission rate to the EpiModel from the configuration dictionary.

    Parameters
    ----------
            model (EpiModel): The EpiModel instance to apply seasonality to.
            config (RootConfig): The configuration object containing seasonality parameters.

    Returns
    -------
            EpiModel: EpiModel instance with seasonal transmission applied.
    """
    import datetime as dt

    import numpy as np
    from epydemix.utils import compute_simulation_dates

    from .seasonality import get_seasonal_transmission_balcan

    # Check that required attributes of model configuration are not None
    if config.model.parameters is None or config.model.seasonality is None:
        return model

    def format_date(datestring: str) -> dt.date:
        """Format a date from a string (YYYY-MM-DD)"""
        date_format = "%Y-%m-%d"
        return dt.datetime.strptime(datestring, date_format).date()

    # Parameter must already be defined
    try:
        previous_value = model.get_parameter(config.model.seasonality.target_parameter)
    except KeyError:
        raise ValueError(
            f"Attempted to apply seasonality to undefined parameter {config.model.seasonality.target_parameter}"
        )

    # Calculate rescaling factor with requested method
    if config.model.seasonality.method == "balcan":
        # Minimum transmission date is optional
        if config.model.seasonality.seasonality_min_date is not None:
            date_tmin = config.model.seasonality.seasonality_min_date
        else:
            date_tmin = None
        # Do the calculation
        dates, st = get_seasonal_transmission_balcan(
            date_start=config.model.simulation.start_date,
            date_stop=config.model.simulation.stop_date,
            date_tmax=config.model.seasonality.seasonality_max_date,
            date_tmin=date_tmin,
            R_min=config.model.seasonality.transmissibility_min,
            R_max=config.model.seasonality.transmissibility_max,
        )
    else:
        raise ValueError(f"Undefined seasonality method recieved: {config.model.seasonality.method}")

    # Handle possibilities for previous parameter value (expressions should already be evaluated at parameter definition)
    # If existing parameter is constant, transform to array of size (T,) with time-varying values
    if not hasattr(previous_value, "__len__"):
        new_value = st * np.array(previous_value)
    # If existing parameter is age-varying (array of size (1, N)), transform to array of size (T, N) with time-varying and age-varying values
    elif previous_value.shape == (1, model.population.num_groups):
        new_value = np.zeros(
            (
                len(
                    compute_simulation_dates(
                        start_date=config.model.simulation.start_date, end_date=config.model.simulation.end_date
                    )
                ),
                model.population.num_groups,
            )
        )
        for i in range(model.population.num_groups):
            new_value[:, i] = st * np.array(previous_value[0, i])
    # Uncertain how this will work for priors
    else:
        raise ValueError(
            f"Cannot apply seasonality to existing parameter {config.model.seasonality.target_parameter} = {previous_value}"
        )

    # Overwrite parameter with new seasonal values
    try:
        model.add_parameter(config.model.seasonality.target_parameter, new_value)
        logger.info(f"Added seasonality to parameter {config.model.seasonality.target_parameter}")
    except Exception as e:
        raise ValueError(f"Error adding parameters to model: {e}")

    return model


def _add_vaccination_schedules_from_config(model: EpiModel, config: RootConfig) -> EpiModel:
    """
    Add transitions between compartments due to vaccination to the EpiModel instance from the configuration dictionary.

    Parameters
    ----------
            model (EpiModel): The EpiModel instance to which vaccination schedules will be added.
            config (RootConfig): The configuration object containing vaccination schedule details.

    Returns
    -------
            EpiModel: EpiModel instance with vaccination schedules added.
    """
    import pandas as pd

    from .vaccinations import add_vaccination_schedule, make_vaccination_probability_function, scenario_to_epydemix

    # Check that required attributes of model configuration are not None
    if config.model.transitions is None or config.model.vaccination is None:
        return model

    # Extract compartment transitions due to vaccination
    vaccination_transitions = [
        transition for transition in config.model.transitions if transition.type == "vaccination"
    ]

    # If no vaccination transitions, return model as is
    if not vaccination_transitions:
        logger.info("No vaccination transitions found in configuration.")
        return model

    age_groups = config.model.population.age_groups

    state = config.model.population.name

    if config.model.simulation.delta_t is not None:
        delta_t = config.model.simulation.delta_t
    else:
        logger.info("'delta_t' not found in simulation configuration, defaulting to 1.0 (1 day)")
        delta_t = 1.0

    # Define vaccine probability function
    vaccine_probability_function = make_vaccination_probability_function(
        config.model.vaccination.origin_compartment, config.model.vaccination.eligible_compartments
    )

    # Vaccination schedule data
    preprocessed_vaccination_data_path = config.model.vaccination.preprocessed_vaccination_data_path

    if preprocessed_vaccination_data_path:
        # Load preprocessed vaccination schedule if provided
        vaccination_schedule = pd.read_csv(preprocessed_vaccination_data_path)
        logger.info(f"Loaded preprocessed vaccination schedule from {preprocessed_vaccination_data_path}")
    else:
        # Otherwise, create vaccination schedule from SMH scenario
        scenario_data_path = config.model.vaccination.scenario_data_path
        start_date = config.model.simulation.start_date
        end_date = config.model.simulation.end_date

        try:
            vaccination_schedule = scenario_to_epydemix(
                input_filepath=scenario_data_path,
                start_date=start_date,
                end_date=end_date,
                target_age_groups=age_groups,
                delta_t=delta_t,
                output_filepath=None,
                state=state,
            )
            logger.info(f"Created vaccination schedule from scenario data at {scenario_data_path}")
        except Exception as e:
            raise ValueError(f"Error creating vaccination schedule from scenario data:\n{e}")

    # Add vaccine transitions to the model
    for transition in vaccination_transitions:
        try:
            model = add_vaccination_schedule(
                model=model,
                vaccine_probability_function=vaccine_probability_function,
                source_comp=transition.source,
                target_comp=transition.target,
                vaccination_schedule=vaccination_schedule,
            )
            logger.info(f"Added vaccination transition: {transition.source} -> {transition.target}")
        except Exception as e:
            raise ValueError(f"Error adding vaccination transition {transition}: {e}")

    return model


def _parse_age_group(group_str: str) -> list:
    """
    Parse an age group string like "0-4", "65+" into a list of individual age labels.
    For "a-b", returns [str(a), str(a+1), ..., str(b)].
    For "c+", returns [str(c), ..., "84", "84+"].

    Parameters
    ----------
            group_str (str): Age group string to parse.

    Returns
    -------
            list: List of individual age labels as strings.
    """
    if group_str.endswith("+"):
        # e.g. "65+" -> start=65, end at 84 then add "84+"
        start = int(group_str[:-1])
        end = 84
        labels = [str(i) for i in range(start, end)] + [f"{end}+"]
    else:
        # e.g. "5-17" -> start=5, end=17
        start, end = map(int, group_str.split("-"))
        labels = [str(i) for i in range(start, end + 1)]
    return labels


def _set_populations_from_config(model: EpiModel, config: RootConfig) -> list[EpiModel]:
    """
    Use the supplied EpiModel to create EpiModel instances with populations set from config.

    Parameters
    ----------
            model (EpiModel): The EpiModel instance for which the population will be set. If multiple populations, multiple EpiModels will be created based on the supplied instance.
            config (RootConfig): The configuration object containing population details.

    Returns
    -------
            list[EpiModel]: EpiModel instances with populations set from config.
    """
    import copy

    from epydemix.population import load_epydemix_population

    from .utils import convert_location_name_format

    # Check that required attributes of model configuration are not None
    if config.model.population is None:
        return [model]

    try:
        # Get population name, and convert to corresponding "epydemix_population" name
        population_names = [
            convert_location_name_format(name, "epydemix_population") for name in config.model.population.names
        ]

        # Get age groups
        age_groups = config.model.population.age_groups

        # Create age group mapping
        if age_groups:
            age_group_mapping = {group: _parse_age_group(group) for group in age_groups}
        else:
            age_group_mapping = None

        populations = [
            load_epydemix_population(population_name=name, age_group_mapping=age_group_mapping)
            for name in population_names
        ]

        models = []
        for pop in populations:
            mod = copy.deepcopy(model)
            mod.set_population(pop)
            models.append(mod)

        logger.info(f"Model populations set for: {population_names}")
    except Exception as e:
        raise ValueError(f"Error setting population: {e}")

    return models


def _add_school_closure_intervention_from_config(model: EpiModel, config: RootConfig) -> EpiModel:
    """
    Apply a school closure intervention to the EpiModel instance.

    Parameters
    ----------
            model (EpiModel): The EpiModel instance to which the intervention will be applied.
            config (RootConfig): The configuration object containing intervention details.

    Returns
    -------
            EpiModel: EpiModel instance with the intervention applied.
    """
    # Check that required attributes of model configuration are not None
    if config.model.interventions is None:
        return model

    # Load school closure functions
    from .school_closures import add_school_closure_interventions, make_school_closure_dict

    # Extract school closure interventions
    school_closures_interventions = [
        intervention for intervention in config.model.interventions if intervention.type == "school_closure"
    ]

    # Confirm that there are school closure interventions to apply
    if len(school_closures_interventions) == 0:
        return model

    for intervention in school_closures_interventions:
        try:
            closure_dict = make_school_closure_dict(intervention.years)
            add_school_closure_interventions(
                model=model, closure_dict=closure_dict, reduction_factor=intervention.reduction_factor
            )
            logger.info(
                f"Applied school closure intervention for years: {intervention.years} with reduction factor: {intervention.reduction_factor}"
            )
        except Exception as e:
            raise ValueError(f"Error applying school closure intervention {intervention}:\n{e}")

    return model


def load_model_config_from_file(path: str) -> RootConfig:
    """
    Load model configuration YAML from the given path and validate against the schema.

    Parameters
    ----------
            path (str): The file path to the YAML configuration file.

    Returns
    -------
            RootConfig: The validated configuration object.
    """
    from pathlib import Path

    import yaml

    with Path(path).open() as f:
        raw = yaml.safe_load(f)

    root = validate_config(raw)
    logger.info("Configuration loaded successfully.")
    return root


def setup_epimodels_from_config(config: RootConfig) -> list[EpiModel]:
    """
    Set up EpiModel instances from a RootConfig instance which may contain scanned parameters or multiple populations.

    Parameters
    ----------
            config (RootConfig): RootConfig instance containing model details. Use `load_model_config_from_file(path_to_yaml)`.

    Returns
    -------
            list[EpiModel]: EpiModel instances configured according to the provided settings.
    """
    # Validate that 'model' exists in config
    if config.model is None:
        raise ValueError("Configuration must contain a 'model' key.")

    # Create an empty instance of EpiModel
    base_model = EpiModel()

    # Set the model name if provided in the config
    if config.model.name is not None:
        base_model.name = config.model.name

    # Set up compartments
    base_model = _add_model_compartments_from_config(base_model, config)

    # Set up transitions
    base_model = _add_model_transitions_from_config(base_model, config)

    # Set population
    models = _set_populations_from_config(base_model, config)

    # Set up parameters
    models = [_add_model_parameters_from_config(mod, config) for mod in models]

    # Flatten if nested list due to scanned parameters
    if any(isinstance(_, list) for _ in models):
        models = sum(models, [])

    # Everything below is sensitive to start date (calibrated or scanned), everything above is not
    # Seasonality is also sensitive to calibration on the target parameter

    # Apply seasonality
    models = [_add_seasonality_from_config(mod, config) for mod in models]

    # Set up vaccination schedules
    models = [_add_vaccination_schedules_from_config(mod, config) for mod in models]

    # Apply school closure
    models = [_add_school_closure_intervention_from_config(mod, config) for mod in models]

    return models  # noqa: RET504


def setup_epimodel_from_config(config: RootConfig) -> EpiModel:
    """
    Set up an EpiModel instance from a RootConfig instance. Assumes only 1 population and no scanned parameters.

    Parameters
    ----------
            config (RootConfig): RootConfig instance containing model details. Use `load_model_config_from_file(path_to_yaml)`.

    Returns
    -------
            EpiModel: An instance of EpiModel configured according to the provided settings.
    """
    # Validate that 'model' exists in config
    if config.model is None:
        raise ValueError("Configuration must contain a 'model' key.")

    # Create an empty instance of EpiModel
    model = EpiModel()

    # Set the model name if provided in the config
    if config.model.name is not None:
        model.name = config.model.name

    # Set population
    model = _set_populations_from_config(model, config)[0]

    # Set up compartments
    model = _add_model_compartments_from_config(model, config)

    # Set up transitions
    model = _add_model_transitions_from_config(model, config)

    # Set up parameters
    model = _add_model_parameters_from_config(model, config)

    # Apply seasonality
    model = _add_seasonality_from_config(model, config)

    # Set up vaccination schedules
    model = _add_vaccination_schedules_from_config(model, config)

    # Apply school closure
    model = _add_school_closure_intervention_from_config(model, config)

    return model  # noqa: RET504
