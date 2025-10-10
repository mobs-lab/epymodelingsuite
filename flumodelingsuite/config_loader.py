### config_loader.py
# Functions for loading and validating configuration files (defined in YAML format).

import ast
import logging
import operator
from typing import Any

import numpy as np
import scipy
from epydemix.model import EpiModel
from pandas import DataFrame

from .basemodel_validator import (
    BasemodelConfig,
    Compartment,
    Intervention,
    Parameter,
    Seasonality,
    Timespan,
    Transition,
    Vaccination,
    validate_basemodel,
)
from .calibration_validator import CalibrationConfig, validate_calibration
from .sampling_validator import SamplingConfig, validate_sampling

__all__ = [
    "load_basemodel_config_from_file",
    "load_calibration_config_from_file",
    "load_sampling_config_from_file",
]

logger = logging.getLogger(__name__)

# === Utility functions ===

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


class RetrieveName(ast.NodeTransformer):
    """
    A NodeTransformer for substituting terms in an expression with parameter values or contact matrix eigenvalue from an EpiModel.
    Used for calculated parameters.
    Constructor requires an EpiModel with contact matrices.
    """

    def __init__(self, model: EpiModel):
        self.model = model

    def visit_Name(self, node):
        if node.id not in _allowed_modules:
            if node.id == "eigenvalue":
                try:
                    C = np.sum([c for _, c in self.model.population.contact_matrices.items()], axis=0)
                    eigenvalue = np.linalg.eigvals(C).real.max()
                    return ast.Constant(value=eigenvalue)
                except Exception as e:
                    raise ValueError(f"Error calculating eigenvalue of contact matrix: {e}")
            else:
                try:
                    value = self.model.get_parameter(node.id)
                    return ast.Constant(value=value)
                except Exception as e:
                    raise ValueError(f"Error obtaining parameter value during calculation: {e}")


def _safe_eval(expr: str) -> Any:
    """
    Safely evaluate a numeric expression from a string, allowing literal numbers,
    basic arithmetic operators, and functions from numpy and scipy.

    Parameters
    ----------
        expr: The expression to evaluate (e.g. "1/10" or "np.exp(-2) + 3 * np.sqrt(4)").

    Returns
    -------
        The result of evaluating the expression. Depending on the expression,
        this may be one of:
            - A Python numeric type: int, float, or complex.
            - A NumPy scalar (e.g. numpy.int64, numpy.float64).
            - A NumPy ndarray.
            - A SciPy sparse matrix (subclass of scipy.sparse.spmatrix).

    Raises
    ------
        ValueError: If the expression contains disallowed operations or syntax.
        SyntaxError: If the expression has invalid Python syntax.

    """
    # Parse into an AST
    tree = ast.parse(expr, mode="eval")

    # Validate AST nodes
    SafeEvalVisitor().visit(tree)

    # Compile and evaluate with restricted globals
    code = compile(tree, filename="<safe_eval>", mode="eval")
    return eval(code, {"__builtins__": None, "np": np, "scipy": scipy}, {})


def _parse_age_group(group_str: str) -> list:
    """
    Parse an age group string like "0-4", "65+" into a list of individual age labels.
    For "a-b", returns [str(a), str(a+1), ..., str(b)].
    For "c+", returns [str(c), ..., "84", "84+"].

    Parameters
    ----------
        group_str: Age group string to parse.

    Returns
    -------
        List of individual age labels as strings.
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


# === Model setup functions ===


def _set_population_from_config(model: EpiModel, population_name: str, age_groups: list[str]) -> EpiModel:
    """
    Set the population for the EpiModel instance.

    Parameters
    ----------
        model: The EpiModel instance for which the population will be set.
        population_name: Name of the population to load.
        age_groups: List of age group strings to map.

    Returns
    -------
        EpiModel instance with the population set.
    """
    from epydemix.population import load_epydemix_population

    from .utils import convert_location_name_format

    try:
        # Convert to "epydemix_population" name
        population_name = convert_location_name_format(population_name, "epydemix_population")

        # Create age group mapping
        age_group_mapping = {group: _parse_age_group(group) for group in age_groups}
        population = load_epydemix_population(population_name=population_name, age_group_mapping=age_group_mapping)
        model.set_population(population)
        logger.info(f"Model population set to: {population_name}")
    except Exception as e:
        raise ValueError(f"Error setting population: {e}")
    return model


def _add_model_compartments_from_config(model: EpiModel, compartments: list[Compartment]) -> EpiModel:
    """
    Add compartments to the EpiModel instance.

    Parameters
    ----------
        model: The EpiModel instance to which compartments will be added.
        compartments: List of Compartment objects containing compartment definitions.

    Returns
    -------
        EpiModel instance with compartments added.
    """
    # Add compartments to the model
    try:
        compartment_ids = [compartment.id for compartment in compartments]
        model.add_compartments(compartment_ids)
        logger.info(f"Added compartments: {compartment_ids}")
    except Exception as e:
        raise ValueError(f"Error adding compartments: {e}")

    return model


def _add_model_transitions_from_config(model: EpiModel, transitions: list[Transition]) -> EpiModel:
    """
    Add transitions between compartments to the EpiModel instance.

    Parameters
    ----------
        model: The EpiModel instance to which compartment transitions will be added.
        transitions: List of Transition objects defining transitions.

    Returns
    -------
        EpiModel instance with compartment transitions added.
    """
    # Check that required attributes of model configuration are not None
    if transitions is None:
        return model

    # Add transitions to the model
    for transition in transitions:
        if transition.type == "mediated":
            try:
                model.add_transition(
                    transition.source,
                    transition.target,
                    params=(transition.rate, transition.mediator),
                    kind=transition.type,
                )
                logger.info(
                    f"Added mediated transition: {transition.source} -> {transition.target} (mediator: {transition.mediator}, rate: {transition.rate})"
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


def _add_model_parameters_from_config(model: EpiModel, parameters: dict[str, Parameter]) -> EpiModel:
    """
    Add parameters to the EpiModel instance.

    Parameters
    ----------
        model: The EpiModel instance to which parameters will be added.
        parameters: Dictionary mapping parameter names to Parameter objects.

    Returns
    -------
        EpiModel instance with parameters added.
    """
    from epydemix.utils import convert_to_2Darray

    # Add parameters to the model
    parameters_dict = {}
    scan_dict = {}
    for key, data in parameters.items():
        if data.type == "scalar":
            if type(data.value) is str:
                parameters_dict[key] = _safe_eval(data.value)
            else:
                parameters_dict[key] = data.value
        elif data.type == "age_varying":  # Ensure array matches population age structure
            if model.population.num_groups == len(data.values):
                parameters_dict[key] = convert_to_2Darray(
                    [_safe_eval(val) if type(val) is str else val for val in data.values]
                )
            else:
                raise ValueError(
                    f"Array values supplied for parameter {key} do not match model population age structure"
                )
        elif data.type in ["sampled", "calibrated", "calculated"]:
            parameters_dict[key] = None

    try:
        model.add_parameter(parameters_dict=parameters_dict)
        logger.info(f"Added parameters: {list(parameters_dict.keys())}")

        return model
    except Exception as e:
        raise ValueError(f"Error adding parameters to model: {e}")


def _calculate_parameters_from_config(model: EpiModel, parameters: dict[str, Parameter]) -> EpiModel:
    """
    Add calculated parameters to the EpiModel, assuming all non-calculated parameters are already in the model.

    Parameters
    ----------
        model: The EpiModel instance to which calculated parameters will be added.
        parameters: Dictionary mapping parameter names to Parameter objects.

    Returns
    -------
        EpiModel instance with calculated parameters added.
    """
    # Extract parameter names and expressions
    calc_params = {name: param.value for name, param in parameters.items() if param.type == "calculated"}

    # Build a dictionary of calculated values
    parameters_dict = {}
    for name, expr in calc_params.items():
        try:
            # Parse the expression into a tree
            tree = ast.parse(expr, mode="eval")
            # Substitute retrieved parameter values or contact matrix eigenvalue into the tree,
            # convert back into expression
            calc_expr = ast.unparse(RetrieveName(model).visit(tree))
            # Evaluate the expression
            parameters_dict[name] = _safe_eval(calc_expr)
        except Exception as e:
            raise ValueError(f"Error calculating parameter {name}: {e}")

    try:
        model.add_parameter(parameters_dict=parameters_dict)
        logger.info(f"Calculated parameters: {list(parameters_dict.keys())}")

        return model
    except Exception as e:
        raise ValueError(f"Error adding parameters to model: {e}")


def _add_vaccination_schedules_from_config(
    model: EpiModel,
    transitions: list[Transition],
    vaccination: Vaccination,
    timespan: Timespan,
    use_schedule: DataFrame | None = None,
) -> EpiModel:
    """
    Add transitions between compartments due to vaccination to the EpiModel instance.

    Parameters
    ----------
        model: The EpiModel instance to which vaccination schedules will be added.
        transitions: List of Transition objects, including vaccination transitions.
        vaccination: Vaccination configuration object.
        timespan: Timespan configuration object with simulation dates.
        use_schedule: Optional pre-loaded vaccination schedule DataFrame.

    Returns
    -------
        EpiModel instance with vaccination schedules added.
    """
    import pandas as pd

    from .vaccinations import add_vaccination_schedule, make_vaccination_probability_function, scenario_to_epydemix

    # Extract compartment transitions due to vaccination
    vaccination_transitions = [transition for transition in transitions if transition.type == "vaccination"]

    # Define vaccine probability function
    vaccine_probability_function = make_vaccination_probability_function(
        vaccination.origin_compartment, vaccination.eligible_compartments
    )

    # Ignore provided data path in vaccination input if use_schedule is provided
    if use_schedule is not None:
        vaccination_schedule = use_schedule
    # Preprocessed vaccination schedule
    elif vaccination.preprocessed_vaccination_data_path:
        vaccination_schedule = pd.read_csv(vaccination.preprocessed_vaccination_data_path)
        logger.info(f"Loaded preprocessed vaccination schedule from {vaccination.preprocessed_vaccination_data_path}")
    # Create schedule from SMH scenario
    else:
        try:
            vaccination_schedule = scenario_to_epydemix(
                input_filepath=vaccination.scenario_data_path,
                start_date=timespan.start_date,
                end_date=timespan.end_date,
                target_age_groups=model.population.Nk_names,
                delta_t=timespan.delta_t,
                states=[model.population.name],
            )
            logger.info(f"Created vaccination schedule from scenario data at {vaccination.scenario_data_path}")
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


def _add_school_closure_intervention_from_config(
    model: EpiModel, interventions: list[Intervention], closure_dict: dict
) -> EpiModel:
    """
    Apply a school closure intervention to the EpiModel instance.

    Parameters
    ----------
        model: The EpiModel instance to which the intervention will be applied.
        interventions: List of Intervention objects.
        closure_dict: Dictionary containing school closure data.

    Returns
    -------
        EpiModel instance with the intervention applied.
    """
    from .school_closures import add_school_closure_interventions

    # Extract school_closure intervention
    # Validator enforces only 1 school_closure intervention, so can just take index here
    try:
        intervention = interventions[[i.type for i in interventions].index("school_closure")]
    except ValueError:  # ValueError thrown by index() if there is no school_closure intervention
        return model

    # Apply the intervention
    try:
        add_school_closure_interventions(
            model=model, closure_dict=closure_dict, reduction_factor=intervention.scaling_factor
        )
        logger.info(f"Applied school closure intervention with reduction factor: {intervention.scaling_factor}")
    except Exception as e:
        raise ValueError(f"Error applying school closure intervention {intervention}:\n{e}")

    return model


def _add_contact_matrix_interventions_from_config(model: EpiModel, interventions: list[Intervention]) -> EpiModel:
    """
    Apply contact matrix interventions.

    Parameters
    ----------
        model: The EpiModel instance to which the intervention will be applied.
        interventions: List of Intervention objects.

    Returns
    -------
        EpiModel instance with contact matrix interventions applied.
    """
    # Extract interventions
    cm_invs = [i for i in interventions if i.type == "contact_matrix"]

    for i in cm_invs:
        # Ensure layer is present
        assert i.contact_matrix_layer in model.population.layers, (
            f"Contact matrix intervention cannot use layer '{i.contact_matrix_layer}'. Available layers are {model.population.layers}."
        )

        # Apply the intervention
        try:
            model.add_intervention(
                layer_name=i.contact_matrix_layer,
                start_date=i.start_date,
                end_date=i.end_date,
                reduction_factor=i.scaling_factor,
            )
            logger.info(
                f"Applied contact matrix intervention to layer '{i.contact_matrix_layer}' with scaling factor {i.scaling_factor}."
            )
        except Exception as e:
            raise ValueError(f"Error applying contact matrix intervention {i}:\n{e}")

    return model


def _add_seasonality_from_config(model: EpiModel, seasonality: Seasonality, timespan: Timespan) -> EpiModel:
    """
    Add seasonally varying transmission rate to the EpiModel.

    Parameters
    ----------
        model: The EpiModel instance to apply seasonality to.
        seasonality: Seasonality configuration object.
        timespan: Timespan configuration object with simulation dates.

    Returns
    -------
        EpiModel instance with seasonal transmission applied.
    """
    import numpy as np

    from .seasonality import get_seasonal_transmission_balcan

    # Parameter must already be defined
    try:
        previous_value = model.get_parameter(seasonality.target_parameter)
    except KeyError:
        raise ValueError(f"Attempted to apply seasonality to undefined parameter {seasonality.target_parameter}")

    # Calculate rescaling factor with requested method
    if seasonality.method == "balcan":
        # Minimum transmission date is optional
        if seasonality.seasonality_min_date is not None:
            date_tmin = seasonality.seasonality_min_date
        else:
            date_tmin = None
        # Do the calculation
        dates, st = get_seasonal_transmission_balcan(
            date_start=timespan.start_date,
            date_stop=timespan.end_date,
            date_tmax=seasonality.seasonality_max_date,
            date_tmin=date_tmin,
            val_min=seasonality.min_value,
            val_max=seasonality.max_value,
            delta_t=timespan.delta_t,
        )
    else:
        raise ValueError(f"Undefined seasonality method recieved: {seasonality.method}")

    # Handle possibilities for previous parameter value (expressions should already be evaluated at parameter definition)
    T = len(st)
    N = model.population.num_groups
    # If existing parameter is constant, transform to array of size (T, 1) with time-varying values
    # If existing parameter is time-varying (array of size (T, 1)), do piecewise multiplication
    if (not hasattr(previous_value, "__len__")) or previous_value.shape == (T,):
        new_value = np.array(st) * np.array(previous_value)
    # If existing parameter is age-varying (array of size (1, N)), transform to array of size (T, N) with time-varying and age-varying values
    # If existing parameter is time-varying and age-varying (array of size (T, N)), do piecewise for each age group
    elif previous_value.shape == (T, N) or previous_value.shape == (1, N):
        new_value = np.zeros((T, N))
        for i in range(N):
            new_value[:, i] = np.array(st) * np.array(previous_value[:, i])
    # Uncertain how this will work for priors
    else:
        raise ValueError(
            f"Cannot apply seasonality to existing parameter {seasonality.target_parameter} = {previous_value}"
        )

    # Overwrite parameter with new seasonal values
    try:
        model.add_parameter(seasonality.target_parameter, new_value)
        logger.info(f"Added seasonality to parameter {seasonality.target_parameter}")
    except Exception as e:
        raise ValueError(f"Error adding parameters to model: {e}")

    return model


def _add_parameter_interventions_from_config(
    model: EpiModel, interventions: list[Intervention], timespan: Timespan
) -> EpiModel:
    """
    Apply parameter interventions to the EpiModel instance.

    Handles both scaling factor interventions and parameter override interventions.
    Override interventions are applied last to ensure they are the final parameter values.

    Parameters
    ----------
        model: The EpiModel instance to apply interventions to.
        interventions: List of Intervention objects.
        timespan: Timespan configuration object with simulation dates.

    Returns
    -------
        EpiModel instance with parameter interventions applied.
    """
    import numpy as np

    from .seasonality import get_scaled_parameter

    # Extract parameter interventions
    param_invs = [i for i in interventions if i.type == "parameter"]

    # Apply scaling interventions
    for i in [inv for inv in param_invs if inv.scaling_factor]:
        # Target parameter must already exist
        try:
            previous_value = model.get_parameter(i.target_parameter)
        except KeyError:
            raise ValueError(
                f"Attempted to apply scaling factor parameter intervention to undefined parameter {i.target_parameter}"
            )

        # Calculate rescaling vector
        dates, st = get_scaled_parameter(
            date_start=timespan.start_date,
            date_stop=timespan.end_date,
            scaling_start=i.start_date,
            scaling_stop=i.end_date,
            scaling_factor=i.scaling_factor,
            delta_t=timespan.delta_t,
        )

        # Handle possibilities for previous parameter value (expressions should already be evaluated at parameter definition)
        T = len(st)
        N = model.population.num_groups
        # If existing parameter is constant, transform to array of size (T, 1) with time-varying values
        # If existing parameter is time-varying (array of size (T, 1)), do piecewise multiplication
        if (not hasattr(previous_value, "__len__")) or previous_value.shape == (T,):
            new_value = np.array(st) * np.array(previous_value)
        # If existing parameter is age-varying (array of size (1, N)), transform to array of size (T, N) with time-varying and age-varying values
        # If existing parameter is time-varying and age-varying (array of size (T, N)), do piecewise for each age group
        elif previous_value.shape == (T, N) or previous_value.shape == (1, N):
            new_value = np.zeros((T, N))
            for i in range(N):
                new_value[:, i] = np.array(st) * np.array(previous_value[:, i])
        # Uncertain how this will work for priors
        else:
            raise ValueError(
                f"Cannot apply scaling intervention to existing parameter {i.target_parameter} = {previous_value}"
            )

        # Overwrite parameter with new scaled values
        try:
            model.add_parameter(i.target_parameter, new_value)
            logger.info(f"Added scaling intervention to parameter {i.target_parameter}")
        except Exception as e:
            raise ValueError(f"Error adding parameter scaling intervention to model: {e}")

    # Apply override interventions.
    # This must occur at the end to ensure override values are final parameter values.
    for i in [inv for inv in param_invs if inv.override_value]:
        try:
            model.override_parameter(
                start_date=i.start_date, end_date=i.end_date, parameter_name=i.target_parameter, value=i.override_value
            )
            logger.info(f"Added override intervention to parameter {i.target_parameter}")
        except Exception as e:
            raise ValueError(f"Error adding parameter override intervention to model: {e}")

    return model


def load_basemodel_config_from_file(path: str) -> BasemodelConfig:
    """
    Load model configuration YAML from the given path and validate against the schema.

    Parameters
    ----------
        path: The file path to the YAML configuration file.

    Returns
    -------
        The validated configuration object.
    """
    from pathlib import Path

    import yaml

    with Path(path).open() as f:
        raw = yaml.safe_load(f)

    root = validate_basemodel(raw)
    logger.info("Basemodel configuration loaded successfully.")
    return root


def load_sampling_config_from_file(path: str) -> SamplingConfig:
    """
    Load sampling configuration YAML from the given path and validate against the schema.

    Parameters
    ----------
        path: The file path to the YAML configuration file.

    Returns
    -------
        The validated configuration object.
    """
    from pathlib import Path

    import yaml

    with Path(path).open() as f:
        raw = yaml.safe_load(f)

    root = validate_sampling(raw)
    logger.info("Sampling configuration loaded successfully.")
    return root


def load_calibration_config_from_file(path: str) -> CalibrationConfig:
    """
    Load calibration configuration YAML from the given path and validate against the schema.

    Parameters
    ----------
        path: The file path to the YAML configuration file.

    Returns
    -------
        The validated configuration object.
    """
    from pathlib import Path

    import yaml

    with Path(path).open() as f:
        raw = yaml.safe_load(f)

    root = validate_calibration(raw)
    logger.info("Calibration configuration loaded successfully.")
    return root
