"""Core model building functions for EpiModel instances."""

import ast
import logging
import operator
from typing import Any

import numpy as np
import scipy
from epydemix.model import EpiModel
from epydemix.population import load_epydemix_population
from epydemix.utils import convert_to_2Darray

from ..schema.basemodel import Compartment, Parameter, Transition
from ..utils import convert_location_name_format

logger = logging.getLogger(__name__)

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
    """
    A NodeVisitor that only allows numeric, numpy, and scipy expressions,
    and enables binary operations on numpy arrays.
    """

    def visit(self, node):
        t = type(node)
        # Permit only these node types
        if t in (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Constant,
            ast.List,
            ast.Load,
            ast.Name,
            ast.Attribute,
            ast.Call,
        ):
            return super().visit(node)
        raise ValueError(f"Disallowed expression: {t.__name__}")

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)

        if type(node.op) not in _allowed_operators:
            raise ValueError(f"Operator {type(node.op).__name__} not allowed")

        # Access node data, handle arrays
        if isinstance(left, ast.Constant) and isinstance(right, ast.Constant):
            left = left.value
            right = right.value
        elif isinstance(left, ast.Constant) and isinstance(right, ast.List):
            left = np.array([left.value])
            right = np.array([c.value for c in right.elts], dtype=float)
        elif isinstance(left, ast.List) and isinstance(right, ast.Constant):
            left = np.array([c.value for c in left.elts], dtype=float)
            right = np.array([right.value])
        elif isinstance(left, ast.List) and isinstance(right, ast.List):
            left = np.array([c.value for c in left.elts], dtype=float)
            right = np.array([c.value for c in right.elts], dtype=float)
        else:
            raise ValueError(f"Attempted BinOp on unsupported objects:\n\n{left}\n\n{right}")

        # Perform calculation
        calc_val = None
        if isinstance(node.op, ast.Add):
            calc_val = np.add(left, right, dtype=float)
        elif isinstance(node.op, ast.Sub):
            calc_val = np.subtract(left, right, dtype=float)
        elif isinstance(node.op, ast.Mult):
            calc_val = np.multiply(left, right, dtype=float)
        elif isinstance(node.op, ast.Div):
            calc_val = np.divide(left, right, dtype=float)

        if isinstance(calc_val, np.ndarray):
            ast_nodes = [ast.Constant(value=item) for item in calc_val.flatten()]
            ast_list = ast.List(elts=ast_nodes, ctx=ast.Load())
            return ast.fix_missing_locations(ast_list)
        return ast.fix_missing_locations(ast.Constant(value=calc_val))

    def visit_UnaryOp(self, node):
        self.visit(node.operand)
        if type(node.op) not in _allowed_unary_operators:
            raise ValueError(f"Unary operator {type(node.op).__name__} not allowed")

    def visit_Constant(self, node):
        # Only allow numeric constants
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"Constant of type {type(node.value).__name__} not allowed")
        return node

    def visit_List(self, node):
        for v in node.elts:
            self.visit(v)
        return node

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
                    return ast.fix_missing_locations(ast.Constant(value=float(eigenvalue)))
                except Exception as e:
                    raise ValueError(f"Error calculating eigenvalue of contact matrix: {e}")
            else:
                try:
                    value = self.model.get_parameter(node.id)
                    if type(value) == np.ndarray:
                        assert value.shape[0] == 1, (
                            "Parameter calculation using parameters with array values is only implemented for age-varying parameters."
                        )
                        ast_nodes = [ast.Constant(value=float(item)) for item in value.flatten()]
                        return ast.fix_missing_locations(ast.List(elts=ast_nodes, ctx=ast.Load()))
                    return ast.fix_missing_locations(ast.Constant(value=float(value)))
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


def set_population_from_config(model: EpiModel, population_name: str, age_groups: list[str]) -> EpiModel:
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


def add_model_compartments_from_config(model: EpiModel, compartments: list[Compartment]) -> EpiModel:
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


def add_model_transitions_from_config(model: EpiModel, transitions: list[Transition]) -> EpiModel:
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


def add_model_parameters_from_config(model: EpiModel, parameters: dict[str, Parameter]) -> EpiModel:
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
            # Skip parameters without values.
            # They will be set later during calibration/sampling or calculated after all other parameters are defined
            pass

    try:
        model.add_parameter(parameters_dict=parameters_dict)
        logger.info(f"Added parameters: {list(parameters_dict.keys())}")

        return model
    except Exception as e:
        raise ValueError(f"Error adding parameters to model: {e}")


def calculate_parameters_from_config(model: EpiModel, parameters: dict[str, Parameter]) -> EpiModel:
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
        logger.info(f"Calculating parameter {name} using expression: {expr}")
        try:
            # Parse the expression into a tree
            tree = ast.parse(expr, mode="eval")

            # Substitute retrieved parameter values or contact matrix eigenvalue into the tree
            RetrieveName(model).visit(tree)

            # Validate the expression
            SafeEvalVisitor().visit(tree)

            # Evaluate the expression
            code = compile(tree, filename="<calc_eval>", mode="eval")
            parameters_dict[name] = eval(code, {"__builtins__": None, "np": np, "scipy": scipy}, {})
            logger.info(f"Calculated parameter {name}: {parameters_dict[name]}")
        except Exception as e:
            raise ValueError(f"Error calculating parameter {name}: {e}")

    try:
        model.add_parameter(parameters_dict=parameters_dict)
        logger.info(f"Calculated parameters: {list(parameters_dict.keys())}")

        return model
    except Exception as e:
        raise ValueError(f"Error adding parameters to model: {e}")


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
        List of Compartment objects from the configuration.
    population_array : np.ndarray
        Array of population counts by age group.
    sampled_compartments : dict | None, optional
        Dictionary of compartment names to sampled values (overrides config values).

    Returns
    -------
    dict | None
        Dictionary mapping compartment names to initial condition arrays,
        or None if no initial conditions are specified.

    Raises
    ------
    ValueError
        If multiple compartments are marked as default, or if initialization
        logic produces invalid values.

    Examples
    --------
    >>> compartments = [
    ...     Compartment(id="S", initial=None, default=True),
    ...     Compartment(id="I", initial=100, default=False),
    ...     Compartment(id="R", initial=0.1, default=False)
    ... ]
    >>> population = np.array([1000, 2000, 3000])
    >>> calculate_compartment_initial_conditions(compartments, population)
    {'S': array([...]), 'I': array([...]), 'R': array([...])}
    """
    # Initialize tracking variables
    default_compartment_id = None
    initial_conditions_dict = {}
    remaining_population = population_array.copy()

    # First pass: identify default compartment and validate
    for compartment in compartments:
        if compartment.default:
            if default_compartment_id is not None:
                raise ValueError("Multiple compartments cannot be marked as default")
            default_compartment_id = compartment.id

    # Second pass: calculate non-default compartment initial conditions
    for compartment in compartments:
        # Skip default compartment for now
        if compartment.default:
            continue

        # Get initialization value (from samples if provided, otherwise from config)
        initial_value = (
            sampled_compartments.get(compartment.id, compartment.initial)
            if sampled_compartments
            else compartment.initial
        )

        # Skip compartments with no initial value
        if initial_value is None:
            continue

        # Case 1: Count-based initialization (value >= 1)
        if initial_value >= 1:
            # Distribute total count proportionally across age groups
            compartment_total = int(initial_value)
            proportions = population_array / population_array.sum()
            initial_conditions = (proportions * compartment_total).astype(int)

            # Adjust for rounding errors to ensure exact total
            difference = compartment_total - initial_conditions.sum()
            if difference != 0:
                # Add/subtract difference to the largest age group
                max_idx = np.argmax(population_array)
                initial_conditions[max_idx] += difference

            initial_conditions_dict[compartment.id] = initial_conditions
            remaining_population -= initial_conditions

        # Case 2: Proportion-based initialization (0 < value < 1)
        elif 0 < initial_value < 1:
            # Apply proportion to each age group
            initial_conditions = (population_array * initial_value).astype(int)
            initial_conditions_dict[compartment.id] = initial_conditions
            remaining_population -= initial_conditions

        # Case 3: Zero initialization
        elif initial_value == 0:
            initial_conditions_dict[compartment.id] = np.zeros_like(population_array, dtype=int)

        else:
            raise ValueError(f"Invalid initial value for compartment {compartment.id}: {initial_value}")

    # Third pass: assign remaining population to default compartment
    if default_compartment_id:
        if np.any(remaining_population < 0):
            raise ValueError(
                f"Initial conditions exceed population in some age groups. "
                f"Remaining population: {remaining_population}"
            )
        initial_conditions_dict[default_compartment_id] = remaining_population.astype(int)

    return initial_conditions_dict if initial_conditions_dict else None
