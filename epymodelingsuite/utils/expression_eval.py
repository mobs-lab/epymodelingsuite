"""Safe expression evaluation utilities for model parameters.

Expressions are evaluated by walking the parsed AST directly, never by ``eval``. Only numbers, lists
(evaluated as elementwise NumPy arrays), arithmetic operators, a fixed set of ``np.<name>`` functions and
constants, and names resolved by a caller-supplied function are allowed. Arbitrary attribute access
(e.g. ``np.ctypeslib``) is rejected.

Functions
---------
safe_eval : Safely evaluate a numeric expression from a string
resolve_model_name : Resolve a name in a calculated-parameter expression from an EpiModel
"""

import ast
import operator
from collections.abc import Callable

import numpy as np
from epydemix.model import EpiModel

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Functions and constants callable as np.<name>
_NUMPY_FUNCTIONS = {
    name: getattr(np, name)
    for name in (
        "abs",
        "exp",
        "log",
        "log10",
        "log2",
        "sqrt",
        "sin",
        "cos",
        "tan",
        "minimum",
        "maximum",
        "min",
        "max",
        "sum",
        "mean",
    )
}
_NUMPY_CONSTANTS = {"pi": np.pi, "e": np.e}


def safe_eval(expr: str, resolve_name: Callable[[str], float | np.ndarray] | None = None) -> float | np.ndarray:
    """
    Safely evaluate a numeric expression from a string.

    Parameters
    ----------
    expr : str
        The expression to evaluate (e.g. ``"1/10"``, ``"np.exp(-2) + 3 * np.sqrt(4)"``, ``"[1, 2] * 2"``).
    resolve_name : Callable[[str], float | np.ndarray] | None
        Function returning the value of a bare name in the expression (e.g. a model parameter).
        If None, bare names are rejected.

    Returns
    -------
    float | np.ndarray
        The value of the expression. Lists evaluate to NumPy arrays with elementwise arithmetic.

    Raises
    ------
    ValueError
        If the expression contains disallowed operations or syntax.
    SyntaxError
        If the expression has invalid Python syntax.
    """
    tree = ast.parse(expr, mode="eval")
    return _evaluate(tree.body, resolve_name)


def _evaluate(node: ast.AST, resolve_name: Callable[[str], float | np.ndarray] | None) -> float | np.ndarray:  # noqa: PLR0911
    match node:
        # bool is a subclass of int; reject it along with strings and other constants
        case ast.Constant(value=value) if type(value) in (int, float):
            # Floats keep Pow from building huge integers (e.g. 9**9**9)
            return float(value)
        case ast.List(elts=elements):
            return np.array([_evaluate(element, resolve_name) for element in elements], dtype=float)
        case ast.BinOp(left=left, op=op, right=right) if type(op) in _BINARY_OPERATORS:
            return _BINARY_OPERATORS[type(op)](_evaluate(left, resolve_name), _evaluate(right, resolve_name))
        case ast.UnaryOp(op=op, operand=operand) if type(op) in _UNARY_OPERATORS:
            return _UNARY_OPERATORS[type(op)](_evaluate(operand, resolve_name))
        case ast.Attribute(value=ast.Name(id="np"), attr=attr) if attr in _NUMPY_CONSTANTS:
            return _NUMPY_CONSTANTS[attr]
        case ast.Call(func=ast.Attribute(value=ast.Name(id="np"), attr=attr), args=args, keywords=[]) if (
            attr in _NUMPY_FUNCTIONS
        ):
            return _NUMPY_FUNCTIONS[attr](*[_evaluate(arg, resolve_name) for arg in args])
        case ast.Name(id=name) if resolve_name is not None and name != "np":
            return resolve_name(name)
    msg = f"Disallowed expression: {ast.unparse(node)}"
    raise ValueError(msg)


def resolve_model_name(
    model: EpiModel, compartment_init: dict[str, np.ndarray] | None, name: str
) -> float | np.ndarray:
    """
    Resolve a name in a calculated-parameter expression.

    Parameters
    ----------
    model : EpiModel
        Model with contact matrices and the parameters referenced by the expression.
    compartment_init : dict[str, np.ndarray] | None
        Initial conditions by compartment, needed when the expression references a compartment.
    name : str
        ``eigenvalue`` (spectral radius of the summed contact matrices), a compartment name
        (its initial proportion of the population), or a model parameter name.

    Returns
    -------
    float | np.ndarray
        The resolved value. Age-varying parameters resolve to a 1D array.
    """
    # Eigenvalue of contact matrix
    if name == "eigenvalue":
        try:
            contact_matrix = np.sum(list(model.population.contact_matrices.values()), axis=0)
            return float(np.linalg.eigvals(contact_matrix).real.max())
        except Exception as e:
            msg = f"Error calculating eigenvalue of contact matrix: {e}"
            raise ValueError(msg) from e

    # Proportion of population in compartment from initial conditions
    if name in model.compartments:
        if compartment_init is None:
            msg = f"Parameter calculation received compartment id {name} but initial conditions were not provided."
            raise ValueError(msg)
        if name not in compartment_init:
            msg = (
                f"Parameter calculation received compartment id {name} "
                "but compartment is missing from provided initial conditions."
            )
            raise ValueError(msg)
        return float(compartment_init[name].sum() / model.population.Nk.sum())

    # Model parameter
    try:
        value = model.get_parameter(name)
    except Exception as e:
        msg = f"Error obtaining parameter value during calculation: {e}"
        raise ValueError(msg) from e
    if isinstance(value, np.ndarray):
        if value.ndim > 1 and value.shape[0] != 1:
            msg = "Parameter calculation using parameters with array values is only implemented for age-varying parameters."
            raise ValueError(msg)
        return value.astype(float).flatten()
    return float(value)
