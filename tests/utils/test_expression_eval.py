"""Tests for the safe expression evaluator used by calculated parameters."""

import ast
from unittest.mock import Mock

import numpy as np

from epymodelingsuite.utils.expression_eval import RetrieveName, SafeEvalVisitor


def _evaluate(expr: str, param_values: dict[str, float]) -> float:
    """Mirror calculate_parameters_from_config's evaluation path for a scalar expr."""
    model = Mock()
    model.compartments = {}
    model.get_parameter = Mock(side_effect=lambda name: param_values[name])

    tree = ast.parse(expr, mode="eval")
    RetrieveName(model, compartment_init=None).visit(tree)
    SafeEvalVisitor().visit(tree)
    code = compile(tree, filename="<test>", mode="eval")
    return eval(code, {"__builtins__": None, "np": np}, {})


def test_np_call_in_expression_builds():
    """A calculated parameter using np.xxx(...) must survive substitution.

    Regression for RetrieveName.visit_Name dropping the 'np' node (NodeTransformer
    returning None), which corrupted np.exp(...) and raised at build time. This is
    the shape of the covid beta formula: Reff*(1-np.exp(-mu*delta_t))/...
    """
    mu, delta_t = 0.5, 1.0
    result = _evaluate("1 - np.exp(-mu*delta_t)", {"mu": mu, "delta_t": delta_t})
    assert np.isclose(result, 1 - np.exp(-mu * delta_t))


def test_beta_style_expression_builds():
    """The full covid-style beta expression evaluates to the expected scalar."""
    params = {"Reff": 1.2, "mu": 0.5, "delta_t": 1.0, "eig": 15.0, "R": 0.3}
    expr = "Reff*(1-np.exp(-mu*delta_t))/(eig*delta_t*(1-R))"
    result = _evaluate(expr, params)
    expected = (
        params["Reff"]
        * (1 - np.exp(-params["mu"] * params["delta_t"]))
        / (params["eig"] * params["delta_t"] * (1 - params["R"]))
    )
    assert np.isclose(result, expected)


def test_plain_arithmetic_still_builds():
    """Non-np calculated expressions are unaffected by the np-preservation guard."""
    result = _evaluate("1/(omega_months*days_per_month)", {"omega_months": 5, "days_per_month": 30})
    assert np.isclose(result, 1 / (5 * 30))
