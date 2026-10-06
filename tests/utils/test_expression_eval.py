"""Tests for safe expression evaluation."""

from __future__ import annotations

import numpy as np
import pytest
from epydemix.model import EpiModel

from epymodelingsuite.builders.base import calculate_parameters_from_config
from epymodelingsuite.schema.basemodel import Parameter
from epymodelingsuite.utils.expression_eval import safe_eval


class TestSafeEval:
    @pytest.mark.parametrize(
        ("expr", "expected"),
        [
            ("1/10", 0.1),
            ("-2 + 3", 1.0),
            ("2 ** 3 % 5", 3.0),
            ("np.exp(-2) + 3 * np.sqrt(4)", np.exp(-2) + 6),
            ("np.pi / 2", np.pi / 2),
        ],
    )
    def test_numeric_expressions(self, expr, expected):
        assert np.isclose(safe_eval(expr), expected)

    def test_lists_use_elementwise_arithmetic(self):
        np.testing.assert_allclose(safe_eval("[1, 2] + [3, 4]"), [4.0, 6.0])
        np.testing.assert_allclose(safe_eval("-[1, 2] * 2"), [-2.0, -4.0])

    @pytest.mark.parametrize(
        "expr",
        [
            "np.ctypeslib.ctypes._os.getpid()",
            "np.ctypeslib",
            "np.load('x')",
            "scipy.special.expit(1)",
            "__import__('os')",
            "(1).__class__",
            "'a'",
            "True + 1",
            "np.exp(x=1)",
            "[x for x in [1]]",
            "beta * 2",
        ],
    )
    def test_rejects_disallowed_expressions(self, expr):
        with pytest.raises(ValueError, match="Disallowed expression"):
            safe_eval(expr)

    def test_resolves_names(self):
        values = {"beta": 0.3, "gamma": 0.1}
        assert np.isclose(safe_eval("beta / gamma", values.__getitem__), 3.0)

    def test_np_is_not_a_name(self):
        with pytest.raises(ValueError, match="Disallowed expression"):
            safe_eval("np", {"np": 1.0}.__getitem__)


class TestCalculateParameters:
    @pytest.fixture
    def model(self):
        model = EpiModel()
        model.add_parameter(parameters_dict={"beta": 0.3, "gamma": 0.1, "susceptibility": np.array([[1.0, 2.0]])})
        return model

    def test_scalar(self, model):
        parameters = {"R0": Parameter(type="calculated", value="beta / gamma")}
        calculate_parameters_from_config(model, parameters, None)
        assert np.isclose(model.get_parameter("R0"), 3.0)

    @pytest.mark.parametrize(
        ("expr", "values", "expected"),
        [
            ("1 - np.exp(-mu*delta_t)", {"mu": 0.5, "delta_t": 1.0}, 1 - np.exp(-0.5)),
            (
                "Reff*(1-np.exp(-mu*delta_t))/(eig*delta_t*(1-R))",
                {"Reff": 1.2, "mu": 0.5, "delta_t": 1.0, "eig": 15.0, "R": 0.3},
                1.2 * (1 - np.exp(-0.5)) / (15.0 * 0.7),
            ),
            ("1/(omega_months*days_per_month)", {"omega_months": 5, "days_per_month": 30}, 1 / 150),
        ],
        ids=["np_call", "beta_expression", "plain_arithmetic"],
    )
    def test_calculated_expression_regressions(self, model, expr, values, expected):
        """Cover the parameter-substitution expressions reported in PR #290."""
        model.add_parameter(parameters_dict=values)
        parameters = {"result": Parameter(type="calculated", value=expr)}
        calculate_parameters_from_config(model, parameters, None)
        assert np.isclose(model.get_parameter("result"), expected)

    def test_age_varying(self, model):
        parameters = {"scaled": Parameter(type="calculated", value="susceptibility / (1 - 0.5)")}
        calculate_parameters_from_config(model, parameters, None)
        np.testing.assert_allclose(model.get_parameter("scaled"), [[2.0, 4.0]])
