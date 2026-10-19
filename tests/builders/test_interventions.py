"""Tests for intervention builder functions."""

import datetime as dt

import numpy as np
from epydemix.model import EpiModel

from epymodelingsuite.builders.interventions import add_parameter_interventions_from_config
from epymodelingsuite.schema.basemodel import Intervention, Timespan
from epymodelingsuite.utils import load_epydemix_population
from tests.conftest import AGE_GROUP_MAPPING


class TestAddParameterInterventionsFromConfig:
    def test_age_varying_parameter_scaling(self):
        """Scaling an age-varying (1, N) parameter produces a (T, N) array scaled within the window."""
        model = EpiModel()
        model.set_population(
            load_epydemix_population(
                population_name="United_States__Massachusetts", age_group_mapping=AGE_GROUP_MAPPING
            )
        )
        num_groups = model.population.num_groups
        age_values = np.arange(1, num_groups + 1, dtype=float).reshape(1, num_groups) * 0.1
        model.add_parameter("beta", age_values)

        timespan = Timespan(start_date=dt.date(2025, 1, 1), end_date=dt.date(2025, 1, 10), delta_t=1.0)
        interventions = [
            Intervention(
                type="parameter",
                target_parameter="beta",
                scaling_factor=0.5,
                start_date=dt.date(2025, 1, 5),
                end_date=dt.date(2025, 1, 10),
            )
        ]

        add_parameter_interventions_from_config(model, interventions, timespan)

        new_value = model.get_parameter("beta")
        assert new_value.shape[1] == num_groups
        np.testing.assert_allclose(new_value[0], age_values[0])
        np.testing.assert_allclose(new_value[-1], age_values[0] * 0.5)


def _apply(**intervention_kwargs) -> EpiModel:
    model = EpiModel()
    model.add_parameter(parameters_dict={"beta": 0.2})
    intervention = Intervention(
        type="parameter",
        target_parameter="beta",
        start_date=dt.date(2025, 1, 5),
        end_date=dt.date(2025, 1, 10),
        **intervention_kwargs,
    )
    timespan = Timespan(start_date=dt.date(2025, 1, 1), end_date=dt.date(2025, 1, 20), delta_t=1.0)
    return add_parameter_interventions_from_config(model, [intervention], timespan)


def test_zero_scaling_factor_is_applied():
    beta = np.asarray(_apply(scaling_factor=0).get_parameter("beta"))
    assert beta.min() == 0
    assert beta.max() == 0.2


def test_zero_override_value_is_applied():
    overrides = _apply(override_value=0).overrides["beta"]
    assert len(overrides) == 1
    assert overrides[0]["value"] == 0
