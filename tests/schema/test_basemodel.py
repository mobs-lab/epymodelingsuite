"""Tests for basemodel schema validation."""

import logging
from datetime import date

import pytest

from epymodelingsuite.schema.basemodel import (
    BaseEpiModel,
    Compartment,
    Parameter,
    Population,
    Simulation,
    Timespan,
    Transition,
)


@pytest.fixture
def base_model_params():
    """Create minimal base model parameters for testing."""
    return {
        "population": Population(age_groups=["0-17", "18-64", "65+"]),
        "compartments": [
            Compartment(id="S", init="default"),
            Compartment(id="I", init=100),
            Compartment(id="R"),
        ],
        "transitions": [
            Transition(type="mediated", source="S", target="I", rate="beta", mediator="I"),
            Transition(type="spontaneous", source="I", target="R", rate=0.1),
        ],
        "parameters": {
            "beta": Parameter(type="scalar", value=0.3),
        },
    }


class TestWeeklyFrequencyEndDateValidation:
    """Test validation of weekly resample_frequency against end_date."""

    def test_wsat_with_saturday_end_date_valid(self, base_model_params):
        """Test that W-SAT with Saturday end_date is valid."""
        # 2024-01-06 is a Saturday
        model = BaseEpiModel(
            timespan=Timespan(start_date=date(2024, 1, 1), end_date=date(2024, 1, 6)),
            simulation=Simulation(n_sims=10, resample_frequency="W-SAT"),
            **base_model_params,
        )
        assert model.timespan.end_date.weekday() == 5

    def test_wsat_with_non_saturday_end_date_warns(self, base_model_params, caplog):
        """Test that W-SAT with non-Saturday end_date logs a warning."""
        caplog.set_level(logging.WARNING)
        # 2024-01-07 is a Sunday
        model = BaseEpiModel(
            timespan=Timespan(start_date=date(2024, 1, 1), end_date=date(2024, 1, 7)),
            simulation=Simulation(n_sims=10, resample_frequency="W-SAT"),
            **base_model_params,
        )
        assert model is not None
        assert "Saturday is preferred" in caplog.text
        assert "Sunday" in caplog.text

    def test_wsat_with_friday_end_date_shows_actual_day(self, base_model_params, caplog):
        """Test that W-SAT with Friday end_date shows actual day in warning."""
        caplog.set_level(logging.WARNING)
        # 2024-01-05 is a Friday
        model = BaseEpiModel(
            timespan=Timespan(start_date=date(2024, 1, 1), end_date=date(2024, 1, 5)),
            simulation=Simulation(n_sims=10, resample_frequency="W-SAT"),
            **base_model_params,
        )
        assert model is not None
        assert "Friday" in caplog.text

    def test_wsun_with_sunday_end_date_valid(self, base_model_params):
        """Test that W-SUN with Sunday end_date is valid."""
        # 2024-01-07 is a Sunday
        model = BaseEpiModel(
            timespan=Timespan(start_date=date(2024, 1, 1), end_date=date(2024, 1, 7)),
            simulation=Simulation(n_sims=10, resample_frequency="W-SUN"),
            **base_model_params,
        )
        assert model.timespan.end_date.weekday() == 6

    def test_wsun_with_non_sunday_end_date_warns(self, base_model_params, caplog):
        """Test that W-SUN with non-Sunday end_date logs a warning."""
        caplog.set_level(logging.WARNING)
        # 2024-01-06 is a Saturday
        model = BaseEpiModel(
            timespan=Timespan(start_date=date(2024, 1, 1), end_date=date(2024, 1, 6)),
            simulation=Simulation(n_sims=10, resample_frequency="W-SUN"),
            **base_model_params,
        )
        assert model is not None
        assert "Sunday is preferred" in caplog.text

    def test_wmon_with_monday_end_date_valid(self, base_model_params):
        """Test that W-MON with Monday end_date is valid."""
        # 2024-01-08 is a Monday
        model = BaseEpiModel(
            timespan=Timespan(start_date=date(2024, 1, 1), end_date=date(2024, 1, 8)),
            simulation=Simulation(n_sims=10, resample_frequency="W-MON"),
            **base_model_params,
        )
        assert model.timespan.end_date.weekday() == 0

    def test_wmon_with_non_monday_end_date_warns(self, base_model_params, caplog):
        """Test that W-MON with non-Monday end_date logs a warning."""
        caplog.set_level(logging.WARNING)
        # 2024-01-07 is a Sunday
        model = BaseEpiModel(
            timespan=Timespan(start_date=date(2024, 1, 1), end_date=date(2024, 1, 7)),
            simulation=Simulation(n_sims=10, resample_frequency="W-MON"),
            **base_model_params,
        )
        assert model is not None
        assert "Monday is preferred" in caplog.text

    def test_non_weekly_frequency_allows_any_day(self, base_model_params):
        """Test that non-weekly frequency allows any end_date."""
        # 2024-01-07 is a Sunday - should be valid with daily frequency
        model = BaseEpiModel(
            timespan=Timespan(start_date=date(2024, 1, 1), end_date=date(2024, 1, 7)),
            simulation=Simulation(n_sims=10, resample_frequency="D"),
            **base_model_params,
        )
        assert model is not None

    def test_no_simulation_allows_any_day(self, base_model_params):
        """Test that missing simulation field allows any end_date."""
        # 2024-01-07 is a Sunday - should be valid without simulation
        model = BaseEpiModel(
            timespan=Timespan(start_date=date(2024, 1, 1), end_date=date(2024, 1, 7)),
            simulation=None,
            **base_model_params,
        )
        assert model is not None

    def test_no_resample_frequency_allows_any_day(self, base_model_params):
        """Test that None resample_frequency allows any end_date."""
        # 2024-01-07 is a Sunday
        model = BaseEpiModel(
            timespan=Timespan(start_date=date(2024, 1, 1), end_date=date(2024, 1, 7)),
            simulation=Simulation(n_sims=10, resample_frequency=None),
            **base_model_params,
        )
        assert model is not None

    def test_unrecognized_weekly_frequency_skipped(self, base_model_params):
        """Test that unrecognized W-XXX frequency is skipped (no validation)."""
        # W-XYZ is not a valid day, so validation is skipped
        model = BaseEpiModel(
            timespan=Timespan(start_date=date(2024, 1, 1), end_date=date(2024, 1, 7)),
            simulation=Simulation(n_sims=10, resample_frequency="W-XYZ"),
            **base_model_params,
        )
        assert model is not None


class TestDeltaTValidation:
    """Test validation of delta_t in Timespan schema."""

    def test_delta_t_rejects_zero(self):
        """Test that delta_t=0 raises ValidationError."""
        with pytest.raises(Exception, match="delta_t"):
            Timespan(start_date=date(2024, 1, 1), end_date=date(2024, 1, 31), delta_t=0)

    def test_delta_t_rejects_negative(self):
        """Test that delta_t=-1 raises ValidationError."""
        with pytest.raises(Exception, match="delta_t"):
            Timespan(start_date=date(2024, 1, 1), end_date=date(2024, 1, 31), delta_t=-1)

    @pytest.mark.parametrize("delta_t", [0.5, 0.25, 0.1])
    def test_delta_t_accepts_subdaily(self, delta_t):
        """Test that subdaily delta_t values are accepted."""
        t = Timespan(start_date=date(2024, 1, 1), end_date=date(2024, 1, 31), delta_t=delta_t)
        assert t.delta_t == delta_t

    def test_delta_t_coerces_int_to_float(self):
        """Test that integer delta_t is coerced to float."""
        t = Timespan(start_date=date(2024, 1, 1), end_date=date(2024, 1, 31), delta_t=2)
        assert type(t.delta_t) is float
        assert t.delta_t == 2.0

    def test_delta_t_defaults_to_one(self):
        """Test that delta_t defaults to 1.0 when omitted."""
        t = Timespan(start_date=date(2024, 1, 1), end_date=date(2024, 1, 31))
        assert t.delta_t == 1.0
