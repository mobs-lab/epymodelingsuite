"""Tests for calibration comparison formatting."""

from datetime import date
from unittest.mock import Mock

import numpy as np
import pytest

from epymodelingsuite.builders.comparison import format_comparison_data
from epymodelingsuite.schema.calibration import ComparisonSpec

RANDOM_STATE = {"bit_generator": "PCG64", "state": {"state": 12345, "inc": 67890}, "has_uint32": 0, "uinteger": 0}


def _spec(simulation, name=None, column="value"):
    return ComparisonSpec(name=name, observed_value_column=column, observed_date_column="date", simulation=simulation)


def _results(dates, transitions):
    results = Mock()
    results.dates = dates
    results.transitions = transitions
    return results


class TestFormatComparisonData:
    """Tests for format_comparison_data."""

    def test_aligns_simulation_to_observed_dates(self):
        """Transitions are summed and only the observed dates are kept."""
        results = _results(
            [date(2024, 1, d) for d in range(1, 6)],
            {"Hosp_vax": np.array([10, 20, 30, 40, 50]), "Hosp_unvax": np.array([5, 10, 15, 20, 25])},
        )
        data_dates = [date(2024, 1, 2), date(2024, 1, 4)]

        result = format_comparison_data(results, [_spec(["Hosp_vax", "Hosp_unvax"])], data_dates, RANDOM_STATE)

        assert set(result) == {"data", "date", "random_state"}
        np.testing.assert_array_equal(result["data"], [30, 60])
        assert result["date"] == data_dates
        assert result["random_state"] == RANDOM_STATE

    def test_pads_with_nan_when_simulation_starts_later(self):
        """Observed dates before the simulation start become NaN."""
        results = _results([date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)], {"Hosp": np.array([30, 40, 50])})
        data_dates = [date(2024, 1, d) for d in range(1, 6)]

        result = format_comparison_data(results, [_spec(["Hosp"])], data_dates, RANDOM_STATE)

        np.testing.assert_array_equal(result["data"], [np.nan, np.nan, 30, 40, 50])

    def test_handles_multiple_transitions(self):
        """All listed transitions are summed."""
        transitions = {
            "Hosp_vax_0_4": np.array([1, 2]),
            "Hosp_vax_5_17": np.array([3, 4]),
            "Hosp_unvax_0_4": np.array([5, 6]),
            "Hosp_unvax_5_17": np.array([7, 8]),
        }
        data_dates = [date(2024, 1, 1), date(2024, 1, 2)]

        result = format_comparison_data(
            _results(data_dates, transitions), [_spec(list(transitions))], data_dates, RANDOM_STATE
        )

        np.testing.assert_array_equal(result["data"], [16, 20])

    def test_lists_missing_dates_from_actual_start(self):
        # The first resampled date can be later than the actual simulation start.
        observations = [date(2024, 1, d) for d in range(1, 7)]
        results = _results([observations[2], observations[4]], {"Hosp": np.array([10, 20])})
        with pytest.raises(ValueError) as error:
            format_comparison_data(results, [_spec(["Hosp"])], observations, RANDOM_STATE, start_date=observations[1])
        assert "2024-01-01" not in str(error.value)
        for day in ("2024-01-02", "2024-01-04", "2024-01-06"):
            assert day in str(error.value)

    def test_multiple_comparisons_add_named_series(self):
        """Multiple comparisons keep the first series as "data" and add every series by name."""
        data_dates = [date(2024, 1, 1), date(2024, 1, 2)]
        results = _results(data_dates, {"ED": np.array([60, 70]), "Hosp": np.array([10, 12])})
        comparisons = [_spec(["ED"], name="ed", column="ed_count"), _spec(["Hosp"], column="hosp_count")]

        result = format_comparison_data(results, comparisons, data_dates, RANDOM_STATE)

        assert set(result) == {"data", "date", "random_state", "ed", "hosp_count"}
        np.testing.assert_array_equal(result["data"], [60, 70])
        np.testing.assert_array_equal(result["ed"], [60, 70])
        np.testing.assert_array_equal(result["hosp_count"], [10, 12])

    def test_failed_simulation_returns_zeros_without_date(self):
        """A failed simulation returns zero-filled series and no "date", single or multiple."""
        data_dates = [date(2024, 1, 1), date(2024, 1, 2)]

        single = format_comparison_data(None, [_spec(["Hosp"])], data_dates, RANDOM_STATE)
        multiple = format_comparison_data(None, [_spec(["A"], name="a"), _spec(["B"], name="b")], data_dates, {})

        assert set(single) == {"data"}
        assert set(multiple) == {"data", "a", "b"}
        for values in [*single.values(), *multiple.values()]:
            np.testing.assert_array_equal(values, [0, 0])
