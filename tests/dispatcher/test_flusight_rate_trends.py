"""Tests for FluSight rate trend functions in epymodelingsuite.dispatcher.output module."""

from datetime import date

import numpy as np
import pytest

from epymodelingsuite.dispatcher.output import (
    categorize_rate_change_flusightforecast,
    compare_thresholds_flusightforecast,
    get_projected_value,
    make_rate_trends_flusightforecast,
)


class TestCompareThresholdsFlusightforecast:
    """Tests for compare_thresholds_flusightforecast function."""

    def test_stable_below_threshold(self):
        """Rate change below stable threshold returns stable."""
        result = compare_thresholds_flusightforecast(0.3, 1.7, 0.1, 50)
        assert result == "stable"

    def test_stable_count_below_10(self):
        """Count change below 10 returns stable regardless of rate."""
        result = compare_thresholds_flusightforecast(0.3, 1.7, 5.0, 5)
        assert result == "stable"

    def test_increase_between_thresholds(self):
        """Positive rate change between thresholds returns increase."""
        result = compare_thresholds_flusightforecast(0.3, 1.7, 0.5, 50)
        assert result == "increase"

    def test_large_increase_at_threshold(self):
        """Rate change at large threshold returns large_increase."""
        result = compare_thresholds_flusightforecast(0.3, 1.7, 1.7, 50)
        assert result == "large_increase"

    def test_decrease_between_thresholds(self):
        """Negative rate change between thresholds returns decrease."""
        result = compare_thresholds_flusightforecast(0.3, 1.7, -0.5, 50)
        assert result == "decrease"

    def test_large_decrease_at_threshold(self):
        """Rate change at negative large threshold returns large_decrease."""
        result = compare_thresholds_flusightforecast(0.3, 1.7, -1.7, 50)
        assert result == "large_decrease"


class TestCategorizeRateChangeFlusightforecast:
    """Tests for categorize_rate_change_flusightforecast function.

    FluSight thresholds per 100k population by horizon:
    - Horizon 0: stable < 0.3, large >= 1.7
    - Horizon 1: stable < 0.5, large >= 3.0
    - Horizon 2: stable < 0.7, large >= 4.0
    - Horizon 3: stable < 1.0, large >= 5.0
    """

    @pytest.mark.parametrize(
        ("rate_change", "count_change", "expected"),
        [
            (0.1, 50, "stable"),
            (0.5, 50, "increase"),
            (1.7, 50, "large_increase"),
            (-0.5, 50, "decrease"),
            (-1.7, 50, "large_decrease"),
            (5.0, 5, "stable"),  # count < 10 override
        ],
    )
    def test_horizon_0(self, rate_change, count_change, expected):
        """Test horizon 0 categorization."""
        result = categorize_rate_change_flusightforecast(rate_change, count_change, 0, 100000)
        assert result == expected

    @pytest.mark.parametrize(
        ("rate_change", "count_change", "expected"),
        [
            (0.3, 50, "stable"),
            (1.5, 50, "increase"),
            (3.0, 50, "large_increase"),
            (-1.5, 50, "decrease"),
            (-3.0, 50, "large_decrease"),
        ],
    )
    def test_horizon_1(self, rate_change, count_change, expected):
        """Test horizon 1 categorization."""
        result = categorize_rate_change_flusightforecast(rate_change, count_change, 1, 100000)
        assert result == expected

    @pytest.mark.parametrize(
        ("rate_change", "count_change", "expected"),
        [
            (0.5, 50, "stable"),
            (2.0, 50, "increase"),
            (4.0, 50, "large_increase"),
            (-2.0, 50, "decrease"),
            (-4.0, 50, "large_decrease"),
        ],
    )
    def test_horizon_2(self, rate_change, count_change, expected):
        """Test horizon 2 categorization."""
        result = categorize_rate_change_flusightforecast(rate_change, count_change, 2, 100000)
        assert result == expected

    @pytest.mark.parametrize(
        ("rate_change", "count_change", "expected"),
        [
            (0.8, 50, "stable"),
            (3.0, 50, "increase"),
            (5.0, 50, "large_increase"),
            (-3.0, 50, "decrease"),
            (-5.0, 50, "large_decrease"),
        ],
    )
    def test_horizon_3(self, rate_change, count_change, expected):
        """Test horizon 3 categorization."""
        result = categorize_rate_change_flusightforecast(rate_change, count_change, 3, 100000)
        assert result == expected

    def test_invalid_horizon_raises(self):
        """Invalid horizon raises ValueError."""
        with pytest.raises(ValueError, match="invalid horizon"):
            categorize_rate_change_flusightforecast(1.0, 50, 4, 100000)

    @pytest.mark.parametrize(
        ("rate_changes", "expected"),
        [
            # Thresholds: stable < [0.3, 0.5, 0.7, 1.0], large >= [1.7, 3.0, 4.0, 5.0]
            # Different rate change at each horizon (h0, h1, h2, h3)
            ([0.2, 0.4, 0.6, 0.8], ["stable", "stable", "stable", "stable"]),
            ([0.4, 0.6, 0.8, 1.5], ["increase", "increase", "increase", "increase"]),
            ([2.0, 3.5, 4.5, 6.0], ["large_increase", "large_increase", "large_increase", "large_increase"]),
            ([-0.4, -0.6, -0.8, -1.5], ["decrease", "decrease", "decrease", "decrease"]),
            ([-2.0, -3.5, -4.5, -6.0], ["large_decrease", "large_decrease", "large_decrease", "large_decrease"]),
        ],
    )
    def test_different_rate_changes_per_horizon(self, rate_changes, expected):
        """Test different rate changes at each horizon."""
        count_change = 50
        for horizon, (rate_change, exp) in enumerate(zip(rate_changes, expected, strict=True)):
            result = categorize_rate_change_flusightforecast(rate_change, count_change, horizon, 100000)
            assert result == exp, f"Horizon {horizon}: rate_change={rate_change}, expected={exp}, got={result}"


class TestGetProjectedValue:
    """Tests for get_projected_value function."""

    def test_retrieves_correct_value(self):
        """Retrieves the value at the target date."""
        import pandas as pd

        dates = np.array(
            [
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-08"),
                pd.Timestamp("2024-01-15"),
            ]
        )
        values = np.array([100, 150, 200])

        assert get_projected_value(dates, values, date(2024, 1, 1)) == 100
        assert get_projected_value(dates, values, date(2024, 1, 8)) == 150
        assert get_projected_value(dates, values, date(2024, 1, 15)) == 200

    def test_mismatched_lengths_raises(self):
        """Mismatched array lengths raises AssertionError."""
        import pandas as pd

        dates = np.array([pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-08")])
        values = np.array([100, 150, 200])

        with pytest.raises(AssertionError, match="must match"):
            get_projected_value(dates, values, date(2024, 1, 1))


class TestMakeRateTrendsFlusightforecast:
    """Tests for make_rate_trends_flusightforecast function."""

    @pytest.fixture
    def basic_setup(self):
        """Create basic test data."""
        import pandas as pd

        reference_date = date(2024, 1, 15)
        population = 1_000_000

        observed = pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-08")],
                "value": [100],
            }
        )

        proj_dates = np.array(
            [
                pd.Timestamp("2024-01-15"),
                pd.Timestamp("2024-01-22"),
                pd.Timestamp("2024-01-29"),
                pd.Timestamp("2024-02-05"),
            ]
        )

        return {
            "reference_date": reference_date,
            "population": population,
            "observed": observed,
            "proj_dates": proj_dates,
        }

    def test_output_structure(self, basic_setup):
        """Output has correct columns."""
        proj_dates = np.array([basic_setup["proj_dates"]])
        proj_values = np.array([[100, 100, 100, 100]])

        result = make_rate_trends_flusightforecast(
            reference_date=basic_setup["reference_date"],
            proj_dates=proj_dates,
            proj_values=proj_values,
            observed=basic_setup["observed"],
            population=basic_setup["population"],
        )

        assert "horizon" in result.columns
        assert "target_end_date" in result.columns
        assert "output_type_id" in result.columns
        assert "value" in result.columns

    def test_all_horizons_present(self, basic_setup):
        """Output includes all horizons 0-3."""
        proj_dates = np.array([basic_setup["proj_dates"]])
        proj_values = np.array([[100, 100, 100, 100]])

        result = make_rate_trends_flusightforecast(
            reference_date=basic_setup["reference_date"],
            proj_dates=proj_dates,
            proj_values=proj_values,
            observed=basic_setup["observed"],
            population=basic_setup["population"],
        )

        assert set(result["horizon"].unique()) == {0, 1, 2, 3}

    def test_probabilities_sum_to_one(self, basic_setup):
        """PMF probabilities sum to 1 for each horizon."""
        proj_dates = np.array(
            [
                basic_setup["proj_dates"],
                basic_setup["proj_dates"],
                basic_setup["proj_dates"],
            ]
        )
        proj_values = np.array(
            [
                [110, 120, 130, 140],  # Slight increase from observed (100)
                [200, 300, 400, 500],  # Large increase
                [90, 80, 70, 60],  # Decrease
            ]
        )

        result = make_rate_trends_flusightforecast(
            reference_date=basic_setup["reference_date"],
            proj_dates=proj_dates,
            proj_values=proj_values,
            observed=basic_setup["observed"],
            population=basic_setup["population"],
        )

        for horizon in range(4):
            prob_sum = result[result["horizon"] == horizon]["value"].sum()
            assert abs(prob_sum - 1.0) < 1e-10

    def test_stable_trajectory(self, basic_setup):
        """Trajectory with no change is categorized as stable."""
        proj_dates = np.array([basic_setup["proj_dates"]])
        proj_values = np.array([[100, 100, 100, 100]])

        result = make_rate_trends_flusightforecast(
            reference_date=basic_setup["reference_date"],
            proj_dates=proj_dates,
            proj_values=proj_values,
            observed=basic_setup["observed"],
            population=basic_setup["population"],
        )

        for horizon in range(4):
            horizon_data = result[result["horizon"] == horizon]
            stable_prob = horizon_data[horizon_data["output_type_id"] == "stable"]["value"].iloc[0]
            assert stable_prob == 1.0
