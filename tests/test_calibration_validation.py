"""Tests for calibration configuration validation, specifically start_date vs fitting_window."""

from __future__ import annotations

import logging
from datetime import date

import pytest

from flumodelingsuite.schema.calibration import (
    CalibrationConfiguration,
    CalibrationStrategy,
    ComparisonSpec,
    FittingWindow,
)
from flumodelingsuite.schema.common import DateParameter, Distribution


class TestSampledStartDateValidation:
    """Test validation of sampled start_date against fitting_window."""

    def test_valid_randint_within_fitting_window(self, caplog):
        """Test that valid randint configuration does not produce warnings."""
        with caplog.at_level(logging.WARNING):
            config = CalibrationConfiguration(
                strategy=CalibrationStrategy(name="SMC"),
                distance_function="rmse",
                observed_data_path="data/test.csv",
                comparison=[
                    ComparisonSpec(
                        observed_value_column="value",
                        observed_date_column="date",
                        simulation=["I_to_R"],
                    )
                ],
                fitting_window=FittingWindow(
                    start_date=date(2024, 9, 1),
                    end_date=date(2024, 12, 1),
                ),
                start_date=DateParameter(
                    reference_date=date(2024, 10, 1),
                    prior=Distribution(type="scipy", name="randint", args=[0, 30]),  # Max: 29 days
                ),
            )

        # Max sampled date: 2024-10-01 + 29 = 2024-10-30, which is before 2024-12-01
        assert "could extend beyond fitting window" not in caplog.text
        assert config is not None

    def test_invalid_randint_exceeds_fitting_window(self):
        """Test that randint configuration exceeding fitting window raises ValueError."""
        # Max sampled date: 2024-10-01 + 90 = 2024-12-30, which is after 2024-12-01
        with pytest.raises(ValueError, match="could extend beyond fitting window"):
            CalibrationConfiguration(
                strategy=CalibrationStrategy(name="SMC"),
                distance_function="rmse",
                observed_data_path="data/test.csv",
                comparison=[
                    ComparisonSpec(
                        observed_value_column="value",
                        observed_date_column="date",
                        simulation=["I_to_R"],
                    )
                ],
                fitting_window=FittingWindow(
                    start_date=date(2024, 9, 1),
                    end_date=date(2024, 12, 1),
                ),
                start_date=DateParameter(
                    reference_date=date(2024, 10, 1),
                    prior=Distribution(type="scipy", name="randint", args=[0, 91]),  # Max: 90 days
                ),
            )

    def test_valid_uniform_within_fitting_window(self, caplog):
        """Test that valid uniform configuration does not produce warnings."""
        with caplog.at_level(logging.WARNING):
            config = CalibrationConfiguration(
                strategy=CalibrationStrategy(name="SMC"),
                distance_function="rmse",
                observed_data_path="data/test.csv",
                comparison=[
                    ComparisonSpec(
                        observed_value_column="value",
                        observed_date_column="date",
                        simulation=["I_to_R"],
                    )
                ],
                fitting_window=FittingWindow(
                    start_date=date(2024, 9, 1),
                    end_date=date(2024, 12, 1),
                ),
                start_date=DateParameter(
                    reference_date=date(2024, 10, 1),
                    prior=Distribution(type="scipy", name="uniform", args=[0, 30]),  # Max: 30 days
                ),
            )

        # Max sampled date: 2024-10-01 + 30 = 2024-10-31, which is before 2024-12-01
        assert "could extend beyond fitting window" not in caplog.text
        assert config is not None

    def test_invalid_uniform_exceeds_fitting_window(self):
        """Test that uniform configuration exceeding fitting window raises ValueError."""
        # Max sampled date: 2024-10-01 + 90 = 2024-12-30, which is after 2024-12-01
        with pytest.raises(ValueError, match="could extend beyond fitting window"):
            CalibrationConfiguration(
                strategy=CalibrationStrategy(name="SMC"),
                distance_function="rmse",
                observed_data_path="data/test.csv",
                comparison=[
                    ComparisonSpec(
                        observed_value_column="value",
                        observed_date_column="date",
                        simulation=["I_to_R"],
                    )
                ],
                fitting_window=FittingWindow(
                    start_date=date(2024, 9, 1),
                    end_date=date(2024, 12, 1),
                ),
                start_date=DateParameter(
                    reference_date=date(2024, 10, 1),
                    prior=Distribution(type="scipy", name="uniform", args=[0, 90]),  # Max: 90 days
                ),
            )

    def test_edge_case_max_date_equals_fitting_window_end(self, caplog):
        """Test edge case where max sampled date exactly equals fitting window end."""
        with caplog.at_level(logging.WARNING):
            config = CalibrationConfiguration(
                strategy=CalibrationStrategy(name="SMC"),
                distance_function="rmse",
                observed_data_path="data/test.csv",
                comparison=[
                    ComparisonSpec(
                        observed_value_column="value",
                        observed_date_column="date",
                        simulation=["I_to_R"],
                    )
                ],
                fitting_window=FittingWindow(
                    start_date=date(2024, 9, 1),
                    end_date=date(2024, 12, 1),
                ),
                start_date=DateParameter(
                    reference_date=date(2024, 10, 1),
                    prior=Distribution(type="scipy", name="randint", args=[0, 62]),  # Max: 61 days
                ),
            )

        # Max sampled date: 2024-10-01 + 61 = 2024-12-01, which equals 2024-12-01
        assert "could extend beyond fitting window" not in caplog.text
        assert config is not None

    def test_no_start_date_specified(self, caplog):
        """Test that validation is skipped when start_date is not specified."""
        from flumodelingsuite.schema.calibration import CalibrationParameter

        with caplog.at_level(logging.WARNING):
            config = CalibrationConfiguration(
                strategy=CalibrationStrategy(name="SMC"),
                distance_function="rmse",
                observed_data_path="data/test.csv",
                comparison=[
                    ComparisonSpec(
                        observed_value_column="value",
                        observed_date_column="date",
                        simulation=["I_to_R"],
                    )
                ],
                fitting_window=FittingWindow(
                    start_date=date(2024, 9, 1),
                    end_date=date(2024, 12, 1),
                ),
                start_date=None,
                parameters={
                    "Rt": CalibrationParameter(prior=Distribution(type="scipy", name="uniform", args=[1.0, 2.0]))
                },
            )

        # No validation should occur
        assert "could extend beyond fitting window" not in caplog.text
        assert config is not None

    def test_start_date_without_prior(self, caplog):
        """Test that validation is skipped when start_date has no prior."""
        with caplog.at_level(logging.WARNING):
            config = CalibrationConfiguration(
                strategy=CalibrationStrategy(name="SMC"),
                distance_function="rmse",
                observed_data_path="data/test.csv",
                comparison=[
                    ComparisonSpec(
                        observed_value_column="value",
                        observed_date_column="date",
                        simulation=["I_to_R"],
                    )
                ],
                fitting_window=FittingWindow(
                    start_date=date(2024, 9, 1),
                    end_date=date(2024, 12, 1),
                ),
                start_date=DateParameter(
                    reference_date=date(2024, 10, 1),
                    prior=None,
                ),
            )

        # No validation should occur
        assert "could extend beyond fitting window" not in caplog.text
        assert config is not None

    def test_unsupported_distribution_type(self, caplog):
        """Test that unsupported distribution types produce a warning and skip validation."""
        with caplog.at_level(logging.WARNING):
            config = CalibrationConfiguration(
                strategy=CalibrationStrategy(name="SMC"),
                distance_function="rmse",
                observed_data_path="data/test.csv",
                comparison=[
                    ComparisonSpec(
                        observed_value_column="value",
                        observed_date_column="date",
                        simulation=["I_to_R"],
                    )
                ],
                fitting_window=FittingWindow(
                    start_date=date(2024, 9, 1),
                    end_date=date(2024, 12, 1),
                ),
                start_date=DateParameter(
                    reference_date=date(2024, 10, 1),
                    prior=Distribution(type="scipy", name="norm", args=[10, 5]),
                ),
            )

        # Should warn about unsupported distribution
        assert "norm" in caplog.text
        assert "does not get validated" in caplog.text
        assert config is not None
