"""Tests for epymodelingsuite.builders.seasonality module."""

from datetime import date

import numpy as np
import pytest
from epydemix.model import EpiModel

from epymodelingsuite.builders.base import set_population_from_config
from epymodelingsuite.builders.seasonality import add_seasonality_from_config
from epymodelingsuite.schema.basemodel import Seasonality, Timespan


class TestAddSeasonalityFromConfig:
    """Integration tests for add_seasonality_from_config function."""

    @pytest.fixture
    def model_with_beta(self):
        """Create an EpiModel with population and beta parameter set."""
        model = EpiModel()
        set_population_from_config(model, "US-CA", ["0-4", "5-17", "18-49", "50-64", "65+"])
        model.add_parameter("beta", 0.5)
        return model

    @pytest.fixture
    def seasonality_config(self):
        """Create a standard seasonality configuration for Northern Hemisphere flu.

        Realistic flu seasonality pattern:
        - Peak transmission (max): December 31 (late winter)
        - Minimum transmission (min): June 15 (mid-summer)

        Scaling factors:
        - max_value=1.0: Peak transmission equals baseline (no scaling at winter peak)
        - min_value=0.5: Summer trough is 50% of baseline transmission
        """
        return Seasonality(
            method="balcan",
            min_value=0.5,  # Trough scaling factor (50% of baseline)
            max_value=1.0,  # Peak scaling factor (100% of baseline)
            target_parameter="beta",
            seasonality_max_date=date(2025, 12, 31),  # Winter peak
            seasonality_min_date=date(2026, 6, 15),  # Summer trough
        )

    @pytest.fixture
    def timespan(self):
        """Create a timespan for simulation."""
        return Timespan(
            start_date=date(2025, 10, 1),
            end_date=date(2026, 5, 31),
            delta_t=1.0,
        )

    def test_applies_seasonality_to_target_parameter(self, model_with_beta, seasonality_config, timespan):
        """Test that seasonality modifies the target parameter."""
        original_beta = model_with_beta.get_parameter("beta")
        assert original_beta == 0.5  # Scalar before

        add_seasonality_from_config(model_with_beta, seasonality_config, timespan)

        new_beta = model_with_beta.get_parameter("beta")
        # Should now be an array, not a scalar
        assert hasattr(new_beta, "__len__"), "Beta should be an array after seasonality"
        assert isinstance(new_beta, np.ndarray)

    def test_parameter_becomes_time_varying_array(self, model_with_beta, seasonality_config, timespan):
        """Test that the parameter becomes a time-varying array with correct length."""
        add_seasonality_from_config(model_with_beta, seasonality_config, timespan)

        beta = model_with_beta.get_parameter("beta")

        # Length should match number of days in timespan
        expected_length = (timespan.end_date - timespan.start_date).days + 1
        assert len(beta) == expected_length

    def test_seasonal_values_peak_at_max_date(self, model_with_beta, seasonality_config, timespan):
        """Test that seasonal values peak at the configured max date."""
        add_seasonality_from_config(model_with_beta, seasonality_config, timespan)

        beta = model_with_beta.get_parameter("beta")

        # Find index of max date
        max_date_idx = (seasonality_config.seasonality_max_date - timespan.start_date).days

        # Value at max date should be at or near the maximum
        # Since beta_original * seasonal_factor, and seasonal_factor peaks at 1.0 at max_date
        # beta at max_date should be 0.5 * 1.0 = 0.5 (the original value * peak factor)
        assert np.isclose(beta[max_date_idx], 0.5, rtol=1e-6), (
            f"Beta at max_date should be ~0.5, got {beta[max_date_idx]}"
        )

        # The max_date should have the highest value (or very close to it)
        assert np.isclose(beta[max_date_idx], max(beta), rtol=0, atol=1e-6)

    def test_seasonal_values_trough_at_min_date(self, model_with_beta, seasonality_config, timespan):
        """Test that seasonal values are lowest around the min date."""
        # Extend timespan to include min date (June 15, 2026)
        extended_timespan = Timespan(
            start_date=date(2025, 10, 1),
            end_date=date(2026, 8, 31),
            delta_t=1.0,
        )

        add_seasonality_from_config(model_with_beta, seasonality_config, extended_timespan)

        beta = model_with_beta.get_parameter("beta")

        # Find index of min date
        min_date_idx = (seasonality_config.seasonality_min_date - extended_timespan.start_date).days

        # Value at min date should be at the minimum
        # Trough factor = min_value / max_value (equals min_value when max_value=1.0)
        # beta at min_date = baseline * (min_value/max_value) = 0.5 * (0.5/1.0) = 0.25
        trough_factor = seasonality_config.min_value / seasonality_config.max_value
        expected_min_beta = 0.5 * trough_factor
        assert np.isclose(beta[min_date_idx], expected_min_beta, rtol=1e-2), (
            f"Beta at min_date should be ~{expected_min_beta}, got {beta[min_date_idx]}"
        )

    def test_seasonal_values_within_expected_range(self, model_with_beta, seasonality_config, timespan):
        """Test that all seasonal values fall within expected range."""
        original_beta = 0.5
        add_seasonality_from_config(model_with_beta, seasonality_config, timespan)

        beta = model_with_beta.get_parameter("beta")

        # Seasonal factor ranges from (min_value/max_value) to 1.0
        # So beta should range from original * (min_value/max_value) to original * 1.0
        min_factor = seasonality_config.min_value / seasonality_config.max_value
        expected_min = original_beta * min_factor  # 0.5 * (0.5/1.0) = 0.25
        expected_max = original_beta * 1.0  # Peak factor is always 1.0

        assert min(beta) >= expected_min - 1e-10, f"Beta min {min(beta)} below expected {expected_min}"
        assert max(beta) <= expected_max + 1e-10, f"Beta max {max(beta)} above expected {expected_max}"

    def test_raises_error_for_undefined_parameter(self, seasonality_config, timespan):
        """Test that error is raised when target parameter doesn't exist."""
        model = EpiModel()
        set_population_from_config(model, "US-CA", ["0-4", "5-17", "18-49", "50-64", "65+"])
        # Don't add beta parameter

        with pytest.raises(ValueError, match="undefined parameter"):
            add_seasonality_from_config(model, seasonality_config, timespan)

    def test_works_without_min_date(self, model_with_beta, timespan):
        """Test that seasonality works when min_date is not specified."""
        config_no_min = Seasonality(
            method="balcan",
            min_value=0.5,  # Trough scaling factor
            max_value=1.0,  # Peak scaling factor
            target_parameter="beta",
            seasonality_max_date=date(2025, 12, 31),  # Winter peak
            # No seasonality_min_date - will be derived from default period (365 days)
        )

        add_seasonality_from_config(model_with_beta, config_no_min, timespan)

        beta = model_with_beta.get_parameter("beta")
        assert isinstance(beta, np.ndarray)
        assert len(beta) > 0

    def test_with_age_varying_parameter(self, seasonality_config, timespan):
        """Test seasonality applied to an age-varying parameter."""
        model = EpiModel()
        set_population_from_config(model, "US-CA", ["0-4", "5-17", "18-49", "50-64", "65+"])

        # Add age-varying beta (shape: 1 x N_age_groups)
        n_age_groups = 5
        age_varying_beta = np.array([[0.3, 0.4, 0.5, 0.4, 0.3]])
        model.add_parameter("beta", age_varying_beta)

        add_seasonality_from_config(model, seasonality_config, timespan)

        beta = model.get_parameter("beta")

        # Should now be time-varying AND age-varying (T x N)
        expected_T = (timespan.end_date - timespan.start_date).days + 1
        assert beta.shape == (expected_T, n_age_groups), (
            f"Expected shape ({expected_T}, {n_age_groups}), got {beta.shape}"
        )

    def test_winter_higher_than_summer(self, model_with_beta, seasonality_config, timespan):
        """Test that winter transmission is higher than summer (realistic flu pattern)."""
        # Extend to include both peak and trough
        full_year_timespan = Timespan(
            start_date=date(2025, 10, 1),
            end_date=date(2026, 8, 31),
            delta_t=1.0,
        )

        add_seasonality_from_config(model_with_beta, seasonality_config, full_year_timespan)

        beta = model_with_beta.get_parameter("beta")

        # December 31 (winter peak) index
        dec_idx = (date(2025, 12, 31) - date(2025, 10, 1)).days
        # June 15 (summer trough) index
        jun_idx = (date(2026, 6, 15) - date(2025, 10, 1)).days

        assert beta[dec_idx] > beta[jun_idx], (
            f"Winter beta ({beta[dec_idx]}) should be higher than summer ({beta[jun_idx]})"
        )
