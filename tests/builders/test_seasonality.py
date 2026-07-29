"""Tests for epymodelingsuite.builders.seasonality module."""

from datetime import date
from unittest.mock import Mock

import numpy as np
import pytest
from epydemix.model import EpiModel

from epymodelingsuite.builders.base import set_population_from_config
from epymodelingsuite.builders.seasonality import (
    add_seasonality_from_config,
    resolve_seasonality_location_from_population,
)
from epymodelingsuite.schema.basemodel import Population, Seasonality, Timespan


def _model_with_mock_population(n_age_groups: int = 5) -> EpiModel:
    """EpiModel with a lightweight mock population (no network fetch)."""
    model = EpiModel()
    mock_pop = Mock()
    mock_pop.num_groups = n_age_groups
    mock_pop.name = "United_States__California"
    model.population = mock_pop
    return model


class TestAddSeasonalityFromConfig:
    """Integration tests for add_seasonality_from_config function."""

    @pytest.fixture
    def model_with_beta(self):
        """Create an EpiModel with population and beta parameter set."""
        model = EpiModel()
        set_population_from_config(model, Population(name="US-CA", age_groups=["0-4", "5-17", "18-49", "50-64", "65+"]))
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
        set_population_from_config(model, Population(name="US-CA", age_groups=["0-4", "5-17", "18-49", "50-64", "65+"]))
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
        set_population_from_config(model, Population(name="US-CA", age_groups=["0-4", "5-17", "18-49", "50-64", "65+"]))

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

    def test_beta_higher_in_winter_than_summer(self, model_with_beta, seasonality_config, timespan):
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


class TestParameterToArrayConversion:
    """Tests for parameter shape conversion logic in add_seasonality_from_config."""

    @pytest.fixture
    def seasonality_config(self):
        """Create a seasonality configuration."""
        return Seasonality(
            method="balcan",
            min_value=0.5,
            max_value=1.0,
            target_parameter="beta",
            seasonality_max_date=date(2025, 12, 31),
            seasonality_min_date=date(2026, 6, 15),
        )

    @pytest.fixture
    def timespan(self):
        """Create a short timespan for testing."""
        return Timespan(
            start_date=date(2025, 12, 1),
            end_date=date(2025, 12, 31),
            delta_t=1.0,
        )

    @pytest.fixture
    def model_with_population(self):
        """Create an EpiModel with population set (5 age groups)."""
        model = EpiModel()
        set_population_from_config(model, Population(name="US-CA", age_groups=["0-4", "5-17", "18-49", "50-64", "65+"]))
        return model

    def test_scalar_converts_to_time_varying_array(self, model_with_population, seasonality_config, timespan):
        """Test that a scalar parameter becomes a (T,) array."""
        model_with_population.add_parameter("beta", 0.5)

        add_seasonality_from_config(model_with_population, seasonality_config, timespan)

        beta = model_with_population.get_parameter("beta")
        expected_T = (timespan.end_date - timespan.start_date).days + 1

        assert isinstance(beta, np.ndarray)
        assert beta.shape == (expected_T,)

    def test_scalar_values_are_multiplied_by_seasonal_factor(self, model_with_population, seasonality_config, timespan):
        """Test that scalar values are correctly multiplied by seasonal factors."""
        baseline = 0.5
        model_with_population.add_parameter("beta", baseline)

        add_seasonality_from_config(model_with_population, seasonality_config, timespan)

        beta = model_with_population.get_parameter("beta")

        # At peak (Dec 31), seasonal factor is 1.0, so beta should equal baseline
        peak_idx = (seasonality_config.seasonality_max_date - timespan.start_date).days
        assert np.isclose(beta[peak_idx], baseline * 1.0, rtol=1e-6)

        # At other times, beta should be less than baseline (seasonal factor < 1.0)
        assert np.all(beta <= baseline + 1e-10)

    def test_age_varying_expands_to_time_age_varying(self, model_with_population, seasonality_config, timespan):
        """Test that (1, N) array expands to (T, N) array."""
        n_age_groups = 5
        age_varying_beta = np.array([[0.3, 0.4, 0.5, 0.4, 0.3]])
        model_with_population.add_parameter("beta", age_varying_beta)

        add_seasonality_from_config(model_with_population, seasonality_config, timespan)

        beta = model_with_population.get_parameter("beta")
        expected_T = (timespan.end_date - timespan.start_date).days + 1

        assert beta.shape == (expected_T, n_age_groups)

    def test_age_varying_preserves_relative_differences(self, model_with_population, seasonality_config, timespan):
        """Test that age-varying values maintain relative differences after seasonality."""
        n_age_groups = 5
        age_varying_beta = np.array([[0.2, 0.4, 0.6, 0.4, 0.2]])
        model_with_population.add_parameter("beta", age_varying_beta)

        add_seasonality_from_config(model_with_population, seasonality_config, timespan)

        beta = model_with_population.get_parameter("beta")

        # At any time t, the ratio between age groups should be preserved
        # beta[t, i] / beta[t, j] should equal original[i] / original[j]
        for t in range(beta.shape[0]):
            # Check ratio of age group 2 (0.6) to age group 0 (0.2) = 3.0
            ratio = beta[t, 2] / beta[t, 0]
            expected_ratio = 0.6 / 0.2
            assert np.isclose(ratio, expected_ratio, rtol=1e-6)

    def test_time_age_varying_multiplied_piecewise(self, model_with_population, seasonality_config, timespan):
        """Test that (T, N) array is multiplied piecewise by seasonal factors."""
        n_age_groups = 5
        expected_T = (timespan.end_date - timespan.start_date).days + 1

        # Create a time-varying and age-varying parameter
        time_age_beta = np.ones((expected_T, n_age_groups)) * 0.5
        # Make it vary by age
        for i in range(n_age_groups):
            time_age_beta[:, i] *= (i + 1) * 0.1  # 0.05, 0.1, 0.15, 0.2, 0.25

        model_with_population.add_parameter("beta", time_age_beta)

        add_seasonality_from_config(model_with_population, seasonality_config, timespan)

        beta = model_with_population.get_parameter("beta")

        assert beta.shape == (expected_T, n_age_groups)

        # At peak, seasonal factor is 1.0, so values should equal original
        peak_idx = (seasonality_config.seasonality_max_date - timespan.start_date).days
        for i in range(n_age_groups):
            expected = time_age_beta[peak_idx, i]
            assert np.isclose(beta[peak_idx, i], expected, rtol=1e-6)

    def test_invalid_shape_raises_error(self, model_with_population, seasonality_config, timespan):
        """Test that unsupported parameter shapes raise ValueError."""
        # Create a 3D array which is not supported
        invalid_beta = np.ones((10, 5, 3))
        model_with_population.add_parameter("beta", invalid_beta)

        with pytest.raises(ValueError, match="Cannot apply seasonality"):
            add_seasonality_from_config(model_with_population, seasonality_config, timespan)

    def test_wrong_time_dimension_raises_error(self, model_with_population, seasonality_config, timespan):
        """Test that array with wrong time dimension raises ValueError."""
        n_age_groups = 5
        expected_T = (timespan.end_date - timespan.start_date).days + 1

        # Create array with wrong T dimension
        wrong_T = expected_T + 10
        wrong_beta = np.ones((wrong_T, n_age_groups)) * 0.5
        model_with_population.add_parameter("beta", wrong_beta)

        with pytest.raises(ValueError, match="Cannot apply seasonality"):
            add_seasonality_from_config(model_with_population, seasonality_config, timespan)


class TestAddSeasonalityWithSubdailyTimesteps:
    """Tests for add_seasonality_from_config with subdaily delta_t values."""

    @pytest.fixture
    def seasonality_config(self):
        """Create a standard seasonality configuration."""
        return Seasonality(
            method="balcan",
            min_value=0.5,
            max_value=1.0,
            target_parameter="beta",
            seasonality_max_date=date(2025, 12, 31),
            seasonality_min_date=date(2026, 6, 15),
        )

    @pytest.mark.parametrize(
        ("delta_t", "days", "expected_T"),
        [
            (1.0, 10, 11),
            (0.5, 10, 21),
            (0.25, 10, 41),
        ],
    )
    def test_beta_array_length_matches_timestep_count(self, seasonality_config, delta_t, days, expected_T):
        """Verify beta array length matches expected timestep count for various delta_t."""
        model = EpiModel()
        set_population_from_config(model, Population(name="US-CA", age_groups=["0-4", "5-17", "18-49", "50-64", "65+"]))
        model.add_parameter("beta", 0.5)

        start = date(2025, 12, 1)
        end = date(2025, 12, 1 + days)
        timespan = Timespan(start_date=start, end_date=end, delta_t=delta_t)

        add_seasonality_from_config(model, seasonality_config, timespan)

        beta = model.get_parameter("beta")
        assert beta.shape[0] == expected_T, f"Expected {expected_T} timesteps, got {beta.shape[0]}"

    def test_seasonality_values_match_across_dt(self, seasonality_config):
        """Values at overlapping timesteps should match between dt=1.0 and dt=0.5."""
        start = date(2025, 12, 1)
        end = date(2025, 12, 11)

        # Model with dt=1.0
        model_dt10 = EpiModel()
        set_population_from_config(
            model_dt10, Population(name="US-CA", age_groups=["0-4", "5-17", "18-49", "50-64", "65+"])
        )
        model_dt10.add_parameter("beta", 0.5)
        timespan_dt10 = Timespan(start_date=start, end_date=end, delta_t=1.0)
        add_seasonality_from_config(model_dt10, seasonality_config, timespan_dt10)
        beta_dt10 = model_dt10.get_parameter("beta")

        # Model with dt=0.5
        model_dt05 = EpiModel()
        set_population_from_config(
            model_dt05, Population(name="US-CA", age_groups=["0-4", "5-17", "18-49", "50-64", "65+"])
        )
        model_dt05.add_parameter("beta", 0.5)
        timespan_dt05 = Timespan(start_date=start, end_date=end, delta_t=0.5)
        add_seasonality_from_config(model_dt05, seasonality_config, timespan_dt05)
        beta_dt05 = model_dt05.get_parameter("beta")

        # Every other entry of dt=0.5 should match dt=1.0
        for k in range(len(beta_dt10)):
            np.testing.assert_allclose(
                beta_dt05[2 * k],
                beta_dt10[k],
                rtol=1e-10,
                err_msg=f"Mismatch at day index {k}",
            )

    def test_seasonality_with_age_varying_beta_subdaily(self, seasonality_config):
        """With dt=0.5 and age-varying beta, output shape should be (T, N_age)."""
        n_age_groups = 5
        start = date(2025, 12, 1)
        end = date(2025, 12, 11)  # 10 days
        delta_t = 0.5

        model = EpiModel()
        set_population_from_config(model, Population(name="US-CA", age_groups=["0-4", "5-17", "18-49", "50-64", "65+"]))
        age_varying_beta = np.array([[0.3, 0.4, 0.5, 0.4, 0.3]])
        model.add_parameter("beta", age_varying_beta)

        timespan = Timespan(start_date=start, end_date=end, delta_t=delta_t)
        add_seasonality_from_config(model, seasonality_config, timespan)

        beta = model.get_parameter("beta")
        total_days = (end - start).days
        expected_T = int(total_days / delta_t) + 1

        assert beta.shape == (expected_T, n_age_groups), f"Expected ({expected_T}, {n_age_groups}), got {beta.shape}"

        # Age ratios should be preserved at every timestep
        for t in range(beta.shape[0]):
            ratio = beta[t, 2] / beta[t, 0]
            expected_ratio = 0.5 / 0.3
            assert np.isclose(ratio, expected_ratio, rtol=1e-6), f"Age ratio not preserved at t={t}"


class TestAddClimateSeasonalityFromConfig:
    """Integration tests for climate seasonality."""

    CLIMATE_TEST_CSV = "tests/data/climate_daily_test.csv"

    def test_resolve_climate_location_from_iso_population(self):
        assert resolve_seasonality_location_from_population("United_States__California") == "US-CA"
        assert resolve_seasonality_location_from_population("US-MA") == "US-MA"

    def test_resolve_climate_location_from_metrocast_population(self):
        assert resolve_seasonality_location_from_population("metrocast_location_denver") == "US-CO"

    @pytest.fixture
    def climate_seasonality_config(self):
        return Seasonality(
            method="data_driven",
            target_parameter="beta",
            seasonality_data_path=TestAddClimateSeasonalityFromConfig.CLIMATE_TEST_CSV,
        )

    @pytest.fixture
    def climate_timespan(self):
        return Timespan(start_date=date(2024, 1, 1), end_date=date(2024, 1, 3), delta_t=1.0)

    @pytest.fixture
    def model_with_climate_coeffs(self):
        model = _model_with_mock_population()
        model.add_parameter("beta", 0.5)
        model.add_parameter("b1", 0.001)
        model.add_parameter("s_min", 0.2)
        model.add_parameter("b3", 0.05)
        return model

    def test_applies_climate_seasonality_to_scalar_beta(
        self, model_with_climate_coeffs, climate_seasonality_config, climate_timespan
    ):
        add_seasonality_from_config(model_with_climate_coeffs, climate_seasonality_config, climate_timespan)
        beta = model_with_climate_coeffs.get_parameter("beta")
        assert beta.shape[0] == 3
        assert np.all(beta > 0)

    def test_param_overrides_change_scaling(
        self, model_with_climate_coeffs, climate_seasonality_config, climate_timespan
    ):
        add_seasonality_from_config(
            model_with_climate_coeffs,
            climate_seasonality_config,
            climate_timespan,
            param_overrides={"b3": 0.0},
        )
        beta_no_temp = model_with_climate_coeffs.get_parameter("beta").copy()

        model2 = _model_with_mock_population()
        model2.add_parameter("beta", 0.5)
        model2.add_parameter("b1", 0.001)
        model2.add_parameter("s_min", 0.2)
        model2.add_parameter("b3", 0.05)
        add_seasonality_from_config(model2, climate_seasonality_config, climate_timespan)
        beta_with_temp = model2.get_parameter("beta")

        assert not np.allclose(beta_no_temp, beta_with_temp)

    def test_missing_coefficient_raises(self, climate_seasonality_config, climate_timespan):
        model = _model_with_mock_population()
        model.add_parameter("beta", 0.5)
        with pytest.raises(ValueError, match="Data-driven seasonality requires coefficient"):
            add_seasonality_from_config(model, climate_seasonality_config, climate_timespan)

    def test_climate_with_age_varying_beta(
        self, climate_seasonality_config, climate_timespan
    ):
        model = _model_with_mock_population()
        model.add_parameter("beta", np.array([[0.3, 0.4, 0.5, 0.4, 0.3]]))
        model.add_parameter("b1", 0.001)
        model.add_parameter("s_min", 0.2)
        model.add_parameter("b3", 0.05)
        add_seasonality_from_config(model, climate_seasonality_config, climate_timespan)
        beta = model.get_parameter("beta")
        assert beta.shape == (3, 5)
