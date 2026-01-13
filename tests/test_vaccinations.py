"""Tests for vaccination data processing functions."""

import os
import tempfile
from datetime import date

import pandas as pd
import pytest

from epymodelingsuite.vaccinations import (
    _get_vaccination_scaling_factors,
    resample_vaccination_schedule,
    scenario_to_epydemix,
)


class TestResampleVaccinationSchedule:
    """Tests for resample_vaccination_schedule function."""

    def test_forward_fill_subdaily(self):
        """Test forward-fill for dt < 1 without scaling."""
        # Create daily schedule with location column
        daily_schedule = pd.DataFrame(
            {
                "dates": pd.date_range("2024-10-01", periods=3, freq="D"),
                "location": ["US-MA", "US-MA", "US-MA"],
                "0-4": [100.0, 200.0, 300.0],
                "5-17": [150.0, 250.0, 350.0],
            }
        )

        # Resample to 12-hour timesteps (dt=0.5)
        resampled = resample_vaccination_schedule(daily_schedule, delta_t=0.5)

        # Values should be forward-filled (repeated), not scaled
        # Check that first day's values are repeated
        assert resampled["0-4"].iloc[0] == 100.0
        assert resampled["0-4"].iloc[1] == 100.0
        # Check that second day's values are repeated
        assert resampled["0-4"].iloc[2] == 200.0
        assert resampled["0-4"].iloc[3] == 200.0

    def test_no_resampling_when_daily(self):
        """Test that dt=1.0 returns a copy without resampling."""
        daily_schedule = pd.DataFrame(
            {
                "dates": pd.date_range("2024-10-01", periods=3, freq="D"),
                "location": ["US-MA", "US-MA", "US-MA"],
                "0-4": [100.0, 200.0, 300.0],
                "5-17": [150.0, 250.0, 350.0],
            }
        )

        resampled = resample_vaccination_schedule(daily_schedule, delta_t=1.0)

        # Should be identical to input
        assert len(resampled) == len(daily_schedule)
        pd.testing.assert_frame_equal(resampled.reset_index(drop=True), daily_schedule)

    def test_preserves_location_and_structure(self):
        """Test that resampling preserves location and column structure."""
        daily_schedule = pd.DataFrame(
            {
                "dates": pd.date_range("2024-10-01", periods=7, freq="D"),
                "location": ["US-MA"] * 7,
                "0-4": [100.0] * 7,
                "5-17": [150.0] * 7,
            }
        )

        # Resample to 12-hour timesteps
        resampled = resample_vaccination_schedule(daily_schedule, delta_t=0.5)

        # Verify structure is preserved
        assert "dates" in resampled.columns
        assert "location" in resampled.columns
        assert "0-4" in resampled.columns
        assert "5-17" in resampled.columns

        # Verify location is preserved
        assert all(resampled["location"] == "US-MA")


class TestScenarioToEpydemix:
    """Tests for scenario_to_epydemix function."""

    def test_no_delta_t_parameter(self):
        """Test that scenario_to_epydemix no longer accepts delta_t parameter."""
        # Create a minimal temporary CSV file with properly formatted data
        vaccination_data = pd.DataFrame(
            {
                "Week_Ending_Sat": ["2024-10-05", "2024-10-05", "2024-10-05"],
                "Geography": ["California", "California", "California"],
                "Age": ["6 Months - 4 Years", "5-17 Years", "6 Months - 17 Years"],
                "Population": [500000, 1000000, 1500000],
                "Coverage": [0.10, 0.15, 0.125],
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as temp_file:
            vaccination_data.to_csv(temp_file.name, index=False)
            temp_filepath = temp_file.name

        try:
            # This should raise TypeError if delta_t parameter is passed
            with pytest.raises(TypeError, match="delta_t"):
                scenario_to_epydemix(
                    input_filepath=temp_filepath,
                    start_date=date(2024, 10, 1),
                    end_date=date(2024, 10, 7),
                    target_age_groups=["0-4", "5-17", "18+"],
                    delta_t=0.5,  # This should cause an error
                )

        finally:
            # Clean up temporary file
            if os.path.exists(temp_filepath):
                os.unlink(temp_filepath)


class TestGetVaccinationScalingFactors:
    """Tests for _get_vaccination_scaling_factors function."""

    def test_iso_location_returns_all_ones(self):
        """Test that ISO (state-level) locations return scaling factors of 1.0."""
        age_groups = ["0-4", "5-17", "18-49", "50-64", "65+"]

        result = _get_vaccination_scaling_factors("United_States_Colorado", age_groups)

        assert all(factor == 1.0 for factor in result.values())
        assert list(result.keys()) == age_groups

    def test_metrocast_location_returns_scaled_factors(self):
        """Test that sub-state locations return age-stratified scaling factors < 1.0."""
        age_groups = ["0-4", "5-17", "18-49", "50-64", "65+"]

        result = _get_vaccination_scaling_factors("metrocast_location_denver", age_groups)

        # Denver is part of Colorado, so all factors should be < 1.0
        assert all(0 < factor < 1.0 for factor in result.values())
        # Should have same age groups
        assert list(result.keys()) == age_groups

    def test_scaling_factors_vary_by_age_group(self):
        """Test that different age groups can have different scaling factors."""
        age_groups = ["0-4", "5-17", "18-49", "50-64", "65+"]

        result = _get_vaccination_scaling_factors("metrocast_location_denver", age_groups)

        # Scaling factors should not all be identical (demographic differences)
        factors = list(result.values())
        # At least some variation expected (not all exactly equal)
        assert len(set(round(f, 4) for f in factors)) > 1

    def test_scaling_factors_sum_is_reasonable(self):
        """Test that scaling factors are in reasonable range for Denver/Colorado."""
        age_groups = ["0-4", "5-17", "18-49", "50-64", "65+"]

        result = _get_vaccination_scaling_factors("metrocast_location_denver", age_groups)

        # Denver is roughly 50% of Colorado's population
        # Average scaling factor should be around 0.4-0.6
        avg_factor = sum(result.values()) / len(result)
        assert 0.3 < avg_factor < 0.7

    def test_unknown_metrocast_location_returns_ones(self):
        """Test that unknown sub-state location returns 1.0 with warning."""
        age_groups = ["0-4", "5+"]

        result = _get_vaccination_scaling_factors("metrocast_location_nonexistent", age_groups)

        assert all(factor == 1.0 for factor in result.values())

    def test_boston_massachusetts_scaling(self):
        """Test scaling factors for Boston (Massachusetts sub-state region)."""
        age_groups = ["0-4", "5-17", "18-49", "50-64", "65+"]

        result = _get_vaccination_scaling_factors("metrocast_location_boston", age_groups)

        # Boston is part of Massachusetts, factors should be < 1.0
        assert all(0 < factor < 1.0 for factor in result.values())

    def test_nc_flu_region_scaling(self):
        """Test scaling factors for NC flu region (North Carolina sub-state)."""
        age_groups = ["0-4", "5-17", "18-49", "50-64", "65+"]

        result = _get_vaccination_scaling_factors("metrocast_location_nenc", age_groups)

        # NENC is part of North Carolina, factors should be < 1.0
        assert all(0 < factor < 1.0 for factor in result.values())
