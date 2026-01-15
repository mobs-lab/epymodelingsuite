"""Tests for vaccination data processing functions."""

import os
import tempfile
from datetime import date

import numpy as np
import pandas as pd
import pytest

from epymodelingsuite.vaccinations import (
    add_vaccination_schedule,
    make_vaccination_rate_function,
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


class TestVaccinationIntegration:
    """Integration tests for vaccination schedule application to models."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple SIR model with vaccination compartments."""
        from epydemix.model import EpiModel

        from epymodelingsuite.builders.base import set_population_from_config

        model = EpiModel()
        set_population_from_config(model, "US-CA", ["0-4", "5-17", "18-49", "50-64", "65+"])

        # Add compartments
        model.add_compartments(["S", "S_vax", "I", "R"])

        # Add basic transitions (non-vaccination)
        model.add_transition("S", "I", params=("beta", "I"), kind="mediated")
        model.add_transition("I", "R", params="gamma", kind="spontaneous")

        # Add parameters
        model.add_parameter(parameters_dict={"beta": 0.3, "gamma": 0.1})

        return model

    @pytest.fixture
    def vaccination_schedule(self):
        """Create a simple vaccination schedule DataFrame."""
        dates = pd.date_range("2024-10-01", periods=30, freq="D")
        return pd.DataFrame(
            {
                "dates": dates,
                "location": ["US-CA"] * 30,
                "0-4": [100.0] * 30,
                "5-17": [200.0] * 30,
                "18-49": [500.0] * 30,
                "50-64": [300.0] * 30,
                "65+": [400.0] * 30,
            }
        )

    def test_vaccination_transitions_added_to_model(self, simple_model, vaccination_schedule):
        """Verify vaccination config creates transitions in the model."""
        # Create vaccine rate function
        vaccine_rate_function = make_vaccination_rate_function(origin_compartment="S", eligible_compartments=["S", "R"])

        # Count transitions before
        transitions_before = len(simple_model.transitions_list)

        # Add vaccination schedule
        model = add_vaccination_schedule(
            model=simple_model,
            vaccine_rate_function=vaccine_rate_function,
            source_comp="S",
            target_comp="S_vax",
            vaccination_schedule=vaccination_schedule,
        )

        # Verify transitions were added
        transitions_after = len(model.transitions_list)
        assert transitions_after > transitions_before, "Vaccination transition should be added to model"

        # Verify the vaccination transition exists
        vaccination_transitions = [t for t in model.transitions_list if t.kind == "vaccination"]
        assert len(vaccination_transitions) == 1, "Should have exactly one vaccination transition"
        assert vaccination_transitions[0].source == "S"
        assert vaccination_transitions[0].target == "S_vax"

    def test_vaccination_rate_function_computes_correctly(self):
        """Verify vaccination rate function computes rates correctly.

        The vaccination rate function uses proportional dose allocation:
        1. fraction_origin = S / (S + R)  -- fraction of eligible pop in origin compartment
        2. effective_doses = doses * fraction_origin  -- doses allocated to origin
        3. rate = effective_doses / S  -- per-person rate in origin compartment

        This simplifies to: rate = doses / (S + R)

        This model assumes doses are split between eligible compartments based on their relative sizes.
        """
        # Create vaccine rate function with S and R as eligible compartments
        # This means doses are allocated proportionally between S and R
        vaccine_rate_function = make_vaccination_rate_function(origin_compartment="S", eligible_compartments=["S", "R"])

        # Pop array structure: (num_compartments, num_age_groups)
        # Compartments: S=0, S_vax=1, I=2, R=3
        # Age groups: 5 groups
        pop = np.array(
            [
                [1000, 2000, 3000, 4000, 5000],  # S
                [0, 0, 0, 0, 0],  # S_vax
                [10, 20, 30, 40, 50],  # I
                [100, 200, 300, 400, 500],  # R
            ]
        )

        # Vaccination schedule: shape (timesteps, age_groups)
        # Each row is a timestep, each column is an age group: [0-4, 5-17, 18-49, 50-64, 65+]
        # Values are number of doses available for that age group on that day
        vaccination_schedule = np.array(
            [
                [100, 200, 500, 300, 400],  # t=0: doses for each age group
                [110, 210, 510, 310, 410],  # t=1: doses for each age group
            ]
        )

        data = {
            "t": 0,
            "dt": 1.0,
            "pop": pop,
            "comp_indices": {"S": 0, "S_vax": 1, "I": 2, "R": 3},
        }

        params = [vaccination_schedule]

        # Compute rate
        rate = vaccine_rate_function(params, data)

        # Rate should be positive and finite
        assert np.all(rate >= 0), "Vaccination rate should be non-negative"
        assert np.all(np.isfinite(rate)), "Vaccination rate should be finite"

        # Verify rate computation for first age group (age 0-4):
        #   doses = 100, S = 1000, R = 100
        #   eligible = S + R = 1100
        #
        # Step 1: fraction_origin = S / eligible = 1000/1100 ≈ 0.909
        # Step 2: effective_doses = doses * fraction_origin = 100 * 0.909 ≈ 90.9
        # Step 3: rate = effective_doses / S = 90.9 / 1000 ≈ 0.0909
        #
        # Simplified: rate = doses / (S + R) = 100 / 1100 ≈ 0.0909
        s_pop = pop[0, 0]  # 1000
        r_pop = pop[3, 0]  # 100
        doses_at_t0 = vaccination_schedule[0, 0]  # 100
        expected_rate_0 = doses_at_t0 / (s_pop + r_pop)  # 100 / 1100
        assert np.isclose(rate[0], expected_rate_0, rtol=1e-5)

    def test_vaccination_rate_handles_zero_population(self):
        """Verify vaccination rate handles zero population gracefully."""
        vaccine_rate_function = make_vaccination_rate_function(origin_compartment="S", eligible_compartments=["S", "R"])

        # Pop array structure: (num_compartments, num_age_groups)
        # Age group 0 has zero S population
        pop = np.array(
            [
                [0, 1000],  # S: first age group has 0
                [0, 0],  # S_vax
                [0, 10],  # I
                [0, 100],  # R
            ]
        )

        # Vaccination schedule: shape (timesteps, age_groups)
        # Columns are age groups (only 2 here to match pop array)
        vaccination_schedule = np.array(
            [
                [100, 200],  # t=0: doses for age group 0 and 1
            ]
        )

        data = {
            "t": 0,
            "dt": 1.0,
            "pop": pop,
            "comp_indices": {"S": 0, "S_vax": 1, "I": 2, "R": 3},
        }

        params = [vaccination_schedule]

        # Should not raise, should return 0 for zero population
        rate = vaccine_rate_function(params, data)

        assert rate[0] == 0, "Rate should be 0 when origin population is 0"
        assert rate[1] > 0, "Rate should be positive for non-zero population"

    def test_add_vaccination_schedule_validates_location(self, simple_model, vaccination_schedule):
        """Verify that add_vaccination_schedule validates location."""
        vaccine_rate_function = make_vaccination_rate_function(origin_compartment="S", eligible_compartments=["S"])

        # Create schedule with wrong location
        wrong_location_schedule = vaccination_schedule.copy()
        wrong_location_schedule["location"] = "US-TX"

        with pytest.raises(ValueError, match="not found in vaccination schedule"):
            add_vaccination_schedule(
                model=simple_model,
                vaccine_rate_function=vaccine_rate_function,
                source_comp="S",
                target_comp="S_vax",
                vaccination_schedule=wrong_location_schedule,
            )

    def test_add_vaccination_schedule_validates_age_groups(self, simple_model):
        """Verify that add_vaccination_schedule validates age groups match model."""
        vaccine_rate_function = make_vaccination_rate_function(origin_compartment="S", eligible_compartments=["S"])

        # Create schedule with missing age groups
        incomplete_schedule = pd.DataFrame(
            {
                "dates": pd.date_range("2024-10-01", periods=10, freq="D"),
                "location": ["US-CA"] * 10,
                "0-4": [100.0] * 10,
                # Missing other age groups
            }
        )

        with pytest.raises(ValueError, match="Age groups"):
            add_vaccination_schedule(
                model=simple_model,
                vaccine_rate_function=vaccine_rate_function,
                source_comp="S",
                target_comp="S_vax",
                vaccination_schedule=incomplete_schedule,
            )
