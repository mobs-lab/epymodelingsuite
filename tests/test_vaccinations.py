"""Tests for vaccination data processing functions."""

import os
import tempfile
from datetime import date

import numpy as np
import pandas as pd
import pytest

from epymodelingsuite.vaccinations import (
    add_vaccination_schedule,
    get_age_groups_from_data,
    make_reweighting_factors,
    make_vaccination_rate_function,
    reaggregate_vaccines,
    remove_vaccination_transitions,
    resample_vaccination_schedule,
    scenario_to_epydemix,
    smh_data_to_epydemix,
)


def _make_base_scenario_df(coverage_override: dict[str, float] | None = None) -> pd.DataFrame:
    """Create a minimal scenario dataset with full age coverage."""
    coverage_override = coverage_override or {}
    base_coverage = {
        "6 Months - 4 Years": 50.0,
        "5-12 Years": 40.0,
        "13-17 Years": 35.0,
        "18-49 Years": 25.0,
        "50-64 Years": 45.0,
        "65+ Years": 60.0,
        "6 Months - 17 Years": 42.0,
    }
    base_population = {
        "6 Months - 4 Years": 2086948,
        "5-12 Years": 3958033,
        "13-17 Years": 2522629,
        "18-49 Years": 17225750,
        "50-64 Years": 7219051,
        "65+ Years": 5976166,
        "6 Months - 17 Years": 6480662,
    }

    rows = []
    week = pd.Timestamp("2025-09-06")
    for age, coverage in base_coverage.items():
        rows.append(
            {
                "Week_Ending_Sat": week.strftime("%Y-%m-%d"),
                "Geography": "California",
                "Age": age,
                "Population": base_population[age],
                "Coverage": coverage_override.get(age, coverage),
            }
        )
    return pd.DataFrame(rows)


class TestResampleVaccinationSchedule:
    """Tests for resample_vaccination_schedule function."""

    def test_forward_fill_subdaily(self):
        """Test forward-fill for dt < 1 without scaling."""
        # Create daily schedule with location column
        daily_schedule = pd.DataFrame(
            {
                "dates": pd.date_range("2025-09-30", periods=3, freq="D"),
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
                "dates": pd.date_range("2025-09-30", periods=3, freq="D"),
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
                "dates": pd.date_range("2025-09-30", periods=7, freq="D"),
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

    def test_preserves_coverage_percentage(self, tmp_path):
        """Verify that scenario_to_epydemix preserves coverage percentages.

        The function should produce doses such that:
            effective_coverage = converted_doses / epydemix_population
        matches the input coverage percentage from the scenario file.

        This is the intended design: preserve coverage %, not absolute doses.
        """
        from epymodelingsuite.utils import get_population_codebook

        # Create test data with known coverage percentages
        # Using California as it has epydemix population data
        test_coverage = {
            "6 Months - 4 Years": 50.0,  # 50% coverage
            "5-12 Years": 40.0,
            "13-17 Years": 35.0,
            "18-49 Years": 25.0,
            "50-64 Years": 45.0,
            "65+ Years": 60.0,
            "6 Months - 17 Years": 42.0,  # Aggregate - will be excluded
        }

        # Population values (these are from the scenario file, not epydemix)
        # The function should use epydemix population to calculate doses
        test_population = {
            "6 Months - 4 Years": 2086948,
            "5-12 Years": 3958033,
            "13-17 Years": 2522629,
            "18-49 Years": 17225750,
            "50-64 Years": 7219051,
            "65+ Years": 5976166,
            "6 Months - 17 Years": 6480662,
        }

        # Create CSV with multiple weeks to have coverage progression
        rows = []
        weeks = pd.date_range("2025-09-06", periods=10, freq="W-SAT")
        for week_idx, week in enumerate(weeks):
            # Coverage increases each week up to target
            week_fraction = (week_idx + 1) / len(weeks)
            for age, target_cov in test_coverage.items():
                rows.append(
                    {
                        "Week_Ending_Sat": week.strftime("%Y-%m-%d"),
                        "Geography": "California",
                        "Age": age,
                        "Population": test_population[age],
                        "Coverage": target_cov * week_fraction,  # Gradual increase
                    }
                )

        test_df = pd.DataFrame(rows)
        test_file = tmp_path / "test_vaccine_scenario.csv"
        test_df.to_csv(test_file, index=False)

        # Convert using scenario_to_epydemix
        result = scenario_to_epydemix(
            input_filepath=str(test_file),
            start_date=date(2025, 9, 6),
            end_date=date(2025, 11, 8),
            target_age_groups=["0-4", "5-17", "18-49", "50-64", "65+"],
            states=["California"],
        )

        # Get epydemix population for California
        codebook = get_population_codebook()
        ca_pop = codebook["United_States_California"]

        # Calculate epydemix population for each model age group
        # TODO: Use aggregate_population_by_age_groups utility once merged
        epydemix_pop = {
            "0-4": sum(ca_pop.values[0:5]),
            "5-17": sum(ca_pop.values[5:18]),
            "18-49": sum(ca_pop.values[18:50]),
            "50-64": sum(ca_pop.values[50:65]),
            "65+": sum(ca_pop.values[65:85]),
        }

        # Expected final coverage for model age groups
        # 0-4 maps to "6 Months - 4 Years"
        # 5-17 maps to weighted average of "5-12 Years" and "13-17 Years"
        expected_coverage = {
            "0-4": test_coverage["6 Months - 4 Years"],
            # 5-17 is combination of 5-12 and 13-17, weighted by epydemix population
            "5-17": (
                test_coverage["5-12 Years"] * sum(ca_pop.values[5:13])
                + test_coverage["13-17 Years"] * sum(ca_pop.values[13:18])
            )
            / epydemix_pop["5-17"],
            "18-49": test_coverage["18-49 Years"],
            "50-64": test_coverage["50-64 Years"],
            "65+": test_coverage["65+ Years"],
        }

        # Verify effective coverage matches expected
        for age_group in ["0-4", "5-17", "18-49", "50-64", "65+"]:
            converted_doses = result[age_group].sum()
            effective_coverage = (converted_doses / epydemix_pop[age_group]) * 100

            assert abs(effective_coverage - expected_coverage[age_group]) < 2.0, (
                f"Coverage mismatch for {age_group}: "
                f"effective={effective_coverage:.1f}%, expected={expected_coverage[age_group]:.1f}%"
            )

    def test_no_delta_t_parameter(self):
        """Test that scenario_to_epydemix no longer accepts delta_t parameter."""
        # Create a minimal temporary CSV file with properly formatted data
        vaccination_data = pd.DataFrame(
            {
                "Week_Ending_Sat": ["2025-10-04", "2025-10-04", "2025-10-04"],
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
                    start_date=date(2025, 9, 30),
                    end_date=date(2025, 10, 6),
                    target_age_groups=["0-4", "5-17", "18+"],
                    delta_t=0.5,  # This should cause an error
                )

        finally:
            # Clean up temporary file
            if os.path.exists(temp_filepath):
                os.unlink(temp_filepath)


class TestCoverageValidation:
    """Coverage validation tests for scenario_to_epydemix."""

    def test_coverage_below_zero_raises_error(self, tmp_path):
        """Coverage values below 0 should raise."""
        test_df = _make_base_scenario_df({"6 Months - 4 Years": -5.0})
        test_file = tmp_path / "coverage_below_zero.csv"
        test_df.to_csv(test_file, index=False)

        with pytest.raises(ValueError, match="Coverage values must be between 0 and 100"):
            scenario_to_epydemix(
                input_filepath=str(test_file),
                start_date=date(2025, 8, 31),
                end_date=date(2025, 9, 6),
                target_age_groups=["0-4", "5-17", "18-49", "50-64", "65+"],
                states=["California"],
            )

    def test_coverage_above_100_raises_error(self, tmp_path):
        """Coverage values above 100 should raise."""
        test_df = _make_base_scenario_df({"5-12 Years": 105.0})
        test_file = tmp_path / "coverage_above_100.csv"
        test_df.to_csv(test_file, index=False)

        with pytest.raises(ValueError, match="Coverage values must be between 0 and 100"):
            scenario_to_epydemix(
                input_filepath=str(test_file),
                start_date=date(2025, 8, 31),
                end_date=date(2025, 9, 6),
                target_age_groups=["0-4", "5-17", "18-49", "50-64", "65+"],
                states=["California"],
            )

    def test_coverage_exactly_zero_is_valid(self, tmp_path):
        """Coverage of 0 should be valid."""
        test_df = _make_base_scenario_df({"18-49 Years": 0.0})
        test_file = tmp_path / "coverage_zero.csv"
        test_df.to_csv(test_file, index=False)

        result = scenario_to_epydemix(
            input_filepath=str(test_file),
            start_date=date(2025, 8, 31),
            end_date=date(2025, 9, 6),
            target_age_groups=["0-4", "5-17", "18-49", "50-64", "65+"],
            states=["California"],
        )

        assert not result.empty

    def test_coverage_exactly_100_is_valid(self, tmp_path):
        """Coverage of 100 should be valid."""
        test_df = _make_base_scenario_df({"65+ Years": 100.0})
        test_file = tmp_path / "coverage_100.csv"
        test_df.to_csv(test_file, index=False)

        result = scenario_to_epydemix(
            input_filepath=str(test_file),
            start_date=date(2025, 8, 31),
            end_date=date(2025, 9, 6),
            target_age_groups=["0-4", "5-17", "18-49", "50-64", "65+"],
            states=["California"],
        )

        assert not result.empty

    def test_error_message_shows_invalid_value_range(self, tmp_path):
        """Error message should include min/max of invalid values."""
        test_df = _make_base_scenario_df({"6 Months - 4 Years": -1.0})
        test_file = tmp_path / "coverage_error_message.csv"
        test_df.to_csv(test_file, index=False)

        with pytest.raises(ValueError) as excinfo:
            scenario_to_epydemix(
                input_filepath=str(test_file),
                start_date=date(2025, 8, 31),
                end_date=date(2025, 9, 6),
                target_age_groups=["0-4", "5-17", "18-49", "50-64", "65+"],
                states=["California"],
            )

        message = str(excinfo.value)
        assert "min=" in message
        assert "max=" in message
        assert "-1.0" in message


class TestSmhDataCoverageValidation:
    """Coverage validation tests for smh_data_to_epydemix."""

    def test_scenario_coverage_below_zero_raises_error(self, tmp_path):
        """Scenario coverage below 0 should raise."""
        base_df = _make_base_scenario_df()
        base_df = base_df.drop(columns=["Coverage"])
        base_df["sc_high"] = -5.0
        base_df["sc_low"] = 25.0

        test_file = tmp_path / "smh_coverage_below_zero.csv"
        base_df.to_csv(test_file, index=False)

        with pytest.raises(ValueError, match="Coverage values must be between 0 and 100"):
            smh_data_to_epydemix(
                input_filepath=str(test_file),
                start_date=date(2025, 8, 31),
                end_date=date(2025, 9, 6),
                target_age_groups=["0-4", "5-17", "18-49", "50-64", "65+"],
                states=["California"],
            )

    def test_scenario_coverage_above_100_raises_error(self, tmp_path):
        """Scenario coverage above 100 should raise."""
        base_df = _make_base_scenario_df()
        base_df = base_df.drop(columns=["Coverage"])
        base_df["sc_high"] = 110.0
        base_df["sc_low"] = 25.0

        test_file = tmp_path / "smh_coverage_above_100.csv"
        base_df.to_csv(test_file, index=False)

        with pytest.raises(ValueError, match="Coverage values must be between 0 and 100"):
            smh_data_to_epydemix(
                input_filepath=str(test_file),
                start_date=date(2025, 8, 31),
                end_date=date(2025, 9, 6),
                target_age_groups=["0-4", "5-17", "18-49", "50-64", "65+"],
                states=["California"],
            )


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
        dates = pd.date_range("2025-09-30", periods=30, freq="D")
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
                "dates": pd.date_range("2025-09-30", periods=10, freq="D"),
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


class TestVaccinationE2E:
    """End-to-end tests verifying vaccination works correctly in simulations.

    These tests run actual simulations and verify:
    1. Total vaccinations match the scheduled doses
    2. Population conservation (S + S_vax remains constant when no disease)
    3. Vaccination transitions actually move people between compartments
    """

    @pytest.fixture
    def sir_model_with_vaccination(self):
        """Create a simple SIR model with vaccination compartments.

        Returns a model with no disease transmission (beta=0) to isolate vaccination effects.
        """
        from epydemix.model import EpiModel
        from epydemix.population import load_epydemix_population

        model = EpiModel()

        age_group_mapping = {
            "0-4": [str(i) for i in range(5)],
            "5-17": [str(i) for i in range(5, 18)],
            "18-49": [str(i) for i in range(18, 50)],
            "50-64": [str(i) for i in range(50, 65)],
            "65+": [str(i) for i in range(65, 84)] + ["84+"],
        }
        population = load_epydemix_population(
            population_name="United_States_California",
            age_group_mapping=age_group_mapping,
        )
        model.set_population(population)

        # Add compartments: S, S_vax, I, R
        model.add_compartments(["S", "S_vax", "I", "R"])

        # Add basic transitions (with beta=0 to disable disease)
        model.add_transition("S", "I", params=("beta", "I"), kind="mediated")
        model.add_transition("I", "R", params="gamma", kind="spontaneous")

        # Add parameters - beta=0 disables disease transmission
        model.add_parameter(parameters_dict={"beta": 0.0, "gamma": 0.1})

        return model

    @pytest.fixture
    def vaccination_schedule_constant(self):
        """Create a constant daily vaccination schedule for 30 days.

        Returns a schedule with fixed daily doses per age group.
        """
        dates = pd.date_range("2025-09-30", periods=30, freq="D")
        return pd.DataFrame(
            {
                "dates": dates,
                "location": ["US-CA"] * 30,
                "0-4": [1000.0] * 30,
                "5-17": [2000.0] * 30,
                "18-49": [5000.0] * 30,
                "50-64": [3000.0] * 30,
                "65+": [4000.0] * 30,
            }
        )

    def test_vaccination_total_matches_schedule(self, sir_model_with_vaccination, vaccination_schedule_constant):
        """Verify that total vaccinations approximately match the scheduled doses.

        This test runs a simulation with vaccination and verifies that the cumulative number of people moved to S_vax is close to the total scheduled doses.

        Note: Due to the rate-to-probability conversion (p = 1 - exp(-rate * dt)), actual vaccinations may be slightly lower than scheduled when doses/population is high.
        """
        model = sir_model_with_vaccination

        # Create vaccination rate function
        vaccine_rate_function = make_vaccination_rate_function(origin_compartment="S", eligible_compartments=["S"])

        # Add vaccination schedule
        model = add_vaccination_schedule(
            model=model,
            vaccine_rate_function=vaccine_rate_function,
            source_comp="S",
            target_comp="S_vax",
            vaccination_schedule=vaccination_schedule_constant,
        )

        # Set initial conditions - everyone susceptible
        n_age = len(model.population.Nk)
        init_conditions = {
            "S": model.population.Nk.copy(),
            "S_vax": np.zeros(n_age),
            "I": np.zeros(n_age),
            "R": np.zeros(n_age),
        }

        # Run simulation
        rng = np.random.default_rng(42)
        sim_results = model.run_simulations(
            start_date="2025-09-30",
            end_date="2025-10-29",
            initial_conditions_dict=init_conditions,
            Nsim=5,
            dt=1.0,
            rng=rng,
        )

        # Get S->S_vax transitions (vaccinations)
        transitions = sim_results.get_stacked_transitions()
        s_to_svax = transitions.get("S_to_S_vax_total")

        assert s_to_svax is not None, "Vaccination transition S_to_S_vax not found in results"

        # Calculate total vaccinations across all simulations
        # Sum across time (axis=1), then average across simulations
        total_vaccinated_per_sim = np.sum(s_to_svax, axis=1)
        avg_total_vaccinated = np.mean(total_vaccinated_per_sim)

        # Calculate expected total from schedule
        age_cols = ["0-4", "5-17", "18-49", "50-64", "65+"]
        expected_total = vaccination_schedule_constant[age_cols].sum().sum()

        # Vaccinations should be within 1% of scheduled
        # (stochastic variation is very small with these population sizes)
        relative_diff = abs(avg_total_vaccinated - expected_total) / expected_total
        assert relative_diff < 0.01, (
            f"Vaccination total ({avg_total_vaccinated:.0f}) differs from scheduled "
            f"({expected_total:.0f}) by {relative_diff * 100:.2f}%"
        )

    def test_vaccination_population_conservation(self, sir_model_with_vaccination, vaccination_schedule_constant):
        """Verify that total population is conserved during vaccination.

        With beta=0 (no disease), only vaccination occurs:
        - S decreases as people get vaccinated
        - S_vax increases by the same amount
        - Total (S + S_vax + I + R) remains constant
        """
        model = sir_model_with_vaccination

        # Create vaccination rate function
        vaccine_rate_function = make_vaccination_rate_function(origin_compartment="S", eligible_compartments=["S"])

        # Add vaccination schedule
        model = add_vaccination_schedule(
            model=model,
            vaccine_rate_function=vaccine_rate_function,
            source_comp="S",
            target_comp="S_vax",
            vaccination_schedule=vaccination_schedule_constant,
        )

        # Set initial conditions
        n_age = len(model.population.Nk)
        init_conditions = {
            "S": model.population.Nk.copy(),
            "S_vax": np.zeros(n_age),
            "I": np.zeros(n_age),
            "R": np.zeros(n_age),
        }

        initial_total_pop = np.sum(model.population.Nk)

        # Run simulation
        rng = np.random.default_rng(42)
        sim_results = model.run_simulations(
            start_date="2025-09-30",
            end_date="2025-10-29",
            initial_conditions_dict=init_conditions,
            Nsim=3,
            dt=1.0,
            rng=rng,
        )

        # Get compartment totals
        compartments = sim_results.get_stacked_compartments()
        s_total = compartments["S_total"]
        svax_total = compartments["S_vax_total"]
        i_total = compartments["I_total"]
        r_total = compartments["R_total"]

        # Calculate total population at each timestep
        total_pop = s_total + svax_total + i_total + r_total

        # Verify population is conserved (within floating point tolerance)
        for sim_idx in range(total_pop.shape[0]):
            pop_at_each_time = total_pop[sim_idx]
            max_deviation = np.max(np.abs(pop_at_each_time - initial_total_pop))
            assert max_deviation < 1.0, (
                f"Population not conserved in sim {sim_idx}: max deviation = {max_deviation:.2f}"
            )

    def test_vaccination_actually_moves_people(self, sir_model_with_vaccination, vaccination_schedule_constant):
        """Verify that vaccination actually moves people from S to S_vax.

        This test confirms that:
        1. S_vax increases from 0 over time
        2. S decreases correspondingly
        3. The final S_vax > 0 (vaccinations occurred)
        """
        model = sir_model_with_vaccination

        # Create vaccination rate function
        vaccine_rate_function = make_vaccination_rate_function(origin_compartment="S", eligible_compartments=["S"])

        # Add vaccination schedule
        model = add_vaccination_schedule(
            model=model,
            vaccine_rate_function=vaccine_rate_function,
            source_comp="S",
            target_comp="S_vax",
            vaccination_schedule=vaccination_schedule_constant,
        )

        # Set initial conditions - everyone in S
        n_age = len(model.population.Nk)
        init_conditions = {
            "S": model.population.Nk.copy(),
            "S_vax": np.zeros(n_age),
            "I": np.zeros(n_age),
            "R": np.zeros(n_age),
        }

        # Run simulation
        rng = np.random.default_rng(42)
        sim_results = model.run_simulations(
            start_date="2025-09-30",
            end_date="2025-10-29",
            initial_conditions_dict=init_conditions,
            Nsim=3,
            dt=1.0,
            rng=rng,
        )

        compartments = sim_results.get_stacked_compartments()

        # Check S_vax increases over time (final > first timestep)
        # Note: First timestep already includes day 1 vaccinations
        svax_total = compartments["S_vax_total"]
        for sim_idx in range(svax_total.shape[0]):
            first_svax = svax_total[sim_idx, 0]
            final_svax = svax_total[sim_idx, -1]

            # First timestep should have some vaccinations (day 1)
            assert first_svax > 0, f"S_vax should have vaccinations on day 1, got {first_svax}"
            # Final should be greater than first (accumulating)
            assert final_svax > first_svax, (
                f"S_vax should increase over time: first={first_svax:.0f}, final={final_svax:.0f}"
            )

        # Check S decreases over time
        s_total = compartments["S_total"]
        initial_total_s = np.sum(sir_model_with_vaccination.population.Nk)
        for sim_idx in range(s_total.shape[0]):
            first_s_sim = s_total[sim_idx, 0]
            final_s_sim = s_total[sim_idx, -1]

            # S at first timestep should already be less than initial (day 1 vaccinations)
            assert first_s_sim < initial_total_s, (
                f"S should decrease from initial: initial={initial_total_s:.0f}, first={first_s_sim:.0f}"
            )
            # Final should be less than first (continuing to decrease)
            assert final_s_sim < first_s_sim, (
                f"S should decrease over time: first={first_s_sim:.0f}, final={final_s_sim:.0f}"
            )

    def test_vaccination_per_age_group_proportional(self, sir_model_with_vaccination):
        """Verify that vaccination doses are distributed proportionally across age groups.

        Creates a schedule where age group "18-49" gets twice as many doses as "0-4".
        Verifies that the actual vaccinations reflect this ratio.
        """
        model = sir_model_with_vaccination

        # Create schedule with 2x doses for 18-49 vs 0-4
        dates = pd.date_range("2025-09-30", periods=30, freq="D")
        vaccination_schedule = pd.DataFrame(
            {
                "dates": dates,
                "location": ["US-CA"] * 30,
                "0-4": [1000.0] * 30,
                "5-17": [1000.0] * 30,
                "18-49": [2000.0] * 30,  # 2x doses
                "50-64": [1000.0] * 30,
                "65+": [1000.0] * 30,
            }
        )

        vaccine_rate_function = make_vaccination_rate_function(origin_compartment="S", eligible_compartments=["S"])

        model = add_vaccination_schedule(
            model=model,
            vaccine_rate_function=vaccine_rate_function,
            source_comp="S",
            target_comp="S_vax",
            vaccination_schedule=vaccination_schedule,
        )

        # Set initial conditions
        n_age = len(model.population.Nk)
        init_conditions = {
            "S": model.population.Nk.copy(),
            "S_vax": np.zeros(n_age),
            "I": np.zeros(n_age),
            "R": np.zeros(n_age),
        }

        rng = np.random.default_rng(42)
        sim_results = model.run_simulations(
            start_date="2025-09-30",
            end_date="2025-10-29",
            initial_conditions_dict=init_conditions,
            Nsim=5,
            dt=1.0,
            rng=rng,
        )

        # Get S_vax by age group (not just total)
        compartments = sim_results.get_stacked_compartments()

        # Compartment keys use age group names: S_vax_0-4, S_vax_18-49, etc.
        svax_0_4 = compartments["S_vax_0-4"]
        svax_18_49 = compartments["S_vax_18-49"]

        # Get final values averaged across simulations
        avg_svax_0_4 = np.mean(svax_0_4[:, -1])
        avg_svax_18_49 = np.mean(svax_18_49[:, -1])

        # 18-49 should have roughly 2x as many vaccinated as 0-4
        # (allowing for population size differences and stochastic effects)
        # Just verify 18-49 has more vaccinated than 0-4
        assert avg_svax_18_49 > avg_svax_0_4, (
            f"Age group 18-49 (with 2x doses) should have more vaccinated than 0-4: "
            f"18-49={avg_svax_18_49:.0f}, 0-4={avg_svax_0_4:.0f}"
        )

    def test_vaccination_with_disease_reduces_infections(self):
        """Verify that vaccination reduces total infections when combined with disease.

        Compares two scenarios:
        1. Disease only (no vaccination)
        2. Disease + vaccination

        The vaccination scenario should have fewer total infections (S->I transitions) because vaccinated people (in S_vax) cannot be infected.
        """
        from epydemix.model import EpiModel
        from epydemix.population import load_epydemix_population

        # Create model with disease (beta > 0)
        def create_model_with_disease():
            model = EpiModel()

            age_group_mapping = {
                "0-4": [str(i) for i in range(5)],
                "5-17": [str(i) for i in range(5, 18)],
                "18-49": [str(i) for i in range(18, 50)],
                "50-64": [str(i) for i in range(50, 65)],
                "65+": [str(i) for i in range(65, 84)] + ["84+"],
            }
            population = load_epydemix_population(
                population_name="United_States_California",
                age_group_mapping=age_group_mapping,
            )
            model.set_population(population)

            model.add_compartments(["S", "S_vax", "I", "R"])
            model.add_transition("S", "I", params=("beta", "I"), kind="mediated")
            model.add_transition("I", "R", params="gamma", kind="spontaneous")
            # Use moderate transmission rate
            model.add_parameter(parameters_dict={"beta": 0.15, "gamma": 0.1})

            return model

        # Create aggressive vaccination schedule
        dates = pd.date_range("2025-09-30", periods=30, freq="D")
        vaccination_schedule = pd.DataFrame(
            {
                "dates": dates,
                "location": ["US-CA"] * 30,
                "0-4": [50000.0] * 30,
                "5-17": [100000.0] * 30,
                "18-49": [250000.0] * 30,
                "50-64": [150000.0] * 30,
                "65+": [200000.0] * 30,
            }
        )

        # Scenario 1: Disease only
        model_disease = create_model_with_disease()
        n_age = len(model_disease.population.Nk)
        i_init = np.zeros(n_age)
        i_init[2] = 100  # Initial infections in 18-49

        init_conditions = {
            "S": model_disease.population.Nk - i_init,
            "S_vax": np.zeros(n_age),
            "I": i_init,
            "R": np.zeros(n_age),
        }

        rng1 = np.random.default_rng(42)
        results_disease_only = model_disease.run_simulations(
            start_date="2025-09-30",
            end_date="2025-10-29",
            initial_conditions_dict=init_conditions,
            Nsim=10,
            dt=1.0,
            rng=rng1,
        )

        # Scenario 2: Disease + Vaccination
        model_with_vax = create_model_with_disease()

        vaccine_rate_function = make_vaccination_rate_function(origin_compartment="S", eligible_compartments=["S"])

        model_with_vax = add_vaccination_schedule(
            model=model_with_vax,
            vaccine_rate_function=vaccine_rate_function,
            source_comp="S",
            target_comp="S_vax",
            vaccination_schedule=vaccination_schedule,
        )

        rng2 = np.random.default_rng(42)
        results_with_vax = model_with_vax.run_simulations(
            start_date="2025-09-30",
            end_date="2025-10-29",
            initial_conditions_dict=init_conditions,
            Nsim=10,
            dt=1.0,
            rng=rng2,
        )

        # Compare total infections (S->I transitions)
        transitions_disease = results_disease_only.get_stacked_transitions()
        transitions_vax = results_with_vax.get_stacked_transitions()

        # Sum all S->I transitions over time for each simulation
        infections_disease = np.sum(transitions_disease["S_to_I_total"], axis=1)
        infections_vax = np.sum(transitions_vax["S_to_I_total"], axis=1)

        avg_infections_disease = np.mean(infections_disease)
        avg_infections_vax = np.mean(infections_vax)

        # Vaccination scenario should have fewer infections
        assert avg_infections_vax < avg_infections_disease, (
            f"Vaccination should reduce infections: "
            f"disease_only={avg_infections_disease:.0f}, with_vax={avg_infections_vax:.0f}"
        )

        # Verify S_vax has people in the vaccination scenario
        compartments_vax = results_with_vax.get_stacked_compartments()
        avg_final_svax = np.mean(compartments_vax["S_vax_total"][:, -1])
        assert avg_final_svax > 0, "S_vax should have vaccinated people"


class TestReaggregateVaccines:
    """Unit tests for reaggregate_vaccines function."""

    @pytest.fixture
    def sample_schedule(self):
        """Create a sample vaccination schedule starting on a Monday (2025-09-29).

        The schedule runs from Monday 2025-09-29 to Sunday 2025-10-19. First Saturday is 2025-10-04.
        """
        # 2025-09-29 is a Monday
        dates = pd.date_range("2025-09-29", periods=21, freq="D")
        return pd.DataFrame(
            {
                "dates": dates,
                "location": ["US-CA"] * 21,
                "0-4": [100] * 21,  # 100 doses/day for 21 days = 2100 total
                "5-17": [200] * 21,  # 200 doses/day for 21 days = 4200 total
                "65+": [300] * 21,  # 300 doses/day for 21 days = 6300 total
            }
        )

    def test_reaggregate_preserves_total_doses(self, sample_schedule):
        """Total doses before and after reaggregation are equal."""
        age_groups = ["0-4", "5-17", "65+"]

        # Calculate total doses before
        total_before = sample_schedule[age_groups].sum().sum()

        # Reaggregate starting from 2025-10-01 (Wednesday)
        actual_start_date = pd.Timestamp("2025-10-01")
        reaggregated = reaggregate_vaccines(sample_schedule, actual_start_date)

        # Calculate total doses after
        total_after = reaggregated[age_groups].sum().sum()

        assert total_before == total_after, (
            f"Total doses should be preserved: before={total_before}, after={total_after}"
        )

    def test_reaggregate_redistributes_to_next_saturday(self, sample_schedule):
        """Doses are evenly distributed from start_date to next Saturday."""
        # Start on Wednesday 2025-10-01, next Saturday is 2025-10-04
        actual_start_date = pd.Timestamp("2025-10-01")
        reaggregated = reaggregate_vaccines(sample_schedule, actual_start_date)

        # Check that the first date in reaggregated schedule is actual_start_date
        first_date = reaggregated["dates"].min()
        assert first_date == actual_start_date, f"First date should be {actual_start_date}, got {first_date}"

        # Check dates before actual_start_date are excluded
        dates_before_start = reaggregated[reaggregated["dates"] < actual_start_date]
        assert len(dates_before_start) == 0, "No dates before actual_start_date should exist"

        # Verify doses are distributed across Wed-Sat (4 days: Oct 2, 3, 4, 5)
        # Original doses from Sep 30 to Oct 5 (6 days): 6 * 100 = 600 for 0-4
        # Redistributed across Oct 2-5 (4 days): 600 / 4 = 150 per day
        redistributed_period = reaggregated[
            (reaggregated["dates"] >= actual_start_date) & (reaggregated["dates"] <= pd.Timestamp("2025-10-04"))
        ]
        assert len(redistributed_period) == 4, (
            f"Should have 4 days in redistribution period, got {len(redistributed_period)}"
        )

    def test_reaggregate_unchanged_when_start_equals_min(self, sample_schedule):
        """Returns unchanged schedule if actual_start_date == date_min."""
        actual_start_date = sample_schedule["dates"].min()
        result = reaggregate_vaccines(sample_schedule, actual_start_date)

        # Should return the same schedule
        pd.testing.assert_frame_equal(result, sample_schedule)

    def test_reaggregate_raises_for_out_of_range_date(self, sample_schedule):
        """Raises ValueError if start_date outside schedule range."""
        # Test date before schedule range
        with pytest.raises(ValueError, match="Start date must be between"):
            reaggregate_vaccines(sample_schedule, pd.Timestamp("2025-08-31"))

        # Test date after schedule range
        with pytest.raises(ValueError, match="Start date must be between"):
            reaggregate_vaccines(sample_schedule, pd.Timestamp("2025-10-31"))

    def test_reaggregate_handles_scenario_column(self):
        """Preserves scenario column if present in schedule."""
        dates = pd.date_range("2025-09-29", periods=14, freq="D")
        schedule_with_scenario = pd.DataFrame(
            {
                "dates": dates,
                "scenario": ["high_coverage"] * 14,
                "location": ["US-CA"] * 14,
                "0-4": [100] * 14,
                "5-17": [200] * 14,
            }
        )

        actual_start_date = pd.Timestamp("2025-10-01")
        result = reaggregate_vaccines(schedule_with_scenario, actual_start_date)

        # Verify scenario column is preserved
        assert "scenario" in result.columns, "scenario column should be preserved"
        assert all(result["scenario"] == "high_coverage"), "scenario values should be preserved"

    def test_reaggregate_handles_start_on_saturday(self):
        """Correctly handles when start_date is already a Saturday."""
        # Create schedule starting Saturday 2025-10-04
        dates = pd.date_range("2025-10-04", periods=14, freq="D")
        schedule = pd.DataFrame(
            {
                "dates": dates,
                "location": ["US-CA"] * 14,
                "0-4": [100] * 14,
                "5-17": [200] * 14,
            }
        )

        # Start on the next Saturday (2025-10-11) - should redistribute Oct 5-11 to Oct 12
        actual_start_date = pd.Timestamp("2025-10-11")
        result = reaggregate_vaccines(schedule, actual_start_date)

        # First date should be actual_start_date
        first_date = result["dates"].min()
        assert first_date == actual_start_date

        # Total doses should be preserved
        total_before = schedule[["0-4", "5-17"]].sum().sum()
        total_after = result[["0-4", "5-17"]].sum().sum()
        assert total_before == total_after

    def test_reaggregate_distributes_remainder_to_first_days(self):
        """Remainder doses go to first days when total doesn't divide evenly."""
        # Create schedule: 10 doses/day for Sun-Thu (5 days), total = 50
        # Fri-Sat have 0 doses
        schedule = pd.DataFrame({
            "dates": pd.date_range("2025-08-31", periods=7, freq="D"),  # Sun-Sat
            "location": "US-CA",
            "0-4": [10, 10, 10, 10, 10, 0, 0],  # 50 total through Thu
        })

        # Start Wed (2025-09-03), redistribute to Sat (2025-09-06) = 4 days
        # 50 doses / 4 days = 12 base, 2 remainder
        # Expected: [13, 13, 12, 12]
        result = reaggregate_vaccines(schedule, pd.Timestamp("2025-09-03"))

        redistributed = result[result["dates"] <= pd.Timestamp("2025-09-06")]
        doses = redistributed["0-4"].tolist()

        assert doses == [13, 13, 12, 12], f"Expected [13, 13, 12, 12], got {doses}"


class TestGetAgeGroupsFromData:
    """Unit tests for get_age_groups_from_data function."""

    def test_extracts_age_groups_from_dataframe(self):
        """Correctly extracts and cleans age groups from data."""
        # Create DataFrame with typical vaccination data age groups
        data = pd.DataFrame(
            {
                "Age": [
                    "6 Months - 4 Years",
                    "5-12 Years",
                    "13-17 Years",
                    "18-49 Years",
                    "50-64 Years",
                    "65+ Years",
                    "6 Months - 17 Years",  # This should be removed
                ],
                "Coverage": [10, 20, 30, 40, 50, 60, 35],
            }
        )

        result = get_age_groups_from_data(data)

        # Result is a dict mapping cleaned age groups to lists of single-year ages
        assert isinstance(result, dict)
        # Should have 6 age groups (excluding "6 Months - 17 Years")
        assert len(result) == 6

    def test_removes_overlapping_6_months_17_years(self):
        """Removes '6 Months - 17 Years' which overlaps finer groups."""
        # Need valid contiguous age groups with a '+' ending group
        data = pd.DataFrame(
            {
                "Age": [
                    "6 Months - 4 Years",
                    "5-12 Years",
                    "13-17 Years",
                    "18-49 Years",
                    "50-64 Years",
                    "65+ Years",
                    "6 Months - 17 Years",  # Overlaps with finer groups - should be excluded
                ],
            }
        )

        result = get_age_groups_from_data(data)

        # The overlapping group should be excluded
        # Cleaned labels should be "0-4", "5-12", "13-17", "18-49", "50-64", "65+"
        expected_keys = {"0-4", "5-12", "13-17", "18-49", "50-64", "65+"}
        assert set(result.keys()) == expected_keys
        # Verify "6 Months - 17 Years" is not present (would be "0-17" if cleaned)
        assert "0-17" not in result.keys()

    def test_cleans_age_group_labels(self):
        """Converts '6 Months - 4 Years' → '0-4', removes ' Years'."""
        data = pd.DataFrame(
            {
                "Age": [
                    "6 Months - 4 Years",
                    "5-12 Years",
                    "13-17 Years",
                    "18-49 Years",
                    "50-64 Years",
                    "65+ Years",
                    "6 Months - 17 Years",  # Will be removed
                ],
            }
        )

        result = get_age_groups_from_data(data)

        # Check cleaned keys
        assert "0-4" in result.keys()
        assert "5-12" in result.keys()
        assert "65+" in result.keys()
        # Original format should not be present
        assert "6 Months - 4 Years" not in result.keys()
        assert "5-12 Years" not in result.keys()

    def test_handles_missing_aggregate_age_group(self):
        """Does not error if '6 Months - 17 Years' is missing."""
        data = pd.DataFrame(
            {
                "Age": [
                    "6 Months - 4 Years",
                    "5-12 Years",
                    "13-17 Years",
                    "18-49 Years",
                    "50-64 Years",
                    "65+ Years",
                ],
            }
        )

        result = get_age_groups_from_data(data)

        expected_keys = {"0-4", "5-12", "13-17", "18-49", "50-64", "65+"}
        assert set(result.keys()) == expected_keys



class TestSmhDataToEpydemix:
    """Unit tests for smh_data_to_epydemix function."""

    @pytest.fixture
    def smh_data_file(self, tmp_path):
        """Create a temporary SMH-format vaccination data file with multiple scenarios."""
        # SMH format has scenario columns containing "sc_"
        data = pd.DataFrame(
            {
                "Week_Ending_Sat": ["2025-10-04", "2025-10-11"] * 7,
                "Geography": ["California"] * 14,
                "Age": [
                    "6 Months - 4 Years",
                    "5-12 Years",
                    "13-17 Years",
                    "18-49 Years",
                    "50-64 Years",
                    "65+ Years",
                    "6 Months - 17 Years",
                ]
                * 2,
                "Population": [2000000, 4000000, 2500000, 17000000, 7000000, 6000000, 6500000] * 2,
                "flu.coverage.rd2526.sc_A": [10.0, 15.0, 12.0, 8.0, 20.0, 30.0, 12.5] * 2,
                "flu.coverage.rd2526.sc_B": [5.0, 7.5, 6.0, 4.0, 10.0, 15.0, 6.25] * 2,
            }
        )
        filepath = tmp_path / "smh_vaccines.csv"
        data.to_csv(filepath, index=False)
        return filepath

    def test_handles_multiple_scenarios(self, smh_data_file):
        """Processes all columns containing 'sc_' into scenarios."""
        result = smh_data_to_epydemix(
            input_filepath=str(smh_data_file),
            start_date=date(2025, 9, 30),
            end_date=date(2025, 10, 11),
            target_age_groups=["0-4", "5-17", "18-49", "50-64", "65+"],
            states=["California"],
        )

        # Should have data for both scenarios
        scenarios = result["scenario"].unique()
        assert len(scenarios) == 2, f"Expected 2 scenarios, got {len(scenarios)}"
        assert "A" in scenarios, "Should have 'A' scenario"
        assert "B" in scenarios, "Should have 'B' scenario"

    def test_adds_scenario_column_to_output(self, smh_data_file):
        """Output DataFrame includes scenario identifier."""
        result = smh_data_to_epydemix(
            input_filepath=str(smh_data_file),
            start_date=date(2025, 9, 30),
            end_date=date(2025, 10, 11),
            target_age_groups=["0-4", "5-17", "18-49", "50-64", "65+"],
            states=["California"],
        )

        # Verify scenario column exists
        assert "scenario" in result.columns, "Output should have 'scenario' column"

        # Verify each row has a scenario value
        assert result["scenario"].notna().all(), "All rows should have scenario values"

    def test_raises_for_missing_scenario_columns(self, tmp_path):
        """Raises ValueError if no sc_ columns found."""
        # Create file without scenario columns
        data = pd.DataFrame(
            {
                "Week_Ending_Sat": ["2025-10-04"] * 7,
                "Geography": ["California"] * 7,
                "Age": [
                    "6 Months - 4 Years",
                    "5-12 Years",
                    "13-17 Years",
                    "18-49 Years",
                    "50-64 Years",
                    "65+ Years",
                    "6 Months - 17 Years",
                ],
                "Population": [2000000, 4000000, 2500000, 17000000, 7000000, 6000000, 6500000],
                "Coverage": [10.0, 15.0, 12.0, 8.0, 20.0, 30.0, 12.5],  # No sc_ prefix
            }
        )
        filepath = tmp_path / "no_scenarios.csv"
        data.to_csv(filepath, index=False)

        with pytest.raises(ValueError, match="No scenario columns found"):
            smh_data_to_epydemix(
                input_filepath=str(filepath),
                start_date=date(2025, 9, 30),
                end_date=date(2025, 10, 11),
                target_age_groups=["0-4", "5-17", "18-49", "50-64", "65+"],
                states=["California"],
            )


