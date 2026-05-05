"""Tests for epymodelingsuite.builders.base module."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from epymodelingsuite.builders.base import (
    _parse_age_group,
    calculate_compartment_initial_conditions,
    load_iso_population,
    load_metrocast_population,
)
from epymodelingsuite.utils.location import get_metrocast_population_data


@dataclass
class DummyCompartment:
    """Mock compartment for testing."""

    id: str
    init: float | int | str | list[float | int]


class TestCalculateCompartmentInitialConditions:
    """Tests for calculate_compartment_initial_conditions function."""

    @pytest.fixture
    def population_array(self):
        """Sample population array with 5 age groups."""
        return np.array([10000, 20000, 30000, 40000, 50000])

    @pytest.fixture
    def total_population(self, population_array):
        """Total population sum."""
        return sum(population_array)

    def test_no_compartments_returns_none(self, population_array):
        """Test that empty compartment list returns None."""
        result = calculate_compartment_initial_conditions([], population_array)
        assert result is None

    def test_count_initialization_distributes_proportionally(self, population_array, total_population):
        """Test that counts (init >= 1) are distributed proportionally across age groups."""
        compartments = [
            DummyCompartment(id="L", init=10),
            DummyCompartment(id="I", init=100),
        ]

        result = calculate_compartment_initial_conditions(compartments, population_array)

        # Check that counts are distributed proportionally
        assert "L" in result
        assert "I" in result

        # Check L compartment distribution
        expected_L = 10 * population_array / total_population
        np.testing.assert_array_almost_equal(result["L"], expected_L)

        # Check I compartment distribution
        expected_I = 100 * population_array / total_population
        np.testing.assert_array_almost_equal(result["I"], expected_I)

        # Verify sum equals original count
        assert np.isclose(sum(result["L"]), 10)
        assert np.isclose(sum(result["I"]), 100)

    def test_proportion_initialization_applies_to_population(self, population_array):
        """Test that proportions (init < 1) are applied directly to population array."""
        compartments = [
            DummyCompartment(id="I", init=0.02),
            DummyCompartment(id="R", init=0.5),
        ]

        result = calculate_compartment_initial_conditions(compartments, population_array)

        # Check that proportions are applied to each age group
        expected_I = 0.02 * population_array
        expected_R = 0.5 * population_array

        np.testing.assert_array_almost_equal(result["I"], expected_I)
        np.testing.assert_array_almost_equal(result["R"], expected_R)

    def test_default_initialization_uses_remaining_population(self, population_array, total_population):
        """Test that default compartments get remaining population distributed per age group."""
        compartments = [
            DummyCompartment(id="S", init="default"),
            DummyCompartment(id="I", init=0.02),
        ]

        result = calculate_compartment_initial_conditions(compartments, population_array)

        # Calculate expected remaining population per age group
        sum_age_structured = 0.02 * population_array
        expected_remaining = population_array - sum_age_structured

        np.testing.assert_array_almost_equal(result["S"], expected_remaining)

        # Verify population conservation
        total_initial = sum(result["S"]) + sum(result["I"])
        assert np.isclose(total_initial, total_population)

    def test_multiple_default_compartments_split_remaining_equally(self, population_array, total_population):
        """Test that multiple default compartments split remaining population equally per age group."""
        compartments = [
            DummyCompartment(id="S", init="default"),
            DummyCompartment(id="S_vax", init="default"),
            DummyCompartment(id="I", init=100),
        ]

        result = calculate_compartment_initial_conditions(compartments, population_array)

        # Calculate expected distribution
        count_distributed = 100 * population_array / total_population
        remaining = population_array - count_distributed

        expected_per_default = remaining / 2  # 2 default compartments

        np.testing.assert_array_almost_equal(result["S"], expected_per_default)
        np.testing.assert_array_almost_equal(result["S_vax"], expected_per_default)

        # Verify population conservation
        total_initial = sum(result["S"]) + sum(result["S_vax"]) + sum(result["I"])
        assert np.isclose(total_initial, total_population)

    def test_mixed_initialization_types(self, population_array, total_population):
        """Test combination of counts, proportions, and default."""
        compartments = [
            DummyCompartment(id="S", init="default"),
            DummyCompartment(id="L", init=10),  # count
            DummyCompartment(id="I", init=0.02),  # proportion
        ]

        result = calculate_compartment_initial_conditions(compartments, population_array)

        # All should be numpy arrays
        assert isinstance(result["S"], np.ndarray)
        assert isinstance(result["L"], np.ndarray)
        assert isinstance(result["I"], np.ndarray)

        # All should have same shape as population
        assert result["S"].shape == population_array.shape
        assert result["L"].shape == population_array.shape
        assert result["I"].shape == population_array.shape

        # Verify population conservation
        total_initial = sum(result["S"]) + sum(result["L"]) + sum(result["I"])
        assert np.isclose(total_initial, total_population)

    def test_sampled_compartments_count_override(self, population_array, total_population):
        """Test that sampled compartments with counts override base configuration."""
        compartments = [
            DummyCompartment(id="L", init=10),
            DummyCompartment(id="I", init=100),
        ]

        sampled_compartments = {"L": 50}  # Override L with higher count

        result = calculate_compartment_initial_conditions(compartments, population_array, sampled_compartments)

        # L should use sampled value
        expected_L = 50 * population_array / total_population
        np.testing.assert_array_almost_equal(result["L"], expected_L)
        assert np.isclose(sum(result["L"]), 50)

        # I should use original value
        expected_I = 100 * population_array / total_population
        np.testing.assert_array_almost_equal(result["I"], expected_I)
        assert np.isclose(sum(result["I"]), 100)

    def test_sampled_compartments_proportion_override(self, population_array):
        """Test that sampled compartments with proportions override base configuration."""
        compartments = [
            DummyCompartment(id="I", init=0.02),
        ]

        sampled_compartments = {"I": 0.05}  # Override with higher proportion

        result = calculate_compartment_initial_conditions(compartments, population_array, sampled_compartments)

        # I should use sampled proportion
        expected_I = 0.05 * population_array
        np.testing.assert_array_almost_equal(result["I"], expected_I)

    def test_sampled_compartments_only_affect_valid_ids(self, population_array):
        """Test that sampled compartments with invalid IDs are ignored."""
        compartments = [
            DummyCompartment(id="L", init=10),
        ]

        sampled_compartments = {
            "L": 50,  # Valid compartment
            "X": 100,  # Invalid compartment - should be ignored
        }

        result = calculate_compartment_initial_conditions(compartments, population_array, sampled_compartments)

        # Only L should be in result
        assert "L" in result
        assert "X" not in result

    def test_with_default_and_sampled_compartments(self, population_array, total_population):
        """Test that default compartments work correctly with sampled compartments."""
        compartments = [
            DummyCompartment(id="S", init="default"),
            DummyCompartment(id="L", init=10),
        ]

        sampled_compartments = {"L": 50}

        result = calculate_compartment_initial_conditions(compartments, population_array, sampled_compartments)

        # L should use sampled value
        sampled_L = 50 * population_array / total_population

        # S should get remaining population
        remaining = population_array - sampled_L
        np.testing.assert_array_almost_equal(result["S"], remaining)

        # Verify population conservation
        total_initial = sum(result["S"]) + sum(result["L"])
        assert np.isclose(total_initial, total_population)

    def test_none_sampled_compartments_is_handled(self, population_array):
        """Test that None sampled_compartments parameter is handled correctly."""
        compartments = [DummyCompartment(id="I", init=10)]

        result = calculate_compartment_initial_conditions(compartments, population_array, None)

        assert "I" in result
        assert isinstance(result["I"], np.ndarray)

    def test_empty_sampled_compartments_dict(self, population_array):
        """Test that empty sampled_compartments dict is handled correctly."""
        compartments = [DummyCompartment(id="I", init=10)]

        result = calculate_compartment_initial_conditions(compartments, population_array, {})

        assert "I" in result
        assert isinstance(result["I"], np.ndarray)

    def test_population_conservation_complex_scenario(self, population_array, total_population):
        """Test population conservation in a complex scenario with all initialization types."""
        compartments = [
            DummyCompartment(id="S", init="default"),
            DummyCompartment(id="S_vax", init="default"),
            DummyCompartment(id="L", init=10),
            DummyCompartment(id="L_vax", init=5),
            DummyCompartment(id="I", init=0.02),
            DummyCompartment(id="I_vax", init=0.01),
            DummyCompartment(id="R", init=100),
            DummyCompartment(id="R_vax", init=50),
        ]

        sampled_compartments = {
            "L": 20,  # Override count
            "I_vax": 0.005,  # Override proportion
        }

        result = calculate_compartment_initial_conditions(compartments, population_array, sampled_compartments)

        # Sum all compartments
        total_initial = sum(sum(result[comp.id]) for comp in compartments)

        # Should conserve population
        assert np.isclose(total_initial, total_population, rtol=1e-5)

    def test_all_results_are_numpy_arrays(self, population_array):
        """Test that all returned values are numpy arrays, not scalars."""
        compartments = [
            DummyCompartment(id="S", init="default"),
            DummyCompartment(id="L", init=10),
            DummyCompartment(id="I", init=0.02),
        ]

        result = calculate_compartment_initial_conditions(compartments, population_array)

        # All values should be numpy arrays
        for key, val in result.items():
            assert isinstance(val, np.ndarray), f"{key} should be array, got {type(val)}"
            assert val.shape == population_array.shape, f"{key} should match population shape"

    def test_numpy_numeric_types_handled(self, population_array, total_population):
        """Test that numpy numeric types (np.int64, np.float64) are handled correctly."""
        compartments = [
            DummyCompartment(id="L", init=np.int64(10)),
            DummyCompartment(id="I", init=np.float64(0.02)),
        ]

        result = calculate_compartment_initial_conditions(compartments, population_array)

        # Should handle numpy types same as Python types
        expected_L = 10 * population_array / total_population
        expected_I = 0.02 * population_array

        np.testing.assert_array_almost_equal(result["L"], expected_L)
        np.testing.assert_array_almost_equal(result["I"], expected_I)

    def test_edge_case_init_exactly_one(self, population_array, total_population):
        """Test boundary case where init == 1.0 is treated as count."""
        compartments = [
            DummyCompartment(id="L", init=1.0),  # Exactly 1.0 should be treated as count
        ]

        result = calculate_compartment_initial_conditions(compartments, population_array)

        # Should be distributed as a count
        expected = 1.0 * population_array / total_population
        np.testing.assert_array_almost_equal(result["L"], expected)
        assert np.isclose(sum(result["L"]), 1.0)

    def test_edge_case_init_just_below_one(self, population_array):
        """Test boundary case where init is just below 1.0 is treated as proportion."""
        compartments = [
            DummyCompartment(id="I", init=0.9999),  # Just below 1.0 should be proportion
        ]

        result = calculate_compartment_initial_conditions(compartments, population_array)

        # Should be applied as proportion
        expected = 0.9999 * population_array
        np.testing.assert_array_almost_equal(result["I"], expected)

    def test_age_varying_proportion_initialization(self, population_array):
        """Test age-varying initialization with proportions."""
        compartments = [
            DummyCompartment(id="M", init=[0.3, 0.0, 0.0, 0.0, 0.0]),  # Maternal immunity in first age group
        ]

        result = calculate_compartment_initial_conditions(compartments, population_array)

        # Should apply proportions to each age group
        expected = np.array(
            [
                population_array[0] * 0.3,
                population_array[1] * 0.0,
                population_array[2] * 0.0,
                population_array[3] * 0.0,
                population_array[4] * 0.0,
            ]
        )
        np.testing.assert_array_almost_equal(result["M"], expected)

    def test_age_varying_count_initialization(self, population_array):
        """Test age-varying initialization with counts."""
        compartments = [
            DummyCompartment(id="I", init=[10, 20, 30, 40, 50]),  # Specific counts per age group
        ]

        result = calculate_compartment_initial_conditions(compartments, population_array)

        # Counts should be applied directly to each age group
        expected = np.array([10, 20, 30, 40, 50], dtype=float)
        np.testing.assert_array_almost_equal(result["I"], expected)

    def test_age_varying_mixed_values(self, population_array):
        """Test age-varying initialization with mixed counts and proportions."""
        compartments = [
            DummyCompartment(id="I", init=[0.1, 10, 0.05, 0, 100]),  # Mix of proportions, counts, and zeros
        ]

        result = calculate_compartment_initial_conditions(compartments, population_array)

        # Mixed values should be handled correctly per age group
        expected = np.array(
            [
                population_array[0] * 0.1,  # Proportion
                10,  # Count
                population_array[2] * 0.05,  # Proportion
                0,  # Zero
                100,  # Count
            ],
            dtype=float,
        )
        np.testing.assert_array_almost_equal(result["I"], expected)

    def test_age_varying_with_default_compartment(self, population_array, total_population):
        """Test age-varying initialization combined with default compartment."""
        compartments = [
            DummyCompartment(id="S", init="default"),
            DummyCompartment(id="M", init=[0.3, 0.0, 0.0, 0.0, 0.0]),
            DummyCompartment(id="I", init=[0.02, 0.01, 0.0, 0.0, 0.0]),
        ]

        result = calculate_compartment_initial_conditions(compartments, population_array)

        # Calculate expected values
        expected_M = np.array([population_array[0] * 0.3, 0, 0, 0, 0])
        expected_I = np.array([population_array[0] * 0.02, population_array[1] * 0.01, 0, 0, 0])
        expected_S = population_array - expected_M - expected_I

        np.testing.assert_array_almost_equal(result["M"], expected_M)
        np.testing.assert_array_almost_equal(result["I"], expected_I)
        np.testing.assert_array_almost_equal(result["S"], expected_S)

        # Verify population conservation
        total_initial = sum(result["S"]) + sum(result["M"]) + sum(result["I"])
        assert np.isclose(total_initial, total_population)

    def test_age_varying_and_scalar_mixed(self, population_array, total_population):
        """Test mixing age-varying and scalar initialization types."""
        compartments = [
            DummyCompartment(id="S", init="default"),
            DummyCompartment(id="M", init=[0.3, 0.0, 0.0, 0.0, 0.0]),  # Age-varying
            DummyCompartment(id="I", init=0.02),  # Scalar proportion
            DummyCompartment(id="R", init=100),  # Scalar count
        ]

        result = calculate_compartment_initial_conditions(compartments, population_array)

        # All compartments should have correct values
        assert "S" in result
        assert "M" in result
        assert "I" in result
        assert "R" in result

        # Verify population conservation
        total_initial = sum(result["S"]) + sum(result["M"]) + sum(result["I"]) + sum(result["R"])
        assert np.isclose(total_initial, total_population, rtol=1e-5)

    def test_age_varying_all_zeros(self, population_array):
        """Test age-varying initialization with all zeros."""
        compartments = [
            DummyCompartment(id="I", init=[0, 0, 0, 0, 0]),
        ]

        result = calculate_compartment_initial_conditions(compartments, population_array)

        expected = np.zeros_like(population_array)
        np.testing.assert_array_almost_equal(result["I"], expected)


class TestParseAgeGroup:
    """Tests for _parse_age_group function."""

    def test_parse_simple_range(self):
        """Test parsing a simple age range like '0-4'."""
        result = _parse_age_group("0-4")
        expected = ["0", "1", "2", "3", "4"]
        assert result == expected

    def test_parse_teenage_range(self):
        """Test parsing a teenage age range like '5-17'."""
        result = _parse_age_group("5-17")
        expected = [str(i) for i in range(5, 18)]
        assert result == expected
        assert len(result) == 13

    def test_parse_adult_range(self):
        """Test parsing an adult age range like '18-49'."""
        result = _parse_age_group("18-49")
        expected = [str(i) for i in range(18, 50)]
        assert result == expected
        assert len(result) == 32

    def test_parse_middle_age_range(self):
        """Test parsing a middle age range like '50-64'."""
        result = _parse_age_group("50-64")
        expected = [str(i) for i in range(50, 65)]
        assert result == expected
        assert len(result) == 15

    def test_parse_plus_notation(self):
        """Test parsing plus notation like '65+'."""
        result = _parse_age_group("65+")
        # Should return ["65", "66", ..., "83", "84+"]
        expected = [str(i) for i in range(65, 84)] + ["84+"]
        assert result == expected
        assert len(result) == 20  # 65-83 (19 ages) + "84+"

    def test_parse_plus_notation_starts_with_84(self):
        """Test parsing '84+' edge case."""
        result = _parse_age_group("84+")
        # Should return just ["84+"] since range(84, 84) is empty
        expected = ["84+"]
        assert result == expected

    def test_parse_plus_notation_zero(self):
        """Test parsing '0+' which represents all ages."""
        result = _parse_age_group("0+")
        # Should return ["0", "1", ..., "83", "84+"]
        expected = [str(i) for i in range(84)] + ["84+"]
        assert result == expected
        assert len(result) == 85  # 0-83 (84 ages) + "84+"

    def test_parse_single_age_range(self):
        """Test parsing a single age like '5-5'."""
        result = _parse_age_group("5-5")
        expected = ["5"]
        assert result == expected

    def test_result_types_are_strings(self):
        """Test that all returned values are strings."""
        result = _parse_age_group("0-4")
        assert all(isinstance(age, str) for age in result)

        result_plus = _parse_age_group("65+")
        assert all(isinstance(age, str) for age in result_plus)


class TestLoadMetrocastPopulation:
    """Tests for load_metrocast_population function."""

    def test_loads_denver_population(self):
        """Test loading population for Denver metrocast location."""
        age_groups = ["0-4", "5-17", "18-49", "50-64", "65+"]
        population = load_metrocast_population("denver", age_groups)

        assert population is not None
        assert population.name == "metrocast_location_denver"
        assert len(population.Nk) == len(age_groups)
        assert all(nk > 0 for nk in population.Nk)

    def test_loads_boston_population(self):
        """Test loading population for Boston metrocast location."""
        age_groups = ["0-4", "5-17", "18-49", "50-64", "65+"]
        population = load_metrocast_population("boston", age_groups)

        assert population is not None
        assert population.name == "metrocast_location_boston"
        assert len(population.Nk) == len(age_groups)
        assert all(nk > 0 for nk in population.Nk)

    def test_loads_nc_flu_region_population(self):
        """Test loading population for NC flu region (nenc)."""
        age_groups = ["0-4", "5-17", "18-49", "50-64", "65+"]
        population = load_metrocast_population("nenc", age_groups)

        assert population is not None
        assert population.name == "metrocast_location_nenc"
        assert len(population.Nk) == len(age_groups)

    def test_has_contact_matrices(self):
        """Test that loaded population has contact matrices from parent region.

        This is a regression test for the fix where age_group_mapping was
        incorrectly passed as contacts_source parameter.
        """
        age_groups = ["0-4", "5-17", "18-49", "50-64", "65+"]
        population = load_metrocast_population("denver", age_groups)

        # Should have contact matrices inherited from parent region (Colorado)
        assert population.contact_matrices is not None
        assert len(population.contact_matrices) > 0
        # Should have standard layers
        assert "home" in population.layers or "community" in population.layers

    def test_contact_matrix_dimensions_match_age_groups(self):
        """Test that contact matrix dimensions match the number of age groups."""
        age_groups = ["0-4", "5-17", "18-49", "50-64", "65+"]
        population = load_metrocast_population("denver", age_groups)

        for layer, matrix in population.contact_matrices.items():
            assert matrix.shape == (len(age_groups), len(age_groups)), (
                f"Contact matrix for layer '{layer}' should have shape "
                f"({len(age_groups)}, {len(age_groups)}), got {matrix.shape}"
            )

    def test_invalid_location_raises_error(self):
        """Test that invalid metrocast location raises ValueError."""
        age_groups = ["0-4", "5-17", "18-49", "50-64", "65+"]
        with pytest.raises(ValueError, match="No population data found"):
            load_metrocast_population("invalid_location", age_groups)

    def test_with_contact_matrix_override(self):
        """Test loading population with contact matrix override."""
        age_groups = ["0-4", "5-17", "18-49", "50-64", "65+"]
        # Use Texas contact matrix instead of Colorado for Denver
        population = load_metrocast_population(
            "denver",
            age_groups,
            contact_matrix_override="US-TX",
        )

        assert population is not None
        assert population.name == "metrocast_location_denver"
        # Contact matrices should be loaded from Texas
        assert len(population.contact_matrices) > 0

    def test_total_population_matches_input_data(self):
        """Test that total population Nk matches the input metrocast data."""
        age_groups = ["0-4", "5-17", "18-49", "50-64", "65+"]

        # Load population using the function
        population = load_metrocast_population("denver", age_groups)

        # Load raw data to compare
        raw_data = get_metrocast_population_data()
        denver_data = raw_data[raw_data["metrocast_location_id"] == "denver"]
        expected_total = denver_data["population"].sum()

        # Compare total population
        actual_total = sum(population.Nk)
        assert actual_total == expected_total, (
            f"Total population mismatch for denver: expected {expected_total}, got {actual_total}"
        )

    def test_age_group_populations_match_input_data(self):
        """Test that each age group population matches the aggregated input data."""
        age_groups = ["0-4", "5-17", "18-49", "50-64", "65+"]

        # Load population using the function
        population = load_metrocast_population("boston", age_groups)

        # Load raw data to compare
        raw_data = get_metrocast_population_data()
        boston_data = raw_data[raw_data["metrocast_location_id"] == "boston"]

        # Define age mapping (same as in load_metrocast_population)
        age_group_ranges = {
            "0-4": ["0", "1", "2", "3", "4"],
            "5-17": [str(i) for i in range(5, 18)],
            "18-49": [str(i) for i in range(18, 50)],
            "50-64": [str(i) for i in range(50, 65)],
            "65+": [str(i) for i in range(65, 84)] + ["84+"],
        }

        # Check each age group
        for i, age_group in enumerate(age_groups):
            ages = age_group_ranges[age_group]
            expected_pop = 0
            for age in ages:
                age_rows = boston_data[boston_data["age"].astype(str) == age]
                expected_pop += age_rows["population"].sum()

            assert population.Nk[i] == expected_pop, (
                f"Population mismatch for age group {age_group}: expected {expected_pop}, got {population.Nk[i]}"
            )

    def test_contact_matrix_inherited_from_parent_region(self):
        """Test that contact matrices are inherited from parent region.

        Denver is in Colorado, so its contact matrix should match US-CO.
        """
        age_groups = ["0-4", "5-17", "18-49", "50-64", "65+"]

        # Load metrocast population for Denver
        denver_pop = load_metrocast_population("denver", age_groups)

        # Load ISO population for Colorado (parent region)
        colorado_pop = load_iso_population("US-CO", age_groups)

        # Verify they have the same layers
        assert set(denver_pop.layers) == set(colorado_pop.layers), (
            f"Layer mismatch: Denver has {denver_pop.layers}, Colorado has {colorado_pop.layers}"
        )

        # Verify contact matrices are identical for each layer
        for layer in denver_pop.layers:
            np.testing.assert_array_equal(
                denver_pop.contact_matrices[layer],
                colorado_pop.contact_matrices[layer],
                err_msg=f"Contact matrix mismatch for layer '{layer}' between Denver and Colorado",
            )

    def test_contact_matrix_inherited_boston_massachusetts(self):
        """Test that Boston inherits contact matrix from Massachusetts (US-MA)."""
        age_groups = ["0-4", "5-17", "18-49", "50-64", "65+"]

        boston_pop = load_metrocast_population("boston", age_groups)
        ma_pop = load_iso_population("US-MA", age_groups)

        assert set(boston_pop.layers) == set(ma_pop.layers)

        for layer in boston_pop.layers:
            np.testing.assert_array_equal(
                boston_pop.contact_matrices[layer],
                ma_pop.contact_matrices[layer],
                err_msg=f"Contact matrix mismatch for layer '{layer}' between Boston and Massachusetts",
            )

    def test_contact_matrix_inherited_nenc_north_carolina(self):
        """Test that NENC (NC flu region) inherits contact matrix from North Carolina (US-NC)."""
        age_groups = ["0-4", "5-17", "18-49", "50-64", "65+"]

        nenc_pop = load_metrocast_population("nenc", age_groups)
        nc_pop = load_iso_population("US-NC", age_groups)

        assert set(nenc_pop.layers) == set(nc_pop.layers)

        for layer in nenc_pop.layers:
            np.testing.assert_array_equal(
                nenc_pop.contact_matrices[layer],
                nc_pop.contact_matrices[layer],
                err_msg=f"Contact matrix mismatch for layer '{layer}' between NENC and North Carolina",
            )

    def test_state_level_location_uses_iso_population(self):
        """Test that state-level metrocast locations use ISO population data."""
        age_groups = ["0-4", "5-17", "18-49", "50-64", "65+"]

        # Load state-level metrocast location
        colorado_metrocast = load_metrocast_population("colorado", age_groups)

        # Load the corresponding ISO location
        colorado_iso = load_iso_population("US-CO", age_groups)

        # Both should have the same population values
        np.testing.assert_array_equal(
            colorado_metrocast.Nk,
            colorado_iso.Nk,
            err_msg="State-level metrocast population should match ISO population",
        )

    def test_state_level_location_has_metrocast_naming(self):
        """Test that state-level metrocast locations use metrocast naming convention."""
        age_groups = ["0-4", "5-17", "18-49", "50-64", "65+"]

        colorado_pop = load_metrocast_population("colorado", age_groups)

        # Should use metrocast naming convention for output formatting
        assert colorado_pop.name == "metrocast_location_colorado"

    def test_state_level_location_has_contact_matrices(self):
        """Test that state-level metrocast locations have contact matrices."""
        age_groups = ["0-4", "5-17", "18-49", "50-64", "65+"]

        georgia_pop = load_metrocast_population("georgia", age_groups)

        # Should have contact matrices from the state
        assert georgia_pop.contact_matrices is not None
        assert len(georgia_pop.contact_matrices) > 0

    def test_state_level_contact_matrix_matches_iso(self):
        """Test that state-level metrocast location contact matrices match ISO location."""
        age_groups = ["0-4", "5-17", "18-49", "50-64", "65+"]

        massachusetts_metrocast = load_metrocast_population("massachusetts", age_groups)
        massachusetts_iso = load_iso_population("US-MA", age_groups)

        # Contact matrices should be identical
        assert set(massachusetts_metrocast.layers) == set(massachusetts_iso.layers)

        for layer in massachusetts_metrocast.layers:
            np.testing.assert_array_equal(
                massachusetts_metrocast.contact_matrices[layer],
                massachusetts_iso.contact_matrices[layer],
                err_msg=f"Contact matrix mismatch for layer '{layer}'",
            )

    def test_multiple_state_level_locations(self):
        """Test multiple state-level locations to verify consistent behavior."""
        age_groups = ["0-4", "5-17", "18-49", "50-64", "65+"]

        state_locations = ["colorado", "georgia", "texas", "north-carolina"]
        iso_codes = ["US-CO", "US-GA", "US-TX", "US-NC"]

        for state_loc, iso_code in zip(state_locations, iso_codes):
            metrocast_pop = load_metrocast_population(state_loc, age_groups)
            iso_pop = load_iso_population(iso_code, age_groups)

            # Population values should match
            np.testing.assert_array_equal(
                metrocast_pop.Nk,
                iso_pop.Nk,
                err_msg=f"Population mismatch for {state_loc}",
            )

            # Name should use metrocast convention
            assert metrocast_pop.name == f"metrocast_location_{state_loc}"
