"""Tests for flumodelingsuite.builders.base module."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from flumodelingsuite.builders.base import calculate_compartment_initial_conditions


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
