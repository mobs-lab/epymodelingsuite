from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from flumodelingsuite.workflow_dispatcher import (
    _create_model_collection,
    calculate_compartment_initial_conditions,
)


@dataclass
class DummyCompartment:
    """Mock compartment for testing."""

    id: str
    init: float | int | str


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


class TestCreateModelCollection:
    """Tests for _create_model_collection function."""

    @pytest.fixture
    def base_model_config(self):
        """Create a minimal BaseEpiModel configuration for testing."""
        from datetime import date

        from flumodelingsuite.validation.basemodel_validator import (
            BaseEpiModel,
            Compartment,
            Parameter,
            Population,
            Simulation,
            Timespan,
            Transition,
        )

        compartments = [
            Compartment(id="S", label="Susceptible", init="default"),
            Compartment(id="I", label="Infected", init=10),
            Compartment(id="R", label="Recovered", init=0),
        ]

        transitions = [
            Transition(
                source="S",
                target="I",
                type="mediated",
                rate="beta",
                mediator="I",
            ),
            Transition(
                source="I",
                target="R",
                type="spontaneous",
                rate="gamma",
            ),
        ]

        parameters = {
            "beta": Parameter(type="scalar", value=0.5),
            "gamma": Parameter(type="scalar", value=0.1),
        }

        population = Population(name="US-CA", age_groups=["0-4", "5-17", "18-49", "50-64", "65+"])

        timespan = Timespan(start_date=date(2024, 1, 1), end_date=date(2024, 12, 31), delta_t=1.0)

        simulation = Simulation(n_sims=10, resample_frequency="W-SAT")

        return BaseEpiModel(
            name="test_model",
            compartments=compartments,
            transitions=transitions,
            parameters=parameters,
            population=population,
            timespan=timespan,
            simulation=simulation,
        )

    def test_creates_single_model_when_population_names_none(self, base_model_config):
        """Test that a single model is created when population_names is None."""
        models, resolved_names = _create_model_collection(base_model_config, None)

        # Should create exactly one model
        assert len(models) == 1
        assert len(resolved_names) == 1

        # Should use the population name from basemodel
        assert resolved_names[0] == "US-CA"

        # Model should have correct population (converted to epydemix format)
        assert models[0].population.name == "United_States_California"

    def test_creates_multiple_models_for_multiple_populations(self, base_model_config):
        """Test that multiple models are created for a list of population names."""
        population_names = ["US-CA", "US-TX", "US-NY"]
        models, resolved_names = _create_model_collection(base_model_config, population_names)

        # Should create one model per population
        assert len(models) == 3
        assert len(resolved_names) == 3
        assert resolved_names == population_names

        # Each model should have the correct population (converted to epydemix format)
        expected_names = ["United_States_California", "United_States_Texas", "United_States_New_York"]
        for model, expected_name in zip(models, expected_names, strict=False):
            assert model.population.name == expected_name

    def test_expands_all_keyword_to_all_locations(self, base_model_config):
        """Test that 'all' in population_names expands to all locations in codebook."""
        from flumodelingsuite.utils import get_location_codebook

        population_names = ["all"]
        models, resolved_names = _create_model_collection(base_model_config, population_names)

        # Should create models for all locations in codebook
        codebook = get_location_codebook()
        expected_locations = codebook["location_name_epydemix"]

        assert len(models) == len(expected_locations)
        # resolved_names is a pandas Series when "all" is used
        # Convert to list for comparison
        import pandas as pd

        if isinstance(resolved_names, pd.Series):
            assert list(resolved_names) == list(expected_locations)
        else:
            assert resolved_names == list(expected_locations)

    def test_all_models_share_compartments(self, base_model_config):
        """Test that all models have the same compartments."""
        population_names = ["US-CA", "US-TX"]
        models, _ = _create_model_collection(base_model_config, population_names)

        # Check compartments exist on all models
        # In epydemix, model.compartments is a list of compartment IDs (strings)
        for model in models:
            assert len(model.compartments) == 3
            assert "S" in model.compartments
            assert "I" in model.compartments
            assert "R" in model.compartments

    def test_all_models_share_transitions(self, base_model_config):
        """Test that all models have the same transitions."""
        population_names = ["US-CA", "US-TX"]
        models, _ = _create_model_collection(base_model_config, population_names)

        # Check transitions exist on all models
        # In epydemix, model.transitions is a dict mapping compartment IDs to transition objects
        for model in models:
            # Check that all compartments have transition entries
            assert "S" in model.transitions
            assert "I" in model.transitions
            assert "R" in model.transitions
            # Verify there are transitions from S and from I
            assert len(model.transitions["S"]) > 0  # S -> I transition
            assert len(model.transitions["I"]) > 0  # I -> R transition

    def test_all_models_share_parameters(self, base_model_config):
        """Test that all models have the same non-sampled parameters."""
        population_names = ["US-CA", "US-TX"]
        models, _ = _create_model_collection(base_model_config, population_names)

        # Check parameters exist on all models
        for model in models:
            assert "beta" in model.parameters
            assert "gamma" in model.parameters
            assert model.parameters["beta"] == 0.5
            assert model.parameters["gamma"] == 0.1

    def test_models_have_different_populations(self, base_model_config):
        """Test that each model has a different population assigned."""
        population_names = ["US-CA", "US-TX", "US-NY"]
        models, _ = _create_model_collection(base_model_config, population_names)

        # Extract population names from models
        model_pop_names = [model.population.name for model in models]

        # All should be different (converted to epydemix format)
        assert len(set(model_pop_names)) == 3
        assert "United_States_California" in model_pop_names
        assert "United_States_Texas" in model_pop_names
        assert "United_States_New_York" in model_pop_names

    def test_model_name_is_set_from_config(self, base_model_config):
        """Test that model name is set from basemodel config."""
        models, _ = _create_model_collection(base_model_config, ["US-CA"])

        assert models[0].name == "test_model"

    def test_model_name_not_set_when_none(self, base_model_config):
        """Test that model name is not set when basemodel.name is None."""
        base_model_config.name = None
        models, _ = _create_model_collection(base_model_config, ["US-CA"])

        # epydemix.EpiModel initializes name as 'EpiModel' by default when not set
        assert models[0].name == "EpiModel"

    def test_models_have_age_structure(self, base_model_config):
        """Test that all models have proper age-structured population."""
        population_names = ["US-CA", "US-TX"]
        models, _ = _create_model_collection(base_model_config, population_names)

        for model in models:
            # Check that population is set and has age structure
            assert model.population is not None
            assert hasattr(model.population, "Nk")
            assert isinstance(model.population.Nk, np.ndarray)
            # Should match the age groups from config
            assert len(model.population.Nk) == 5

    def test_returns_tuple_of_list_and_names(self, base_model_config):
        """Test that function returns tuple of (models, resolved_names)."""
        result = _create_model_collection(base_model_config, ["US-CA"])

        assert isinstance(result, tuple)
        assert len(result) == 2

        models, resolved_names = result
        assert isinstance(models, list)
        assert isinstance(resolved_names, list)

    def test_resolved_names_matches_models_length(self, base_model_config):
        """Test that resolved_names list has same length as models list."""
        population_names = ["US-CA", "US-TX", "US-NY"]
        models, resolved_names = _create_model_collection(base_model_config, population_names)

        assert len(models) == len(resolved_names)

    def test_models_are_independent_copies(self, base_model_config):
        """Test that models are independent deepcopies, not references."""
        population_names = ["US-CA", "US-TX"]
        models, _ = _create_model_collection(base_model_config, population_names)

        # Modify first model's parameter
        models[0].parameters["beta"] = 0.9

        # Second model should not be affected
        assert models[1].parameters["beta"] == 0.5

    def test_handles_empty_population_list(self, base_model_config):
        """Test behavior with empty population list."""
        # Empty list creates a dummy model with base population (implementation detail)
        models, resolved_names = _create_model_collection(base_model_config, [])

        # Actually creates one model from basemodel population when list is empty
        # This is because the code path falls through to use the base population
        assert len(models) == 1
        assert len(resolved_names) == 1
        assert resolved_names[0] == "US-CA"

    def test_maintains_model_configuration_integrity(self, base_model_config):
        """Test that model configuration is properly maintained across copies."""
        population_names = ["US-CA", "US-TX"]
        models, _ = _create_model_collection(base_model_config, population_names)

        for model in models:
            # Check compartments
            assert len(model.compartments) == 3

            # Check transitions (dict with 3 compartments as keys)
            assert len(model.transitions) == 3

            # Check parameters
            assert len(model.parameters) >= 2  # At least beta and gamma
