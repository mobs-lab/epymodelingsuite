"""Tests for flumodelingsuite.builders.orchestrators module."""

from __future__ import annotations

from datetime import date
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from flumodelingsuite.builders.orchestrators import (
    apply_calibrated_parameters,
    apply_seasonality_with_sampled_min,
    apply_vaccination_for_sampled_start,
    compute_simulation_start_date,
    create_model_collection,
    flatten_simulation_results,
    format_calibration_data,
    format_projection_trajectories,
    get_aggregated_comparison_transition,
    setup_vaccination_schedules,
)
from flumodelingsuite.schema.basemodel import (
    BaseEpiModel,
    Compartment,
    Parameter,
    Population,
    Seasonality,
    Simulation,
    Timespan,
    Transition,
    Vaccination,
)
from flumodelingsuite.schema.calibration import ComparisonSpec


class TestCreateModelCollection:
    """Tests for create_model_collection function."""

    @pytest.fixture
    def base_model_config(self):
        """Create a minimal BaseEpiModel configuration for testing."""
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
        models, resolved_names = create_model_collection(base_model_config, None)

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
        models, resolved_names = create_model_collection(base_model_config, population_names)

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
        # Mock codebook with 3 locations (major + minor states) for faster testing
        mock_codebook = pd.DataFrame(
            {
                "location_name_epydemix": [
                    "United_States_California",
                    "United_States_Vermont",
                    "United_States_Washington",
                ]
            }
        )

        with patch("flumodelingsuite.builders.orchestrators.get_location_codebook", return_value=mock_codebook):
            population_names = ["all"]
            models, resolved_names = create_model_collection(base_model_config, population_names)

            # Should create models for all locations in mocked codebook
            expected_locations = mock_codebook["location_name_epydemix"]

            assert len(models) == len(expected_locations)
            # resolved_names is a pandas Series when "all" is used
            # Convert to list for comparison
            if isinstance(resolved_names, pd.Series):
                assert list(resolved_names) == list(expected_locations)
            else:
                assert resolved_names == list(expected_locations)

    def test_all_models_share_compartments(self, base_model_config):
        """Test that all models have the same compartments."""
        population_names = ["US-CA", "US-TX"]
        models, _ = create_model_collection(base_model_config, population_names)

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
        models, _ = create_model_collection(base_model_config, population_names)

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
        models, _ = create_model_collection(base_model_config, population_names)

        # Check parameters exist on all models
        for model in models:
            assert "beta" in model.parameters
            assert "gamma" in model.parameters
            assert model.parameters["beta"] == 0.5
            assert model.parameters["gamma"] == 0.1

    def test_models_have_different_populations(self, base_model_config):
        """Test that each model has a different population assigned."""
        population_names = ["US-CA", "US-TX", "US-NY"]
        models, _ = create_model_collection(base_model_config, population_names)

        # Extract population names from models
        model_pop_names = [model.population.name for model in models]

        # All should be different (converted to epydemix format)
        assert len(set(model_pop_names)) == 3
        assert "United_States_California" in model_pop_names
        assert "United_States_Texas" in model_pop_names
        assert "United_States_New_York" in model_pop_names

    def test_model_name_is_set_from_config(self, base_model_config):
        """Test that model name is set from basemodel config."""
        models, _ = create_model_collection(base_model_config, ["US-CA"])

        assert models[0].name == "test_model"

    def test_model_name_not_set_when_none(self, base_model_config):
        """Test that model name is not set when basemodel.name is None."""
        base_model_config.name = None
        models, _ = create_model_collection(base_model_config, ["US-CA"])

        # epydemix.EpiModel initializes name as 'EpiModel' by default when not set
        assert models[0].name == "EpiModel"

    def test_models_have_age_structure(self, base_model_config):
        """Test that all models have proper age-structured population."""
        population_names = ["US-CA", "US-TX"]
        models, _ = create_model_collection(base_model_config, population_names)

        for model in models:
            # Check that population is set and has age structure
            assert model.population is not None
            assert hasattr(model.population, "Nk")
            assert isinstance(model.population.Nk, np.ndarray)
            # Should match the age groups from config
            assert len(model.population.Nk) == 5

    def test_returns_tuple_of_list_and_names(self, base_model_config):
        """Test that function returns tuple of (models, resolved_names)."""
        result = create_model_collection(base_model_config, ["US-CA"])

        assert isinstance(result, tuple)
        assert len(result) == 2

        models, resolved_names = result
        assert isinstance(models, list)
        assert isinstance(resolved_names, list)

    def test_resolved_names_matches_models_length(self, base_model_config):
        """Test that resolved_names list has same length as models list."""
        population_names = ["US-CA", "US-TX", "US-NY"]
        models, resolved_names = create_model_collection(base_model_config, population_names)

        assert len(models) == len(resolved_names)

    def test_models_are_independent_copies(self, base_model_config):
        """Test that models are independent deepcopies, not references."""
        population_names = ["US-CA", "US-TX"]
        models, _ = create_model_collection(base_model_config, population_names)

        # Modify first model's parameter
        models[0].parameters["beta"] = 0.9

        # Second model should not be affected
        assert models[1].parameters["beta"] == 0.5

    def test_handles_empty_population_list(self, base_model_config):
        """Test behavior with empty population list."""
        # Empty list creates a dummy model with base population (implementation detail)
        models, resolved_names = create_model_collection(base_model_config, [])

        # Actually creates one model from basemodel population when list is empty
        # This is because the code path falls through to use the base population
        assert len(models) == 1
        assert len(resolved_names) == 1
        assert resolved_names[0] == "US-CA"

    def test_maintains_model_configuration_integrity(self, base_model_config):
        """Test that model configuration is properly maintained across copies."""
        population_names = ["US-CA", "US-TX"]
        models, _ = create_model_collection(base_model_config, population_names)

        for model in models:
            # Check compartments
            assert len(model.compartments) == 3

            # Check transitions (dict with 3 compartments as keys)
            assert len(model.transitions) == 3

            # Check parameters
            assert len(model.parameters) >= 2  # At least beta and gamma


class TestSetupVaccinationSchedules:
    """Tests for setup_vaccination_schedules function."""

    @pytest.fixture
    def base_model_config(self):
        """Create a minimal BaseEpiModel configuration for testing."""
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

    @pytest.fixture
    def base_model_with_vaccination(self, base_model_config, tmp_path):
        """Create a BaseEpiModel with vaccination configuration."""
        # Create a minimal vaccination data file
        import pandas as pd

        vax_data = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-08", "2024-01-15"],
                "location": ["06", "06", "06"],  # California FIPS
                "age_group": ["0-4", "0-4", "0-4"],
                "doses": [100, 150, 200],
            }
        )
        vax_file = tmp_path / "vaccination_data.csv"
        vax_data.to_csv(vax_file, index=False)

        vaccination = Vaccination(
            scenario_data_path=str(vax_file),
            origin_compartment="S",
            eligible_compartments=["S"],
        )

        # Add vaccination transition
        vax_transition = Transition(
            source="S",
            target="R",
            type="vaccination",
            rate=None,
        )
        base_model_config.transitions.append(vax_transition)
        base_model_config.vaccination = vaccination
        return base_model_config

    @pytest.fixture
    def sample_models(self, base_model_config):
        """Create a collection of test models."""
        models, population_names = create_model_collection(base_model_config, ["US-CA", "US-TX"])
        return models, population_names

    def test_returns_models_and_none_when_no_vaccination(self, base_model_config, sample_models):
        """Test that function returns (models, None) when vaccination is not configured."""
        base_model_config.vaccination = None
        models, population_names = sample_models

        result_models, earliest_vax = setup_vaccination_schedules(
            basemodel=base_model_config, models=models, sampled_start_timespan=None, population_names=population_names
        )

        # Should return the same models unchanged and None for earliest_vax
        assert result_models == models
        assert earliest_vax is None
        assert len(result_models) == len(models)

    def test_returns_models_and_earliest_vax_when_start_date_sampled(self, base_model_with_vaccination, sample_models):
        """Test that function returns (models, earliest_vax) when start_date is sampled."""
        models, population_names = sample_models

        sampled_start_timespan = Timespan(start_date=date(2024, 1, 1), end_date=date(2024, 12, 31), delta_t=1.0)

        # Mock scenario_to_epydemix to return a predictable DataFrame
        import pandas as pd

        mock_vax_schedule = pd.DataFrame({"date": ["2024-01-01"], "doses": [100]})

        with patch(
            "flumodelingsuite.builders.orchestrators.scenario_to_epydemix", return_value=mock_vax_schedule
        ) as mock_scenario_to_epydemix:
            result_models, earliest_vax = setup_vaccination_schedules(
                basemodel=base_model_with_vaccination,
                models=models,
                sampled_start_timespan=sampled_start_timespan,
                population_names=population_names,
            )

            # Should return models unchanged and the earliest vaccination schedule
            assert len(result_models) == len(models)
            assert earliest_vax is not None
            assert isinstance(earliest_vax, pd.DataFrame)

            # Verify scenario_to_epydemix was called with correct parameters
            mock_scenario_to_epydemix.assert_called_once()
            call_kwargs = mock_scenario_to_epydemix.call_args[1]
            assert call_kwargs["start_date"] == date(2024, 1, 1)
            assert call_kwargs["end_date"] == date(2024, 12, 31)
            assert call_kwargs["states"] == population_names

    def test_adds_vaccination_to_models_when_start_date_not_sampled(self, base_model_with_vaccination, sample_models):
        """Test that vaccination is added directly to models when start_date is not sampled."""
        models, population_names = sample_models

        # Mock _add_vaccination_schedules_from_config
        with patch("flumodelingsuite.builders.orchestrators.add_vaccination_schedules_from_config") as mock_add_vax:
            result_models, earliest_vax = setup_vaccination_schedules(
                basemodel=base_model_with_vaccination,
                models=models,
                sampled_start_timespan=None,
                population_names=population_names,
            )

            # Should return models and None
            assert len(result_models) == len(models)
            assert earliest_vax is None

            # Verify _add_vaccination_schedules_from_config was called for each model
            assert mock_add_vax.call_count == len(models)
            for call in mock_add_vax.call_args_list:
                assert call[0][0] in models  # First arg is a model
                assert call[0][1] == base_model_with_vaccination.transitions
                assert call[0][2] == base_model_with_vaccination.vaccination
                assert call[0][3] == base_model_with_vaccination.timespan

    def test_scenario_to_epydemix_not_called_when_no_vaccination(self, base_model_config, sample_models):
        """Test that scenario_to_epydemix is not called when vaccination is None."""
        base_model_config.vaccination = None
        models, population_names = sample_models

        sampled_start_timespan = Timespan(start_date=date(2024, 1, 1), end_date=date(2024, 12, 31), delta_t=1.0)

        with patch("flumodelingsuite.builders.orchestrators.scenario_to_epydemix") as mock_scenario_to_epydemix:
            setup_vaccination_schedules(
                basemodel=base_model_config,
                models=models,
                sampled_start_timespan=sampled_start_timespan,
                population_names=population_names,
            )

            # scenario_to_epydemix should not be called when vaccination is None
            mock_scenario_to_epydemix.assert_not_called()

    def test_add_vaccination_not_called_when_start_date_sampled(self, base_model_with_vaccination, sample_models):
        """Test that _add_vaccination_schedules_from_config is not called when start_date is sampled."""
        models, population_names = sample_models

        sampled_start_timespan = Timespan(start_date=date(2024, 1, 1), end_date=date(2024, 12, 31), delta_t=1.0)

        import pandas as pd

        mock_vax_schedule = pd.DataFrame({"date": ["2024-01-01"], "doses": [100]})

        with patch("flumodelingsuite.builders.orchestrators.scenario_to_epydemix", return_value=mock_vax_schedule):
            with patch("flumodelingsuite.builders.vaccination.add_vaccination_schedules_from_config") as mock_add_vax:
                setup_vaccination_schedules(
                    basemodel=base_model_with_vaccination,
                    models=models,
                    sampled_start_timespan=sampled_start_timespan,
                    population_names=population_names,
                )

                # _add_vaccination_schedules_from_config should not be called when start_date is sampled
                mock_add_vax.assert_not_called()


class TestComputeSimulationStartDate:
    """Tests for compute_simulation_start_date function."""

    def test_no_sampling_returns_basemodel_start_date(self):
        """Test that without sampling, returns basemodel start date."""
        params = {"projection": False}
        basemodel_timespan = Timespan(start_date=date(2024, 1, 1), end_date=date(2024, 12, 31), delta_t=1.0)
        # reference_start_date is None when calibration.start_date is not defined in YAML
        # (i.e., start_date is not being calibrated)
        reference_start_date = None

        result = compute_simulation_start_date(params, basemodel_timespan, reference_start_date)

        assert result == date(2024, 1, 1)

    def test_projection_mode_uses_sampled_offset(self):
        """Test that projection mode applies sampled offset (same as calibration)."""
        params = {"projection": True, "start_date": 10}
        basemodel_timespan = Timespan(start_date=date(2024, 1, 1), end_date=date(2024, 12, 31), delta_t=1.0)
        reference_start_date = date(2024, 1, 5)

        result = compute_simulation_start_date(params, basemodel_timespan, reference_start_date)

        # Should use earliest + offset (Jan 5 + 10 days = Jan 15), same as calibration
        assert result == date(2024, 1, 15)

    def test_calibration_mode_uses_sampled_offset(self):
        """Test that calibration mode applies sampled offset to earliest start."""
        params = {"projection": False, "start_date": 10}
        basemodel_timespan = Timespan(start_date=date(2024, 1, 1), end_date=date(2024, 12, 31), delta_t=1.0)
        reference_start_date = date(2024, 1, 5)

        result = compute_simulation_start_date(params, basemodel_timespan, reference_start_date)

        # Should use earliest + offset (Jan 5 + 10 days = Jan 15)
        assert result == date(2024, 1, 15)

    def test_negative_offset(self):
        """Test that negative offset works correctly."""
        params = {"projection": False, "start_date": -5}
        basemodel_timespan = Timespan(start_date=date(2024, 1, 1), end_date=date(2024, 12, 31), delta_t=1.0)
        reference_start_date = date(2024, 1, 10)

        result = compute_simulation_start_date(params, basemodel_timespan, reference_start_date)

        # Jan 10 - 5 days = Jan 5
        assert result == date(2024, 1, 5)


class TestApplyCalibratedParameters:
    """Tests for apply_calibrated_parameters function."""

    def test_applies_calibrated_parameters(self):
        """Test that calibrated parameters are extracted and applied to model."""
        model = Mock()
        params = {"beta": 0.5, "gamma": 0.1, "epimodel": "something", "projection": False}
        parameter_config = {
            "beta": Parameter(type="calibrated"),
            "gamma": Parameter(type="calibrated"),
            "alpha": Parameter(type="scalar", value=0.1),
        }

        with patch("flumodelingsuite.builders.orchestrators.add_model_parameters_from_config") as mock_add:
            apply_calibrated_parameters(model, params, parameter_config)

            # Should only extract beta and gamma (calibrated params)
            mock_add.assert_called_once()
            call_args = mock_add.call_args
            assert call_args[0][0] == model
            added_params = call_args[0][1]
            assert "beta" in added_params
            assert "gamma" in added_params
            assert "alpha" not in added_params
            assert added_params["beta"].value == 0.5
            assert added_params["gamma"].value == 0.1

    def test_no_calibrated_parameters_skips_add(self):
        """Test that if no calibrated parameters, nothing is added."""
        model = Mock()
        params = {"epimodel": "something", "projection": False}
        parameter_config = {"alpha": Parameter(type="scalar", value=0.1)}

        with patch("flumodelingsuite.builders.orchestrators.add_model_parameters_from_config") as mock_add:
            apply_calibrated_parameters(model, params, parameter_config)

            # Should not call add since no calibrated params
            mock_add.assert_not_called()

    def test_recalculates_derived_parameters(self):
        """Test that calculated parameters trigger recalculation."""
        model = Mock()
        params = {"beta": 0.5}
        parameter_config = {
            "beta": Parameter(type="calibrated"),
            "derived": Parameter(type="calculated", value="beta * 2"),
        }

        with (
            patch("flumodelingsuite.builders.orchestrators.add_model_parameters_from_config") as mock_add,
            patch("flumodelingsuite.builders.orchestrators.calculate_parameters_from_config") as mock_calc,
        ):
            apply_calibrated_parameters(model, params, parameter_config)

            # Should call both add and calculate
            mock_add.assert_called_once()
            mock_calc.assert_called_once_with(model, parameter_config)


class TestApplyVaccinationForSampledStart:
    """Tests for apply_vaccination_for_sampled_start function."""

    def test_no_vaccination_returns_early(self):
        """Test that without vaccination config, function returns early."""
        model = Mock()
        basemodel = Mock(vaccination=None)
        timespan = Timespan(start_date=date(2024, 1, 1), end_date=date(2024, 12, 31), delta_t=1.0)

        with (
            patch("flumodelingsuite.builders.orchestrators.reaggregate_vaccines") as mock_reagg,
            patch("flumodelingsuite.builders.orchestrators.add_vaccination_schedules_from_config") as mock_add,
        ):
            apply_vaccination_for_sampled_start(model, basemodel, timespan, None, None)

            # Should not call vaccination functions
            mock_reagg.assert_not_called()
            mock_add.assert_not_called()

    def test_no_sampling_returns_early(self):
        """Test that without start_date sampling, vaccination already applied so returns early."""
        model = Mock()
        basemodel = Mock(vaccination=Mock())
        timespan = Timespan(start_date=date(2024, 1, 1), end_date=date(2024, 12, 31), delta_t=1.0)

        with (
            patch("flumodelingsuite.builders.orchestrators.reaggregate_vaccines") as mock_reagg,
            patch("flumodelingsuite.builders.orchestrators.add_vaccination_schedules_from_config") as mock_add,
        ):
            apply_vaccination_for_sampled_start(model, basemodel, timespan, None, sampled_start_timespan=None)

            # Should not reaggregate since start_date not sampled
            mock_reagg.assert_not_called()
            mock_add.assert_not_called()

    def test_reaggregates_vaccination_when_start_date_sampled(self):
        """Test that vaccination is reaggregated when start_date is sampled."""
        model = Mock()
        basemodel = Mock(vaccination=Mock(), transitions=[])
        timespan = Timespan(start_date=date(2024, 1, 15), end_date=date(2024, 12, 31), delta_t=1.0)
        earliest_vax = {"vax_data": "some_data"}
        sampled_start_timespan = Timespan(start_date=date(2024, 1, 1), end_date=date(2024, 12, 31), delta_t=1.0)

        with (
            patch("flumodelingsuite.builders.orchestrators.reaggregate_vaccines") as mock_reagg,
            patch("flumodelingsuite.builders.orchestrators.resample_dataframe") as mock_resample,
            patch("flumodelingsuite.builders.orchestrators.add_vaccination_schedules_from_config") as mock_add,
        ):
            mock_reagg.return_value = {"reaggregated": "data"}
            mock_resample.return_value = {"resampled": "data"}

            apply_vaccination_for_sampled_start(model, basemodel, timespan, earliest_vax, sampled_start_timespan)

            # Should reaggregate, resample, and add
            mock_reagg.assert_called_once_with(earliest_vax, date(2024, 1, 15))
            mock_resample.assert_called_once_with({"reaggregated": "data"}, 1.0)
            mock_add.assert_called_once()


class TestApplySeasonalityWithSampledMin:
    """Tests for apply_seasonality_with_sampled_min function."""

    def test_no_seasonality_returns_early(self):
        """Test that without seasonality config, function returns early."""
        model = Mock()
        basemodel = Mock(seasonality=None)
        timespan = Timespan(start_date=date(2024, 1, 1), end_date=date(2024, 12, 31), delta_t=1.0)
        params = {}

        with patch("flumodelingsuite.builders.orchestrators.add_seasonality_from_config") as mock_add:
            apply_seasonality_with_sampled_min(model, basemodel, timespan, params)

            # Should not call seasonality functions
            mock_add.assert_not_called()

    def test_applies_seasonality_without_sampled_min(self):
        """Test that seasonality is applied without modifying min_value."""
        model = Mock()
        seasonality_config = Seasonality(
            method="balcan",
            latitude=40.0,
            min_value=0.5,
            max_value=1.5,
            target_parameter="beta",
            seasonality_min_date=date(2024, 7, 1),
            seasonality_max_date=date(2024, 1, 1),
        )
        basemodel = Mock(seasonality=seasonality_config)
        timespan = Timespan(start_date=date(2024, 1, 1), end_date=date(2024, 12, 31), delta_t=1.0)
        params = {}

        with patch("flumodelingsuite.builders.orchestrators.add_seasonality_from_config") as mock_add:
            apply_seasonality_with_sampled_min(model, basemodel, timespan, params)

            # Should call with copied config
            mock_add.assert_called_once()
            call_args = mock_add.call_args
            applied_config = call_args[0][1]
            assert applied_config.min_value == 0.5  # Original value

    def test_applies_seasonality_with_sampled_min(self):
        """Test that seasonality min_value is overridden when sampled."""
        model = Mock()
        seasonality_config = Seasonality(
            method="balcan",
            latitude=40.0,
            min_value=0.5,
            max_value=1.5,
            target_parameter="beta",
            seasonality_min_date=date(2024, 7, 1),
            seasonality_max_date=date(2024, 1, 1),
        )
        basemodel = Mock(seasonality=seasonality_config)
        timespan = Timespan(start_date=date(2024, 1, 1), end_date=date(2024, 12, 31), delta_t=1.0)
        params = {"seasonality_min": 0.3}

        with patch("flumodelingsuite.builders.orchestrators.add_seasonality_from_config") as mock_add:
            apply_seasonality_with_sampled_min(model, basemodel, timespan, params)

            # Should call with modified min_value
            mock_add.assert_called_once()
            call_args = mock_add.call_args
            applied_config = call_args[0][1]
            assert applied_config.min_value == 0.3  # Sampled value

    def test_does_not_mutate_original_seasonality(self):
        """Test that original basemodel seasonality is not mutated."""
        model = Mock()
        seasonality_config = Seasonality(
            method="balcan",
            latitude=40.0,
            min_value=0.5,
            max_value=1.5,
            target_parameter="beta",
            seasonality_min_date=date(2024, 7, 1),
            seasonality_max_date=date(2024, 1, 1),
        )
        basemodel = Mock(seasonality=seasonality_config)
        timespan = Timespan(start_date=date(2024, 1, 1), end_date=date(2024, 12, 31), delta_t=1.0)
        params = {"seasonality_min": 0.3}

        with patch("flumodelingsuite.builders.orchestrators.add_seasonality_from_config"):
            apply_seasonality_with_sampled_min(model, basemodel, timespan, params)

            # Original should not be mutated
            assert basemodel.seasonality.min_value == 0.5


class TestFormatCalibrationData:
    """Tests for format_calibration_data function."""

    def test_aligns_simulation_to_observed_dates(self):
        """Test that simulation results are aligned to observation dates."""
        # Mock simulation results
        results = Mock()
        results.dates = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)]
        results.transitions = {
            "Hosp_vax": np.array([10, 20, 30, 40, 50]),
            "Hosp_unvax": np.array([5, 10, 15, 20, 25]),
        }

        comparison_transitions = ["Hosp_vax", "Hosp_unvax"]
        data_dates = [date(2024, 1, 2), date(2024, 1, 4)]

        result = format_calibration_data(results, comparison_transitions, data_dates)

        # Should return dict with "data" and "date" keys
        assert isinstance(result, dict)
        assert "data" in result
        assert "date" in result

        # Should sum transitions and extract only dates matching data_dates
        expected_data = np.array([30, 60])  # [20+10, 40+20] for dates Jan 2 and Jan 4
        np.testing.assert_array_equal(result["data"], expected_data)
        assert result["date"] == data_dates

    def test_pads_with_zeros_when_simulation_shorter(self):
        """Test that zeros are padded when simulation is shorter than observations."""
        results = Mock()
        results.dates = [date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)]
        results.transitions = {
            "Hosp": np.array([30, 40, 50]),
        }

        comparison_transitions = ["Hosp"]
        data_dates = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)]

        result = format_calibration_data(results, comparison_transitions, data_dates)

        # Should return dict with "data" and "date" keys
        assert isinstance(result, dict)
        assert "data" in result
        assert "date" in result

        # Should pad with 2 zeros at beginning
        expected_data = np.array([0, 0, 30, 40, 50])
        np.testing.assert_array_equal(result["data"], expected_data)
        assert result["date"] == data_dates

    def test_handles_multiple_transitions(self):
        """Test that multiple transitions are summed correctly."""
        results = Mock()
        results.dates = [date(2024, 1, 1), date(2024, 1, 2)]
        results.transitions = {
            "Hosp_vax_0_4": np.array([1, 2]),
            "Hosp_vax_5_17": np.array([3, 4]),
            "Hosp_unvax_0_4": np.array([5, 6]),
            "Hosp_unvax_5_17": np.array([7, 8]),
        }

        comparison_transitions = ["Hosp_vax_0_4", "Hosp_vax_5_17", "Hosp_unvax_0_4", "Hosp_unvax_5_17"]
        data_dates = [date(2024, 1, 1), date(2024, 1, 2)]

        result = format_calibration_data(results, comparison_transitions, data_dates)

        # Should return dict with "data" and "date" keys
        assert isinstance(result, dict)
        assert "data" in result
        assert "date" in result

        # Should sum all transitions: [1+3+5+7, 2+4+6+8]
        expected_data = np.array([16, 20])
        np.testing.assert_array_equal(result["data"], expected_data)
        assert result["date"] == data_dates


class TestFlattenSimulationResults:
    """Tests for flatten_simulation_results function."""

    def test_flattens_results_structure(self):
        """Test that results are flattened into single dict."""
        results = Mock()
        results.dates = [date(2024, 1, 1), date(2024, 1, 2)]
        results.transitions = {
            "Hosp_vax": np.array([10, 20]),
            "Hosp_unvax": np.array([5, 10]),
        }
        results.compartments = {
            "S": np.array([1000, 990]),
            "I": np.array([10, 20]),
            "R": np.array([0, 0]),
        }

        result = flatten_simulation_results(results)

        # Should have date at top level plus all transitions and compartments
        assert "date" in result
        assert result["date"] == results.dates
        assert "Hosp_vax" in result
        assert "Hosp_unvax" in result
        assert "S" in result
        assert "I" in result
        assert "R" in result
        np.testing.assert_array_equal(result["Hosp_vax"], np.array([10, 20]))
        np.testing.assert_array_equal(result["S"], np.array([1000, 990]))

    def test_handles_empty_transitions_and_compartments(self):
        """Test that empty transitions/compartments are handled."""
        results = Mock()
        results.dates = [date(2024, 1, 1)]
        results.transitions = {}
        results.compartments = {}

        result = flatten_simulation_results(results)

        # Should only have date
        assert result == {"date": [date(2024, 1, 1)]}


class TestFormatProjectionTrajectories:
    """Tests for format_projection_trajectories function."""

    def test_no_padding_when_parameters_none(self):
        """Test that no padding occurs when optional parameters are None."""
        results = Mock()
        results.dates = [date(2024, 1, 1), date(2024, 1, 2)]
        results.transitions = {"Hosp": np.array([10, 20])}
        results.compartments = {"S": np.array([1000, 990])}

        result = format_projection_trajectories(
            results=results,
            reference_start_date=None,
            actual_start_date=None,
            target_end_date=None,
        )

        # Should just flatten without padding
        assert result["date"] == results.dates
        np.testing.assert_array_equal(result["Hosp"], np.array([10, 20]))

    def test_pads_to_target_length(self):
        """Test that trajectories are padded to match target length."""
        results = Mock()
        # Simulate starting at week 2 (Jan 13), missing first week (Jan 6)
        # Jan 6, 2024 is the first Saturday of the year
        results.dates = [date(2024, 1, 13), date(2024, 1, 20)]
        results.transitions = {"Hosp": np.array([10, 20])}
        results.compartments = {"S": np.array([1000, 990])}

        result = format_projection_trajectories(
            results=results,
            reference_start_date=date(2024, 1, 6),
            actual_start_date=date(2024, 1, 13),
            target_end_date=date(2024, 1, 20),
            resample_frequency="W-SAT",
        )

        # Should pad to 3 weeks total (Jan 6, 13, 20)
        assert len(result["date"]) == 3
        # First date should be padded (Jan 6)
        assert result["date"][0] == pd.Timestamp(date(2024, 1, 6))
        # Hosp should have one zero padded at beginning
        np.testing.assert_array_equal(result["Hosp"], np.array([0, 10, 20]))
        np.testing.assert_array_equal(result["S"], np.array([0, 1000, 990]))

    def test_handles_different_pad_lengths(self):
        """Test that different trajectories pad to same final length."""
        # Trajectory 1: starts early, needs little padding
        results1 = Mock()
        results1.dates = [date(2024, 1, 8), date(2024, 1, 15)]
        results1.transitions = {"Hosp": np.array([10, 20])}
        results1.compartments = {}

        result1 = format_projection_trajectories(
            results=results1,
            reference_start_date=date(2024, 1, 1),
            actual_start_date=date(2024, 1, 8),
            target_end_date=date(2024, 1, 15),
            resample_frequency="W-SAT",
        )

        # Trajectory 2: starts late, needs more padding
        results2 = Mock()
        results2.dates = [date(2024, 1, 15)]
        results2.transitions = {"Hosp": np.array([30])}
        results2.compartments = {}

        result2 = format_projection_trajectories(
            results=results2,
            reference_start_date=date(2024, 1, 1),
            actual_start_date=date(2024, 1, 15),
            target_end_date=date(2024, 1, 15),
            resample_frequency="W-SAT",
        )

        # Both should have same final length
        assert len(result1["date"]) == len(result2["date"])
        assert len(result1["Hosp"]) == len(result2["Hosp"])

    def test_all_trajectories_same_shape_for_stacking(self):
        """Test that multiple trajectories with different starts have identical shapes."""
        # Simulate the real-world scenario: different sampled start dates
        # but all should end at the same projection end date
        trajectories = []

        # Create 5 trajectories with different start dates
        start_dates = [date(2024, 9, 21), date(2024, 9, 28), date(2024, 10, 5), date(2024, 10, 12), date(2024, 10, 19)]
        trajectory_lengths = [37, 36, 35, 34, 33]  # Decreasing as start gets later

        reference_start = date(2024, 9, 21)  # Earliest start
        target_end = date(2025, 5, 31)  # Fixed projection end

        for start_date, length in zip(start_dates, trajectory_lengths, strict=False):
            results = Mock()
            # Create dates for this trajectory
            results.dates = pd.date_range(start=start_date, periods=length, freq="W-SAT").tolist()
            results.transitions = {
                "Hosp": np.random.rand(length),
                "ICU": np.random.rand(length),
            }
            results.compartments = {
                "S": np.random.rand(length),
                "I": np.random.rand(length),
            }

            formatted = format_projection_trajectories(
                results=results,
                reference_start_date=reference_start,
                actual_start_date=start_date,
                target_end_date=target_end,
                resample_frequency="W-SAT",
            )
            trajectories.append(formatted)

        # Verify all trajectories have the same shape
        expected_length = len(trajectories[0]["date"])
        for i, traj in enumerate(trajectories):
            assert len(traj["date"]) == expected_length, (
                f"Trajectory {i} has length {len(traj['date'])}, expected {expected_length}"
            )
            assert len(traj["Hosp"]) == expected_length, (
                f"Trajectory {i} Hosp has length {len(traj['Hosp'])}, expected {expected_length}"
            )
            assert len(traj["ICU"]) == expected_length
            assert len(traj["S"]) == expected_length
            assert len(traj["I"]) == expected_length

        # Verify all trajectories start from the same reference date
        for traj in trajectories:
            assert traj["date"][0] == pd.Timestamp(reference_start)

        # Verify all trajectories end on the same date
        end_dates = [traj["date"][-1] for traj in trajectories]
        assert len(set(end_dates)) == 1, f"End dates differ: {end_dates}"

        # Verify arrays can be stacked (this is what epydemix does)
        hosp_arrays = [traj["Hosp"] for traj in trajectories]
        stacked = np.stack(hosp_arrays)  # This should not raise ValueError
        assert stacked.shape == (len(trajectories), expected_length)

    def test_aggregates_comparison_transitions(self):
        """Test that comparison_specs are aggregated and included in output."""
        results = Mock()
        results.dates = [date(2024, 1, 1), date(2024, 1, 2)]
        results.transitions = {
            "Hosp_vax": np.array([10, 20]),
            "Hosp_unvax": np.array([5, 10]),
            "Death": np.array([1, 2]),
        }
        results.compartments = {"S": np.array([1000, 990])}

        # Define comparison specs
        comparison_specs = [
            ComparisonSpec(
                observed_value_column="total_hosp",
                observed_date_column="date",
                simulation=["Hosp_vax", "Hosp_unvax"],
            ),
        ]

        result = format_projection_trajectories(
            results=results,
            comparison_specs=comparison_specs,
        )

        # Should include aggregated transition
        assert "total_hosp" in result
        np.testing.assert_array_equal(result["total_hosp"], np.array([15, 30]))
        # Should also include original transitions
        np.testing.assert_array_equal(result["Hosp_vax"], np.array([10, 20]))
        np.testing.assert_array_equal(result["Hosp_unvax"], np.array([5, 10]))

    def test_aggregates_multiple_comparison_specs(self):
        """Test that multiple comparison specs are all aggregated."""
        results = Mock()
        results.dates = [date(2024, 1, 1), date(2024, 1, 2)]
        results.transitions = {
            "Hosp_vax": np.array([10, 20]),
            "Hosp_unvax": np.array([5, 10]),
            "Death_vax": np.array([1, 2]),
            "Death_unvax": np.array([0.5, 1]),
        }
        results.compartments = {}

        # Define multiple comparison specs
        comparison_specs = [
            ComparisonSpec(
                observed_value_column="total_hosp",
                observed_date_column="date",
                simulation=["Hosp_vax", "Hosp_unvax"],
            ),
            ComparisonSpec(
                observed_value_column="total_deaths",
                observed_date_column="date",
                simulation=["Death_vax", "Death_unvax"],
            ),
        ]

        result = format_projection_trajectories(
            results=results,
            comparison_specs=comparison_specs,
        )

        # Should include both aggregated transitions
        assert "total_hosp" in result
        assert "total_deaths" in result
        np.testing.assert_array_equal(result["total_hosp"], np.array([15, 30]))
        np.testing.assert_array_equal(result["total_deaths"], np.array([1.5, 3]))

    def test_aggregated_data_is_padded(self):
        """Test that aggregated comparison transitions are padded along with other arrays."""
        results = Mock()
        results.dates = [date(2024, 1, 13), date(2024, 1, 20)]
        results.transitions = {
            "Hosp_vax": np.array([10, 20]),
            "Hosp_unvax": np.array([5, 10]),
        }
        results.compartments = {}

        comparison_specs = [
            ComparisonSpec(
                observed_value_column="total_hosp",
                observed_date_column="date",
                simulation=["Hosp_vax", "Hosp_unvax"],
            ),
        ]

        result = format_projection_trajectories(
            results=results,
            reference_start_date=date(2024, 1, 6),
            actual_start_date=date(2024, 1, 13),
            target_end_date=date(2024, 1, 20),
            resample_frequency="W-SAT",
            comparison_specs=comparison_specs,
        )

        # Should pad aggregated transition to 3 weeks total
        assert len(result["total_hosp"]) == 3
        # First value should be zero (padding)
        np.testing.assert_array_equal(result["total_hosp"], np.array([0, 15, 30]))


class TestGetAggregatedComparisonTransition:
    """Tests for get_aggregated_comparison_transition function."""

    def test_aggregates_single_transition(self):
        """Test aggregation of a single transition."""
        results = Mock()
        results.transitions = {"Hosp": np.array([10, 20, 30])}

        aggregated = get_aggregated_comparison_transition(results, ["Hosp"])

        np.testing.assert_array_equal(aggregated, np.array([10, 20, 30]))

    def test_aggregates_multiple_transitions(self):
        """Test aggregation of multiple transitions."""
        results = Mock()
        results.transitions = {
            "Hosp_vax": np.array([10, 20, 30]),
            "Hosp_unvax": np.array([5, 10, 15]),
        }

        aggregated = get_aggregated_comparison_transition(results, ["Hosp_vax", "Hosp_unvax"])

        np.testing.assert_array_equal(aggregated, np.array([15, 30, 45]))

    def test_aggregates_three_transitions(self):
        """Test aggregation of three transitions."""
        results = Mock()
        results.transitions = {
            "A": np.array([1, 2, 3]),
            "B": np.array([10, 20, 30]),
            "C": np.array([100, 200, 300]),
        }

        aggregated = get_aggregated_comparison_transition(results, ["A", "B", "C"])

        np.testing.assert_array_equal(aggregated, np.array([111, 222, 333]))
