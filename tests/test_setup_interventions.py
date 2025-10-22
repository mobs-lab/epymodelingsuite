from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pytest

from flumodelingsuite.workflow_dispatcher import _create_model_collection, _setup_interventions


class TestSetupInterventions:
    """Tests for _setup_interventions function."""

    @pytest.fixture
    def base_model_config(self):
        """Create a minimal BaseEpiModel configuration for testing."""
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

    @pytest.fixture
    def sample_models(self, base_model_config):
        """Create a collection of test models."""
        models, population_names = _create_model_collection(base_model_config, ["US-CA", "US-TX"])
        return models

    def test_returns_models_unchanged_when_no_interventions(self, base_model_config, sample_models):
        """Test that models are returned unchanged when basemodel has no interventions."""
        base_model_config.interventions = None
        models = sample_models
        intervention_types = []

        result = _setup_interventions(
            models=models,
            basemodel=base_model_config,
            intervention_types=intervention_types,
            sampled_start_timespan=None,
        )

        # Should return the same models unchanged
        assert result == models
        assert len(result) == len(models)

    def test_returns_models_unchanged_when_interventions_empty_list(self, base_model_config, sample_models):
        """Test that models are returned unchanged when basemodel.interventions is an empty list."""
        base_model_config.interventions = []
        models = sample_models
        intervention_types = []

        result = _setup_interventions(
            models=models,
            basemodel=base_model_config,
            intervention_types=intervention_types,
            sampled_start_timespan=None,
        )

        # Should return the same models unchanged
        assert result == models
        assert len(result) == len(models)

    def test_adds_school_closure_when_in_intervention_types(self, base_model_config, sample_models):
        """Test that school closure intervention is added when 'school_closure' is in intervention_types."""
        from flumodelingsuite.validation.basemodel_validator import Intervention

        # Mock interventions configuration
        base_model_config.interventions = [
            Intervention(
                type="school_closure",
                label="School Closures",
            )
        ]
        models = sample_models
        intervention_types = ["school_closure"]

        with (
            patch("flumodelingsuite.workflow_dispatcher.make_school_closure_dict") as mock_closure_dict,
            patch(
                "flumodelingsuite.workflow_dispatcher._add_school_closure_intervention_from_config"
            ) as mock_add_closure,
        ):
            mock_closure_dict.return_value = {"2024": []}

            result = _setup_interventions(
                models=models,
                basemodel=base_model_config,
                intervention_types=intervention_types,
                sampled_start_timespan=None,
            )

            # Verify make_school_closure_dict was called once per model
            assert mock_closure_dict.call_count == len(models)
            # Check that it was called with correct year range
            for call_args in mock_closure_dict.call_args_list:
                year_range = list(call_args[0][0])
                assert 2024 in year_range

            # Verify _add_school_closure_intervention_from_config was called for each model
            assert mock_add_closure.call_count == len(models)
            for mock_call in mock_add_closure.call_args_list:
                assert mock_call[0][0] in models  # First arg is a model
                assert mock_call[0][1] == base_model_config.interventions
                assert mock_call[0][2] == {"2024": []}

            # Result should be the models
            assert result == models

    def test_adds_contact_matrix_when_in_intervention_types(self, base_model_config, sample_models):
        """Test that contact matrix intervention is added when 'contact_matrix' is in intervention_types."""
        from flumodelingsuite.validation.basemodel_validator import Intervention

        # Mock interventions configuration
        base_model_config.interventions = [
            Intervention(
                type="contact_matrix",
                label="Contact Matrix Changes",
                contact_matrix_layer="work",
                start_date=date(2024, 3, 1),
                end_date=date(2024, 6, 1),
            )
        ]
        models = sample_models
        intervention_types = ["contact_matrix"]

        with patch(
            "flumodelingsuite.workflow_dispatcher._add_contact_matrix_interventions_from_config"
        ) as mock_add_contact:
            result = _setup_interventions(
                models=models,
                basemodel=base_model_config,
                intervention_types=intervention_types,
                sampled_start_timespan=None,
            )

            # Verify _add_contact_matrix_interventions_from_config was called for each model
            assert mock_add_contact.call_count == len(models)
            for mock_call in mock_add_contact.call_args_list:
                assert mock_call[0][0] in models  # First arg is a model
                assert mock_call[0][1] == base_model_config.interventions

            # Result should be the models
            assert result == models

    def test_adds_both_interventions_when_both_in_types(self, base_model_config, sample_models):
        """Test that both school closure and contact matrix are added when both are in intervention_types."""
        from flumodelingsuite.validation.basemodel_validator import Intervention

        # Mock interventions configuration
        base_model_config.interventions = [
            Intervention(type="school_closure", label="School Closures"),
            Intervention(
                type="contact_matrix",
                label="Contact Matrix",
                contact_matrix_layer="work",
                start_date=date(2024, 3, 1),
                end_date=date(2024, 6, 1),
            ),
        ]
        models = sample_models
        intervention_types = ["school_closure", "contact_matrix"]

        with (
            patch("flumodelingsuite.workflow_dispatcher.make_school_closure_dict") as mock_closure_dict,
            patch(
                "flumodelingsuite.workflow_dispatcher._add_school_closure_intervention_from_config"
            ) as mock_add_closure,
            patch(
                "flumodelingsuite.workflow_dispatcher._add_contact_matrix_interventions_from_config"
            ) as mock_add_contact,
        ):
            mock_closure_dict.return_value = {"2024": []}

            result = _setup_interventions(
                models=models,
                basemodel=base_model_config,
                intervention_types=intervention_types,
                sampled_start_timespan=None,
            )

            # Both should be called for each model
            assert mock_add_closure.call_count == len(models)
            assert mock_add_contact.call_count == len(models)

            # Result should be the models
            assert result == models

    def test_uses_basemodel_timespan_when_no_sampled_start(self, base_model_config, sample_models):
        """Test that basemodel.timespan is used when sampled_start_timespan is None."""
        from flumodelingsuite.validation.basemodel_validator import Intervention

        base_model_config.interventions = [Intervention(type="school_closure", label="School Closures")]
        models = sample_models
        intervention_types = ["school_closure"]

        with (
            patch("flumodelingsuite.workflow_dispatcher.make_school_closure_dict") as mock_closure_dict,
            patch(
                "flumodelingsuite.workflow_dispatcher._add_school_closure_intervention_from_config"
            ) as mock_add_closure,
        ):
            mock_closure_dict.return_value = {"2024": []}

            _setup_interventions(
                models=models,
                basemodel=base_model_config,
                intervention_types=intervention_types,
                sampled_start_timespan=None,
            )

            # Verify the year range comes from basemodel.timespan
            call_args = mock_closure_dict.call_args[0][0]
            assert 2024 in list(call_args)  # basemodel timespan is 2024-01-01 to 2024-12-31

    def test_uses_sampled_start_timespan_when_provided(self, base_model_config, sample_models):
        """Test that sampled_start_timespan is used when provided instead of basemodel.timespan."""
        from flumodelingsuite.validation.basemodel_validator import Intervention, Timespan

        base_model_config.interventions = [Intervention(type="school_closure", label="School Closures")]
        models = sample_models
        intervention_types = ["school_closure"]

        # Create a different timespan for sampling
        sampled_start_timespan = Timespan(start_date=date(2023, 1, 1), end_date=date(2025, 12, 31), delta_t=1.0)

        with (
            patch("flumodelingsuite.workflow_dispatcher.make_school_closure_dict") as mock_closure_dict,
            patch(
                "flumodelingsuite.workflow_dispatcher._add_school_closure_intervention_from_config"
            ) as mock_add_closure,
        ):
            mock_closure_dict.return_value = {"2023": [], "2024": [], "2025": []}

            _setup_interventions(
                models=models,
                basemodel=base_model_config,
                intervention_types=intervention_types,
                sampled_start_timespan=sampled_start_timespan,
            )

            # Verify the year range comes from sampled_start_timespan (2023-2025)
            call_args = mock_closure_dict.call_args[0][0]
            year_list = list(call_args)
            assert 2023 in year_list
            assert 2024 in year_list
            assert 2025 in year_list

    def test_handles_multi_year_timespan(self, base_model_config, sample_models):
        """Test that school closures are created for all years in the timespan."""
        from flumodelingsuite.validation.basemodel_validator import Intervention, Timespan

        # Create a multi-year timespan
        base_model_config.timespan = Timespan(start_date=date(2022, 6, 1), end_date=date(2024, 3, 31), delta_t=1.0)
        base_model_config.interventions = [Intervention(type="school_closure", label="School Closures")]
        models = sample_models
        intervention_types = ["school_closure"]

        with (
            patch("flumodelingsuite.workflow_dispatcher.make_school_closure_dict") as mock_closure_dict,
            patch(
                "flumodelingsuite.workflow_dispatcher._add_school_closure_intervention_from_config"
            ) as mock_add_closure,
        ):
            mock_closure_dict.return_value = {"2022": [], "2023": [], "2024": []}

            _setup_interventions(
                models=models,
                basemodel=base_model_config,
                intervention_types=intervention_types,
                sampled_start_timespan=None,
            )

            # Verify all years are included (2022, 2023, 2024)
            call_args = mock_closure_dict.call_args[0][0]
            year_list = list(call_args)
            assert 2022 in year_list
            assert 2023 in year_list
            assert 2024 in year_list

    def test_does_not_add_school_closure_if_not_in_types(self, base_model_config, sample_models):
        """Test that school closure is not added if 'school_closure' is not in intervention_types."""
        from flumodelingsuite.validation.basemodel_validator import Intervention

        base_model_config.interventions = [
            Intervention(type="school_closure", label="School Closures", scaling_factor=0.5),
            Intervention(
                type="contact_matrix",
                label="Contact Matrix",
                contact_matrix_layer="work",
                start_date=date(2024, 3, 1),
                end_date=date(2024, 6, 1),
                scaling_factor=0.7,
            ),
        ]
        models = sample_models
        intervention_types = ["contact_matrix"]  # school_closure not included

        with (
            patch("flumodelingsuite.workflow_dispatcher.make_school_closure_dict") as mock_closure_dict,
            patch(
                "flumodelingsuite.workflow_dispatcher._add_school_closure_intervention_from_config"
            ) as mock_add_closure,
        ):
            _setup_interventions(
                models=models,
                basemodel=base_model_config,
                intervention_types=intervention_types,
                sampled_start_timespan=None,
            )

            # School closure functions should not be called
            mock_closure_dict.assert_not_called()
            mock_add_closure.assert_not_called()

    def test_does_not_add_contact_matrix_if_not_in_types(self, base_model_config, sample_models):
        """Test that contact matrix is not added if 'contact_matrix' is not in intervention_types."""
        from flumodelingsuite.validation.basemodel_validator import Intervention

        base_model_config.interventions = [
            Intervention(type="school_closure", label="School Closures", scaling_factor=0.5),
            Intervention(
                type="contact_matrix",
                label="Contact Matrix",
                contact_matrix_layer="work",
                start_date=date(2024, 3, 1),
                end_date=date(2024, 6, 1),
                scaling_factor=0.7,
            ),
        ]
        models = sample_models
        intervention_types = ["school_closure"]  # contact_matrix not included

        with patch(
            "flumodelingsuite.workflow_dispatcher._add_contact_matrix_interventions_from_config"
        ) as mock_add_contact:
            _setup_interventions(
                models=models,
                basemodel=base_model_config,
                intervention_types=intervention_types,
                sampled_start_timespan=None,
            )

            # Should not be called
            mock_add_contact.assert_not_called()

    def test_handles_empty_intervention_types_list(self, base_model_config, sample_models):
        """Test that no interventions are added when intervention_types is an empty list."""
        from flumodelingsuite.validation.basemodel_validator import Intervention

        base_model_config.interventions = [
            Intervention(type="school_closure", label="School Closures"),
            Intervention(
                type="contact_matrix",
                label="Contact Matrix",
                contact_matrix_layer="work",
                start_date=date(2024, 3, 1),
                end_date=date(2024, 6, 1),
            ),
        ]
        models = sample_models
        intervention_types = []

        with (
            patch("flumodelingsuite.workflow_dispatcher.make_school_closure_dict") as mock_closure_dict,
            patch(
                "flumodelingsuite.workflow_dispatcher._add_school_closure_intervention_from_config"
            ) as mock_add_closure,
            patch(
                "flumodelingsuite.workflow_dispatcher._add_contact_matrix_interventions_from_config"
            ) as mock_add_contact,
        ):
            result = _setup_interventions(
                models=models,
                basemodel=base_model_config,
                intervention_types=intervention_types,
                sampled_start_timespan=None,
            )

            # None should be called
            mock_closure_dict.assert_not_called()
            mock_add_closure.assert_not_called()
            mock_add_contact.assert_not_called()

            # Result should be the models unchanged
            assert result == models

    def test_handles_single_model_in_list(self, base_model_config):
        """Test that interventions are added correctly for a single model."""
        from flumodelingsuite.validation.basemodel_validator import Intervention

        # Create a single model
        models, _ = _create_model_collection(base_model_config, ["US-CA"])

        # Add school closure
        base_model_config.interventions = [Intervention(type="school_closure", label="School Closures")]
        intervention_types = ["school_closure"]

        with (
            patch("flumodelingsuite.workflow_dispatcher.make_school_closure_dict") as mock_closure_dict,
            patch(
                "flumodelingsuite.workflow_dispatcher._add_school_closure_intervention_from_config"
            ) as mock_add_closure,
        ):
            mock_closure_dict.return_value = {"2024": []}

            result = _setup_interventions(
                models=models,
                basemodel=base_model_config,
                intervention_types=intervention_types,
                sampled_start_timespan=None,
            )

            # Should be called once for the single model
            assert mock_add_closure.call_count == 1
            assert result == models

    def test_handles_many_models(self, base_model_config):
        """Test that interventions are added correctly for many models."""
        from flumodelingsuite.validation.basemodel_validator import Intervention

        # Create multiple models
        models, _ = _create_model_collection(base_model_config, ["US-CA", "US-TX", "US-NY", "US-FL"])

        # Add contact matrix intervention
        base_model_config.interventions = [
            Intervention(
                type="contact_matrix",
                label="Contact Matrix",
                contact_matrix_layer="work",
                start_date=date(2024, 3, 1),
                end_date=date(2024, 6, 1),
            )
        ]
        intervention_types = ["contact_matrix"]

        with patch(
            "flumodelingsuite.workflow_dispatcher._add_contact_matrix_interventions_from_config"
        ) as mock_add_contact:
            result = _setup_interventions(
                models=models,
                basemodel=base_model_config,
                intervention_types=intervention_types,
                sampled_start_timespan=None,
            )

            # Should be called once for each model
            assert mock_add_contact.call_count == 4
            assert result == models

    def test_returns_same_model_list_reference(self, base_model_config, sample_models):
        """Test that the function returns the same list object (models are modified in place)."""
        from flumodelingsuite.validation.basemodel_validator import Intervention

        base_model_config.interventions = [Intervention(type="school_closure", label="School Closures")]
        models = sample_models
        intervention_types = ["school_closure"]

        with patch("flumodelingsuite.workflow_dispatcher.make_school_closure_dict", return_value={"2024": []}):
            with patch("flumodelingsuite.workflow_dispatcher._add_school_closure_intervention_from_config"):
                result = _setup_interventions(
                    models=models,
                    basemodel=base_model_config,
                    intervention_types=intervention_types,
                    sampled_start_timespan=None,
                )

                # Should return the same list object
                assert result is models

    def test_intervention_types_with_parameter_type_ignored(self, base_model_config, sample_models):
        """Test that 'parameter' intervention type is ignored by _setup_interventions."""
        from flumodelingsuite.validation.basemodel_validator import Intervention

        base_model_config.interventions = [
            Intervention(
                type="parameter",
                label="Parameter Intervention",
                target_parameter="beta",
                scaling_factor=0.5,
                start_date=date(2024, 3, 1),
                end_date=date(2024, 6, 1),
            )
        ]
        models = sample_models
        intervention_types = ["parameter"]  # This should be ignored by _setup_interventions

        with (
            patch("flumodelingsuite.workflow_dispatcher.make_school_closure_dict") as mock_closure_dict,
            patch(
                "flumodelingsuite.workflow_dispatcher._add_school_closure_intervention_from_config"
            ) as mock_add_closure,
            patch(
                "flumodelingsuite.workflow_dispatcher._add_contact_matrix_interventions_from_config"
            ) as mock_add_contact,
        ):
            result = _setup_interventions(
                models=models,
                basemodel=base_model_config,
                intervention_types=intervention_types,
                sampled_start_timespan=None,
            )

            # None should be called (parameter type not handled here)
            mock_closure_dict.assert_not_called()
            mock_add_closure.assert_not_called()
            mock_add_contact.assert_not_called()

            # Result should be the models unchanged
            assert result == models
