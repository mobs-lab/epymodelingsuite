"""Tests to verify that builder functions use non-mutating pattern (deepcopy)."""

from datetime import date

from epydemix.model import EpiModel

from flumodelingsuite.builders.base import (
    add_model_compartments_from_config,
    add_model_parameters_from_config,
    add_model_transitions_from_config,
    calculate_parameters_from_config,
    set_population_from_config,
)
from flumodelingsuite.builders.interventions import (
    add_contact_matrix_interventions_from_config,
    add_parameter_interventions_from_config,
)
from flumodelingsuite.builders.seasonality import add_seasonality_from_config
from flumodelingsuite.schema.basemodel import (
    Compartment,
    Intervention,
    Parameter,
    Seasonality,
    Timespan,
    Transition,
)


class TestNonMutatingBuilders:
    """Test that builder functions don't mutate input models."""

    def test_set_population_from_config_does_not_mutate(self):
        """Verify that set_population_from_config doesn't mutate the input model."""
        model = EpiModel()
        original_id = id(model)
        original_population_name = model.population.name
        original_num_groups = model.population.num_groups

        result = set_population_from_config(model, "US-CA", ["0-4", "5-17", "18-64", "65+"])

        # Result should be a different object
        assert id(result) != original_id
        # Original model population should be unchanged
        assert model.population.name == original_population_name
        assert model.population.num_groups == original_num_groups
        # Result should have different population
        assert result.population.name != original_population_name
        assert result.population.num_groups != original_num_groups

    def test_add_model_compartments_from_config_does_not_mutate(self):
        """Verify that add_model_compartments_from_config doesn't mutate the input model."""
        model = EpiModel()
        original_id = id(model)
        original_compartments = len(model.compartments_idx)

        compartments = [
            Compartment(id="S", label="Susceptible", init="default"),
            Compartment(id="I", label="Infected", init=10),
            Compartment(id="R", label="Recovered", init=0),
        ]

        result = add_model_compartments_from_config(model, compartments)

        # Result should be a different object
        assert id(result) != original_id
        # Original model should be unchanged
        assert len(model.compartments_idx) == original_compartments
        # Result should have new compartments
        assert len(result.compartments_idx) == original_compartments + 3

    def test_add_model_transitions_from_config_does_not_mutate(self):
        """Verify that add_model_transitions_from_config doesn't mutate the input model."""
        model = EpiModel()
        model.add_compartments(["S", "I", "R"])
        original_id = id(model)
        original_transitions = len(model.transitions_list)

        transitions = [
            Transition(source="S", target="I", type="mediated", rate="beta", mediator="I"),
            Transition(source="I", target="R", type="spontaneous", rate="gamma"),
        ]

        result = add_model_transitions_from_config(model, transitions)

        # Result should be a different object
        assert id(result) != original_id
        # Original model should be unchanged
        assert len(model.transitions_list) == original_transitions
        # Result should have new transitions
        assert len(result.transitions_list) == original_transitions + 2

    def test_add_model_parameters_from_config_does_not_mutate(self):
        """Verify that add_model_parameters_from_config doesn't mutate the input model."""
        model = EpiModel()
        original_id = id(model)
        original_params = len(model.parameters)

        parameters = {
            "beta": Parameter(type="scalar", value=0.5),
            "gamma": Parameter(type="scalar", value=0.1),
        }

        result = add_model_parameters_from_config(model, parameters)

        # Result should be a different object
        assert id(result) != original_id
        # Original model should be unchanged
        assert len(model.parameters) == original_params
        # Result should have new parameters
        assert len(result.parameters) == original_params + 2

    def test_calculate_parameters_from_config_does_not_mutate(self):
        """Verify that calculate_parameters_from_config doesn't mutate the input model."""
        model = EpiModel()
        model.add_parameter(parameters_dict={"R0": 2.5, "gamma": 0.1})
        original_id = id(model)
        original_params = len(model.parameters)

        parameters = {
            "R0": Parameter(type="scalar", value=2.5),
            "gamma": Parameter(type="scalar", value=0.1),
            "beta": Parameter(type="calculated", value="R0 * gamma"),
        }

        result = calculate_parameters_from_config(model, parameters)

        # Result should be a different object
        assert id(result) != original_id
        # Original model should be unchanged
        assert len(model.parameters) == original_params
        assert "beta" not in model.parameters
        # Result should have calculated parameter
        assert "beta" in result.parameters

    def test_add_seasonality_from_config_does_not_mutate(self):
        """Verify that add_seasonality_from_config doesn't mutate the input model."""
        model = EpiModel()
        model.add_parameter(parameters_dict={"beta": 0.5})
        original_id = id(model)
        original_beta = model.get_parameter("beta")

        seasonality = Seasonality(
            method="balcan",
            target_parameter="beta",
            seasonality_min_date=date(2024, 7, 1),
            seasonality_max_date=date(2024, 1, 15),
            min_value=0.8,
            max_value=1.2,
        )
        timespan = Timespan(start_date=date(2024, 1, 1), end_date=date(2024, 12, 31), delta_t=1.0)

        result = add_seasonality_from_config(model, seasonality, timespan)

        # Result should be a different object
        assert id(result) != original_id
        # Original model parameter should be unchanged
        assert model.get_parameter("beta") == original_beta
        # Result parameter should be different (time-varying)
        import numpy as np

        assert not np.array_equal(result.get_parameter("beta"), original_beta)

    def test_add_contact_matrix_interventions_from_config_does_not_mutate(self):
        """Verify that add_contact_matrix_interventions_from_config doesn't mutate the input model."""
        model = EpiModel()
        model = set_population_from_config(model, "US-CA", ["0-4", "5-17", "18-64", "65+"])
        original_id = id(model)
        original_interventions = len(model.interventions)

        interventions = [
            Intervention(
                type="contact_matrix",
                label="Test Intervention",
                contact_matrix_layer="home",
                start_date=date(2024, 3, 1),
                end_date=date(2024, 6, 1),
                scaling_factor=0.5,
            )
        ]

        result = add_contact_matrix_interventions_from_config(model, interventions)

        # Result should be a different object
        assert id(result) != original_id
        # Original model should be unchanged
        assert len(model.interventions) == original_interventions
        # Result should have new intervention
        assert len(result.interventions) == original_interventions + 1

    def test_add_parameter_interventions_from_config_does_not_mutate(self):
        """Verify that add_parameter_interventions_from_config doesn't mutate the input model."""
        model = EpiModel()
        model.add_parameter(parameters_dict={"beta": 0.5})
        original_id = id(model)
        original_beta = model.get_parameter("beta")

        interventions = [
            Intervention(
                type="parameter",
                label="Test Intervention",
                target_parameter="beta",
                start_date=date(2024, 3, 1),
                end_date=date(2024, 6, 1),
                scaling_factor=0.7,
            )
        ]
        timespan = Timespan(start_date=date(2024, 1, 1), end_date=date(2024, 12, 31), delta_t=1.0)

        result = add_parameter_interventions_from_config(model, interventions, timespan)

        # Result should be a different object
        assert id(result) != original_id
        # Original model parameter should be unchanged
        assert model.get_parameter("beta") == original_beta
        # Result parameter should be modified (time-varying)
        import numpy as np

        assert not np.array_equal(result.get_parameter("beta"), original_beta)

    def test_chained_builders_do_not_mutate(self):
        """Verify that chaining builder functions doesn't mutate the original model."""
        original_model = EpiModel()
        original_id = id(original_model)
        original_num_compartments = len(original_model.compartments_idx)
        original_num_transitions = len(original_model.transitions_list)
        original_num_parameters = len(original_model.parameters)

        # Chain multiple builder operations
        model = set_population_from_config(original_model, "US-CA", ["0-4", "5-17", "18-64", "65+"])

        compartments = [
            Compartment(id="S", label="Susceptible", init="default"),
            Compartment(id="I", label="Infected", init=10),
            Compartment(id="R", label="Recovered", init=0),
        ]
        model = add_model_compartments_from_config(model, compartments)

        transitions = [
            Transition(source="S", target="I", type="mediated", rate="beta", mediator="I"),
            Transition(source="I", target="R", type="spontaneous", rate="gamma"),
        ]
        model = add_model_transitions_from_config(model, transitions)

        parameters = {
            "beta": Parameter(type="scalar", value=0.5),
            "gamma": Parameter(type="scalar", value=0.1),
        }
        model = add_model_parameters_from_config(model, parameters)

        # Original model should remain completely unchanged
        assert id(model) != original_id
        assert len(original_model.compartments_idx) == original_num_compartments
        assert len(original_model.transitions_list) == original_num_transitions
        assert len(original_model.parameters) == original_num_parameters

        # Final model should have all the components
        assert model.population is not None
        assert len(model.compartments_idx) == 3
        assert len(model.transitions_list) == 2
        assert len(model.parameters) == 2
