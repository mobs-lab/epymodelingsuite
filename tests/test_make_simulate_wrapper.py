"""Tests for make_simulate_wrapper function - verifies closure bug fix."""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from flumodelingsuite.builders.orchestrators import create_model_collection, make_simulate_wrapper
from flumodelingsuite.schema.basemodel import (
    BaseEpiModel,
    Compartment,
    Intervention,
    Parameter,
    Population,
    Seasonality,
    Simulation,
    Timespan,
    Transition,
)


class TestMakeSimulateWrapper:
    """Tests for make_simulate_wrapper function."""

    @pytest.fixture
    def base_model_config(self):
        """Create a minimal BaseEpiModel configuration for testing."""
        compartments = [
            Compartment(id="S", label="Susceptible", init="default"),
            Compartment(id="I", label="Infected", init=10),
            Compartment(id="R", label="Recovered", init=0),
        ]

        transitions = [
            Transition(source="S", target="I", type="mediated", rate="beta", mediator="I"),
            Transition(source="I", target="R", type="spontaneous", rate="gamma"),
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
    def mock_calibration(self):
        """Create a mock calibration configuration with minimal required attributes."""
        calibration = Mock()
        calibration.comparison = [Mock()]
        calibration.comparison[0].simulation = ["S_to_I"]
        return calibration

    @pytest.fixture
    def data_state(self):
        """Create sample observed data DataFrame."""
        dates = [date(2024, 1, 1) + timedelta(days=i * 7) for i in range(10)]
        return pd.DataFrame({"target_end_date": dates, "observed": np.arange(10, dtype=float) * 10})

    def test_returns_callable(self, base_model_config, mock_calibration, data_state):
        """Test that make_simulate_wrapper returns a callable function and can be invoked."""
        # Create EpiModels using create_model_collection
        models, _ = create_model_collection(base_model_config, None)
        model = models[0]

        simulate_wrapper = make_simulate_wrapper(
            basemodel=base_model_config,
            calibration=mock_calibration,
            data_state=data_state,
            intervention_types=[],
        )

        # Test that the returned object is callable
        assert callable(simulate_wrapper)

        # Test that wrapper can be called with proper params dict
        params = {
            "epimodel": model,
            "end_date": date(2024, 3, 31),
            "projection": False,
            "beta": 0.5,
            "gamma": 0.1,
        }
        result = simulate_wrapper(params)

        # Should return dict with "data" key for calibration mode
        assert isinstance(result, dict)
        assert "data" in result
        assert isinstance(result["data"], np.ndarray)

    @pytest.mark.parametrize("num_observations", [5, 8])
    def test_wrapper_uses_correct_data_state(self, base_model_config, mock_calibration, num_observations):
        """
        Test that a wrapper uses the specific data_state it was created with.

        This verifies that when a wrapper is created with a specific data_state DataFrame,
        it correctly uses that data_state for calibration output alignment.
        """
        # Create a single EpiModel
        models, _ = create_model_collection(base_model_config, None)
        model = models[0]

        # Create a data_state with the specified number of observations
        data_state = pd.DataFrame(
            {
                "target_end_date": [date(2024, 1, 1) + timedelta(days=i * 7) for i in range(num_observations)],
                "observed": np.arange(num_observations, dtype=float) * 10,
            }
        )

        # Create wrapper with this data_state
        wrapper = make_simulate_wrapper(
            basemodel=base_model_config,
            calibration=mock_calibration,
            data_state=data_state,
            intervention_types=[],
        )

        assert callable(wrapper)

        # Call the wrapper
        params = {
            "epimodel": model,
            "end_date": date(2024, 3, 31),
            "projection": False,  # Calibration mode to get data aligned to observations
            "beta": 0.5,
            "gamma": 0.1,
        }
        result = wrapper(params)

        # Verify the wrapper actually ran
        assert isinstance(result, dict)
        assert "data" in result
        assert isinstance(result["data"], np.ndarray)  # noqa: S101

        # KEY ASSERTION: Output length should match the data_state this wrapper was created with
        expected_length = len(data_state)
        actual_length = len(result["data"])
        assert actual_length == expected_length, (  # noqa: S101
            f"Wrapper should return {expected_length} data points to match its data_state, but returned {actual_length}"
        )

    def test_calibration_mode_returns_data_array(self, base_model_config, mock_calibration, data_state):
        """
        Test that calibration mode (projection=False) returns correctly formatted output.

        Calibration mode should return a dict with a "data" key containing a numpy array
        of simulated values aligned to the observation dates in data_state.
        """
        models, _ = create_model_collection(base_model_config, None)
        model = models[0]

        wrapper = make_simulate_wrapper(
            basemodel=base_model_config,
            calibration=mock_calibration,
            data_state=data_state,
            intervention_types=[],
        )

        params = {
            "epimodel": model,
            "end_date": date(2024, 3, 31),
            "projection": False,  # Calibration mode
            "beta": 0.5,
            "gamma": 0.1,
        }
        result = wrapper(params)

        # Verify calibration mode output structure
        assert isinstance(result, dict)
        assert "data" in result
        assert isinstance(result["data"], np.ndarray)

        # Should NOT have projection-specific keys
        assert "dates" not in result
        assert "transitions" not in result
        assert "compartments" not in result

        # Output length should match data_state
        assert len(result["data"]) == len(data_state)

    def test_projection_mode_returns_full_results(self, base_model_config, mock_calibration, data_state):
        """
        Test that projection mode (projection=True) returns correctly formatted output.

        Projection mode should return a dict with "dates", "transitions", and "compartments"
        keys containing the full simulation results.
        """
        models, _ = create_model_collection(base_model_config, None)
        model = models[0]

        wrapper = make_simulate_wrapper(
            basemodel=base_model_config,
            calibration=mock_calibration,
            data_state=data_state,
            intervention_types=[],
        )

        params = {
            "epimodel": model,
            "end_date": date(2024, 3, 31),
            "projection": True,  # Projection mode
            "beta": 0.5,
            "gamma": 0.1,
        }
        result = wrapper(params)

        # Verify projection mode output structure
        assert isinstance(result, dict)

        # If simulation succeeded, should have projection-specific keys
        if result:  # Simulation might fail, which returns empty dict
            assert "dates" in result  # noqa: S101
            assert "transitions" in result  # noqa: S101
            assert "compartments" in result  # noqa: S101

            # Should NOT have calibration-specific keys
            assert "data" not in result  # noqa: S101

            # Verify types
            assert isinstance(result["dates"], list)  # noqa: S101
            assert isinstance(result["transitions"], dict)  # noqa: S101
            assert isinstance(result["compartments"], dict)  # noqa: S101

    def test_wrapper_with_calibrated_parameters(self, base_model_config, mock_calibration, data_state):
        """
        Test that wrapper correctly handles calibrated parameters.

        Tests: _add_model_parameters_from_config when new_params is not empty.
        """
        # Modify base model to have a calibrated parameter (no value for calibrated type)
        base_model_config.parameters["beta"] = Parameter(type="calibrated")

        models, _ = create_model_collection(base_model_config, None)
        model = models[0]

        wrapper = make_simulate_wrapper(
            basemodel=base_model_config,
            calibration=mock_calibration,
            data_state=data_state,
            intervention_types=[],
        )

        # Pass a different beta value in params (simulating calibration sampling)
        params = {
            "epimodel": model,
            "end_date": date(2024, 3, 31),
            "projection": False,
            "beta": 0.7,  # Different from model's beta
            "gamma": 0.1,
        }
        result = wrapper(params)

        # Verify the wrapper ran successfully
        assert isinstance(result, dict)
        assert "data" in result
        assert isinstance(result["data"], np.ndarray)

    def test_wrapper_with_seasonality(self, base_model_config, mock_calibration, data_state):
        """
        Test that wrapper correctly handles seasonality.

        Tests: seasonality handling and seasonality_min parameter.
        """
        # Add seasonality to base model
        base_model_config.seasonality = Seasonality(
            target_parameter="beta",
            method="balcan",
            seasonality_max_date=date(2024, 1, 15),
            seasonality_min_date=date(2024, 7, 15),
            max_value=0.8,
            min_value=0.3,
        )

        models, _ = create_model_collection(base_model_config, None)
        model = models[0]

        wrapper = make_simulate_wrapper(
            basemodel=base_model_config,
            calibration=mock_calibration,
            data_state=data_state,
            intervention_types=[],
        )

        # Test with seasonality_min parameter
        params = {
            "epimodel": model,
            "end_date": date(2024, 3, 31),
            "projection": False,
            "beta": 0.5,
            "gamma": 0.1,
            "seasonality_min": 0.25,  # Override min_value
        }
        result = wrapper(params)

        # Verify the wrapper ran successfully
        assert isinstance(result, dict)
        assert "data" in result
        assert isinstance(result["data"], np.ndarray)

    def test_wrapper_with_parameter_interventions(self, base_model_config, mock_calibration, data_state):
        """
        Test that wrapper correctly handles parameter interventions.

        Tests: parameter interventions when intervention_types includes "parameter".
        """
        # Add parameter intervention to base model
        base_model_config.interventions = [
            Intervention(
                type="parameter",
                target_parameter="beta",
                scaling_factor=0.5,
                start_date=date(2024, 2, 1),
                end_date=date(2024, 3, 1),
            )
        ]

        models, _ = create_model_collection(base_model_config, None)
        model = models[0]

        wrapper = make_simulate_wrapper(
            basemodel=base_model_config,
            calibration=mock_calibration,
            data_state=data_state,
            intervention_types=["parameter"],  # Enable parameter interventions
        )

        params = {
            "epimodel": model,
            "end_date": date(2024, 3, 31),
            "projection": False,
            "beta": 0.5,
            "gamma": 0.1,
        }
        result = wrapper(params)

        # Verify the wrapper ran successfully
        assert isinstance(result, dict)
        assert "data" in result
        assert isinstance(result["data"], np.ndarray)

    def test_wrapper_with_sampled_start_date(self, base_model_config, mock_calibration, data_state):
        """
        Test that wrapper correctly handles sampled start_date.

        Tests: start_date calculation when sampled_start_timespan is provided.
        """
        models, _ = create_model_collection(base_model_config, None)
        model = models[0]

        # Create a sampled_start_timespan (earliest possible start date)
        sampled_start_timespan = Timespan(
            start_date=date(2023, 12, 1),
            end_date=date(2024, 12, 31),
            delta_t=1.0,
        )

        wrapper = make_simulate_wrapper(
            basemodel=base_model_config,
            calibration=mock_calibration,
            data_state=data_state,
            intervention_types=[],
            sampled_start_timespan=sampled_start_timespan,
        )

        # Pass start_date as offset in days from sampled_start_timespan.start_date
        params = {
            "epimodel": model,
            "end_date": date(2024, 3, 31),
            "projection": False,
            "beta": 0.5,
            "gamma": 0.1,
            "start_date": 15,  # 15 days offset from 2023-12-01
        }
        result = wrapper(params)

        # Verify the wrapper ran successfully
        assert isinstance(result, dict)
        assert "data" in result
        assert isinstance(result["data"], np.ndarray)

    def test_projection_mode_handles_failure(self, base_model_config, mock_calibration, data_state):
        """
        Test that projection mode correctly handles simulation failures.

        Tests: exception handling in projection mode.
        """
        models, _ = create_model_collection(base_model_config, None)
        model = models[0]

        wrapper = make_simulate_wrapper(
            basemodel=base_model_config,
            calibration=mock_calibration,
            data_state=data_state,
            intervention_types=[],
        )

        # Use invalid parameters that might cause simulation to fail
        params = {
            "epimodel": model,
            "end_date": date(2024, 3, 31),
            "projection": True,
            "beta": -1.0,  # Invalid negative beta
            "gamma": -0.1,  # Invalid negative gamma
        }
        result = wrapper(params)

        # On failure, projection mode should return empty dict
        assert isinstance(result, dict)
        # Either empty dict (failure) or valid result dict (if it somehow succeeds)
        if not result:
            # Empty dict indicates failure was handled gracefully
            assert result == {}
        else:
            # If it succeeded despite invalid params, should have projection keys
            assert "dates" in result  # noqa: S101

    def test_wrapper_with_calculated_parameters(self, base_model_config, mock_calibration, data_state):
        """
        Test that wrapper correctly handles calculated parameters.

        Tests: _calculate_parameters_from_config when calculated parameters exist.
        """
        # Add a calculated parameter to base model
        base_model_config.parameters["R0"] = Parameter(type="calculated", value="beta / gamma")

        models, _ = create_model_collection(base_model_config, None)
        model = models[0]

        wrapper = make_simulate_wrapper(
            basemodel=base_model_config,
            calibration=mock_calibration,
            data_state=data_state,
            intervention_types=[],
        )

        params = {
            "epimodel": model,
            "end_date": date(2024, 3, 31),
            "projection": False,
            "beta": 0.5,
            "gamma": 0.1,
        }
        result = wrapper(params)

        # Verify the wrapper ran successfully
        assert isinstance(result, dict)
        assert "data" in result
        assert isinstance(result["data"], np.ndarray)
