"""Tests for make_simulate_wrapper function - verifies closure bug fix."""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from epymodelingsuite.builders.orchestrators import create_model_collection, make_simulate_wrapper
from epymodelingsuite.schema.basemodel import (
    BaseEpiModel,
    Intervention,
    Parameter,
    Seasonality,
    Timespan,
)
from tests.conftest import make_sir_config


class TestMakeSimulateWrapper:
    """Tests for make_simulate_wrapper function."""

    @pytest.fixture
    def base_model_config(self):
        """Create a minimal BaseEpiModel configuration for testing."""
        return make_sir_config()

    @pytest.fixture
    def mock_calibration(self):
        """Create a mock calibration configuration with minimal required attributes."""
        calibration = Mock()
        calibration.comparison = [Mock()]
        calibration.comparison[0].simulation = ["S_to_I_total"]  # Transition key format: source_to_target_total
        calibration.comparison[0].observed_date_column = "target_end_date"
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
            observed_data=data_state,
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
            observed_data=data_state,
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
            observed_data=data_state,
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

        # Should have "date" key (from format_calibration_data)
        assert "date" in result

        # Should NOT have nested projection-specific keys
        assert "transitions" not in result
        assert "compartments" not in result

        # Output length should match data_state
        assert len(result["data"]) == len(data_state)
        assert len(result["date"]) == len(data_state)

    def test_projection_mode_returns_full_results(self, base_model_config, mock_calibration, data_state):
        """
        Test that projection mode (projection=True) returns correctly formatted output.

        Projection mode should return a flattened dict with "date" key and all
        transition/compartment keys at the top level.
        """
        models, _ = create_model_collection(base_model_config, None)
        model = models[0]

        wrapper = make_simulate_wrapper(
            basemodel=base_model_config,
            calibration=mock_calibration,
            observed_data=data_state,
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

        # If simulation succeeded, should have flattened structure with date and all transitions/compartments
        if result:  # Simulation might fail, which returns empty dict
            assert "date" in result  # noqa: S101

            # Should NOT have calibration-specific keys
            assert "data" not in result  # noqa: S101

            # Verify date type
            assert isinstance(result["date"], list)  # noqa: S101

            # Verify flattened structure has transition and compartment keys at top level
            # (not nested under "transitions" or "compartments" keys)
            # Should have transition keys like "S_to_I_total", "I_to_R_total", etc.
            # Should have compartment keys like "S_0-4", "I_0-4", "R_0-4", etc.
            assert any(key.startswith("S_to_I") for key in result)  # noqa: S101
            assert any(key.startswith("S_") for key in result if "_to_" not in key)  # noqa: S101

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
            observed_data=data_state,
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
            observed_data=data_state,
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
            observed_data=data_state,
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
            observed_data=data_state,
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
            observed_data=data_state,
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
            assert "date" in result  # noqa: S101

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
            observed_data=data_state,
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

    def test_no_shared_state_mutation_across_wrappers(self, base_model_config, mock_calibration, data_state):
        """
        Test that wrappers don't mutate shared basemodel config (P0 Critical Issue).

        This verifies the fix for the seasonality.min_value mutation bug described in the review.
        Multiple wrappers sharing the same basemodel config should not interfere with each other.
        """
        # Add seasonality to base model
        base_model_config.seasonality = Seasonality(
            target_parameter="beta",
            method="balcan",
            seasonality_max_date=date(2024, 1, 15),
            seasonality_min_date=date(2024, 7, 15),
            max_value=1.0,
            min_value=0.5,
        )
        original_min = base_model_config.seasonality.min_value

        # Create two wrappers sharing the same basemodel config
        models, _ = create_model_collection(base_model_config, None)
        model = models[0]

        wrapper1 = make_simulate_wrapper(
            basemodel=base_model_config,
            calibration=mock_calibration,
            observed_data=data_state,
            intervention_types=[],
        )

        wrapper2 = make_simulate_wrapper(
            basemodel=base_model_config,
            calibration=mock_calibration,
            observed_data=data_state,
            intervention_types=[],
        )

        # First wrapper uses custom seasonality_min
        params1 = {
            "epimodel": model,
            "end_date": date(2024, 3, 31),
            "projection": False,
            "beta": 0.5,
            "gamma": 0.1,
            "seasonality_min": 0.25,  # Different from original
        }
        wrapper1(params1)

        # Verify basemodel config wasn't mutated
        assert base_model_config.seasonality.min_value == original_min

        # Second wrapper should still see original value (not 0.25 from wrapper1)
        params2 = {
            "epimodel": model,
            "end_date": date(2024, 3, 31),
            "projection": False,
            "beta": 0.5,
            "gamma": 0.1,
            # No seasonality_min specified - should use original 0.5
        }
        wrapper2(params2)

        # Verify basemodel config still wasn't mutated
        assert base_model_config.seasonality.min_value == original_min

    def test_duplicate_dates_raises_error(self, base_model_config, mock_calibration):
        """Test that duplicate dates in observed_data raise a clear error."""
        # Create observed_data with duplicate dates (simulating mixed location data)
        dates = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 1)]  # Date 1 appears twice
        observed_data_with_duplicates = pd.DataFrame(
            {
                "target_end_date": dates,
                "observed": [10.0, 20.0, 30.0],
            }
        )

        # Attempt to create wrapper should raise ValueError
        with pytest.raises(ValueError, match="Duplicate dates found in observed_data"):
            make_simulate_wrapper(
                basemodel=base_model_config,
                calibration=mock_calibration,
                observed_data=observed_data_with_duplicates,
                intervention_types=[],
            )

    def test_wrapper_with_post_hoc_transformation(self, base_model_config, mock_calibration, data_state):
        """
        Test that wrapper correctly applies post-hoc transformation function.

        Tests:
        - Post-hoc transformation is applied to simulation results
        - Transformation function receives Trajectory object
        - Transformation output is used in final results
        - Works in both calibration and projection modes
        """
        models, _ = create_model_collection(base_model_config, None)
        model = models[0]

        # Create a simple transformation function that adds a new compartment
        def test_transformation(trajectory):
            """Add a new compartment 'I_plus_R' that sums I and R compartments."""
            import copy

            traj = copy.deepcopy(trajectory)
            # Sum all I compartments with all R compartments (across age groups)
            i_keys = [k for k in traj.compartments.keys() if k.startswith("I_")]
            r_keys = [k for k in traj.compartments.keys() if k.startswith("R_")]

            if i_keys and r_keys:
                i_total = sum(traj.compartments[k] for k in i_keys)
                r_total = sum(traj.compartments[k] for k in r_keys)
                traj.compartments["I_plus_R_total"] = i_total + r_total

            return traj

        wrapper = make_simulate_wrapper(
            basemodel=base_model_config,
            calibration=mock_calibration,
            observed_data=data_state,
            intervention_types=[],
            post_hoc_transformation=test_transformation,
        )

        # Test in projection mode
        params_projection = {
            "epimodel": model,
            "end_date": date(2024, 3, 31),
            "projection": True,
            "beta": 0.5,
            "gamma": 0.1,
        }
        result_projection = wrapper(params_projection)

        # Verify projection mode results include transformed compartment
        if result_projection:  # If simulation succeeded
            assert "I_plus_R_total" in result_projection
            assert isinstance(result_projection["I_plus_R_total"], np.ndarray)

        # Test in calibration mode
        params_calibration = {
            "epimodel": model,
            "end_date": date(2024, 3, 31),
            "projection": False,
            "beta": 0.5,
            "gamma": 0.1,
        }
        result_calibration = wrapper(params_calibration)

        # Verify calibration mode still works (transformation applied before aggregation)
        assert isinstance(result_calibration, dict)
        assert "data" in result_calibration
        assert isinstance(result_calibration["data"], np.ndarray)

    def test_wrapper_with_failing_post_hoc_transformation(
        self, base_model_config, mock_calibration, data_state, caplog
    ):
        """
        Test that wrapper handles post-hoc transformation failures gracefully.

        Tests:
        - Exceptions in transformation are caught
        - Non-transformed results are returned on failure
        - Simulation continues despite transformation error
        - Warning is logged
        """
        models, _ = create_model_collection(base_model_config, None)
        model = models[0]

        # Create a transformation function that always raises an exception
        def failing_transformation(trajectory):
            """Transformation that always fails."""
            raise ValueError("Intentional test failure")

        wrapper = make_simulate_wrapper(
            basemodel=base_model_config,
            calibration=mock_calibration,
            observed_data=data_state,
            intervention_types=[],
            post_hoc_transformation=failing_transformation,
        )

        # Test in projection mode
        params = {
            "epimodel": model,
            "end_date": date(2024, 3, 31),
            "projection": True,
            "beta": 0.5,
            "gamma": 0.1,
        }

        # Should not raise exception - wrapper handles it gracefully
        result = wrapper(params)

        # Verify simulation completed (either succeeded or failed, but didn't crash)
        assert isinstance(result, dict)

        # Verify warning was logged
        assert any("Post-hoc transformation failed" in record.message for record in caplog.records)

        # If simulation succeeded, result should have standard keys (not transformed ones)
        if result:  # Non-empty result means simulation succeeded
            assert "date" in result
            # Should NOT have any custom transformed compartments
            assert "I_plus_R_total" not in result

    def test_wrapper_with_post_hoc_transformation_receives_context(
        self, base_model_config, mock_calibration, data_state
    ):
        """
        Test that post-hoc transformation function receives context dict.

        Tests:
        - Transformation function is called with context keyword argument
        - context contains expected keys (params, basemodel, timespan, observed_data, etc.)
        - Function can access calibrated parameter values via context['params']
        - Function can access model config via context['basemodel']
        - Function can access actual dates via context['timespan']
        - Works with optional context argument (backward compatible)
        """
        models, _ = create_model_collection(base_model_config, None)
        model = models[0]

        # Track what context was passed to the transformation
        captured_context = {}

        def context_aware_transformation(trajectory, context=None):
            """Transformation that captures context for inspection."""
            import copy

            # Capture context for testing
            if context is not None:
                captured_context.update(context)

            traj = copy.deepcopy(trajectory)
            # Use context if available
            if context:
                beta = context["params"].get("beta", 0)
                location = context["location"]
                # Add compartment with info from context
                traj.compartments["test_compartment"] = traj.compartments["S_total"] * beta

            return traj

        wrapper = make_simulate_wrapper(
            basemodel=base_model_config,
            calibration=mock_calibration,
            observed_data=data_state,
            intervention_types=[],
            post_hoc_transformation=context_aware_transformation,
        )

        # Run in projection mode
        params = {
            "epimodel": model,
            "end_date": date(2024, 3, 31),
            "projection": True,
            "beta": 0.6,
            "gamma": 0.1,
        }
        result = wrapper(params)

        # Verify context was passed
        assert captured_context, "Context should have been passed to transformation function"

        # Verify context has expected keys
        assert "params" in captured_context
        assert "basemodel" in captured_context
        assert "timespan" in captured_context
        assert "observed_data" in captured_context
        assert "intervention_types" in captured_context
        assert "projection" in captured_context
        assert "location" in captured_context

        # Verify context content
        assert captured_context["params"]["beta"] == 0.6
        assert captured_context["params"]["gamma"] == 0.1
        assert captured_context["params"]["projection"] is True
        assert captured_context["projection"] is True
        # Location name is transformed from "US-CA" to "United_States_California"
        assert captured_context["location"] == "United_States_California"
        assert isinstance(captured_context["basemodel"], BaseEpiModel)
        assert isinstance(captured_context["timespan"], Timespan)
        assert isinstance(captured_context["observed_data"], pd.DataFrame)
        assert isinstance(captured_context["intervention_types"], list)

        # Verify transformation still worked
        if result:
            assert "test_compartment" in result

    def test_wrapper_with_post_hoc_transformation_without_context_arg(
        self, base_model_config, mock_calibration, data_state
    ):
        """
        Test that transformation functions without context argument still work.

        Tests:
        - Functions with basic signature (only accepting trajectory) continue to work
        - No error when function doesn't accept context keyword
        - TypeError is caught and function is retried without context
        - Backward compatibility is maintained
        """
        models, _ = create_model_collection(base_model_config, None)
        model = models[0]

        # Track that the function was called
        call_count = {"count": 0}

        def basic_transformation(trajectory):
            """Transformation with basic signature that doesn't accept context."""
            import copy

            call_count["count"] += 1
            traj = copy.deepcopy(trajectory)
            traj.compartments["basic_compartment"] = traj.compartments["S_total"] * 2
            return traj

        wrapper = make_simulate_wrapper(
            basemodel=base_model_config,
            calibration=mock_calibration,
            observed_data=data_state,
            intervention_types=[],
            post_hoc_transformation=basic_transformation,
        )

        # Run in projection mode
        params = {
            "epimodel": model,
            "end_date": date(2024, 3, 31),
            "projection": True,
            "beta": 0.5,
            "gamma": 0.1,
        }
        result = wrapper(params)

        # Verify function was called
        assert call_count["count"] > 0, "Transformation function should have been called"

        # Verify transformation still worked despite not accepting context
        if result:
            assert "basic_compartment" in result
            assert isinstance(result["basic_compartment"], np.ndarray)
