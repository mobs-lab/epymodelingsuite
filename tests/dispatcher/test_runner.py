"""Tests for epymodelingsuite.dispatcher.runner module.

This module contains two types of tests:
1. Unit tests (mocked) - Fast tests for routing, error handling, and edge cases
2. Integration tests (real EpiModel) - Tests with actual model execution

The integration tests verify that runners work correctly with real EpiModel objects,
while unit tests verify routing logic and error handling without expensive model runs.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from epydemix.calibration import ABCSampler, CalibrationResults
from epydemix.calibration.metrics import rmse
from epydemix.model.simulation_results import SimulationResults
from scipy.stats import uniform

from epymodelingsuite.builders.orchestrators import create_model_collection
from epymodelingsuite.dispatcher.runner import (
    dispatch_runner,
    run_calibration,
    run_calibration_with_projection,
    run_simulation,
)
from epymodelingsuite.schema.basemodel import (
    BaseEpiModel,
    Compartment,
    Parameter,
    Population,
    Simulation,
    Timespan,
    Transition,
)
from epymodelingsuite.schema.calibration import CalibrationStrategy
from epymodelingsuite.schema.dispatcher import (
    BuilderOutput,
    CalibrationOutput,
    ProjectionArguments,
    SimulationArguments,
    SimulationOutput,
)

# =============================================================================
# Helper Functions
# =============================================================================


def create_builder_output(**kwargs):
    """Create a BuilderOutput bypassing Pydantic validation for testing with mocks."""
    return BuilderOutput.model_construct(**kwargs)


# =============================================================================
# Fixtures for Integration Tests
# =============================================================================


@pytest.fixture
def base_epimodel_config():
    """Create a minimal BaseEpiModel configuration for testing.

    This creates a simple SIR model that can be used to create real EpiModel objects.
    """
    compartments = [
        Compartment(id="S", label="Susceptible", init="default"),
        Compartment(id="I", label="Infected", init=100),
        Compartment(id="R", label="Recovered", init=0),
    ]

    transitions = [
        Transition(source="S", target="I", type="mediated", rate="beta", mediator="I"),
        Transition(source="I", target="R", type="spontaneous", rate="gamma"),
    ]

    parameters = {
        "beta": Parameter(type="scalar", value=0.3),
        "gamma": Parameter(type="scalar", value=0.1),
    }

    population = Population(name="US-MA", age_groups=["0-4", "5-17", "18-49", "50-64", "65+"])

    timespan = Timespan(start_date=date(2024, 1, 1), end_date=date(2024, 2, 1), delta_t=1.0)

    simulation = Simulation(n_sims=3, resample_frequency="W-SAT")

    return BaseEpiModel(
        name="test_sir_model",
        compartments=compartments,
        transitions=transitions,
        parameters=parameters,
        population=population,
        timespan=timespan,
        simulation=simulation,
    )


@pytest.fixture
def real_simulation_builder_output(base_epimodel_config):
    """Create a BuilderOutput with a real EpiModel for simulation testing."""
    models, _ = create_model_collection(base_epimodel_config, None)
    model = models[0]

    # Calculate initial conditions with numeric values
    # The model has 5 age groups, distribute infections across them
    n_age_groups = len(model.population.Nk)
    i_init = np.zeros(n_age_groups)
    i_init[2] = 100  # Put initial infections in 18-49 age group
    r_init = np.zeros(n_age_groups)
    s_init = model.population.Nk - i_init - r_init  # Remaining population

    simulation_args = SimulationArguments(
        start_date=base_epimodel_config.timespan.start_date,
        end_date=date(2024, 2, 3),  # Use a Sunday for W-SAT resampling
        initial_conditions_dict={"S": s_init, "I": i_init, "R": r_init},
        Nsim=3,
        dt=1.0,
        resample_frequency="W-SAT",
    )

    return BuilderOutput(
        primary_id=0,
        seed=42,
        delta_t=1.0,
        model=model,
        simulation=simulation_args,
    )


@pytest.fixture
def real_calibration_builder_output(base_epimodel_config):
    """Create a BuilderOutput with a real ABCSampler for calibration testing.

    Uses minimal settings (few particles, large epsilon) for fast execution.
    The goal is to verify the runner executes calibration correctly, not to
    test calibration convergence or fit quality.
    """
    models, _ = create_model_collection(base_epimodel_config, None)
    model = models[0]

    # Calculate initial conditions
    n_age_groups = len(model.population.Nk)
    i_init = np.zeros(n_age_groups)
    i_init[2] = 100  # Put initial infections in 18-49 age group
    r_init = np.zeros(n_age_groups)
    s_init = model.population.Nk - i_init - r_init

    # Generate synthetic observed data (simple increasing values like new infections)
    start_date = base_epimodel_config.timespan.start_date
    end_date = start_date + pd.Timedelta(days=9)
    observed_data = np.array([10, 20, 35, 55, 80, 110, 140, 165, 185, 200])

    # Simulate wrapper following the pattern from epydemix tutorial:
    # - Takes params dict, unpacks to simulate()
    # - Returns {"data": transitions_array}
    def simulate_wrapper(params):
        """Simulate the model and return I_to_R transitions."""
        from epydemix import simulate

        # Update beta in the model
        params["epimodel"].parameters["beta"] = params["beta"]

        # Call simulate with unpacked params (matches tutorial pattern)
        results = simulate(
            epimodel=params["epimodel"],
            start_date=start_date,
            end_date=end_date,
            initial_conditions_dict={"S": s_init, "I": i_init, "R": r_init},
            dt=1.0,
        )

        # Return transitions as dict with "data" key
        return {"data": results.transitions["I_to_R_total"]}

    # Create ABCSampler with minimal settings for speed
    # observed_data: pass raw array (ABCSampler wraps it as {'data': ...})
    # simulation return: must be dict with "data" key
    abc_sampler = ABCSampler(
        simulation_function=simulate_wrapper,
        priors={"beta": uniform(loc=0.1, scale=0.4)},  # uniform(0.1, 0.5)
        parameters={"epimodel": model},
        observed_data=observed_data,
        distance_function=rmse,
    )

    # Minimal calibration strategy for fast execution
    strategy = CalibrationStrategy(name="rejection", options={"num_particles": 10, "epsilon": 1e9})

    return BuilderOutput(
        primary_id=0,
        seed=42,
        delta_t=1.0,
        model=model,
        calibrator=abc_sampler,
        calibration=strategy,
    )


# =============================================================================
# PART 1: Unit Tests (Mocked) - Routing and Error Handling
# =============================================================================


class TestDispatchRunnerRouting:
    """Unit tests for dispatch_runner routing logic."""

    def test_routes_to_run_simulation(self):
        """Test that dispatch_runner routes to run_simulation for simulation workflow."""
        mock_model = MagicMock()
        mock_model.population.name = "test_location"

        simulation_args = SimulationArguments(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 31),
            initial_conditions_dict={"S": 1000},
            Nsim=5,
            dt=1.0,
        )

        builder_output = create_builder_output(
            primary_id=0,
            seed=42,
            delta_t=1.0,
            model=mock_model,
            simulation=simulation_args,
        )

        with patch("epymodelingsuite.dispatcher.runner.run_simulation") as mock_run:
            mock_run.return_value = SimulationOutput.model_construct(
                primary_id=0, seed=42, delta_t=1.0, population="test_location", results=None
            )

            result = dispatch_runner(builder_output)

            mock_run.assert_called_once()
            assert isinstance(result, SimulationOutput)

    def test_routes_to_run_calibration(self):
        """Test that dispatch_runner routes to run_calibration for calibration workflow."""
        mock_model = MagicMock()
        mock_model.population.name = "test_location"
        mock_calibrator = MagicMock()

        strategy = CalibrationStrategy(name="rejection", options={"n_samples": 100})

        builder_output = create_builder_output(
            primary_id=0,
            seed=42,
            delta_t=1.0,
            model=mock_model,
            calibrator=mock_calibrator,
            calibration=strategy,
        )

        with patch("epymodelingsuite.dispatcher.runner.run_calibration") as mock_run:
            mock_run.return_value = CalibrationOutput.model_construct(
                primary_id=0, seed=42, delta_t=1.0, population="test_location", results=None
            )

            result = dispatch_runner(builder_output)

            mock_run.assert_called_once()
            assert isinstance(result, CalibrationOutput)

    def test_routes_to_run_calibration_with_projection(self):
        """Test that dispatch_runner routes to run_calibration_with_projection for projection workflow."""
        mock_model = MagicMock()
        mock_model.population.name = "test_location"
        mock_calibrator = MagicMock()
        mock_calibrator.parameters = {"epimodel": MagicMock()}
        mock_calibrator.parameters["epimodel"].population.name = "test_location"

        strategy = CalibrationStrategy(name="smc", options={"n_samples": 100})
        projection = ProjectionArguments(end_date=date(2024, 6, 30), n_trajectories=50)

        builder_output = create_builder_output(
            primary_id=0,
            seed=42,
            delta_t=1.0,
            model=mock_model,
            calibrator=mock_calibrator,
            calibration=strategy,
            projection=projection,
        )

        with patch("epymodelingsuite.dispatcher.runner.run_calibration_with_projection") as mock_run:
            mock_run.return_value = CalibrationOutput.model_construct(
                primary_id=0, seed=42, delta_t=1.0, population="test_location", results=None
            )

            result = dispatch_runner(builder_output)

            mock_run.assert_called_once()
            assert isinstance(result, CalibrationOutput)

    def test_raises_assertion_error_for_invalid_config(self):
        """Test that AssertionError is raised for BuilderOutput without simulation or calibration."""
        invalid_output = create_builder_output(
            primary_id=0,
            seed=42,
            delta_t=1.0,
            model=MagicMock(),
        )

        with pytest.raises(AssertionError, match="Runner called without simulation or calibration specs"):
            dispatch_runner(invalid_output)


class TestErrorHandling:
    """Unit tests for error handling in runner functions."""

    def test_simulation_raises_runtime_error_on_failure(self):
        """Test that RuntimeError is raised when simulation fails."""
        mock_model = MagicMock()
        mock_model.population.name = "test_location"
        mock_model.run_simulations.side_effect = Exception("Simulation failed")

        simulation_args = SimulationArguments(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 31),
            initial_conditions_dict={"S": 1000},
            Nsim=5,
            dt=1.0,
        )

        builder_output = create_builder_output(
            primary_id=0,
            seed=42,
            delta_t=1.0,
            model=mock_model,
            simulation=simulation_args,
        )

        with pytest.raises(RuntimeError, match="Error during simulation"):
            run_simulation(builder_output)

    def test_simulation_error_includes_original_message(self):
        """Test that the RuntimeError message includes the original error message."""
        mock_model = MagicMock()
        mock_model.population.name = "test_location"
        mock_model.run_simulations.side_effect = ValueError("Invalid parameter value")

        simulation_args = SimulationArguments(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 31),
            initial_conditions_dict={"S": 1000},
            Nsim=5,
            dt=1.0,
        )

        builder_output = create_builder_output(
            primary_id=0,
            seed=42,
            delta_t=1.0,
            model=mock_model,
            simulation=simulation_args,
        )

        with pytest.raises(RuntimeError, match="Invalid parameter value"):
            run_simulation(builder_output)

    def test_calibration_raises_runtime_error_on_failure(self):
        """Test that RuntimeError is raised when calibration fails."""
        mock_calibrator = MagicMock()
        mock_calibrator.calibrate.side_effect = Exception("Calibration failed")

        mock_model = MagicMock()
        mock_model.population.name = "test_location"

        strategy = CalibrationStrategy(name="rejection", options={"n_samples": 100})

        builder_output = create_builder_output(
            primary_id=0,
            seed=42,
            delta_t=1.0,
            model=mock_model,
            calibrator=mock_calibrator,
            calibration=strategy,
        )

        with pytest.raises(RuntimeError, match="Error during calibration"):
            run_calibration(builder_output)


class TestProjectionFallbackBehavior:
    """Unit tests for projection fallback behavior when projection fails."""

    @pytest.fixture
    def projection_builder_output(self):
        """Create a BuilderOutput configured for calibration with projection."""
        mock_calibration_results = MagicMock(spec=CalibrationResults)
        mock_projection_results = MagicMock(spec=CalibrationResults)

        mock_calibrator = MagicMock()
        mock_calibrator.calibrate.return_value = mock_calibration_results
        mock_calibrator.run_projections.return_value = mock_projection_results
        mock_calibrator.parameters = {"epimodel": MagicMock()}
        mock_calibrator.parameters["epimodel"].population.name = "test_location"

        mock_model = MagicMock()
        mock_model.population.name = "test_location"

        strategy = CalibrationStrategy(name="smc", options={"n_samples": 100, "generations": 2})
        projection = ProjectionArguments(end_date=date(2024, 6, 30), n_trajectories=50, generation_number=1)

        return create_builder_output(
            primary_id=5,
            seed=777,
            delta_t=1.0,
            model=mock_model,
            calibrator=mock_calibrator,
            calibration=strategy,
            projection=projection,
            start_date_reference=date(2024, 2, 1),
        )

    def test_calibration_error_raises_runtime_error(self, projection_builder_output):
        """Test that RuntimeError is raised when calibration fails in projection workflow."""
        projection_builder_output.calibrator.calibrate.side_effect = Exception("Calibration failed")

        with pytest.raises(RuntimeError, match="Error during calibration"):
            run_calibration_with_projection(projection_builder_output)

        # run_projections should not be called if calibration fails
        projection_builder_output.calibrator.run_projections.assert_not_called()

    def test_projection_error_returns_calibration_results(self, projection_builder_output):
        """Test that projection error returns calibration results instead of raising.

        This is important fallback behavior: if calibration succeeds but projection fails,
        we return the calibration results rather than losing all work.
        """
        projection_builder_output.calibrator.run_projections.side_effect = Exception("Projection failed")

        result = run_calibration_with_projection(projection_builder_output)

        # Should return calibration results, not raise
        assert result.results is projection_builder_output.calibrator.calibrate.return_value

    def test_projection_error_logs_warning(self, projection_builder_output, caplog):
        """Test that projection error logs a warning message."""
        projection_builder_output.calibrator.run_projections.side_effect = Exception("Projection failed")

        with caplog.at_level("WARNING"):
            run_calibration_with_projection(projection_builder_output)

        assert "projection failed" in caplog.text.lower()
        assert "primary_id=5" in caplog.text


# =============================================================================
# PART 2: Integration Tests (Real EpiModel)
# =============================================================================


@pytest.mark.dynamics
class TestRunSimulationIntegration:
    """Integration tests for run_simulation with real EpiModel objects.

    These tests verify that the runner correctly executes simulations and
    produces valid results, not just that it passes arguments correctly.
    """

    def test_produces_valid_simulation_results(self, real_simulation_builder_output):
        """Test that run_simulation produces valid SimulationResults with real model."""
        result = run_simulation(real_simulation_builder_output)

        assert isinstance(result, SimulationOutput)
        assert isinstance(result.results, SimulationResults)
        assert result.primary_id == 0
        assert result.seed == 42
        assert result.population == "United_States_Massachusetts"

    def test_results_have_expected_compartments(self, real_simulation_builder_output):
        """Test that simulation results contain expected compartments."""
        result = run_simulation(real_simulation_builder_output)

        compartments = result.results.get_stacked_compartments()

        # Should have S, I, R compartments (with _total suffix for aggregated)
        assert "S_total" in compartments or "S" in compartments
        assert "I_total" in compartments or "I" in compartments
        assert "R_total" in compartments or "R" in compartments

    def test_results_have_correct_number_of_simulations(self, real_simulation_builder_output):
        """Test that the correct number of simulations are produced."""
        result = run_simulation(real_simulation_builder_output)

        assert result.results.Nsim == 3

    def test_results_have_no_nan_values(self, real_simulation_builder_output):
        """Test that simulation results contain no NaN values."""
        result = run_simulation(real_simulation_builder_output)

        compartments = result.results.get_stacked_compartments()

        for name, values in compartments.items():
            assert not np.any(np.isnan(values)), f"NaN values found in {name}"

    def test_sir_dynamics_are_correct(self, real_simulation_builder_output):
        """Test that SIR dynamics behave as expected: S decreases, R increases."""
        result = run_simulation(real_simulation_builder_output)

        compartments = result.results.get_stacked_compartments()

        # Get first simulation trajectory
        s_key = "S_total" if "S_total" in compartments else "S"
        r_key = "R_total" if "R_total" in compartments else "R"

        s_values = compartments[s_key][0]  # First simulation
        r_values = compartments[r_key][0]

        # S should decrease over time (epidemic spreads)
        assert s_values[0] > s_values[-1], "Susceptibles should decrease during epidemic"

        # R should increase over time (people recover)
        assert r_values[-1] > r_values[0], "Recovered should increase during epidemic"

    def test_seed_produces_reproducible_results(self, base_epimodel_config):
        """Test that the same seed produces reproducible results."""
        models1, _ = create_model_collection(base_epimodel_config, None)
        models2, _ = create_model_collection(base_epimodel_config, None)

        # Calculate initial conditions with numeric values
        n_age_groups = len(models1[0].population.Nk)
        i_init = np.zeros(n_age_groups)
        i_init[2] = 100
        r_init = np.zeros(n_age_groups)
        s_init = models1[0].population.Nk - i_init - r_init

        simulation_args = SimulationArguments(
            start_date=base_epimodel_config.timespan.start_date,
            end_date=date(2024, 2, 3),  # Use a Sunday for W-SAT resampling
            initial_conditions_dict={"S": s_init, "I": i_init, "R": r_init},
            Nsim=3,
            dt=1.0,
            resample_frequency="W-SAT",
        )

        builder_output1 = BuilderOutput(
            primary_id=0, seed=42, delta_t=1.0, model=models1[0], simulation=simulation_args
        )
        builder_output2 = BuilderOutput(
            primary_id=0, seed=42, delta_t=1.0, model=models2[0], simulation=simulation_args
        )

        result1 = run_simulation(builder_output1)
        result2 = run_simulation(builder_output2)

        # Results should be identical with same seed
        compartments1 = result1.results.get_stacked_compartments()
        compartments2 = result2.results.get_stacked_compartments()

        for key in compartments1:
            np.testing.assert_array_equal(
                compartments1[key], compartments2[key], err_msg=f"Results differ for {key} with same seed"
            )

    def test_different_seeds_produce_different_results(self, base_epimodel_config):
        """Test that different seeds produce different results."""
        models1, _ = create_model_collection(base_epimodel_config, None)
        models2, _ = create_model_collection(base_epimodel_config, None)

        # Calculate initial conditions with numeric values
        n_age_groups = len(models1[0].population.Nk)
        i_init = np.zeros(n_age_groups)
        i_init[2] = 100
        r_init = np.zeros(n_age_groups)
        s_init = models1[0].population.Nk - i_init - r_init

        simulation_args = SimulationArguments(
            start_date=base_epimodel_config.timespan.start_date,
            end_date=date(2024, 2, 3),  # Use a Sunday for W-SAT resampling
            initial_conditions_dict={"S": s_init, "I": i_init, "R": r_init},
            Nsim=3,
            dt=1.0,
            resample_frequency="W-SAT",
        )

        builder_output1 = BuilderOutput(
            primary_id=0, seed=42, delta_t=1.0, model=models1[0], simulation=simulation_args
        )
        builder_output2 = BuilderOutput(
            primary_id=0, seed=123, delta_t=1.0, model=models2[0], simulation=simulation_args
        )

        result1 = run_simulation(builder_output1)
        result2 = run_simulation(builder_output2)

        # Results should differ with different seeds
        compartments1 = result1.results.get_stacked_compartments()
        compartments2 = result2.results.get_stacked_compartments()

        # At least one compartment should have different values
        any_different = False
        for key in compartments1:
            if not np.array_equal(compartments1[key], compartments2[key]):
                any_different = True
                break

        assert any_different, "Different seeds should produce different results"


class TestDispatchRunnerIntegration:
    """Integration tests for dispatch_runner with real EpiModel objects."""

    def test_dispatch_runner_executes_simulation(self, real_simulation_builder_output):
        """Test that dispatch_runner correctly executes simulation with real model."""
        result = dispatch_runner(real_simulation_builder_output)

        assert isinstance(result, SimulationOutput)
        assert isinstance(result.results, SimulationResults)
        assert result.results.Nsim == 3


class TestRunCalibrationIntegration:
    """Integration tests for run_calibration with real ABCSampler.

    These tests use minimal particles and large epsilon for speed.
    They verify the runner executes calibration correctly, not that
    calibration converges or fits well.
    """

    def test_produces_valid_calibration_results(self, real_calibration_builder_output):
        """Test that run_calibration produces CalibrationResults with real ABCSampler."""
        result = run_calibration(real_calibration_builder_output)

        assert isinstance(result, CalibrationOutput)
        assert isinstance(result.results, CalibrationResults)
        assert result.results is not None

        # Verify posterior samples exist (don't check values)
        posterior = result.results.get_posterior_distribution()
        assert len(posterior) > 0

    def test_calibration_output_has_correct_metadata(self, real_calibration_builder_output):
        """Test that CalibrationOutput metadata is populated correctly."""
        result = run_calibration(real_calibration_builder_output)

        assert result.primary_id == real_calibration_builder_output.primary_id
        assert result.seed == real_calibration_builder_output.seed
        assert result.population == "United_States_Massachusetts"
