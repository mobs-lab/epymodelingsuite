"""End-to-end integration tests for the full pipeline.

Tests the complete workflow: YAML config -> dispatch_builder -> dispatch_runner -> output.
These tests use minimal configs to verify the pipeline works correctly without mocks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from epymodelingsuite.config_loader import (
    load_basemodel_config_from_file,
    load_calibration_config_from_file,
)
from epymodelingsuite.dispatcher.builder import dispatch_builder
from epymodelingsuite.dispatcher.runner import dispatch_runner
from epymodelingsuite.schema.dispatcher import CalibrationOutput, SimulationOutput

# Path to test fixtures
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


class TestSimulationPipelineE2E:
    """End-to-end tests for the simulation pipeline."""

    def test_yaml_config_to_working_simulation(self):
        """Full pipeline: YAML -> EpiModel -> simulation results.

        Verifies:
        - Config loads successfully from YAML
        - dispatch_builder creates valid BuilderOutput
        - dispatch_runner executes simulation
        - SimulationOutput has valid trajectories
        - No NaN/inf values in results
        - Population conservation (total population remains constant)
        """
        # Load config from YAML fixture
        basemodel_config = load_basemodel_config_from_file(str(FIXTURES_DIR / "minimal_basemodel.yaml"))

        # Build model
        builder_output = dispatch_builder(basemodel_config=basemodel_config)

        # Verify builder output
        assert builder_output is not None
        assert builder_output.model is not None
        assert builder_output.simulation is not None
        assert builder_output.seed == 42

        # Run simulation
        result = dispatch_runner(builder_output)

        # Verify result type
        assert isinstance(result, SimulationOutput)
        assert result.results is not None
        assert result.population == "United_States_Massachusetts"

        # Verify trajectories exist and have valid structure
        sim_results = result.results
        assert sim_results.Nsim == 5

        # Get stacked compartments (dict of compartment_name -> array[n_sims, n_timesteps])
        compartments = sim_results.get_stacked_compartments()
        assert len(compartments) > 0

        # Verify no NaN or inf values
        for name, values in compartments.items():
            assert not np.any(np.isnan(values)), f"NaN values found in {name}"
            assert not np.any(np.isinf(values)), f"Inf values found in {name}"

        # Verify population conservation
        # Sum S + I + R totals across all age groups
        s_total = compartments.get("S_total")
        i_total = compartments.get("I_total")
        r_total = compartments.get("R_total")

        if s_total is not None and i_total is not None and r_total is not None:
            total_pop = s_total + i_total + r_total
            # Population should be approximately constant across time (axis=1)
            for sim_idx in range(total_pop.shape[0]):
                pop_std = np.std(total_pop[sim_idx])
                assert pop_std < 1.0, f"Population not conserved in sim {sim_idx}, std: {pop_std}"

    def test_simulation_produces_expected_compartment_dynamics(self):
        """Verify SIR dynamics: S decreases, I peaks then decreases, R increases."""
        basemodel_config = load_basemodel_config_from_file(str(FIXTURES_DIR / "minimal_basemodel.yaml"))

        builder_output = dispatch_builder(basemodel_config=basemodel_config)
        result = dispatch_runner(builder_output)

        compartments = result.results.get_stacked_compartments()

        # Get first simulation's trajectory (index 0)
        s_total = compartments["S_total"][0]  # shape: (n_timesteps,)
        i_total = compartments["I_total"][0]
        r_total = compartments["R_total"][0]

        # S should decrease overall (first > last)
        assert s_total[0] > s_total[-1], "Susceptibles should decrease over epidemic"

        # R should increase overall (last > first)
        assert r_total[-1] > r_total[0], "Recovered should increase over epidemic"

        # I should peak somewhere in the middle (not at start or end)
        i_max_idx = np.argmax(i_total)
        assert 0 < i_max_idx < len(i_total) - 1, "Infectious should peak in middle of simulation"


class TestCalibrationPipelineE2E:
    """End-to-end tests for the calibration pipeline."""

    @pytest.fixture
    def synthetic_observed_data(self, tmp_path):
        """Generate synthetic observed data by running actual simulation.

        Uses known parameter values (beta=0.25, I_init=100) to generate
        "observed" data that the calibration should be able to recover.
        """
        from epydemix.model import EpiModel
        from epydemix.population import load_epydemix_population

        # Create a simple SIR model with known parameters
        model = EpiModel()

        # Load population from epydemix
        age_groups = ["0-4", "5-17", "18-49", "50-64", "65+"]
        age_group_mapping = {
            "0-4": [str(i) for i in range(5)],
            "5-17": [str(i) for i in range(5, 18)],
            "18-49": [str(i) for i in range(18, 50)],
            "50-64": [str(i) for i in range(50, 65)],
            "65+": [str(i) for i in range(65, 84)] + ["84+"],
        }
        population = load_epydemix_population(
            population_name="United_States_Massachusetts",
            age_group_mapping=age_group_mapping,
        )
        model.set_population(population)

        model.add_compartments(["S", "I", "R"])
        model.add_transition("S", "I", kind="mediated", params=("beta", "I"))
        model.add_transition("I", "R", kind="spontaneous", params="mu")
        model.add_parameter("beta", 0.25)  # True value within prior [0.1, 0.5]
        model.add_parameter("mu", 0.1)

        # Initial conditions: 100 infected (true value within prior [50, 200))
        total_pop = np.sum(model.population.Nk)
        n_age = len(model.population.Nk)
        i_init = np.zeros(n_age)
        i_init[2] = 100  # Put initial infections in 18-49 age group
        s_init = model.population.Nk - i_init

        init_conditions = {
            "S": s_init,
            "I": i_init,
            "R": np.zeros(n_age),
        }

        # Run simulation
        rng = np.random.default_rng(123)
        sim_results = model.run_simulations(
            start_date="2025-01-01",
            end_date="2025-03-01",
            initial_conditions_dict=init_conditions,
            Nsim=1,
            dt=1.0,
            resample_frequency="W-SAT",
            rng=rng,
        )

        # Extract I->R transitions as "observed" data
        transitions = sim_results.get_stacked_transitions()
        i_to_r = transitions["I_to_R_total"][0]  # First (only) simulation

        # Create DataFrame with dates
        dates = sim_results.dates

        df = pd.DataFrame(
            {
                "date": dates,
                "value": i_to_r.astype(int),
                "location": "US-MA",
            }
        )

        # Save to temp file
        data_path = tmp_path / "synthetic_observed.csv"
        df.to_csv(data_path, index=False)

        return str(data_path)

    def test_calibration_end_to_end(self, synthetic_observed_data, tmp_path):
        """Full calibration: config -> ABCSampler -> posteriors.

        Verifies:
        - Configs load successfully
        - dispatch_builder creates ABCSampler
        - Calibration runs and produces posterior samples
        - Acceptance rate > 0 (at least some particles accepted)
        """
        import yaml

        # Load basemodel config
        basemodel_config = load_basemodel_config_from_file(str(FIXTURES_DIR / "minimal_basemodel_calibration.yaml"))

        # Load and modify calibration config to use synthetic data
        with open(FIXTURES_DIR / "minimal_modelset_calibration.yaml") as f:
            calibration_raw = yaml.safe_load(f)

        # Set the observed data path
        calibration_raw["modelset"]["calibration"]["observed_data_path"] = synthetic_observed_data

        # Write modified config to temp file
        modified_config_path = tmp_path / "calibration_config.yaml"
        with open(modified_config_path, "w") as f:
            yaml.dump(calibration_raw, f)

        calibration_config = load_calibration_config_from_file(str(modified_config_path))

        # Build calibrator
        builder_outputs = dispatch_builder(
            basemodel_config=basemodel_config,
            calibration_config=calibration_config,
        )

        # Should return list for calibration workflow
        assert isinstance(builder_outputs, list)
        assert len(builder_outputs) == 1  # One population

        builder_output = builder_outputs[0]

        # Verify builder output has calibrator
        assert builder_output.calibrator is not None
        assert builder_output.calibration is not None
        assert builder_output.model is not None

        # Run calibration
        result = dispatch_runner(builder_output)

        # Verify result type
        assert isinstance(result, CalibrationOutput)
        assert result.results is not None
        assert result.population == "United_States_Massachusetts"

        # Verify posterior samples
        posterior_df = result.results.get_posterior_distribution()
        assert len(posterior_df) > 0, "No posterior samples produced"

        # Verify calibrated parameters are in results
        assert "beta" in posterior_df.columns, "beta parameter not in posteriors"
        assert "I" in posterior_df.columns, "I compartment not in posteriors"

    def test_calibration_respects_priors(self, synthetic_observed_data, tmp_path):
        """Verify that posterior samples respect prior bounds."""
        import yaml

        basemodel_config = load_basemodel_config_from_file(str(FIXTURES_DIR / "minimal_basemodel_calibration.yaml"))

        with open(FIXTURES_DIR / "minimal_modelset_calibration.yaml") as f:
            calibration_raw = yaml.safe_load(f)

        calibration_raw["modelset"]["calibration"]["observed_data_path"] = synthetic_observed_data

        modified_config_path = tmp_path / "calibration_config.yaml"
        with open(modified_config_path, "w") as f:
            yaml.dump(calibration_raw, f)

        calibration_config = load_calibration_config_from_file(str(modified_config_path))

        builder_outputs = dispatch_builder(
            basemodel_config=basemodel_config,
            calibration_config=calibration_config,
        )

        result = dispatch_runner(builder_outputs[0])

        # Get posterior samples
        posterior_df = result.results.get_posterior_distribution()

        # beta prior: uniform(0.1, 0.4) -> range [0.1, 0.5]
        beta_values = posterior_df["beta"].values
        assert np.all(beta_values >= 0.1), "beta below prior minimum"
        assert np.all(beta_values <= 0.5), "beta above prior maximum"

        # I prior: randint(50, 200) -> range [50, 200)
        i_values = posterior_df["I"].values
        assert np.all(i_values >= 50), "I below prior minimum"
        assert np.all(i_values < 200), "I above prior maximum"

    @pytest.mark.parametrize(
        "strategy_name,strategy_options",
        [
            ("rejection", {"num_particles": 30, "epsilon": 100000}),
            ("smc", {"num_particles": 20, "num_generations": 2}),
            ("top_fraction", {"top_fraction": 0.5, "Nsim": 30}),
        ],
        ids=["rejection", "smc", "top_fraction"],
    )
    def test_calibration_strategies(self, synthetic_observed_data, tmp_path, strategy_name, strategy_options):
        """Verify each calibration strategy runs and produces posteriors.

        Tests that rejection, smc, and top_fraction strategies all:
        - Execute without errors
        - Produce posterior samples
        - Record the correct strategy in results
        """
        import yaml

        basemodel_config = load_basemodel_config_from_file(str(FIXTURES_DIR / "minimal_basemodel_calibration.yaml"))

        with open(FIXTURES_DIR / "minimal_modelset_calibration.yaml") as f:
            calibration_raw = yaml.safe_load(f)

        # Set strategy and options
        calibration_raw["modelset"]["calibration"]["strategy"] = {
            "name": strategy_name,
            "options": strategy_options,
        }
        calibration_raw["modelset"]["calibration"]["observed_data_path"] = synthetic_observed_data

        modified_config_path = tmp_path / "calibration_config.yaml"
        with open(modified_config_path, "w") as f:
            yaml.dump(calibration_raw, f)

        calibration_config = load_calibration_config_from_file(str(modified_config_path))

        # Verify the strategy was set correctly in config
        assert calibration_config.modelset.calibration.strategy.name == strategy_name

        builder_outputs = dispatch_builder(
            basemodel_config=basemodel_config,
            calibration_config=calibration_config,
        )

        result = dispatch_runner(builder_outputs[0])

        # Verify calibration completed
        assert isinstance(result, CalibrationOutput)
        assert result.results is not None

        # Verify posterior samples were produced
        posterior_df = result.results.get_posterior_distribution()
        assert len(posterior_df) > 0, f"No posterior samples for {strategy_name}"

        # Verify the strategy was recorded in results
        assert result.results.calibration_strategy == strategy_name

    @pytest.mark.parametrize(
        "distance_function",
        ["rmse", "wrmse", "wmape", "mae", "mape"],
        ids=["rmse", "wrmse", "wmape", "mae", "mape"],
    )
    def test_distance_functions(self, synthetic_observed_data, tmp_path, distance_function):
        """Verify each distance function is used correctly in calibration.

        Tests that rmse, wrmse, wmape, mae, mape distance functions all:
        - Are accepted in config
        - Execute calibration without errors
        - Produce posterior samples
        """
        import yaml

        basemodel_config = load_basemodel_config_from_file(str(FIXTURES_DIR / "minimal_basemodel_calibration.yaml"))

        with open(FIXTURES_DIR / "minimal_modelset_calibration.yaml") as f:
            calibration_raw = yaml.safe_load(f)

        # Set distance function
        calibration_raw["modelset"]["calibration"]["distance_function"] = distance_function
        calibration_raw["modelset"]["calibration"]["observed_data_path"] = synthetic_observed_data

        modified_config_path = tmp_path / "calibration_config.yaml"
        with open(modified_config_path, "w") as f:
            yaml.dump(calibration_raw, f)

        calibration_config = load_calibration_config_from_file(str(modified_config_path))

        # Verify the distance function was set correctly in config
        assert calibration_config.modelset.calibration.distance_function == distance_function

        builder_outputs = dispatch_builder(
            basemodel_config=basemodel_config,
            calibration_config=calibration_config,
        )

        # Verify the ABCSampler was built with a distance function
        assert builder_outputs[0].calibrator is not None
        assert builder_outputs[0].calibrator.distance_function is not None

        result = dispatch_runner(builder_outputs[0])

        # Verify calibration completed with posteriors
        assert isinstance(result, CalibrationOutput)
        assert result.results is not None

        posterior_df = result.results.get_posterior_distribution()
        assert len(posterior_df) > 0, f"No posterior samples with {distance_function}"
