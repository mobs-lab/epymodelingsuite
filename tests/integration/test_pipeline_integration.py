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
        s_total = compartments["S_total"]
        i_total = compartments["I_total"]
        r_total = compartments["R_total"]

        total_pop_per_timestep = s_total + i_total + r_total

        # Get expected total population from model
        expected_total_pop = np.sum(builder_output.model.population.Nk)

        # Population should be exactly conserved at every timestep for every simulation
        for sim_idx in range(total_pop_per_timestep.shape[0]):
            for t_idx in range(total_pop_per_timestep.shape[1]):
                actual_pop = total_pop_per_timestep[sim_idx, t_idx]
                np.testing.assert_allclose(
                    actual_pop,
                    expected_total_pop,
                    rtol=1e-10,
                    err_msg=f"Population not conserved at sim={sim_idx}, t={t_idx}: "
                    f"expected={expected_total_pop}, got={actual_pop}",
                )

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

    @pytest.mark.dynamics
    def test_sir_mean_approximates_ode(self):
        """Verify stochastic SIR mean approximates deterministic ODE solution.

        Uses a large population (100k) and many simulations (50) to verify that
        the stochastic model's mean behavior matches the deterministic ODE solution
        within 10% relative error for the final epidemic size.

        This test validates that the epydemix simulation engine correctly implements
        the underlying epidemic dynamics.
        """
        from scipy.integrate import solve_ivp

        from epydemix.model import EpiModel

        # Parameters for a simple SIR model
        N = 100_000  # Large population to reduce stochastic noise
        I0 = 100  # Initial infections
        beta = 0.3  # Transmission rate
        gamma = 0.1  # Recovery rate
        t_span = 60  # 60 days

        # Define ODE system for SIR
        def sir_ode(t, y, beta, gamma, N):  # noqa: ARG001
            S, I, _ = y
            dS = -beta * S * I / N
            dI = beta * S * I / N - gamma * I
            dR = gamma * I
            return [dS, dI, dR]

        # Solve ODE
        y0 = [N - I0, I0, 0]
        sol = solve_ivp(
            sir_ode,
            [0, t_span],
            y0,
            args=(beta, gamma, N),
            t_eval=np.arange(0, t_span + 1, 1),
            method="RK45",
        )
        ode_final_s = sol.y[0][-1]

        # Create stochastic model with single age group
        model = EpiModel()

        # Use a simple single-age-group setup
        from epydemix.population import Population

        pop = Population("Test_Population")
        pop.add_population(Nk=[float(N)], Nk_names=["all"])
        pop.add_contact_matrix(np.array([[1.0]]), layer_name="all")
        model.set_population(pop)

        model.add_compartments(["S", "I", "R"])
        model.add_transition("S", "I", params=("beta", "I"), kind="mediated")
        model.add_transition("I", "R", params="gamma", kind="spontaneous")
        model.add_parameter(parameters_dict={"beta": beta, "gamma": gamma})

        # Initial conditions
        init_conditions = {
            "S": np.array([N - I0]),
            "I": np.array([I0]),
            "R": np.array([0]),
        }

        # Run stochastic simulations
        rng = np.random.default_rng(42)
        results = model.run_simulations(
            start_date="2025-01-01",
            end_date="2025-03-01",
            initial_conditions_dict=init_conditions,
            Nsim=50,
            dt=1.0,
            rng=rng,
        )

        # Get final S from stochastic model
        compartments = results.get_stacked_compartments()
        s_final_stochastic = compartments["S_total"][:, -1]  # Shape: (Nsim,)
        mean_final_s_stochastic = np.mean(s_final_stochastic)

        # Compare with ODE solution
        relative_error = abs(mean_final_s_stochastic - ode_final_s) / ode_final_s

        assert relative_error < 0.10, (
            f"Stochastic mean final S ({mean_final_s_stochastic:.0f}) differs from "
            f"ODE solution ({ode_final_s:.0f}) by {relative_error * 100:.1f}% (> 10%)"
        )


@pytest.mark.slow
class TestCalibrationPipelineE2E:
    """End-to-end tests for the calibration pipeline."""

    @pytest.fixture(scope="class")
    def synthetic_observed_data(self, tmp_path_factory):
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

        # Save to temp file (use tmp_path_factory for class-scoped fixture)
        tmp_path = tmp_path_factory.mktemp("data")
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

    def test_start_date_sampling(self, synthetic_observed_data, tmp_path):
        """Verify start_date sampling works in calibration.

        Tests that when start_date is configured with reference_date and prior:
        - Calibration executes successfully
        - start_date appears in posterior distribution
        - Sampled start_date values are within prior bounds
        - CalibrationOutput has start_date_reference set
        """
        import yaml

        # Load basemodel config with calibrated start_date
        basemodel_config = load_basemodel_config_from_file(str(FIXTURES_DIR / "minimal_basemodel_start_date.yaml"))

        # Load calibration config and set observed_data_path
        with open(FIXTURES_DIR / "minimal_modelset_start_date.yaml") as f:
            calibration_raw = yaml.safe_load(f)

        calibration_raw["modelset"]["calibration"]["observed_data_path"] = synthetic_observed_data

        modified_config_path = tmp_path / "calibration_start_date.yaml"
        with open(modified_config_path, "w") as f:
            yaml.dump(calibration_raw, f)

        calibration_config = load_calibration_config_from_file(str(modified_config_path))

        # Verify configs loaded correctly
        # calibration.start_date defines epidemic start sampling
        assert calibration_config.modelset.calibration.start_date is not None
        assert calibration_config.modelset.calibration.start_date.reference_date is not None
        assert calibration_config.modelset.calibration.start_date.prior is not None

        # Build and run calibration
        builder_outputs = dispatch_builder(
            basemodel_config=basemodel_config,
            calibration_config=calibration_config,
        )

        builder_output = builder_outputs[0]

        # Verify start_date is in priors
        assert "start_date" in builder_output.calibrator.priors

        result = dispatch_runner(builder_output)

        # Verify calibration completed
        assert isinstance(result, CalibrationOutput)
        assert result.results is not None

        # Verify start_date_reference is set in output
        assert result.start_date_reference is not None

        # Verify start_date appears in posterior distribution
        posterior_df = result.results.get_posterior_distribution()
        assert len(posterior_df) > 0, "No posterior samples produced"
        assert "start_date" in posterior_df.columns, "start_date not in posteriors"

        # Verify sampled start_date values are within prior bounds [0, 14)
        start_date_values = posterior_df["start_date"].values
        assert np.all(start_date_values >= 0), "start_date below prior minimum"
        assert np.all(start_date_values < 14), "start_date above prior maximum"


class TestSimulationPipelineSubdailyTimesteps:
    """Integration tests for subdaily timestep (delta_t=0.5) pipeline."""

    def test_pipeline_with_half_day_timestep(self):
        """Full pipeline with delta_t=0.5: YAML -> build -> run -> valid output."""
        basemodel_config = load_basemodel_config_from_file(str(FIXTURES_DIR / "minimal_basemodel_half_day.yaml"))

        builder_output = dispatch_builder(basemodel_config=basemodel_config)

        assert builder_output is not None
        assert builder_output.model is not None
        assert builder_output.simulation is not None
        assert builder_output.delta_t == 0.5
        assert builder_output.simulation.dt == 0.5

        result = dispatch_runner(builder_output)

        assert isinstance(result, SimulationOutput)
        assert result.results is not None
        assert result.delta_t == 0.5

        compartments = result.results.get_stacked_compartments()
        assert len(compartments) > 0

        # Verify no NaN or inf values
        for name, values in compartments.items():
            assert not np.any(np.isnan(values)), f"NaN values found in {name}"
            assert not np.any(np.isinf(values)), f"Inf values found in {name}"

        # Verify population conservation
        s_total = compartments["S_total"]
        i_total = compartments["I_total"]
        r_total = compartments["R_total"]
        total_pop = s_total + i_total + r_total
        expected_total_pop = np.sum(builder_output.model.population.Nk)

        # Subdaily timesteps may introduce small numerical drift in stochastic simulation
        np.testing.assert_allclose(
            total_pop,
            expected_total_pop,
            rtol=1e-5,
            err_msg="Population not conserved with subdaily timesteps",
        )

    def test_sir_dynamics_with_subdaily_timesteps(self):
        """Verify SIR dynamics with dt=0.5: S decreases, R increases, I peaks in middle."""
        basemodel_config = load_basemodel_config_from_file(str(FIXTURES_DIR / "minimal_basemodel_half_day.yaml"))
        builder_output = dispatch_builder(basemodel_config=basemodel_config)
        result = dispatch_runner(builder_output)

        compartments = result.results.get_stacked_compartments()
        s_total = compartments["S_total"][0]
        i_total = compartments["I_total"][0]
        r_total = compartments["R_total"][0]

        # S should decrease overall
        assert s_total[0] > s_total[-1], "Susceptibles should decrease over epidemic"

        # R should increase overall
        assert r_total[-1] > r_total[0], "Recovered should increase over epidemic"

        # I should peak somewhere in the middle
        i_max_idx = np.argmax(i_total)
        assert 0 < i_max_idx < len(i_total) - 1, "Infectious should peak in middle of simulation"

    def test_subdaily_with_weekly_resampling_produces_weekly_output(self):
        """dt=0.5 + resample_frequency=W-SAT -> output dates are weekly Saturdays."""
        basemodel_config = load_basemodel_config_from_file(str(FIXTURES_DIR / "minimal_basemodel_half_day.yaml"))
        builder_output = dispatch_builder(basemodel_config=basemodel_config)
        result = dispatch_runner(builder_output)

        dates = result.results.dates
        assert len(dates) > 1

        # Verify dates are weekly (7 days apart) and are Saturdays (weekday=5)
        for i in range(len(dates) - 1):
            d1 = pd.Timestamp(dates[i])
            d2 = pd.Timestamp(dates[i + 1])
            assert (d2 - d1).days == 7, f"Dates not 7 days apart: {d1} -> {d2}"
            assert d1.weekday() == 5, f"Date {d1} is not a Saturday"

    @pytest.mark.nightly
    def test_resampled_output_similar_across_dt(self):
        """Compare dt=1.0 vs dt=0.5 resampled to W-SAT: mean final S within 10%."""
        import yaml

        from epymodelingsuite.schema.basemodel import validate_basemodel

        # Build dt=1.0 config with same date range as half-day (Jan 1 - Feb 1)
        with open(FIXTURES_DIR / "minimal_basemodel.yaml") as f:
            raw_dt10 = yaml.safe_load(f)
        raw_dt10["model"]["timespan"]["end_date"] = "2025-02-01"
        raw_dt10["model"]["simulation"]["n_sims"] = 50
        raw_dt10["model"]["random_seed"] = 99
        basemodel_config_dt10 = validate_basemodel(raw_dt10)

        # Build dt=0.5 config with same sims and seed
        with open(FIXTURES_DIR / "minimal_basemodel_half_day.yaml") as f:
            raw_dt05 = yaml.safe_load(f)
        raw_dt05["model"]["simulation"]["n_sims"] = 50
        raw_dt05["model"]["random_seed"] = 99
        basemodel_config_dt05 = validate_basemodel(raw_dt05)

        # Build and run dt=1.0
        builder_dt10 = dispatch_builder(basemodel_config=basemodel_config_dt10)
        result_dt10 = dispatch_runner(builder_dt10)

        # Build and run dt=0.5
        builder_dt05 = dispatch_builder(basemodel_config=basemodel_config_dt05)
        result_dt05 = dispatch_runner(builder_dt05)

        # Compare mean final S (resampled to weekly)
        s_dt10 = result_dt10.results.get_stacked_compartments()["S_total"]
        s_dt05 = result_dt05.results.get_stacked_compartments()["S_total"]

        mean_final_s_dt10 = np.mean(s_dt10[:, -1])
        mean_final_s_dt05 = np.mean(s_dt05[:, -1])

        # Should be within 10% tolerance
        np.testing.assert_allclose(
            mean_final_s_dt05,
            mean_final_s_dt10,
            rtol=0.10,
            err_msg=f"Mean final S differs: dt=1.0 ({mean_final_s_dt10:.0f}) vs dt=0.5 ({mean_final_s_dt05:.0f})",
        )
