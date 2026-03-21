"""Tests for epymodelingsuite.dispatcher.builder module."""

from __future__ import annotations

from datetime import date
from unittest.mock import Mock, patch

import numpy as np
import pytest
from epydemix.model import EpiModel

from epymodelingsuite.dispatcher.builder import (
    build_basemodel,
    build_calibration,
    build_sampling,
    count_nans_at_start,
    dispatch_builder,
    dist_func_date_alignment_wrapper,
    dist_func_dict,
)
from epymodelingsuite.schema.basemodel import (
    BaseEpiModel,
    BasemodelConfig,
    Compartment,
    Parameter,
    Population,
    Simulation,
    Timespan,
    Transition,
)
from epymodelingsuite.schema.dispatcher import BuilderOutput, SimulationArguments
from tests.conftest import create_builder_output, make_sir_config


class TestCountNansAtStart:
    """Tests for count_nans_at_start helper function."""

    def test_empty_array_returns_zero(self):
        """Test that empty array returns 0."""
        arr = np.array([])
        assert count_nans_at_start(arr) == 0

    def test_no_nans_returns_zero(self):
        """Test that array with no NaNs returns 0."""
        arr = np.array([1, 2, 3, 4, 5])
        assert count_nans_at_start(arr) == 0

    def test_all_nans_returns_length(self):
        """Test that array with all NaNs returns length."""
        arr = np.array([np.nan, np.nan, np.nan])
        assert count_nans_at_start(arr) == 3

    def test_nans_at_start_counted(self):
        """Test that NaNs at start are counted correctly."""
        arr = np.array([np.nan, np.nan, 1, 2, 3])
        assert count_nans_at_start(arr) == 2

    def test_nans_in_middle_not_counted(self):
        """Test that NaNs after first non-NaN are not counted."""
        arr = np.array([1, np.nan, 2, np.nan, 3])
        assert count_nans_at_start(arr) == 0

    def test_single_nan_at_start(self):
        """Test single NaN at start."""
        arr = np.array([np.nan, 1, 2])
        assert count_nans_at_start(arr) == 1


class TestDistFuncDateAlignmentWrapper:
    """Tests for dist_func_date_alignment_wrapper function."""

    def test_truncates_prepended_nans(self):
        """Test that wrapper truncates prepended NaNs before calling distance function."""
        # Create a mock distance function that returns sum of differences
        mock_dist_func = Mock(return_value=10.0)

        wrapped = dist_func_date_alignment_wrapper(mock_dist_func)

        data = {"data": np.array([1, 2, 3, 4, 5])}
        simulation = {"data": np.array([np.nan, np.nan, 3, 4, 5])}

        result = wrapped(data, simulation)

        # Should have called with truncated arrays
        mock_dist_func.assert_called_once()
        call_data = mock_dist_func.call_args[0][0]
        call_sim = mock_dist_func.call_args[0][1]

        # Should have truncated first 2 elements (NaN positions)
        np.testing.assert_array_equal(call_data["data"], np.array([3, 4, 5]))
        np.testing.assert_array_equal(call_sim["data"], np.array([3, 4, 5]))
        assert result == 10.0

    def test_no_nans_passes_through_unchanged(self):
        """Test that arrays without NaNs pass through unchanged."""
        mock_dist_func = Mock(return_value=5.0)
        wrapped = dist_func_date_alignment_wrapper(mock_dist_func)

        data = {"data": np.array([1, 2, 3])}
        simulation = {"data": np.array([1, 2, 3])}

        wrapped(data, simulation)

        call_data = mock_dist_func.call_args[0][0]
        call_sim = mock_dist_func.call_args[0][1]
        np.testing.assert_array_equal(call_data["data"], np.array([1, 2, 3]))
        np.testing.assert_array_equal(call_sim["data"], np.array([1, 2, 3]))


class TestDistFuncDict:
    """Tests for dist_func_dict containing available distance functions."""

    def test_contains_expected_functions(self):
        """Test that dist_func_dict contains all expected distance functions."""
        expected_funcs = ["rmse", "wmape", "ae", "mae", "mape", "wrmse"]
        for func_name in expected_funcs:
            assert func_name in dist_func_dict, f"Missing distance function: {func_name}"

    def test_all_functions_are_callable(self):
        """Test that all functions in dict are callable."""
        for name, func in dist_func_dict.items():
            assert callable(func), f"Distance function '{name}' is not callable"


class TestBuildBasemodel:
    """Tests for build_basemodel function."""

    @pytest.fixture
    def minimal_basemodel_config(self):
        """Create a minimal BasemodelConfig for testing."""
        return BasemodelConfig(model=make_sir_config(end_date=date(2024, 3, 31), n_sims=5, random_seed=42))

    def test_returns_builder_output(self, minimal_basemodel_config):
        """Test that build_basemodel returns a BuilderOutput object."""
        result = build_basemodel(basemodel_config=minimal_basemodel_config)

        assert isinstance(result, BuilderOutput)

    def test_builder_output_has_model(self, minimal_basemodel_config):
        """Test that BuilderOutput contains an EpiModel."""
        result = build_basemodel(basemodel_config=minimal_basemodel_config)

        assert result.model is not None
        assert isinstance(result.model, EpiModel)

    def test_builder_output_has_simulation_args(self, minimal_basemodel_config):
        """Test that BuilderOutput contains SimulationArguments."""
        result = build_basemodel(basemodel_config=minimal_basemodel_config)

        assert result.simulation is not None
        assert isinstance(result.simulation, SimulationArguments)

    def test_builder_output_has_no_calibrator(self, minimal_basemodel_config):
        """Test that BuilderOutput has no calibrator for simulation workflow."""
        result = build_basemodel(basemodel_config=minimal_basemodel_config)

        assert result.calibrator is None

    def test_simulation_args_match_config(self, minimal_basemodel_config):
        """Test that SimulationArguments match the configuration."""
        result = build_basemodel(basemodel_config=minimal_basemodel_config)

        assert result.simulation.start_date == date(2024, 1, 1)
        assert result.simulation.end_date == date(2024, 3, 31)
        assert result.simulation.Nsim == 5
        assert result.simulation.dt == 1.0
        assert result.simulation.resample_frequency == "W-SAT"

    def test_model_has_compartments(self, minimal_basemodel_config):
        """Test that EpiModel has compartments from config."""
        result = build_basemodel(basemodel_config=minimal_basemodel_config)

        assert "S" in result.model.compartments
        assert "I" in result.model.compartments
        assert "R" in result.model.compartments

    def test_model_has_parameters(self, minimal_basemodel_config):
        """Test that EpiModel has parameters from config."""
        result = build_basemodel(basemodel_config=minimal_basemodel_config)

        assert "beta" in result.model.parameters
        assert "gamma" in result.model.parameters
        assert result.model.parameters["beta"] == 0.5
        assert result.model.parameters["gamma"] == 0.1

    def test_model_has_correct_population(self, minimal_basemodel_config):
        """Test that EpiModel has correct population."""
        result = build_basemodel(basemodel_config=minimal_basemodel_config)

        # Population name is converted to epydemix format
        assert result.model.population.name == "United_States_California"

    def test_seed_is_preserved(self, minimal_basemodel_config):
        """Test that random seed is preserved in BuilderOutput."""
        result = build_basemodel(basemodel_config=minimal_basemodel_config)

        assert result.seed == 42

    def test_delta_t_is_preserved(self, minimal_basemodel_config):
        """Test that delta_t is preserved in BuilderOutput."""
        result = build_basemodel(basemodel_config=minimal_basemodel_config)

        assert result.delta_t == 1.0

    def test_primary_id_is_zero(self, minimal_basemodel_config):
        """Test that primary_id is 0 for single model build."""
        result = build_basemodel(basemodel_config=minimal_basemodel_config)

        assert result.primary_id == 0

    @pytest.mark.parametrize("delta_t", [0.5, 0.25, 2.0])
    def test_delta_t_is_preserved_across_values(self, delta_t):
        """Test that various delta_t values are preserved through build_basemodel."""
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
        timespan = Timespan(start_date=date(2024, 1, 1), end_date=date(2024, 3, 31), delta_t=delta_t)
        simulation = Simulation(n_sims=5)

        basemodel = BaseEpiModel(
            name="test_model",
            compartments=compartments,
            transitions=transitions,
            parameters=parameters,
            population=population,
            timespan=timespan,
            simulation=simulation,
            random_seed=42,
        )
        config = BasemodelConfig(model=basemodel)

        result = build_basemodel(basemodel_config=config)

        assert result.delta_t == delta_t
        assert result.simulation.dt == delta_t


class TestDispatchBuilder:
    """Tests for dispatch_builder function."""

    @pytest.fixture
    def minimal_basemodel_config(self):
        """Create a minimal BasemodelConfig for testing."""
        return BasemodelConfig(model=make_sir_config(end_date=date(2024, 3, 31), n_sims=5, random_seed=42))

    def test_dispatch_with_basemodel_only_returns_builder_output(self, minimal_basemodel_config):
        """Test that dispatch_builder with only basemodel returns BuilderOutput."""
        result = dispatch_builder(basemodel_config=minimal_basemodel_config)

        assert isinstance(result, BuilderOutput)

    def test_dispatch_with_basemodel_only_has_model(self, minimal_basemodel_config):
        """Test that result has EpiModel for simulation workflow."""
        result = dispatch_builder(basemodel_config=minimal_basemodel_config)

        assert result.model is not None
        assert isinstance(result.model, EpiModel)

    def test_dispatch_with_basemodel_only_has_simulation_args(self, minimal_basemodel_config):
        """Test that result has SimulationArguments for simulation workflow."""
        result = dispatch_builder(basemodel_config=minimal_basemodel_config)

        assert result.simulation is not None
        assert isinstance(result.simulation, SimulationArguments)

    def test_dispatch_with_basemodel_only_no_calibrator(self, minimal_basemodel_config):
        """Test that result has no calibrator for simulation workflow."""
        result = dispatch_builder(basemodel_config=minimal_basemodel_config)

        assert result.calibrator is None
        assert result.calibration is None
        assert result.projection is None

    def test_dispatch_routes_to_correct_builder(self, minimal_basemodel_config):
        """Test that dispatch_builder routes to build_basemodel for basemodel-only config."""
        with patch("epymodelingsuite.dispatcher.builder.BUILDER_REGISTRY") as mock_registry:
            mock_builder = Mock(return_value=create_builder_output(primary_id=0, model=Mock()))
            mock_registry.__getitem__ = Mock(return_value=mock_builder)

            dispatch_builder(basemodel_config=minimal_basemodel_config)

            # Should have looked up the builder for basemodel_config only
            mock_registry.__getitem__.assert_called_once()
            called_key = mock_registry.__getitem__.call_args[0][0]
            assert "basemodel_config" in called_key


class TestBuildSampling:
    """Tests for build_sampling function."""

    @pytest.fixture
    def minimal_basemodel_config(self):
        """Create a minimal BasemodelConfig for testing."""
        return BasemodelConfig(model=make_sir_config(end_date=date(2024, 3, 31), n_sims=5, random_seed=42))

    @pytest.fixture
    def minimal_sampling_config(self):
        """Create a minimal SamplingConfig for testing with multiple populations."""
        from epymodelingsuite.schema.sampling import SamplingConfig, SamplingModelset

        modelset = SamplingModelset(population_names=["US-CA", "US-TX"], sampling=None)

        return SamplingConfig(modelset=modelset)

    def test_returns_list_of_builder_outputs(self, minimal_basemodel_config, minimal_sampling_config):
        """Test that build_sampling returns list of BuilderOutput objects."""
        result = build_sampling(basemodel_config=minimal_basemodel_config, sampling_config=minimal_sampling_config)

        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(item, BuilderOutput) for item in result)

    def test_creates_model_per_population(self, minimal_basemodel_config, minimal_sampling_config):
        """Test that build_sampling creates one model per population."""
        result = build_sampling(basemodel_config=minimal_basemodel_config, sampling_config=minimal_sampling_config)

        # Should have 2 models (US-CA and US-TX)
        assert len(result) == 2

    def test_each_output_has_model(self, minimal_basemodel_config, minimal_sampling_config):
        """Test that each BuilderOutput has an EpiModel."""
        result = build_sampling(basemodel_config=minimal_basemodel_config, sampling_config=minimal_sampling_config)

        for output in result:
            assert output.model is not None
            assert isinstance(output.model, EpiModel)

    def test_each_output_has_simulation_args(self, minimal_basemodel_config, minimal_sampling_config):
        """Test that each BuilderOutput has SimulationArguments."""
        result = build_sampling(basemodel_config=minimal_basemodel_config, sampling_config=minimal_sampling_config)

        for output in result:
            assert output.simulation is not None
            assert isinstance(output.simulation, SimulationArguments)

    def test_outputs_have_sequential_ids(self, minimal_basemodel_config, minimal_sampling_config):
        """Test that BuilderOutputs have sequential primary_ids."""
        result = build_sampling(basemodel_config=minimal_basemodel_config, sampling_config=minimal_sampling_config)

        ids = [output.primary_id for output in result]
        assert ids == list(range(len(result)))

    def test_models_have_different_populations(self, minimal_basemodel_config, minimal_sampling_config):
        """Test that each model has a different population."""
        result = build_sampling(basemodel_config=minimal_basemodel_config, sampling_config=minimal_sampling_config)

        populations = {output.model.population.name for output in result}
        assert len(populations) == len(result)
        assert "United_States_California" in populations
        assert "United_States_Texas" in populations


# Calibration strategy configurations for parametrized tests
CALIBRATION_STRATEGIES = [
    pytest.param("rejection", {"n_samples": 10, "epsilon": 100}, id="rejection"),
    pytest.param("smc", {"n_samples": 10, "generations": 3, "perturbation_kernel_scale": 0.5}, id="smc"),
    pytest.param("top_fraction", {"n_samples": 10, "top_fraction": 0.1}, id="top_fraction"),
]


class TestBuildCalibration:
    """Tests for build_calibration function with different strategies."""

    @pytest.fixture
    def minimal_basemodel_config(self):
        """Create a minimal BasemodelConfig for calibration testing (beta is calibrated)."""
        basemodel = make_sir_config(end_date=date(2024, 3, 31), n_sims=5, random_seed=42)
        basemodel.parameters["beta"] = Parameter(type="calibrated")
        return BasemodelConfig(model=basemodel)

    @pytest.fixture
    def calibration_config_factory(self, tmp_path):
        """Factory fixture to create CalibrationConfig with different strategies."""
        import pandas as pd

        from epymodelingsuite.schema.calibration import (
            CalibrationConfig,
            CalibrationConfiguration,
            CalibrationModelset,
            CalibrationParameter,
            CalibrationStrategy,
            ComparisonSpec,
            FittingWindow,
        )
        from epymodelingsuite.schema.common import Distribution

        def _create_config(strategy_name: str, strategy_options: dict):
            # Create temporary observed data file
            observed_data = pd.DataFrame(
                {
                    "date": pd.date_range(start="2024-01-01", periods=12, freq="W-SAT"),
                    "location": ["United_States_California"] * 12,
                    "value": np.random.randint(10, 100, size=12),
                }
            )
            observed_path = tmp_path / f"observed_{strategy_name}.csv"
            observed_data.to_csv(observed_path, index=False)

            strategy = CalibrationStrategy(name=strategy_name, options=strategy_options)

            fitting_window = FittingWindow(start_date=date(2024, 1, 1), end_date=date(2024, 2, 1))

            comparison = ComparisonSpec(
                observed_value_column="value",
                observed_date_column="date",
                observed_location_column="location",
                simulation=["S_to_I_total"],
            )

            calibration = CalibrationConfiguration(
                strategy=strategy,
                fitting_window=fitting_window,
                observed_data_path=str(observed_path),
                distance_function="rmse",
                comparison=[comparison],
                parameters={
                    "beta": CalibrationParameter(prior=Distribution(name="uniform", kwargs={"loc": 0, "scale": 1}))
                },
                compartments={},
            )

            modelset = CalibrationModelset(population_names=["US-CA"], calibration=calibration)

            return CalibrationConfig(modelset=modelset)

        return _create_config

    @pytest.mark.parametrize(("strategy_name", "strategy_options"), CALIBRATION_STRATEGIES)
    def test_returns_list_of_builder_outputs(
        self, minimal_basemodel_config, calibration_config_factory, strategy_name, strategy_options
    ):
        """Test that build_calibration returns list of BuilderOutput objects."""
        calibration_config = calibration_config_factory(strategy_name, strategy_options)
        result = build_calibration(basemodel_config=minimal_basemodel_config, calibration_config=calibration_config)

        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(item, BuilderOutput) for item in result)

    @pytest.mark.parametrize(("strategy_name", "strategy_options"), CALIBRATION_STRATEGIES)
    def test_each_output_has_calibrator(
        self, minimal_basemodel_config, calibration_config_factory, strategy_name, strategy_options
    ):
        """Test that each BuilderOutput has an ABCSampler."""
        from epydemix.calibration import ABCSampler

        calibration_config = calibration_config_factory(strategy_name, strategy_options)
        result = build_calibration(basemodel_config=minimal_basemodel_config, calibration_config=calibration_config)

        for output in result:
            assert output.calibrator is not None
            assert isinstance(output.calibrator, ABCSampler)

    @pytest.mark.parametrize(("strategy_name", "strategy_options"), CALIBRATION_STRATEGIES)
    def test_each_output_has_calibration_strategy(
        self, minimal_basemodel_config, calibration_config_factory, strategy_name, strategy_options
    ):
        """Test that each BuilderOutput has calibration strategy."""
        from epymodelingsuite.schema.calibration import CalibrationStrategy

        calibration_config = calibration_config_factory(strategy_name, strategy_options)
        result = build_calibration(basemodel_config=minimal_basemodel_config, calibration_config=calibration_config)

        for output in result:
            assert output.calibration is not None
            assert isinstance(output.calibration, CalibrationStrategy)

    @pytest.mark.parametrize(("strategy_name", "strategy_options"), CALIBRATION_STRATEGIES)
    def test_calibration_strategy_has_correct_name(
        self, minimal_basemodel_config, calibration_config_factory, strategy_name, strategy_options
    ):
        """Test that calibration strategy has correct name from config."""
        calibration_config = calibration_config_factory(strategy_name, strategy_options)
        result = build_calibration(basemodel_config=minimal_basemodel_config, calibration_config=calibration_config)

        assert result[0].calibration.name == strategy_name

    @pytest.mark.parametrize(("strategy_name", "strategy_options"), CALIBRATION_STRATEGIES)
    def test_calibration_strategy_has_correct_options(
        self, minimal_basemodel_config, calibration_config_factory, strategy_name, strategy_options
    ):
        """Test that calibration strategy has correct options from config."""
        calibration_config = calibration_config_factory(strategy_name, strategy_options)
        result = build_calibration(basemodel_config=minimal_basemodel_config, calibration_config=calibration_config)

        assert result[0].calibration.options is not None
        for key, value in strategy_options.items():
            assert key in result[0].calibration.options
            assert result[0].calibration.options[key] == value

    @pytest.mark.parametrize(("strategy_name", "strategy_options"), CALIBRATION_STRATEGIES)
    def test_no_simulation_args_for_calibration(
        self, minimal_basemodel_config, calibration_config_factory, strategy_name, strategy_options
    ):
        """Test that BuilderOutput has no simulation args for calibration workflow."""
        calibration_config = calibration_config_factory(strategy_name, strategy_options)
        result = build_calibration(basemodel_config=minimal_basemodel_config, calibration_config=calibration_config)

        for output in result:
            assert output.simulation is None

    def test_calibrator_has_priors_for_calibrated_parameters(
        self, minimal_basemodel_config, calibration_config_factory
    ):
        """Test that ABCSampler has priors set for calibrated parameters."""
        calibration_config = calibration_config_factory("rejection", {"n_samples": 10, "epsilon": 100})
        result = build_calibration(basemodel_config=minimal_basemodel_config, calibration_config=calibration_config)

        calibrator = result[0].calibrator
        assert calibrator is not None
        assert hasattr(calibrator, "priors")
        assert "beta" in calibrator.priors

    def test_calibrator_priors_are_scipy_distributions(self, minimal_basemodel_config, calibration_config_factory):
        """Test that priors are scipy frozen distribution objects."""
        calibration_config = calibration_config_factory("rejection", {"n_samples": 10, "epsilon": 100})
        result = build_calibration(basemodel_config=minimal_basemodel_config, calibration_config=calibration_config)

        calibrator = result[0].calibrator
        beta_prior = calibrator.priors["beta"]

        # Check it's a scipy frozen distribution (has rvs method)
        assert hasattr(beta_prior, "rvs")
        assert hasattr(beta_prior, "pdf")
        assert callable(beta_prior.rvs)

    def test_calibrator_prior_has_correct_distribution_type(self, minimal_basemodel_config, calibration_config_factory):
        """Test that prior has the correct distribution type (uniform)."""
        calibration_config = calibration_config_factory("rejection", {"n_samples": 10, "epsilon": 100})
        result = build_calibration(basemodel_config=minimal_basemodel_config, calibration_config=calibration_config)

        calibrator = result[0].calibrator
        beta_prior = calibrator.priors["beta"]

        # The fixture uses uniform(loc=0, scale=1), so check the distribution name
        assert beta_prior.dist.name == "uniform"

    def test_calibrator_prior_has_correct_parameters(self, minimal_basemodel_config, calibration_config_factory):
        """Test that prior distribution has correct loc and scale parameters."""
        calibration_config = calibration_config_factory("rejection", {"n_samples": 10, "epsilon": 100})
        result = build_calibration(basemodel_config=minimal_basemodel_config, calibration_config=calibration_config)

        calibrator = result[0].calibrator
        beta_prior = calibrator.priors["beta"]

        # The fixture uses uniform(loc=0, scale=1)
        # For scipy uniform, kwds contains loc and scale
        assert beta_prior.kwds["loc"] == 0
        assert beta_prior.kwds["scale"] == 1

    def test_calibrator_prior_can_generate_samples(self, minimal_basemodel_config, calibration_config_factory):
        """Test that prior can generate random samples in expected range."""
        calibration_config = calibration_config_factory("rejection", {"n_samples": 10, "epsilon": 100})
        result = build_calibration(basemodel_config=minimal_basemodel_config, calibration_config=calibration_config)

        calibrator = result[0].calibrator
        beta_prior = calibrator.priors["beta"]

        # Generate samples and verify they're in the expected range [0, 1]
        samples = beta_prior.rvs(size=100, random_state=42)
        assert len(samples) == 100
        assert all(0 <= s <= 1 for s in samples)

    @pytest.fixture
    def calibration_config_with_distance_function_factory(self, tmp_path):
        """Factory fixture to create CalibrationConfig with customizable distance function."""
        import pandas as pd

        from epymodelingsuite.schema.calibration import (
            CalibrationConfig,
            CalibrationConfiguration,
            CalibrationModelset,
            CalibrationParameter,
            CalibrationStrategy,
            ComparisonSpec,
            FittingWindow,
        )
        from epymodelingsuite.schema.common import Distribution

        def _create_config(distance_function):
            # Create temporary observed data file
            observed_data = pd.DataFrame(
                {
                    "date": pd.date_range(start="2024-01-01", periods=12, freq="W-SAT"),
                    "location": ["United_States_California"] * 12,
                    "value": np.random.randint(10, 100, size=12),
                }
            )
            observed_path = tmp_path / "observed_distfunc.csv"
            observed_data.to_csv(observed_path, index=False)

            strategy = CalibrationStrategy(name="top_fraction", options={"n_samples": 10, "top_fraction": 0.1})
            fitting_window = FittingWindow(start_date=date(2024, 1, 1), end_date=date(2024, 2, 1))

            comparison = ComparisonSpec(
                observed_value_column="value",
                observed_date_column="date",
                observed_location_column="location",
                simulation=["S_to_I_total"],
            )

            calibration = CalibrationConfiguration(
                strategy=strategy,
                fitting_window=fitting_window,
                observed_data_path=str(observed_path),
                distance_function=distance_function,
                comparison=[comparison],
                parameters={
                    "beta": CalibrationParameter(prior=Distribution(name="uniform", kwargs={"loc": 0, "scale": 1}))
                },
                compartments={},
            )

            modelset = CalibrationModelset(population_names=["US-CA"], calibration=calibration)

            return CalibrationConfig(modelset=modelset)

        return _create_config

    @pytest.mark.parametrize("dist_func_name", ["rmse", "wrmse", "mae", "mape", "wmape"])
    def test_distance_function_from_config_passed_to_calibrator(
        self, minimal_basemodel_config, calibration_config_with_distance_function_factory, dist_func_name
    ):
        """Test that distance_function string in config selects correct function."""
        calibration_config = calibration_config_with_distance_function_factory(dist_func_name)
        result = build_calibration(basemodel_config=minimal_basemodel_config, calibration_config=calibration_config)

        calibrator = result[0].calibrator
        assert calibrator is not None
        assert hasattr(calibrator, "distance_function")

        # Use test data where different distance functions produce distinguishable results.
        # With data=[1,2,3] and sim=[2,2,2], differences=[1,0,1]:
        #   rmse = sqrt(2/3) ≈ 0.816
        #   mae = 2/3 ≈ 0.667
        #   mape, wmape, wrmse all produce different values
        data_dict = {"data": np.array([1.0, 2.0, 3.0])}
        sim_dict = {"data": np.array([2.0, 2.0, 2.0])}

        wrapped_result = calibrator.distance_function(data_dict, sim_dict)

        # Verify it returns a valid distance value
        assert isinstance(wrapped_result, (int, float, np.floating))

        # Verify the calibrator uses the exact function specified in config
        # by comparing against the expected function from dist_func_dict
        expected_result = dist_func_dict[dist_func_name](data_dict, sim_dict)
        assert np.isclose(wrapped_result, expected_result), (
            f"Expected {dist_func_name} to return {expected_result}, got {wrapped_result}"
        )

    def test_custom_distance_function_passed_to_calibrator(
        self, minimal_basemodel_config, calibration_config_with_distance_function_factory, tmp_path
    ):
        """Test that user-defined distance function is passed to ABCSampler."""
        from epymodelingsuite.schema.calibration import UserDefinedFunction

        # Create a temporary script with a custom distance function
        custom_script = tmp_path / "custom_distance.py"
        custom_script.write_text(
            """
def custom_dist(data, simulation):
    '''Custom distance function that returns sum of absolute differences.'''
    import numpy as np
    return np.sum(np.abs(data['data'] - simulation['data']))
"""
        )

        # Create a UserDefinedFunction pointing to our custom script
        udf = UserDefinedFunction(
            user_script_path=str(custom_script),
            user_function_name="custom_dist",
        )

        calibration_config = calibration_config_with_distance_function_factory(udf)
        result = build_calibration(basemodel_config=minimal_basemodel_config, calibration_config=calibration_config)

        calibrator = result[0].calibrator
        assert calibrator is not None
        assert hasattr(calibrator, "distance_function")

        # Test the custom function behavior (sum of absolute differences)
        data_dict = {"data": np.array([1.0, 2.0, 3.0])}
        sim_dict = {"data": np.array([2.0, 3.0, 4.0])}

        result_value = calibrator.distance_function(data_dict, sim_dict)

        # Expected: |1-2| + |2-3| + |3-4| = 1 + 1 + 1 = 3
        assert result_value == 3.0

    def test_distance_function_wrapped_with_nan_alignment(
        self, minimal_basemodel_config, calibration_config_with_distance_function_factory
    ):
        """Test that distance function is wrapped with dist_func_date_alignment_wrapper."""
        calibration_config = calibration_config_with_distance_function_factory("rmse")
        result = build_calibration(basemodel_config=minimal_basemodel_config, calibration_config=calibration_config)

        calibrator = result[0].calibrator
        dist_func = calibrator.distance_function

        # The wrapper should truncate prepended NaNs in simulated data
        # Test: simulated data has NaNs at start, observed data is complete
        data_dict = {"data": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        sim_dict_with_nans = {"data": np.array([np.nan, np.nan, 3.0, 4.0, 5.0])}

        # If properly wrapped, it should truncate first 2 elements from both arrays
        # and compute RMSE on [3,4,5] vs [3,4,5] = 0
        result_with_nans = dist_func(data_dict, sim_dict_with_nans)

        # Perfect match after NaN alignment should give 0 (or very close to 0)
        assert np.isclose(result_with_nans, 0.0, atol=1e-10)

        # Compare with case where simulation exactly matches observed (no NaNs)
        sim_dict_no_nans = {"data": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        result_no_nans = dist_func(data_dict, sim_dict_no_nans)

        # Also should be 0 since it's a perfect match
        assert np.isclose(result_no_nans, 0.0, atol=1e-10)
