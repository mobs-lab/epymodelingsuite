"""Statistical validation tests for ABC calibration.

These tests verify that ABC calibration can statistically recover known parameters
from synthetic data. Unlike the @pytest.mark.slow tests which only verify code execution,
these tests validate that the posterior distributions contain the true parameter values.

Tests are marked with @pytest.mark.nightly and skipped by default.
Run with: pytest -m nightly tests/integration/test_abc_calibration.py
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from epydemix.model import EpiModel
from epydemix.population import load_epydemix_population

from epymodelingsuite.dispatcher.builder import dispatch_builder
from epymodelingsuite.dispatcher.runner import dispatch_runner
from epymodelingsuite.schema.basemodel import (
    BaseEpiModel,
    Compartment,
    Parameter,
    Population,
    Simulation,
    Timespan,
    Transition,
)
from epymodelingsuite.schema.calibration import (
    CalibrationConfig,
    CalibrationConfiguration,
    CalibrationModelset,
    CalibrationParameter,
    CalibrationStrategy,
    ComparisonSpec,
    FittingWindow,
)
from epymodelingsuite.schema.common import Distribution, Meta


def get_credible_interval(samples: np.ndarray, level: float = 0.80) -> tuple[float, float]:
    """Return (lower, upper) for given credible level.

    Parameters
    ----------
    samples : np.ndarray
        Array of posterior samples.
    level : float
        Credible interval level (e.g., 0.80 for 80% CI).

    Returns
    -------
    tuple[float, float]
        Lower and upper bounds of the credible interval.
    """
    alpha = (1 - level) / 2
    return np.percentile(samples, [alpha * 100, (1 - alpha) * 100])


def is_in_interval(value: float, lower: float, upper: float) -> bool:
    """Check if value is within [lower, upper]."""
    return lower <= value <= upper


# =============================================================================
# Shared config builders
# =============================================================================

_AGE_GROUPS = ["0-4", "5-17", "18-49", "50-64", "65+"]
_AGE_GROUP_MAPPING = {
    "0-4": [str(i) for i in range(5)],
    "5-17": [str(i) for i in range(5, 18)],
    "18-49": [str(i) for i in range(18, 50)],
    "50-64": [str(i) for i in range(50, 65)],
    "65+": [str(i) for i in range(65, 84)] + ["84+"],
}


def _make_basemodel_config(
    calibrated_params: list[str],
    fixed_params: dict[str, float],
    initial_infected: int = 100,
) -> BaseEpiModel:
    """Build a BaseEpiModel config for calibration testing.

    Parameters
    ----------
    calibrated_params : list[str]
        Parameter names to mark as calibrated.
    fixed_params : dict[str, float]
        Parameter names to fix at given values.
    initial_infected : int
        Number of initial infections.
    """
    parameters = {}
    for name in calibrated_params:
        parameters[name] = Parameter(type="calibrated")
    for name, value in fixed_params.items():
        parameters[name] = Parameter(type="scalar", value=value)

    return BaseEpiModel(
        name="sir_calibration_test",
        meta=Meta(description="SIR calibration test", version="1.0.0", date="2025-01-01"),
        timespan=Timespan(start_date=date(2025, 1, 1), end_date=date(2025, 3, 1), delta_t=1.0),
        simulation=Simulation(resample_frequency="W-SAT"),
        random_seed=42,
        population=Population(name="US-MA", age_groups=_AGE_GROUPS),
        compartments=[
            Compartment(id="S", label="Susceptible", init="default"),
            Compartment(id="I", label="Infectious", init=initial_infected),
            Compartment(id="R", label="Recovered"),
        ],
        transitions=[
            Transition(type="mediated", source="S", target="I", mediator="I", rate="beta"),
            Transition(type="spontaneous", source="I", target="R", rate="mu"),
        ],
        parameters=parameters,
    )


def _make_calibration_config(
    observed_data_path: str,
    prior_bounds: dict[str, tuple[float, float]],
    nsim: int = 500,
) -> CalibrationConfig:
    """Build a CalibrationConfig for testing.

    Parameters
    ----------
    observed_data_path : str
        Path to observed data CSV.
    prior_bounds : dict[str, tuple[float, float]]
        Prior bounds for each calibrated parameter as {name: (low, high)}.
    nsim : int
        Number of simulations for top_fraction strategy.
    """
    cal_params = {}
    for name, (low, high) in prior_bounds.items():
        cal_params[name] = CalibrationParameter(
            prior=Distribution(type="scipy", name="uniform", args=[low, high - low]),
        )

    return CalibrationConfig(
        modelset=CalibrationModelset(
            meta=Meta(description="Calibration test", version="1.0.0"),
            population_names=["US-MA"],
            calibration=CalibrationConfiguration(
                strategy=CalibrationStrategy(
                    name="top_fraction",
                    options={"Nsim": nsim, "top_fraction": 0.1},
                ),
                distance_function="rmse",
                observed_data_path=observed_data_path,
                comparison=[
                    ComparisonSpec(
                        observed_date_column="date",
                        observed_value_column="value",
                        observed_location_column="location",
                        simulation=["I_to_R_total"],
                    ),
                ],
                fitting_window=FittingWindow(
                    start_date=date(2025, 1, 4),
                    end_date=date(2025, 2, 22),
                ),
                compartments={},
                parameters=cal_params,
            ),
        ),
    )


def _run_calibration(
    observed_data_path: str,
    calibrated_params: list[str],
    fixed_params: dict[str, float],
    prior_bounds: dict[str, tuple[float, float]],
    nsim: int = 500,
) -> pd.DataFrame:
    """Build configs, run calibration, return posterior DataFrame.

    Parameters
    ----------
    observed_data_path : str
        Path to observed data CSV.
    calibrated_params : list[str]
        Parameter names to calibrate.
    fixed_params : dict[str, float]
        Parameter names to fix at given values.
    prior_bounds : dict[str, tuple[float, float]]
        Prior bounds for calibrated parameters.
    nsim : int
        Number of simulations for top_fraction strategy.

    Returns
    -------
    pd.DataFrame
        Posterior distribution samples.
    """
    basemodel_config = _make_basemodel_config(calibrated_params, fixed_params)
    calibration_config = _make_calibration_config(observed_data_path, prior_bounds, nsim)

    builder_outputs = dispatch_builder(
        basemodel_config=basemodel_config,
        calibration_config=calibration_config,
    )

    result = dispatch_runner(builder_outputs[0])
    return result.results.get_posterior_distribution()


@pytest.mark.nightly
class TestABCParameterRecovery:
    """Statistical validation that ABC calibration recovers known parameters.

    Uses a simple SIR model with known parameters to generate synthetic data,
    then verifies that ABC calibration can recover those parameters within
    credible intervals.
    """

    # True parameter values
    TRUE_BETA = 0.30
    TRUE_MU = 0.10

    # Prior bounds (uniform priors): (lower, upper)
    PRIOR_BOUNDS = {
        "beta": (0.15, 0.45),
        "mu": (0.05, 0.15),
    }

    @pytest.fixture(scope="class")
    def synthetic_sir_scenario(self, tmp_path_factory) -> SimpleNamespace:
        """Generate synthetic SIR data with known parameters.

        Creates a simple SIR model with true parameters, runs a simulation,
        and extracts I->R transitions as "observed" data for calibration.

        Returns
        -------
        SimpleNamespace
            Contains:
            - observed_data_path: Path to CSV with observed I->R transitions
        """
        model = EpiModel()

        population = load_epydemix_population(
            population_name="United_States_Massachusetts",
            age_group_mapping=_AGE_GROUP_MAPPING,
        )
        model.set_population(population)

        model.add_compartments(["S", "I", "R"])
        model.add_transition("S", "I", params=("beta", "I"), kind="mediated")
        model.add_transition("I", "R", params="mu", kind="spontaneous")

        model.add_parameter("beta", self.TRUE_BETA)
        model.add_parameter("mu", self.TRUE_MU)

        # Initial conditions: 100 infected in 18-49 age group
        n_age = len(model.population.Nk)
        i_init = np.zeros(n_age)
        i_init[2] = 100
        s_init = model.population.Nk - i_init

        init_conditions = {
            "S": s_init,
            "I": i_init,
            "R": np.zeros(n_age),
        }

        rng = np.random.default_rng(42)
        results = model.run_simulations(
            start_date="2025-01-01",
            end_date="2025-03-01",
            initial_conditions_dict=init_conditions,
            Nsim=1,
            dt=1.0,
            resample_frequency="W-SAT",
            rng=rng,
        )

        # Extract I->R transitions as "observed" data
        transitions = results.get_stacked_transitions()
        observed = transitions["I_to_R_total"][0]

        df = pd.DataFrame(
            {
                "date": results.dates,
                "value": observed.astype(int),
                "location": "US-MA",
            }
        )

        tmp_path = tmp_path_factory.mktemp("data")
        data_path = tmp_path / "synthetic_observed.csv"
        df.to_csv(data_path, index=False)

        return SimpleNamespace(observed_data_path=str(data_path))

    @pytest.fixture(scope="class")
    def beta_only_calibration_result(self, synthetic_sir_scenario):
        """Run calibration for beta only with fixed mu."""
        posterior = _run_calibration(
            observed_data_path=synthetic_sir_scenario.observed_data_path,
            calibrated_params=["beta"],
            fixed_params={"mu": self.TRUE_MU},
            prior_bounds={"beta": self.PRIOR_BOUNDS["beta"]},
        )
        return SimpleNamespace(posterior=posterior, beta_samples=posterior["beta"].values)

    @pytest.fixture(scope="class")
    def mu_only_calibration_result(self, synthetic_sir_scenario):
        """Run calibration for mu only with fixed beta."""
        posterior = _run_calibration(
            observed_data_path=synthetic_sir_scenario.observed_data_path,
            calibrated_params=["mu"],
            fixed_params={"beta": self.TRUE_BETA},
            prior_bounds={"mu": self.PRIOR_BOUNDS["mu"]},
        )
        return SimpleNamespace(posterior=posterior, mu_samples=posterior["mu"].values)

    @pytest.fixture(scope="class")
    def multi_param_calibration_result(self, synthetic_sir_scenario):
        """Run calibration for both beta and mu together."""
        posterior = _run_calibration(
            observed_data_path=synthetic_sir_scenario.observed_data_path,
            calibrated_params=["beta", "mu"],
            fixed_params={},
            prior_bounds=self.PRIOR_BOUNDS,
            nsim=750,
        )
        return SimpleNamespace(
            posterior=posterior,
            beta_samples=posterior["beta"].values,
            mu_samples=posterior["mu"].values,
        )

    def test_beta_recovery_in_credible_interval(self, beta_only_calibration_result):
        """Verify true beta falls within 90% posterior credible interval."""
        beta_samples = beta_only_calibration_result.beta_samples
        beta_low, beta_high = get_credible_interval(beta_samples, level=0.90)

        assert is_in_interval(self.TRUE_BETA, beta_low, beta_high), (
            f"True beta ({self.TRUE_BETA}) not in 90% CI [{beta_low:.4f}, {beta_high:.4f}]. "
            f"Posterior mean={np.mean(beta_samples):.4f}, std={np.std(beta_samples):.4f}"
        )

    def test_mu_recovery_in_credible_interval(self, mu_only_calibration_result):
        """Verify true mu falls within 90% posterior credible interval."""
        mu_samples = mu_only_calibration_result.mu_samples
        mu_low, mu_high = get_credible_interval(mu_samples, level=0.90)

        assert is_in_interval(self.TRUE_MU, mu_low, mu_high), (
            f"True mu ({self.TRUE_MU}) not in 90% CI [{mu_low:.4f}, {mu_high:.4f}]. "
            f"Posterior mean={np.mean(mu_samples):.4f}, std={np.std(mu_samples):.4f}"
        )

    def test_posterior_narrower_than_prior(self, beta_only_calibration_result, mu_only_calibration_result):
        """Verify calibration learned something (posterior more concentrated than prior).

        If calibration is informative, the posterior variance should be substantially
        smaller than the prior variance. We require at least 90% reduction.
        """
        # Uniform distribution variance = (b - a)^2 / 12
        beta_low, beta_high = self.PRIOR_BOUNDS["beta"]
        prior_var_beta = (beta_high - beta_low) ** 2 / 12
        posterior_var_beta = np.var(beta_only_calibration_result.beta_samples)

        assert posterior_var_beta < 0.1 * prior_var_beta, (
            f"Beta posterior variance ({posterior_var_beta:.6f}) not < 10% of prior variance ({prior_var_beta:.6f})"
        )

        mu_low, mu_high = self.PRIOR_BOUNDS["mu"]
        prior_var_mu = (mu_high - mu_low) ** 2 / 12
        posterior_var_mu = np.var(mu_only_calibration_result.mu_samples)

        assert posterior_var_mu < 0.1 * prior_var_mu, (
            f"Mu posterior variance ({posterior_var_mu:.8f}) not < 10% of prior variance ({prior_var_mu:.8f})"
        )

    def test_posterior_mean_near_true_value(self, beta_only_calibration_result, mu_only_calibration_result):
        """Verify posterior mean is within 20% of true value.

        A well-calibrated model should have its posterior mean close to the true
        parameter value, not just containing it within the CI.
        """
        # Beta
        posterior_mean_beta = np.mean(beta_only_calibration_result.beta_samples)
        relative_error_beta = abs(posterior_mean_beta - self.TRUE_BETA) / self.TRUE_BETA

        assert relative_error_beta < 0.20, (
            f"Beta posterior mean ({posterior_mean_beta:.4f}) differs from true value "
            f"({self.TRUE_BETA}) by {relative_error_beta * 100:.1f}% (> 20%)"
        )

        # Mu
        posterior_mean_mu = np.mean(mu_only_calibration_result.mu_samples)
        relative_error_mu = abs(posterior_mean_mu - self.TRUE_MU) / self.TRUE_MU

        assert relative_error_mu < 0.20, (
            f"Mu posterior mean ({posterior_mean_mu:.4f}) differs from true value "
            f"({self.TRUE_MU}) by {relative_error_mu * 100:.1f}% (> 20%)"
        )

    def test_multi_parameter_joint_recovery(self, multi_param_calibration_result):
        """Verify both beta and mu are recovered when calibrated jointly.

        This is a more challenging test: both parameters must be recovered
        simultaneously, which tests the identifiability of the model.
        """
        # Check beta
        beta_samples = multi_param_calibration_result.beta_samples
        beta_low, beta_high = get_credible_interval(beta_samples, level=0.90)

        assert is_in_interval(self.TRUE_BETA, beta_low, beta_high), (
            f"True beta ({self.TRUE_BETA}) not in 90% CI [{beta_low:.4f}, {beta_high:.4f}] during joint calibration. "
            f"Posterior mean={np.mean(beta_samples):.4f}, std={np.std(beta_samples):.4f}"
        )

        # Check mu
        mu_samples = multi_param_calibration_result.mu_samples
        mu_low, mu_high = get_credible_interval(mu_samples, level=0.90)

        assert is_in_interval(self.TRUE_MU, mu_low, mu_high), (
            f"True mu ({self.TRUE_MU}) not in 90% CI [{mu_low:.4f}, {mu_high:.4f}] during joint calibration. "
            f"Posterior mean={np.mean(mu_samples):.4f}, std={np.std(mu_samples):.4f}"
        )
