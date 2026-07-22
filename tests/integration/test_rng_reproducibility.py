"""RNG reproducibility for simulation, calibration, and projection through the dispatcher.

These exercise the real pipeline end-to-end (dispatch_builder -> dispatch_runner) and
verify that seeding from ``random_seed`` makes forward simulation, the ABC parameter
search, and the projections reproducible, while distinct seeds / locations stay
independent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from epymodelingsuite.dispatcher.builder import _ABCSAMPLER_SUPPORTS_RNG, dispatch_builder
from epymodelingsuite.dispatcher.runner import dispatch_runner
from epymodelingsuite.schema.basemodel import validate_basemodel
from epymodelingsuite.schema.calibration import validate_calibration
from epymodelingsuite.schema.dispatcher import CalibrationOutput, SimulationOutput

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

# ISO location code -> epydemix population name, for synthesizing multi-location data.
ISO_TO_POPULATION = {
    "US-MA": "United_States__Massachusetts",
    "US-CA": "United_States__California",
}

AGE_GROUP_MAPPING = {
    "0-4": [str(i) for i in range(5)],
    "5-17": [str(i) for i in range(5, 18)],
    "18-49": [str(i) for i in range(18, 50)],
    "50-64": [str(i) for i in range(50, 65)],
    "65+": [str(i) for i in range(65, 84)] + ["84+"],
}

# The calibration/projection paths need rng support in epydemix
requires_abcsampler_rng = pytest.mark.skipif(
    not _ABCSAMPLER_SUPPORTS_RNG,
    reason="requires epydemix with ABCSampler(rng=...) support",
)


# =============================================================================
# Helpers
# =============================================================================


def _synthesize_observed_data(tmp_path, iso_locations):
    """Write synthetic observed I_to_R data for the given ISO locations to one CSV.

    Each location is simulated once with a fixed (data-generating) seed so the
    "observed" series is deterministic and independent of the calibration seed.
    """
    from epydemix.model import EpiModel

    from epymodelingsuite.utils import load_epydemix_population

    frames = []
    for data_seed, iso in enumerate(iso_locations):
        model = EpiModel()
        population = load_epydemix_population(
            population_name=ISO_TO_POPULATION[iso],
            age_group_mapping=AGE_GROUP_MAPPING,
        )
        model.set_population(population)
        model.add_compartments(["S", "I", "R"])
        model.add_transition("S", "I", kind="mediated", params=("beta", "I"))
        model.add_transition("I", "R", kind="spontaneous", params="mu")
        model.add_parameter("beta", 0.25)
        model.add_parameter("mu", 0.1)

        n_age = len(model.population.Nk)
        i_init = np.zeros(n_age)
        i_init[2] = 100
        init_conditions = {"S": model.population.Nk - i_init, "I": i_init, "R": np.zeros(n_age)}

        sim_results = model.run_simulations(
            start_date="2025-01-01",
            end_date="2025-03-01",
            initial_conditions_dict=init_conditions,
            Nsim=1,
            dt=1.0,
            resample_frequency="W-SAT",
            rng=np.random.default_rng(1000 + data_seed),
        )
        i_to_r = sim_results.get_stacked_transitions()["I_to_R_total"][0]
        frames.append(pd.DataFrame({"date": sim_results.dates, "value": i_to_r.astype(int), "location": iso}))

    data_path = tmp_path / "synthetic_observed.csv"
    pd.concat(frames, ignore_index=True).to_csv(data_path, index=False)
    return str(data_path)


def _run_simulation(*, seed=42):
    """Build + run a plain forward simulation end-to-end; return the SimulationOutput."""
    with open(FIXTURES_DIR / "minimal_basemodel.yaml") as f:
        basemodel_raw = yaml.safe_load(f)
    basemodel_raw["model"]["random_seed"] = seed
    builder_output = dispatch_builder(basemodel_config=validate_basemodel(basemodel_raw))
    return dispatch_runner(builder_output)


def _get_transitions(output: SimulationOutput) -> dict[str, list]:
    """A simulation's stacked transition arrays as plain lists, comparable with ``==``."""
    return {key: value.tolist() for key, value in output.results.get_stacked_transitions().items()}


def _make_configs(observed_path, *, seed=42, populations=("US-MA",), projection=None):
    """Load the minimal calibration fixtures and override seed / populations / projection."""
    with open(FIXTURES_DIR / "minimal_basemodel_calibration.yaml") as f:
        basemodel_raw = yaml.safe_load(f)
    with open(FIXTURES_DIR / "minimal_modelset_calibration.yaml") as f:
        modelset_raw = yaml.safe_load(f)

    basemodel_raw["model"]["random_seed"] = seed
    modelset_raw["modelset"]["population_names"] = list(populations)
    calibration = modelset_raw["modelset"]["calibration"]
    calibration["observed_data_path"] = observed_path
    if projection is not None:
        calibration["projection"] = projection

    return validate_basemodel(basemodel_raw), validate_calibration(modelset_raw)


def _build_calibration(observed_path, *, seed=42, populations=("US-MA",), projection=None):
    """Build (but do not run) the per-location calibrators; return the builder outputs."""
    basemodel_config, calibration_config = _make_configs(
        observed_path, seed=seed, populations=populations, projection=projection
    )
    return dispatch_builder(basemodel_config=basemodel_config, calibration_config=calibration_config)


def _run_calibration(observed_path, *, seed=42, populations=("US-MA",), projection=None):
    """Build + run the calibration (optionally with projection); return the outputs list."""
    builder_outputs = _build_calibration(observed_path, seed=seed, populations=populations, projection=projection)
    return [dispatch_runner(builder_output) for builder_output in builder_outputs]


def _get_posterior(output: CalibrationOutput) -> pd.DataFrame:
    """A run's posterior distribution, normalized for equality comparison."""
    return output.results.get_posterior_distribution().reset_index(drop=True)


def _get_projections(output: CalibrationOutput, scenario="baseline") -> list[dict]:
    """A run's projection trajectories as plain nested lists, comparable with ``==``.

    Arrays are converted to lists so two runs' projections can be compared exactly
    (and symmetrically) with ``==`` / ``!=``
    """
    return [
        {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in trajectory.items()}
        for trajectory in output.results.projections[scenario]
    ]


def _find_location(outputs: list[CalibrationOutput], iso: str) -> CalibrationOutput:
    """Return the per-location result for the given ISO code from a multi-location run."""
    population = ISO_TO_POPULATION[iso]
    return next(output for output in outputs if output.population == population)


@pytest.fixture(scope="module")
def observed_ma(tmp_path_factory):
    """Synthetic observed data for a single location (US-MA)."""
    return _synthesize_observed_data(tmp_path_factory.mktemp("data_ma"), ["US-MA"])


@pytest.fixture(scope="module")
def observed_ma_ca(tmp_path_factory):
    """Synthetic observed data for two locations (US-MA, US-CA)."""
    return _synthesize_observed_data(tmp_path_factory.mktemp("data_ma_ca"), ["US-MA", "US-CA"])


# =============================================================================
# Simulation reproducibility (e2e, no calibration; runs on any epydemix build)
# =============================================================================


@pytest.mark.dynamics
class TestSimulationReproducibility:
    """Plain forward simulation seeded from ``random_seed`` via run_simulation."""

    def test_same_seed_is_reproducible(self):
        """Positive: two runs with the same seed produce identical trajectories."""
        result_a = _run_simulation(seed=42)
        result_b = _run_simulation(seed=42)

        transitions_a = _get_transitions(result_a)
        transitions_b = _get_transitions(result_b)
        assert transitions_a == transitions_b, "same seed should reproduce identical simulation trajectories"

    def test_different_seed_differs(self):
        """Negative: a different seed produces different trajectories."""
        result_a = _run_simulation(seed=42)
        result_b = _run_simulation(seed=123)

        transitions_a = _get_transitions(result_a)
        transitions_b = _get_transitions(result_b)
        assert transitions_a != transitions_b, "different seeds should produce different simulation trajectories"

    def test_unseeded_is_nondeterministic(self):
        """random_seed=None runs without error and is not reproducible across runs
        (no accidental forced determinism).
        """
        result_a = _run_simulation(seed=None)
        result_b = _run_simulation(seed=None)

        assert isinstance(result_a, SimulationOutput) and result_a.results is not None, (
            "unseeded run should still produce valid results"
        )
        transitions_a = _get_transitions(result_a)
        transitions_b = _get_transitions(result_b)
        assert transitions_a != transitions_b, "unseeded runs should not be reproducible"


# =============================================================================
# Calibration reproducibility
# =============================================================================


@requires_abcsampler_rng
@pytest.mark.slow
class TestCalibrationReproducibility:
    def test_same_seed_is_reproducible(self, observed_ma):
        """Positive: two runs with the same seed produce identical posteriors."""
        (result_a,) = _run_calibration(observed_ma, seed=42)
        (result_b,) = _run_calibration(observed_ma, seed=42)

        assert _get_posterior(result_a).equals(_get_posterior(result_b)), "same seed should reproduce the posterior"
        assert result_a.results.get_distances() == pytest.approx(result_b.results.get_distances()), (
            "same seed should reproduce the ABC distances"
        )
        assert np.asarray(result_a.results.get_weights()) == pytest.approx(
            np.asarray(result_b.results.get_weights())
        ), "same seed should reproduce the ABC weights"

    def test_different_seed_differs(self, observed_ma):
        """Negative: a different seed produces a different posterior of the same shape."""
        (result_a,) = _run_calibration(observed_ma, seed=42)
        (result_b,) = _run_calibration(observed_ma, seed=123)

        posterior_a, posterior_b = _get_posterior(result_a), _get_posterior(result_b)
        assert list(posterior_a.columns) == list(posterior_b.columns), (
            "posteriors should share the same parameter columns"
        )
        assert not posterior_a.equals(posterior_b), "different seeds should yield different posteriors"

    def test_unseeded_is_nondeterministic(self, observed_ma):
        """random_seed=None runs without error and is not reproducible across runs
        (no accidental forced determinism).
        """
        (result_a,) = _run_calibration(observed_ma, seed=None)
        (result_b,) = _run_calibration(observed_ma, seed=None)

        assert isinstance(result_a, CalibrationOutput) and result_a.results is not None, (
            "unseeded calibration should still produce valid results"
        )
        assert not _get_posterior(result_a).equals(_get_posterior(result_b)), (
            "unseeded calibration runs should not be reproducible"
        )


# =============================================================================
# Projection reproducibility
# =============================================================================


@requires_abcsampler_rng
@pytest.mark.slow
class TestProjectionReproducibility:
    def test_same_seed_is_reproducible(self, observed_ma):
        """Positive: same seed -> identical projection trajectories."""
        (result_a,) = _run_calibration(observed_ma, seed=42, projection={"n_trajectories": 10})
        (result_b,) = _run_calibration(observed_ma, seed=42, projection={"n_trajectories": 10})

        projections_a = _get_projections(result_a)
        projections_b = _get_projections(result_b)
        assert projections_a, "expected a non-empty set of trajectories"
        assert projections_a == projections_b, "same seed should reproduce identical projection trajectories"

    def test_different_seed_differs(self, observed_ma):
        """Negative: a different seed changes the projection trajectories."""
        (result_a,) = _run_calibration(observed_ma, seed=42, projection={"n_trajectories": 10})
        (result_b,) = _run_calibration(observed_ma, seed=7, projection={"n_trajectories": 10})

        projections_a = _get_projections(result_a)
        projections_b = _get_projections(result_b)
        assert projections_a != projections_b, "different seeds should yield different projection trajectories"

    def test_prefix_is_stable_across_trajectory_count(self, observed_ma):
        """A larger n_trajectories run reproduces a smaller run's trajectories as a
        prefix. Each trajectory's child rng is keyed by its index alone
        (spawn_key = base + (i,)), independent of the total count, so with the same
        seed the first n of an n+k run match the n run exactly.
        """
        (result_few,) = _run_calibration(observed_ma, seed=42, projection={"n_trajectories": 5})
        (result_many,) = _run_calibration(observed_ma, seed=42, projection={"n_trajectories": 10})

        projections_few = _get_projections(result_few)
        projections_many = _get_projections(result_many)

        assert len(projections_few) == 5, "expected 5 raw trajectories"
        assert len(projections_many) == 10, "expected 10 raw trajectories"
        assert projections_many[:5] == projections_few, (
            "the first n trajectories should be identical regardless of the total count"
        )


# =============================================================================
# Multi-location independence (guards the per-location rng spawn in build_calibration)
# =============================================================================


@requires_abcsampler_rng
@pytest.mark.slow
class TestMultiLocationIndependence:
    def test_locations_reproducible_but_independent(self, observed_ma_ca):
        """Each location is reproducible across runs (same seed), but the two
        locations draw independent parameter streams (not identical to each other).
        """
        results_a = _run_calibration(observed_ma_ca, seed=42, populations=("US-MA", "US-CA"))
        results_b = _run_calibration(observed_ma_ca, seed=42, populations=("US-MA", "US-CA"))
        assert len(results_a) == len(results_b) == 2, "expected one result per location"

        # Reproducible per location across runs.
        for location_a, location_b in zip(results_a, results_b, strict=True):
            assert _get_posterior(location_a).equals(_get_posterior(location_b)), (
                "each location should be reproducible across runs"
            )

        # Independent between locations: the two per-location samplers use different
        # spawned child rngs, so their posteriors are not identical.
        posterior_ma, posterior_ca = _get_posterior(results_a[0]), _get_posterior(results_a[1])
        assert not posterior_ma.equals(posterior_ca), "distinct locations should not share an rng stream"

    def test_location_result_independent_of_order(self, observed_ma_ca):
        """A location's result is the same regardless of its position in the batch:
        MA in [MA, CA] must match MA in [CA, MA] (seeding is name-keyed, not order-keyed).
        """
        ma_first = _run_calibration(observed_ma_ca, seed=42, populations=("US-MA", "US-CA"))
        ma_last = _run_calibration(observed_ma_ca, seed=42, populations=("US-CA", "US-MA"))

        assert _get_posterior(_find_location(ma_first, "US-MA")).equals(
            _get_posterior(_find_location(ma_last, "US-MA"))
        ), "a location's result should not depend on its position in the batch"

    def test_location_result_matches_single_location_run(self, observed_ma_ca):
        """A location calibrated within a batch matches the same location run alone:
        MA in [CA, MA] must match MA in [MA].
        """
        in_batch = _run_calibration(observed_ma_ca, seed=42, populations=("US-CA", "US-MA"))
        (solo,) = _run_calibration(observed_ma_ca, seed=42, populations=("US-MA",))

        assert _get_posterior(_find_location(in_batch, "US-MA")).equals(_get_posterior(solo)), (
            "a batched location should match the same location run alone"
        )


# =============================================================================
# Builder-level guards (fast: no calibration run)
# =============================================================================


@requires_abcsampler_rng
class TestBuilderSeeding:
    """Regression guards so build_calibration cannot silently stop seeding the sampler."""

    def test_samplers_are_seeded_with_independent_streams(self, observed_ma_ca):
        """Every per-location sampler is seeded, and their rng streams are independent."""
        builder_outputs = _build_calibration(observed_ma_ca, seed=42, populations=("US-MA", "US-CA"))

        assert len(builder_outputs) == 2, "expected one builder output per location"
        samplers = [builder_output.calibrator for builder_output in builder_outputs]

        # Each sampler requested seeding (rng was threaded in).
        assert all(sampler._seed_requested for sampler in samplers), "every per-location sampler should be seeded"

        # Per-location spawned children are distinct: first draws differ between locations.
        first_draws = [sampler.rng.random() for sampler in samplers]
        assert first_draws[0] != first_draws[1], "per-location rng streams should be independent"

    def test_rebuild_is_reproducible(self, observed_ma_ca):
        """Rebuilding with the same seed reproduces each location's rng stream; a
        different seed changes it.
        """

        def _location_draws(seed):
            outputs = _build_calibration(observed_ma_ca, seed=seed, populations=("US-MA", "US-CA"))
            # First draw from each location's sampler identifies its rng stream.
            return [builder_output.calibrator.rng.random() for builder_output in outputs]

        draws_a = _location_draws(42)
        draws_b = _location_draws(42)
        draws_c = _location_draws(7)

        assert draws_a == draws_b, "same seed should reproduce each location's rng stream"

        # Negative control: a different seed yields different rng streams.
        assert draws_a != draws_c, "a different seed should change each location's rng stream"

    def test_location_stream_is_stable_across_batch_composition(self, observed_ma_ca):
        """A location's rng stream depends only on its name + seed, not on which other
        locations are in the batch or their order (name-keyed, not position-keyed).
        """

        def _ma_first_draw(populations):
            outputs = _build_calibration(observed_ma_ca, seed=42, populations=populations)
            ma = next(
                builder_output
                for builder_output in outputs
                if builder_output.model.population.name == "United_States__Massachusetts"
            )
            return ma.calibrator.rng.random()

        solo = _ma_first_draw(("US-MA",))
        in_batch_last = _ma_first_draw(("US-CA", "US-MA"))  # MA is not at index 0 here

        assert solo == in_batch_last, "US-MA's rng stream should not depend on batch composition"
