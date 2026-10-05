"""Tests for epymodelingsuite.output.trajectory_samples."""

import gzip
import io
from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from epymodelingsuite.dispatcher.output import generate_calibration_outputs
from epymodelingsuite.output import trajectory_samples as ts
from epymodelingsuite.schema.output import (
    FlusightForecastOutput,
    ModelMetaOutput,
    OutputConfig,
    OutputConfiguration,
    TabularOutputTypeEnum,
)

HORIZONS = [-1, 0, 1, 2, 3]
REFERENCE_DATE = date(2026, 10, 10)
# Include extra projection weeks on either side of the submitted horizons.
# With this start date, columns 3:8 correspond to horizons -1 through 3.
DATES = pd.date_range(pd.Timestamp(REFERENCE_DATE) - pd.Timedelta(weeks=4), periods=9, freq="7D")


def _trajectories(n: int, seed: int = 0) -> dict:
    # Deterministic paths make value comparisons repeatable; ed_prop supplies a
    # separate projection variable for testing the ED transition strategy.
    rng = np.random.default_rng(seed)
    hosp = rng.gamma(2.0, 100.0, (n, len(DATES)))
    return {
        "date": [DATES.to_numpy() for _ in range(n)],
        "hospitalizations": list(hosp),
        "ed_prop": list(hosp / 10_000),
    }


def _calibration(population: str, trajectories: dict, seed: int | None = 42) -> MagicMock:
    # Supply the result data and metadata consumed by the output code without fitting a model.
    calibration = MagicMock()
    calibration.primary_id = 1
    calibration.seed = seed
    calibration.population = population
    calibration.delta_t = 1.0
    calibration.start_date_reference = None
    calibration.results.get_projection_trajectories.return_value = trajectories
    return calibration


def _flusight(**kwargs) -> FlusightForecastOutput:
    return FlusightForecastOutput(reference_date=REFERENCE_DATE, samples={"n_samples": 100}, **kwargs)


def _concat(rows: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(rows, ignore_index=True)


class TestSelectTrajectoryIndices:
    """Tests for sample selection."""

    def test_selects_n_distinct_rows(self):
        """Test that n distinct trajectories are selected."""
        # Requesting 100 of 1,000 forces subsampling; unique indices rule out replacement.
        values = np.arange(1000 * 5, dtype=float).reshape(1000, 5)
        idx = ts.select_trajectory_indices(values, 100, seed=1)
        assert len(idx) == 100
        assert len(set(idx)) == 100

    def test_same_seed_same_selection(self):
        """Test that the same seed gives the same selection."""
        values = np.random.default_rng(0).random((500, 5))
        # Hold the input fixed and verify that the selection seed reproduces the same indices.
        assert (
            ts.select_trajectory_indices(values, 100, seed=7) == ts.select_trajectory_indices(values, 100, seed=7)
        ).all()

    def test_fewer_than_n_returns_all_without_resampling(self):
        """Test that all trajectories are returned, without resampling, when fewer than n exist."""
        values = np.ones((60, 5))
        # Each original index should appear once; the result must not be padded to 100.
        assert ts.select_trajectory_indices(values, 100, seed=1).tolist() == list(range(60))

    def test_trajectories_with_nan_are_never_selected(self):
        """Test that trajectories with NaN are never selected."""
        values = np.ones((150, 5))
        # A single missing horizon invalidates the whole trajectory, leaving 75 complete rows.
        values[::2, 3] = np.nan
        idx = ts.select_trajectory_indices(values, 100, seed=1)
        assert len(idx) == 75
        assert not np.isnan(values[idx]).any()

    def test_registered_selector_is_used(self, monkeypatch):
        """Test that a newly registered selector is used and its indices map to the original rows."""
        # A deterministic selector makes the mapping from filtered rows to input rows observable.
        monkeypatch.setitem(ts.SAMPLE_SELECTORS, "first", lambda values, n, rng: np.arange(min(n, len(values))))
        values = np.ones((10, 5))
        values[0, 0] = np.nan
        # Indices refer to the original rows, after dropping the incomplete row 0
        assert ts.select_trajectory_indices(values, 3, method="first").tolist() == [1, 2, 3]


class TestTrajectoriesToSampleRows:
    """Tests for building hub sample rows."""

    def test_ids_horizons_and_dates(self):
        """Test sample ids, horizons and target end dates of the rows."""
        rows = ts.trajectories_to_sample_rows(
            np.ones((3, 5)), HORIZONS, date(2026, 10, 10), "25", "wk inc flu hosp", "MA", integer=True
        )
        # Three trajectories create 15 rows, with one sample ID shared across five horizons.
        assert len(rows) == 15
        assert rows.output_type_id.unique().tolist() == ["MA00", "MA01", "MA02"]
        assert rows.groupby("output_type_id").horizon.apply(list).map(lambda h: h == HORIZONS).all()
        first = rows[rows.output_type_id == "MA00"]
        # Horizon -1 is the Saturday before the reference date; subsequent horizons advance weekly.
        assert first.target_end_date.tolist() == [date(2026, 10, 3) + pd.Timedelta(weeks=i) for i in range(5)]
        assert set(rows.output_type) == {"sample"}

    def test_counts_are_rounded_non_negative_integers(self):
        """Test that count values are clipped at 0 and rounded to integers."""
        # Cover negative clipping, rounding down/up, and ties-to-even rounding (10.5 -> 10).
        values = np.array([[-2.0, 0.4, 1.6, 10.5, 3.2]])
        rows = ts.trajectories_to_sample_rows(
            values, HORIZONS, date(2026, 10, 10), "US", "wk inc flu hosp", "US", integer=True
        )
        assert rows.value.tolist() == [0.0, 0.0, 2.0, 10.0, 3.0]

    def test_proportions_are_clipped_to_unit_interval(self):
        """Test that proportion values are clipped to [0, 1]."""
        # Include values outside both bounds and valid fractions that must remain unchanged.
        values = np.array([[-0.1, 0.2, 1.3, 0.5, 1.0]])
        rows = ts.trajectories_to_sample_rows(
            values, HORIZONS, date(2026, 10, 10), "US", "wk inc flu prop ed visits", "US", upper=1
        )
        assert rows.value.tolist() == [0.0, 0.2, 1.0, 0.5, 1.0]


class TestBuildFlusightTrajectorySamples:
    """Tests for building FluSight sample tables from calibration results."""

    def test_one_hundred_integer_samples_per_location(self):
        """Test that each location gets 100 non-negative integer samples over horizons -1..3."""
        # National and state outputs exercise both US/FIPS identifiers and separate sample counts.
        calibrations = [
            _calibration("United_States", _trajectories(500)),
            _calibration("United_States_Massachusetts", _trajectories(500, seed=1)),
        ]
        rows, warns = ts.build_flusight_trajectory_samples(calibrations, _flusight(), pd.DataFrame())
        df = _concat(rows)

        assert warns == []
        assert set(df.target) == {"wk inc flu hosp"}
        assert df.groupby("location").output_type_id.nunique().to_dict() == {"25": 100, "US": 100}
        assert df[df.location == "25"].output_type_id.str.startswith("MA").all()
        assert df.groupby("output_type_id").horizon.apply(sorted).map(lambda h: h == [-1, 0, 1, 2, 3]).all()
        assert (df.value % 1 == 0).all()
        assert (df.value >= 0).all()

    def test_values_are_whole_trajectories(self):
        """Test that each sample is one projection trajectory, not values mixed across trajectories."""
        traj = _trajectories(200)
        rows, _ = ts.build_flusight_trajectory_samples(
            [_calibration("United_States", traj)], _flusight(), pd.DataFrame()
        )
        sample = _concat(rows).query("output_type_id == 'US00'").sort_values("horizon").value.to_numpy()
        candidates = np.rint(np.stack(traj["hospitalizations"])[:, 3:8])  # horizons -1..3
        # All five values must match one source trajectory after count rounding.
        # Matching each horizon to an arbitrary source row would allow mixed trajectories.
        assert (candidates == sample).all(axis=1).any()

    def test_seed_makes_selection_reproducible(self):
        """Test that the same seed selects the same samples."""
        # The output sample seed is unset, so reproducibility must come from calibration.seed.
        traj = _trajectories(500)
        a, _ = ts.build_flusight_trajectory_samples([_calibration("United_States", traj)], _flusight(), pd.DataFrame())
        b, _ = ts.build_flusight_trajectory_samples([_calibration("United_States", traj)], _flusight(), pd.DataFrame())
        pd.testing.assert_frame_equal(_concat(a), _concat(b))

    def test_fewer_trajectories_are_all_submitted_with_warning(self):
        """Test that all trajectories are submitted with a warning when fewer than requested exist."""
        # The config requests 100, but only 60 exist: return 60 unique IDs and explain the shortfall.
        rows, warns = ts.build_flusight_trajectory_samples(
            [_calibration("United_States", _trajectories(60))], _flusight(), pd.DataFrame()
        )
        assert _concat(rows).output_type_id.nunique() == 60
        assert any("only 60 complete trajectories" in w for w in warns)

    def test_trajectories_missing_a_horizon_are_skipped(self):
        """Test that trajectories with NaN in a horizon are not submitted."""
        traj = _trajectories(120)
        # Break one horizon in 30 paths: 90 complete trajectories remain, below the requested 100.
        for i in range(30):
            traj["hospitalizations"][i] = traj["hospitalizations"][i].copy()
            traj["hospitalizations"][i][5] = np.nan  # horizon 1
        rows, warns = ts.build_flusight_trajectory_samples(
            [_calibration("United_States", traj)], _flusight(), pd.DataFrame()
        )
        assert _concat(rows).output_type_id.nunique() == 90
        assert not _concat(rows).value.isna().any()
        assert warns

    def test_duplicate_location_keeps_first_with_warning(self):
        """Test that only the first model per location is sampled, with a warning."""
        # Repeated calibrations for one location must not create a second sample table with the same IDs.
        calibrations = [
            _calibration("United_States", _trajectories(200)),
            _calibration("United_States", _trajectories(200)),
        ]
        rows, warns = ts.build_flusight_trajectory_samples(calibrations, _flusight(), pd.DataFrame())
        assert len(rows) == 1
        assert any("more than one model" in w for w in warns)

    def test_transition_strategy_uses_transition_trajectories(self):
        """Test that the transition strategy samples the transition trajectories within [0, 1]."""
        # Disable hospitalization output so this exercises ED-only generation from ed_prop.
        flusight = _flusight(hospitalizations=None, prop_ed={"strategy": "transition", "transition_name": "ed_prop"})
        traj = _trajectories(300)
        rows, warns = ts.build_flusight_trajectory_samples(
            [_calibration("United_States", traj)], flusight, pd.DataFrame()
        )
        df = _concat(rows)

        assert warns == []
        assert set(df.target) == {"wk inc flu prop ed visits"}
        assert df.output_type_id.nunique() == 100
        assert df.value.between(0, 1).all()
        sample = df.query("output_type_id == 'US00'").sort_values("horizon").value.to_numpy()
        # Range checks alone are insufficient: the values must match a whole ED projection path.
        assert np.isclose(np.stack(traj["ed_prop"])[:, 3:8], sample).all(axis=1).any()

    @pytest.mark.parametrize("seed", [42, None])
    @pytest.mark.parametrize("strategy", ["surveillance_window", "calibration_window"])
    def test_window_strategies_scale_the_hosp_samples(self, strategy, seed, monkeypatch):
        """Test that window strategies scale the hosp samples by the rescaling factor, sharing ids."""
        # Both window strategies consume precomputed factors here; fitting the factor
        # is outside this function. These fields satisfy each strategy's config requirements.
        extra = (
            {"ed_source": "ed", "hosp_source": "hosp", "fit_start": "2026-09-01", "fit_end": "2026-10-01"}
            if strategy == "surveillance_window"
            else {"ed_source": "ed", "num_fit_weeks": 4}
        )
        flusight = _flusight(prop_ed={"strategy": strategy, **extra})
        traj = _trajectories(300)
        if seed is None:
            # Two separate unseeded selections would use seeds 1 and 2, yielding
            # different rows. This catches re-sampling ED instead of reusing hosp
            # without depending on a chance failure from system randomness.
            rng_factory = np.random.default_rng
            entropy = iter([1, 2])
            monkeypatch.setattr(np.random, "default_rng", lambda _seed: rng_factory(next(entropy)))
        factors = pd.DataFrame({"population": ["United_States"], "rescaling_factor": [2e-4]})
        rows, warns = ts.build_flusight_trajectory_samples(
            [_calibration("United_States", traj, seed=seed)], flusight, factors
        )
        df = _concat(rows)
        hosp = df[df.target == "wk inc flu hosp"].set_index(["output_type_id", "horizon"]).value
        ed = df[df.target == "wk inc flu prop ed visits"].set_index(["output_type_id", "horizon"]).value

        assert warns == []
        # Pair rows by sample ID and horizon. Undo ED scaling before rounding to
        # compare with the submitted integer counts: ED uses unrounded hosp values.
        assert hosp.index.equals(ed.index)
        assert np.allclose(hosp, np.rint(ed / 2e-4))

    def test_window_strategy_without_factor_skips_ed(self):
        """Test that ED samples are skipped with a warning when the location has no rescaling factor."""
        flusight = _flusight(prop_ed={"strategy": "calibration_window", "ed_source": "ed", "num_fit_weeks": 4})
        # An empty factor table prevents ED conversion, but hospitalization output must survive.
        rows, warns = ts.build_flusight_trajectory_samples(
            [_calibration("United_States", _trajectories(200))], flusight, pd.DataFrame()
        )
        assert set(_concat(rows).target) == {"wk inc flu hosp"}
        assert any("no prop ED rescaling factor" in w for w in warns)


def test_samples_rejected_for_metrocast():
    """Test that samples cannot be configured for metrocast outputs."""
    # An empty samples mapping enables the default sample settings, so the config must reject it.
    with pytest.raises(ValueError, match="metrocast"):
        FlusightForecastOutput(reference_date=REFERENCE_DATE, metrocast=True, samples={})


def test_unknown_sample_method_rejected():
    """Test that an unregistered sample selection method is rejected."""
    # Reject an unknown selector during configuration, before output generation can use it.
    with pytest.raises(ValueError, match="Unknown sample selection method"):
        FlusightForecastOutput(reference_date=REFERENCE_DATE, samples={"method": "nope"})


def test_ed_model_parquet_output_is_submission_ready():
    """Production ED model config (hospitalizations: null, transition ed_prop) with samples and Parquet output."""
    traj = _trajectories(300)
    calibration = _calibration("United_States", traj)
    # The dispatcher requires projection metadata as well as trajectory/quantile results.
    calibration.results.projections = {"baseline": [{"date": list(DATES)}]}
    quantiles = [0.025, 0.5, 0.975]
    # Provide quantiles over the same nine weeks as the trajectories; the submission
    # should keep only five horizons and include both quantile and sample rows.
    calibration.results.get_projection_quantiles.return_value = pd.DataFrame(
        {
            "date": np.tile(DATES, len(quantiles)),
            "quantile": np.repeat(quantiles, len(DATES)),
            "ed_prop": np.tile(np.linspace(0.01, 0.02, len(DATES)), len(quantiles)),
        }
    )
    config = OutputConfig(
        output=OutputConfiguration(
            tabular_output_types=[
                TabularOutputTypeEnum.Parquet,
                TabularOutputTypeEnum.CSVBytes,
                TabularOutputTypeEnum.DataFrame,
            ],
            flusight_format={
                "reference_date": REFERENCE_DATE,
                "hospitalizations": None,
                "prop_ed": {"strategy": "transition", "transition_name": "ed_prop"},
                "quantiles": quantiles,
                "samples": {"n_samples": 100},
            },
            model_meta=ModelMetaOutput(projection_parameters=False),
        )
    )

    outputs = generate_calibration_outputs(calibrations=[calibration], output_config=config)

    hub, csv, dataframe = outputs["output_hub_formatted"]
    assert hub.name == "output_hub_formatted.parquet"
    table = pq.read_table(io.BytesIO(hub.data))
    # Check the actual file schema produced through the dispatcher and shared writer.
    assert [f.type for f in table.schema] == [
        "string",
        "int32",
        "string",
        "string",
        "string",
        "string",
        "string",
        "double",
    ]
    df = table.to_pandas()
    # Three quantile levels x five horizons = 15 rows; 100 samples x five horizons = 500.
    assert df.groupby("output_type").size().to_dict() == {"quantile": 15, "sample": 500}
    assert set(df.target) == {"wk inc flu prop ed visits"}
    assert df.query("output_type == 'quantile'").output_type_id.unique().tolist() == ["0.025", "0.5", "0.975"]

    # Requesting Parquet first must not normalize the later CSV/DataFrame outputs.
    # Preserve their existing column order, Python dates, integer horizons, and numeric quantile IDs.
    legacy = dataframe.data
    assert legacy.columns.tolist() == [
        "location",
        "reference_date",
        "horizon",
        "target_end_date",
        "target",
        "output_type",
        "output_type_id",
        "value",
    ]
    assert legacy.reference_date.iloc[0] == REFERENCE_DATE
    assert legacy.horizon.dtype == np.dtype("int64")
    assert legacy.output_type_id.iloc[0] == 0.025
    # Compare decompressed CSV text so gzip metadata does not affect the compatibility check.
    assert gzip.decompress(csv.data).decode() == legacy.to_csv(index=False, date_format="%Y-%m-%d")
