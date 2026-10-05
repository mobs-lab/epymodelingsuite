"""Tests for epymodelingsuite.output.trajectory_samples."""

import gzip
import io
from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from epydemix.calibration import CalibrationResults

from epymodelingsuite.dispatcher.output import generate_calibration_outputs
from epymodelingsuite.output import trajectory_samples as ts
from epymodelingsuite.schema.dispatcher import CalibrationOutput
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


class TestSelectRandomIndices:
    """Tests specific to the random trajectory sampler."""

    def test_selects_n_distinct_rows(self):
        """Test that n distinct trajectories are selected."""
        # Requesting 100 of 1,000 forces subsampling; unique indices rule out replacement.
        values = np.arange(1000 * 5, dtype=float).reshape(1000, 5)
        idx = ts.select_random_indices(values, 100, np.random.default_rng(1))
        assert len(idx) == 100
        assert len(set(idx)) == 100
        assert ((0 <= idx) & (idx < len(values))).all()

    def test_same_seed_same_selection(self):
        """Test that the same seed gives the same selection."""
        values = np.random.default_rng(0).random((500, 5))
        # Hold the input fixed and verify that the selection seed reproduces the same indices.
        assert (
            ts.select_random_indices(values, 100, np.random.default_rng(7))
            == ts.select_random_indices(values, 100, np.random.default_rng(7))
        ).all()

    def test_different_seeds_select_different_rows(self):
        """Test that the random sampler uses the supplied generator to select rows."""
        values = np.ones((500, 5))
        # Fixed seeds make the check repeatable and reject always selecting the first rows.
        # Compare sets so changing only the order of the same rows cannot satisfy this check.
        first = ts.select_random_indices(values, 100, np.random.default_rng(7))
        second = ts.select_random_indices(values, 100, np.random.default_rng(8))
        assert set(first) != set(second)

    def test_fewer_than_n_returns_all_without_resampling(self):
        """Test that all trajectories are returned, without resampling, when fewer than n exist."""
        values = np.ones((60, 5))
        # Each original index should appear once; the result must not be padded to 100.
        assert ts.select_random_indices(values, 100, np.random.default_rng(1)).tolist() == list(range(60))


class TestSelectTrajectoryIndices:
    """Tests for filtering and dispatch shared by trajectory samplers."""

    def test_trajectories_with_nan_are_never_selected(self, monkeypatch):
        """Test that trajectories with NaN are never selected."""
        # Use a deterministic selector to exercise shared filtering independently of randomness.
        monkeypatch.setitem(ts.SAMPLE_SELECTORS, "first", lambda values, n, rng: np.arange(min(n, len(values))))
        values = np.ones((150, 5))
        # A single missing horizon invalidates the whole trajectory, leaving 75 complete rows.
        values[::2, 3] = np.nan
        idx = ts.select_trajectory_indices(values, 100, method="first")
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
            np.ones((3, 5)), HORIZONS, date(2026, 10, 10), "25", "wk inc flu hosp", "MA"
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
        # Cover negative clipping and rounding down/up. NumPy rint, like pandas round,
        # rounds exact .5 ties to the nearest even integer (10.5 -> 10).
        values = np.array([[-2.0, 0.4, 1.6, 10.5, 3.2]])
        rows = ts.trajectories_to_sample_rows(values, HORIZONS, date(2026, 10, 10), "US", "wk inc flu hosp", "US")
        assert rows.value.tolist() == [0.0, 0.0, 2.0, 10.0, 3.0]

    def test_proportions_are_clipped_to_unit_interval(self):
        """Test that proportion values are clipped to [0, 1]."""
        # Include values outside both bounds and valid fractions that must remain unchanged.
        values = np.array([[-0.1, 0.2, 1.3, 0.5, 1.0]])
        rows = ts.trajectories_to_sample_rows(
            values, HORIZONS, date(2026, 10, 10), "US", "wk inc flu prop ed visits", "US"
        )
        assert rows.value.tolist() == [0.0, 0.2, 1.0, 0.5, 1.0]

    @pytest.mark.parametrize("target", ["Flu ED visits pct", "ILI ED visits pct"])
    def test_percentage_targets_are_clipped_without_unit_conversion(self, target):
        """Percentage targets retain decimals and values above 1, clipping only outside [0, 100]."""
        values = np.array([[-2.0, 2.5, 12.5, 120.0, 100.0]])
        # The target alone determines the bounds; no hub flag or clipping arguments are passed.
        rows = ts.trajectories_to_sample_rows(values, HORIZONS, REFERENCE_DATE, "denver", target, None)
        assert rows.value.tolist() == [0.0, 2.5, 12.5, 100.0, 100.0]

    def test_custom_target_values_are_unchanged(self):
        """Custom targets preserve signed values without assuming bounds or integer counts."""
        values = np.array([[-2.0, 0.5, 2.5, 12.5, 120.0]])
        rows = ts.trajectories_to_sample_rows(values, HORIZONS, REFERENCE_DATE, "US", "custom_target", "US")
        assert rows.value.tolist() == [-2.0, 0.5, 2.5, 12.5, 120.0]


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

    def test_different_calibration_seeds_change_random_selection(self):
        """Test that calibration seeds affect the submitted samples when using random selection."""
        traj = _trajectories(500)
        flusight = FlusightForecastOutput(reference_date=REFERENCE_DATE, samples={"n_samples": 100, "method": "random"})
        # Keep the input paths and config fixed, with no output-level seed override.
        # Different calibration seeds must change selection for the random method.
        a, _ = ts.build_flusight_trajectory_samples(
            [_calibration("United_States", traj, seed=7)], flusight, pd.DataFrame()
        )
        b, _ = ts.build_flusight_trajectory_samples(
            [_calibration("United_States", traj, seed=8)], flusight, pd.DataFrame()
        )
        assert not _concat(a).equals(_concat(b))

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

    @pytest.mark.parametrize(
        "metrocast,target,expected",
        [
            (True, "wk inc flu prop ed visits", [0, 1, 1, 1, 1]),
            (False, "Flu ED visits pct", [0, 2.5, 12.5, 100, 100]),
        ],
    )
    def test_ed_clipping_depends_on_target_not_hub(self, metrocast, target, expected):
        """The configured target determines ED bounds even when hub defaults suggest other units."""
        flusight = _flusight(
            metrocast=metrocast,
            horizons=HORIZONS,
            hospitalizations=None,
            prop_ed={"target": target, "strategy": "transition", "transition_name": "ed_prop"},
        )
        traj = {
            "date": [DATES[3:8].to_numpy() for _ in range(100)],
            "ed_prop": list(np.tile([-2.0, 2.5, 12.5, 120.0, 100.0], (100, 1))),
        }
        rows, warns = ts.build_flusight_trajectory_samples(
            [_calibration("United_States", traj)], flusight, pd.DataFrame()
        )
        assert not warns
        assert _concat(rows).value.tolist() == expected * 100

    @pytest.mark.parametrize("strategy", ["transition", "surveillance_window", "calibration_window"])
    def test_metrocast_ed_samples_use_percentages_and_horizons_zero_to_three(self, strategy):
        """Metrocast preserves percentage units and indexes each location's trajectories from 1."""
        prop_ed = {"target": "Flu ED visits pct", "strategy": strategy}
        if strategy == "transition":
            prop_ed["transition_name"] = "ed_prop"
        elif strategy == "surveillance_window":
            prop_ed.update(ed_source="ed", hosp_source="hosp", fit_start=REFERENCE_DATE, fit_end=REFERENCE_DATE)
        else:
            prop_ed.update(ed_source="ed", num_fit_weeks=4)
        flusight = _flusight(metrocast=True, hospitalizations=None, prop_ed=prop_ed)
        calibrations = []
        for offset, location in enumerate(["denver", "colorado"]):
            # Two locations in the same state must retain their own values. Their
            # required horizon 0-3 paths have no preceding week, which is valid here.
            values = np.tile([-2.0, 2.5 + offset, 12.5 + offset, 120.0], (100, 1))
            traj = {
                "date": [DATES[4:8].to_numpy() for _ in range(100)],
                "ed_prop": list(values),
                "hospitalizations": list(values * 10),
            }
            calibrations.append(_calibration(f"metrocast_location_{location}", traj))
        # Window factors convert counts directly to percentage units, as in the
        # existing Metrocast quantile path; do not apply another factor of 100.
        factors = pd.DataFrame({"population": [c.population for c in calibrations], "rescaling_factor": [0.1, 0.1]})
        rows, warns = ts.build_flusight_trajectory_samples(calibrations, flusight, factors)
        assert not warns
        df = _concat(rows)
        assert set(df.target) == {"Flu ED visits pct"}
        assert len(df) == 2 * 100 * 4
        for offset, location in enumerate(["denver", "colorado"]):
            output = df[df.location == location]
            assert output.output_type_id.tolist() == np.repeat([str(i) for i in range(1, 101)], 4).tolist()
            assert output.horizon.tolist() == [0, 1, 2, 3] * 100
            assert output.target_end_date.tolist() == [d.date() for d in DATES[4:8]] * 100
            # Clip only outside [0, 100]; preserve decimals and percentages above 1.
            np.testing.assert_allclose(output.value, [0, 2.5 + offset, 12.5 + offset, 100] * 100)


def test_samples_enabled_for_metrocast():
    """Test that Metrocast accepts the default sample configuration."""
    # Metrocast 2026-27 requires 100 trajectory samples per location and target.
    config = FlusightForecastOutput(reference_date=REFERENCE_DATE, metrocast=True, samples={})
    assert config.samples.n_samples == 100
    assert config.samples.method == "random"


@pytest.mark.parametrize("strategy", ["surveillance_window", "calibration_window"])
def test_dispatcher_generates_window_ed_samples_without_hospitalization_output(tmp_path, strategy):
    """Fit ED factors and emit ED-only forecasts through the real dispatcher."""
    dates = pd.date_range(REFERENCE_DATE, periods=4, freq="7D").to_list()
    calibration = CalibrationOutput(
        primary_id=0,
        population="United_States",
        seed=42,
        delta_t=1.0,
        results=CalibrationResults(
            selected_trajectories={0: [{"date": dates, "data": np.full(4, 100.0)}]},
            projections={"baseline": [{"date": dates, "hospitalizations": np.full(4, 20.0)} for _ in range(100)]},
        ),
    )
    surveillance = {}
    for name, value in [("ed", 0.01), ("hosp", 100.0)]:
        path = tmp_path / f"{name}.csv"
        pd.DataFrame({"date": dates, "location": "US", "value": value}).to_csv(path, index=False)
        surveillance[name] = {
            "data_path": str(path),
            "value_column": "value",
            "date_column": "date",
            "location_column": "location",
            "location_format": "FIPS",
        }
    extra = (
        {"hosp_source": "hosp", "fit_start": dates[0], "fit_end": dates[-1]}
        if strategy == "surveillance_window"
        else {"num_fit_weeks": 4}
    )
    config = OutputConfig(
        output=OutputConfiguration(
            tabular_output_types=[TabularOutputTypeEnum.DataFrame],
            options={"surveillance": surveillance},
            flusight_format={
                "reference_date": REFERENCE_DATE,
                "horizons": [0, 1, 2, 3],
                "hospitalizations": None,
                "prop_ed": {"strategy": strategy, "ed_source": "ed", **extra},
                "samples": {},
            },
        )
    )

    outputs = generate_calibration_outputs(calibrations=[calibration], output_config=config)
    (hub,) = outputs["output_hub_formatted"]
    assert set(hub.data.target) == {"wk inc flu prop ed visits"}
    assert hub.data.groupby("output_type").size().to_dict() == {"quantile": 23 * 4, "sample": 100 * 4}
    # Both fitting strategies recover 0.01 / 100 and apply it to the projected 20 admissions.
    np.testing.assert_allclose(hub.data.value.to_numpy(dtype=float), 0.002)


def test_unknown_sample_method_rejected():
    """Test that an unregistered sample selection method is rejected."""
    # Reject an unknown selector during configuration, before output generation can use it.
    with pytest.raises(ValueError, match="Unknown sample selection method"):
        FlusightForecastOutput(reference_date=REFERENCE_DATE, samples={"method": "invalid_method"})


@pytest.mark.parametrize("horizons", [None, [0, 2, 4]])
@pytest.mark.parametrize("metrocast", [False, True])
def test_ed_model_parquet_output_is_submission_ready(metrocast, horizons):
    """ED model output preserves FluSight formats and emits only samples for Metrocast."""
    traj = _trajectories(300)
    population = "metrocast_location_denver" if metrocast else "United_States"
    target = "Flu ED visits pct" if metrocast else "wk inc flu prop ed visits"
    expected_horizons = horizons if horizons is not None else ([0, 1, 2, 3] if metrocast else HORIZONS)
    calibration = _calibration(population, traj)
    # The dispatcher requires projection metadata as well as trajectory/quantile results.
    calibration.results.projections = {"baseline": [{"date": list(DATES)}]}
    quantiles = [0.025, 0.5, 0.975]
    # Provide quantiles over the same nine weeks as the trajectories; the submission
    # should retain the hub's horizons. Metrocast must exclude quantile rows even
    # though they are available; its 2026-27 specification accepts samples only.
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
                "prop_ed": {"target": target, "strategy": "transition", "transition_name": "ed_prop"},
                "quantiles": quantiles,
                "samples": {"n_samples": 100},
                "metrocast": metrocast,
                "horizons": horizons,
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
    assert set(df.target) == {target}
    # Explicit horizons must reach both quantile and sample generation, including
    # horizon 4 outside either hub's defaults and the omission of horizons 1 and 3.
    for _, rows in df.groupby("output_type"):
        assert rows.horizon.unique().tolist() == expected_horizons
    if metrocast:
        # 100 draws over the configured horizons, without submitted quantiles.
        assert df.groupby("output_type").size().to_dict() == {"sample": 100 * len(expected_horizons)}
        assert set(df.location) == {"denver"}
        assert df.output_type_id.unique().tolist() == [str(i) for i in range(1, 101)]
    else:
        assert df.groupby("output_type").size().to_dict() == {
            "quantile": 3 * len(expected_horizons),
            "sample": 100 * len(expected_horizons),
        }
        assert df.query("output_type == 'quantile'").output_type_id.unique().tolist() == ["0.025", "0.5", "0.975"]

    # Requesting Parquet first must not normalize the later CSV/DataFrame outputs.
    # Preserve the original table's column order, dates and integer horizons. FluSight
    # also retains numeric quantile IDs; Metrocast has only string sample indexes.
    legacy = dataframe.data
    leading_columns = (
        ["reference_date", "horizon", "target_end_date", "location"]
        if metrocast
        else ["location", "reference_date", "horizon", "target_end_date"]
    )
    assert legacy.columns.tolist() == leading_columns + [
        "target",
        "output_type",
        "output_type_id",
        "value",
    ]
    assert legacy.reference_date.iloc[0] == REFERENCE_DATE
    assert legacy.horizon.dtype == np.dtype("int64")
    assert legacy.output_type_id.iloc[0] == ("1" if metrocast else 0.025)
    # Compare decompressed CSV text so gzip metadata does not affect the compatibility check.
    assert gzip.decompress(csv.data).decode() == legacy.to_csv(index=False, date_format="%Y-%m-%d")
