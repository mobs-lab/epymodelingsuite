"""Long dashboard contracts shared by direct and aggregated output."""

import io

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from epymodelingsuite.output.dashboard_trajectories import (
    DASHBOARD_TRAJECTORY_SCHEMA,
    dashboard_trajectories_from_wide,
    format_dashboard_trajectories,
    validate_dashboard_trajectories,
)

REFERENCE = "2026-10-10"
TARGET = "wk inc flu hosp"


def _source(strains=("single",)):
    frame = pd.DataFrame({"sample_id": [0, 0, 1, 1]})
    for strain in strains:
        frame[f"sim_id_{strain}"] = [10, 10, 20, 20]
    frame["location"] = "United_States"
    frame["date"] = pd.to_datetime(["2026-10-03", REFERENCE] * 2).as_unit("ms")
    for i, strain in enumerate(strains):
        frame[f"target_{strain}"] = [np.nan, 2.0 + i, 4.0 + i, 6.0 + i]
    frame["target_total"] = frame[[f"target_{s}" for s in strains]].sum(axis=1, min_count=1)
    return frame


def _formatted():
    frame = dashboard_trajectories_from_wide(_source(), ["single"], REFERENCE, TARGET)
    native = pa.Table.from_pandas(frame).to_pandas(ignore_metadata=True)
    native["sim_id"] = native.sim_id.astype("Int64")
    return native


@pytest.mark.parametrize("strains", [("single",), ("h1", "h3", "b")])
@pytest.mark.parametrize("baselines", [(), ("target_baseline",), ("target_baseline_k5", "target_baseline_k10")])
def test_format_roundtrip_schema_and_values(strains, baselines):
    source = _source(strains)
    for column in baselines:
        source[column] = source.target_total + 0.5
    original = source.copy(deep=True)
    formatted = dashboard_trajectories_from_wide(source, strains, REFERENCE, TARGET)
    pd.testing.assert_frame_equal(source, original)
    table = pq.read_table(io.BytesIO(formatted.to_parquet(index=False)))
    assert table.column_names == DASHBOARD_TRAJECTORY_SCHEMA.names
    assert [f.type for f in table.schema] == [f.type for f in DASHBOARD_TRAJECTORY_SCHEMA]
    assert validate_dashboard_trajectories(table.to_pandas(), strains) == []
    assert len(formatted) == len(source) * (len(strains) + 1 + len(baselines))
    assert formatted.target.eq(TARGET).all()
    total = formatted.loc[formatted.series.eq("total")]
    np.testing.assert_array_equal(total.value.to_numpy(dtype=float, na_value=np.nan), source.target_total.to_numpy())
    assert total.sim_id.isna().all()
    assert total.horizon.tolist() == [-1, 0, -1, 0]
    for strain in strains:
        rows = formatted.loc[formatted.series.eq(strain)]
        assert rows.sim_id.tolist() == source[f"sim_id_{strain}"].tolist()
        assert rows.value.isna().tolist() == [True, False, False, False]


def test_direct_long_producer_needs_no_aggregation():
    direct = _formatted().drop(columns=["reference_date", "horizon", "epiweek"])
    original = direct.copy(deep=True)
    result = format_dashboard_trajectories(direct, ["single"], REFERENCE)
    pd.testing.assert_frame_equal(direct, original)
    assert validate_dashboard_trajectories(result, ["single"]) == []


@pytest.mark.parametrize(
    ("series", "column", "value", "message"),
    [
        ("total", "value", 99.0, "sum of available strain"),
        ("single", "value", np.inf, "finite"),
        ("single", "value", -1.0, "non-negative"),
        (None, "location", "", "blank"),
        (None, "location", None, "missing"),
        (None, "target", "", "blank"),
        (None, "horizon", 9, "horizon"),
        (None, "epiweek", "190001", "epiweek"),
        (None, "reference_date", pd.Timestamp("2026-10-17"), "single value"),
        (None, "date", pd.Timestamp("2026-10-04"), "Saturday"),
        (None, "date", pd.Timestamp("2026-10-03 12:00"), "midnight"),
        (None, "date", pd.NaT, "missing"),
        ("single", "sim_id", 21, "stay constant"),
        ("single", "sim_id", pd.NA, "require sim_id"),
        ("total", "sim_id", 10, "null sim_id"),
        (None, "sample_id", -1, "non-negative"),
    ],
)
def test_validator_rejects_bad_data(series, column, value, message):
    frame = _formatted()
    mask = frame.sample_id.eq(0) & frame.date.eq(pd.Timestamp("2026-10-03"))
    if series is not None:
        mask &= frame.series.eq(series)
    frame.loc[mask, column] = value
    assert any(message in error for error in validate_dashboard_trajectories(frame, ["single"]))


def test_validator_rejects_duplicates_missing_series_and_noncontiguous_samples():
    frame = _formatted()
    assert any("Duplicate" in e for e in validate_dashboard_trajectories(pd.concat([frame, frame]), ["single"]))
    assert any("all series" in e for e in validate_dashboard_trajectories(frame.drop(index=0), ["single"]))
    frame["sample_id"] += 1
    assert any("0 .. n_samples" in e for e in validate_dashboard_trajectories(frame, ["single"]))


def test_validator_rejects_schema_errors():
    frame = _formatted()
    for invalid in [frame.drop(columns="target"), frame.assign(extra=1), frame[frame.columns[::-1]]]:
        assert "Columns" in validate_dashboard_trajectories(invalid, ["single"])[0]
    assert validate_dashboard_trajectories(frame.iloc[:0], ["single"]) == ["Table has no rows."]
    frame["date"] = frame.date.dt.as_unit("ms")
    assert any("timestamp[us]" in e for e in validate_dashboard_trajectories(frame, ["single"]))


@pytest.mark.parametrize("strains", [[], ["single", "h1"], ["h1", "h1"], ["total"], ["baseline"], [""]])
def test_rejects_ambiguous_labels(strains):
    with pytest.raises(ValueError, match="strain labels"):
        dashboard_trajectories_from_wide(_source(), strains, REFERENCE, TARGET)


def test_rejects_bad_baselines():
    for columns in [("target_baseline", "target_baseline_k5"), ("target_baseline_unknown",)]:
        source = _source()
        for column in columns:
            source[column] = source.target_total + 1
        with pytest.raises(ValueError, match="Baseline"):
            dashboard_trajectories_from_wide(source, ["single"], REFERENCE, TARGET)
    source = _source()
    source["target_baseline"] = source.target_total * 0.9
    with pytest.raises(ValueError, match="at least total"):
        dashboard_trajectories_from_wide(source, ["single"], REFERENCE, TARGET)


def test_format_does_not_drop_columns_or_truncate_ids():
    with pytest.raises(ValueError, match="Columns"):
        dashboard_trajectories_from_wide(_source().assign(extra=1), ["single"], REFERENCE, TARGET)
    source = _source()
    source["sim_id_single"] = source.sim_id_single + 0.5
    with pytest.raises(ValueError, match="Cannot cast"):
        dashboard_trajectories_from_wide(source, ["single"], REFERENCE, TARGET)


def test_format_rejects_timezones_and_fractional_microseconds():
    source = _source()
    source["date"] = source.date.dt.tz_localize("UTC")
    with pytest.raises(ValueError, match="timezone-naive"):
        dashboard_trajectories_from_wide(source, ["single"], REFERENCE, TARGET)
    source = _source()
    source["date"] = source.date.dt.as_unit("ns") + pd.Timedelta(nanoseconds=1)
    with pytest.raises(ValueError, match="Cannot cast"):
        dashboard_trajectories_from_wide(source, ["single"], REFERENCE, TARGET)


def test_validator_rejects_arrow_nan_without_null_bitmap():
    frame = dashboard_trajectories_from_wide(_source(), ["single"], REFERENCE, TARGET)
    values = frame.value.to_numpy(dtype=float, na_value=np.nan).copy()
    values[frame.series.eq("total").to_numpy(dtype=bool)] = np.nan
    frame["value"] = pd.Series(pa.array(values), dtype=pd.ArrowDtype(pa.float64()))
    assert any("sum of available strain" in e for e in validate_dashboard_trajectories(frame, ["single"]))


def test_targets_have_independent_values_and_simulation_ids():
    hosp = _formatted()
    ed = hosp.copy()
    ed["target"] = "wk inc flu prop ed visits"
    ed["value"] /= 100
    ed["sim_id"] += 100
    combined = pd.concat([hosp, ed], ignore_index=True)
    assert validate_dashboard_trajectories(combined, ["single"]) == []


def test_each_target_can_have_its_own_baseline_series():
    hosp_source = _source()
    hosp_source["target_baseline_k10"] = hosp_source.target_total + 1
    hosp = dashboard_trajectories_from_wide(hosp_source, ["single"], REFERENCE, TARGET)
    ed_source = _source()
    ed_source["target_baseline"] = ed_source.target_total + 0.1
    ed = dashboard_trajectories_from_wide(ed_source, ["single"], REFERENCE, "wk inc flu prop ed visits")
    assert validate_dashboard_trajectories(pd.concat([hosp, ed], ignore_index=True), ["single"]) == []


@pytest.mark.parametrize(("position", "value"), [(0, 0.0), (1, np.nan)])
def test_total_missing_exactly_when_all_strains_missing(position, value):
    frame = _formatted()
    index = frame.index[frame.series.eq("total")][position]
    frame.loc[index, "value"] = value
    assert any("missing iff all strains" in error for error in validate_dashboard_trajectories(frame, ["single"]))


@pytest.mark.parametrize("baseline", ["baseline", "baseline_k10"])
@pytest.mark.parametrize(("position", "value"), [(0, 0.0), (1, np.nan)])
def test_baseline_missingness_must_match_total(baseline, position, value):
    source = _source()
    source[f"target_{baseline}"] = source.target_total + 1
    frame = dashboard_trajectories_from_wide(source, ["single"], REFERENCE, TARGET)
    index = frame.index[frame.series.eq(baseline)][position]
    frame.loc[index, "value"] = value
    assert any("missing exactly where total" in error for error in validate_dashboard_trajectories(frame, ["single"]))


def test_all_nan_values_are_valid():
    frame = dashboard_trajectories_from_wide(_source(), ["single"], REFERENCE, TARGET)
    frame["value"] = pd.Series(pa.array([np.nan] * len(frame)), dtype=pd.ArrowDtype(pa.float64()))
    assert validate_dashboard_trajectories(frame, ["single"]) == []
