"""Long-form trajectory contracts shared by direct and aggregated output."""

import io
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from epymodelingsuite.output.long_form_trajectories import (
    LONG_FORM_TRAJECTORY_SCHEMA,
    format_long_form_trajectories,
    long_form_trajectories_from_wide,
    read_long_form_trajectories,
    validate_long_form_trajectories,
    write_long_form_trajectories,
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
    frame = long_form_trajectories_from_wide(_source(), ["single"], REFERENCE, TARGET)
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
    formatted = long_form_trajectories_from_wide(source, strains, REFERENCE, TARGET)
    pd.testing.assert_frame_equal(source, original)
    table = pq.read_table(io.BytesIO(formatted.to_parquet(index=False)))
    assert table.column_names == LONG_FORM_TRAJECTORY_SCHEMA.names
    assert [f.type for f in table.schema] == [f.type for f in LONG_FORM_TRAJECTORY_SCHEMA]
    assert validate_long_form_trajectories(table.to_pandas(), strains) == []
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
    result = format_long_form_trajectories(direct, ["single"], REFERENCE)
    pd.testing.assert_frame_equal(direct, original)
    assert validate_long_form_trajectories(result, ["single"]) == []


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
    assert any(message in error for error in validate_long_form_trajectories(frame, ["single"]))


def test_validator_rejects_duplicates_missing_series_and_noncontiguous_samples():
    frame = _formatted()
    assert any("Duplicate" in e for e in validate_long_form_trajectories(pd.concat([frame, frame]), ["single"]))
    assert any("all series" in e for e in validate_long_form_trajectories(frame.drop(index=0), ["single"]))
    frame["sample_id"] += 1
    assert any("0 .. n_samples" in e for e in validate_long_form_trajectories(frame, ["single"]))


def test_validator_rejects_schema_errors():
    frame = _formatted()
    for invalid in [frame.drop(columns="target"), frame.assign(extra=1), frame[frame.columns[::-1]]]:
        assert "Columns" in validate_long_form_trajectories(invalid, ["single"])[0]
    assert validate_long_form_trajectories(frame.iloc[:0], ["single"]) == ["Table has no rows."]
    frame["date"] = frame.date.dt.as_unit("ms")
    assert any("timestamp[us]" in e for e in validate_long_form_trajectories(frame, ["single"]))


@pytest.mark.parametrize("strains", [[], ["single", "h1"], ["h1", "h1"], ["total"], ["baseline"], [""]])
def test_rejects_ambiguous_labels(strains):
    with pytest.raises(ValueError, match="strain labels"):
        long_form_trajectories_from_wide(_source(), strains, REFERENCE, TARGET)


def test_rejects_bad_baselines():
    for columns in [("target_baseline", "target_baseline_k5"), ("target_baseline_unknown",)]:
        source = _source()
        for column in columns:
            source[column] = source.target_total + 1
        with pytest.raises(ValueError, match="Baseline"):
            long_form_trajectories_from_wide(source, ["single"], REFERENCE, TARGET)
    source = _source()
    source["target_baseline"] = source.target_total * 0.9
    with pytest.raises(ValueError, match="at least total"):
        long_form_trajectories_from_wide(source, ["single"], REFERENCE, TARGET)


def test_format_does_not_drop_columns_or_truncate_ids():
    with pytest.raises(ValueError, match="Columns"):
        long_form_trajectories_from_wide(_source().assign(extra=1), ["single"], REFERENCE, TARGET)
    source = _source()
    source["sim_id_single"] = source.sim_id_single + 0.5
    with pytest.raises(ValueError, match="Cannot cast"):
        long_form_trajectories_from_wide(source, ["single"], REFERENCE, TARGET)


def test_format_rejects_timezones_and_fractional_microseconds():
    source = _source()
    source["date"] = source.date.dt.tz_localize("UTC")
    with pytest.raises(ValueError, match="timezone-naive"):
        long_form_trajectories_from_wide(source, ["single"], REFERENCE, TARGET)
    source = _source()
    source["date"] = source.date.dt.as_unit("ns") + pd.Timedelta(nanoseconds=1)
    with pytest.raises(ValueError, match="Cannot cast"):
        long_form_trajectories_from_wide(source, ["single"], REFERENCE, TARGET)


def test_validator_rejects_arrow_nan_without_null_bitmap():
    frame = long_form_trajectories_from_wide(_source(), ["single"], REFERENCE, TARGET)
    values = frame.value.to_numpy(dtype=float, na_value=np.nan).copy()
    values[frame.series.eq("total").to_numpy(dtype=bool)] = np.nan
    frame["value"] = pd.Series(pa.array(values), dtype=pd.ArrowDtype(pa.float64()))
    assert any("sum of available strain" in e for e in validate_long_form_trajectories(frame, ["single"]))


def test_targets_share_simulation_ids_but_have_independent_values():
    hosp = _formatted()
    ed = hosp.copy()
    ed["target"] = "wk inc flu prop ed visits"
    ed["value"] /= 100
    combined = pd.concat([hosp, ed], ignore_index=True)
    assert validate_long_form_trajectories(combined, ["single"]) == []


def test_each_target_can_have_its_own_baseline_series():
    hosp_source = _source()
    hosp_source["target_baseline_k10"] = hosp_source.target_total + 1
    hosp = long_form_trajectories_from_wide(hosp_source, ["single"], REFERENCE, TARGET)
    ed_source = _source()
    ed_source["target_baseline"] = ed_source.target_total + 0.1
    ed = long_form_trajectories_from_wide(ed_source, ["single"], REFERENCE, "wk inc flu prop ed visits")
    assert validate_long_form_trajectories(pd.concat([hosp, ed], ignore_index=True), ["single"]) == []


@pytest.mark.parametrize(("position", "value"), [(0, 0.0), (1, np.nan)])
def test_total_missing_exactly_when_all_strains_missing(position, value):
    frame = _formatted()
    index = frame.index[frame.series.eq("total")][position]
    frame.loc[index, "value"] = value
    assert any("missing iff all strains" in error for error in validate_long_form_trajectories(frame, ["single"]))


@pytest.mark.parametrize("baseline", ["baseline", "baseline_k10"])
@pytest.mark.parametrize(("position", "value"), [(0, 0.0), (1, np.nan)])
def test_baseline_missingness_must_match_total(baseline, position, value):
    source = _source()
    source[f"target_{baseline}"] = source.target_total + 1
    frame = long_form_trajectories_from_wide(source, ["single"], REFERENCE, TARGET)
    index = frame.index[frame.series.eq(baseline)][position]
    frame.loc[index, "value"] = value
    assert any("missing exactly where total" in error for error in validate_long_form_trajectories(frame, ["single"]))


def test_all_nan_values_are_valid():
    frame = long_form_trajectories_from_wide(_source(), ["single"], REFERENCE, TARGET)
    frame["value"] = pd.Series(pa.array([np.nan] * len(frame)), dtype=pd.ArrowDtype(pa.float64()))
    assert validate_long_form_trajectories(frame, ["single"]) == []


@pytest.mark.parametrize(
    "sources",
    [
        None,
        [{"target": TARGET, "strain": "single", "experiment": "local-test"}],
        [
            {
                "target": TARGET,
                "strain": "single",
                "experiment": "202640/hosp",
                "run_id": "20261007-182529-061e374e",
                "trajectory_file": "gs://bucket/202640/hosp/20261007-182529-061e374e/outputs/20261013/data.csv.gz",
            }
        ],
    ],
)
def test_versioned_parquet_roundtrip(tmp_path, sources):
    frame = long_form_trajectories_from_wide(_source(), ["single"], REFERENCE, TARGET)
    path = tmp_path / "long_form_trajectories.parquet"
    write_long_form_trajectories(frame, path, sources=sources)
    schema = pq.read_schema(path)
    assert schema.metadata[b"schema_version"] == b"1"
    assert schema.names == LONG_FORM_TRAJECTORY_SCHEMA.names
    result = read_long_form_trajectories(path)
    pd.testing.assert_frame_equal(result, frame)
    assert result.attrs["schema_version"] == "1"
    assert result.attrs.get("sources") == sources
    if sources is None:
        assert b"sources" not in schema.metadata
    else:
        assert json.loads(schema.metadata[b"sources"]) == sources
    copy = tmp_path / "copy.parquet"
    write_long_form_trajectories(result, copy)
    assert read_long_form_trajectories(copy).attrs == result.attrs


@pytest.mark.parametrize("version", [None, b"2", b"999"])
def test_reader_rejects_missing_or_unknown_version(tmp_path, version):
    frame = long_form_trajectories_from_wide(_source(), ["single"], REFERENCE, TARGET)
    table = pa.Table.from_pandas(frame)
    metadata = dict(table.schema.metadata or {})
    if version is not None:
        metadata[b"schema_version"] = version
    path = tmp_path / "long_form_trajectories.parquet"
    pq.write_table(table.replace_schema_metadata(metadata), path)
    with pytest.raises(ValueError, match="schema_version"):
        read_long_form_trajectories(path)


@pytest.mark.parametrize(
    "sources",
    [
        {},
        [None],
        [{"target": TARGET}],
        [{"target": TARGET, "strain": "single", "run_id": "latest"}],
        [{"target": TARGET, "strain": "single", "run_id": "any"}],
        [{"target": TARGET, "strain": "single", "experiment": " "}],
        [{"target": TARGET, "strain": "single", "run_id": None}],
        [{"target": TARGET, "strain": "single", "extra": "ignored?"}],
        [{"target": TARGET, "strain": "total"}],
        [{"target": TARGET, "strain": "h1"}],
        [{"target": TARGET, "strain": "single"}] * 2,
    ],
)
def test_invalid_sources_rejected_before_write_and_after_read(tmp_path, sources):
    frame = long_form_trajectories_from_wide(_source(), ["single"], REFERENCE, TARGET)
    path = tmp_path / "long_form_trajectories.parquet"
    with pytest.raises(ValueError, match="[Ss]ource"):
        write_long_form_trajectories(frame, path, sources=sources)
    assert not path.exists()
    write_long_form_trajectories(frame, path)
    table = pq.read_table(path)
    metadata = {**table.schema.metadata, b"sources": json.dumps(sources).encode()}
    pq.write_table(table.replace_schema_metadata(metadata), path)
    with pytest.raises(ValueError, match="[Ss]ource"):
        read_long_form_trajectories(path)


def test_invalid_file_data_does_not_overwrite_existing_file(tmp_path):
    frame = _formatted()
    path = tmp_path / "long_form_trajectories.parquet"
    write_long_form_trajectories(frame, path)
    before = path.read_bytes()
    frame.loc[frame.series.eq("total"), "value"] = 999.0
    with pytest.raises(ValueError, match="sum of available"):
        write_long_form_trajectories(frame, path)
    assert path.read_bytes() == before


def test_reader_rejects_malformed_sources_json(tmp_path):
    path = tmp_path / "long_form_trajectories.parquet"
    write_long_form_trajectories(_formatted(), path)
    table = pq.read_table(path)
    table = table.replace_schema_metadata({**table.schema.metadata, b"sources": b"not JSON"})
    pq.write_table(table, path)
    with pytest.raises(ValueError, match="sources JSON"):
        read_long_form_trajectories(path)


def _age_rows():
    all_ages = _formatted().assign(target="hospitalizations")
    young = all_ages.assign(age_group="0-4", value=all_ages.value * 0.25)
    older = all_ages.assign(age_group="5+", value=all_ages.value * 0.75)
    return pd.concat([young, older, all_ages], ignore_index=True)


@pytest.mark.parametrize("include_all", [False, True])
def test_age_groups_and_model_targets_roundtrip(tmp_path, include_all):
    hosp = _age_rows()
    if not include_all:
        hosp = hosp.loc[hosp.age_group.ne("all")]
    ed = hosp.assign(target="ed_signal", value=hosp.value * 2.8)
    rows = pd.concat([hosp, ed], ignore_index=True)
    formatted = format_long_form_trajectories(rows, ["single"], REFERENCE)
    path = tmp_path / "ages.parquet"
    write_long_form_trajectories(formatted, path)
    result = read_long_form_trajectories(path)
    pd.testing.assert_frame_equal(result, formatted)
    assert set(result.age_group) == ({"0-4", "5+", "all"} if include_all else {"0-4", "5+"})
    assert set(result.target) == {"hospitalizations", "ed_signal"}


@pytest.mark.parametrize("value", [None, "", "   "])
def test_age_group_required_nonblank(value):
    rows = _formatted()
    rows["age_group"] = value
    with pytest.raises(ValueError, match="age_group"):
        format_long_form_trajectories(rows, ["single"], REFERENCE)


def test_direct_long_requires_age_group():
    with pytest.raises(ValueError, match="Columns"):
        format_long_form_trajectories(_formatted().drop(columns="age_group"), ["single"], REFERENCE)


def test_wide_adapter_preserves_age_groups_and_defaults_legacy_to_all():
    source = _source()
    assert long_form_trajectories_from_wide(source, ["single"], REFERENCE, TARGET).age_group.eq("all").all()
    ages = pd.concat([source.assign(age_group="0-4"), source.assign(age_group="5+")], ignore_index=True)
    result = long_form_trajectories_from_wide(ages, ["single"], REFERENCE, TARGET)
    assert set(result.age_group) == {"0-4", "5+"}
    assert len(result) == len(ages) * 2


def test_rejects_missing_age_rows_even_when_all_series_removed():
    rows = _age_rows()
    rows = rows.loc[~(rows.age_group.eq("0-4") & rows.sample_id.eq(0) & rows.horizon.eq(0))]
    assert any("same age groups" in e for e in validate_long_form_trajectories(rows, ["single"]))


@pytest.mark.parametrize("bad_value", [0.0, 99.0, np.nan])
def test_rejects_incorrect_all_age_sum(bad_value):
    rows = _age_rows()
    rows.loc[rows.age_group.eq("all") & rows.horizon.eq(0), "value"] = bad_value
    assert any("sum of all age groups" in e for e in validate_long_form_trajectories(rows, ["single"]))


def test_missing_age_cannot_be_treated_as_zero():
    rows = _age_rows()
    rows.loc[rows.age_group.eq("0-4"), "value"] = np.nan
    assert any("missing if any age" in e for e in validate_long_form_trajectories(rows, ["single"]))
    rows.loc[rows.age_group.eq("all"), "value"] = np.nan
    assert validate_long_form_trajectories(rows, ["single"]) == []


@pytest.mark.parametrize("dimension", ["age_group", "target"])
def test_rejects_different_source_ids_across_ages_and_targets(tmp_path, dimension):
    rows = _age_rows()
    if dimension == "target":
        ed = rows.assign(target="ed_signal", sim_id=rows.sim_id + 100)
        rows = pd.concat([rows, ed], ignore_index=True)
    else:
        rows.loc[rows.age_group.eq("0-4"), "sim_id"] += 100
    assert any("stay constant" in e for e in validate_long_form_trajectories(rows, ["single"]))
    with pytest.raises(ValueError, match="stay constant"):
        write_long_form_trajectories(rows, tmp_path / "bad.parquet")


def test_age_and_strain_sums_are_separate_dimensions(tmp_path):
    source = _source(("a", "b"))
    source["target_a"] = [np.nan, 0.0, 4.0, 6.0]
    source["target_b"] = [np.nan, np.nan, 5.0, 7.0]
    source["target_total"] = source[["target_a", "target_b"]].sum(axis=1, min_count=1)
    source["target_baseline_k10"] = source.target_total + 1
    all_ages = long_form_trajectories_from_wide(source, ["a", "b"], REFERENCE, "hospitalizations")
    young = all_ages.assign(age_group="0-4", value=all_ages.value * 0.25)
    older = all_ages.assign(age_group="5+", value=all_ages.value * 0.75)
    combined = format_long_form_trajectories(
        pd.concat([all_ages, young, older], ignore_index=True), ["a", "b"], REFERENCE
    )
    path = tmp_path / "multistrain_ages.parquet"
    write_long_form_trajectories(combined, path)
    restored = read_long_form_trajectories(path)
    pd.testing.assert_frame_equal(restored, combined)
    zeros = restored.loc[restored.sample_id.eq(0) & restored.horizon.eq(0) & restored.series.eq("total")]
    assert zeros.value.eq(0).all()
    assert len(zeros) == 3
    invalid = restored.copy()
    invalid.loc[invalid.age_group.eq("0-4") & invalid.series.eq("total") & invalid.horizon.eq(0), "value"] = 99.0
    assert any("sum of available strain" in e for e in validate_long_form_trajectories(invalid, ["a", "b"]))
