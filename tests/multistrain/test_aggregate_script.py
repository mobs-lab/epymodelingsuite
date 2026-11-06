"""Check the aggregate CLI's serialized dates for single and multiple strains."""

import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from epiweeks import Week

from epymodelingsuite.multistrain import aggregator
from epymodelingsuite.output.long_form_trajectories import (
    LONG_FORM_TRAJECTORY_SCHEMA,
    long_form_trajectories_from_wide,
    validate_long_form_trajectories,
)
from epymodelingsuite.schema.aggregation import AggregationConfiguration, SourceExperiment


@pytest.mark.parametrize("unit", ["ms", "us", "ns", "invalid"])
@pytest.mark.parametrize("labels", [("single",), ("h1", "h3", "b")])
@pytest.mark.parametrize("with_ages", [False, True])
def test_aggregate_dates_are_microseconds(tmp_path, monkeypatch, unit, labels, with_ages):
    reference = pd.Timestamp(Week.fromstring("202640").enddate())
    dates = pd.date_range(reference - pd.Timedelta(weeks=1), periods=2, freq="7D").as_unit(
        "ms" if unit == "invalid" else unit
    )
    trajectories = {
        label: pd.DataFrame(
            {
                "sim_id": [0, 0],
                "population": ["United_States"] * 2,
                "date": dates,
                "hospitalizations": [float("nan"), -2.0 if unit == "invalid" else 2.0],
            }
        )
        for label in labels
    }
    age_columns = {"0-4": "young", "5+": "older"} if with_ages else {}
    if with_ages:
        for frame in trajectories.values():
            frame["young"] = frame.hospitalizations * 0.25
            frame["older"] = frame.hospitalizations * 0.75
    config = tmp_path / "example.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "aggregation": {
                    "bucket": "gs://unused",
                    "submission_week": "202640",
                    "target": "wk inc flu hosp",
                    "random_seed": 1337,
                    "sources": [
                        {
                            "strain": label,
                            "experiment": "unused",
                            "trajectory_file": "unused.csv",
                            **({"age_group_columns": age_columns} if with_ages else {}),
                        }
                        for label in labels
                    ],
                    "sampling": {"method": "random", "n_samples": 1},
                    "aggregate_method": "sum",
                }
            }
        )
    )
    sources = [
        {"target": "wk inc flu hosp", "strain": label, "experiment": "example", "run_id": "20261007-182529-061e374e"}
        for label in labels
    ]
    monkeypatch.setattr(aggregator, "pull_trajectory_projections", lambda *_: (trajectories, sources))
    monkeypatch.setattr(sys, "argv", ["aggregate.py", "--config", str(config), "--output", str(tmp_path)])
    if unit == "invalid":
        with pytest.raises(ValueError, match="non-negative"):
            runpy.run_path(
                str(Path(__file__).resolve().parents[2] / "scripts/postprocess/aggregate.py"), run_name="__main__"
            )
        assert not list(tmp_path.glob("*.parquet"))
        return
    runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/postprocess/aggregate.py"), run_name="__main__")

    path = tmp_path / "long_form_trajectories_example.parquet"
    schema = pq.read_schema(path)
    assert schema.metadata[b"schema_version"] == b"1"
    assert json.loads(schema.metadata[b"sources"]) == sources
    assert schema.field("date").type == pa.timestamp("us")
    assert schema.field("reference_date").type == pa.timestamp("us")
    assert [f.type for f in schema] == [f.type for f in LONG_FORM_TRAJECTORY_SCHEMA]
    result = pd.read_parquet(path)
    assert validate_long_form_trajectories(result, labels) == []
    assert len(result) == len(dates) * (len(labels) + 1) * (len(age_columns) + 1)
    assert result.target.eq("wk inc flu hosp").all()
    assert set(result.series) == {*labels, "total"}
    assert set(result.age_group) == {"all", *age_columns}
    for age, fraction in {"all": 1.0, **({"0-4": 0.25, "5+": 0.75} if with_ages else {})}.items():
        age_total = result.loc[result.series.eq("total") & result.age_group.eq(age)]
        np.testing.assert_array_equal(age_total.value.to_numpy(), [np.nan, 2.0 * len(labels) * fraction])
    total = result.loc[result.series.eq("total") & result.age_group.eq("all")]
    assert total.date.tolist() == dates.tolist()
    assert result.reference_date.eq(reference).all()
    assert total.horizon.tolist() == [-1, 0]
    assert total.epiweek.tolist() == [Week.fromdate(date).cdcformat() for date in dates]
    assert pd.isna(total.value.iloc[0])
    assert total.value.iloc[1] == 2.0 * len(labels)
    assert total.sim_id.isna().all()


def test_sum_preserves_all_missing_and_real_zero():
    labels = ["h1", "h3", "b"]
    config = AggregationConfiguration(
        bucket="gs://unused",
        submission_week="202640",
        target="wk inc flu hosp",
        sources=[{"strain": s, "experiment": "unused", "trajectory_file": "unused.csv"} for s in labels],
        sampling={"method": "random", "n_samples": 4},
        aggregate_method="sum",
    )
    merged = pd.DataFrame(
        {
            "sample_id": range(4),
            "location": "United_States",
            "date": pd.Timestamp("2026-10-10"),
            "target_h1": [10.0, 10.0, np.nan, 0.0],
            "target_h3": [20.0, np.nan, np.nan, np.nan],
            "target_b": [30.0, 30.0, np.nan, np.nan],
        }
    )
    for label in labels:
        merged[f"sim_id_{label}"] = range(4)
    original = merged.copy(deep=True)
    result = aggregator.dispatch_strain_aggregator(merged, config)
    pd.testing.assert_frame_equal(merged, original)
    np.testing.assert_array_equal(result.target_total.to_numpy(), [60.0, 40.0, np.nan, 0.0])
    trajectories = long_form_trajectories_from_wide(result, labels, "2026-10-10", config.target)
    assert validate_long_form_trajectories(trajectories, labels) == []


@pytest.mark.parametrize("with_ages", [False, True])
def test_age_groups_use_the_same_sampled_trajectories(with_ages):
    config = AggregationConfiguration(
        bucket="gs://unused",
        submission_week="202640",
        target="hospitalizations",
        random_seed=42,
        sources=[
            {
                "strain": label,
                "experiment": "unused",
                "trajectory_file": "unused.csv",
                "target_column": f"{label}_all",
                "age_group_columns": {"0-4": f"{label}_young", "5+": f"{label}_older"} if with_ages else {},
            }
            for label in ("a", "b")
        ],
        sampling={"method": "random", "n_samples": 2},
        aggregate_method="sum",
    )
    trajectories = {}
    for label, offset in [("a", 10), ("b", 20)]:
        ids = np.arange(offset, offset + 4)
        trajectories[label] = pd.DataFrame(
            {
                "sim_id": ids,
                "population": "United_States",
                "date": pd.Timestamp("2026-10-10"),
                f"{label}_all": ids * 4.0,
                f"{label}_young": ids * 1.0,
                f"{label}_older": ids * 3.0,
            }
        )
    originals = {label: frame.copy(deep=True) for label, frame in trajectories.items()}
    mapping = aggregator.dispatch_strain_sampler(trajectories, config)
    merged = aggregator.merge_strain_trajectories(trajectories, mapping, config)
    aggregated = aggregator.dispatch_strain_aggregator(merged, config)
    ages = {"all": 4.0, **({"0-4": 1.0, "5+": 3.0} if with_ages else {})}
    assert len(aggregated) == 2 * len(ages)
    for age, multiplier in ages.items():
        rows = aggregated.loc[aggregated.age_group.eq(age)].sort_values("sample_id")
        for label in ("a", "b"):
            np.testing.assert_array_equal(rows[f"sim_id_{label}"], mapping[f"sim_id_{label}"])
            np.testing.assert_array_equal(rows[f"target_{label}"], mapping[f"sim_id_{label}"] * multiplier)
        np.testing.assert_array_equal(rows.target_total, (mapping.sim_id_a + mapping.sim_id_b) * multiplier)
    for label, original in originals.items():
        pd.testing.assert_frame_equal(trajectories[label], original)

    trajectories["b"]["date"] += pd.Timedelta(weeks=1)
    merged = aggregator.merge_strain_trajectories(trajectories, mapping, config)
    aggregated = aggregator.dispatch_strain_aggregator(merged, config)
    assert len(aggregated) == 4 * len(ages)
    assert aggregated.loc[aggregated.date.eq(pd.Timestamp("2026-10-10")), "target_b"].isna().all()
    assert aggregated.loc[aggregated.date.eq(pd.Timestamp("2026-10-17")), "target_a"].isna().all()
    long_form = long_form_trajectories_from_wide(aggregated, ["a", "b"], "2026-10-10", config.target)
    assert validate_long_form_trajectories(long_form, ["a", "b"]) == []

    missing_column = "b_young" if with_ages else "b_all"
    trajectories["b"] = trajectories["b"].drop(columns=missing_column)
    with pytest.raises(ValueError, match=f"Source 'b'.*{missing_column}"):
        aggregator.merge_strain_trajectories(trajectories, mapping, config)


@pytest.mark.parametrize(
    "columns",
    [
        {"all": "young"},
        {" ": "young"},
        {"0-4": " "},
        {"0-4": "hospitalizations"},
        {"0-4": "same", "5+": "same"},
    ],
)
def test_invalid_age_column_mapping(columns):
    with pytest.raises(ValueError, match="age_group_columns"):
        SourceExperiment(strain="single", experiment="unused", trajectory_file="unused.csv", age_group_columns=columns)


@pytest.mark.parametrize("incompatible", ["different_labels", "missing_labels", "baseline"])
def test_reject_incompatible_age_configuration(incompatible):
    source = {
        "strain": "a",
        "experiment": "unused",
        "trajectory_file": "unused.csv",
        "age_group_columns": {"0-4": "young"},
    }
    sources = [source]
    extra = {}
    if incompatible == "baseline":
        extra["baseline"] = {"method": "negative_binomial", "observed_means": "unused.csv", "dispersion_values": 10}
    else:
        sources.append(
            {
                **source,
                "strain": "b",
                "age_group_columns": {"5+": "older"} if incompatible == "different_labels" else {},
            }
        )
    with pytest.raises(ValueError, match="age_group_columns"):
        AggregationConfiguration(
            bucket="gs://unused",
            submission_week="202640",
            target="hospitalizations",
            sources=sources,
            sampling={"method": "random", "n_samples": 1},
            aggregate_method="sum",
            **extra,
        )
