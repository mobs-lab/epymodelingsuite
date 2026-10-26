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
from epymodelingsuite.output.dashboard_trajectories import (
    DASHBOARD_TRAJECTORY_SCHEMA,
    dashboard_trajectories_from_wide,
    validate_dashboard_trajectories,
)
from epymodelingsuite.schema.aggregation import AggregationConfiguration


@pytest.mark.parametrize("unit", ["ms", "us", "ns", "invalid"])
@pytest.mark.parametrize("labels", [("single",), ("h1", "h3", "b")])
def test_aggregate_dates_are_microseconds(tmp_path, monkeypatch, unit, labels):
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
                        {"strain": label, "experiment": "unused", "trajectory_file": "unused.csv"} for label in labels
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

    path = tmp_path / "dashboard_trajectories_example.parquet"
    schema = pq.read_schema(path)
    assert schema.metadata[b"schema_version"] == b"1"
    assert json.loads(schema.metadata[b"sources"]) == sources
    assert schema.field("date").type == pa.timestamp("us")
    assert schema.field("reference_date").type == pa.timestamp("us")
    assert [f.type for f in schema] == [f.type for f in DASHBOARD_TRAJECTORY_SCHEMA]
    result = pd.read_parquet(path)
    assert validate_dashboard_trajectories(result, labels) == []
    assert len(result) == len(dates) * (len(labels) + 1)
    assert result.target.eq("wk inc flu hosp").all()
    assert set(result.series) == {*labels, "total"}
    total = result.loc[result.series.eq("total")]
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
    dashboard = dashboard_trajectories_from_wide(result, labels, "2026-10-10", config.target)
    assert validate_dashboard_trajectories(dashboard, labels) == []
