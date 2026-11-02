"""Resolve provenance from the concrete GCS object that is downloaded."""

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from epymodelingsuite.multistrain import aggregator
from epymodelingsuite.output.long_form_trajectories import read_long_form_trajectories
from epymodelingsuite.schema.aggregation import AggregationConfiguration


def _config(run_id, filename="trajectories.csv.gz"):
    return AggregationConfiguration(
        bucket="gs://bucket/pipeline",
        submission_week="202640",
        target="wk inc flu hosp",
        sources=[
            {
                "strain": "single",
                "experiment": "202640/hosp",
                "run_id": run_id,
                "trajectory_file": filename,
            }
        ],
        sampling={"method": "random", "n_samples": 1},
        aggregate_method="sum",
    )


@pytest.mark.parametrize("selector", ["latest", "any", "20261007-182529-061e374e"])
@pytest.mark.parametrize("filename", ["trajectories.csv.gz", "trajectories.parquet"])
def test_download_records_resolved_run_and_file(tmp_path, monkeypatch, selector, filename):
    run_id = "20261007-182529-061e374e"
    prefix = "gs://bucket/pipeline/202640/hosp/"
    source_path = f"{prefix}{run_id}/outputs/20261013-224354/{filename}"
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[2] == "cp":
            import gzip

            if filename.endswith(".parquet"):
                pd.DataFrame(
                    {"sim_id": [0], "population": ["United_States"], "date": ["2026-10-10"], "hospitalizations": [2.0]}
                ).to_parquet(command[-1])
            else:
                with gzip.open(command[-1], "wt") as output:
                    output.write("sim_id,population,date,hospitalizations\n0,United_States,2026-10-10,2.0\n")
            return SimpleNamespace(returncode=0)
        if command[-1] == prefix.rstrip("/"):
            # Listing order must not determine which run is latest.
            stdout = f"{prefix}{run_id}/\n{prefix}20261001-182529-061e374e/\n"
        else:
            stdout = source_path + "\n"
        return SimpleNamespace(returncode=0, stdout=stdout)

    monkeypatch.setattr(aggregator.subprocess, "run", run)
    trajectories, sources = aggregator.pull_trajectory_projections(_config(selector, filename), str(tmp_path))
    assert trajectories["single"].hospitalizations.tolist() == [2.0]
    assert trajectories["single"].date.tolist() == [pd.Timestamp("2026-10-10")]
    assert sources == [
        {
            "target": "wk inc flu hosp",
            "strain": "single",
            "experiment": "202640/hosp",
            "run_id": run_id,
            "trajectory_file": source_path,
        }
    ]
    assert calls[-1] == ["gcloud", "storage", "cp", source_path, str(tmp_path / f"single_{filename}")]
    pattern = calls[-2][-1]
    assert f"/{'*' if selector == 'any' else run_id}/outputs/" in pattern


@pytest.mark.parametrize("stdout,code", [("", 0), ("gs://a\ngs://b\n", 0), ("", 1)])
def test_ambiguous_missing_or_failed_source_is_not_downloaded(tmp_path, monkeypatch, stdout, code):
    def run(command, **kwargs):
        assert command[2] == "ls"
        return SimpleNamespace(returncode=code, stdout=stdout, stderr="listing failed")

    monkeypatch.setattr(aggregator.subprocess, "run", run)
    with pytest.raises(ValueError, match="exactly one|Failed to resolve"):
        aggregator.pull_trajectory_projections(_config("any"), str(tmp_path))
    assert not list(Path(tmp_path).iterdir())


def test_download_failure_keeps_diagnostic(tmp_path, monkeypatch):
    def run(command, **kwargs):
        if command[2] == "ls":
            return SimpleNamespace(
                returncode=0, stdout="gs://bucket/pipeline/202640/hosp/run/outputs/time/trajectories.csv.gz\n"
            )
        return SimpleNamespace(returncode=1, stderr="Permission denied")

    monkeypatch.setattr(aggregator.subprocess, "run", run)
    with pytest.raises(ValueError, match="Permission denied"):
        aggregator.pull_trajectory_projections(_config("any"), str(tmp_path))


def test_parquet_source_runs_aggregate_cli(tmp_path, monkeypatch):
    config = _config("run-1", "trajectories.parquet")
    source = "gs://bucket/pipeline/202640/hosp/run-1/outputs/time/trajectories.parquet"
    config_path = tmp_path / "example.yml"
    config_path.write_text(yaml.safe_dump({"aggregation": config.model_dump(mode="json")}))

    def run(command, **kwargs):
        if command[2] == "ls":
            return SimpleNamespace(returncode=0, stdout=source + "\n")
        pd.DataFrame(
            {"sim_id": [7], "population": ["United_States"], "date": ["2026-10-10"], "hospitalizations": [2.0]}
        ).to_parquet(command[-1])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(aggregator.subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", ["aggregate.py", "--config", str(config_path), "--output", str(tmp_path)])
    runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/postprocess/aggregate.py"), run_name="__main__")
    result = read_long_form_trajectories(tmp_path / "long_form_trajectories_example.parquet")
    assert set(result.series) == {"single", "total"}
    assert result.value.tolist() == [2.0, 2.0]
    assert result.attrs["sources"][0]["trajectory_file"] == source
