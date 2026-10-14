"""Resolve provenance from the concrete GCS object that is downloaded."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from epymodelingsuite.multistrain import aggregator
from epymodelingsuite.schema.aggregation import AggregationConfiguration


def _config(run_id):
    return AggregationConfiguration(
        bucket="gs://bucket/pipeline",
        submission_week="202640",
        target="wk inc flu hosp",
        sources=[
            {
                "strain": "single",
                "experiment": "202640/hosp",
                "run_id": run_id,
                "trajectory_file": "trajectories.csv.gz",
            }
        ],
        sampling={"method": "random", "n_samples": 1},
        aggregate_method="sum",
    )


@pytest.mark.parametrize("selector", ["latest", "any", "20261007-182529-061e374e"])
def test_download_records_resolved_run_and_file(tmp_path, monkeypatch, selector):
    run_id = "20261007-182529-061e374e"
    prefix = "gs://bucket/pipeline/202640/hosp/"
    source_path = f"{prefix}{run_id}/outputs/20261013-224354/trajectories.csv.gz"
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[2] == "cp":
            import gzip

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
    trajectories, sources = aggregator.pull_trajectory_projections(_config(selector), str(tmp_path))
    assert trajectories["single"].hospitalizations.tolist() == [2.0]
    assert sources == [
        {
            "target": "wk inc flu hosp",
            "strain": "single",
            "experiment": "202640/hosp",
            "run_id": run_id,
            "trajectory_file": source_path,
        }
    ]
    assert calls[-1] == ["gcloud", "storage", "cp", source_path, str(tmp_path / "single_trajectories.csv.gz")]
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
