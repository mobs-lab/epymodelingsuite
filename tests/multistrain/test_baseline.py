import numpy as np
import pandas as pd
import pytest

from epymodelingsuite.multistrain.aggregator import dispatch_baseline
from epymodelingsuite.schema.aggregation import AggregationConfiguration


def _config(observed_means: str, method: str = "negative_binomial", seed: int | None = 1) -> AggregationConfiguration:
    return AggregationConfiguration(
        bucket="gs://unused",
        random_seed=seed,
        submission_week=202616,
        sources=[],
        sampling={"method": "random", "n_samples": 1},
        aggregate_method="sum",
        baseline={"method": method, "observed_means": observed_means, "dispersion_values": [5, 50]},
    )


def _trajectories(locations: list[str], n_samples: int = 200) -> pd.DataFrame:
    rows = [(loc, s) for loc in locations for s in range(n_samples)]
    df = pd.DataFrame(rows, columns=["location", "sample_id"])
    df["target_total"] = 0.0
    # Non-default index, as after filtering or concatenation
    return df.sample(frac=1, random_state=0).set_index(np.arange(len(df)) * 3)


def test_baseline_rows_stay_aligned_with_locations():
    # US mean (889) >> Alaska mean (3.4); rows are shuffled with a non-default index
    df = _trajectories(["United_States", "United_States__Alaska"])
    out = dispatch_baseline(df, _config("baselines_hosp_2025.csv"))

    assert out.index.equals(df.index)
    means = out.groupby("location")["target_baseline_k50"].mean()
    assert means["United_States"] == pytest.approx(889, rel=0.1)
    assert means["United_States__Alaska"] == pytest.approx(3.4, rel=0.5)


def test_baseline_is_reproducible_with_seed():
    df = _trajectories(["United_States"])
    a = dispatch_baseline(df, _config("baselines_hosp_2025.csv", seed=7))
    b = dispatch_baseline(df, _config("baselines_hosp_2025.csv", seed=7))
    pd.testing.assert_frame_equal(a, b)


def test_baseline_missing_location_raises():
    # Missouri (US-MO) is excluded from the ED baseline file due to missing data in surveillance
    df = _trajectories(["United_States", "United_States__Missouri"])
    with pytest.raises(ValueError, match="United_States__Missouri"):
        dispatch_baseline(df, _config("baselines_ed_2025.csv", method="beta"))


def test_beta_baseline_for_ed_proportions():
    df = _trajectories(["United_States", "United_States__Alabama"])
    out = dispatch_baseline(df, _config("baselines_ed_2025.csv", method="beta"))

    assert out["target_baseline_k5"].between(0, 1).all()
    assert out.loc[out.location == "United_States", "target_baseline_k50"].mean() == pytest.approx(0.0013, rel=0.3)


def test_metrocast_baseline_locations():
    df = _trajectories(["metrocast_location_athens"])
    out = dispatch_baseline(df, _config("baselines_metro_2025.csv"))
    assert out["target_baseline_k5"].notna().all()
