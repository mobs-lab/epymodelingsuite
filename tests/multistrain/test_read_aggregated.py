"""Long dashboard files feed existing submission and plot calculations."""

import pandas as pd
import pytest

from epymodelingsuite.multistrain.formatter import compute_quantiles, create_submission, read_aggregated
from epymodelingsuite.output.dashboard_trajectories import dashboard_trajectories_from_wide
from epymodelingsuite.schema.output import FlusightTrajectorySamples
from epymodelingsuite.schema.submission import AggregatedTrajectoriesConfig

TARGET = "wk inc flu prop ed visits"


def _wide():
    dates = pd.date_range("2026-10-03", periods=5, freq="7D")
    return pd.DataFrame(
        [
            {
                "sample_id": sample,
                "sim_id_single": sample + 10,
                "location": "United_States",
                "date": date,
                "target_single": (sample + day + 1) / 100,
                "target_total": (sample + day + 1) / 100,
                "target_baseline": (sample + day + 2) / 100,
            }
            for sample in range(2)
            for day, date in enumerate(dates)
        ]
    )


@pytest.mark.parametrize("series", ["target_total", "target_baseline", "total", "baseline"])
def test_long_and_wide_produce_same_submissions_and_quantiles(tmp_path, series):
    wide = _wide()
    long = dashboard_trajectories_from_wide(wide, ["single"], "2026-10-10", TARGET)
    path = tmp_path / "long.parquet"
    long.to_parquet(path, index=False)
    actual = read_aggregated(str(path), AggregatedTrajectoriesConfig(target_column=series))
    wide_column = series if series.startswith("target_") else "target_" + series
    expected = wide[["date", "location", wide_column, "sample_id"]].rename(columns={wide_column: series})
    pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected, check_dtype=False)
    pd.testing.assert_frame_equal(
        compute_quantiles(actual, value_col=series), compute_quantiles(expected, value_col=series), check_dtype=False
    )
    kwargs = dict(
        reference_date="2026-10-10",
        profile="flusight_ed",
        value_col=series,
        trajectory_samples=FlusightTrajectorySamples(n_samples=2, seed=1337),
    )
    pd.testing.assert_frame_equal(create_submission(actual, **kwargs), create_submission(expected, **kwargs))


def test_multiple_targets_require_explicit_selection(tmp_path):
    ed = dashboard_trajectories_from_wide(_wide(), ["single"], "2026-10-10", TARGET)
    hosp = ed.copy()
    hosp["target"] = "wk inc flu hosp"
    path = tmp_path / "multiple.parquet"
    pd.concat([ed, hosp], ignore_index=True).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="multiple targets"):
        read_aggregated(str(path), AggregatedTrajectoriesConfig(target_column="total"))
    result = read_aggregated(str(path), AggregatedTrajectoriesConfig(target=TARGET, target_column="total"))
    assert len(result) == len(_wide())
    with pytest.raises(ValueError, match="no rows"):
        read_aggregated(str(path), AggregatedTrajectoriesConfig(target="unknown", target_column="total"))


def test_unknown_series_and_corrupt_long_data_fail(tmp_path):
    long = dashboard_trajectories_from_wide(_wide(), ["single"], "2026-10-10", TARGET)
    path = tmp_path / "long.parquet"
    long.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="not found"):
        read_aggregated(str(path), AggregatedTrajectoriesConfig(target_column="h1"))
    long.iloc[1:].to_parquet(path, index=False)
    with pytest.raises(ValueError, match="all series"):
        read_aggregated(str(path), AggregatedTrajectoriesConfig(target_column="total"))


@pytest.mark.parametrize("suffix", ["parquet", "csv"])
def test_legacy_wide_files_still_read(tmp_path, suffix):
    path = tmp_path / f"wide.{suffix}"
    wide = _wide()
    if suffix == "parquet":
        wide.to_parquet(path, index=False)
    else:
        wide.to_csv(path, index=False)
    result = read_aggregated(str(path), AggregatedTrajectoriesConfig(target_column="target_total"))
    pd.testing.assert_frame_equal(result, wide[["date", "location", "target_total", "sample_id"]])
