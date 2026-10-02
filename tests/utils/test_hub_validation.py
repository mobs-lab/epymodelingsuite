from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from epymodelingsuite.utils.hub_validation import load_tasks, validate_model_output
from epymodelingsuite.utils.trajectory_samples import make_sample_rows

FIXTURES = Path(__file__).parent.parent / "fixtures" / "flusight"


@pytest.fixture(scope="module")
def tasks() -> dict:
    return load_tasks(FIXTURES / "tasks.json")


@pytest.fixture
def example() -> pd.DataFrame:
    """FluSight 2026-27 example submission (passes the hub's validation)."""
    return pd.read_parquet(FIXTURES / "2026-10-10-example-submission.parquet")


def _errors(df, tasks) -> str:
    return "\n".join(validate_model_output(df, tasks))


def _is(df, target, output_type) -> pd.Series:
    return (df.target == target) & (df.output_type == output_type)


def test_example_submission_passes(example, tasks):
    assert validate_model_output(example, tasks) == []


def test_csv_round_trip_passes(example, tasks, tmp_path):
    example.to_csv(tmp_path / "sub.csv", index=False)
    df = pd.read_csv(tmp_path / "sub.csv", dtype={"location": str, "output_type_id": str})
    assert validate_model_output(df, tasks) == []


def test_generated_samples_pass(example, tasks):
    # Replace the example's hosp samples with ones built by make_sample_rows
    hosp = _is(example, "wk inc flu hosp", "sample")
    locations = example.loc[hosp, "location"].unique()
    generated = [
        make_sample_rows(
            np.full((100, 5), 50.0), [-1, 0, 1, 2, 3], "2026-10-10", loc, "wk inc flu hosp", f"{loc}x", integer=True
        )
        for loc in locations
    ]
    df = pd.concat([example[~hosp], *generated], ignore_index=True)
    assert validate_model_output(df, tasks) == []


def test_extra_or_missing_column(example, tasks):
    assert "Columns must be exactly" in _errors(example.assign(extra=1), tasks)
    assert "Columns must be exactly" in _errors(example.drop(columns="value"), tasks)


def test_bad_types(example, tasks):
    assert "not YYYY-MM-DD" in _errors(example.assign(reference_date="10/10/2026"), tasks)
    df = example.astype({"horizon": "float"})
    df.loc[df.horizon.notna(), "horizon"] += 0.5
    assert "non-integer" in _errors(df, tasks)
    assert "non-numeric" in _errors(example.assign(value="x"), tasks)


def test_multiple_reference_dates(example, tasks):
    df = example.copy()
    df.loc[0, "reference_date"] = "2026-10-17"
    assert "single value" in _errors(df, tasks)


def test_unknown_target_and_location(example, tasks):
    assert "match no model task" in _errors(
        example.replace({"target": {"wk inc flu hosp": "wk inc covid hosp"}}), tasks
    )
    df = example.copy()
    df.loc[0, "location"] = "99"
    assert "`location` must be an allowed value" in _errors(df, tasks)


def test_peak_rows_must_have_no_horizon(example, tasks):
    df = example.copy()
    df.loc[df.target == "peak inc flu hosp", "horizon"] = 1
    assert "`horizon` must be NA" in _errors(df, tasks)


def test_duplicate_rows(example, tasks):
    assert "duplicate" in _errors(pd.concat([example, example.head(1)], ignore_index=True), tasks)


def test_missing_required_quantile(example, tasks):
    df = example[~(_is(example, "wk inc flu hosp", "quantile") & (example.output_type_id == "0.5"))]
    assert "miss required output_type_ids" in _errors(df, tasks)


def test_missing_required_output_type(example, tasks):
    # Samples without the required quantiles for the same target/location/horizon
    df = example[~_is(example, "wk inc flu hosp", "quantile")]
    assert "required output type `quantile`" in _errors(df, tasks)


def test_value_bounds_and_integers(example, tasks):
    df = example.copy()
    df.loc[_is(df, "wk inc flu prop ed visits", "quantile").idxmax(), "value"] = 1.5
    assert "above maximum 1" in _errors(df, tasks)
    df = example.copy()
    df.loc[_is(df, "wk inc flu hosp", "sample").idxmax(), "value"] = 10.5
    assert "must be integers" in _errors(df, tasks)
    df = example.copy()
    df.loc[_is(df, "wk inc flu hosp", "sample").idxmax(), "value"] = -1
    assert "below minimum 0" in _errors(df, tasks)


def test_decreasing_quantiles(example, tasks):
    df = example.copy()
    q = _is(df, "wk inc flu hosp", "quantile") & (df.output_type_id == "0.99")
    df.loc[q, "value"] = -0.0
    assert "values decrease" in _errors(df, tasks)


def test_pmf_must_sum_to_one(example, tasks):
    df = example.copy()
    df.loc[_is(df, "wk flu hosp rate change", "pmf").idxmax(), "value"] += 0.1
    assert "don't sum to 1" in _errors(df, tasks)


def test_sample_count(example, tasks):
    s = _is(example, "wk inc flu hosp", "sample")
    keep = ~(s & example.output_type_id.str.endswith("99"))
    assert "sample count outside [100, 100]" in _errors(example[keep], tasks)


def test_sample_id_spanning_locations(example, tasks):
    df = example.copy()
    s = _is(df, "wk inc flu hosp", "sample")
    second = df.loc[s, "location"].unique()[1]
    df.loc[s & (df.location == second), "output_type_id"] = df.loc[
        s & (df.location == second), "output_type_id"
    ].str.replace(r"^\D+", "US", regex=True)
    assert "span more than one" in _errors(df, tasks)


def test_samples_with_different_horizons(example, tasks):
    s = _is(example, "wk inc flu hosp", "sample")
    drop = s & (example.output_type_id == example.loc[s, "output_type_id"].iloc[0]) & (example.horizon == 3)
    assert "cover different" in _errors(example[~drop], tasks)


def test_target_end_date_matches_horizon(example, tasks):
    df = example.copy()
    df.loc[df.horizon == 0, "target_end_date"] = "2026-10-17"
    assert "target_end_date != reference_date" in _errors(df, tasks)
