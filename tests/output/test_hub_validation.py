"""Tests for epymodelingsuite.output.hub_validation."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from epymodelingsuite.output.hub_validation import load_hub_tasks, validate_model_output
from epymodelingsuite.output.trajectory_samples import trajectories_to_sample_rows

FIXTURES = Path(__file__).parent.parent / "fixtures" / "flusight"

# Invalid-input tests modify a valid official submission. One change can violate
# multiple rules, so each assertion looks for the diagnostic that test targets.


@pytest.fixture(scope="module")
def tasks() -> dict:
    """FluSight 2026-27 tasks.json."""
    return load_hub_tasks(FIXTURES / "tasks.json")


@pytest.fixture
def example() -> pd.DataFrame:
    """FluSight 2026-27 example submission, with sample ids made unique per target.

    The published example reuses each sample id for hosp and ED, which the hub's validator
    rejects (spl_mt_unique), so ED ids get an ``ed_`` prefix to make a valid baseline.
    """
    df = pd.read_parquet(FIXTURES / "2026-10-10-example-submission.parquet")
    ed = _is(df, "wk inc flu prop ed visits", "sample")
    df.loc[ed, "output_type_id"] = "ed_" + df.loc[ed, "output_type_id"]
    return df


def _errors(df, tasks) -> str:
    return "\n".join(validate_model_output(df, tasks))


def _is(df, target, output_type) -> pd.Series:
    return (df.target == target) & (df.output_type == output_type)


def test_example_submission_passes(example, tasks):
    """Test that the FluSight example submission passes."""
    # Establish that the unchanged fixture is a valid baseline for the mutations below.
    assert validate_model_output(example, tasks) == []


def test_csv_round_trip_passes(example, tasks, tmp_path):
    """Test that the example passes after a round trip through csv."""
    example.to_csv(tmp_path / "sub.csv", index=False)
    # Preserve IDs such as location "01" when CSV has no stored column types.
    df = pd.read_csv(tmp_path / "sub.csv", dtype={"location": str, "output_type_id": str})
    assert validate_model_output(df, tasks) == []


def test_generated_samples_pass(example, tasks):
    """Test that samples built by trajectories_to_sample_rows pass."""
    # Replace only hospitalization samples, keeping the example's required quantiles
    # and other targets. This checks that generated sample rows fit a valid submission.
    hosp = _is(example, "wk inc flu hosp", "sample")
    locations = example.loc[hosp, "location"].unique()
    generated = [
        trajectories_to_sample_rows(
            np.full((100, 5), 50.0), [-1, 0, 1, 2, 3], "2026-10-10", loc, "wk inc flu hosp", f"{loc}x"
        )
        for loc in locations
    ]
    df = pd.concat([example[~hosp], *generated], ignore_index=True)
    assert validate_model_output(df, tasks) == []


def test_extra_or_missing_column(example, tasks):
    """Test that extra or missing columns are reported."""
    # The hub schema requires an exact column set; check both directions of mismatch.
    assert "Columns must be exactly" in _errors(example.assign(extra=1), tasks)
    assert "Columns must be exactly" in _errors(example.drop(columns="value"), tasks)


def test_bad_types(example, tasks):
    """Test that non-date, non-integer horizon and non-numeric values are reported."""
    # A recognizable date still has to use the required YYYY-MM-DD representation.
    assert "not YYYY-MM-DD" in _errors(example.assign(reference_date="10/10/2026"), tasks)
    # Allow fractional values in the test input so the validator, rather than pandas
    # assignment, rejects horizons that are not whole weeks. Preserve missing horizons.
    df = example.astype({"horizon": "float"})
    df.loc[df.horizon.notna(), "horizon"] += 0.5
    assert "non-integer" in _errors(df, tasks)
    assert "non-numeric" in _errors(example.assign(value="x"), tasks)


# Exercise a second allowed date and one outside the configured submission dates.
@pytest.mark.parametrize("reference_date", ["2026-10-17", "2099-01-01"])
@pytest.mark.parametrize("missing_date", [False, True])
def test_multiple_reference_dates(example, tasks, missing_date, reference_date):
    """Report multiple reference dates even when a missing date is also present."""
    df = example.copy()
    # This second date causes the "single value" error in both missing_date cases.
    df.loc[0, "reference_date"] = reference_date
    if missing_date:
        # The same error must be reported without crashing when dates and missing
        # values are sorted to build the error message.
        df.loc[1, "reference_date"] = None
    assert "single value" in _errors(df, tasks)


def test_unknown_target_and_location(example, tasks):
    """Test that unknown targets and locations are reported."""
    assert "match no model task" in _errors(example.replace({"target": {"wk inc flu hosp": "invalid_target"}}), tasks)
    # Start with valid targets again so unmatched-target filtering cannot hide a bad location.
    df = example.copy()
    df.loc[0, "location"] = "999"  # Invalid location ID.
    assert "`location` must be an allowed value" in _errors(df, tasks)


def test_peak_rows_must_have_no_horizon(example, tasks):
    """Test that peak target rows with a horizon are reported."""
    df = example.copy()
    # The peak target has no weekly horizon in tasks.json; assigning one must be rejected.
    df.loc[df.target == "peak inc flu hosp", "horizon"] = 1
    assert "`horizon` must be NA" in _errors(df, tasks)


def test_duplicate_rows(example, tasks):
    """Test that duplicate rows are reported."""
    # Repeating an existing row makes all forecast-identifying columns collide exactly.
    assert "duplicate" in _errors(pd.concat([example, example.head(1)], ignore_index=True), tasks)


def test_missing_required_quantile(example, tasks):
    """Test that a missing required quantile level is reported."""
    # Leave the other quantile levels in place but remove the required median.
    df = example[~(_is(example, "wk inc flu hosp", "quantile") & (example.output_type_id == "0.5"))]
    assert "miss required output_type_ids" in _errors(df, tasks)


def test_equivalent_quantile_ids_are_duplicates(example, tasks):
    """Reject a repeated numeric quantile even when its string spelling differs."""
    row = example[example.output_type.eq("quantile") & example.output_type_id.eq("0.01")].head(1)
    duplicate = row.assign(output_type_id="0.010")
    assert "duplicate" in _errors(pd.concat([example, duplicate], ignore_index=True), tasks)
    # Alternative spellings alone are valid and validation must not modify the input.
    alternate = example.replace({"output_type_id": {"0.01": "0.010"}})
    original = alternate.copy(deep=True)
    assert validate_model_output(alternate, tasks) == []
    pd.testing.assert_frame_equal(alternate, original)


@pytest.mark.parametrize("output_type", ["pmf", "quantile"])
def test_missing_and_invalid_output_ids_return_diagnostics(example, tasks, output_type):
    """Report mixed missing and invalid IDs without failing to sort them."""
    df = example.copy()
    indices = df.index[df.output_type.eq(output_type)][:2]
    df.loc[indices, "output_type_id"] = [None, "invalid"]
    assert "invalid output_type_id" in _errors(df, tasks)


def test_missing_required_output_type(example, tasks):
    """Test that samples without the required quantiles are reported."""
    # Samples alone are insufficient: hospitalization forecasts also require quantiles.
    df = example[~_is(example, "wk inc flu hosp", "quantile")]
    assert "required output type `quantile`" in _errors(df, tasks)


def test_value_bounds_and_integers(example, tasks):
    """Test that values outside the bounds or non-integer counts are reported."""
    # Use a fresh copy for each case: a proportion above 1, a fractional count,
    # and a negative count should each produce its own validation diagnostic.
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
    """Test that quantile values decreasing with level are reported."""
    df = example.copy()
    q = _is(df, "wk inc flu hosp", "quantile") & (df.output_type_id == "0.99")
    # Zero respects the lower bound but puts the 99th percentile below lower quantiles.
    # This makes the failure about quantile ordering rather than negative counts.
    df.loc[q, "value"] = -0.0
    assert "values decrease" in _errors(df, tasks)


def test_pmf_must_sum_to_one(example, tasks):
    """Test that pmf probabilities not summing to 1 are reported."""
    df = example.copy()
    # Changing one category breaks the total probability without removing any category.
    df.loc[_is(df, "wk flu hosp rate change", "pmf").idxmax(), "value"] += 0.1
    assert "don't sum to 1" in _errors(df, tasks)


def test_sample_count(example, tasks):
    """Test that a sample count other than 100 is reported."""
    s = _is(example, "wk inc flu hosp", "sample")
    # Remove sample 99 across all horizons at each location: 99 complete trajectories
    # remain, isolating the sample-count requirement from horizon coverage.
    keep = ~(s & example.output_type_id.str.endswith("99"))
    assert "sample count outside [100, 100]" in _errors(example[keep], tasks)


def test_missing_sample_id(example, tasks):
    """Reject an extra unidentified trajectory even when 100 valid sample ids remain."""
    sample = example[_is(example, "wk inc flu hosp", "sample") & (example.output_type_id == "US00")]
    # Keep all 100 named samples and append one without an ID. Counting unique IDs
    # alone misses this extra trajectory because pandas excludes missing IDs by default.
    df = pd.concat([example, sample.assign(output_type_id=None)], ignore_index=True)
    assert "missing output_type_id" in _errors(df, tasks)


def test_sample_id_spanning_locations(example, tasks):
    """Test that a sample id used for two locations is reported."""
    df = example.copy()
    s = _is(df, "wk inc flu hosp", "sample")
    second = df.loc[s, "location"].unique()[1]
    # Reuse the US sample prefixes at a second location while leaving its location
    # column unchanged. The same sample IDs now incorrectly identify two locations.
    df.loc[s & (df.location == second), "output_type_id"] = df.loc[
        s & (df.location == second), "output_type_id"
    ].str.replace(r"^\D+", "US", regex=True)
    assert "span more than one" in _errors(df, tasks)


def test_sample_id_shared_across_targets(example, tasks):
    """Test that a sample id used for both hosp and ED is reported, as the hub's spl_mt_unique check does."""
    df = example.copy()
    ed = _is(df, "wk inc flu prop ed visits", "sample")
    df.loc[ed, "output_type_id"] = df.loc[ed, "output_type_id"].str.removeprefix("ed_")
    assert "shared across targets" in _errors(df, tasks)


def test_samples_with_different_horizons(example, tasks):
    """Test that samples covering different horizons are reported."""
    s = _is(example, "wk inc flu hosp", "sample")
    # Drop one horizon from one sample. The sample count stays at 100, but that
    # trajectory no longer covers the same weeks as the other samples.
    drop = s & (example.output_type_id == example.loc[s, "output_type_id"].iloc[0]) & (example.horizon == 3)
    assert "cover different" in _errors(example[~drop], tasks)


def test_target_end_date_matches_horizon(example, tasks):
    """Test that target_end_date inconsistent with the horizon is reported."""
    df = example.copy()
    # Horizon zero must end on the reference date (2026-10-10); move it one week ahead.
    df.loc[df.horizon == 0, "target_end_date"] = "2026-10-17"
    assert "target_end_date != reference_date" in _errors(df, tasks)
