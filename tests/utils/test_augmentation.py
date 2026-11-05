"""Tests for HHS reporting-fraction augmentation."""

import numpy as np
import pandas as pd
import pytest

from epymodelingsuite.utils.augmentation import HHSDataNotReadyError, augment_hhs_hospitalizations


@pytest.fixture
def snapshot():
    """Two weeks for two locations; only the second week should be adjusted."""
    return pd.DataFrame(
        {
            "location_iso": ["US-CA", "US-CA", "US-TX", "US-TX"],
            "location_code": ["06", "06", "48", "48"],
            "target_end_date": pd.to_datetime(["2026-05-16", "2026-05-23"] * 2),
            "epiweek": [202619, 202620] * 2,
            "hospitalizations": [np.nan, 100.0, 40.0, 20.0],
            "reporting_frac": [np.nan, 0.5, 0.9, 1.0],
        }
    )


def test_adjusts_only_latest_week(snapshot):
    """Test that only the latest week is scaled and the input is not mutated."""
    original = snapshot.copy()
    augmented, reporting = augment_hhs_hospitalizations(snapshot, method="weighted")

    assert "reporting_frac" not in augmented.columns
    # Latest week: 100 at 50% -> 150; 20 at 100% -> 20
    assert augmented["hospitalizations"].iloc[[1, 3]].tolist() == [150.0, 20.0]
    # Earlier week unchanged, including missing counts
    assert pd.isna(augmented["hospitalizations"].iloc[0])
    assert augmented["hospitalizations"].iloc[2] == 40.0
    assert reporting["values_augment"].tolist() == [150.0, 20.0]
    assert reporting["estimated_reporting"].tolist() == [1.5, 1.0]
    pd.testing.assert_frame_equal(snapshot, original)


def test_missing_latest_fraction_is_not_ready(snapshot):
    """Test that a missing latest-week reporting fraction raises HHSDataNotReadyError."""
    snapshot.loc[1, "reporting_frac"] = np.nan
    with pytest.raises(HHSDataNotReadyError):
        augment_hhs_hospitalizations(snapshot, method="weighted")


def test_missing_latest_count_is_not_ready(snapshot):
    """Test that a missing latest-week count raises HHSDataNotReadyError."""
    snapshot.loc[3, "hospitalizations"] = np.nan
    with pytest.raises(HHSDataNotReadyError):
        augment_hhs_hospitalizations(snapshot, method="weighted")


@pytest.mark.parametrize("fraction", [0.0, 1.5])
def test_out_of_range_fraction_raises(snapshot, fraction):
    """Test that reporting fractions outside 0 < r <= 1 raise ValueError."""
    snapshot.loc[1, "reporting_frac"] = fraction
    with pytest.raises(ValueError, match="0 < r <= 1"):
        augment_hhs_hospitalizations(snapshot, method="weighted")


def test_average_augments_without_reducing_or_mutating_snapshot(snapshot):
    original = snapshot.copy()
    augmented, reporting = augment_hhs_hospitalizations(
        snapshot, method="average", average_reporting_frac={"US-CA": 0.75, "US-TX": 0.8}
    )
    assert augmented["hospitalizations"].iloc[[1, 3]].tolist() == [150.0, 20.0]
    assert pd.isna(augmented["hospitalizations"].iloc[0])
    assert augmented["hospitalizations"].iloc[2] == 40.0
    assert reporting["estimated_reporting"].tolist() == [1.5, 1.0]
    assert reporting["average_reporting_frac"].tolist() == [0.75, 0.8]
    assert reporting["method"].tolist() == ["average", "average"]
    assert reporting["unreported_weight"].isna().all()
    pd.testing.assert_frame_equal(snapshot, original)


@pytest.mark.parametrize("average", [0, -0.1, 70.2, np.nan, np.inf])
def test_invalid_average_raises(snapshot, average):
    with pytest.raises(ValueError, match="Average reporting fractions"):
        augment_hhs_hospitalizations(
            snapshot, method="average", average_reporting_frac={"US-CA": average, "US-TX": 0.8}
        )


def test_missing_state_average_raises(snapshot):
    with pytest.raises(ValueError, match="US-TX"):
        augment_hhs_hospitalizations(snapshot, method="average", average_reporting_frac={"US-CA": 0.75})


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"method": "unknown"}, "Unknown augmentation method"),
        ({"method": "average"}, "requires average_reporting_frac"),
        ({"method": "average", "unreported_weight": 0.5}, "does not accept unreported_weight"),
        ({"method": "weighted", "average_reporting_frac": {}}, "only accepted"),
        ({"method": "weighted", "unreported_weight": -0.1}, "between 0 and 1"),
        ({"method": "weighted", "unreported_weight": np.nan}, "between 0 and 1"),
    ],
)
def test_invalid_method_arguments_raise(snapshot, kwargs, message):
    with pytest.raises(ValueError, match=message):
        augment_hhs_hospitalizations(snapshot, **kwargs)


@pytest.mark.parametrize(("weight", "expected"), [(0, [100.0, 20.0]), (1, [200.0, 20.0])])
def test_weighted_endpoints(snapshot, weight, expected):
    augmented, _ = augment_hhs_hospitalizations(snapshot, method="weighted", unreported_weight=weight)
    assert augmented["hospitalizations"].iloc[[1, 3]].tolist() == expected


def test_average_accepts_integer_counts_from_csv(snapshot):
    snapshot["hospitalizations"] = snapshot["hospitalizations"].fillna(0).astype(int)
    augmented, _ = augment_hhs_hospitalizations(
        snapshot, method="average", average_reporting_frac={"US-CA": 0.70213, "US-TX": 0.77076}
    )
    assert augmented["hospitalizations"].iloc[[1, 3]].tolist() == pytest.approx([140.426, 20.0])


@pytest.mark.parametrize(
    ("average", "expected_factor", "expected_count"),
    [(0.25, 1.0, 100.0), (0.5, 1.0, 100.0), (0.8, 1.6, 160.0)],
)
def test_average_factor_at_below_and_above_current_reporting(snapshot, average, expected_factor, expected_count):
    augmented, reporting = augment_hhs_hospitalizations(
        snapshot, method="average", average_reporting_frac={"US-CA": average, "US-TX": 0.8}
    )
    assert augmented.loc[1, "hospitalizations"] == expected_count
    assert reporting.loc[1, "values_augment"] == expected_count
    assert reporting.loc[1, "estimated_reporting"] == expected_factor
