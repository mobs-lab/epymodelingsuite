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
    augmented, reporting = augment_hhs_hospitalizations(snapshot)

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
        augment_hhs_hospitalizations(snapshot)


def test_missing_latest_count_is_not_ready(snapshot):
    """Test that a missing latest-week count raises HHSDataNotReadyError."""
    snapshot.loc[3, "hospitalizations"] = np.nan
    with pytest.raises(HHSDataNotReadyError):
        augment_hhs_hospitalizations(snapshot)


@pytest.mark.parametrize("fraction", [0.0, 1.5])
def test_out_of_range_fraction_raises(snapshot, fraction):
    """Test that reporting fractions outside 0 < r <= 1 raise ValueError."""
    snapshot.loc[1, "reporting_frac"] = fraction
    with pytest.raises(ValueError, match="0 < r <= 1"):
        augment_hhs_hospitalizations(snapshot)
