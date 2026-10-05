"""Tests for shared hub target value formatting."""

import numpy as np
import pandas as pd
import pytest

from epymodelingsuite.output.hub_format import normalize_target_values


@pytest.mark.parametrize("container", [np.array, pd.Series])
@pytest.mark.parametrize(
    "target,inputs,expected",
    [
        # Counts: clip negatives, round down/up and both .5 ties, with no upper bound.
        ("wk inc flu hosp", [-2, 0, 0.4, 1.6, 10.5, 11.5, 150.2, np.nan], [0, 0, 0, 2, 10, 12, 150, np.nan]),
        # Proportions: preserve valid fractions and both boundaries; clip outside [0, 1].
        ("wk inc flu prop ed visits", [-0.1, 0, 0.2, 0.5, 1, 1.3, np.nan], [0, 0, 0.2, 0.5, 1, 1, np.nan]),
        # Flu percentages: preserve decimals below/above 1 and the boundary at 100.
        ("Flu ED visits pct", [-2, 0, 0.5, 2.5, 100, 120, np.nan], [0, 0, 0.5, 2.5, 100, 100, np.nan]),
        # ILI percentages: check values just below/above the upper boundary without rounding.
        ("ILI ED visits pct", [-0.5, 0, 1.25, 99.9, 100, 100.1, np.nan], [0, 0, 1.25, 99.9, 100, 100, np.nan]),
    ],
)
def test_normalizes_target_values_without_modifying_input(container, target, inputs, expected):
    """Array samples and Series quantiles use the same rules while retaining missing values."""
    values = container(inputs)
    if isinstance(values, pd.Series):
        # A non-default index and name expose accidental conversion to an ndarray.
        values.index = range(10, 10 + len(values))
        values.name = "value"
    original = values.copy()
    normalized = normalize_target_values(values, target)
    assert type(normalized) is type(values)
    np.testing.assert_allclose(normalized, expected, equal_nan=True)
    if isinstance(values, pd.Series):
        pd.testing.assert_series_equal(values, original)
        assert normalized.index.equals(values.index)
        assert normalized.name == values.name
    else:
        np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize("container", [np.array, pd.Series])
def test_unknown_target_values_are_unchanged(container):
    """Unknown targets do not imply bounds or integer rounding."""
    # Signed values and values above 100 must survive without assuming count,
    # proportion or percentage units. Missing values must also remain missing.
    values = container([-120.75, -0.5, 0, 120.5, np.nan])
    normalized = normalize_target_values(values, "custom_target")
    assert normalized is not values
    if isinstance(values, pd.Series):
        pd.testing.assert_series_equal(normalized, values)
    else:
        np.testing.assert_array_equal(normalized, values)


def test_nullable_counts_preserve_missing_values():
    """Nullable pandas counts remain missing until the caller chooses the output dtype."""
    values = pd.Series([-2, 11, pd.NA], dtype="Int64", name="value")
    normalized = normalize_target_values(values, "wk inc flu hosp").astype("Int64")
    pd.testing.assert_series_equal(normalized, pd.Series([0, 11, pd.NA], dtype="Int64", name="value"))
