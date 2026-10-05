"""Tests for shared hub target value formatting."""

import numpy as np
import pandas as pd
import pytest

from epymodelingsuite.output.hub_format import normalize_target_values


@pytest.mark.parametrize("container", [np.array, pd.Series])
@pytest.mark.parametrize(
    "target,expected",
    [
        ("wk inc flu hosp", [0, 0, 2, 10, 120, np.nan]),
        ("wk inc flu prop ed visits", [0, 0.5, 1, 1, 1, np.nan]),
        ("Flu ED visits pct", [0, 0.5, 2.5, 10.5, 100, np.nan]),
        ("ILI ED visits pct", [0, 0.5, 2.5, 10.5, 100, np.nan]),
        ("custom_target", [0, 0.5, 2.5, 10.5, 120, np.nan]),
    ],
)
def test_normalizes_target_values_without_modifying_input(container, target, expected):
    """Array samples and Series quantiles use the same rules while retaining missing values."""
    values = container([-2.0, 0.5, 2.5, 10.5, 120.0, np.nan])
    if isinstance(values, pd.Series):
        # A non-default index and name expose accidental conversion to an ndarray.
        values.index = list("abcdef")
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


def test_nullable_counts_preserve_missing_values():
    """Nullable pandas counts remain missing until the caller chooses the output dtype."""
    values = pd.Series([-2, 11, pd.NA], dtype="Int64", name="value")
    normalized = normalize_target_values(values, "wk inc flu hosp").astype("Int64")
    pd.testing.assert_series_equal(normalized, pd.Series([0, 11, pd.NA], dtype="Int64", name="value"))
