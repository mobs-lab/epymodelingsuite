"""Tests for sample selection and rows in epymodelingsuite.output.samples."""

from datetime import date

import numpy as np
import pandas as pd

from epymodelingsuite.output import samples as ts

HORIZONS = [-1, 0, 1, 2, 3]


class TestSelectSamples:
    """Tests for sample selection."""

    def test_selects_n_distinct_rows(self):
        """Test that n distinct trajectories are selected."""
        values = np.arange(1000 * 5, dtype=float).reshape(1000, 5)
        idx = ts.select_samples(values, 100, seed=1)
        assert len(idx) == 100
        assert len(set(idx)) == 100

    def test_same_seed_same_selection(self):
        """Test that the same seed gives the same selection."""
        values = np.random.default_rng(0).random((500, 5))
        assert (ts.select_samples(values, 100, seed=7) == ts.select_samples(values, 100, seed=7)).all()

    def test_fewer_than_n_returns_all_without_resampling(self):
        """Test that all trajectories are returned, without resampling, when fewer than n exist."""
        values = np.ones((60, 5))
        assert ts.select_samples(values, 100, seed=1).tolist() == list(range(60))

    def test_trajectories_with_nan_are_never_selected(self):
        """Test that trajectories with NaN are never selected."""
        values = np.ones((150, 5))
        values[::2, 3] = np.nan  # 75 incomplete rows
        idx = ts.select_samples(values, 100, seed=1)
        assert len(idx) == 75
        assert not np.isnan(values[idx]).any()

    def test_registered_selector_is_used(self, monkeypatch):
        """Test that a newly registered selector is used and its indices map to the original rows."""
        monkeypatch.setitem(ts.SAMPLE_SELECTORS, "first", lambda values, n, rng: np.arange(min(n, len(values))))
        values = np.ones((10, 5))
        values[0, 0] = np.nan
        # Indices refer to the original rows, after dropping the incomplete row 0
        assert ts.select_samples(values, 3, method="first").tolist() == [1, 2, 3]


class TestMakeSampleRows:
    """Tests for building hub sample rows."""

    def test_ids_horizons_and_dates(self):
        """Test sample ids, horizons and target end dates of the rows."""
        rows = ts.make_sample_rows(
            np.ones((3, 5)), HORIZONS, date(2026, 10, 10), "25", "wk inc flu hosp", "MA", integer=True
        )
        assert len(rows) == 15
        assert rows.output_type_id.unique().tolist() == ["MA00", "MA01", "MA02"]
        assert rows.groupby("output_type_id").horizon.apply(list).map(lambda h: h == HORIZONS).all()
        first = rows[rows.output_type_id == "MA00"]
        assert first.target_end_date.tolist() == [date(2026, 10, 3) + pd.Timedelta(weeks=i) for i in range(5)]
        assert set(rows.output_type) == {"sample"}

    def test_counts_are_rounded_non_negative_integers(self):
        """Test that count values are clipped at 0 and rounded to integers."""
        values = np.array([[-2.0, 0.4, 1.6, 10.5, 3.2]])
        rows = ts.make_sample_rows(values, HORIZONS, date(2026, 10, 10), "US", "wk inc flu hosp", "US", integer=True)
        assert rows.value.tolist() == [0.0, 0.0, 2.0, 10.0, 3.0]

    def test_proportions_are_clipped_to_unit_interval(self):
        """Test that proportion values are clipped to [0, 1]."""
        values = np.array([[-0.1, 0.2, 1.3, 0.5, 1.0]])
        rows = ts.make_sample_rows(
            values, HORIZONS, date(2026, 10, 10), "US", "wk inc flu prop ed visits", "US", upper=1
        )
        assert rows.value.tolist() == [0.0, 0.2, 1.0, 0.5, 1.0]
