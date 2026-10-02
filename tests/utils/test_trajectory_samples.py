import io
from datetime import date

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from epymodelingsuite.utils import trajectory_samples as ts

HORIZONS = [-1, 0, 1, 2, 3]

# Schema of the FluSight 2026-27 example submission (auxiliary-data/2026-10-10-example-submission.parquet)
FLUSIGHT_EXAMPLE_SCHEMA = {
    "reference_date": pa.string(),
    "horizon": pa.int32(),
    "target_end_date": pa.string(),
    "location": pa.string(),
    "target": pa.string(),
    "output_type": pa.string(),
    "output_type_id": pa.string(),
    "value": pa.float64(),
}


class TestSelectSamples:
    def test_selects_n_distinct_rows(self):
        values = np.arange(1000 * 5, dtype=float).reshape(1000, 5)
        idx = ts.select_samples(values, 100, seed=1)
        assert len(idx) == 100
        assert len(set(idx)) == 100

    def test_same_seed_same_selection(self):
        values = np.random.default_rng(0).random((500, 5))
        assert (ts.select_samples(values, 100, seed=7) == ts.select_samples(values, 100, seed=7)).all()

    def test_fewer_than_n_returns_all_without_resampling(self):
        values = np.ones((60, 5))
        assert ts.select_samples(values, 100, seed=1).tolist() == list(range(60))

    def test_trajectories_with_nan_are_never_selected(self):
        values = np.ones((150, 5))
        values[::2, 3] = np.nan  # 75 incomplete rows
        idx = ts.select_samples(values, 100, seed=1)
        assert len(idx) == 75
        assert not np.isnan(values[idx]).any()

    def test_registered_selector_is_used(self, monkeypatch):
        monkeypatch.setitem(ts.SAMPLE_SELECTORS, "first", lambda values, n, rng: np.arange(min(n, len(values))))
        values = np.ones((10, 5))
        values[0, 0] = np.nan
        # Indices refer to the original rows, after dropping the incomplete row 0
        assert ts.select_samples(values, 3, method="first").tolist() == [1, 2, 3]


class TestMakeSampleRows:
    def test_ids_horizons_and_dates(self):
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
        values = np.array([[-2.0, 0.4, 1.6, 10.5, 3.2]])
        rows = ts.make_sample_rows(values, HORIZONS, date(2026, 10, 10), "US", "wk inc flu hosp", "US", integer=True)
        assert rows.value.tolist() == [0.0, 0.0, 2.0, 10.0, 3.0]

    def test_proportions_are_clipped_to_unit_interval(self):
        values = np.array([[-0.1, 0.2, 1.3, 0.5, 1.0]])
        rows = ts.make_sample_rows(
            values, HORIZONS, date(2026, 10, 10), "US", "wk inc flu prop ed visits", "US", upper=1
        )
        assert rows.value.tolist() == [0.0, 0.2, 1.0, 0.5, 1.0]


def _quantile_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "location": "25",
            "reference_date": date(2026, 10, 10),
            "horizon": [0, 0],
            "target_end_date": [date(2026, 10, 10)] * 2,
            "target": "wk inc flu hosp",
            "output_type": "quantile",
            "output_type_id": [0.025, 0.5],
            "value": pd.array([3, 7], dtype="Int64"),
        }
    )


def test_hub_parquet_matches_flusight_example_schema():
    samples = ts.make_sample_rows(
        np.ones((2, 5)), HORIZONS, date(2026, 10, 10), "25", "wk inc flu hosp", "MA", integer=True
    )
    data = ts.hub_parquet_bytes(pd.concat([_quantile_rows(), samples], ignore_index=True))

    table = pq.read_table(io.BytesIO(data))
    assert {f.name: f.type for f in table.schema} == FLUSIGHT_EXAMPLE_SCHEMA
    df = table.to_pandas()
    assert df.output_type_id.tolist()[:2] == ["0.025", "0.5"]
    assert df.reference_date.unique().tolist() == ["2026-10-10"]
    assert df.location.unique().tolist() == ["25"]


class TestCombineSubmissions:
    def test_concatenates_hosp_and_ed(self):
        hosp = ts.make_sample_rows(np.ones((2, 5)), HORIZONS, date(2026, 10, 10), "25", "wk inc flu hosp", "MA")
        ed = ts.make_sample_rows(np.ones((2, 5)), HORIZONS, date(2026, 10, 10), "25", "wk inc flu prop ed visits", "MA")
        combined = ts.combine_submissions([hosp, ed])
        assert len(combined) == 20
        assert set(combined.target) == {"wk inc flu hosp", "wk inc flu prop ed visits"}

    def test_same_forecast_from_two_tables_raises(self):
        hosp = ts.make_sample_rows(np.ones((2, 5)), HORIZONS, date(2026, 10, 10), "25", "wk inc flu hosp", "MA")
        with pytest.raises(ValueError, match="more than one table"):
            ts.combine_submissions([hosp, hosp.copy()])
