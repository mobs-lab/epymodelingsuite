"""Tests for epymodelingsuite.output.hub_files."""

import io
from datetime import date

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from epymodelingsuite.output import hub_files
from epymodelingsuite.output.samples import make_sample_rows

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
    """Test that hub parquet bytes use the column types of the FluSight example submission."""
    samples = make_sample_rows(
        np.ones((2, 5)), HORIZONS, date(2026, 10, 10), "25", "wk inc flu hosp", "MA", integer=True
    )
    data = hub_files.hub_parquet_bytes(pd.concat([_quantile_rows(), samples], ignore_index=True))

    table = pq.read_table(io.BytesIO(data))
    assert {f.name: f.type for f in table.schema} == FLUSIGHT_EXAMPLE_SCHEMA
    df = table.to_pandas()
    assert df.output_type_id.tolist()[:2] == ["0.025", "0.5"]
    assert df.reference_date.unique().tolist() == ["2026-10-10"]
    assert df.location.unique().tolist() == ["25"]


class TestCombineSubmissions:
    """Tests for combining hub tables into one submission."""

    def test_concatenates_hosp_and_ed(self):
        """Test that hosp and ED tables are concatenated into one submission."""
        hosp = make_sample_rows(np.ones((2, 5)), HORIZONS, date(2026, 10, 10), "25", "wk inc flu hosp", "MA")
        ed = make_sample_rows(np.ones((2, 5)), HORIZONS, date(2026, 10, 10), "25", "wk inc flu prop ed visits", "MA")
        combined = hub_files.combine_submissions([hosp, ed])
        assert len(combined) == 20
        assert set(combined.target) == {"wk inc flu hosp", "wk inc flu prop ed visits"}

    def test_same_forecast_from_two_tables_raises(self):
        """Test that the same location/target/output_type from two tables raises."""
        hosp = make_sample_rows(np.ones((2, 5)), HORIZONS, date(2026, 10, 10), "25", "wk inc flu hosp", "MA")
        with pytest.raises(ValueError, match="more than one table"):
            hub_files.combine_submissions([hosp, hosp.copy()])
