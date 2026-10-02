"""File layout of hubverse (FluSight) submissions: column types, combining and reading."""

from pathlib import Path

import pandas as pd
import pyarrow as pa

HUB_COLUMNS = [
    "reference_date",
    "horizon",
    "target_end_date",
    "location",
    "target",
    "output_type",
    "output_type_id",
    "value",
]

# Column types of the FluSight example submission parquet (auxiliary-data/2026-10-10-example-submission.parquet)
HUB_SCHEMA = pa.schema(
    [
        ("reference_date", pa.string()),
        ("horizon", pa.int32()),
        ("target_end_date", pa.string()),
        ("location", pa.string()),
        ("target", pa.string()),
        ("output_type", pa.string()),
        ("output_type_id", pa.string()),
        ("value", pa.float64()),
    ]
)


def cast_hub_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast a hub table to the column types of the FluSight example submission parquet.

    Dates become 'YYYY-MM-DD' strings, `horizon` nullable int32, `output_type_id` and the other id columns
    strings, `value` float64. Columns are put in hub order. Arrow-backed dtypes preserve the exact schema
    when the returned copy is saved with DataFrame.to_parquet().
    """
    out = df[HUB_COLUMNS].copy()
    for col in ["reference_date", "target_end_date"]:
        out[col] = pd.to_datetime(out[col]).dt.strftime("%Y-%m-%d").astype("string")
    out["horizon"] = out["horizon"].astype("Int32")
    for col in ["location", "target", "output_type", "output_type_id"]:
        out[col] = out[col].astype("string")
    out["value"] = out["value"].astype("float64")
    return out.astype({field.name: pd.ArrowDtype(field.type) for field in HUB_SCHEMA})


def combine_submissions(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate hub tables (e.g. a hosp and an ED output) into one submission.

    Raises if two tables contribute the same (location, target, output_type), since a submission can hold
    only one forecast for each.
    """
    combined = pd.concat([cast_hub_dtypes(t) for t in tables], ignore_index=True)
    keys = ["location", "target", "output_type"]
    per_table = [cast_hub_dtypes(t)[keys].drop_duplicates() for t in tables]
    duplicated = pd.concat(per_table).duplicated(keep=False)
    if duplicated.any():
        clashes = pd.concat(per_table)[duplicated].drop_duplicates().to_records(index=False).tolist()
        msg = f"Forecasts for the same location/target/output_type come from more than one table: {clashes[:5]}"
        raise ValueError(msg)
    return combined.sort_values(["location", "target", "output_type", "output_type_id", "horizon"]).reset_index(
        drop=True
    )


def read_hub_table(path: str | Path) -> pd.DataFrame:
    """Read a hub table from parquet or (gzipped) csv, keeping ids such as FIPS '01' as strings."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype={"location": str, "output_type_id": str})
