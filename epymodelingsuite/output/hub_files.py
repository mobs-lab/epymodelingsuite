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

    Parameters
    ----------
    df : pd.DataFrame
        Forecast table containing all ``HUB_COLUMNS``. Dates must be parseable,
        horizons must be nullable integers, and values must be numeric.

    Returns
    -------
    pd.DataFrame
        New table in hub column order with Arrow-backed dtypes: dates and IDs
        as strings, horizon as nullable int32, and value as float64. Dates use
        ``YYYY-MM-DD`` format. Extra columns are omitted; the input is unchanged.

    Raises
    ------
    KeyError
        If a required hub column is missing.
    TypeError or ValueError
        If dates or numeric columns cannot be converted to the required types.

    Notes
    -----
    The returned dtypes preserve ``HUB_SCHEMA`` when written with
    ``DataFrame.to_parquet`` using PyArrow.
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

    Parameters
    ----------
    tables : list[pd.DataFrame]
        Non-empty collection of hub tables, each containing ``HUB_COLUMNS``.
        Different tables must contribute distinct location/target/output-type
        combinations. The input tables are not modified.

    Returns
    -------
    pd.DataFrame
        Combined table with hub dtypes, sorted by location, target, output type,
        output type ID and horizon, with a fresh integer index.

    Raises
    ------
    ValueError
        If the list is empty or multiple tables supply the same
        location/target/output-type combination.
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
    """Read a hub table from Parquet or CSV.

    Parameters
    ----------
    path : str or pathlib.Path
        Input file. A ``.parquet`` suffix selects the Parquet reader; other
        suffixes use the CSV reader, including compressed files such as ``.csv.gz``.

    Returns
    -------
    pd.DataFrame
        Loaded table. CSV location and output type IDs are read as strings to
        preserve leading zeros; Parquet retains its stored column types.
    """
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype={"location": str, "output_type_id": str})
