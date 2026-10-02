"""Trajectory samples and file layout for hubverse (FluSight) submissions."""

from collections.abc import Callable
from datetime import date
from pathlib import Path

import io

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

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


def select_random(values: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Pick `n` rows of `values` (trajectories x horizons) uniformly without replacement, or all if fewer."""
    if len(values) <= n:
        return np.arange(len(values))
    return np.sort(rng.choice(len(values), size=n, replace=False))


# Name -> selector. A selector gets one location's complete trajectories (rows x horizons), the number of
# samples wanted and a seeded generator, and returns the selected row indices. Register new methods here.
SAMPLE_SELECTORS: dict[str, Callable[[np.ndarray, int, np.random.Generator], np.ndarray]] = {
    "random": select_random,
}


def select_samples(values: np.ndarray, n: int, method: str = "random", seed: int | None = None) -> np.ndarray:
    """
    Select up to `n` trajectories to submit as samples.

    Trajectories with a NaN in any horizon can't be submitted and are never selected. If fewer than `n`
    complete trajectories remain, all of them are returned (no resampling).

    Parameters
    ----------
    values : np.ndarray
        Values with shape (trajectories, horizons).
    n : int
        Number of samples wanted.
    method : str
        Key of `SAMPLE_SELECTORS`.
    seed : int | None
        Seed for the selector's random generator.

    Returns
    -------
    np.ndarray
        Row indices into `values`.
    """
    complete = np.flatnonzero(~np.isnan(values).any(axis=1))
    chosen = SAMPLE_SELECTORS[method](values[complete], n, np.random.default_rng(seed))
    return complete[chosen]


def make_sample_rows(  # noqa: PLR0913
    values: np.ndarray,
    horizons: list[int],
    reference_date: date,
    location: str,
    target: str,
    id_prefix: str,
    *,
    integer: bool = False,
    upper: float | None = None,
) -> pd.DataFrame:
    """
    Hub rows for selected sample trajectories.

    Parameters
    ----------
    values : np.ndarray
        Selected trajectories with shape (samples, horizons).
    horizons : list[int]
        Horizon of each column of `values`.
    reference_date : date
        Submission reference date.
    location : str
        Hub location id (e.g. FIPS code).
    target : str
        Hub target name.
    id_prefix : str
        Prefix of `output_type_id`; samples are numbered `<prefix>00`, `<prefix>01`, ...
    integer : bool
        Round values to integers (counts).
    upper : float | None
        Clip values to at most this (e.g. 1 for proportions). Values are always clipped to be non-negative.

    Returns
    -------
    pd.DataFrame
        One row per sample and horizon with the hub columns.
    """
    values = np.clip(values, 0, upper)
    if integer:
        values = np.rint(values)
    n_samples, n_horizons = values.shape
    ref = pd.Timestamp(reference_date)
    return pd.DataFrame(
        {
            "reference_date": ref.date(),
            "horizon": np.tile(horizons, n_samples),
            "target_end_date": [(ref + pd.Timedelta(weeks=h)).date() for h in horizons] * n_samples,
            "location": location,
            "target": target,
            "output_type": "sample",
            "output_type_id": np.repeat([f"{id_prefix}{i:02d}" for i in range(n_samples)], n_horizons),
            "value": values.reshape(-1).astype(float),
        }
    )


def to_hub_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast a hub table to the column types of the FluSight example submission parquet.

    Dates become 'YYYY-MM-DD' strings, `horizon` nullable int32, `output_type_id` and the other id columns
    strings, `value` float64. Columns are put in hub order.
    """
    out = df[HUB_COLUMNS].copy()
    for col in ["reference_date", "target_end_date"]:
        out[col] = pd.to_datetime(out[col]).dt.strftime("%Y-%m-%d").astype("string")
    out["horizon"] = out["horizon"].astype("Int32")
    for col in ["location", "target", "output_type", "output_type_id"]:
        out[col] = out[col].astype("string")
    out["value"] = out["value"].astype("float64")
    return out


def hub_parquet_bytes(df: pd.DataFrame) -> bytes:
    """Hub table as parquet bytes with exactly `HUB_SCHEMA`, ready to submit."""
    table = pa.Table.from_pandas(to_hub_dtypes(df), schema=HUB_SCHEMA, preserve_index=False)
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    return buffer.getvalue()


def combine_submissions(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate hub tables (e.g. a hosp and an ED output) into one submission.

    Raises if two tables contribute the same (location, target, output_type), since a submission can hold
    only one forecast for each.
    """
    combined = pd.concat([to_hub_dtypes(t) for t in tables], ignore_index=True)
    keys = ["location", "target", "output_type"]
    per_table = [to_hub_dtypes(t)[keys].drop_duplicates() for t in tables]
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
