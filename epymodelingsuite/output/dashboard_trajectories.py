"""Shared long-form dashboard trajectory schema, formatting and validation."""

import re
from collections.abc import Sequence
from datetime import date

import numpy as np
import pandas as pd
import pyarrow as pa
from epiweeks import Week

DASHBOARD_TRAJECTORY_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.int64(), nullable=False),
        pa.field("sim_id", pa.int64()),
        pa.field("location", pa.string(), nullable=False),
        pa.field("date", pa.timestamp("us"), nullable=False),
        pa.field("target", pa.string(), nullable=False),
        pa.field("series", pa.string(), nullable=False),
        pa.field("value", pa.float64()),
        pa.field("reference_date", pa.timestamp("us"), nullable=False),
        pa.field("horizon", pa.int64(), nullable=False),
        pa.field("epiweek", pa.string(), nullable=False),
    ]
)
KEY = ["sample_id", "location", "date", "target"]


def _check_labels(strains: Sequence[str], baselines: Sequence[str]) -> None:
    if (
        isinstance(strains, str)
        or not strains
        or len(set(strains)) != len(strains)
        or any(not isinstance(s, str) or not s.strip() or s == "total" or s.startswith("baseline") for s in strains)
        or ("single" in strains and len(strains) != 1)
    ):
        raise ValueError("Provide unique, non-reserved strain labels; 'single' must be used alone.")
    if any(re.fullmatch(r"baseline(?:_k[1-9][0-9]*)?", b) is None for b in baselines) or (
        "baseline" in baselines and len(baselines) != 1
    ):
        raise ValueError("Baseline series must be baseline or baseline_k<positive integer>; do not mix them.")


def validate_dashboard_trajectories(df: pd.DataFrame, strains: Sequence[str]) -> list[str]:
    """Return long-form schema/data errors without casting or changing the table.

    Each sample/location/date/target must have all strain series, total, and every
    baseline series present in the file. Strain rows require sim_id and allow
    missing value; total is missing iff all strains are missing. Baselines must
    share total's missingness. Derived rows require null sim_id.
    String/large_string are equivalent. Dates must be timezone-naive timestamp[us].
    """
    schema = DASHBOARD_TRAJECTORY_SCHEMA
    if list(df.columns) != schema.names:
        return [f"Columns must be exactly {schema.names}, in that order."]
    if df.empty:
        return ["Table has no rows."]
    try:
        actual = pa.Schema.from_pandas(df, preserve_index=False)
    except (pa.ArrowException, TypeError, ValueError) as exc:
        return [f"Cannot determine column types: {exc}"]
    errors = []
    for field in schema:
        dtype = actual.field(field.name).type
        if dtype != field.type and not (field.type == pa.string() and dtype == pa.large_string()):
            errors.append(f"{field.name} must have type {field.type}; got {dtype}.")
        if not field.nullable and df[field.name].isna().any():
            errors.append(f"{field.name} must not contain missing values.")
    if errors:
        return errors
    if df.target.nunique() > 1:
        return [
            f"{target}: {error}"
            for target, rows in df.groupby("target")
            for error in validate_dashboard_trajectories(rows, strains)
        ]
    labels = df.series.unique().tolist()
    baselines = [s for s in labels if s.startswith("baseline")]
    try:
        _check_labels(strains, baselines)
    except ValueError as exc:
        return [str(exc)]
    expected_series = [*strains, "total", *baselines]
    if set(labels) != set(expected_series):
        return [f"series must contain exactly {expected_series}; got {labels}."]
    if df.duplicated([*KEY, "series"]).any():
        return ["Duplicate (sample_id, location, date, target, series) rows."]
    if not df.groupby(KEY).size().eq(len(expected_series)).all():
        return ["Every sample/location/date/target must contain all series, including missing strain values."]
    strain_rows = df.series.isin(strains)
    if df.loc[strain_rows, "sim_id"].isna().any():
        errors.append("Strain rows require sim_id.")
    if df.loc[~strain_rows, "sim_id"].notna().any():
        errors.append("Total/baseline rows must have null sim_id.")
    if (df.sample_id < 0).any() or (df.sim_id.dropna() < 0).any():
        errors.append("Sample and simulation IDs must be non-negative.")
    samples = sorted(df.sample_id.unique())
    if samples != list(range(len(samples))):
        errors.append("sample_id must cover 0 .. n_samples-1.")
    if (df.loc[strain_rows].groupby(["location", "sample_id", "target", "series"]).sim_id.nunique() > 1).any():
        errors.append("Source simulation IDs must stay constant within each location/sample/strain.")
    if df.location.str.strip().eq("").any():
        errors.append("location must not be blank.")
    if df.target.str.strip().eq("").any():
        errors.append("target must not be blank; use the original hub target name.")
    dates = pd.DatetimeIndex(df.date)
    references = pd.DatetimeIndex(df.reference_date)
    for name, values in [("date", dates), ("reference_date", references)]:
        if not ((values.dayofweek == 5) & (values == values.normalize())).all():
            errors.append(f"{name} must contain Saturday dates at midnight.")
    if references.nunique() != 1:
        errors.append("reference_date must have a single value.")
    if not np.array_equal(df.horizon.to_numpy(), (dates - references).days // 7):
        errors.append("horizon must equal (date - reference_date) in weeks.")
    weeks = {d: Week.fromdate(d).cdcformat() for d in dates.unique()}
    if not df.epiweek.eq(df.date.map(weeks)).all():
        errors.append("epiweek must match the CDC epiweek of date.")
    values = df.value.to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(values).any() or (values < 0).any():
        errors.append("Non-missing value must be finite and non-negative.")
    wide = df.pivot(index=KEY, columns="series", values="value")
    total = wide["total"].to_numpy(dtype=float, na_value=np.nan)
    expected = wide[list(strains)].sum(axis=1, skipna=True, min_count=1).to_numpy(dtype=float, na_value=np.nan)
    if not np.allclose(total, expected, rtol=1e-10, atol=1e-12, equal_nan=True):
        errors.append(
            "total must equal the sum of available strain values, and be missing iff all strains are missing."
        )
    for baseline in baselines:
        baseline_values = wide[baseline].to_numpy(dtype=float, na_value=np.nan)
        if not np.array_equal(np.isnan(baseline_values), np.isnan(total)):
            errors.append(f"{baseline} must be missing exactly where total is missing.")
        if (baseline_values < total).any():
            errors.append(f"{baseline} must be at least total.")
    return errors


def format_dashboard_trajectories(
    df: pd.DataFrame,
    strains: Sequence[str],
    reference_date: date | str | pd.Timestamp,
) -> pd.DataFrame:
    """Add weekly metadata, safely cast and validate a long-form producer table.

    Requires sample_id, sim_id, location, date, target, series and value. Does not sample
    or aggregate. Recomputes existing weekly metadata; rejects extra columns.
    Returns a new Arrow-backed frame preserving column types in Parquet.
    """
    schema = DASHBOARD_TRAJECTORY_SCHEMA
    if not df.columns.is_unique or "date" not in df:
        raise ValueError("Column names must be unique and include date.")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    reference = pd.Timestamp(reference_date)
    if out.date.dt.tz is not None or reference.tzinfo is not None:
        raise ValueError("date and reference_date must be timezone-naive.")
    out["reference_date"] = reference
    out["horizon"] = (out.date - reference).dt.days // 7
    weeks = {d: Week.fromdate(d).cdcformat() for d in out.date.dropna().unique()}
    out["epiweek"] = out.date.map(weeks)
    if set(out.columns) != set(schema.names):
        raise ValueError(f"Columns must be exactly {schema.names}; got {list(out.columns)}.")
    try:
        table = pa.Table.from_pandas(out[schema.names], schema=schema, preserve_index=False, safe=True)
    except (pa.ArrowException, TypeError, ValueError) as exc:
        raise ValueError(f"Cannot cast dashboard trajectories: {exc}") from exc
    out = table.to_pandas(types_mapper=pd.ArrowDtype)
    errors = validate_dashboard_trajectories(out, strains)
    if errors:
        raise ValueError("Invalid dashboard trajectories: " + "; ".join(errors))
    return out


def dashboard_trajectories_from_wide(
    df: pd.DataFrame,
    strains: Sequence[str],
    reference_date: date | str | pd.Timestamp,
    target: str,
) -> pd.DataFrame:
    """Adapt aggregate results (or legacy wide files) to the long-form contract.

    Preserve source IDs only on strain rows. Total and baseline provenance is
    recovered by joining strain rows on sample_id/location/date/target. Preserve NaN
    strain rows so missing history is distinguishable from zero.
    """
    if not df.columns.is_unique or not all(isinstance(c, str) for c in df):
        raise ValueError("Column names must be unique strings.")
    baselines = [c.removeprefix("target_") for c in df if c.startswith("target_baseline")]
    _check_labels(strains, baselines)
    series = [*strains, "total", *baselines]
    source_key = ["sample_id", "location", "date"]
    required = {*source_key, *[f"sim_id_{s}" for s in strains], *[f"target_{s}" for s in series]}
    if not required.issubset(df.columns) or set(df.columns) - required - {"reference_date", "horizon", "epiweek"}:
        raise ValueError(f"Columns must contain {sorted(required)} and optional weekly metadata.")
    frames = []
    for label in series:
        frame = df[source_key].copy()
        frame["target"] = target
        frame["sim_id"] = df[f"sim_id_{label}"] if label in strains else pd.Series(pd.NA, index=df.index, dtype="Int64")
        frame["series"] = label
        frame["value"] = df[f"target_{label}"]
        frames.append(frame)
    return format_dashboard_trajectories(pd.concat(frames, ignore_index=True), strains, reference_date)
