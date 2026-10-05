"""Hubverse content validation based on the R package hubValidations.

Upstream: https://github.com/hubverse-org/hubValidations

This module implements a subset of hubValidations checks in Python for the output types we use
(``quantile``, ``pmf``, ``sample``), driven by the hub's ``tasks.json``.

Covers what our submissions can get wrong: columns and types, allowed task id / output type values, duplicate
rows, required output type ids and output types, value ranges, quantile ordering, pmf sums, sample structure and
``target_end_date = reference_date + horizon`` weeks. File name, location, metadata and submission time checks
are left to the hub's own validation.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .hub_files import HUB_COLUMNS

TASK_ID_COLUMNS = ["reference_date", "target", "horizon", "location", "target_end_date"]
DATE_COLUMNS = ["reference_date", "target_end_date"]
# Same tolerance as R's all.equal(), used by hubValidations for pmf sums
SUM1_TOLERANCE = 1.5e-8


def load_tasks(path: str | Path) -> dict:
    """Read a hub's task configuration from JSON.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the hub's ``tasks.json`` file.

    Returns
    -------
    dict
        Parsed task configuration, without schema validation.
    """
    return json.loads(Path(path).read_text())


def _get_allowed_values(task_id: dict) -> set | None:
    """Collect the allowed values for a task ID.

    Parameters
    ----------
    task_id : dict
        Task ID specification with ``required`` and/or ``optional`` value lists.
        Missing or null lists are treated as empty.

    Returns
    -------
    set[str] or None
        Required and optional values converted to strings. None means that no
        non-missing value is allowed for this task ID.
    """
    values = (task_id.get("required") or []) + (task_id.get("optional") or [])
    return {str(v) for v in values} if values else None


def _normalize(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Normalize hub columns for comparison and collect conversion errors.

    Parameters
    ----------
    df : pd.DataFrame
        Input table containing all ``HUB_COLUMNS``. The input is not modified.

    Returns
    -------
    normalized : pd.DataFrame
        Copy with dates as ``YYYY-MM-DD`` strings, horizons as integer strings,
        identifier columns as strings, and values as numbers. Missing values
        remain missing.
    errors : list[str]
        Problems found while converting dates, horizons or values. The returned
        table should only be validated further when this list is empty.
    """
    errors = []
    out = df.copy()
    for col in DATE_COLUMNS:
        parsed = pd.to_datetime(out[col], format="%Y-%m-%d", errors="coerce")
        if (parsed.isna() & out[col].notna() & (out[col].astype(str) != "")).any():
            errors.append(f"`{col}` has values that are not YYYY-MM-DD dates.")
        out[col] = parsed.dt.strftime("%Y-%m-%d")
    horizon = pd.to_numeric(out["horizon"], errors="coerce")
    if (horizon.notna() & (horizon % 1 != 0)).any() or (horizon.isna() & out["horizon"].notna()).any():
        errors.append("`horizon` has non-integer values.")
        horizon = horizon.round()
    out["horizon"] = horizon.astype("Int64").astype("string")
    for col in ["location", "target", "output_type", "output_type_id"]:
        out[col] = out[col].astype("string")
    value = pd.to_numeric(out["value"], errors="coerce")
    if value.isna().any():
        errors.append(f"`value` has {int(value.isna().sum())} missing or non-numeric values.")
    out["value"] = value
    return out, errors


def _parse_quantile_levels(ids: pd.Series) -> pd.Series:
    """Parse quantile IDs as numeric probability levels.

    Parameters
    ----------
    ids : pd.Series
        Quantile output type IDs, which may contain numeric strings.

    Returns
    -------
    pd.Series
        Numeric levels with the original index. Unparseable IDs become missing
        values, allowing the caller to report invalid quantile levels.
    """
    return pd.to_numeric(ids, errors="coerce")


def validate_model_output(df: pd.DataFrame, tasks: dict) -> list[str]:  # noqa: C901, PLR0912, PLR0915
    """
    Check a model output table against the hub's task configuration.

    Parameters
    ----------
    df : pd.DataFrame
        Model output with the hubverse columns, as read from csv or parquet.
    tasks : dict
        Parsed `tasks.json` of the hub (single round keyed by `reference_date`, as in FluSight).

    Returns
    -------
    list[str]
        Problems found; empty if the table passes.
    """
    missing = [c for c in HUB_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in HUB_COLUMNS]
    if missing or extra:
        return [f"Columns must be exactly {HUB_COLUMNS}; missing {missing}, unexpected {extra}."]
    if df.empty:
        return ["Table has no rows."]

    tbl, errors = _normalize(df)
    if errors:
        return errors

    if tbl["reference_date"].nunique() != 1:  # noqa: PD101
        errors.append(f"`reference_date` must have a single value, got {sorted(tbl['reference_date'].unique(), key=str)}.")

    # Assign each row to the model task (target + output type) it belongs to
    model_tasks = [mt for rnd in tasks["rounds"] for mt in rnd["model_tasks"]]
    tbl["_task"] = -1
    for i, mt in enumerate(model_tasks):
        targets = _get_allowed_values(mt["task_ids"]["target"]) or set()
        in_task = tbl["target"].isin(targets) & tbl["output_type"].isin(mt["output_type"].keys())
        tbl.loc[in_task & (tbl["_task"] == -1), "_task"] = i
    unmatched = tbl["_task"] == -1
    if unmatched.any():
        combos = tbl.loc[unmatched, ["target", "output_type"]].drop_duplicates().to_records(index=False).tolist()
        errors.append(f"{int(unmatched.sum())} rows match no model task (target, output_type): {combos[:5]}.")
    tbl = tbl[~unmatched]

    key = [*TASK_ID_COLUMNS, "output_type", "output_type_id"]
    dupes = tbl.duplicated(subset=key, keep=False)
    if dupes.any():
        errors.append(f"{int(dupes.sum())} rows duplicate the same task ids, output type and output type id.")

    for i, mt in enumerate(model_tasks):
        rows = tbl[tbl["_task"] == i]
        if rows.empty:
            continue
        target = sorted(rows["target"].unique())
        # Task id values
        for col in TASK_ID_COLUMNS:
            allowed = _get_allowed_values(mt["task_ids"].get(col, {}))
            bad = rows[col].notna() if allowed is None else ~rows[col].isin(allowed)
            if bad.any():
                expected = "NA" if allowed is None else "an allowed value"
                errors.append(
                    f"{target}: `{col}` must be {expected}; invalid {sorted(rows.loc[bad, col].astype(str).unique(), key=str)[:5]}."
                )

        for output_type, spec in mt["output_type"].items():
            out = rows[rows["output_type"] == output_type]
            if out.empty:
                if spec.get("is_required"):
                    errors.append(f"{target}: required output type `{output_type}` is missing.")
                continue
            errors += _check_values(out, spec, target, output_type)
            if output_type == "sample":
                errors += _check_samples(out, spec["output_type_id_params"], target)
            else:
                errors += _check_output_type_ids(out, spec, target, output_type)

        # Required output types must accompany every submitted task id combination
        combos = rows[TASK_ID_COLUMNS].drop_duplicates()
        for output_type, spec in mt["output_type"].items():
            if not spec.get("is_required"):
                continue
            have = rows.loc[rows["output_type"] == output_type, TASK_ID_COLUMNS].drop_duplicates()
            lacking = len(combos.merge(have, how="left", indicator=True).query("_merge == 'left_only'"))
            if lacking:
                errors.append(f"{target}: {lacking} task id combinations lack required output type `{output_type}`.")

    # target_end_date = reference_date + horizon weeks
    dated = tbl[tbl["horizon"].notna() & tbl["target_end_date"].notna()]
    expected_end = pd.to_datetime(dated["reference_date"]) + pd.to_timedelta(dated["horizon"].astype(int) * 7, "D")
    off = expected_end.dt.strftime("%Y-%m-%d") != dated["target_end_date"]
    if off.any():
        errors.append(f"{int(off.sum())} rows have target_end_date != reference_date + 7 * horizon days.")
    return errors


def _check_values(out: pd.DataFrame, spec: dict, target: list, output_type: str) -> list[str]:
    """Check numeric forecast values against their configured type and bounds.

    Parameters
    ----------
    out : pd.DataFrame
        Normalized rows for one model task and output type, with numeric values.
    spec : dict
        Output type specification whose ``value`` section defines the type and
        any minimum or maximum bounds.
    target : list[str]
        Target names used to label error messages.
    output_type : str
        Output type name used to label error messages.

    Returns
    -------
    list[str]
        Integer-type and bounds violations, or an empty list when values pass.
    """
    errors = []
    value_spec = spec.get("value", {})
    if value_spec.get("type") == "integer" and (out["value"] % 1 != 0).any():
        errors.append(f"{target} {output_type}: values must be integers.")
    if "minimum" in value_spec and (out["value"] < value_spec["minimum"]).any():
        errors.append(f"{target} {output_type}: values below minimum {value_spec['minimum']}.")
    if "maximum" in value_spec and (out["value"] > value_spec["maximum"]).any():
        errors.append(f"{target} {output_type}: values above maximum {value_spec['maximum']}.")
    return errors


def _check_output_type_ids(out: pd.DataFrame, spec: dict, target: list, output_type: str) -> list[str]:
    """Check quantile or PMF IDs and the associated distribution values.

    Parameters
    ----------
    out : pd.DataFrame
        Normalized rows for one model task and output type.
    spec : dict
        Output type specification with required and optional ``output_type_id``
        values.
    target : list[str]
        Target names used to label error messages.
    output_type : str
        ``"quantile"`` to check increasing quantile values, or ``"pmf"`` to
        check probabilities sum to one within ``SUM1_TOLERANCE``.

    Returns
    -------
    list[str]
        Invalid or missing ID, quantile ordering, and PMF sum errors for each
        task ID combination. Empty when all checks pass.
    """
    errors = []
    ids = spec["output_type_id"]
    required = {str(v) for v in ids.get("required") or []}
    allowed = required | {str(v) for v in ids.get("optional") or []}
    if output_type == "quantile":
        # Compare quantile levels numerically ("0.1" == "0.10")
        canon = {str(float(v)) for v in allowed}
        out = out.assign(output_type_id=_parse_quantile_levels(out["output_type_id"]).map(lambda v: str(float(v))))
        required = {str(float(v)) for v in required}
        allowed = canon
    bad = ~out["output_type_id"].isin(allowed)
    if bad.any():
        errors.append(
            f"{target} {output_type}: invalid output_type_id {sorted(out.loc[bad, 'output_type_id'].unique())[:5]}."
        )

    groups = out.groupby(TASK_ID_COLUMNS, dropna=False)
    incomplete = groups["output_type_id"].agg(lambda s: not required <= set(s)).sum()
    if incomplete:
        errors.append(f"{target} {output_type}: {int(incomplete)} task id combinations miss required output_type_ids.")

    if output_type == "quantile":
        descending = groups.apply(
            lambda g: (
                np.diff(g.sort_values("output_type_id", key=_parse_quantile_levels)["value"].to_numpy()) < 0
            ).any(),
            include_groups=False,
        )
        if descending.any():
            errors.append(
                f"{target} quantile: values decrease with quantile level in {int(descending.sum())} combinations."
            )
    if output_type == "pmf":
        off = (groups["value"].sum() - 1).abs() > SUM1_TOLERANCE
        if off.any():
            errors.append(f"{target} pmf: probabilities don't sum to 1 in {int(off.sum())} combinations.")
    return errors


def _check_samples(out: pd.DataFrame, params: dict, target: list) -> list[str]:
    """Check sample identifiers, counts and trajectory structure.

    Parameters
    ----------
    out : pd.DataFrame
        Normalized sample rows for one model task.
    params : dict
        ``output_type_id_params`` configuration defining compound task ID
        columns, maximum identifier length, and sample count bounds. Task ID
        columns outside the compound set must have matching combinations across
        samples within each compound group.
    target : list[str]
        Target names used to label error messages.

    Returns
    -------
    list[str]
        Identifier, sample count and trajectory coverage violations, or an
        empty list when the samples pass these checks.
    """
    errors = []
    compound = params.get("compound_taskid_set", TASK_ID_COLUMNS)
    non_compound = [c for c in TASK_ID_COLUMNS if c not in compound]

    if out["output_type_id"].isna().any():
        errors.append(f"{target} sample: missing output_type_id.")

    if (out["output_type_id"].str.len() > params.get("max_length", np.inf)).any():
        errors.append(f"{target} sample: output_type_id longer than {params['max_length']} characters.")

    spans = out.groupby("output_type_id")[compound].nunique(dropna=False)
    spanning = spans.index[(spans > 1).any(axis=1)]
    if len(spanning):
        errors.append(
            f"{target} sample: {len(spanning)} sample ids span more than one {compound} combination, e.g. {list(spanning[:3])}."
        )

    n = out.groupby(compound, dropna=False)["output_type_id"].nunique()
    lo, hi = params.get("min_samples_per_task", 1), params.get("max_samples_per_task", np.inf)
    wrong_n = n[(n < lo) | (n > hi)]
    if len(wrong_n):
        errors.append(
            f"{target} sample: {len(wrong_n)} {compound} combinations have a sample count outside [{lo}, {hi}], "
            f"e.g. {wrong_n.head(3).to_dict()}."
        )

    if non_compound:
        # Every sample of a compound combination covers the same non-compound (e.g. horizon) values
        per_sample = out.groupby([*compound, "output_type_id"], dropna=False)[non_compound].apply(
            lambda g: tuple(sorted(map(tuple, g.astype(str).to_numpy())))
        )
        mixed = per_sample.groupby(level=list(range(len(compound)))).nunique()
        if (mixed > 1).any():
            errors.append(
                f"{target} sample: samples cover different {non_compound} values in {int((mixed > 1).sum())} combinations."
            )
    return errors
