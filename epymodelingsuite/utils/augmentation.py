"""Adjust the latest preliminary HHS week for incomplete hospital reporting."""

from collections.abc import Mapping
from typing import Literal

import pandas as pd

HHS_ID_COLUMNS = ["location_iso", "location_code", "target_end_date", "epiweek"]


class HHSDataNotReadyError(ValueError):
    """The latest week lacks admission counts or reporting fractions; retry after the next release."""


def augment_hhs_hospitalizations(
    snapshot: pd.DataFrame,
    *,
    method: Literal["weighted", "average"],
    unreported_weight: float | None = None,
    average_reporting_frac: Mapping[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Adjust the latest week's admissions using the selected reporting correction.

    The latest week is the most recent ``target_end_date`` in the snapshot; earlier weeks are left unchanged.

    Parameters
    ----------
    snapshot : pd.DataFrame
        Output of ``fetch_hhs_hospitalizations(include_reporting_frac=True, preserve_missing_counts=True)``.
    method : {"weighted", "average"}
        Weighted correction for non-reporting hospitals, or normalization to average reporting coverage.
    unreported_weight : float, optional
        Share of the non-reporting hospitals' admissions to add back, between 0 and 1.
        0 keeps reported counts; 1 applies the full inverse of the reporting fraction. Default is 0.5.
        Only accepted with ``method="weighted"``.
    average_reporting_frac : Mapping[str, float], optional
        Required with ``method="average"``. Maps location ISO codes, such as ``US-TX``, to average
        reporting fractions in (0, 1]. Must cover every location in the latest week.

    Returns
    -------
    augmented : pd.DataFrame
        Snapshot without ``reporting_frac``, with only the latest week's ``hospitalizations`` adjusted.
    reporting_data : pd.DataFrame
        Latest-week rows with ``values`` (reported), ``reporting_frac``, ``estimated_reporting``
        (multiplier), ``values_augment`` (adjusted), ``method``, ``unreported_weight``, and
        ``average_reporting_frac``. Parameters unused by the method are missing.

    Raises
    ------
    HHSDataNotReadyError
        A latest-week count or reporting fraction is missing.
    ValueError
        A method or its arguments are invalid, an average is missing, or a reporting fraction is out of range.

    Notes
    -----
    Weighted: ``(r + unreported_weight * (1 - r)) / r``. With the default weight,
    100 admissions at 50% reporting become 150.
    Average: ``average_reporting_frac / r``. This can reduce counts when reporting exceeds the average.
    """
    # Select the latest week and check its inputs.
    latest_date = snapshot["target_end_date"].max()
    latest = snapshot["target_end_date"] == latest_date
    counts = snapshot.loc[latest, "hospitalizations"]
    fractions = snapshot.loc[latest, "reporting_frac"]
    if counts.isna().any() or fractions.isna().any():
        msg = f"Admission counts or reporting fractions are missing for {latest_date:%Y-%m-%d}"
        raise HHSDataNotReadyError(msg)
    if not (fractions.gt(0) & fractions.le(1)).all():
        msg = f"Reporting fractions for {latest_date:%Y-%m-%d} must satisfy 0 < r <= 1"
        raise ValueError(msg)

    average_fractions = float("nan")
    if method == "weighted":
        if average_reporting_frac is not None:
            msg = "average_reporting_frac is only accepted with method='average'"
            raise ValueError(msg)
        unreported_weight = 0.5 if unreported_weight is None else unreported_weight
        if not 0 <= unreported_weight <= 1:
            msg = f"unreported_weight must be between 0 and 1, got {unreported_weight}"
            raise ValueError(msg)
        factor = (fractions + unreported_weight * (1 - fractions)) / fractions
    elif method == "average":
        if average_reporting_frac is None or unreported_weight is not None:
            msg = "method='average' requires average_reporting_frac and does not accept unreported_weight"
            raise ValueError(msg)
        averages = pd.Series(average_reporting_frac, dtype=float)
        if not (averages.gt(0) & averages.le(1)).all():
            msg = "Average reporting fractions must satisfy 0 < r <= 1"
            raise ValueError(msg)

        locations = snapshot.loc[latest, "location_iso"]
        average_fractions = locations.map(averages)
        if average_fractions.isna().any():
            missing = locations[average_fractions.isna()].unique().tolist()
            msg = f"Missing average reporting fractions for {missing}"
            raise ValueError(msg)
        factor = average_fractions / fractions
        unreported_weight = float("nan")
    else:
        msg = f"Unknown augmentation method: {method!r}"
        raise ValueError(msg)

    augmented = snapshot.drop(columns="reporting_frac").astype({"hospitalizations": float})
    augmented.loc[latest, "hospitalizations"] = counts * factor
    # Record the inputs and multipliers used for the latest week
    reporting_data = snapshot.loc[latest, HHS_ID_COLUMNS].assign(
        values=counts,
        reporting_frac=fractions,
        estimated_reporting=factor,
        values_augment=counts * factor,
        unreported_weight=unreported_weight,
        average_reporting_frac=average_fractions,
        method=method,
    )
    return augmented, reporting_data
