"""Adjust the latest preliminary HHS week for incomplete hospital reporting."""

import pandas as pd

HHS_ID_COLUMNS = ["location_iso", "location_code", "target_end_date", "epiweek"]


class HHSDataNotReadyError(ValueError):
    """The latest week lacks admission counts or reporting fractions; retry after the next release."""


def augment_hhs_hospitalizations(
    snapshot: pd.DataFrame, *, unreported_weight: float = 0.5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scale the latest week's admissions up by its reporting fraction.

    The latest week is the most recent ``target_end_date`` in the snapshot; earlier weeks are left unchanged.

    Parameters
    ----------
    snapshot : pd.DataFrame
        Output of ``fetch_hhs_hospitalizations(include_reporting_frac=True, preserve_missing_counts=True)``.
    unreported_weight : float, optional
        Share of the non-reporting hospitals' admissions to add back, between 0 and 1.
        0 keeps reported counts; 1 applies the full inverse of the reporting fraction. Default is 0.5.

    Returns
    -------
    augmented : pd.DataFrame
        Snapshot without ``reporting_frac``, with only the latest week's ``hospitalizations`` adjusted.
    reporting_data : pd.DataFrame
        Latest-week rows with ``values`` (reported), ``reporting_frac``, ``estimated_reporting``
        (multiplier), ``values_augment`` (adjusted), and ``unreported_weight``.

    Raises
    ------
    HHSDataNotReadyError
        A latest-week count or reporting fraction is missing.
    ValueError
        ``unreported_weight`` or a latest-week reporting fraction is out of range.

    Notes
    -----
    For reporting fraction ``r``, the multiplier is ``(r + unreported_weight * (1 - r)) / r``.
    With the default weight, 100 admissions at 50% reporting become 150.
    """
    if not 0 <= unreported_weight <= 1:
        msg = f"unreported_weight must be between 0 and 1, got {unreported_weight}"
        raise ValueError(msg)

    # Select the latest week and check its inputs
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

    # Scale the latest week's counts
    factor = (fractions + unreported_weight * (1 - fractions)) / fractions
    augmented = snapshot.drop(columns="reporting_frac")
    augmented.loc[latest, "hospitalizations"] = counts * factor
    # Record the inputs and multipliers used for the latest week
    reporting_data = snapshot.loc[latest, HHS_ID_COLUMNS].assign(
        values=counts,
        reporting_frac=fractions,
        estimated_reporting=factor,
        values_augment=counts * factor,
        unreported_weight=unreported_weight,
    )
    return augmented, reporting_data
