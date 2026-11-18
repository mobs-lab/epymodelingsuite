"""Prepare aligned series for ABC calibration."""

from datetime import date
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from epydemix.model.simulation_output import Trajectory

    from epymodelingsuite.schema.calibration import ComparisonSpec


def comparison_observations(
    observed_data: pd.DataFrame, comparisons: list["ComparisonSpec"]
) -> np.ndarray | dict[str, np.ndarray]:
    """Return one observed array or a mapping of comparison names to arrays."""
    if len(comparisons) == 1:
        return observed_data[comparisons[0].observed_value_column].values
    return {comp.key: observed_data[comp.observed_value_column].values for comp in comparisons}


def format_comparison_data(
    results: "Trajectory | None",
    comparisons: list["ComparisonSpec"],
    data_dates: list,
    random_state: dict[str, Any],
    *,
    start_date: date | None = None,
) -> dict[str, Any]:
    """
    Sum each comparison's transitions and align them to the observation dates.

    Missing dates before the simulation starts become NaN. Later missing dates raise ValueError.
    "data" holds the first comparison; multiple comparisons also add each series under
    its name. A failed simulation (``results=None``) returns zero-filled series without
    "date", so output filtering drops it.
    """
    multiple = len(comparisons) > 1
    if results is None:
        zeros = np.full(len(data_dates), 0)
        return {"data": zeros, **({comp.key: zeros for comp in comparisons} if multiple else {})}

    simulation_dates = pd.DatetimeIndex(results.dates)
    target_dates = pd.DatetimeIndex(data_dates)
    missing = target_dates.difference(simulation_dates)
    if start_date is not None or len(simulation_dates):
        start = pd.Timestamp(start_date) if start_date is not None else simulation_dates.min()
        missing = missing[missing >= start]
    if len(missing):
        msg = f"Observed dates missing from the simulation grid: {', '.join(missing.astype(str))}"
        raise ValueError(msg)
    series = {
        comp.key: pd.Series(sum(results.transitions[key] for key in comp.simulation), index=simulation_dates)
        .reindex(target_dates)
        .to_numpy()
        for comp in comparisons
    }

    # TODO: Stop duplicating the first series as "data" for multiple comparisons once
    # dispatcher/output.py and visualization/preparation.py read calibration series by name.
    output = {"data": series[comparisons[0].key], "date": data_dates, "random_state": random_state}
    if multiple:
        output.update(series)
    return output
