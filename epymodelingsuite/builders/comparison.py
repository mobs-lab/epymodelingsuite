"""Prepare aligned series for ABC calibration."""

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


def format_multi_comparison_data(
    results: "Trajectory | None",
    comparisons: list["ComparisonSpec"],
    data_dates: list,
    random_state: dict[str, Any],
) -> dict[str, Any]:
    """Sum each comparison's transitions and align them to the observation dates."""
    if results is None:
        return {key: np.zeros(len(data_dates), dtype=int) for key in ["data", *(comp.key for comp in comparisons)]}

    simulation_dates = pd.DatetimeIndex(results.dates)
    series = {}
    for comp in comparisons:
        values = sum(results.transitions[key] for key in comp.simulation)
        series[comp.key] = pd.Series(values, index=simulation_dates).reindex(data_dates).to_numpy()

    return {
        "data": series[comparisons[0].key],
        "date": data_dates,
        "random_state": random_state,
        **series,
    }
