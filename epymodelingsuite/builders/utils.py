"""Helper utilities for filtering and processing calibration data."""

import pandas as pd

from ..schema.calibration import CalibrationConfig
from ..utils.location import resolve_location
from ..utils.resolved_location import ResolvedLocation


def get_data_in_window(data: pd.DataFrame, calibration: CalibrationConfig, sort: bool = True) -> pd.DataFrame:
    """
    Get data within a specified time window.

    Parameters
    ----------
    data : pd.DataFrame
        The full dataset to filter.
    calibration : CalibrationConfig
        Calibration configuration containing fitting window dates.
    sort : bool, optional
        If True, sort the filtered data by date in ascending order (oldest to newest).
        Default is True to ensure consistent ordering for ABC distance functions.

    Returns
    -------
    pd.DataFrame
        Filtered data within the specified time window, optionally sorted by date.
    """
    if calibration.fitting_window.start_date:
        window_start = calibration.fitting_window.start_date
        window_end = calibration.fitting_window.end_date
    else:
        window_start = calibration.fitting_window.epiweek_start_date
        window_end = calibration.fitting_window.epiweek_end_date

    date_col_name = calibration.comparison[0].observed_date_column
    date_col = pd.to_datetime(data[date_col_name]).dt.date

    mask = (date_col >= window_start) & (date_col <= window_end)
    filtered = data.loc[mask]

    if sort:
        filtered = filtered.sort_values(by=date_col_name).reset_index(drop=True)

    return filtered


def get_data_in_location(
    data: pd.DataFrame,
    population_name: str | ResolvedLocation,
    location_key: str,
    data_location_format: str = "ISO",
) -> pd.DataFrame:
    """Select a region using its saved external identifier, without re-resolving it."""
    location = resolve_location(population_name)
    target = location.format(data_location_format)
    values = data[location_key].astype(str)
    if data_location_format == "epydemix_population":
        return data[values.isin((target, *location.aliases))]
    if data_location_format == "FIPS":
        values = values.str.zfill(2)
    return data[values == target]
