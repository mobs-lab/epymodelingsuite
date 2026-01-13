"""Helper utilities for filtering and processing calibration data."""

import pandas as pd

from ..schema.calibration import CalibrationConfig
from ..utils import convert_location_name_format


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
    window_start = calibration.fitting_window.start_date
    window_end = calibration.fitting_window.end_date

    date_col_name = calibration.comparison[0].observed_date_column
    date_col = pd.to_datetime(data[date_col_name]).dt.date

    mask = (date_col >= window_start) & (date_col <= window_end)
    filtered = data.loc[mask]

    if sort:
        filtered = filtered.sort_values(by=date_col_name).reset_index(drop=True)

    return filtered


def get_data_in_location(
    data: pd.DataFrame, population_name: str, location_key: str, location_format: str = "ISO"
) -> pd.DataFrame:
    """
    Get data for a specific location.

    Parameters
    ----------
    data : pd.DataFrame
        The full dataset to filter.
    population_name : str
        Name of the population/location.
    location_key : str
        The column name containing location identifiers.
    location_format : str
        Format of location identifiers in the data. Options: ISO, FIPS, abbreviation, name, epydemix_population.
        Default is "ISO".

    Returns
    -------
    pd.DataFrame
        Filtered data for the location.
    """
    location_iso = convert_location_name_format(population_name, "ISO")

    if location_format == "ISO":
        # Direct match (current behavior)
        return data[data[location_key] == location_iso]
    # Convert observed data values to ISO using known input format
    data_iso = data[location_key].apply(
        lambda x: convert_location_name_format(str(x), "ISO", input_format=location_format)
    )
    return data[data_iso == location_iso]
