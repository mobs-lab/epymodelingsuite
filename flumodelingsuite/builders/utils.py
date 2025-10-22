"""Helper utilities for filtering and processing calibration data."""

import pandas as pd
from epydemix.model import EpiModel

from ..schema.calibration import CalibrationConfig
from ..utils import convert_location_name_format


def get_data_in_window(data: pd.DataFrame, calibration: CalibrationConfig) -> pd.DataFrame:
    """
    Get data within a specified time window.

    Parameters
    ----------
    data : pd.DataFrame
        The full dataset to filter.
    calibration : CalibrationConfig
        Calibration configuration containing fitting window dates.

    Returns
    -------
    pd.DataFrame
        Filtered data within the specified time window.
    """
    window_start = calibration.fitting_window.start_date
    window_end = calibration.fitting_window.end_date

    date_col_name = calibration.comparison[0].observed_date_column
    date_col = pd.to_datetime(data[date_col_name]).dt.date

    mask = (date_col >= window_start) & (date_col <= window_end)
    return data.loc[mask]


def get_data_in_location(data: pd.DataFrame, model: EpiModel, location_key: str) -> pd.DataFrame:
    """
    Get data for a specific location.

    Parameters
    ----------
    data : pd.DataFrame
        The full dataset to filter.
    model : EpiModel
        The model whose population name will be used to filter data.
    location_key : str
        The column name containing location identifiers.

    Returns
    -------
    pd.DataFrame
        Filtered data for the model's location.
    """
    location_iso = convert_location_name_format(model.population.name, "ISO")
    # TODO: geo_value column name should be configurable.
    return data[data[location_key] == location_iso]
