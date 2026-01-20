"""Data validation functions for calibration pipeline.

This module provides functions to validate that observed data exists for
calibration before running computationally expensive calibration jobs.
"""

import logging

import pandas as pd

from ..builders.utils import get_data_in_location
from .calibration import CalibrationConfig

logger = logging.getLogger(__name__)


def validate_calibration_data(
    calibration_config: CalibrationConfig,
    population_names: list[str],
    observed_data: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """
    Validate data availability for each location in fitting window.

    This is the core validation logic that operates on in-memory objects.
    Use this when configs and data are already loaded (e.g., inside
    build_calibration). For file-path-based validation, use
    validate_calibration_data_for_config_set instead.

    This function checks whether observed data exists for each location
    within the fitting window. It logs detailed errors for invalid locations
    (useful for CI debugging) and returns lists of valid and invalid locations.

    Parameters
    ----------
    calibration_config : CalibrationConfig
        Calibration configuration containing fitting window and data file info.
    population_names : list[str]
        List of population/location names to validate.
    observed_data : pd.DataFrame
        Observed data already filtered to the fitting window.

    Returns
    -------
    tuple[list[str], list[str]]
        (valid_locations, invalid_locations) - lists of location names.
    """
    calibration = calibration_config.modelset.calibration
    fitting_window = calibration.fitting_window

    # Get fitting window dates for logging
    if fitting_window.start_date:
        window_start = fitting_window.start_date
        window_end = fitting_window.end_date
    else:
        window_start = fitting_window.epiweek_start_date
        window_end = fitting_window.epiweek_end_date

    # Get column info from comparison spec
    comparison = calibration.comparison[0]
    location_column = comparison.observed_location_column
    location_format = comparison.observed_location_format
    date_column = comparison.observed_date_column

    valid = []
    invalid = []

    for location in population_names:
        location_data = get_data_in_location(observed_data, location, location_column, location_format)

        if location_data.empty:
            logger.error(
                "DATA VALIDATION FAILED: No data for location '%s' in fitting window [%s, %s]. Data file: %s",
                location,
                window_start,
                window_end,
                calibration.observed_data_path,
            )
            invalid.append(location)
        else:
            logger.debug(
                "Data validation passed for '%s': %d data points",
                location,
                len(location_data),
            )
            valid.append(location)

            # Warn if data doesn't cover the full fitting window
            data_start = pd.to_datetime(location_data[date_column]).min()
            data_end = pd.to_datetime(location_data[date_column]).max()
            if data_start > pd.to_datetime(window_start) or data_end < pd.to_datetime(window_end):
                logger.warning(
                    "Incomplete data coverage for '%s': data spans [%s, %s] but fitting window is [%s, %s]",
                    location,
                    data_start.date(),
                    data_end.date(),
                    window_start,
                    window_end,
                )

    return valid, invalid


def validate_calibration_data_for_config_set(
    basemodel_path: str,
    calibration_path: str,
) -> tuple[bool, str | None]:
    """
    Validate calibration data availability from config file paths.

    Convenience wrapper that loads configs and data from files, then
    calls validate_calibration_data internally. Designed for external
    scripts (like validate_configs.py) that only have file paths.

    Parameters
    ----------
    basemodel_path : str
        Path to the basemodel YAML configuration file.
    calibration_path : str
        Path to the calibration YAML configuration file.

    Returns
    -------
    tuple[bool, str | None]
        (success, error_message) where success indicates if validation
        passed and error_message contains the error if validation failed.
    """
    try:
        from ..config_loader import (
            load_basemodel_config_from_file,
            load_calibration_config_from_file,
        )
        from ..builders.utils import get_data_in_window
        from ..utils import get_location_codebook

        # Load configs
        basemodel_config = load_basemodel_config_from_file(basemodel_path)
        calibration_config = load_calibration_config_from_file(calibration_path)

        calibration = calibration_config.modelset.calibration

        # Resolve population names (same logic as create_model_collection)
        population_names = calibration_config.modelset.population_names
        if "all" in population_names:
            # Expand to all locations from codebook
            population_names = get_location_codebook()["location_name_epydemix"].tolist()
        elif not population_names:
            # Use population from basemodel
            population_names = [basemodel_config.model.population.name]

        # Load and filter observed data
        observed_raw = pd.read_csv(calibration.observed_data_path)
        observed_in_window = get_data_in_window(observed_raw, calibration)

        # Check if any data in window at all
        if observed_in_window.empty:
            fitting_window = calibration.fitting_window
            if fitting_window.start_date:
                window_start = fitting_window.start_date
                window_end = fitting_window.end_date
            else:
                window_start = fitting_window.epiweek_start_date
                window_end = fitting_window.epiweek_end_date
            return False, f"No data found in fitting window [{window_start}, {window_end}]"

        # Validate each location
        valid_locations, invalid_locations = validate_calibration_data(
            calibration_config=calibration_config,
            population_names=population_names,
            observed_data=observed_in_window,
        )

        if not valid_locations:
            return False, f"No valid data for any location. Invalid: {invalid_locations}"

        if invalid_locations:
            return False, f"Data validation failed for locations: {invalid_locations}"

        return True, None

    except FileNotFoundError as e:
        return False, f"File not found: {e}"
    except Exception as e:
        return False, str(e)
