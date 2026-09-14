from typing import Callable
import pandas as pd
import numpy as np
from epymodelingsuite.builders.utils import get_data_in_location
from epymodelingsuite.utils.location import convert_location_name_format
from epymodelingsuite.schema.dispatcher import CalibrationOutput
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def anchor_projections_on_data(
    runner_output: CalibrationOutput,
    surveillance_data: pd.DataFrame,
    top_fraction: float,
    anchor_start_date: str | pd.Timestamp | datetime,
    anchor_end_date: str | pd.Timestamp | datetime,
    distance_function: Callable[[dict, dict], float],
    surveillance_location_col: str = "location_iso",
    surveillance_target_col: str = "hospitalizations",
    surveillance_date_col: str = "target_end_date",
    simulation_target: str | list[str] = "hospitalizations",
) -> tuple[list[dict], pd.DataFrame]:
    """
    Post process projections to anchor on specific data point(s).

    For each trajectory in the runner output, calculates a distance to the observed
    surveillance data within the specified date range. Returns only the top fraction
    of trajectories with the smallest distances.

    If `simulation_target` is a list, sums the simulation values for all specified targets.
    For "point anchoring", set the same date for both `anchor_start_date` and `anchor_end_date`.

    Parameters
    ----------
    runner_output : CalibrationOutput
        Object containing simulation results, must have `results.projections["baseline"]`
        and `results.projection_parameters["baseline"]`.
    surveillance_data : pd.DataFrame
        DataFrame containing observed data.
    top_fraction : float
        Float between 0 and 1 specifying fraction of projections to keep.
    anchor_start_date : str | pd.Timestamp | datetime
        Start date (inclusive) of the anchor date range for filtering observed data.
    anchor_end_date : str | pd.Timestamp | datetime
        End date (inclusive) of the anchor date range for filtering observed data.
    distance_function : Callable[[dict, dict], float]
        Callable that takes two dicts with key 'data' and returns a float distance.
        The 'data' key should contain a vector of values for comparison.
    surveillance_location_col : str, optional
        Column in `surveillance_data` specifying location, by default "location_iso".
    surveillance_target_col : str, optional
        Column in `surveillance_data` with observed values, by default "hospitalizations".
    surveillance_date_col : str, optional
        Column in `surveillance_data` with dates, by default "target_end_date".
    simulation_target : str | list[str], optional
        Column name or list of column names in simulation projections to compare,
        by default "hospitalizations".

    Returns
    -------
    tuple[list[dict], pd.DataFrame]
        Tuple containing:
        - filtered_projections : list[dict]
            List of projections corresponding to the top fraction (in the same format
            as they appear in CalibrationOutput).
        - filtered_projection_parameters : pd.DataFrame
            DataFrame of parameters corresponding to these projections.

    Raises
    ------
    ValueError
        If `top_fraction` is not in (0, 1], or if no data exists for the anchor date range.
    KeyError
        If expected columns are missing from data or simulation trajectories.
    """
    # Validate inputs
    if not (0 < top_fraction <= 1):
        raise ValueError(f"`top_fraction` must be in (0, 1], got {top_fraction}")

    if surveillance_location_col not in surveillance_data.columns:
        raise KeyError(f"Column '{surveillance_location_col}' not found in surveillance_data")
    if surveillance_target_col not in surveillance_data.columns:
        raise KeyError(f"Column '{surveillance_target_col}' not found in surveillance_data")
    if surveillance_date_col not in surveillance_data.columns:
        raise KeyError(f"Column '{surveillance_date_col}' not found in surveillance_data")

    # Convert anchor dates to Timestamp
    anchor_start_date = pd.Timestamp(anchor_start_date)
    anchor_end_date = pd.Timestamp(anchor_end_date)

    if anchor_start_date > anchor_end_date:
        raise ValueError(
            f"`anchor_start_date` ({anchor_start_date}) must be <= `anchor_end_date` ({anchor_end_date})"
        )

    # Get location and filter data
    location = convert_location_name_format(runner_output.population, "ISO")
    surveillance_data_at_location = get_data_in_location(
        surveillance_data,
        population_name=location,
        location_key=surveillance_location_col
    ).copy()
    surveillance_data_at_location[surveillance_date_col] = pd.to_datetime(
        surveillance_data_at_location[surveillance_date_col]
    )

    # Extract data within the anchor date range
    mask = (
        (surveillance_data_at_location[surveillance_date_col] >= anchor_start_date) &
        (surveillance_data_at_location[surveillance_date_col] <= anchor_end_date)
    )
    filtered_data = surveillance_data_at_location[mask]
    if filtered_data.empty:
        raise ValueError(
            f"No surveillance data available for date range [{anchor_start_date}, {anchor_end_date}] "
            f"at location {location}"
        )

    # Get vector of all data points within the date range 
    surveillance_data_vector = filtered_data[surveillance_target_col].values
    surveillance_data_dict = {"data": surveillance_data_vector}

    # Ensure simulation_target is a list
    if isinstance(simulation_target, str):
        simulation_target_list = [simulation_target]
    else:
        simulation_target_list = simulation_target

    # Check all targets exist in trajectories 
    if len(runner_output.results.projections["baseline"]) == 0:
        raise ValueError("No projections found in runner_output.results.projections['baseline']")
    
    first_trajectory = runner_output.results.projections["baseline"][0]
    for target in simulation_target_list:
        if target not in first_trajectory:
            raise KeyError(f"Simulation trajectory missing expected target column '{target}'")

    # Extract date indices for anchor date range (assuming all trajectories have same length)
    # Get date indices for the anchor range from the first trajectory
    trajectory_dates = first_trajectory["date"]
    date_indices = []
    for idx, date_val in enumerate(trajectory_dates):
        date_ts = pd.Timestamp(date_val)
        if anchor_start_date <= date_ts <= anchor_end_date:
            date_indices.append(idx)
    
    if not date_indices:
        raise ValueError(
            f"No trajectory dates found in range [{anchor_start_date}, {anchor_end_date}]"
        )

    # Compute distances
    distances = []
    for trajectory in runner_output.results.projections["baseline"]:
        # Extract simulation values for all targets within the anchor date range
        simulation_values = []
        for date_idx in date_indices:
            # Sum over all targets if multiple
            value_at_date = sum(trajectory[target][date_idx] for target in simulation_target_list)
            simulation_values.append(value_at_date)
        
        simulation_data_dict = {"data": np.array(simulation_values)}
        distances.append(distance_function(surveillance_data_dict, simulation_data_dict))

    distances = np.array(distances)
    cut = np.quantile(distances, top_fraction)
    idx = np.where(distances <= cut)[0]

    if len(idx) == 0:
        raise ValueError("No projections fall within the top_fraction threshold")

    filtered_projections = [runner_output.results.projections["baseline"][i] for i in idx]
    filtered_projection_parameters = runner_output.results.projection_parameters["baseline"].iloc[idx]

    # Summary statement
    logger.info(
        f"Filtered projections: {len(filtered_projections)} out of "
        f"{len(runner_output.results.projections['baseline'])} trajectories remain after "
        f"applying top_fraction={top_fraction}"
    )
    return filtered_projections, filtered_projection_parameters
