from typing import Callable
import pandas as pd
import numpy as np
from epymodelingsuite.builders.utils import get_data_in_location
from epymodelingsuite.utils.location import convert_location_name_format
from .schema.dispatcher import CalibrationOutput
from datetime import datetime

def filter_projections_on_data_point(
    runner_output: CalibrationOutput,
    surveillance_data: pd.DataFrame,
    top_fraction: float,
    final_fitting_date: str | pd.Timestamp | datetime,
    distance_function: Callable[[dict, dict], float],
    surveillance_location_col: str = "location_iso",
    surveillance_target_col: str = "hospitalizations",
    surveillance_date_col: str = "target_end_date",
    simulation_target: str | list[str] = "hospitalizations",
) -> tuple[list[dict], pd.DataFrame]:
    """
    Filters simulation projections based on their similarity to observed surveillance data.

    For each trajectory in the runner output, calculates a distance to the observed
    surveillance data at the specified `final_fitting_date`. Returns only the top
    fraction of trajectories with the smallest distances.

    If `simulation_target` is a list, sums the simulation values at the fitting date
    for all specified targets.

    Args:
        runner_output: Object containing simulation results, must have 
            `results.projections["baseline"]` and `results.projection_parameters["baseline"]`.
        surveillance_data: DataFrame containing observed data.
        top_fraction: Float between 0 and 1 specifying fraction of projections to keep.
        final_fitting_date: Date (string, pd.Timestamp, or datetime) to filter observed data.
        distance_function: Callable that takes two dicts with key 'data' and returns a float.
        surveillance_location_col: Column in `surveillance_data` specifying location.
        surveillance_target_col: Column in `surveillance_data` with observed values.
        surveillance_date_col: Column in `surveillance_data` with dates.
        simulation_target: Column name or list of column names in simulation projections to compare.

    Returns:
        Tuple of:
            - filtered_projections: List of projections corresponding to the top fraction (in the 
            same format as they appear in CalibrationOutput). 
            - filtered_projection_parameters: DataFrame of parameters corresponding to these projections.

    Raises:
        ValueError: If `top_fraction` is not in (0, 1], or if no data exists for `final_fitting_date`.
        KeyError: If expected columns are missing from data.
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

    # Convert final date to Timestamp
    final_fitting_date = pd.Timestamp(final_fitting_date)

    # Get location and filter data
    location = convert_location_name_format(runner_output.population, "ISO")
    surveillance_location = get_data_in_location(
        surveillance_data,
        population_name=location,
        location_key=surveillance_location_col
    ).copy()
    surveillance_location[surveillance_date_col] = pd.to_datetime(surveillance_location[surveillance_date_col])

    # Extract data at the final fitting date
    filtered_data = surveillance_location[surveillance_location[surveillance_date_col] == final_fitting_date]
    if filtered_data.empty:
        raise ValueError(f"No surveillance data available for date {final_fitting_date} at location {location}")

    surveillance_data_at_date = filtered_data[surveillance_target_col].values[0]
    surveillance_data_dict = {"data": surveillance_data_at_date}

    # Ensure simulation_target is a list
    if isinstance(simulation_target, str):
        simulation_target_list = [simulation_target]
    else:
        simulation_target_list = simulation_target

    # Compute distances
    distances = []
    for trajectory in runner_output.results.projections["baseline"]:
        # Check all targets exist in trajectory
        for target in simulation_target_list:
            if target not in trajectory:
                raise KeyError(f"Simulation trajectory missing expected target column '{target}'")

        try:
            date_index = trajectory["date"].index(final_fitting_date)
        except ValueError:
            raise ValueError(f"Final fitting date {final_fitting_date} not found in trajectory dates")

        # Sum over all targets if multiple
        simulation_at_date = sum(trajectory[target][date_index] for target in simulation_target_list)
        simulation_data_dict = {"data": simulation_at_date}
        distances.append(distance_function(surveillance_data_dict, simulation_data_dict))

    distances = np.array(distances)
    cut = np.quantile(distances, top_fraction)
    idx = np.where(distances <= cut)[0]

    if len(idx) == 0:
        raise ValueError("No projections fall within the top_fraction threshold")

    filtered_projections = [runner_output.results.projections["baseline"][i] for i in idx]
    filtered_projection_parameters = runner_output.results.projection_parameters["baseline"].iloc[idx]

    # Summary statement
    print(f"Filtered projections: {len(filtered_projections)} out of {len(runner_output.results.projections['baseline'])} trajectories remain after applying top_fraction={top_fraction}")

    return filtered_projections, filtered_projection_parameters
