"""Output generation functions for formatting and saving results."""

import copy
import io
import logging
from collections import defaultdict
from datetime import date, timedelta

import numpy as np
import pandas as pd
from epydemix.calibration import CalibrationResults

from ..schema.dispatcher import CalibrationOutput, SimulationOutput
from ..schema.output import (
    FlusightPropED,
    ObservedValuesConfig,
    OutputConfig,
    OutputObject,
    TabularOutputTypeEnum,
    get_flusight_quantiles,
)
from ..telemetry import ExecutionTelemetry
from ..utils.location import convert_location_name_format, get_flusight_population
from ..visualization.generators import (
    generate_categorical_plots,
    generate_posterior_grid_plot,
    generate_quantile_grid_plot,
    generate_single_location_posterior_plots,
    generate_single_quantile_plots,
)

logger = logging.getLogger(__name__)


# ===== Output Generator Helper Functions =====


def filter_failed_projections(calibration_results: CalibrationResults) -> CalibrationResults:
    """
    Filter out failed projections (empty dicts) from projection results.

    When projections fail, the simulation wrapper returns {}. This function
    removes empty dicts before quantile/trajectory calculations to prevent KeyError when
    epydemix tries to access keys like "date" in get_projection_quantiles() or
    get_projection_trajectories().

    Modifies the calibration_results object in-place by filtering the projections lists.
    Also stores the filtered count on the results object as `_filtered_count` for later reference.

    Parameters
    ----------
    calibration_results : CalibrationResults
        Calibration results with projections attribute (dict mapping scenario_id to list of
        projection dicts). Failed projections are empty dicts {}.

    Returns
    -------
    CalibrationResults
        The same object (modified in-place) with empty dicts filtered out.
        The `_filtered_count` attribute is set to the total number of filtered projections.
    """
    total_filtered = 0
    if hasattr(calibration_results, "projections") and calibration_results.projections:
        # There can be multiple scenarios. The default is "baseline".
        for scenario_id in calibration_results.projections:
            projections = calibration_results.projections[scenario_id]
            if projections:
                # Extract valid projections (non-empty dicts)
                valid_projections = [proj for proj in projections if proj]
                calibration_results.projections[scenario_id] = valid_projections

                # Count and log filtered projections
                filtered_count = len(projections) - len(valid_projections)
                total_filtered += filtered_count
                if filtered_count > 0:
                    logger.warning(
                        "Filtered out %d failed projection(s) for scenario '%s' (kept %d/%d)",
                        filtered_count,
                        scenario_id,
                        len(valid_projections),
                        len(projections),
                    )

    # Store filtered count on results object
    calibration_results._filtered_count = total_filtered

    return calibration_results


def filter_failed_calibration_trajectories(calibration_results: CalibrationResults) -> CalibrationResults:
    """
    Filter out failed calibration trajectories from calibration results.

    When calibration simulations fail, the simulation wrapper returns {}. This function
    removes failed trajectories before quantile/trajectory calculations.

    A valid trajectory dict must contain 'data' and 'date' keys with array-like values.
    Other keys like 'random_state' (which is a dict) are allowed and expected.

    Modifies the calibration_results object in-place by filtering the selected_trajectories dict.

    Parameters
    ----------
    calibration_results : CalibrationResults
        Calibration results with selected_trajectories attribute (dict mapping generation to list of trajectory dicts).
        Failed trajectories are empty dicts {} or missing required keys.

    Returns
    -------
    CalibrationResults
        The same object (modified in-place) with failed trajectories filtered out.
    """
    if hasattr(calibration_results, "selected_trajectories") and calibration_results.selected_trajectories:
        total_filtered = 0
        # Filter each generation's trajectories
        for generation in calibration_results.selected_trajectories:
            trajectories = calibration_results.selected_trajectories[generation]
            if trajectories:
                original_count = len(trajectories)
                # Filter out failed trajectories
                valid_trajectories = []
                for traj in trajectories:
                    # Check if trajectory is empty dict
                    if not traj:
                        continue

                    # Check if trajectory has required keys with valid data
                    is_valid = True

                    # Must have 'data' key with array-like value
                    if "data" not in traj or not isinstance(traj["data"], (list, tuple, np.ndarray)):
                        is_valid = False

                    # Must have 'date' key with array-like value
                    if (is_valid and "date" not in traj) or (
                        is_valid and not isinstance(traj["date"], (list, tuple, np.ndarray))
                    ):
                        is_valid = False

                    if is_valid:
                        valid_trajectories.append(traj)

                calibration_results.selected_trajectories[generation] = valid_trajectories

                filtered_count = original_count - len(valid_trajectories)
                total_filtered += filtered_count
                if filtered_count > 0:
                    logger.warning(
                        "Filtered out %d failed calibration trajectory/trajectories in generation %d (kept %d/%d)",
                        filtered_count,
                        generation,
                        len(valid_trajectories),
                        original_count,
                    )

    return calibration_results


def format_quantiles_flusightforecast(quantiles_df: pd.DataFrame, reference_date: date) -> pd.DataFrame:
    """
    Create FluSight forecast formatted quantile outputs for a single model. Rate-trends are handled separately.

    Parameters
    ----------
    quantiles_df : pd.DataFrame
        Quantile forecast data with columns: date, quantile, hospitalizations
    reference_date : date
        Reference date for calculating forecast horizons

    Returns
    -------
    pd.DataFrame
        Formatted quantile forecasts with FluSight columns (horizon, target, output_type, output_type_id, target_end_date, value)
    """
    formatted = copy.deepcopy(quantiles_df)

    # Horizons required for quantile outputs
    flusight_horizons = range(-1, 4)

    # Create horizon column and filter for appropriate horizons
    formatted.insert(
        0,
        "horizon",
        (formatted.date - pd.to_datetime(reference_date)).apply(lambda x: x / np.timedelta64(1, "W")).astype(int),
    )
    formatted = formatted[formatted.horizon.isin(flusight_horizons)]

    # Name and format remaining fields
    # FRAGILE: the name 'hospitalizations' is user-supplied in the modelset as the column to look for in the surveillance data.
    # Use nullable integer dtype to handle potential NaN values
    formatted.hospitalizations = formatted.hospitalizations.round().astype("Int64")
    formatted.rename(
        columns={"date": "target_end_date", "hospitalizations": "value", "quantile": "output_type_id"}, inplace=True
    )
    formatted.insert(2, "output_type", "quantile")
    formatted.insert(2, "target", "wk inc flu hosp")
    formatted.target_end_date = formatted.target_end_date.apply(lambda x: x.date())

    return formatted


def compare_thresholds_flusightforecast(
    stable_thres: float, change_thres: float, rate_change: float, count_change: float
) -> str:
    """
    Compare the simulated rate-change and count-change against the provided thresholds.

    Comparisons against thresholds are defined in FluSight documentation.
    https://github.com/cdcepi/FluSight-forecast-hub/tree/main/model-output#rate-trend-forecast-specifications

    Parameters
    ----------
    stable_thres : float
        A simulated rate-change with magnitude less than this threshold is stable (unless count_change < 10).
    change_thres : float
        This threshold defines whether non-stable rate-changes are a large increase/decrease or not.
    rate_change : float
        The difference between the last observed rate (/100k population) and the simulated rate (diff = simulated - observed).
    count_change : float
        The difference between the last observed count and the simulated count (diff = simulated - observed).

    Returns
    -------
    str
        A string representing the category of the rate-change ("stable", "increase", "large_increase", "decrease", "large_decrease").
    """
    if abs(rate_change) < stable_thres or abs(count_change) < 10:
        return "stable"
    if 0 < rate_change < change_thres:
        return "increase"
    if change_thres <= rate_change:
        return "large_increase"
    if -change_thres < rate_change < 0:
        return "decrease"
    if rate_change <= -change_thres:
        return "large_decrease"

    msg = f"Unexpected rate_change value: {rate_change} (thresholds: stable={stable_thres}, change={change_thres})"
    raise ValueError(msg)


def categorize_rate_change_flusightforecast(
    rate_change: float, count_change: float, horizon: int, rate_population_scale: int
) -> str:
    """
    Categorize the simulated rate-change using the appropriate thresholds for the horizon.

    Thresholds for different horizons are defined in FluSight documentation.
    https://github.com/cdcepi/FluSight-forecast-hub/tree/main/model-output#rate-trend-forecast-specifications

    Parameters
    ----------
    rate_change : float
        The difference between the last observed rate (/100k population) and the simulated rate (diff = simulated - observed).
    count_change : float
        The difference between the last observed count and the simulated count (diff = simulated - observed).
    horizon : int
        The horizon on which the simulated changes are calculated.

    Returns
    -------
    str
        A string representing the category of the rate-change ("stable", "increase", "large_increase", "decrease", "large_decrease").
    """
    if horizon == 0:
        stable_thres = 0.3
        change_thres = 1.7
        return compare_thresholds_flusightforecast(stable_thres, change_thres, rate_change, count_change)

    if horizon == 1:
        stable_thres = 0.5
        change_thres = 3
        return compare_thresholds_flusightforecast(stable_thres, change_thres, rate_change, count_change)

    if horizon == 2:
        stable_thres = 0.7
        change_thres = 4
        return compare_thresholds_flusightforecast(stable_thres, change_thres, rate_change, count_change)

    if horizon == 3:
        stable_thres = 1
        change_thres = 5
        return compare_thresholds_flusightforecast(stable_thres, change_thres, rate_change, count_change)

    msg = f"Received invalid horizon {horizon}."
    raise ValueError(msg)


def get_projected_value(dates: np.ndarray, values: np.ndarray, target_date: date) -> np.float64:
    """
    Retrieve projected value at a specific target date.

    Parameters
    ----------
    dates : np.ndarray
        Array of dates corresponding to projection time points
    values : np.ndarray
        Array of projected values corresponding to dates
    target_date : date
        The specific date for which to retrieve the projected value

    Returns
    -------
    np.float64
        The projected value at the target date

    Raises
    ------
    AssertionError
        If dates and values arrays have different lengths, or if target_date appears
        multiple times in the dates array
    """
    assert len(dates) == len(values), "Projection dates must match projection values."

    (loc,) = np.where(dates == pd.Timestamp(target_date))

    assert len(loc) == 1, "Received projections with duplicate dates."

    return values[loc[0]]


def make_rate_trends_flusightforecast(
    reference_date: date,
    proj_dates: np.ndarray,
    proj_values: np.ndarray,
    observed: pd.DataFrame,
    population: float,
) -> pd.DataFrame:
    """
    Create FluSight rate-trend forecasts from projection trajectories.

    Parameters
    ----------
    reference_date : date
        Reference date for the forecast
    proj_dates : np.ndarray
        Array of projection date arrays (one per trajectory)
    proj_values : np.ndarray
        Array of projection value arrays (one per trajectory)
    observed : pd.DataFrame
        Observed surveillance data with columns: date, value
    population : float
        Population size for calculating rates per 100k

    Returns
    -------
    pd.DataFrame
        Rate-trend forecasts with columns: horizon, target_end_date, output_type_id, value
    """
    from collections import Counter

    # Horizons required for rate-trend outputs, denominator for rates (i.e. /100k pop)
    flusight_horizons = range(4)  # horizons 0-3
    rate_population_scale = 100000

    # Date of observation for comparison (equivalent to horizon -1)
    obs_date = reference_date - timedelta(weeks=1)
    print(f"obs_date: {obs_date}\nref_date: {reference_date}")

    # Observed value and rate
    obs_val = observed[observed.date == pd.Timestamp(obs_date)].value.iloc[0]
    obs_rate = rate_population_scale * obs_val / population
    print(f"obs_val: {obs_val}\nobs_rate: {obs_rate}")

    # Build list of rows
    rows = []
    for horizon in flusight_horizons:
        # Target date for forecast
        target_date = reference_date + timedelta(weeks=horizon)
        print(f"target_date: {target_date}")

        # Projected values and rates (one for each projection trajectory)
        proj_vals = [
            get_projected_value(dates, values, target_date)
            for dates, values in zip(proj_dates, proj_values, strict=True)
        ]
        proj_rates = (rate_population_scale / population) * np.array(proj_vals)
        print(f"proj_vals: {proj_vals}\nproj_rates: {proj_rates}")

        # Calculate rate-changes and count-changes
        rate_changes = proj_rates - obs_rate
        count_changes = proj_vals - obs_val
        print(f"rate_changes: {rate_changes}\ncount_changes: {count_changes}")

        # Counter containing the categorization for each projection trajectory
        trajectory_categories = Counter(
            [
                categorize_rate_change_flusightforecast(rate_change, count_change, horizon, rate_population_scale)
                for rate_change, count_change in zip(rate_changes, count_changes, strict=True)
            ]
        )
        print(f"traj_cats: {trajectory_categories}")

        # Dict containing the probability of each category
        num_traj = trajectory_categories.total()
        cat_probs = {category: count / num_traj for category, count in trajectory_categories.items()}
        cat_probs.setdefault("stable", 0)
        cat_probs.setdefault("increase", 0)
        cat_probs.setdefault("decrease", 0)
        cat_probs.setdefault("large_increase", 0)
        cat_probs.setdefault("large_decrease", 0)
        print(f"num_traj: {num_traj}\ncat_probs: {cat_probs}")

        # Add rows to the list
        for category, value in cat_probs.items():
            rows.append(
                {"horizon": horizon, "target_end_date": target_date, "output_type_id": category, "value": value}
            )

    return pd.DataFrame.from_records(rows)


def prop_ed_rescaling_factor(observed: np.array, prediction: np.array) -> np.float64:
    """
    Calculate the rescaling factor for converting values from prediction to the scale of values from observed.

    Parameters
    ----------
    observed: np.array
        Array of observed prop ed visits, aligned with `prediction`
    prediction: np.array
        Array of values to rescale to `observed`, aligned with `observed`

    Returns
    -------
    np.float64
        Calculated rescaling factor
    """
    return observed.dot(prediction) / (prediction**2).sum()


def read_surveillance_from_config(config: ObservedValuesConfig) -> pd.DataFrame:
    """
    Read a surveillance file from a configuration object.

    Parameters
    ----------
    config: ObservedValuesConfig

    Returns
    -------
    pd.DataFrame
        Surveillance data read from config
    """
    return pd.read_csv(
        config.data_path, dtype={config.location_column: str}, parse_dates=[config.date_column], date_format="%Y-%m-%d"
    )


def prop_ed_surveillance_window(
    pred_hosp: pd.DataFrame,
    obs_ed: ObservedValuesConfig,
    obs_hosp: ObservedValuesConfig,
    fit_start: date,
    fit_end: date,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create FluSight prop ed forecasts using surveillance_window strategy.

    Parameters
    ----------
    pred_hosp: pd.DataFrame
        Hospitalization quantile forecasts in FluSight format
    obs_ed: ObservedValuesConfig
        Configuration object for observed prop ed visits
    obs_hosp: ObservedValuesConfig
        Configuration object for observed hospitalizations
    fit_start: date
        Start date for rescaling factor fitting window
    fit_end: date
        End date for rescaling factor fitting window

    Returns
    -------
    pd.DataFrame
        Complete prop ed forecasts for a submission (all dates/locations)
    pd.DataFrame
        Record of rescaling factors for each location
    """
    # Read surveillance files from config
    obs_ed_df = read_surveillance_from_config(obs_ed)
    obs_hosp_df = read_surveillance_from_config(obs_hosp)

    # Create forecast
    prop_ed_list = []
    r_dict = defaultdict(list)
    for loc in pred_hosp.location.unique():
        # Convert FIPS location to surveillance data formats for filtering
        loc_hosp = convert_location_name_format(loc, obs_hosp.location_format)
        loc_ed = convert_location_name_format(loc, obs_ed.location_format)

        # Filter forecasts and observations
        filt_pred_hosp = pred_hosp[(pred_hosp.location == loc) & (pred_hosp.output_type == "quantile")].copy(deep=True)
        filt_obs_hosp = (
            obs_hosp_df[
                (obs_hosp_df[obs_hosp.location_column] == loc_hosp)
                & (obs_hosp_df[obs_hosp.date_column] >= pd.to_datetime(fit_start))
                & (obs_hosp_df[obs_hosp.date_column] < pd.to_datetime(fit_end))
            ]
            .dropna(axis=0, subset=obs_hosp.value_column)
            .rename(columns={obs_hosp.value_column: "value_pred"})
        )
        filt_obs_ed = (
            obs_ed_df[
                (obs_ed_df[obs_ed.location_column] == loc_ed)
                & (obs_ed_df[obs_ed.date_column] >= pd.to_datetime(fit_start))
                & (obs_ed_df[obs_ed.date_column] < pd.to_datetime(fit_end))
            ]
            .dropna(axis=0, subset=obs_ed.value_column)
            .rename(columns={obs_ed.value_column: "value_truth"})
        )
        timeseries = filt_obs_hosp.merge(
            filt_obs_ed,
            left_on=[filt_obs_hosp[obs_hosp.location_column], filt_obs_hosp[obs_hosp.date_column]],
            right_on=[filt_obs_ed[obs_ed.location_column], filt_obs_ed[obs_ed.date_column]],
            how="inner",
        )

        # Obtain rescaling and create forecast for location
        r = prop_ed_rescaling_factor(
            np.array(timeseries.value_truth.astype(float)), np.array(timeseries.value_pred.astype(float))
        )
        filt_pred_hosp.value = filt_pred_hosp.value * r
        prop_ed_list.append(filt_pred_hosp)
        r_dict["population"].append(convert_location_name_format(loc, "epydemix_population"))
        r_dict["rescaling_factor"].append(r)

    # Format and return
    prop_ed = pd.concat(prop_ed_list)
    prop_ed.target = "wk inc flu prop ed visits"
    rescaling_factors = pd.DataFrame.from_dict(r_dict, orient="columns")
    return prop_ed, rescaling_factors


def prop_ed_calibration_window(
    pred_hosp: pd.DataFrame, obs_ed: ObservedValuesConfig, calibration_quantiles: pd.DataFrame, num_fit_weeks: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create FluSight prop ed forecasts using calibration_window strategy.

    Parameters
    ----------
    pred_hosp: pd.DataFrame
        Hospitalization quantile forecasts in FluSight format
    obs_ed: ObservedValuesConfig
        Configuration object for observed prop ed visits
    calibration_quantiles: pd.DataFrame | None
        Quantiles from the calibration fitting window
    num_fit_weeks: int
        Number of weeks for rescaling factor fitting window,
        extending back from the end of the calibration fitting window

    Returns
    -------
    pd.DataFrame
        Complete prop ed forecasts for a submission (all dates/locations)
    pd.DataFrame
        Record of rescaling factors for each location
    """
    # Read surveillance file from config
    obs_ed_df = read_surveillance_from_config(obs_ed)

    # TODO: Use ISO in future
    # Convert location codes to FIPS format to match pred_hosp
    obs_ed_df[obs_ed.location_column] = obs_ed_df[obs_ed.location_column].apply(
        lambda x: convert_location_name_format(x, "FIPS")
    )

    # Resolve fitting windows
    fit_end = calibration_quantiles.date.max()
    fit_start = fit_end - timedelta(weeks=num_fit_weeks - 1)
    calibration_window = calibration_quantiles[
        (calibration_quantiles.date >= pd.to_datetime(fit_start)) & (calibration_quantiles["quantile"] == 0.5)
    ].rename(columns={"data": "value_pred"})

    # Create forecast
    prop_ed_list = []
    r_dict = defaultdict(list)
    for loc in pred_hosp.location.unique():
        # Convert FIPS location to surveillance data format for filtering
        loc_ed = convert_location_name_format(loc, obs_ed.location_format)

        # Filter forecasts and observations
        filt_pred_hosp = pred_hosp[(pred_hosp.location == loc) & (pred_hosp.output_type == "quantile")].copy(deep=True)
        filt_obs_ed = (
            obs_ed_df[
                (obs_ed_df[obs_ed.location_column] == loc_ed)
                & (obs_ed_df[obs_ed.date_column] >= fit_start)
                & (obs_ed_df[obs_ed.date_column] <= fit_end)
            ]
            .rename(columns={obs_ed.value_column: "value_truth"})
            .sort_values(by=obs_ed.date_column)
        )
        epy_loc = convert_location_name_format(loc, "epydemix_population")
        filt_calibration = calibration_window[calibration_window.population == epy_loc].copy()
        filt_calibration["location"] = loc
        window = filt_obs_ed.merge(
            filt_calibration,
            left_on=[filt_obs_ed[obs_ed.location_column], filt_obs_ed[obs_ed.date_column]],
            right_on=[filt_calibration["location"], filt_calibration["date"]],
        )
        # Obtain rescaling and create forecast for location
        r = prop_ed_rescaling_factor(np.array(window.value_truth), np.array(window.value_pred))
        filt_pred_hosp.value = filt_pred_hosp.value * r
        prop_ed_list.append(filt_pred_hosp)
        r_dict["population"].append(epy_loc)
        r_dict["rescaling_factor"].append(r)

    # Format and return
    prop_ed = pd.concat(prop_ed_list)
    prop_ed.target = "wk inc flu prop ed visits"
    prop_ed.value = prop_ed.value.apply(lambda x: max(x, 0))
    prop_ed.value = prop_ed.value.apply(lambda x: min(x, 1))
    rescaling_factors = pd.DataFrame.from_dict(r_dict, orient="columns")
    return prop_ed, rescaling_factors


def make_prop_ed_flusightforecast(
    pred_hosp: pd.DataFrame,
    config: FlusightPropED,
    surveillance: dict[str, ObservedValuesConfig],
    calibration_quantiles: pd.DataFrame | None = None,
    projection_quantiles: pd.DataFrame | None = None,
    reference_date: date | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create FluSight prop ed forecasts from hosp forecast and surveillance data.

    Parameters
    ----------
    pred_hosp: pd.DataFrame
        Hospitalization quantile forecasts in FluSight format
    config: FlusightPropED
        Configuration object for target
    surveillance: dict[str, ObservedValuesConfig]
        Dictionary containing configurations for surveillance data (from output.options.surveillance)
    calibration_quantiles: pd.DataFrame | None
        Quantiles from the calibration fitting window (if config.strategy is 'calibration_window')
    projection_quantiles: pd.DataFrame | None
        Quantiles from the projection phase (if config.strategy is 'transition')
    reference_date: date | None
        Reference date for calculating horizons (required if pred_hosp is empty)

    Returns
    -------
    pd.DataFrame
        Complete prop ed forecasts for a submission (all dates/locations)
    pd.DataFrame
        Record of rescaling factors for each location
    """
    match config.strategy:
        case "surveillance_window":
            # Ensure sources are present
            if config.ed_source not in surveillance or config.hosp_source not in surveillance:
                msg = f"prop_ed ed_source '{config.ed_source}' or hosp_source '{config.hosp_source}' not found in output.options.surveillance"
                raise ValueError(msg)

            return prop_ed_surveillance_window(
                pred_hosp,
                surveillance[config.ed_source],
                surveillance[config.hosp_source],
                config.fit_start,
                config.fit_end,
            )
        case "calibration_window":
            # Ensure sources are present
            assert not calibration_quantiles.empty, (
                "Strategy calibration_window requires calibration quantiles, but not received."
            )
            if config.ed_source not in surveillance:
                msg = f"prop_ed ed_source '{config.ed_source}' not found in output.options.surveillance"
                raise ValueError(msg)
            return prop_ed_calibration_window(
                pred_hosp, surveillance[config.ed_source], calibration_quantiles, config.num_fit_weeks
            )
        case "transition":
            # Ensure projection_quantiles is provided
            if projection_quantiles is None or projection_quantiles.empty:
                msg = "Strategy 'transition' requires projection_quantiles, but not received."
                raise ValueError(msg)

            # Ensure transition_name exists in projection_quantiles
            if config.transition_name not in projection_quantiles.columns:
                msg = f"Transition '{config.transition_name}' not found in projection_quantiles. Available columns: {list(projection_quantiles.columns)}"
                raise ValueError(msg)

            # Format the transition quantiles into FluSight format
            formatted = copy.deepcopy(projection_quantiles)

            # Horizons required for quantile outputs
            flusight_horizons = range(-1, 4)

            # Get the reference date from parameter or pred_hosp
            if reference_date is None:
                if not pred_hosp.empty and "reference_date" in pred_hosp.columns:
                    reference_date = pd.to_datetime(pred_hosp.reference_date.iloc[0]).date()
                else:
                    msg = (
                        "Cannot determine reference_date: must provide reference_date parameter or non-empty pred_hosp"
                    )
                    raise ValueError(msg)

            # Create horizon column and filter for appropriate horizons
            formatted.insert(
                0,
                "horizon",
                (formatted.date - pd.to_datetime(reference_date))
                .apply(lambda x: x / np.timedelta64(1, "W"))
                .astype(int),
            )
            formatted = formatted[formatted.horizon.isin(flusight_horizons)]

            # Rename and format columns
            formatted.rename(
                columns={"date": "target_end_date", config.transition_name: "value", "quantile": "output_type_id"},
                inplace=True,
            )
            formatted.insert(2, "output_type", "quantile")
            formatted.insert(2, "target", "wk inc flu prop ed visits")
            formatted.target_end_date = formatted.target_end_date.apply(lambda x: x.date() if hasattr(x, "date") else x)

            # Add location and reference_date columns
            formatted.insert(0, "reference_date", reference_date)
            # Convert population to FIPS format for each row
            formatted.insert(
                0, "location", formatted.population.apply(lambda x: convert_location_name_format(x, "FIPS"))
            )

            # Select only the columns we need for FluSight format
            formatted = formatted[
                [
                    "location",
                    "reference_date",
                    "horizon",
                    "target_end_date",
                    "target",
                    "output_type",
                    "output_type_id",
                    "value",
                ]
            ]

            # Return formatted df and empty rescaling_factors dataframe
            rescaling_factors = pd.DataFrame()
            return formatted, rescaling_factors
        case _:
            raise ValueError(f"Received undefined/unimplemented strategy {config.strategy}")


def format_quantiles_flusmh(quantiles_df: pd.DataFrame) -> pd.DataFrame:
    """"""

    formatted = copy.deepcopy(quantiles_df)
    # TODO
    return pd.DataFrame()


def format_quantiles_covid19forecast(quantiles_df: pd.DataFrame) -> pd.DataFrame:
    """"""

    formatted = copy.deepcopy(quantiles_df)
    # TODO
    return pd.DataFrame()


def format_tabular_object(df: pd.DataFrame, name: str, output_type: TabularOutputTypeEnum) -> OutputObject:
    """
    Create an OutputObject containing tabular data as the requested type.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing tabular data.
    name: str
        Name for identifying tabular data.
    output_type: TabularOutputTypeEnum
        Requested output type, e.g. CSVBytes or DataFrame.

    Returns
    -------
    OutputObject
        Object containing tabular data as requested type.
    """
    match output_type:
        case TabularOutputTypeEnum.CSVBytes:
            return OutputObject(
                output_type=output_type,
                name=f"{name}.csv.gz",
                data=dataframe_to_gzipped_csv(df, header=True, index=False),
            )
        case TabularOutputTypeEnum.DataFrame:
            return OutputObject(output_type=output_type, name=name, data=df)
        case TabularOutputTypeEnum.Parquet:
            msg = "Parquet output not yet implemented."
            logger.warning(msg)
        case _:
            msg = f"Requested undefined tabular object format {output_format}."
            logger.warning(msg)


def dataframe_to_gzipped_csv(df: pd.DataFrame, **csv_kwargs) -> bytes:
    """
    Convert a DataFrame to gzip-compressed CSV bytes.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to convert
    **csv_kwargs
        Additional keyword arguments to pass to DataFrame.to_csv()

    Returns
    -------
    bytes
        Gzip-compressed CSV data as bytes
    """
    buffer = io.BytesIO()
    df.to_csv(buffer, date_format="%Y-%m-%d", compression="gzip", **csv_kwargs)
    return buffer.getvalue()


# ===== Output Generator Registry and Functions =====


OUTPUT_GENERATOR_REGISTRY = {}


def register_output_generator(kind_set):
    """Decorator for output generation dispatch."""

    def deco(fn):
        OUTPUT_GENERATOR_REGISTRY[frozenset(kind_set)] = fn
        return fn

    return deco


@register_output_generator({"simulations", "output_config"})
def generate_simulation_outputs(
    *, simulations: list[SimulationOutput], output_config: OutputConfig, **_
) -> dict[str, list[OutputObject]]:
    """
    Create a dictionary of outputs specified in an OutputConfig for a simulation workflow.

    Parameters
    ----------
        simulations: a list of SimulationOutputs containing SimulationResults.
        output_config: an OutputConfig instance with output specifications.

    Returns
    -------
        A dictionary where keys are intended filenames for writing data, and values are gzip-compressed CSV strings.
    """
    logger.info("OUTPUT GENERATOR: dispatched for simulation")
    output = output_config.output
    warnings = set()

    # Initialize lists for efficient DataFrame concatenation (converted to DataFrames after loops)
    quantiles_compartments_list = []
    quantiles_transitions_list = []
    trajectories_compartments_list = []
    trajectories_transitions_list = []
    hub_format_output = pd.DataFrame()
    model_meta = pd.DataFrame()

    ### Quantiles
    if output.quantiles:
        logger.info("Generating quantile outputs")
        for simulation in simulations:
            # Compartments
            if output.quantiles.compartments:
                quanc_df = simulation.results.get_quantiles_compartments(quantiles=output.quantiles.selections)
                if hasattr(output.quantiles.compartments, "__len__"):
                    try:
                        columns_to_select = ["date", "quantile"]
                        columns_to_select.extend(output.quantiles.compartments)
                        quanc_df = quanc_df[columns_to_select].copy()
                    except Exception as e:
                        warnings.add(
                            f"OUTPUT GENERATOR: Exception occured selecting compartment quantiles, returning all compartments: {e}"
                        )
                quanc_df.insert(0, "primary_id", simulation.primary_id)
                quanc_df.insert(1, "seed", simulation.seed)
                quanc_df.insert(2, "population", simulation.population)
                quantiles_compartments_list.append(quanc_df)

            # Transitions
            if output.quantiles.transitions:
                quant_df = simulation.results.get_quantiles_transitions(quantiles=output.quantiles.selections)
                if hasattr(output.quantiles.transitions, "__len__"):
                    try:
                        columns_to_select = ["date", "quantile"]
                        columns_to_select.extend(output.quantiles.transitions)
                        quant_df = quant_df[columns_to_select].copy()
                    except Exception as e:
                        warnings.add(
                            f"OUTPUT GENERATOR: Exception occured selecting transition quantiles, returning all transitions: {e}"
                        )
                quant_df.insert(0, "primary_id", simulation.primary_id)
                quant_df.insert(1, "seed", simulation.seed)
                quant_df.insert(2, "population", simulation.population)
                quantiles_transitions_list.append(quant_df)

    quantiles_compartments = (
        pd.concat(quantiles_compartments_list, ignore_index=True) if quantiles_compartments_list else pd.DataFrame()
    )
    quantiles_transitions = (
        pd.concat(quantiles_transitions_list, ignore_index=True) if quantiles_transitions_list else pd.DataFrame()
    )

    ### Trajectories
    if output.trajectories:
        logger.info("Generating trajectory outputs")
        for simulation in simulations:
            for i, traj in enumerate(simulation.results.trajectories):
                # Compartments
                if output.trajectories.compartments:
                    trajc_df = pd.DataFrame(traj.compartments)
                    if hasattr(output.trajectories.compartments, "__len__"):
                        try:
                            trajc_df = trajc_df[output.trajectories.compartments]
                        except Exception as e:
                            warnings.add(
                                f"OUTPUT GENERATOR: Exception occured selecting compartment trajectories, returning all compartments: {e}"
                            )
                    trajc_df.insert(0, "primary_id", simulation.primary_id)
                    trajc_df.insert(1, "sim_id", i)
                    trajc_df.insert(2, "seed", simulation.seed)
                    trajc_df.insert(3, "population", simulation.population)
                    trajectories_compartments_list.append(trajc_df)

                # Transitions
                if output.trajectories.transitions:
                    trajt_df = pd.DataFrame(traj.transitions)
                    if hasattr(output.trajectories.transitions, "__len__"):
                        try:
                            trajt_df = trajt_df[output.trajectories.transitions]
                        except Exception as e:
                            warnings.add(
                                f"OUTPUT GENERATOR: Exception occured selecting transition trajectories, returning all transitions: {e}"
                            )
                    trajt_df.insert(0, "primary_id", simulation.primary_id)
                    trajt_df.insert(1, "sim_id", i)
                    trajt_df.insert(2, "seed", simulation.seed)
                    trajt_df.insert(3, "population", simulation.population)
                    trajectories_transitions_list.append(trajt_df)

    trajectories_compartments = (
        pd.concat(trajectories_compartments_list, ignore_index=True)
        if trajectories_compartments_list
        else pd.DataFrame()
    )
    trajectories_transitions = (
        pd.concat(trajectories_transitions_list, ignore_index=True) if trajectories_transitions_list else pd.DataFrame()
    )

    ### Hub Formats

    # Unsupported formats
    if output.flusight_format or output.covid19_format:
        warnings.add("OUTPUT_GENERATOR: Requested forecast hub quantile format for simulation data, ignoring.")

    # Flu SMH
    if output.flusmh_format:
        pass

    ### Model Metadata
    if output.model_meta:
        logger.info("Generating model metadata outputs")
        if output.model_meta.projection_parameters:
            warnings.add("OUTPUT_GENERATOR: Requested projection parameter metadata in simulation workflow, ignoring.")

        meta_dict = defaultdict(list)
        for simulation in simulations:
            meta_dict["primary_id"].append(simulation.primary_id)
            meta_dict["seed"].append(simulation.seed)
            meta_dict["delta_t"].append(simulation.delta_t)
            meta_dict["population"].append(simulation.population)
            meta_dict["n_sims"].append(simulation.results.Nsim)
            meta_dict["start_date"].append(str(sorted(simulation.results.dates)[0]))
            meta_dict["end_date"].append(str(sorted(simulation.results.dates)[-1]))

            # Parameters
            for p, v in simulation.results.parameters.items():
                meta_dict[p].append(str(v))

            # Initial conditions
            inits = {k: [int(v[0]) for v in vs] for k, vs in simulation.results.get_stacked_compartments().items()}
            for c, i in inits:
                colname = f"init_{c}"
                meta_dict[colname].append(str(i))

        model_meta = pd.DataFrame(meta_dict)

    ### Cleanup and return
    for warning in warnings:
        logger.warning(warning)

    logger.info("Formatting tabular outputs")
    out_dict = {}
    if not quantiles_compartments.empty:
        qc_name = "quantiles_compartments"
        qc_objects = [
            format_tabular_object(quantiles_compartments, qc_name, _type) for _type in output.tabular_output_types
        ]
        out_dict[qc_name] = qc_objects
    if not quantiles_transitions.empty:
        qt_name = "quantiles_transitions"
        qt_objects = [
            format_tabular_object(quantiles_transitions, qt_name, _type) for _type in output.tabular_output_types
        ]
        out_dict[qt_name] = qt_objects
    if not trajectories_compartments.empty:
        tc_name = "trajectories_compartments"
        tc_objects = [
            format_tabular_object(trajectories_compartments, tc_name, _type) for _type in output.tabular_output_types
        ]
        out_dict[tc_name] = tc_objects
    if not trajectories_transitions.empty:
        tt_name = "trajectories_transitions"
        tt_objects = [
            format_tabular_object(trajectories_transitions, tt_name, _type) for _type in output.tabular_output_types
        ]
        out_dict[tt_name] = tt_objects
    if not hub_format_output.empty:
        # will want to build filename to be something better, like to fit hub standards
        hf_name = "output_hub_formatted"
        hf_objects = [format_tabular_object(hub_format_output, hf_name, _type) for _type in output.tabular_output_types]
        out_dict[hf_name] = hf_objects
    if not model_meta.empty:
        mm_name = "model_metadata"
        mm_objects = [format_tabular_object(model_meta, mm_name, _type) for _type in output.tabular_output_types]
        out_dict[mm_name] = mm_objects

    logger.info("Output generation complete. Generated %d output types", len(out_dict))
    logger.info("OUTPUT GENERATOR: completed for simulation")

    return out_dict


@register_output_generator({"calibrations", "output_config"})
def generate_calibration_outputs(
    *, calibrations: list[CalibrationOutput], output_config: OutputConfig, **_
) -> dict[str, list[OutputObject]]:
    """
    Create a dictionary of outputs specified in an OutputConfig for a calibration workflow.

    Parameters
    ----------
        calibrations: a list of CalibrationOutputs containing CalibrationResults.
        output_config: an OutputConfig instance with output specifications.

    Returns
    -------
        A dictionary where keys are intended filenames for writing data, and values are gzip-compressed CSV strings.
    """
    logger.info("OUTPUT GENERATOR: dispatched for calibration")
    output = output_config.output
    warnings = set()

    # Initialize structures for efficient DataFrame concatenation (converted to DataFrames after loops)
    quantiles_projection_compartments_list = []
    quantiles_projection_transitions_list = []
    quantiles_calibration_list = []
    trajectories_projection_compartments_list = []
    trajectories_projection_transitions_list = []
    posteriors_list = []
    hub_format_output_list = []
    meta_dict = defaultdict(list)

    # Filter out failed projections
    for calibration in calibrations:
        calibration.results = filter_failed_projections(calibration.results)

    ### Quantiles
    if output.quantiles:
        logger.info("Generating quantile outputs")
        for calibration in calibrations:  # Calibration quantiles (only for calibration comparison target)
            # Calibration quantiles
            if output.quantiles.calibration:
                if hasattr(output.quantiles.calibration, "__len__"):
                    for generation in output.quantiles.calibration:
                        try:
                            quancal_df = calibration.results.get_calibration_quantiles(
                                quantiles=output.quantiles.selections, generation=generation, variables=["data", "date"]
                            )
                            quancal_df.insert(0, "primary_id", calibration.primary_id)
                            quancal_df.insert(1, "seed", calibration.seed)
                            quancal_df.insert(2, "population", calibration.population)
                            quancal_df.insert(3, "generation", generation)
                            quantiles_calibration_list.append(quancal_df)
                        except Exception as e:
                            warnings.add(
                                f"OUTPUT GENERATOR: Exception occured obtaining calibration quantiles for model with primary_id={calibration.primary_id}, generation {generation}, continuing to next generation. Message: {e}"
                            )
                else:
                    try:
                        quancal_df = calibration.results.get_calibration_quantiles(
                            quantiles=output.quantiles.selections, variables=["data", "date"]
                        )
                        quancal_df.insert(0, "primary_id", calibration.primary_id)
                        quancal_df.insert(1, "seed", calibration.seed)
                        quancal_df.insert(2, "population", calibration.population)
                        quantiles_calibration_list.append(quancal_df)
                    except Exception as e:
                        warnings.add(
                            f"OUTPUT GENERATOR: Exception occured obtaining calibration quantiles for model with primary_id={calibration.primary_id}, continuing. Message: {e}"
                        )

            # Projection quantiles
            try:
                quan_df = calibration.results.get_projection_quantiles(quantiles=output.quantiles.selections)
            except ValueError:
                warnings.add(
                    f"OUTPUT GENERATOR: failed to obtain projection quantiles for model with primary_id={calibration.primary_id}, continuing to next model."
                )
                continue

            try:
                transition_columns = [c for c in quan_df.columns if "_to_" in c]

                # Compartments
                if output.quantiles.compartments:
                    if hasattr(output.quantiles.compartments, "__len__"):
                        # Filter for explicitly requested compartments
                        try:
                            columns_to_select = ["date", "quantile"]
                            columns_to_select.extend(output.quantiles.compartments)
                            quanc_df = quan_df[columns_to_select].copy()
                        except KeyError as e:
                            warnings.add(
                                f"OUTPUT GENERATOR: Exception occured selecting compartment quantiles, returning all compartments: {e}"
                            )
                            # Use all compartments, filter out transitions
                            quanc_df = quan_df.copy()
                            quanc_df.drop(columns=transition_columns, inplace=True)
                    else:
                        # Use all compartments, filter out transitions
                        quanc_df = quan_df.copy()
                        quanc_df.drop(columns=transition_columns, inplace=True)
                    quanc_df.insert(0, "primary_id", calibration.primary_id)
                    quanc_df.insert(1, "seed", calibration.seed)
                    quanc_df.insert(2, "population", calibration.population)
                    quantiles_projection_compartments_list.append(quanc_df)

                # Transitions
                if output.quantiles.transitions:
                    if hasattr(output.quantiles.transitions, "__len__"):
                        # Filter for explicitly requested transitions
                        try:
                            columns_to_select = ["date", "quantile"]
                            columns_to_select.extend(output.quantiles.transitions)
                            quant_df = quan_df[columns_to_select].copy()
                        except Exception as e:
                            warnings.add(
                                f"OUTPUT GENERATOR: Exception occured selecting compartment quantiles, returning all transitions: {e}"
                            )
                            # Use all transitions, filter out compartments
                            # TODO: add target prediction data column name below
                            columns_to_select = ["date", "quantile"]
                            columns_to_select.extend(transition_columns)
                            quant_df = quan_df[columns_to_select].copy()
                    else:
                        # Use all transitions, filter out compartments
                        # TODO: add target prediction data column name below
                        columns_to_select = ["date", "quantile"]
                        columns_to_select.extend(transition_columns)
                        quant_df = quan_df[columns_to_select].copy()
                    quant_df.insert(0, "primary_id", calibration.primary_id)
                    quant_df.insert(1, "seed", calibration.seed)
                    quant_df.insert(2, "population", calibration.population)
                    quantiles_projection_transitions_list.append(quant_df)
            except Exception as e:
                warnings.add(
                    f"OUTPUT GENERATOR: Exception occurred processing quantile outputs for model with primary_id={calibration.primary_id}, continuing to next model. Message: {e}"
                )

    quantiles_projection_compartments = (
        pd.concat(quantiles_projection_compartments_list, ignore_index=True)
        if quantiles_projection_compartments_list
        else pd.DataFrame()
    )
    quantiles_projection_transitions = (
        pd.concat(quantiles_projection_transitions_list, ignore_index=True)
        if quantiles_projection_transitions_list
        else pd.DataFrame()
    )
    quantiles_calibration = (
        pd.concat(quantiles_calibration_list, ignore_index=True) if quantiles_calibration_list else pd.DataFrame()
    )

    ### Trajectories
    if output.trajectories:
        logger.info("Generating trajectory outputs")
        for calibration in calibrations:
            # Collect all trajectories
            try:
                traj = calibration.results.get_projection_trajectories()
            except Exception:
                warnings.add(
                    f"OUTPUT GENERATOR: failed to obtain projection trajectories for model with primary_id={calibration.primary_id}, continuing to next model."
                )
                continue

            trajectories_list = []
            for i in range(len(traj["date"])):
                columns = []
                for name, values in traj.items():
                    columns.append(pd.Series(values[i], name=name))
                traj_df = pd.concat(columns, axis=1)
                traj_df.insert(0, "sim_id", i)
                trajectories_list.append(traj_df)

            trajectories = pd.concat(trajectories_list, ignore_index=True) if trajectories_list else pd.DataFrame()
            transition_columns = [c for c in trajectories.columns if "_to_" in c]

            # Compartments
            if output.trajectories.compartments:
                traj_c = trajectories.copy()
                if hasattr(output.trajectories.compartments, "__len__"):
                    # Filter for explicitly requested compartments
                    try:
                        columns_to_select = ["sim_id", "date"]
                        columns_to_select.extend(output.trajectories.compartments)
                        traj_c = traj_c[columns_to_select].copy()
                    except Exception:
                        warnings.add(
                            "OUTPUT GENERATOR: failed to filter trajectories for selected compartments, returning all compartments."
                        )
                        # Use all compartments, filter out transitions
                        traj_c.drop(columns=transition_columns, inplace=True)
                else:
                    # Use all compartments, filter out transitions
                    traj_c.drop(columns=transition_columns, inplace=True)
                traj_c.insert(0, "primary_id", calibration.primary_id)
                traj_c.insert(2, "seed", calibration.seed)
                traj_c.insert(3, "population", calibration.population)
                trajectories_projection_compartments_list.append(traj_c)

            # Transitions
            if output.trajectories.transitions:
                traj_t = trajectories.copy()
                if hasattr(output.trajectories.transitions, "__len__"):
                    # filter for explicitly requested transitions
                    try:
                        columns_to_select = ["sim_id", "date"]
                        columns_to_select.extend(output.trajectories.transitions)
                        traj_t = traj_t[columns_to_select].copy()
                    except Exception:
                        warnings.add(
                            "OUTPUT GENERATOR: failed to filter trajectories for selected transitions, returning all transitions."
                        )
                        # Use all transitions, filter out compartments
                        columns_to_select = ["sim_id", "date"]
                        columns_to_select.extend(transition_columns)
                        traj_t = traj_t[columns_to_select].copy()
                else:
                    # Use all transitions, filter out compartments
                    columns_to_select = ["sim_id", "date"]
                    columns_to_select.extend(transition_columns)
                    traj_t = traj_t[columns_to_select].copy()
                traj_t.insert(0, "primary_id", calibration.primary_id)
                traj_t.insert(2, "seed", calibration.seed)
                traj_t.insert(3, "population", calibration.population)
                trajectories_projection_transitions_list.append(traj_t)

    trajectories_projection_compartments = (
        pd.concat(trajectories_projection_compartments_list, ignore_index=True)
        if trajectories_projection_compartments_list
        else pd.DataFrame()
    )
    trajectories_projection_transitions = (
        pd.concat(trajectories_projection_transitions_list, ignore_index=True)
        if trajectories_projection_transitions_list
        else pd.DataFrame()
    )

    ### Posteriors
    if output.posteriors:
        logger.info("Generating posterior outputs")
        for calibration in calibrations:
            # Output last generation (default)
            if output.posteriors == True:
                post_df = calibration.results.get_posterior_distribution()
            # Output selected generations
            elif output.posteriors.generations:
                post_df_list = []
                for g in output.posteriors.generations:
                    try:
                        post = calibration.results.get_posterior_distribution(generation=g)
                        post.insert(0, "generation", g)
                        post_df_list.append(post)
                    except Exception:
                        warnings.add(
                            f"OUTPUT GENERATOR: failed to obtain posterior for generation {g} from model with primary_id={calibration.primary_id}, continuing."
                        )
                post_df = pd.concat(post_df_list, ignore_index=True) if post_df_list else pd.DataFrame()
            # Undefined behavior
            else:
                msg = f"Received unexpected value for posteriors output config (should be bool or list of int): {output.posteriors}"
                raise ValueError(msg)
            # Record identifiers and add to list
            if not post_df.empty:
                post_df.insert(0, "primary_id", calibration.primary_id)
                post_df.insert(1, "seed", calibration.seed)
                post_df.insert(2, "population", calibration.population)
                posteriors_list.append(post_df)

    posteriors = pd.concat(posteriors_list, ignore_index=True) if posteriors_list else pd.DataFrame()

    ### Hub Formats

    # FluSight Forecast Hub
    logger.info(f"DEBUG: output.flusight_format = {output.flusight_format}")
    if output.flusight_format:
        logger.info("Generating FluSight forecast hub outputs")

        # Quantile forecasts (hospitalizations)
        if output.flusight_format.hospitalizations:
            logger.info("  - Generating FluSight quantile forecasts (hospitalizations)")
            for calibration in calibrations:
                try:
                    # FRAGILE: the name 'hospitalizations' is user-supplied in the modelset as the column to look for in the surveillance data.
                    quanf_df = calibration.results.get_projection_quantiles(
                        quantiles=get_flusight_quantiles(), variables=["date", "quantile", "hospitalizations"]
                    )
                except ValueError:
                    warnings.add(
                        f"OUTPUT GENERATOR: failed to obtain projection quantiles for model with primary_id={calibration.primary_id}, continuing to next model."
                    )
                    continue
                quanf_df = format_quantiles_flusightforecast(quanf_df, output.flusight_format.reference_date)
                quanf_df.insert(0, "reference_date", output.flusight_format.reference_date)
                quanf_df.insert(0, "location", convert_location_name_format(calibration.population, "FIPS"))
                hub_format_output_list.append(quanf_df)

        # Prop ED forecasts
        if output.flusight_format.prop_ed:
            # Get surveillance source configuration (not needed for transition strategy)
            if output.flusight_format.prop_ed.strategy != "transition":
                if not output.options or not output.options.surveillance:
                    msg = "prop_ed specified but no surveillance sources defined in output.options.surveillance"
                    raise ValueError(msg)

            # Make calibration quantiles with flusight required quantiles if using calibration_window strategy
            quantiles_calibration_flusight_list = []
            if output.flusight_format.prop_ed.strategy == "calibration_window":
                for calibration in calibrations:
                    try:
                        quancalflu_df = calibration.results.get_calibration_quantiles(
                            quantiles=get_flusight_quantiles(), variables=["data", "date"]
                        )
                        quancalflu_df.insert(0, "primary_id", calibration.primary_id)
                        quancalflu_df.insert(1, "seed", calibration.seed)
                        quancalflu_df.insert(2, "population", calibration.population)
                        quantiles_calibration_flusight_list.append(quancalflu_df)
                    except Exception as e:
                        warnings.add(
                            f"OUTPUT GENERATOR: Exception occured obtaining calibration quantiles for model with primary_id={calibration.primary_id} during prop_ed forecast generation, continuing. Message: {e}"
                        )

            # Make projection quantiles with transition if using transition strategy
            quantiles_projection_flusight_list = []
            if output.flusight_format.prop_ed.strategy == "transition":
                transition_name = output.flusight_format.prop_ed.transition_name
                for calibration in calibrations:
                    try:
                        quanproj_df = calibration.results.get_projection_quantiles(
                            quantiles=get_flusight_quantiles(), variables=["date", "quantile", transition_name]
                        )
                        quanproj_df.insert(0, "primary_id", calibration.primary_id)
                        quanproj_df.insert(1, "seed", calibration.seed)
                        quanproj_df.insert(2, "population", calibration.population)
                        quantiles_projection_flusight_list.append(quanproj_df)
                    except Exception as e:
                        warnings.add(
                            f"OUTPUT GENERATOR: Exception occured obtaining projection quantiles for transition '{transition_name}' for model with primary_id={calibration.primary_id} during prop_ed forecast generation, continuing. Message: {e}"
                        )

            # Collect data and generate prop ed forecast
            quantiles_calibration_flusight = (
                pd.concat(quantiles_calibration_flusight_list) if quantiles_calibration_flusight_list else None
            )
            quantiles_projection_flusight = (
                pd.concat(quantiles_projection_flusight_list) if quantiles_projection_flusight_list else None
            )
            hosp_forecast = (
                pd.concat(hub_format_output_list, ignore_index=True) if hub_format_output_list else pd.DataFrame()
            )
            prop_ed_df, rescaling_factors = make_prop_ed_flusightforecast(
                hosp_forecast,
                output.flusight_format.prop_ed,
                output.options.surveillance if output.options else {},
                quantiles_calibration_flusight,
                quantiles_projection_flusight,
                output.flusight_format.reference_date,
            )
            hub_format_output_list.append(prop_ed_df)

        # Rate-trend forecasts (only if hospitalizations is enabled)
        if output.flusight_format.rate_trends_source and output.flusight_format.hospitalizations:
            logger.info("  - Generating FluSight rate-trend forecasts")
            # Get surveillance source configuration
            if not output.options or not output.options.surveillance:
                msg = "rate_trends_source specified but no surveillance sources defined in output.options.surveillance"
                raise ValueError(msg)

            source_name = output.flusight_format.rate_trends_source
            if source_name not in output.options.surveillance:
                msg = f"rate_trends_source '{source_name}' not found in output.options.surveillance"
                raise ValueError(msg)

            source_config = output.options.surveillance[source_name]

            # Read surveillance data
            surveillance = read_surveillance_from_config(source_config)

            for calibration in calibrations:
                # Get trajectories
                try:
                    traj = calibration.results.get_projection_trajectories()
                except Exception:
                    warnings.add(
                        f"OUTPUT GENERATOR: failed to obtain projection trajectories for model with primary_id={calibration.primary_id}, continuing to next model."
                    )
                    continue

                # Filter surveillance for location
                # Convert calibration population to surveillance data's location format
                target_location = convert_location_name_format(calibration.population, source_config.location_format)
                surv = surveillance[surveillance[source_config.location_column] == target_location]
                surv = surv.drop(columns=source_config.location_column).rename(
                    columns={
                        source_config.date_column: "date",
                        source_config.value_column: "value",
                    }
                )

                # Calculate rate-trend forecasts and add to output
                # FRAGILE: use of name 'hospitalizations'
                trends_df = make_rate_trends_flusightforecast(
                    reference_date=output.flusight_format.reference_date,
                    proj_dates=traj["date"],
                    proj_values=traj["hospitalizations"],
                    observed=surv,
                    population=get_flusight_population(calibration.population),
                )
                trends_df.insert(0, "target", "wk flu hosp rate change")
                trends_df.insert(0, "output_type", "pmf")
                trends_df.insert(0, "reference_date", output.flusight_format.reference_date)
                trends_df.insert(0, "location", convert_location_name_format(calibration.population, "FIPS"))
                hub_format_output_list.append(trends_df)

        hub_format_output = (
            pd.concat(hub_format_output_list, ignore_index=True) if hub_format_output_list else pd.DataFrame()
        )

    # Covid19 Forecast Hub
    elif output.covid19_format:
        hub_format_output = pd.DataFrame()  # TODO: implement covid19 format

    else:
        # No hub format specified
        hub_format_output = pd.DataFrame()

    ### Model Metadata
    if output.model_meta:
        logger.info("Generating model metadata outputs")
        for calibration in calibrations:
            # Skip calibrations without projections (failed calibrations)
            if not calibration.results.projections:
                logger.warning(
                    "Skipping model metadata for primary_id=%s (%s) - no projections available",
                    calibration.primary_id,
                    calibration.population,
                )
                continue

            meta_dict["primary_id"].append(calibration.primary_id)
            meta_dict["seed"].append(calibration.seed)
            meta_dict["delta_t"].append(calibration.delta_t)
            meta_dict["population"].append(calibration.population)
            if calibration.results.projections:
                for scenario_id in calibration.results.projections:
                    colname = f"n_projections_{scenario_id}"
                    meta_dict[colname].append(len(calibration.results.projections[scenario_id]))

            # Fitting window
            try:
                trajc = calibration.results.get_calibration_trajectories()
                if trajc and "date" in trajc and len(trajc["date"]) > 0:
                    meta_dict["fitting_start"].append(str(sorted(trajc["date"][0])[0].date()))
                    meta_dict["fitting_end"].append(str(sorted(trajc["date"][0])[-1].date()))
                else:
                    meta_dict["fitting_start"].append(None)
                    meta_dict["fitting_end"].append(None)
            except (KeyError, IndexError, AttributeError, TypeError) as e:
                logger.warning("Failed to extract fitting window dates: %s", e)
                meta_dict["fitting_start"].append(None)
                meta_dict["fitting_end"].append(None)

            # Projection window
            try:
                trajp = calibration.results.get_projection_trajectories()
                if trajp and "date" in trajp and len(trajp["date"]) > 0:
                    meta_dict["start_date"].append(str(sorted(trajp["date"][0])[0].date()))
                    meta_dict["end_date"].append(str(sorted(trajp["date"][0])[-1].date()))
                else:
                    meta_dict["start_date"].append(None)
                    meta_dict["end_date"].append(None)
            except (KeyError, IndexError, AttributeError, TypeError, ValueError) as e:
                logger.warning("Failed to extract projection window dates: %s", e)
                meta_dict["start_date"].append(None)
                meta_dict["end_date"].append(None)

            # Projection parameters
            # No calibration parameters, this is covered by posteriors.
            if output.model_meta.projection_parameters:
                for scenario_id in calibration.results.projections:
                    proj_params = calibration.results.projection_parameters[scenario_id]
                    for p in proj_params:
                        colname = f"proj_{scenario_id}_{p}"
                        meta_dict[colname].append(str(proj_params[p]))

        model_meta = pd.DataFrame(meta_dict)
        if output.flusight_format and output.flusight_format.prop_ed and not rescaling_factors.empty:
            model_meta = model_meta.merge(rescaling_factors, on="population")

    ### Cleanup and return
    for warning in warnings:
        logger.warning(warning)

    logger.info("Formatting tabular outputs")
    out_dict = {}
    if not quantiles_projection_compartments.empty:
        qc_name = "quantiles_projection_compartments"
        qc_objects = [
            format_tabular_object(quantiles_projection_compartments, qc_name, _type)
            for _type in output.tabular_output_types
        ]
        out_dict[qc_name] = qc_objects
    if not quantiles_projection_transitions.empty:
        qt_name = "quantiles_projection_transitions"
        qt_objects = [
            format_tabular_object(quantiles_projection_transitions, qt_name, _type)
            for _type in output.tabular_output_types
        ]
        out_dict[qt_name] = qt_objects
    if not quantiles_calibration.empty:
        qcal_name = "quantiles_calibration"
        qcal_objects = [
            format_tabular_object(quantiles_calibration, qcal_name, _type) for _type in output.tabular_output_types
        ]
        out_dict[qcal_name] = qcal_objects
    if not trajectories_projection_compartments.empty:
        tc_name = "trajectories_projection_compartments"
        tc_objects = [
            format_tabular_object(trajectories_projection_compartments, tc_name, _type)
            for _type in output.tabular_output_types
        ]
        out_dict[tc_name] = tc_objects
    if not trajectories_projection_transitions.empty:
        tt_name = "trajectories_projection_transitions"
        tt_objects = [
            format_tabular_object(trajectories_projection_transitions, tt_name, _type)
            for _type in output.tabular_output_types
        ]
        out_dict[tt_name] = tt_objects
    if not posteriors.empty:
        p_name = "posteriors"
        p_objects = [format_tabular_object(posteriors, p_name, _type) for _type in output.tabular_output_types]
        out_dict[p_name] = p_objects
    if not hub_format_output.empty:
        # will want to build filename to be something better, like to fit hub standards
        hf_name = "output_hub_formatted"
        hf_objects = [format_tabular_object(hub_format_output, hf_name, _type) for _type in output.tabular_output_types]
        out_dict[hf_name] = hf_objects
    if not model_meta.empty:
        mm_name = "model_metadata"
        mm_objects = [format_tabular_object(model_meta, mm_name, _type) for _type in output.tabular_output_types]
        out_dict[mm_name] = mm_objects

    ### Visualization plots
    if output.plots:
        logger.info("Generating visualization plots")
        plots_config = output.plots

        # Extract start_date_reference from first calibration (they should all have the same reference date)
        start_date_reference = None
        for calibration in calibrations:
            if hasattr(calibration, "start_date_reference") and calibration.start_date_reference:
                start_date_reference = str(calibration.start_date_reference)
                break

        logger.info("  - Generating quantile plots")
        # Extract surveillance sources from output config if available
        surveillance_sources = None
        if output.options and output.options.surveillance:
            surveillance_sources = output.options.surveillance

        generate_single_quantile_plots(calibrations, plots_config, out_dict, surveillance_sources)
        generate_quantile_grid_plot(calibrations, plots_config, out_dict, surveillance_sources)
        logger.info("  - Generating posterior plots")
        generate_single_location_posterior_plots(calibrations, plots_config, out_dict, start_date_reference)
        generate_posterior_grid_plot(calibrations, plots_config, out_dict, start_date_reference)

        # Generate categorical plots (requires FluSight rate-trend data)
        if plots_config.categorical:
            logger.info("  - Generating categorical plots")
            generate_categorical_plots(plots_config, out_dict, hub_format_output)

    logger.info("Output generation complete. Generated %d output types", len(out_dict))
    return out_dict


def dispatch_output_generator(**configs) -> dict[str, bytes]:
    """
    Dispatch output generator functions.

    Parameters
    ----------
    **configs
        Configuration objects (simulations/calibrations, output_config)

    Returns
    -------
    dict[str, bytes]
        Output data dict of filenames → gzip-compressed CSV bytes
    """
    # Get telemetry from context
    telemetry = ExecutionTelemetry.get_current()

    # Set as current context (for nested calls)
    ExecutionTelemetry.set_current(telemetry)

    try:
        # Enter output stage
        if telemetry:
            telemetry.enter_output()

        # Generate outputs
        kinds = frozenset(k for k, v in configs.items() if v is not None)
        output_data = OUTPUT_GENERATOR_REGISTRY[kinds](**configs)

        # Track output files in telemetry
        if telemetry:
            for output_objects in output_data.values():
                for output_object in output_objects:
                    file_size = len(output_object.data)
                    telemetry.capture_file(output_object.name, file_size)

        # Exit output stage
        if telemetry:
            telemetry.exit_output()

        return output_data
    finally:
        # Clear context when done
        ExecutionTelemetry.set_current(None)
