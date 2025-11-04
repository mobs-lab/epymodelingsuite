import datetime as dt
import logging
from collections.abc import Callable

import pandas as pd
from epydemix.model import EpiModel

logger = logging.getLogger(__name__)


def validate_age_groups(target_age_groups: list[str]) -> None:
    """
    Validate that a list of age group labels is properly formatted, contiguous, and non-overlapping.

    Rules enforced
    --------------
    - The first age group must start at '0'.
    - The last age group must end with '+' (e.g. '80+').
    - All intermediate groups must be in the format 'start-end' (e.g. '0-9', '10-24').
    - Groups must be contiguous: the end of one group plus one must equal the start of the next group.

    Parameters
    ----------
    target_age_groups : list[str]
        List of age group labels to validate.

    Raises
    ------
    ValueError
        If any of the rules above are violated.
    """
    logger.info(f"Validating age groups: {target_age_groups}")

    if target_age_groups[-1][-1] != "+":
        raise ValueError("The last age group must end with '+' e.g. '80+'")

    if target_age_groups[0][0] != "0":
        raise ValueError("The first age group must start at '0'")

    for i in range(len(target_age_groups) - 1):
        if "-" not in target_age_groups[i]:
            raise ValueError("Age groups must be in the format 'start-end' e.g. '0-9', '10-24', '25-32' etc")
        if "-" not in target_age_groups[i + 1] and i + 1 != len(target_age_groups) - 1:
            raise ValueError("Age groups must be in the format 'start-end' e.g. '0-9', '10-24', '25-32' etc")
        i_end = int(target_age_groups[i].split("-")[1].replace("+", ""))
        i1_start = int(target_age_groups[i + 1].split("-")[0].replace("+", ""))
        if i_end + 1 != i1_start:
            raise ValueError("Age groups must be contiguous and not overlapping e.g. '0-9', '10-24', '25-32' etc")


def get_age_group_mapping(target_age_groups: list[str]) -> dict[str, list[str]]:
    """
    Construct a mapping from model age group labels to the individual ages
    they encompass.

    Parameters
    ----------
    target_age_groups : list[str]
        List of contiguous, non-overlapping age group labels. Each group
        should be formatted as:
        - 'start-end', e.g. '0-9', '10-24', '25-32', etc.
        - The final group must use an open-ended format with '+', e.g. '80+'.

    Returns
    -------
    dict[str, list[str]]
        A dictionary mapping each age group label to a list of the string
        representations of ages it covers. The final open-ended group includes
        all ages from its starting value up to 83, plus the label '84+'.
    """
    validate_age_groups(target_age_groups)
    age_group_map = {}
    for i, a in enumerate(target_age_groups):
        if i != len(target_age_groups) - 1:
            start, end = a.split("-")
            age_group_map[a] = [str(j) for j in range(int(start), int(end) + 1)]
        elif "+" in a:
            start = a.split("+")[0]
            age_group_map[a] = [str(j) for j in range(int(start), 84)] + ["84+"]

    return age_group_map


def get_all_age_map() -> dict[str, list[str]]:
    """Create a master population age group map from all age groups."""
    # fmt: off
    map = get_age_group_mapping(
        ["0-0", "1-1", "2-2", "3-3", "4-4", "5-5", "6-6", "7-7", "8-8", "9-9", "10-10", "11-11", "12-12", "13-13", "14-14", "15-15", "16-16", "17-17", "18-18", "19-19", "20-20", "21-21", "22-22", "23-23", "24-24", "25-25", "26-26", "27-27", "28-28", "29-29", "30-30", "31-31", "32-32", "33-33", "34-34", "35-35", "36-36", "37-37", "38-38", "39-39", "40-40", "41-41", "42-42", "43-43", "44-44", "45-45", "46-46", "47-47", "48-48", "49-49", "50-50", "51-51", "52-52", "53-53", "54-54", "55-55", "56-56", "57-57", "58-58", "59-59", "60-60", "61-61", "62-62", "63-63", "64-64", "65-65", "66-66", "67-67", "68-68", "69-69", "70-70", "71-71", "72-72", "73-73", "74-74", "75-75", "76-76", "77-77", "78-78", "79-79", "80-80", "81-81", "82-82", "83-83", "84+"]
    )
    # fmt: on
    return map


def get_age_groups_from_data(data: pd.DataFrame) -> dict[str, str]:
    """
    Extract and clean age group labels from a dataset, returning a mapping from
    data-provided age group names to standardized model age group names.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing an 'Age' column with age group labels. The label
        "6 Months - 17 Years" is excluded, since it overlaps with finer-grained
        groups in the data.

    Returns
    -------
    dict[str, str]
        A dictionary mapping raw age group labels from the data to the cleaned
        and standardized model age group labels.
    """
    age_groups_data = data.Age.unique().tolist()  # Get unique age groups
    age_groups_data.remove(
        "6 Months - 17 Years"
    )  # Remove 6 Months - 17 Years because because data includes finer resolution age groups which cover this range

    age_groups_data_cleaned = []
    for a in age_groups_data:
        b = a.replace(" Years", "").replace("6 Months ", "0").replace(" ", "")
        age_groups_data_cleaned.append(b)

    # data_to_model_age_groups = dict(zip(age_groups_data, age_groups_data_cleaned))

    age_group_map_data = get_age_group_mapping(age_groups_data_cleaned)

    return age_group_map_data


def resample_dataframe(df: pd.DataFrame, delta_t: float) -> pd.DataFrame:
    """
    Resample a vaccination coverage DataFrame to a finer time resolution
    by repeating values (step interpolation) and scaling by delta_t.

    This function preserves the step-wise nature of vaccination schedules,
    where values should remain constant within each day/period rather than
    being smoothly interpolated between periods.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with at least the following columns:
        - 'dates': datetime-like index column
        - 'location': location identifier
        - numeric coverage columns to be resampled
    delta_t : float
        Fraction of a day representing the new time step size. For example,
        `delta_t=0.5` corresponds to 12-hour intervals, `delta_t=0.25` to 6-hour intervals.

    Returns
    -------
    pd.DataFrame
        Resampled DataFrame with:
        - 'dates' as the new index column at the finer resolution
        - 'location' carried forward
        - numeric columns repeated (not interpolated) and scaled by `delta_t`
    """
    import numpy as np

    frequency = f"{24 * delta_t}h"

    # Create new time index
    new_index = pd.date_range(start=df.dates.iloc[0], end=df.dates.iloc[-1] + pd.Timedelta(days=1), freq=frequency)[
        :-1
    ]  # Remove the last point to avoid going beyond end date

    df = df.set_index("dates")
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Use forward fill (step interpolation) instead of linear interpolation
    combined_index = df.index.union(new_index)
    vaccines_step = df[numeric_cols].reindex(combined_index).ffill()
    vaccines_fine = vaccines_step.reindex(new_index)

    # Scale by delta_t to maintain total doses per day
    vaccines_fine = vaccines_fine * delta_t

    vaccines_fine = vaccines_fine.reset_index().rename(columns={"index": "dates"})
    vaccines_fine.insert(1, "location", df["location"].values[0])

    return vaccines_fine


def make_reweighting_factors(
    population_dict_model: dict[str, int], data_age_groups: list[str], loc_epydemix: str
) -> dict[str, list[float]]:
    """
    Calculate reweighting factors that map coverage trajectories from data age groups
    onto model age groups, using population distributions from Epydemix.

    Parameters
    ----------
    population_dict_model : dict[str, int]
        Dictionary mapping model age group names (e.g., "0-4", "5-17") to their populations.
    data_age_groups : list[str]
        Ordered list of data age group labels, with ranges given as "start-end"
        or "start+" for the final open-ended group.
    loc_epydemix : str
        Key used to select the population distribution from the Epydemix codebook.

    Returns
    -------
    dict[str, list[float]]
        A dictionary mapping each model age group name to a list of reweighting factors,
        one for each data age group. Each list is the same length as `data_age_groups`.
    """
    import numpy as np

    from .utils import get_population_codebook

    population_codebook = get_population_codebook()
    population = population_codebook[loc_epydemix].values
    model_age_groups = list(population_dict_model.keys())
    reweighting_factors_dict = {}  # dict of lists of len(model_age_groups) to store reweighting factors for each model age group,
    for i, a in enumerate(model_age_groups):
        reweighting_factors = np.tile(
            0.0, len(data_age_groups)
        )  # list of len(data_age_groups) to store reweighting factors for each data age group
        if i != len(model_age_groups) - 1:
            start_model, end_model = a.split("-")
            start_model = int(start_model)
            end_model = int(end_model)
            for j, b in enumerate(data_age_groups):
                if j != len(data_age_groups) - 1:
                    start_data, end_data = b.split("-")
                    start_data = int(start_data)
                    end_data = int(end_data)
                    if start_data > end_model:
                        continue
                    if end_data < start_model:
                        continue

                    if start_data > start_model and end_data > end_model:
                        factor = sum(population[start_data : end_model + 1]) / sum(
                            population[start_data : end_data + 1]
                        )
                        reweighting_factors[j] = np.min([factor, 1.0])

                    if start_data > start_model and end_data <= end_model:
                        factor = sum(population[start_data : end_data + 1]) / sum(population[start_data : end_data + 1])
                        reweighting_factors[j] = np.min([factor, 1.0])

                    if start_data <= start_model and end_data <= end_model:
                        factor = sum(population[start_model : end_data + 1]) / sum(
                            population[start_data : end_data + 1]
                        )
                        reweighting_factors[j] = np.min([factor, 1.0])

                    if start_data <= start_model and end_data > end_model:
                        factor = sum(population[start_model : end_model + 1]) / sum(
                            population[start_data : end_data + 1]
                        )
                        reweighting_factors[j] = np.min([factor, 1.0])

                elif "+" in b:
                    start_data = int(b.split("+")[0])
                    end_data = 84

                    if start_data > start_model and end_data > end_model:
                        factor = sum(population[start_data : end_model + 1]) / sum(
                            population[start_data : end_data + 1]
                        )
                        reweighting_factors[j] = np.min([factor, 1.0])

                    if start_data > start_model and end_data <= end_model:
                        factor = sum(population[start_data : end_data + 1]) / sum(population[start_data : end_data + 1])
                        reweighting_factors[j] = np.min([factor, 1.0])

                    if start_data <= start_model and end_data <= end_model:
                        factor = sum(population[start_model : end_data + 1]) / sum(
                            population[start_data : end_data + 1]
                        )
                        reweighting_factors[j] = np.min([factor, 1.0])

                    if start_data <= start_model and end_data > end_model:
                        factor = sum(population[start_model : end_model + 1]) / sum(
                            population[start_data : end_data + 1]
                        )
                        reweighting_factors[j] = np.min([factor, 1.0])
        elif "+" in a:
            start_model = a.split("+")[0]
            start_model = int(start_model)
            end_model = 84

            if start_data > start_model and end_data > end_model:
                factor = sum(population[start_data : end_model + 1]) / sum(population[start_data : end_data + 1])
                reweighting_factors[j] = np.min([factor, 1.0])

            if start_data > start_model and end_data <= end_model:
                factor = sum(population[start_data : end_data + 1]) / sum(population[start_data : end_data + 1])
                reweighting_factors[j] = np.min([factor, 1.0])

            if start_data <= start_model and end_data <= end_model:
                factor = sum(population[start_model : end_data + 1]) / sum(population[start_data : end_data + 1])
                reweighting_factors[j] = np.min([factor, 1.0])

            if start_data <= start_model and end_data > end_model:
                factor = sum(population[start_model : end_model + 1]) / sum(population[start_data : end_data + 1])
                reweighting_factors[j] = np.min([factor, 1.0])

        reweighting_factors_dict[a] = reweighting_factors

    return reweighting_factors_dict


def scenario_to_epydemix(
    input_filepath: str,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    target_age_groups: list[str] = ["0-4", "5-17", "18-49", "50-64", "65+"],
    delta_t: float = 1.0,
    output_filepath: str | None = None,
    states: list[str] | None = None,
) -> pd.DataFrame:
    """
    Convert age-specific vaccination coverage data into daily vaccination schedules compatible with Epydemix.

    This function reads vaccination data from a CSV, validates and filters it by date and optionally by state,
    maps reported age groups to the model's age groups, calculates cumulative doses and daily vaccination rates,
    and produces a DataFrame of daily doses per age group and location. Optionally, the resulting data can be
    written to a CSV file.

    Parameters
    ----------
    input_filepath : str
        Path to the CSV file containing vaccination data. The CSV must contain the following columns:
        "Week_Ending_Sat", "Geography", "Age", "Population", "Coverage".
    start_date : str or pd.Timestamp
        Start date for the vaccination schedule (inclusive).
    end_date : str or pd.Timestamp
        End date for the vaccination schedule (inclusive).
    target_age_groups : list of str, default ["0-4", "5-17", "18-49", "50-64", "65+"]
        Age groups to map the data to for the output schedule.
    delta_t : float, default 1.0
        Time step for resampling the daily vaccination data. Default 1.0 means daily.
    output_filepath : str, optional
        If provided, the processed daily vaccination DataFrame will be saved as a CSV at this path.
    states : list[str], optional
        If provided, only data for these specific states/locations will be processed.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing daily vaccination doses for each age group and location. Columns include:
        "dates", "location", and one column per target age group.

    Raises
    ------
    ValueError
        If required columns are missing from the input CSV or if age groups are invalid.

    Notes
    -----
    - The function aggregates weekly coverage into daily doses, distributes them evenly across days,
      and maps input age groups to the target model age groups using population reweighting.
    - Location names are converted to ISO codes for compatibility with Epydemix.
    """
    import numpy as np

    from .utils import convert_location_name_format, get_population_codebook

    validate_age_groups(target_age_groups)
    population_codebook = get_population_codebook()
    if states:
        states = [convert_location_name_format(s, "name") for s in states]
    # ========== LOAD AND FILTER DATA ==========
    vaccines = pd.read_csv(input_filepath)
    # Date and time handling
    vaccines["Week_Ending_Sat"] = pd.to_datetime(vaccines["Week_Ending_Sat"])
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    # Validate required columns
    required_columns = ["Week_Ending_Sat", "Geography", "Age", "Population", "Coverage"]
    missing_columns = [col for col in required_columns if col not in vaccines.columns]
    if missing_columns:
        raise ValueError(
            f"Input data must contain the following columns: {required_columns}. Missing columns: {missing_columns}"
        )

    # Get age groups from data
    data_age_groups_dict = get_age_groups_from_data(vaccines)
    data_age_groups = list(data_age_groups_dict.keys())

    # Filter for date range
    vaccines = vaccines.query("Week_Ending_Sat >= @start_date")

    first_sat = vaccines["Week_Ending_Sat"].min()
    if end_date + pd.Timedelta(days=6) < first_sat:
        raise ValueError(
            f"No data for the requested date range from {start_date.date()} to {end_date.date()}. Earliest data is for week ending {first_sat.date()}."
        )

    # ========== PROCESS ALL GEOGRAPHIES ==========
    # all_locations_data = []
    all_locations_df = pd.DataFrame()

    for location in vaccines["Geography"].unique():
        if states and location not in states:
            continue
        location_data = vaccines.query("Geography == @location").copy()
        loc_epydemix = convert_location_name_format(location, "epydemix_population")

        # Keep other age groups and combine with aggregated youth data
        vaccine_schedule = location_data.query("Age not in ['6 Months - 17 Years']")
        vaccine_schedule = vaccine_schedule.sort_values(["Week_Ending_Sat", "Age"]).reset_index(drop=True)

        # ========== MAP TO MODEL POPULATION ==========
        age_to_index = {
            "6 Months - 4 Years": 0,
            "5-12 Years": 1,
            "13-17 Years": 2,
            "18-49 Years": 3,
            "50-64 Years": 4,
            "65+ Years": 5,
        }
        age_group_map_data = get_age_group_mapping(data_age_groups)
        # population_data = load_epydemix_population(loc_epydemix, age_group_mapping=age_group_map_data)
        population_data = {}
        for key, val in age_group_map_data.items():
            vals = [int(v.replace("+", "")) for v in val]
            population_data[key] = sum(population_codebook[loc_epydemix][vals].values)

        # Add model population and calculate cumulative doses
        vaccine_schedule["population_data"] = vaccine_schedule["Age"].map(
            lambda age: list(population_data.values())[age_to_index[age]] if age in age_to_index else 0
        )

        vaccine_schedule["cumulative_doses"] = (
            vaccine_schedule["Coverage"] / 100 * vaccine_schedule["population_data"]
        ).round()

        # ========== CONVERT TO DAILY VACCINATION RATES ==========
        daily_vaccines_list = []
        for age_group in vaccine_schedule["Age"].unique():
            age_data = vaccine_schedule.query("Age == @age_group").copy()

            # Calculate new doses per week (difference from previous week)
            age_data["new_doses"] = np.diff(age_data["cumulative_doses"], prepend=0)

            # Distribute weekly doses across days
            current_date = start_date

            for _, week_data in age_data.iterrows():
                week_end = week_data["Week_Ending_Sat"]
                days_in_period = (week_end - current_date).days + 1  # Include week_end

                if week_end >= end_date:
                    days_in_period = (end_date - current_date).days + 1
                    daily_rate = week_data["new_doses"] / days_in_period

                    # Add daily entries for this period (including week_end)
                    for date in pd.date_range(current_date, end_date):
                        daily_vaccines_list.append(
                            {"dates": date, "location": location, "age_group": age_group, "doses": round(daily_rate)}
                        )
                    current_date = end_date + pd.Timedelta(days=1)  # End processing
                    break

                if current_date + pd.Timedelta(days=6) <= first_sat:
                    daily_rate = 0
                    for date in pd.date_range(current_date, week_end - pd.Timedelta(days=7)):
                        daily_vaccines_list.append(
                            {"dates": date, "location": location, "age_group": age_group, "doses": round(daily_rate)}
                        )
                    days_in_period = 7

                    daily_rate = week_data["new_doses"] / days_in_period

                    # Add daily entries for this period (including week_end)
                    for date in pd.date_range(week_end - pd.Timedelta(days=7), week_end):
                        daily_vaccines_list.append(
                            {"dates": date, "location": location, "age_group": age_group, "doses": round(daily_rate)}
                        )
                else:
                    # Calculate daily rate for this period
                    daily_rate = week_data["new_doses"] / days_in_period

                    # Add daily entries for this period (including week_end)
                    for date in pd.date_range(current_date, week_end):
                        daily_vaccines_list.append(
                            {"dates": date, "location": location, "age_group": age_group, "doses": round(daily_rate)}
                        )
                current_date = week_end + pd.Timedelta(days=1)  # Start next period

            # Fill remaining days with zeros
            for date in pd.date_range(current_date, end_date):
                daily_vaccines_list.append({"dates": date, "location": location, "age_group": age_group, "doses": 0})

        this_location_data = pd.DataFrame(daily_vaccines_list)

        this_location_wide = this_location_data.pivot_table(
            index=["dates", "location"], columns="age_group", values="doses", aggfunc="first", fill_value=0
        ).reset_index()

        this_location_wide.columns.name = None
        this_location_wide.rename(
            columns={
                "6 Months - 4 Years": "0-4",
                "5-12 Years": "5-12",
                "13-17 Years": "13-17",
                "18-49 Years": "18-49",
                "50-64 Years": "50-64",
                "65+ Years": "65+",
            },
            inplace=True,
        )

        # Format locations as ISO codes
        # this_location_wide["location"] = [
        #     convert_location_name_format(loc, "ISO") for loc in this_location_wide.location
        # ]
        this_location_wide["location"] = convert_location_name_format(location, "ISO")

        daily_vaccines_wide_subset = this_location_wide[data_age_groups]

        age_group_map_model = get_age_group_mapping(target_age_groups)
        # population_model = load_epydemix_population(loc_epydemix, age_group_mapping=age_group_map_model)
        population_dict_model = {}
        for key, val in age_group_map_model.items():
            vals = [int(v.replace("+", "")) for v in val]
            population_dict_model[key] = sum(population_codebook[loc_epydemix][vals].values)

        # population_dict_model= dict(zip(population_model.Nk_names, population_model.Nk))

        reweighting_factors_dict = make_reweighting_factors(population_dict_model, data_age_groups, loc_epydemix)
        new_coverages = {}
        for target_group, weights in reweighting_factors_dict.items():
            # linear combination
            new_coverages[target_group] = daily_vaccines_wide_subset.mul(weights, axis=1).sum(axis=1)

        daily_vaccines_transformed = pd.DataFrame(new_coverages, index=daily_vaccines_wide_subset.index)
        daily_vaccines_transformed.insert(0, "dates", this_location_wide["dates"])
        daily_vaccines_transformed.insert(1, "location", this_location_wide["location"])

        if delta_t != 1.0:
            vaccines_fine = resample_dataframe(daily_vaccines_transformed, delta_t)
        else:
            vaccines_fine = daily_vaccines_transformed.copy()

        all_locations_df = pd.concat([all_locations_df, vaccines_fine], ignore_index=True)
        # all_locations_data.extend(daily_vaccines_list)

    # ========== WRITE OUTPUT CSV ==========
    if output_filepath:
        all_locations_df.to_csv(output_filepath, index=False)

    return all_locations_df


def smh_data_to_epydemix(
    input_filepath: str,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    target_age_groups: list[str] = ["0-4", "5-17", "18-49", "50-64", "65+"],
    delta_t: float = 1.0,
    output_filepath: str | None = None,
    states: list[str] | None = None,
) -> pd.DataFrame:
    """
    Process age-specific influenza vaccine coverage data from the scenario modeling hub into
    daily vaccination schedules by age group for ALL scenarios and ALL locations.

    This function handles multiple scenarios by extracting them from the input data, creating
    temporary single-scenario files, and calling scenario_to_epydemix for each scenario.
    The results are then combined into a single DataFrame with scenario information.

    Args:
        input_filepath (str): Path to CSV containing SMH vaccination data with scenario columns.
        start_date (str or Timestamp): Start date of the simulation period.
        end_date (str or Timestamp): End date of the simulation period.
        target_age_groups (list[str]): Age groups to map the data to for the output schedule.
        delta_t (float): Time step for resampling the daily vaccination data. Default 1.0 means daily.
        output_filepath (str, optional): If provided, the output DataFrame will be saved as a CSV.
        states (list[str], optional): If provided, only data for these specific states/locations will be processed.

    Returns
    -------
        pd.DataFrame: DataFrame with columns ['dates', 'scenario', 'location', <age groups>] giving the
                      daily vaccination counts per age group for each scenario across all geographies.
    """
    import os
    import tempfile

    import pandas as pd

    # ========== LOAD AND EXTRACT SCENARIOS ==========
    vaccines = pd.read_csv(input_filepath)

    # Extract scenario columns directly (keep full column names)
    scenario_columns = [name for name in vaccines.columns if name.find("sc_") > -1]
    scenario_columns = [name.split("sc_")[-1] for name in scenario_columns if name.find("sc_") > -1]

    vaccines = vaccines.rename(
        columns={name: name.split("sc_")[-1] for name in list(vaccines.columns) if name.find("sc_") > -1}
    )

    if not scenario_columns:
        raise ValueError("No scenario columns found in the input data. Expected columns with 'sc_' prefix.")

    # ========== PROCESS EACH SCENARIO ==========
    all_scenarios_df = pd.DataFrame()

    for scenario_column in scenario_columns:
        # Extract scenario name for labeling
        scenario_name = scenario_column

        # Create a temporary dataset for this scenario with single Coverage column
        scenario_data = vaccines.copy()

        # Rename the scenario column to "Coverage" (expected by scenario_to_epydemix)
        scenario_data["Coverage"] = scenario_data[scenario_column]

        # Remove all other scenario columns to avoid confusion
        other_scenario_columns = [col for col in scenario_columns if col != scenario_column]
        scenario_data = scenario_data.drop(columns=other_scenario_columns)

        # Create temporary file for this scenario
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as temp_file:
            scenario_data.to_csv(temp_file.name, index=False)
            temp_filepath = temp_file.name

        try:
            # Call scenario_to_epydemix for this scenario
            scenario_result = scenario_to_epydemix(
                input_filepath=temp_filepath,
                start_date=start_date,
                end_date=end_date,
                target_age_groups=target_age_groups,
                delta_t=delta_t,
                output_filepath=None,  # Don't write individual scenario files
                states=states,
            )

            # Add scenario identifier
            scenario_result.insert(1, "scenario", scenario_name)

            # Combine with other scenarios
            all_scenarios_df = pd.concat([all_scenarios_df, scenario_result], ignore_index=True)

        finally:
            # Clean up temporary file
            if os.path.exists(temp_filepath):
                os.unlink(temp_filepath)

    # ========== WRITE OUTPUT CSV ==========
    if output_filepath:
        all_scenarios_df.to_csv(output_filepath, index=False)

    return all_scenarios_df


def reaggregate_vaccines(schedule: pd.DataFrame, actual_start_date: dt.date | pd.Timestamp) -> pd.DataFrame:
    """
    Reaggregate a vaccination schedule so that it begins at the specified
    actual start date.

    Parameters
    ----------
    schedule : pd.DataFrame
        Vaccination schedule with columns including:
        - 'dates' : pd.Timestamp
        - 'location' : str
        - age group columns (e.g. '0-4', '5-17', '65+')
        Optionally may include:
        - 'scenario' : str
    actual_start_date : pd.Timestamp or datetime.date
        The true start date of the vaccination schedule. Must fall within
        the range of `schedule['dates']`.

    Returns
    -------
    pd.DataFrame
        A reaggregated vaccination schedule where:
        - Doses from the actual start date up to the next Saturday are
        redistributed evenly across that period.
        - All subsequent rows from the original schedule are preserved.
        - Returned DataFrame is sorted by 'dates'.

    Raises
    ------
    ValueError
        If `actual_start_date` is earlier than the first date or later
        than the last date in `schedule['dates']`.
    """
    # Normalize type
    actual_start_date = pd.Timestamp(actual_start_date)

    # Validate actual start dates to fit
    date_min = schedule["dates"].min()
    date_max = schedule["dates"].max()
    if not (date_min <= actual_start_date <= date_max):
        err_msg = f"Start date must be between {date_min} and {date_max}"
        raise ValueError(err_msg)
    if date_min == actual_start_date:
        return schedule

    # Calculate next saturday
    days_until_saturday = (5 - actual_start_date.weekday()) % 7
    if days_until_saturday == 0:  # If actual start date is Saturday
        days_until_saturday = 7
    next_saturday = actual_start_date + pd.Timedelta(days=days_until_saturday)

    # Get age groups
    # e.g. ['0-4', '5-17', '18-49', '50-64', '65+']
    age_groups = [c for c in schedule.columns if "-" in c or "+" in c]

    # Aggregate doses for the period before the next Saturday
    before_saturday = schedule.query("dates < @next_saturday")
    aggregated_doses = before_saturday[age_groups].sum(axis=0)

    # Create a new date range for the redistribution period
    new_dates = pd.date_range(start=actual_start_date, end=next_saturday, freq="D")
    num_days = len(new_dates)

    redistributed_data = []

    # Redistribute doses across the new date range
    for age_group in age_groups:
        total_doses = aggregated_doses[age_group]

        # Calculate daily base doses and reminder to keep the total dose consistent.
        daily_base = (total_doses // num_days).astype(int)
        remainder = (total_doses % num_days).astype(int)

        # List of daily doses for the new date range
        doses = [daily_base] * num_days
        # Remainders would be distributed at the beginning (a dose/day)
        for i in range(remainder):
            doses[i] += 1

        redistributed_data.append(doses)

    # Prepare new rows to be added to the schedule
    new_rows = pd.DataFrame(
        {
            "dates": new_dates,
            "location": schedule["location"].iloc[0],
        }
    )

    # Add scenario column if exists
    if "scenario" in schedule.columns:
        new_rows["scenario"] = schedule["scenario"].iloc[0]

    # Add age group columns with redistributed data
    new_rows = new_rows.assign(**dict(zip(age_groups, redistributed_data, strict=False)))

    # Combine with the original schedule
    after_saturday = schedule.query("dates > @next_saturday")
    reaggregated_schedule = pd.concat([new_rows, after_saturday], ignore_index=True)

    return reaggregated_schedule


def make_vaccination_rate_function(origin_compartment: str, eligible_compartments: list[str]) -> callable:
    """
    Return a vaccination rate function for a given origin compartment and a set of
    eligible compartments used in the denominator when allocating doses.

    Returns a RATE function (vaccinations per person per unit time), not a probability.
    In epydemix ≥ v1.0.2, Callable that computs rates is passed to EpiModel.register_transition_kind(), instead of Callable that computes probabilities. Internally, it will convert the rate to a probability using p = 1 - exp(-rate * dt).

    Args:
            origin_compartment (str): The compartment receiving the vaccination (e.g., 'S').
            eligible_compartments (list of str): Compartments included in dose allocation denominator
                                                                                     (e.g., ['S', 'R']).

    Returns
    -------
            function: A function (params, data) -> np.ndarray suitable for use with
                              model.register_transition_kind.
    """
    import numpy as np
    from epydemix.model.epimodel import validate_transition_function

    def compute_vaccination_rate(params: list, data: dict) -> np.ndarray:
        """
        Compute the vaccination rate (per unit time) required to allocate available doses
        to individuals in the origin compartment, based on the distribution of the population
        across eligible compartments.

        Note: Returns a RATE (vaccinations per person per unit time), not a probability.
        epydemix will convert this to a probability using p = 1 - exp(-rate * dt).

        Args:
                params (list): A list containing one element: a 1D array of daily total doses
                                        (aligned with simulation time steps).
                data (dict): A dictionary with the following keys:
                        - 't': Current time index (int)
                        - 'dt': Duration of the current time step (used by EpiModel, not here)
                        - 'pop': Array of compartment population counts
                        - 'comp_indices': Dict mapping compartment names ('S', 'R', etc.) to their indices in 'pop'

        Returns
        -------
                np.ndarray: Array of vaccination rates (per unit time) for the origin compartment.  Can exceed 1.0 (e.g., if daily doses > origin population).
        """
        total_doses = params[0][data["t"]]
        origin_pop = data["pop"][data["comp_indices"][origin_compartment]]

        denom = sum(data["pop"][data["comp_indices"][comp]] for comp in eligible_compartments)

        with np.errstate(divide="ignore", invalid="ignore"):
            fraction_origin = np.where(denom > 0, origin_pop / denom, 0)

        # Calculate effective doses for origin compartment
        effective_doses = total_doses * fraction_origin

        # Calculate vaccination rate: doses per unit time / population
        with np.errstate(divide="ignore", invalid="ignore"):
            vaccination_rate = np.where(origin_pop > 0, effective_doses / origin_pop, 0)

        # Return rate (can be > 1, no upper cap needed)
        # Only ensure non-negative
        return np.maximum(vaccination_rate, 0)

    validate_transition_function(compute_vaccination_rate)
    return compute_vaccination_rate


def add_vaccination_schedule(
    model: EpiModel,
    vaccine_rate_function: Callable,
    source_comp: str,
    target_comp: str,
    vaccination_schedule: pd.DataFrame,
) -> EpiModel:
    """
    Register and add a vaccination transition to the model using a time-varying schedule by age group.

    Parameters
    ----------
        model (EpiModel): The model object to which the vaccination schedule will be added. Must have a population
                          age groups same as the columns in `vaccination_schedule`.
        vaccine_rate_function (Callable): A function defining time-dependent vaccination rates.
        vaccination_schedule (pd.DataFrame): Vaccination schedule with age groups as columns and time as rows.
                                             Must include all age groups used in the model.
        source_comp (str): The name of the source compartment (e.g., "S").
        target_comp (str): The name of the target compartment (e.g., "SV").

    Returns
    -------
        EpiModel: The model with the vaccination transition added.

    Raises
    ------
        ValueError: If any age groups required by the model are missing from the DataFrame.
    """
    from .utils import convert_location_name_format

    # Location handling
    iso_location = convert_location_name_format(model.population.name, "ISO")

    if "location" not in vaccination_schedule.columns:
        raise ValueError(
            f"'location' column not found in vaccination_schedule.\n"
            f"Available columns: {list(vaccination_schedule.columns)}"
        )

    if iso_location not in vaccination_schedule["location"].unique():
        raise ValueError(f"Location {iso_location} not found in vaccination schedule data.")

    vaccination_schedule = vaccination_schedule.query("location == @iso_location").copy()

    # From epydemix v1.0.2, register_transition_kind accepts Callable for rate not probability
    model.register_transition_kind("vaccination", vaccine_rate_function)

    age_groups_model = model.population.Nk_names
    age_groups_data = vaccination_schedule.columns.tolist()

    missing = [age for age in age_groups_model if age not in age_groups_data]
    if missing:
        raise ValueError(
            "Age groups in model and age groups in data must be the same.\n"
            f"Model age groups: {age_groups_model}\n"
            f"Data frame columns: {age_groups_data}"
        )

    vaccine_schedule = (vaccination_schedule[age_groups_model].values,)

    # Usage:
    model = remove_vaccination_transitions(model, source_comp, target_comp)
    model.add_transition(source_comp, target_comp, params=vaccine_schedule, kind="vaccination")

    return model


def remove_vaccination_transitions(model: EpiModel, source_comp: str, target_comp: str) -> EpiModel:
    """
    Manually remove vaccination transitions from model.
    This prevents `add_vaccination_schedule` from creating duplicate transitions if it is called multiple times.
    """
    # Remove from transitions_list
    model.transitions_list = [
        t
        for t in model.transitions_list
        if not (t.source == source_comp and t.target == target_comp and t.kind == "vaccination")
    ]

    # Remove from transitions dict
    if source_comp in model.transitions:
        model.transitions[source_comp] = [
            t for t in model.transitions[source_comp] if not (t.target == target_comp and t.kind == "vaccination")
        ]

    return model
