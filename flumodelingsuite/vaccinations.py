
from typing import Optional, Callable
import pandas as pd
from epydemix.model import EpiModel 

def validate_age_groups(
    target_age_groups: list[str]
) -> None:
    """
    Validate that a list of age group labels is properly formatted, contiguous, 
    and non-overlapping.

    Rules enforced
    --------------
    - The first age group must start at '0'.
    - The last age group must end with '+' (e.g. '80+').
    - All intermediate groups must be in the format 'start-end' (e.g. '0-9', '10-24').
    - Groups must be contiguous: the end of one group plus one must equal the 
      start of the next group.

    Parameters
    ----------
    target_age_groups : list[str]
        List of age group labels to validate.

    Raises
    ------
    ValueError
        If any of the rules above are violated.
    """
    if target_age_groups[-1][-1] != "+":
        raise ValueError("The last age group must end with '+' e.g. '80+'")

    if target_age_groups[0][0] != '0':
        raise ValueError("The first age group must start at '0'")

    for i in range(len(target_age_groups) - 1):
        if "-" not in target_age_groups[i]:
            raise ValueError("Age groups must be in the format 'start-end' e.g. '0-9', '10-24', '25-32' etc")
        if "-" not in target_age_groups[i+1] and i+1 != len(target_age_groups) - 1:
            raise ValueError("Age groups must be in the format 'start-end' e.g. '0-9', '10-24', '25-32' etc")
        i_end = int(target_age_groups[i].split("-")[1].replace("+", ""))
        i1_start = int(target_age_groups[i + 1].split("-")[0].replace("+", ""))
        if i_end + 1 != i1_start:
            raise ValueError("Age groups must be contiguous and not overlapping e.g. '0-9', '10-24', '25-32' etc")
            
def get_age_group_mapping(
    target_age_groups: list[str]
) -> dict[str, list[str]]:
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
        if i != len(target_age_groups)-1:
            start, end = a.split("-")
            age_group_map[a] = [str(j) for j in range(int(start), int(end)+1)]
        elif '+' in a:
            start = a.split("+")[0]
            age_group_map[a] = [str(j) for j in range(int(start), 84)] + ["84+"]

    return age_group_map

def get_all_age_map():
    """ Create a master population age group map from all age groups. """
    map = get_age_group_mapping(["0-0", "1-1", "2-2", "3-3", "4-4", "5-5", "6-6", "7-7", "8-8", "9-9", "10-10", "11-11", "12-12", "13-13", "14-14", "15-15", "16-16", 
                                 "17-17", "18-18", "19-19", "20-20", "21-21", "22-22", "23-23", "24-24", "25-25", "26-26", "27-27", "28-28", "29-29", "30-30", "31-31", 
                                 "32-32", "33-33", "34-34", "35-35", "36-36", "37-37", "38-38", "39-39", "40-40", "41-41", "42-42", "43-43", "44-44", "45-45", "46-46", 
                                 "47-47", "48-48", "49-49", "50-50", "51-51", "52-52", "53-53", "54-54", "55-55", "56-56", "57-57", "58-58", "59-59", "60-60", "61-61", 
                                 "62-62", "63-63", "64-64", "65-65", "66-66", "67-67", "68-68", "69-69", "70-70", "71-71", "72-72", "73-73", "74-74", "75-75", "76-76", 
                                 "77-77", "78-78", "79-79", "80-80", "81-81", "82-82", "83-83", "84+"])
    return map

def get_age_groups_from_data(
    data: pd.DataFrame
) -> dict[str, str]:
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
    age_groups_data.remove("6 Months - 17 Years") # Remove 6 Months - 17 Years because because data includes finer resolution age groups which cover this range

    age_groups_data_cleaned = []
    for a in age_groups_data:
        b = a.replace(" Years", "").replace("6 Months ", "0").replace(" ","")
        age_groups_data_cleaned.append(b)

    # data_to_model_age_groups = dict(zip(age_groups_data, age_groups_data_cleaned))

    age_group_map_data = get_age_group_mapping(age_groups_data_cleaned)

    return age_group_map_data

def resample_dataframe(
    df: pd.DataFrame,
    dtt: float
) -> pd.DataFrame:
    """
    Resample a vaccination coverage DataFrame to a finer time resolution 
    by linearly interpolating numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with at least the following columns:
        - 'dates': datetime-like index column
        - 'location': location identifier
        - numeric coverage columns to be interpolated
    dtt : float
        Fraction of a day representing the new time step size. For example,
        `dtt=0.5` corresponds to 12-hour intervals, `dtt=0.25` to 6-hour intervals.

    Returns
    -------
    pd.DataFrame
        Resampled DataFrame with:
        - 'dates' as the new index column at the finer resolution
        - 'location' carried forward
        - numeric columns interpolated and scaled by `dtt`
    """
    import numpy as np
    frequency = f"{24*dtt}h"
    new_index = pd.date_range(start=df.dates.iloc[0], 
                            end=df.dates.iloc[-1] + pd.Timedelta(days=1),
                            freq=frequency)
    new_index = new_index[0:-1]

    df = df.set_index('dates')

    # Separate numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    combined_index = df.index.union(new_index)

    # Interpolate only numeric columns
    vaccines_interpolated = df[numeric_cols].reindex(combined_index).interpolate(method='linear')
    vaccines_fine = vaccines_interpolated.reindex(new_index)
    vaccines_fine = vaccines_fine * dtt

    vaccines_fine = vaccines_fine.reset_index().rename(columns={'index': 'dates'})
    vaccines_fine.insert(1, "location", df["location"].values[0])

    return vaccines_fine

def make_reweighting_factors(
    population_dict_model: dict[str, int],
    data_age_groups: list[str],
    loc_epydemix: str
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
    reweighting_factors_dict = {} # dict of lists of len(model_age_groups) to store reweighting factors for each model age group,
    for i, a in enumerate(model_age_groups):
        reweighting_factors = np.tile(0.0, len(data_age_groups)) # list of len(data_age_groups) to store reweighting factors for each data age group                        
        if i != len(model_age_groups)-1:
            start_model, end_model = a.split("-")
            start_model = int(start_model)
            end_model = int(end_model)
            for j, b in enumerate(data_age_groups):
                if j != len(data_age_groups)-1:
                    start_data, end_data = b.split("-")
                    start_data = int(start_data)
                    end_data = int(end_data)
                    if start_data > end_model:
                        continue
                    if end_data < start_model:
                        continue

                    if start_data > start_model and end_data > end_model:
                        factor = sum(population[start_data:end_model+1])/ sum(population[start_data:end_data+1])
                        reweighting_factors[j] = np.min([factor, 1.0])
                    
                    if start_data > start_model and end_data <= end_model:
                        factor = sum(population[start_data:end_data+1])/ sum(population[start_data:end_data+1])
                        reweighting_factors[j] = np.min([factor, 1.0])

                    if start_data <= start_model and end_data <= end_model: 
                        factor = sum(population[start_model:end_data+1])/ sum(population[start_data:end_data+1])
                        reweighting_factors[j] = np.min([factor, 1.0])

                    if start_data <= start_model and end_data > end_model: 
                        factor = sum(population[start_model:end_model+1])/ sum(population[start_data:end_data+1])
                        reweighting_factors[j] = np.min([factor, 1.0])

                elif '+' in b:
                    start_data = int(b.split("+")[0])
                    end_data = 84
                    
                    if start_data > start_model and end_data > end_model:
                        factor = sum(population[start_data:end_model+1])/ sum(population[start_data:end_data+1])
                        reweighting_factors[j] = np.min([factor, 1.0])
                    
                    if start_data > start_model and end_data <= end_model:
                        factor = sum(population[start_data:end_data+1])/ sum(population[start_data:end_data+1])
                        reweighting_factors[j] = np.min([factor, 1.0])

                    if start_data <= start_model and end_data <= end_model: 
                        factor = sum(population[start_model:end_data+1])/ sum(population[start_data:end_data+1])
                        reweighting_factors[j] = np.min([factor, 1.0])

                    if start_data <= start_model and end_data > end_model: 
                        factor = sum(population[start_model:end_model+1])/ sum(population[start_data:end_data+1])
                        reweighting_factors[j] = np.min([factor, 1.0])
        elif '+' in a:
            start_model = a.split("+")[0]
            start_model = int(start_model)
            end_model = 84
            
            if start_data > start_model and end_data > end_model:
                factor = sum(population[start_data:end_model+1])/ sum(population[start_data:end_data+1])
                reweighting_factors[j] = np.min([factor, 1.0])
            
            if start_data > start_model and end_data <= end_model:
                factor = sum(population[start_data:end_data+1])/ sum(population[start_data:end_data+1])
                reweighting_factors[j] = np.min([factor, 1.0])

            if start_data <= start_model and end_data <= end_model: 
                factor = sum(population[start_model:end_data+1])/ sum(population[start_data:end_data+1])
                reweighting_factors[j] = np.min([factor, 1.0])

            if start_data <= start_model and end_data > end_model: 
                factor = sum(population[start_model:end_model+1])/ sum(population[start_data:end_data+1])
                reweighting_factors[j] = np.min([factor, 1.0])
        
        reweighting_factors_dict[a] = reweighting_factors

    return reweighting_factors_dict

def scenario_to_epydemix(
    input_filepath: str,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    target_age_groups: list[str] = ["0-4", "5-17", "18-49", "50-64", "65+"],
    dtt: float = 1.0,
    output_filepath: Optional[str] = None,
    state: Optional[str] = None
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
    dtt : float, default 1.0
        Time step for resampling the daily vaccination data. Default 1.0 means daily.
    output_filepath : str, optional
        If provided, the processed daily vaccination DataFrame will be saved as a CSV at this path.
    state : str, optional
        If provided, only data for this specific state/location will be processed.

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
        raise ValueError(f"Input data must contain the following columns: {required_columns}. Missing columns: {missing_columns}")
    
    # Get age groups from data
    data_age_groups_dict = get_age_groups_from_data(vaccines)
    data_age_groups = list(data_age_groups_dict.keys())

    # Filter for date range
    vaccines = vaccines.query("Week_Ending_Sat >= @start_date")
    
    # ========== PROCESS ALL GEOGRAPHIES ==========
    # all_locations_data = []
    all_locations_df = pd.DataFrame()
    
    for location in vaccines['Geography'].unique():
        if state and location != state:
            continue
        location_data = vaccines.query("Geography == @location").copy()
        loc_epydemix = convert_location_name_format(location, 'epydemix_population')

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
            "65+ Years": 5
        }
        age_group_map_data = get_age_group_mapping(data_age_groups)
        # population_data = load_epydemix_population(loc_epydemix, age_group_mapping=age_group_map_data)
        population_data = {}
        for key, val in age_group_map_data.items():
            vals = [int(v.replace('+','')) for v in val]
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
            age_data["new_doses"] = np.diff(
                age_data["cumulative_doses"], prepend=0
            )
            
            # Distribute weekly doses across days
            current_date = start_date
            
            for _, week_data in age_data.iterrows():
                week_end = week_data["Week_Ending_Sat"]
                days_in_period = (week_end - current_date).days + 1  # Include week_end

                if (week_end >= end_date):
                    days_in_period = (end_date - current_date).days + 1
                    daily_rate = week_data["new_doses"] / days_in_period
                    
                    # Add daily entries for this period (including week_end)
                    for date in pd.date_range(current_date, end_date):
                        daily_vaccines_list.append({
                            "dates": date,
                            "location": location,
                            "age_group": age_group,
                            "doses": round(daily_rate)
                        })
                    current_date = end_date + pd.Timedelta(days=1)  # End processing
                    break

                if days_in_period > 0:
                    # Calculate daily rate for this period
                    daily_rate = week_data["new_doses"] / days_in_period
                    
                    # Add daily entries for this period (including week_end)
                    for date in pd.date_range(current_date, week_end):
                        daily_vaccines_list.append({
                            "dates": date,
                            "location": location,
                            "age_group": age_group,
                            "doses": round(daily_rate)
                        })
                
                current_date = week_end + pd.Timedelta(days=1)  # Start next period
            
            # Fill remaining days with zeros
            for date in pd.date_range(current_date, end_date):
                daily_vaccines_list.append({
                    "dates": date,
                    "location": location,
                    "age_group": age_group,
                    "doses": 0
                })

        this_location_data = pd.DataFrame(daily_vaccines_list)

        this_location_wide = this_location_data.pivot_table(
            index=['dates', 'location'],
            columns='age_group', 
            values='doses',
            aggfunc='first',
            fill_value=0
        ).reset_index()
            
        this_location_wide.columns.name = None
        this_location_wide.rename(columns={
            "6 Months - 4 Years": "0-4",
            "5-12 Years": "5-12",
            "13-17 Years": "13-17", 
            "18-49 Years": "18-49",
            "50-64 Years": "50-64",
            "65+ Years": "65+"
        }, inplace=True)

        # Format locations as ISO codes
        this_location_wide['location'] = [convert_location_name_format(loc, 'ISO') for loc in this_location_wide.location]
        
        daily_vaccines_wide_subset = this_location_wide[data_age_groups]

        age_group_map_model = get_age_group_mapping(target_age_groups)
        # population_model = load_epydemix_population(loc_epydemix, age_group_mapping=age_group_map_model)
        population_dict_model = {}
        for key, val in age_group_map_model.items():
            vals = [int(v.replace('+','')) for v in val]
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

        if dtt != 1.0:
            vaccines_fine = resample_dataframe(daily_vaccines_transformed, dtt)
        else:
            vaccines_fine = daily_vaccines_transformed.copy()
        
        all_locations_df = pd.concat([all_locations_df, vaccines_fine], ignore_index=True)
        # all_locations_data.extend(daily_vaccines_list)

    # ========== WRITE OUTPUT CSV ==========
    if output_filepath:
        vaccines_fine.to_csv(output_filepath, index=False)
    
    return all_locations_df

def smh_data_to_epydemix(
    input_filepath: str,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    target_age_groups: list[str] = ["0-4", "5-17", "18-49", "50-64", "65+"],
    output_filepath: Optional[str] = None,
    state: Optional[str] = None
) -> pd.DataFrame:
    """
    Processes age-specific influenza vaccine coverage data from the scenario modeling hub into 
    daily vaccination schedules by age group for ALL scenarios and ALL locations.

    The function loads weekly cumulative coverage estimates from a CSV file, maps them to 
    the model population, computes new weekly doses, and distributes them evenly across days 
    between weekly reporting dates. Age groups 5-12 and 13-17 are merged into a 5-17 category 
    using population-weighted averages. The output is a daily time series of vaccine doses 
    administered per age group for the specified scenario across all geographies.

    Args:
        input_filepath (str): Path to CSV containing SMH vaccination data.
        start_date (str or Timestamp): Start date of the simulation period (used to build full timeline).
        end_date (str or Timestamp): End date of the simulation period (used to build full timeline).
        model (object): Epydemix EpiModel instance with a .population.Nk list for age group sizes.
        output_filepath (str, optional): If provided, the output DataFrame will be saved as a CSV at the given filepath.

    Returns:
        pd.DataFrame: DataFrame with columns ['dates', 'scenario', 'geography', <age groups>] giving the 
                      daily vaccination counts per age group for the selected scenario across all geographies.
    """
    import numpy as np
    from .utils import convert_location_name_format, get_population_codebook

    validate_age_groups(target_age_groups)
    population_codebook = get_population_codebook()
    # ========== LOAD AND FILTER DATA ==========
    vaccines = pd.read_csv(input_filepath)
    # Date and time handling
    vaccines["Week_Ending_Sat"] = pd.to_datetime(vaccines["Week_Ending_Sat"])
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    
    # Extract scenarios
    scenario_options = [name.split('sc_')[-1] for name in list(vaccines.columns) if name.find('sc_') > -1]
    
    # Filter for date range and rename scenario columns
    vaccines = vaccines.query("Week_Ending_Sat >= @start_date").rename(
        columns={name: 'Coverage_' + name.split('sc_')[-1] 
                 for name in list(vaccines.columns) 
                 if name.find('sc_') > -1}
    )
    
    # ========== PROCESS ALL GEOGRAPHIES ==========
    all_locations_data = []
    
    for location in vaccines['Geography'].unique():
        if state and location != state:
            continue
        location_data = vaccines.query("Geography == @location").copy()
        
        # ========== AGGREGATE AGE GROUPS FOR THIS LOCATION ==========
        # Combine 5-12 and 13-17 year olds into single 5-17 group using population-weighted averages
        youth_data = location_data.query("Age in ['5-12 Years', '13-17 Years']")
        
        def weighted_avg(group, column):
            """Calculate population-weighted average for a column."""
            return (group[column] * group["Population"]).sum() / group["Population"].sum()
        
        if not youth_data.empty:
            youth_aggregated = (
                youth_data.groupby("Week_Ending_Sat")
                .apply(lambda g: pd.Series({
                    "Geography": g["Geography"].iloc[0],
                    "Age": "5-17 Years",
                    "Population": g["Population"].sum(),
                    **{'Coverage_'+scn: weighted_avg(g, 'Coverage_'+scn) for scn in scenario_options}
                }), include_groups=False)
              .reset_index()
            )
        else:
            # Handle case where youth data might be missing for this location
            youth_aggregated = pd.DataFrame()
        
        # Keep other age groups and combine with aggregated youth data
        other_ages = location_data.query("Age not in ['5-12 Years', '13-17 Years', '6 Months - 17 Years']")
        
        if not youth_aggregated.empty:
            vaccine_schedule = pd.concat([other_ages, youth_aggregated], ignore_index=True)
        else:
            vaccine_schedule = other_ages.copy()
            
        vaccine_schedule = vaccine_schedule.sort_values(["Week_Ending_Sat", "Age"]).reset_index(drop=True)
        
        # ========== MAP TO MODEL POPULATION ==========
        age_to_index = {
            "6 Months - 4 Years": 0,
            "5-17 Years": 1,
            "18-49 Years": 2,
            "50-64 Years": 3,
            "65+ Years": 4
        }
        
        # Add model population and calculate cumulative doses for each scenario
        loc_epydemix = convert_location_name_format(location, 'epydemix_population')
        age_group_map_data = get_age_group_mapping(target_age_groups)

        population_model = {}
        for key, val in age_group_map_data.items():
            vals = [int(v.replace('+','')) for v in val]
            population_model[key] = sum(population_codebook[loc_epydemix][vals].values)

        # Add model population and calculate cumulative doses
        vaccine_schedule["population_model"] = vaccine_schedule["Age"].map(
            lambda age: list(population_model.values())[age_to_index[age]] if age in age_to_index else 0
        )
        
        for scn in scenario_options:
            vaccine_schedule[f"cumulative_doses_{scn}"] = (
                vaccine_schedule[f"Coverage_{scn}"] / 100 * vaccine_schedule["population_model"]
            ).round()
        
        # ========== CONVERT TO DAILY VACCINATION RATES ==========
        daily_vaccines_list = []
        
        for age_group in vaccine_schedule["Age"].unique():
            age_data = vaccine_schedule.query("Age == @age_group").copy()
            
            # Calculate new doses per week (difference from previous week)
            for scn in scenario_options:
                age_data[f"new_doses_{scn}"] = np.diff(
                    age_data[f"cumulative_doses_{scn}"], prepend=0
                )
            
            # Distribute weekly doses across days
            current_date = start_date
            
            for _, week_data in age_data.iterrows():
                week_end = week_data["Week_Ending_Sat"]
                days_in_period = (week_end - current_date).days + 1  # Include week_end
                
                if (week_end >= end_date):
                    days_in_period = (end_date - current_date).days + 1
                    daily_rates = {
                        scn: week_data[f"new_doses_{scn}"] / days_in_period 
                        for scn in scenario_options
                    }
                    
                    # Add daily entries for this period (including week_end)
                    for date in pd.date_range(current_date, end_date):
                        daily_vaccines_list.append({
                            "dates": date,
                            "location": location,
                            "age_group": age_group,
                            **{scn: round(daily_rates[scn]) for scn in scenario_options}
                        })
                    current_date = end_date + pd.Timedelta(days=1)  # End processing
                    break

                if days_in_period > 0:
                    # Calculate daily rates for this period
                    daily_rates = {
                        scn: week_data[f"new_doses_{scn}"] / days_in_period 
                        for scn in scenario_options
                    }
                    
                    # Add daily entries for this period (including week_end)
                    for date in pd.date_range(current_date, week_end):
                        daily_vaccines_list.append({
                            "dates": date,
                            "location": location,
                            "age_group": age_group,
                            **{scn: round(daily_rates[scn]) for scn in scenario_options}
                        })
                
                current_date = week_end + pd.Timedelta(days=1)  # Start next period
            
            # Fill remaining days with zeros
            for date in pd.date_range(current_date, end_date):
                daily_vaccines_list.append({
                    "dates": date,
                    "location": location,
                    "age_group": age_group,
                    **{scn: 0 for scn in scenario_options}
                })
        
        all_locations_data.extend(daily_vaccines_list)
    
    # ========== RESHAPE TO FINAL FORMAT ==========
    daily_vaccines_df = pd.DataFrame(all_locations_data)
    
    # Melt scenarios into rows, then pivot age groups into columns
    df_melted = pd.melt(
        daily_vaccines_df,
        id_vars=['dates', 'location', 'age_group'],
        value_vars=scenario_options,
        var_name='scenario',
        value_name='value'
    )
    
    df_final = df_melted.pivot_table(
        index=['dates', 'scenario', 'location'],
        columns='age_group',
        values='value',
        aggfunc='first'
    ).reset_index()
    
    # Clean up column names
    df_final.columns.name = None
    df_final.rename(columns={
        "6 Months - 4 Years": "0-4",
        "5-17 Years": "5-17",
        "18-49 Years": "18-49",
        "50-64 Years": "50-64",
        "65+ Years": "65+"
    }, inplace=True)

    # Format locations as ISO codes
    df_final['location'] = [convert_location_name_format(loc, 'ISO') for loc in df_final.location]

    # ========== WRITE OUTPUT CSV ==========
    if output_filepath:
        df_final.to_csv(output_filepath, index=False)
    
    return df_final


def make_vaccination_probability_function(
    origin_compartment: str,
    eligible_compartments: list[str]
) -> callable:
    """
    Returns a vaccination probability function for a given origin compartment and a set of 
    eligible compartments used in the denominator when allocating doses.

    Args:
        origin_compartment (str): The compartment receiving the vaccination (e.g., 'S').
        eligible_compartments (list of str): Compartments included in dose allocation denominator 
                                             (e.g., ['S', 'R']).

    Returns:
        function: A function (params, data) -> np.ndarray suitable for use with 
                  model.register_transition_kind.
    """
    import numpy as np
    from epydemix.model.epimodel import validate_transition_function

    def compute_vaccination_probability(params: list, data: dict) -> np.ndarray:
        """
        Computes the probability of the spontaneous transition required to move a certain 
        number of individuals out of the origin compartment at the current time step, based 
        on the total available doses and the distribution of the population across compartments.

        Args:
            params (list): A list containing one element: a 1D array of daily total doses 
                        (aligned with simulation time steps).
            data (dict): A dictionary with the following keys:
                - 't': Current time index (int)
                - 'dt': Duration of the current time step
                - 'pop': Array of compartment population counts
                - 'comp_indices': Dict mapping compartment names ('S', 'R', etc.) to their indices in 'pop'

        Returns:
            np.ndarray: Array of vaccination probabilities for the susceptible population, 
                        clipped between 0 and 0.999.
        """
        total_doses = params[0][data["t"]]
        origin_pop = data["pop"][data["comp_indices"][origin_compartment]]

        denom = sum(data["pop"][data["comp_indices"][comp]] for comp in eligible_compartments)

        with np.errstate(divide='ignore', invalid='ignore'):
            fraction_origin = np.where(denom > 0, origin_pop / denom, 0)

        effective_doses = total_doses * data["dt"] * fraction_origin

        with np.errstate(divide='ignore', invalid='ignore'):
            p_vax = np.where(origin_pop > 0, effective_doses / origin_pop, 0)

        return np.clip(p_vax, 0, 0.999)
    
    validate_transition_function(compute_vaccination_probability)
    return compute_vaccination_probability

def add_vaccination_schedule(
    model: EpiModel,
    vaccine_probability_function: Callable,
    source_comp: str,
    target_comp: str,
    vaccination_schedule:pd.DataFrame
) -> EpiModel:
    """
    Register and add a vaccination transition to the model using a time-varying schedule by age group.

    Parameters:
        model (EpiModel): The model object to which the vaccination schedule will be added. Must have a population 
                          age groups same as the columns in `vaccination_schedule`.
        vaccine_probability_function (Callable): A function defining time-dependent vaccination probabilities.
        vaccination_schedule (pd.DataFrame): Vaccination schedule with age groups as columns and time as rows.
                                             Must include all age groups used in the model. 
        source_comp (str): The name of the source compartment (e.g., "S").
        target_comp (str): The name of the target compartment (e.g., "SV").

    Returns:
        EpiModel: The model with the vaccination transition added.

    Raises:
        ValueError: If any age groups required by the model are missing from the DataFrame.
    """
    import copy
    from .utils import convert_location_name_format

    # Make a deep copy of the model to avoid modifying the original
    model = copy.deepcopy(model)

	# Location handling
    iso_location = convert_location_name_format(model.population.name, 'ISO')

    if 'location' not in vaccination_schedule.columns:
        raise ValueError(
            f"'location' column not found in vaccination_schedule.\n"
            f"Available columns: {list(vaccination_schedule.columns)}"
        )

    if iso_location not in vaccination_schedule['location'].unique():
        raise ValueError(
            f"Location {iso_location} not found in vaccination schedule data."
        )
    
    vaccination_schedule = vaccination_schedule.query("location == @iso_location").copy()
    model.register_transition_kind("vaccination", vaccine_probability_function)
    
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

def remove_vaccination_transitions(model, source_comp, target_comp):
        """
        Manually remove vaccination transitions from model. This prevents
        `add_vaccination_schedule` from creating duplicate transitions
        if it is called multiple times.
        """

        import copy

        # Make a deep copy of the model to avoid modifying the original
        model = copy.deepcopy(model)
        
        # Remove from transitions_list
        model.transitions_list = [
            t for t in model.transitions_list 
            if not (t.source == source_comp and t.target == target_comp and t.kind == "vaccination")
        ]
        
        # Remove from transitions dict
        if source_comp in model.transitions:
            model.transitions[source_comp] = [
                t for t in model.transitions[source_comp]
                if not (t.target == target_comp and t.kind == "vaccination")
            ]
        
        return model