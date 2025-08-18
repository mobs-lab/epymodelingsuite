
from typing import Optional, Callable
import pandas as pd
from epydemix.model import EpiModel 

def scenario_to_epydemix(
    input_filepath: str,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    model: EpiModel,
    output_filepath: Optional[str] = None,
) -> pd.DataFrame:
    """
    Processes age-specific influenza vaccine coverage data with a single scenario from a CSV file into 
    daily vaccination schedules by age group for ALL locations.

    The function loads weekly cumulative coverage estimates from a CSV file containing a single scenario
    in a 'Coverage' column, maps them to the model population, computes new weekly doses, and distributes 
    them evenly across days between weekly reporting dates. Age groups 5-12 and 13-17 are merged into a 
    5-17 category using population-weighted averages. The output is a daily time series of vaccine doses 
    administered per age group across all geographies.

    Args:
        input_filepath (str): Path to CSV containing vaccination data with 'Coverage' column.
        start_date (str or Timestamp): Start date of the simulation period (used to build full timeline).
        end_date (str or Timestamp): End date of the simulation period (used to build full timeline).
        model (object): Epydemix EpiModel instance with a .population.Nk list for age group sizes.
        output_filepath (str, optional): If provided, the output DataFrame will be saved as a CSV at the given filepath.

    Returns:
        pd.DataFrame: DataFrame with columns ['dates', 'location', <age groups>] giving the 
                      daily vaccination counts per age group across all geographies.
    """
    import numpy as np
    from .utils import convert_location_name_format
    
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
    
    # Filter for date range
    vaccines = vaccines.query("Week_Ending_Sat >= @start_date")
    
    # ========== PROCESS ALL GEOGRAPHIES ==========
    all_locations_data = []
    
    for location in vaccines['Geography'].unique():
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
                    "Coverage": weighted_avg(g, "Coverage"),
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
        
        # Add model population and calculate cumulative doses
        vaccine_schedule["population_model"] = vaccine_schedule["Age"].map(
            lambda age: model.population.Nk[age_to_index[age]] if age in age_to_index else 0
        )
        
        vaccine_schedule["cumulative_doses"] = (
            vaccine_schedule["Coverage"] / 100 * vaccine_schedule["population_model"]
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
        
        all_locations_data.extend(daily_vaccines_list)
    
    # ========== RESHAPE TO FINAL FORMAT ==========
    daily_vaccines_df = pd.DataFrame(all_locations_data)

    
    df_final = daily_vaccines_df.pivot_table(
        index=['dates', 'location'],
        columns='age_group', 
        values='doses',
        aggfunc='first',
        fill_value=0
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

def smh_data_to_epydemix(
    input_filepath: str,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    model: EpiModel,
    output_filepath: Optional[str] = None,
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
    from .utils import convert_location_name_format
    
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
        vaccine_schedule["population_model"] = vaccine_schedule["Age"].map(
            lambda age: model.population.Nk[age_to_index[age]] if age in age_to_index else 0
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