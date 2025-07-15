
from typing import Optional, Callable
import pandas as pd
from epydemix.model import EpiModel 

def smh_data_to_epydemix(
    start_date: str,
    end_date: str,
    location: str,
    model: EpiModel,
    scenario: str = "C_D",
    output_filename: Optional[str]=None,
) -> pd.DataFrame:
    """
    Processes age-specific influenza vaccine coverage data from the scenario modeling hub into 
    daily vaccination schedules by age group for a given scenario and location.

    The function loads weekly cumulative coverage estimates from a CSV file, maps them to 
    the model population, computes new weekly doses, and distributes them evenly across days 
    between weekly reporting dates. Age groups 5-12 and 13-17 are merged into a 5-17 category 
    using population-weighted averages. The output is a daily time series of vaccine doses 
    administered per age group for the specified scenario.

    Args:
        start_date (str or Timestamp): Start date of the simulation period (used to build full timeline).
        end_date (str or Timestamp): End date of the simulation period (used to build full timeline).
        location (str): Name of the geography to filter vaccine data (e.g., 'California').
        model (object): Epydemix EpiModel instance with a .population.Nk list for age group sizes.
        scenario (str): One of {'A_B', 'C_D', 'E_F'}, indicating which vaccination scenario to extract.
        output_filename (str, optional): If provided, the output DataFrame will be saved as a CSV file with this name.

    Returns:
        pd.DataFrame: DataFrame with columns ['dates', 'scenario', <age groups>] giving the 
                      daily vaccination counts per age group for the selected scenario.
    """
    import numpy as np
    import os
    import sys
    
    # ========== LOAD AND FILTER DATA ==========
    filename = os.path.join(os.path.dirname(sys.modules[__name__].__file__), "data/vaccine_scenarios_2425.csv")
    vaccines = pd.read_csv(filename)
    vaccines["Week_Ending_Sat"] = pd.to_datetime(vaccines["Week_Ending_Sat"])
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    location = location
    if location not in vaccines['Geography'].unique():
        raise ValueError(f"Location '{location}' not found in vaccine data. \n Available locations: {vaccines['Geography'].unique()}")
    vaccines = vaccines.query("Geography == @location and Week_Ending_Sat >= @start_date").rename(
        columns={
            "flu.coverage.rd2425.sc_A_B": "Coverage_A_B", 
            "flu.coverage.rd2425.sc_C_D": "Coverage_C_D",
            "flu.coverage.rd2425.sc_E_F": "Coverage_E_F"
        }
    )
    
    # ========== AGGREGATE AGE GROUPS ==========
    # Combine 5-12 and 13-17 year olds into single 5-17 group using population-weighted averages
    youth_data = vaccines.query("Age in ['5-12 Years', '13-17 Years']")
    
    def weighted_avg(group, column):
        """Calculate population-weighted average for a column."""
        return (group[column] * group["Population"]).sum() / group["Population"].sum()
    
    youth_aggregated = (
        youth_data.groupby("Week_Ending_Sat")
        .apply(lambda g: pd.Series({
            "Geography": g["Geography"].iloc[0],
            "Age": "5-17 Years",
            "Population": g["Population"].sum(),
            "Coverage_A_B": weighted_avg(g, "Coverage_A_B"),
            "Coverage_C_D": weighted_avg(g, "Coverage_C_D"),
            "Coverage_E_F": weighted_avg(g, "Coverage_E_F"),
        }))
        .reset_index()
    )
    
    # Keep other age groups and combine with aggregated youth data
    other_ages = vaccines.query("Age not in ['5-12 Years', '13-17 Years', '6 Months - 17 Years']")
    vaccine_schedule = pd.concat([other_ages, youth_aggregated], ignore_index=True)
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
        lambda age: model.population.Nk[age_to_index[age]]
    )
    
    for scn in ["A_B", "C_D", "E_F"]:
        vaccine_schedule[f"cumulative_doses_{scn}"] = (
            vaccine_schedule[f"Coverage_{scn}"] / 100 * vaccine_schedule["population_model"]
        ).round()
    
    # ========== CONVERT TO DAILY VACCINATION RATES ==========
    # Create date range with buffer before and after data
    daily_vaccines_list = []
    
    for age_group in vaccine_schedule["Age"].unique():
        age_data = vaccine_schedule.query("Age == @age_group").copy()
        
        # Calculate new doses per week (difference from previous week)
        for scn in ["A_B", "C_D", "E_F"]:
            age_data[f"new_doses_{scn}"] = np.diff(
                age_data[f"cumulative_doses_{scn}"], prepend=0
            )
        
        # Distribute weekly doses across days
        current_date = start_date
        
        for _, week_data in age_data.iterrows():
            week_end = week_data["Week_Ending_Sat"]
            days_in_period = (week_end - current_date).days
            
            if days_in_period > 0:
                # Calculate daily rates for this period
                daily_rates = {
                    scn: week_data[f"new_doses_{scn}"] / days_in_period 
                    for scn in ["A_B", "C_D", "E_F"]
                }
                
                # Add daily entries for this period
                for date in pd.date_range(current_date, week_end - pd.Timedelta(days=1)):
                    daily_vaccines_list.append({
                        "dates": date,
                        "age_group": age_group,
                        **{scn: daily_rates[scn] for scn in ["A_B", "C_D", "E_F"]}
                    })
            
            current_date = week_end
        
        # Fill remaining days with zeros
        for date in pd.date_range(current_date, end_date):
            daily_vaccines_list.append({
                "dates": date,
                "age_group": age_group,
                "A_B": 0,
                "C_D": 0,
                "E_F": 0
            })
    
    # ========== RESHAPE TO FINAL FORMAT ==========
    daily_vaccines_df = pd.DataFrame(daily_vaccines_list)
    
    # Melt scenarios into rows, then pivot age groups into columns
    df_melted = pd.melt(
        daily_vaccines_df,
        id_vars=['dates', 'age_group'],
        value_vars=['A_B', 'C_D', 'E_F'],
        var_name='scenario',
        value_name='value'
    )
    
    df_final = df_melted.pivot_table(
        index=['dates', 'scenario'],
        columns='age_group',
        values='value',
        aggfunc='first'
    ).reset_index()
    
    df_final = df_final[df_final['scenario'] == scenario].copy()
    # Clean up column names
    df_final.columns.name = None
    df_final.rename(columns={
        "6 Months - 4 Years": "0-4",
        "5-17 Years": "5-17",
        "18-49 Years": "18-49",
        "50-64 Years": "50-64",
        "65+ Years": "65+"
    }, inplace=True)

    # ========== WRITE OUTPUT CSV ==========
    if output_filename:
        output_path = os.path.join(os.path.dirname(sys.modules[__name__].__file__), "data", output_filename + ".csv")
        df_final.to_csv(output_path, index=False)
    
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

    return compute_vaccination_probability

def add_vaccination_schedule(
    model: EpiModel,
    vaccine_probability_function: Callable,
    location: str,
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
        location (str): Location name suffix (e.g. "California") to be prefixed with "United_States_".
        vaccination_schedule (pd.DataFrame): Vaccination schedule with age groups as columns and time as rows.
                                             Must include all age groups used in the model. 
        source_comp (str): The name of the source compartment (e.g., "S").
        target_comp (str): The name of the target compartment (e.g., "SV").

    Returns:
        EpiModel: The model with the vaccination transition added.

    Raises:
        ValueError: If any age groups required by the model are missing from the DataFrame.
    """

    # import os
    # import sys
    
    # if vaccination_schedule is None:
    #     schedule_path = os.path.join(
    #         os.path.dirname(sys.modules[__name__].__file__),
    #         "data/vaccine_schedule.csv"
    #     )
    #     vaccination_schedule = pd.read_csv(schedule_path)

    location = "United_States_" + location
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
    model.add_transition(source_comp, target_comp, params=vaccine_schedule, kind="vaccination")
    
    return model

