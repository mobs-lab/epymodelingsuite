### school_closures.py
# Functions for calculating and adding school closure interventions to an epydemix EpiModel
import datetime as dt

from epydemix import EpiModel

import logging

logger = logging.getLogger(__name__)


def make_school_closure_dict(
    years: list[int], interval_reduction_iter: int | None = 5
) -> dict[str, set[tuple[dt.date, dt.date, str]]]:
    """
    Create a dictionary where the keys are location abbreviations and the values are sets of named tuples representing closures.
    For example, in normal usage if you want closures for 2024 and 2025 simply call make_school_closure_dict([2024, 2025]).

    The closure tuples in the output dictionary contain 'start_date' and 'end_date' as datetime.Date objects, as well as 'name' strings.

    For the total US model, we take the union of all school closures from our calendar data. The algorithm for merging these date intervals
    can take multiple iterations over the dataset, so interval_reduction_iter specifies the maximum number of iterations to try.

    We also add state-specific and national holidays using the holidays package.

    Parameters
    ----------
            years: a list of integer years for which to calculate closures
            interval_reduction_iter: maximum iterations used when merging date intervals for the total US model

    Returns
    -------
            closure_dict: a dictionary where the keys are location abbreviations and the values are sets of named tuples representing closures
    """
    import datetime as dt
    import os
    import sys
    from collections import namedtuple

    import pandas as pd
    from holidays import country_holidays

    # Read closures from school calendars
    filename = os.path.join(os.path.dirname(sys.modules[__name__].__file__), "data/school_calendars_formatted.csv")
    calendars = pd.read_csv(filename)

    # Initialize the output dict
    # Keys are 2-letter location codes e.g. 'US' or 'AK'
    # Values are sets of named tuples: Closure('start_date': dt.date, 'end_date': dt.date, 'name': string)
    closure_dict = {loc: set() for loc in calendars.state.unique()}
    Closure = namedtuple("Closure", ["start_date", "end_date", "name"])

    ### Determine closures for given years from school calendar data

    logger.info("Calculating closures from school calendars...\n")

    # Iterate through years
    for year in years:
        # Find Thanksgiving day
        thanksgiving = [
            item[0] for item in country_holidays("US", years=year).items() if item[1] == "Thanksgiving Day"
        ][0].day

        # Add closures from school calendars
        for closure in calendars.itertuples(index=False):
            # Place Thanksgiving break heuristically: all breaks are at least 2 days, so if a break is d days
            # the last day will be Black Friday and the first day will be d-1 days prior
            if closure.event_name == "Thanksgiving Break":
                duration = closure.end_day - closure.start_day
                thanksgiving_end = thanksgiving + 1
                thanksgiving_start = thanksgiving_end - duration
                start_date = dt.date(year=year, month=closure.start_month, day=thanksgiving_start)
                end_date = dt.date(year=year, month=closure.end_month, day=thanksgiving_end)
                closure_dict[closure.state].add(Closure(start_date, end_date, closure.event_name))

            # Handle Christmas break, which is the only break spanning across calendar years
            elif closure.end_month < closure.start_month:
                # Create the break for both ends of the year; duplicates will not apply as we are using sets
                start_date = dt.date(year=year, month=closure.start_month, day=closure.start_day)
                end_date = dt.date(year=year + 1, month=closure.end_month, day=closure.end_day)
                closure_dict[closure.state].add(Closure(start_date, end_date, closure.event_name))
                start_date = dt.date(year=year - 1, month=closure.start_month, day=closure.start_day)
                end_date = dt.date(year=year, month=closure.end_month, day=closure.end_day)
                closure_dict[closure.state].add(Closure(start_date, end_date, closure.event_name))

            # Add all other school closures
            else:
                start_date = dt.date(year=year, month=closure.start_month, day=closure.start_day)
                end_date = dt.date(year=year, month=closure.end_month, day=closure.end_day)
                closure_dict[closure.state].add(Closure(start_date, end_date, closure.event_name))

    ### Define functions to combine date intervals
    # Based on https://github.com/kwovadis/merge_overlapping_pandas_intervals

    def reduce_intervals(df: pd.DataFrame, cat: str, start_str: str, end_str: str) -> pd.DataFrame:
        # Based on: https://stackoverflow.com/questions/39143918/reducing-overlapping-intervals-in-pandas-dataframe-in-an-optimal-fashion

        # Identify overlapping intervals
        # Sort table
        tmp = df.sort_values(by=[cat, start_str]).reset_index(drop=True)
        # Create a new column 'InPrev' thats true for all rows > 0, if start time at Event_(X) > end time at Event_(X-1)
        tmp.loc[0, "InPrev"] = False
        tmp.InPrev.to_numpy()[1:] = tmp.loc[1:, start_str].reset_index(drop=True) > tmp.loc[
            : (len(tmp) - 2), end_str
        ].reset_index(drop=True)
        tmp["InPrev"] = tmp["InPrev"].astype("bool")
        # Create a new column 'GrpCount' that creates a cumulative sum of all 'InPrev' column bools
        # If 'GrpCount' does not change value between subsequent rows, these rows will be grouped into a single interval
        tmp["GrpCount"] = tmp.groupby([cat])["InPrev"].cumsum()

        # Group overlapping intervals
        # 1. Group table per 'org' & 'GrpCount'
        # 2. Aggregate data based on min/max within each group
        # 3. Reset index & keep only 'org','start','end' columns (i.e. get rid of 'InPrev' & 'GrpCount' columns)
        df_reduced = (
            tmp.groupby([cat, "GrpCount"])
            .agg({start_str: "min", end_str: "max"})
            .reset_index()[[cat, start_str, end_str]]
        )

        return df_reduced

    def reduction_loop(df_synthetic: pd.DataFrame, n_iter: int, cat: str, start_str: str, end_str: str, DEBUG: int = 1):
        # Identify & group overlapping intervals
        red_df = reduce_intervals(df_synthetic, cat, start_str, end_str)

        # DEBUG: keep track of how the data tables look like
        if DEBUG == 1:
            print("\n Prior df (iteration = 0)")
            print(df_synthetic)

            print("\n Posterior df (iteration = 1)")
            print(red_df)
            print("\n")

        if red_df is None:
            raise ValueError("Output is empty. Critical Error!!!")

        # If red_df == df_synthetic, then stop
        if red_df.equals(df_synthetic):
            print("Max reduction")
            print("Iteration = 1\n")
        else:
            df_synthetic = red_df.copy()

            i = 1
            while i < n_iter:
                print(f"Finished iteration = {i}. Progress to next iteration... \n")
                # Identify & group overlapping intervals
                red_df = reduce_intervals(df_synthetic, cat, start_str, end_str)

                # DEBUG: keep track of how the data tables look like
                if DEBUG == 1:
                    print(f"\n Prior df (iteration = {i})")
                    print(df_synthetic)
                    print(f"\n Posterior df (iteration = {i + 1})")
                    print(red_df)
                    print("\n")

                # If red_df == df_synthetic, then stop
                if red_df.equals(df_synthetic):
                    print(f"Data reduction succesful! Stopped at iteration {i + 1}\n")
                    break
                df_synthetic = red_df.copy()
                i += 1

            # if max possible number of iterations is reached, then stop
            if i == n_iter:
                raise RuntimeError(f"Data reduction failed! Stopped at iteration {i}\n")

        return red_df

    ### Combine all state school closures for total US model, but don't add yet
    # Based on https://github.com/kwovadis/merge_overlapping_pandas_intervals

    logger.info("Merging school closures for total US model...\n")

    # Make a df from the closure dict
    closure_df = pd.DataFrame([_ for sublist in list(closure_dict.values()) for _ in sublist])

    # Make a mapping from dates to integers
    unique_dates = sorted(pd.unique(closure_df[["start_date", "end_date"]].values.ravel("K")))
    date_to_int = {item[1]: item[0] for item in enumerate(unique_dates)}

    # Convert dates into integers in the df
    closure_df = closure_df.assign(
        start=closure_df.start_date.map(date_to_int),
        end=closure_df.end_date.map(date_to_int),
        name="US Combined School Closure",
    )  # setting names equal (we want to combine without grouping)

    # Select, sort, and deduplicate df for supplying to reduction loop
    closure_synthetic = closure_df[["name", "start", "end"]].sort_values(by=["name", "start"]).drop_duplicates().copy()

    # Apply reduction
    closure_reduced = reduction_loop(closure_synthetic, interval_reduction_iter, "name", "start", "end", DEBUG=0)

    # Convert integers back into dates in the reduced df
    int_to_date = {v: k for k, v in date_to_int.items()}
    closure_reduced = closure_reduced.assign(
        start_date=closure_reduced.start.map(int_to_date),
        end_date=closure_reduced.end.map(int_to_date),
        name=closure_reduced.name,
    )[["start_date", "end_date", "name"]]

    # Make a set of Closures from the reduced df, this will be added to the closure dict
    us_closures = set(
        Closure(closure.start_date, closure.end_date, closure.name)
        for closure in closure_reduced.itertuples(index=False)
    )

    ### Add all state and national holidays, check that holidays are not falling within previously defined school closures

    logger.info("Adding state and national holidays...\n")

    # Helper for determining whether a date falls between two other dates
    def date_in_range(start: dt.date, end: dt.date, comp: dt.date) -> bool:
        return start <= comp <= end

    # Add state holidays
    for loc in closure_dict:
        state_holidays = country_holidays("US", subdiv=loc, years=years)
        existing_closures = closure_dict[loc].copy()
        # Only add if holiday is not contained by previously defined school closures for the location
        [
            closure_dict[loc].add(Closure(holiday[0], holiday[0], holiday[1]))
            for holiday in state_holidays.items()
            if not any(
                [date_in_range(closure.start_date, closure.end_date, holiday[0]) for closure in existing_closures]
            )
        ]

    # Add national holidays for US model to the US closures set
    us_holidays = country_holidays("US", years=years)
    existing_closures = us_closures.copy()
    # Only add if holiday is not contained by previously defined school closures
    [
        us_closures.add(Closure(holiday[0], holiday[0], holiday[1]))
        for holiday in us_holidays.items()
        if not any([date_in_range(closure.start_date, closure.end_date, holiday[0]) for closure in existing_closures])
    ]

    # Add the US closures set to the closure dict
    closure_dict["US"] = us_closures

    logger.info("School closures computed.\n")
    return closure_dict


def add_school_closure_interventions(
    model: EpiModel, closure_dict: dict[str, set[tuple[dt.date, dt.date, str]]], reduction_factor: int
) -> EpiModel:
    """
    Add school closure interventions to a model. Called for effect.

    Parameters
    ----------
            model: an already defined epydemix EpiModel. This must be using a US population with contact matrices.
            closure_dict: a dictionary created by calling make_school_closure_dict(...) from this module.
            reduction_factor: the factor by which to reduce the contact matrix.

    Returns
    -------
            None
    """
    from .utils import convert_location_name_format

    # Get the school closures that apply to the location/population of the model
    closures = closure_dict[convert_location_name_format(model.population.name, "abbreviation")]

    # Add all interventions to the model
    [
        model.add_intervention(
            layer_name="school",
            start_date=closure.start_date,
            end_date=closure.end_date,
            reduction_factor=reduction_factor,
            name=closure.name,
        )
        for closure in closures
    ]

    logger.info(f"School closure interventions added to model for {model.population.name}\n")

    return model
