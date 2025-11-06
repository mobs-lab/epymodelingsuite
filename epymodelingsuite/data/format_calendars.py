import pandas as pd

### This script cleans our school closure data and saves it in a consistent format.

# read the uncleaned data
school_calendars = pd.read_csv("school_calendars_corrected.csv", parse_dates=["start", "end"])

# normalize names
school_calendars.event_name = school_calendars.event_name.str.removesuffix("\t").str.strip().str.title()

# NOTE: One state, Virginia, has a Rosh Hashanah break. This may actually be representative of the state, as
# the break exists in at least Fairfax County, Arlington County, and Richmond. This creates a couple problems.
# First, if treating US school holidays as the sum of all state holidays, this will be misrepresentative.
# Second, Rosh Hashanah is irregular. We might be able to handle this like Thanksgiving, but for now I'll
# just exclude Rosh Hashanah.
school_calendars = school_calendars[school_calendars.event_name != "Rosh Hashanah Break"]

# NOTE: Three states have an Easter break. Since Easter is an irregular holiday I'll also drop it for now.
school_calendars = school_calendars[school_calendars.event_name != "Easter Break"]

# NOTE: Federal and state holidays will be added by the functions for adding school closure interventions to models

# create summer breaks from first/last day of school fields
school_calendars = school_calendars.sort_values(["state", "start"]).drop(columns=["NCES_district_id", "name"])

summer_breaks = (
    school_calendars[school_calendars.event_name.isin(["First Day Of School", "Last Day Of School"])]
    .pivot(columns="event_name", index="state", values="start")
    .reset_index()
    .rename(columns={"Last Day Of School": "start", "First Day Of School": "end"})
)
summer_breaks["event_name"] = "Summer Break"

school_calendars = pd.concat([school_calendars, summer_breaks]).dropna(ignore_index=True)

# store only the day and month of closures
school_calendars["start_day"] = school_calendars.start.dt.day
school_calendars["start_month"] = school_calendars.start.dt.month
school_calendars["end_day"] = school_calendars.end.dt.day
school_calendars["end_month"] = school_calendars.end.dt.month
school_calendars.drop(columns=["start", "end"], inplace=True)

# save to csv
school_calendars.to_csv("school_calendars_formatted.csv", index=False)
