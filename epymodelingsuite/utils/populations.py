"""Population-related utilities."""

import logging

import pandas as pd
from epydemix import EpiModel
from epydemix.population import Population

from .location import METROCAST_PREFIX, get_metrocast_locations

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

    Examples
    --------
    >>> get_age_group_mapping(["0-4", "5-17", "18-49", "50-64", "65+"])
    {
        "0-4": ["0", "1", "2", "3", "4"],
        "5-17": ["5", "6", "7", ..., "17"],
        "18-49": ["18", "19", ..., "49"],
        "50-64": ["50", "51", ..., "64"],
        "65+": ["65", "66", ..., "83", "84+"]
    }
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
    """
    Create an age group mapping where each single-year age maps to itself.

    This is useful when working with single-year age data and you need a
    consistent interface with `get_age_group_mapping`.

    Returns
    -------
    dict[str, list[str]]
        Mapping from single-year age groups to their ages (each containing one element).

    Examples
    --------
    >>> get_all_age_map()
    {
        "0-0": ["0"],
        "1-1": ["1"],
        ...
        "83-83": ["83"],
        "84+": ["84+"]
    }
    """
    # fmt: off
    age_map = get_age_group_mapping(
        ["0-0", "1-1", "2-2", "3-3", "4-4", "5-5", "6-6", "7-7", "8-8", "9-9", "10-10", "11-11", "12-12", "13-13", "14-14", "15-15", "16-16", "17-17", "18-18", "19-19", "20-20", "21-21", "22-22", "23-23", "24-24", "25-25", "26-26", "27-27", "28-28", "29-29", "30-30", "31-31", "32-32", "33-33", "34-34", "35-35", "36-36", "37-37", "38-38", "39-39", "40-40", "41-41", "42-42", "43-43", "44-44", "45-45", "46-46", "47-47", "48-48", "49-49", "50-50", "51-51", "52-52", "53-53", "54-54", "55-55", "56-56", "57-57", "58-58", "59-59", "60-60", "61-61", "62-62", "63-63", "64-64", "65-65", "66-66", "67-67", "68-68", "69-69", "70-70", "71-71", "72-72", "73-73", "74-74", "75-75", "76-76", "77-77", "78-78", "79-79", "80-80", "81-81", "82-82", "83-83", "84+"]
    )
    # fmt: on
    return age_map

def get_population_codebook() -> pd.DataFrame:
    """Retrieve the population codebook as a Pandas DataFrame."""
    import os
    import sys

    filename = os.path.join(os.path.dirname(sys.modules[__name__].__file__), "../data/population_codebook.csv")
    population_codebook = pd.read_csv(filename)

    return population_codebook


def get_total_population(epydemix_name: str) -> int:
    """Get total population for a given epydemix location name.

    Handles both ISO locations (countries/states) and metrocast locations.

    Parameters
    ----------
    epydemix_name : str
        Name of epydemix's population object (e.g., "United_States_Alabama" or "metrocast_location_denver")

    Returns
    -------
    int
        Total population for the location

    Raises
    ------
    KeyError
        If the location name is not found in the lookup table
    ValueError
        If a metrocast location is not found in metrocast data
    """
    # Handle metrocast locations
    if epydemix_name.startswith(METROCAST_PREFIX):
        location_name = epydemix_name[len(METROCAST_PREFIX) :]
        metrocast_locs = get_metrocast_locations()
        location_row = metrocast_locs[metrocast_locs["location"] == location_name]

        if location_row.empty:
            msg = f"Metrocast location '{location_name}' not found in metrocast locations data"
            raise ValueError(msg)

        return int(location_row["population"].iloc[0])

    # ISO locations - use hardcoded lookup
    POPULATION_LOOKUP = {
        "United_States": 338120586,
        "United_States_Alabama": 5108468,
        "United_States_Alaska": 733406,
        "United_States_Arizona": 7431344,
        "United_States_Arkansas": 3067732,
        "United_States_California": 38965193,
        "United_States_Colorado": 5877610,
        "United_States_Connecticut": 3617176,
        "United_States_Delaware": 1031890,
        "United_States_District_of_Columbia": 678972,
        "United_States_Florida": 22610726,
        "United_States_Georgia": 11029227,
        "United_States_Hawaii": 1435138,
        "United_States_Idaho": 1964726,
        "United_States_Illinois": 12549689,
        "United_States_Indiana": 6862199,
        "United_States_Iowa": 3207004,
        "United_States_Kansas": 2940546,
        "United_States_Kentucky": 4526154,
        "United_States_Louisiana": 4573749,
        "United_States_Maine": 1395722,
        "United_States_Maryland": 6180253,
        "United_States_Massachusetts": 7001399,
        "United_States_Michigan": 10037261,
        "United_States_Minnesota": 5737915,
        "United_States_Mississippi": 2939690,
        "United_States_Missouri": 6196156,
        "United_States_Montana": 1132812,
        "United_States_Nebraska": 1978379,
        "United_States_Nevada": 3194176,
        "United_States_New_Hampshire": 1402054,
        "United_States_New_Jersey": 9290841,
        "United_States_New_Mexico": 2114371,
        "United_States_New_York": 19571216,
        "United_States_North_Carolina": 10835491,
        "United_States_North_Dakota": 783926,
        "United_States_Ohio": 11785935,
        "United_States_Oklahoma": 4053824,
        "United_States_Oregon": 4233358,
        "United_States_Pennsylvania": 12961683,
        "United_States_Rhode_Island": 1095962,
        "United_States_South_Carolina": 5373555,
        "United_States_South_Dakota": 919318,
        "United_States_Tennessee": 7126489,
        "United_States_Texas": 30503301,
        "United_States_Utah": 3417734,
        "United_States_Vermont": 647464,
        "United_States_Virginia": 8715698,
        "United_States_Washington": 7812880,
        "United_States_West_Virginia": 1770071,
        "United_States_Wisconsin": 5910955,
        "United_States_Wyoming": 584057,
    }

    return POPULATION_LOOKUP[epydemix_name]


def make_dummy_population(model: EpiModel) -> Population:
    """Create a dummy population for testing purposes with 100 individuals per age group."""
    dummy_pop = Population(name="Dummy")
    dummy_pop.add_population(Nk=[100 for _ in model.population.age_groups], Nk_names=model.population.age_groups)
    return dummy_pop
