"""Population-related utilities."""

import logging

import pandas as pd
from epydemix import EpiModel
from epydemix.population import Population

from .location import METROCAST_PREFIX, convert_location_name_format, get_metrocast_locations


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


def aggregate_population_by_age_groups(
    population_data: pd.DataFrame | pd.Series | dict,
    age_groups: list[str],
) -> dict[str, int]:
    """
    Aggregate single-year age population data into age groups.

    Parameters
    ----------
    population_data : pd.DataFrame | pd.Series | dict
        Population data indexed/keyed by single-year ages (0, 1, ..., 83, 84+).
        - DataFrame: must have 'age' and 'population' columns (pre-filtered to location)
        - Series/dict: indexed/keyed by age (int or str)
    age_groups : list[str]
        List of age group strings (e.g., ["0-4", "5-17", "18-49", "50-64", "65+"])

    Returns
    -------
    dict[str, int]
        Mapping of age group -> total population

    Notes
    -----
    This function uses `get_age_group_mapping` internally to expand age group
    labels into individual ages. For example:

    - ``"0-4"`` expands to ``["0", "1", "2", "3", "4"]``
    - ``"65+"`` expands to ``["65", "66", ..., "83", "84+"]``

    The population for each single-year age is then summed to get the total
    for each age group.

    Examples
    --------
    >>> import pandas as pd
    >>> # DataFrame input (metrocast population format)
    >>> df = pd.DataFrame({
    ...     "age": [0, 1, 2, 3, 4, 5, 6],
    ...     "population": [100, 100, 100, 100, 100, 200, 200]
    ... })
    >>> aggregate_population_by_age_groups(df, ["0-4", "5+"])
    {'0-4': 500, '5+': ...}

    >>> # Dict input
    >>> pop_dict = {0: 100, 1: 100, 2: 100, 3: 100, 4: 100}
    >>> aggregate_population_by_age_groups(pop_dict, ["0-2", "3+"])
    {'0-2': 300, '3+': 200}
    """
    age_group_map = get_age_group_mapping(age_groups)
    result = {}

    for age_group, ages in age_group_map.items():
        total = 0
        for age in ages:
            if isinstance(population_data, pd.DataFrame):
                # DataFrame with age, population columns (pre-filtered by caller)
                age_rows = population_data[population_data["age"].astype(str) == str(age)]
                total += age_rows["population"].sum()
            elif isinstance(population_data, pd.Series):
                # Series indexed by age (from population_codebook)
                age_idx = int(str(age).replace("+", ""))
                total += population_data.iloc[age_idx]
            else:
                # Dict keyed by age
                age_key = int(str(age).replace("+", "")) if age != "84+" else 84
                total += population_data.get(age_key, 0)

        result[age_group] = int(total)

    return result


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
        Name of epydemix's population object (e.g., "United_States__Alabama" or "metrocast_location_denver").
        Legacy single-underscore names (e.g., "United_States_Alabama") are normalized via the codebook.

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
        location_row = metrocast_locs[metrocast_locs["metrocast_location_id"] == location_name]

        if location_row.empty:
            msg = f"Metrocast location '{location_name}' not found in metrocast locations data"
            raise ValueError(msg)

        return int(location_row["population"].iloc[0])

    # Normalize to canonical epydemix name (handles legacy single-underscore inputs via deprecated column lookup)
    epydemix_name = convert_location_name_format(epydemix_name, "epydemix_population")

    # ISO locations - use hardcoded lookup
    POPULATION_LOOKUP = {
        "United_States": 338120586,
        "United_States__Alabama": 5108468,
        "United_States__Alaska": 733406,
        "United_States__Arizona": 7431344,
        "United_States__Arkansas": 3067732,
        "United_States__California": 38965193,
        "United_States__Colorado": 5877610,
        "United_States__Connecticut": 3617176,
        "United_States__Delaware": 1031890,
        "United_States__District_of_Columbia": 678972,
        "United_States__Florida": 22610726,
        "United_States__Georgia": 11029227,
        "United_States__Hawaii": 1435138,
        "United_States__Idaho": 1964726,
        "United_States__Illinois": 12549689,
        "United_States__Indiana": 6862199,
        "United_States__Iowa": 3207004,
        "United_States__Kansas": 2940546,
        "United_States__Kentucky": 4526154,
        "United_States__Louisiana": 4573749,
        "United_States__Maine": 1395722,
        "United_States__Maryland": 6180253,
        "United_States__Massachusetts": 7001399,
        "United_States__Michigan": 10037261,
        "United_States__Minnesota": 5737915,
        "United_States__Mississippi": 2939690,
        "United_States__Missouri": 6196156,
        "United_States__Montana": 1132812,
        "United_States__Nebraska": 1978379,
        "United_States__Nevada": 3194176,
        "United_States__New_Hampshire": 1402054,
        "United_States__New_Jersey": 9290841,
        "United_States__New_Mexico": 2114371,
        "United_States__New_York": 19571216,
        "United_States__North_Carolina": 10835491,
        "United_States__North_Dakota": 783926,
        "United_States__Ohio": 11785935,
        "United_States__Oklahoma": 4053824,
        "United_States__Oregon": 4233358,
        "United_States__Pennsylvania": 12961683,
        "United_States__Rhode_Island": 1095962,
        "United_States__South_Carolina": 5373555,
        "United_States__South_Dakota": 919318,
        "United_States__Tennessee": 7126489,
        "United_States__Texas": 30503301,
        "United_States__Utah": 3417734,
        "United_States__Vermont": 647464,
        "United_States__Virginia": 8715698,
        "United_States__Washington": 7812880,
        "United_States__West_Virginia": 1770071,
        "United_States__Wisconsin": 5910955,
        "United_States__Wyoming": 584057,
    }

    return POPULATION_LOOKUP[epydemix_name]


def make_dummy_population(model: EpiModel) -> Population:
    """Create a dummy population for testing purposes with 100 individuals per age group."""
    dummy_pop = Population(name="Dummy")
    dummy_pop.add_population(Nk=[100 for _ in model.population.age_groups], Nk_names=model.population.age_groups)
    return dummy_pop
