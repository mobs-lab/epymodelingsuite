"""Population-related utilities."""

import pandas as pd
from epydemix import EpiModel
from epydemix.population import Population


def get_population_codebook() -> pd.DataFrame:
    """Retrieve the population codebook as a Pandas DataFrame."""
    import os
    import sys

    filename = os.path.join(os.path.dirname(sys.modules[__name__].__file__), "../data/population_codebook.csv")
    population_codebook = pd.read_csv(filename)

    return population_codebook


def get_total_population(epydemix_name: str) -> int:
    """Get total population for a given epydemix location name.

    Population data is sourced from load_epydemix_population(epydemix_name).total_population.

    Parameters
    ----------
    epydemix_name : str
        Location name in epydemix format (e.g., "United_States_Alabama")

    Returns
    -------
    int
        Total population for the location

    Raises
    ------
    KeyError
        If the location name is not found in the lookup table
    """
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
