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


def make_dummy_population(model: EpiModel) -> Population:
    """Create a dummy population for testing purposes with 100 individuals per age group."""
    dummy_pop = Population(name="Dummy")
    dummy_pop.add_population(Nk=[100 for _ in model.population.age_groups], Nk_names=model.population.age_groups)
    return dummy_pop
