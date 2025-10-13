### utils.py
# Utility functions

import pandas as pd
import scipy.stats
from epydemix import EpiModel
from epydemix.population import Population

from .validation.common_validators import Distribution


def get_population_codebook() -> pd.DataFrame:
    """Retrieve the population codebook as a Pandas DataFrame."""
    import os
    import sys

    filename = os.path.join(os.path.dirname(sys.modules[__name__].__file__), "data/population_codebook.csv")
    population_codebook = pd.read_csv(filename)

    return population_codebook


def get_location_codebook() -> pd.DataFrame:
    """Retrieve the location codebook as a Pandas DataFrame."""
    import os
    import sys

    filename = os.path.join(os.path.dirname(sys.modules[__name__].__file__), "data/location_codebook.csv")
    location_codebook = pd.read_csv(filename)

    return location_codebook


def convert_location_name_format(value: str, output_format: str) -> str:
    """
    Convert location name from any valid format to the specified format.

    Available formats:
    "ISO" - Location code in ISO 3166. Countries use ISO 3166-1 alpha-2 country code (e.g. "US") and states/regions use ISO 3166-2 subdivision (e.g. "US-NY").
    "epydemix_population" - Population names used by epydemix (e.g. "United_States", "United_States_New_York").
    "name" - Standard English names (e.g. "United States", "New York").
    "abbreviation" - Two-letter postal abbreviations (e.g. "US", "NY").
    "FIPS" - Two-character Federal Information Processing Standard codes (e.g. "US", "36").

    Parameters
    ----------
            value (str): The location name to convert, in any valid format.
            output_format (str): The location name format to convert to, either "ISO", "epydemix_population", "name", "abbreviation", or "FIPS".

    Returns
    -------
            str: The converted location name.
    """
    # Retrieve codebook
    codebook = get_location_codebook()

    # Find row with input value
    location = codebook[codebook.isin([value]).any(axis=1)]

    # Ensure value exists
    assert not location.empty, f"Supplied location value {value} does not match any valid format"

    # Match format strings to codebook columns
    format_dict = {
        "ISO": "ISO",
        "epydemix_population": "location_name_epydemix",
        "name": "location_name",
        "abbreviation": "location_abbreviation",
        "FIPS": "location_code",
    }

    # Return location name in requested format
    return location[format_dict[output_format]].values[0]


def validate_iso3166(value: str) -> str:
    """
    Validate that `value` follows ISO 3166.

    Parameters
    ----------
                value (str): Location code in ISO 3166. Countries use ISO 3166-1 alpha-2 country code (e.g., "US") and states/regions use ISO 3166-2 subdivision (e.g., "US-NY").

    Returns
    -------
        str: The validated ISO 3166 code. Raises error when the code is invalid.

    """
    # fmt: off
    countries = ["AF", "AX", "AL", "DZ", "AS", "AD", "AO", "AI", "AQ", "AG", "AR", "AM", "AW", "AU", "AT", "AZ", "BS", "BH", "BD", "BB", "BY", "BE", "BZ", "BJ", "BM", "BT", "BO", "BQ", "BA", "BW", "BV", "BR", "IO", "BN", "BG", "BF", "BI", "CV", "KH", "CM", "CA", "KY", "CF", "TD", "CL", "CN", "CX", "CC", "CO", "KM", "CG", "CD", "CK", "CR", "CI", "HR", "CU", "CW", "CY", "CZ", "DK", "DJ", "DM", "DO", "EC", "EG", "SV", "GQ", "ER", "EE", "SZ", "ET", "FK", "FO", "FJ", "FI", "FR", "GF", "PF", "TF", "GA", "GM", "GE", "DE", "GH", "GI", "GR", "GL", "GD", "GP", "GU", "GT", "GG", "GN", "GW", "GY", "HT", "HM", "VA", "HN", "HK", "HU", "IS", "IN", "ID", "IR", "IQ", "IE", "IM", "IL", "IT", "JM", "JP", "JE", "JO", "KZ", "KE", "KI", "KP", "KR", "KW", "KG", "LA", "LV", "LB", "LS", "LR", "LY", "LI", "LT", "LU", "MO", "MG", "MW", "MY", "MV", "ML", "MT", "MH", "MQ", "MR", "MU", "YT", "MX", "FM", "MD", "MC", "MN", "ME", "MS", "MA", "MZ", "MM", "NA", "NR", "NP", "NL", "NC", "NZ", "NI", "NE", "NG", "NU", "NF", "MK", "MP", "NO", "OM", "PK", "PW", "PS", "PA", "PG", "PY", "PE", "PH", "PN", "PL", "PT", "PR", "QA", "RE", "RO", "RU", "RW", "BL", "SH", "KN", "LC", "MF", "PM", "VC", "WS", "SM", "ST", "SA", "SN", "RS", "SC", "SL", "SG", "SX", "SK", "SI", "SB", "SO", "ZA", "GS", "SS", "ES", "LK", "SD", "SR", "SJ", "SE", "CH", "SY", "TW", "TJ", "TZ", "TH", "TL", "TG", "TK", "TO", "TT", "TN", "TR", "TM", "TC", "TV", "UG", "UA", "AE", "GB", "US", "UM", "UY", "UZ", "VU", "VE", "VN", "VG", "VI", "WF", "EH", "YE", "ZM", "ZW"]
    subdivisions = {
        "US": ["US-AL", "US-AK", "US-AZ", "US-AR", "US-CA", "US-CO", "US-CT", "US-DE", "US-FL", "US-GA", "US-HI", "US-ID", "US-IL", "US-IN", "US-IA", "US-KS", "US-KY", "US-LA", "US-ME", "US-MD", "US-MA", "US-MI", "US-MN", "US-MS", "US-MO", "US-MT", "US-NE", "US-NV", "US-NH", "US-NJ", "US-NM", "US-NY", "US-NC", "US-ND", "US-OH", "US-OK", "US-OR", "US-PA", "US-RI", "US-SC", "US-SD", "US-TN", "US-TX", "US-UT", "US-VT", "US-VA", "US-WA", "US-WV", "US-WI", "US-WY", "US-DC", "US-AS", "US-GU", "US-MP", "US-PR", "US-UM", "US-VI"]
    }
    # fmt: on

    # Check ISO 3166-1 (country)
    if value in countries:
        return value

    # Check ISO 3166-2 (subdivision)
    if value.startswith("US-"):
        if value in subdivisions["US"]:
            return value
    else:
        sub = value.split("-")[1]
        if len(sub) == 2:
            return value

    raise ValueError(f"Invalid ISO 3166 code: {value}")


def distribution_to_scipy(distribution: Distribution):
    """
    Convert a Distribution object to a scipy distribution object.

    Parameters
    ----------
    distribution : Distribution
        A Distribution instance containing name, args, and kwargs for the scipy distribution.

    Returns
    -------
    scipy.stats distribution object
        The scipy distribution object created from the Distribution parameters.

    Examples
    --------
    >>> from flumodelingsuite.common_validators import Distribution
    >>> dist_config = Distribution(name="norm", args=[0, 1])
    >>> scipy_dist = distribution_to_scipy(dist_config)
    >>> scipy_dist.rvs(5)  # Generate 5 random samples

    >>> dist_config = Distribution(name="uniform", args=[0, 1])
    >>> scipy_dist = distribution_to_scipy(dist_config)
    >>> scipy_dist.rvs(10)  # Generate 10 random samples from uniform distribution
    """
    if distribution.type == "scipy":
        # Get the distribution class from scipy.stats
        dist_class = getattr(scipy.stats, distribution.name)

        # Create the distribution object with args and kwargs
        kwargs = distribution.kwargs or {}
        dist = dist_class(*distribution.args, **kwargs)
    return dist


def make_dummy_population(model: EpiModel) -> Population:
    dummy_pop = Population(name="Dummy")
    dummy_pop.add_population(Nk=[100 for _ in model.population.age_groups], Nk_names=model.population.age_groups)
    return dummy_pop
