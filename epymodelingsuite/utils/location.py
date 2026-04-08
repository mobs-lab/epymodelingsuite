"""Location validation and conversion utilities."""

import os
import sys

import pandas as pd


def get_location_codebook() -> pd.DataFrame:
    """Retrieve the location codebook as a Pandas DataFrame."""
    filename = os.path.join(os.path.dirname(sys.modules[__name__].__file__), "../data/location_codebook.csv")
    location_codebook = pd.read_csv(filename)

    return location_codebook


def get_flusight_locations() -> pd.DataFrame:
    """Retrieve the FluSight location data as a Pandas DataFrame."""
    filename = os.path.join(os.path.dirname(sys.modules[__name__].__file__), "../data/flusight_locations.csv")
    flusight_locations = pd.read_csv(filename)

    return flusight_locations


def get_flusight_population(location: str) -> int:
    """Retrieve the population of the specified location."""
    flusight_locations = get_flusight_locations()

    return flusight_locations[flusight_locations.location == convert_location_name_format(location, "FIPS")][
        "population"
    ].iloc[0]


METROCAST_PREFIX = "metrocast_location_"


def convert_location_name_format(
    value: str,
    output_format: str,
    input_format: str | None = None,
    location_type: str | None = None,
) -> str:
    """
    Convert location name from any valid format to the specified format.

    Available formats for ISO locations:
    - "ISO" - Location code in ISO 3166 (e.g., "US", "US-NY")
    - "epydemix_population" - Population names used by epydemix (e.g., "United_States_New_York")
    - "name" - Standard English names (e.g., "New York")
    - "abbreviation" - Two-letter postal abbreviations (e.g., "NY")
    - "FIPS" - Federal Information Processing Standard codes (e.g., "36")

    Available formats for metrocast locations:
    - "metrocast_location_id" - Location ID (e.g., "denver", "nenc")
    - "original_location_code" - Original code from source data (e.g., "688", "1")
    - "epydemix_population" - Prefixed location ID (e.g., "metrocast_location_denver")
    - "name" - Human-readable name (e.g., "Denver, CO")
    - "name_short" - Short name for plots (e.g., "Greater Boston, MA")
    - "hsa_counties" - Counties list (e.g., "Adams, Arapahoe, ...")
    - "state" - Full state name (e.g., "Colorado")
    - "abbreviation" - State abbreviation (e.g., "CO")
    - "ISO" - State ISO code (e.g., "US-CO")

    Parameters
    ----------
    value : str
        The location name to convert, in any valid format.
    output_format : str
        The format to convert to.
    input_format : str | None
        Optional. The format of the input value. If provided, enables direct lookup
        instead of auto-detection, which is more efficient for batch conversions.
    location_type : str | None
        Location type: "iso", "metrocast_location", or None for auto-detection.
        Auto-detection checks for metrocast prefix or metrocast location list.

    Returns
    -------
    str
        The converted location name.
    """
    # Auto-detect location_type if not provided
    if location_type is None:
        if value.startswith(METROCAST_PREFIX):
            location_type = "metrocast_location"
            input_format = "epydemix_population"
        elif value in get_metrocast_locations()["metrocast_location_id"].values:
            location_type = "metrocast_location"
        else:
            location_type = "iso"

    # Handle metrocast locations
    if location_type == "metrocast_location":
        # Parse location name from input
        if input_format == "epydemix_population" and value.startswith(METROCAST_PREFIX):
            location_id = value[len(METROCAST_PREFIX) :]
        else:
            location_id = value

        # Convert to output format
        if output_format == "epydemix_population":
            return f"{METROCAST_PREFIX}{location_id}"

        if output_format == "metrocast_location_id":
            return location_id

        # Formats that require CSV lookup
        csv_column_map = {
            "name": "location_name",
            "name_short": "location_name_short",
            "hsa_counties": "hsa_counties",
            "state": "state",
            "abbreviation": "state_abb",
            "ISO": "state_iso",
            "original_location_code": "original_location_code",
        }
        if output_format in csv_column_map:
            metrocast_locs = get_metrocast_locations()
            row = metrocast_locs[metrocast_locs["metrocast_location_id"] == location_id]
            if not row.empty:
                return row[csv_column_map[output_format]].iloc[0]

        # Unknown format - raise error for clarity
        valid_formats = ["epydemix_population", "metrocast_location_id", *csv_column_map.keys()]
        raise ValueError(
            f"Unknown output format '{output_format}' for metrocast location. Valid formats: {valid_formats}"
        )

    # ISO location handling
    codebook = get_location_codebook()

    # Match format strings to codebook columns
    format_dict = {
        "ISO": "ISO",
        "epydemix_population": "location_name_epydemix",
        "name": "location_name",
        "abbreviation": "location_abbreviation",
        "FIPS": "location_code",
    }

    # Find row with input value
    if input_format is not None:
        # Direct lookup using known input format (more efficient)
        input_col = format_dict[input_format]
        location = codebook[codebook[input_col] == value]
    else:
        # Auto-detect by searching all columns (current behavior)
        location = codebook[codebook.isin([value]).any(axis=1)]

    # Ensure value exists
    assert not location.empty, f"Supplied location value {value} does not match any valid format"

    # Location name in requested format
    result = location[format_dict[output_format]].values[0]

    # Zero-pad FIPS codes to 2 characters.
    # FluSight location codes use zero-padded FIPS codes.
    if output_format == "FIPS" and len(str(result)) == 1:
        result = str(result).zfill(2)

    return result


def get_metrocast_locations() -> pd.DataFrame:
    """
    Retrieve the metrocast location metadata as a Pandas DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: metrocast_location_id, original_location_code, state, state_abb,
        location_name, location_name_short, population, location_type, hsa_counties, state_iso
    """
    filename = os.path.join(os.path.dirname(sys.modules[__name__].__file__), "../data/metrocast_locations.csv")
    return pd.read_csv(filename)


def get_metrocast_population_data() -> pd.DataFrame:
    """
    Retrieve granular age-stratified population data for metrocast locations.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: metrocast_location_id, age, population
        Age values are single-year ages (0, 1, 2, ..., 83, 84+)
    """
    filename = os.path.join(os.path.dirname(sys.modules[__name__].__file__), "../data/metrocast_population.csv")
    population_data = pd.read_csv(filename)
    return population_data


def is_state_level_metrocast_location(location_name: str) -> bool:
    """
    Check if a metrocast location is a state-level aggregate.

    State-level metrocast locations have original_location_code == "All" in metrocast_locations.csv. These locations (e.g., "colorado", "georgia") represent entire states rather than sub-state regions like HSAs.

    Parameters
    ----------
    location_name : str
        Metrocast location name (e.g., "colorado", "denver")

    Returns
    -------
    bool
        True if the location is a state-level aggregate, False otherwise
    """
    metrocast_locs = get_metrocast_locations()
    row = metrocast_locs[metrocast_locs["metrocast_location_id"] == location_name]
    if row.empty:
        return False
    return row["original_location_code"].iloc[0] == "All"


def get_parent_region(
    location_name: str,
    output_format: str = "ISO",
    granularity: str = "state",
) -> str:
    """
    Get parent region for a location in specified format.

    Supports metrocast locations and ISO 3166-2 state codes.

    Parameters
    ----------
    location_name : str
        Location name: metrocast location (e.g., "denver", "metrocast_location_denver")
        or ISO 3166-2 state code (e.g., "US-MA")
    output_format : str
        Output format: "ISO", "FIPS", "abbreviation", "name", "epydemix_population"
    granularity : str
        Region level: "state", "country"

    Returns
    -------
    str
        Parent region identifier in requested format

    Examples
    --------
    >>> get_parent_region("denver")
    'US-CO'
    >>> get_parent_region("denver", output_format="abbreviation")
    'CO'
    >>> get_parent_region("denver", granularity="country")
    'US'
    >>> get_parent_region("US-MA", granularity="country")
    'US'
    """
    # Parse location name (handles "metrocast_location_denver" prefix)
    parsed_name, location_type = parse_population_name(location_name)

    # Determine if this is a metrocast location or ISO location
    if location_type == "metrocast_location":
        # Prefixed metrocast location
        metrocast_locs = get_metrocast_locations()
        location_row = metrocast_locs[metrocast_locs["metrocast_location_id"] == parsed_name]
        if location_row.empty:
            raise ValueError(f"Metrocast location '{parsed_name}' not found")
        state_iso = location_row["state_iso"].iloc[0]
    elif parsed_name in get_metrocast_locations()["metrocast_location_id"].values:
        # Raw metrocast location name
        metrocast_locs = get_metrocast_locations()
        location_row = metrocast_locs[metrocast_locs["metrocast_location_id"] == parsed_name]
        state_iso = location_row["state_iso"].iloc[0]
    elif "-" in parsed_name:
        # ISO 3166-2 state code (e.g., "US-MA")
        state_iso = parsed_name
    else:
        raise ValueError(f"Unknown location: '{location_name}'")

    if granularity == "country":
        # Extract country code (e.g., "US" from "US-CO")
        country = state_iso.split("-")[0]
        if output_format == "ISO":
            return country
        return convert_location_name_format(country, output_format, input_format="ISO")
    if granularity == "state":
        # Return state in requested format
        if output_format == "ISO":
            return state_iso
        return convert_location_name_format(state_iso, output_format, input_format="ISO")
    raise ValueError(f"Invalid granularity: {granularity}. Must be 'state' or 'country'")


def validate_iso3166(value: str) -> str:
    """
    Validate that `value` follows ISO 3166.

    Parameters
    ----------
    value : str
        Location code in ISO 3166. Countries use ISO 3166-1 alpha-2 country code
        (e.g., "US") and states/regions use ISO 3166-2 subdivision (e.g., "US-NY").

    Returns
    -------
    str
        The validated ISO 3166 code.

    Raises
    ------
    ValueError
        If the code is invalid.
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
    elif "-" in value:
        # Non-US subdivision: check format XX-YY
        parts = value.split("-")
        if len(parts) >= 2 and len(parts[1]) == 2:
            return value

    raise ValueError(f"Invalid ISO 3166 code: {value}")


def validate_metrocast_location(location_id: str) -> str:
    """
    Validate that metrocast location name exists in metrocast data.

    Parameters
    ----------
    location_id : str
        Location name to validate (e.g., "denver", "nenc")

    Returns
    -------
    str
        The validated location name

    Raises
    ------
    ValueError
        If location is not found in metrocast data
    """
    metrocast_locs = get_metrocast_locations()
    valid_locations = metrocast_locs["metrocast_location_id"].tolist()

    if location_id not in valid_locations:
        raise ValueError(
            f"Invalid metrocast location: {location_id}. Must be one of {', '.join(sorted(valid_locations))}"
        )

    return location_id


# Registry for location validators
LOCATION_VALIDATOR_REGISTRY = {
    "iso": validate_iso3166,
    "metrocast_location": validate_metrocast_location,
}


def validate_location_by_type(location_id: str, location_type: str) -> str:
    """
    Validate location id by type.

    Parameters
    ----------
    location_id : str
        Location identifier to validate
    location_type : str
        Type of location ("iso" or "metrocast_location")

    Returns
    -------
    str
        The validated location identifier

    Raises
    ------
    ValueError
        If location_type is not registered or location is invalid
    """
    if location_type not in LOCATION_VALIDATOR_REGISTRY:
        raise ValueError(
            f"Unknown location type: {location_type}. Must be one of {', '.join(LOCATION_VALIDATOR_REGISTRY.keys())}"
        )

    validator = LOCATION_VALIDATOR_REGISTRY[location_type]
    return validator(location_id)


def parse_population_name(population_name: str) -> tuple[str, str]:
    """
    Parse an epydemix population name into location name and type.

    Epydemix population names use two naming conventions:
    - Countries/states: "United_States_Massachusetts"
    - Metrocast locations: "metrocast_location_denver" (prefixed)

    Parameters
    ----------
    population_name : str
        The epydemix population name (from model.population.name or CalibrationOutput.population)

    Returns
    -------
    tuple[str, str]
        (location_name, location_type) tuple:
        - For countries/states: ("United_States_Massachusetts", "iso")
        - For metrocast: ("denver", "metrocast_location")
    """
    if population_name.startswith(METROCAST_PREFIX):
        location_name = population_name[len(METROCAST_PREFIX) :]
        return location_name, "metrocast_location"
    if population_name in get_metrocast_locations()["metrocast_location_id"].values:
        return population_name, "metrocast_location"
    return population_name, "iso"
