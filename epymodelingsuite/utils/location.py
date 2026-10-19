"""Location validation and conversion utilities."""

import os
import re
import sys

import pandas as pd
import pycountry

from .resolved_location import ResolvedLocation

CUSTOM_PREFIX = "custom_location_"


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


def get_flusight_population(location: str | ResolvedLocation) -> int:
    """Return the region's saved denominator for hub rate calculations."""
    resolved = resolve_location(location)
    if resolved.flusight_population is None:
        raise ValueError(f"No FluSight population available for {resolved.location_id}")
    return int(resolved.flusight_population)


METROCAST_PREFIX = "metrocast_location_"


def convert_location_name_format(
    value: str | ResolvedLocation,
    output_format: str,
    input_format: str | None = None,
    location_type: str | None = None,
) -> str:
    """Adapt a region to an external table format; resolved inputs need no lookup."""
    if isinstance(value, ResolvedLocation):
        return value.format(output_format)
    # Tabular interchange can carry the adapter name without a full snapshot.
    if value.startswith(CUSTOM_PREFIX) and output_format in {"location_id", "epydemix_population"}:
        return value.removeprefix(CUSTOM_PREFIX) if output_format == "location_id" else value
    if (
        input_format is not None
        and (location_type is None or location_type == "iso")
        and not value.startswith(METROCAST_PREFIX)
    ):
        columns = {
            "ISO": "ISO",
            "epydemix_population": "location_name_epydemix",
            "name": "location_name",
            "abbreviation": "location_abbreviation",
            "FIPS": "location_code",
        }
        if input_format in columns:
            codebook = get_location_codebook()
            rows = codebook[codebook[columns[input_format]].astype(str) == str(value)]
            if not rows.empty:
                value = str(rows.iloc[0]["ISO"])
            elif location_type == "iso":
                raise ValueError(f"No ISO location mapping for {value} in the bundled codebook")
    return resolve_location(value, location_type).format(output_format)


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
    location_name: str | ResolvedLocation,
    output_format: str = "ISO",
    granularity: str = "state",
) -> str:
    """Get the inherited data region, or its country, in the requested format."""
    source = resolve_location(location_name).source_iso
    if granularity == "country":
        source = source.split("-")[0]
    elif granularity != "state":
        raise ValueError(f"Invalid granularity: {granularity}. Must be 'state' or 'country'")
    return source if output_format == "ISO" else convert_location_name_format(source, output_format, input_format="ISO")


def validate_iso3166(value: str) -> str:
    """Validate country and subdivision codes against pycountry's bundled ISO data.

    Population data availability is checked separately when loading a model.
    """
    if not re.fullmatch(r"[A-Z]{2}(?:-[A-Z0-9]{1,3})?", value):
        raise ValueError(f"Invalid ISO 3166 code format: {value}")
    if not (pycountry.countries.get(alpha_2=value) or pycountry.subdivisions.get(code=value)):
        raise ValueError(f"Invalid ISO 3166 code: {value}")
    return value


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


def validate_custom_location_id(value: str) -> str:
    """Check an external location identifier without reading its CSV."""
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ValueError(f"Invalid custom location ID: {value}")
    return value


# Registry for location validators
LOCATION_VALIDATOR_REGISTRY = {
    "iso": validate_iso3166,
    "metrocast_location": validate_metrocast_location,
    "custom": validate_custom_location_id,
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
    if population_name.startswith(CUSTOM_PREFIX):
        return population_name[len(CUSTOM_PREFIX) :], "custom"
    if population_name.startswith(METROCAST_PREFIX):
        location_name = population_name[len(METROCAST_PREFIX) :]
        return location_name, "metrocast_location"
    return population_name, "iso"


def validate_population_names(population_names: list[str | dict[str, str]]) -> None:
    """Validate config references; external custom IDs are resolved during building."""
    import warnings

    metrocast_ids = set(get_metrocast_locations()["metrocast_location_id"])
    for population in population_names:
        if isinstance(population, dict):
            name = population.get("name")
            if not name:
                raise ValueError("population dict must have 'name' field")
            validate_location_by_type(name, population.get("type", "iso"))
        elif not isinstance(population, str):
            raise ValueError(f"Invalid population format: {population}")
        elif population == "all":
            warnings.warn(
                "The 'all' keyword is deprecated. Use 'all-states' instead.", DeprecationWarning, stacklevel=2
            )
        elif population in ("all-states", "all-metrocast") or population in metrocast_ids:
            continue
        elif re.fullmatch(r"[A-Z]{2}(?:-[A-Z0-9]+)?", population):
            validate_iso3166(population)
        else:
            validate_custom_location_id(population)


def resolve_location(
    value: str | ResolvedLocation,
    location_type: str | None = None,
    custom_populations: dict | None = None,
) -> ResolvedLocation:
    """Resolve an input alias once, within the current build's catalog."""
    if isinstance(value, ResolvedLocation):
        if location_type is not None and value.kind != location_type:
            raise ValueError(f"Location {value.location_id} is not of type {location_type}")
        return value
    if not isinstance(value, str):
        raise ValueError(f"Invalid population format: {value}")
    if location_type not in {None, "iso", "metrocast_location", "custom"}:
        raise ValueError(f"Unknown location type: {location_type}")
    custom_id = value.removeprefix(CUSTOM_PREFIX)
    if location_type == "custom" or value.startswith(CUSTOM_PREFIX) or custom_id in (custom_populations or {}):
        definition = (custom_populations or {}).get(custom_id)
        if definition is None:
            raise ValueError(f"Unknown custom location: {custom_id}; resolve it with its build catalog")
        resolved = definition if isinstance(definition, ResolvedLocation) else definition.location
        if location_type is not None and resolved.kind != location_type:
            raise ValueError(f"Location {value} is not of type {location_type}")
        return resolved

    metro = get_metrocast_locations()
    metro_id = value.removeprefix(METROCAST_PREFIX)
    if location_type == "metrocast_location" or (
        location_type is None
        and (value.startswith(METROCAST_PREFIX) or metro_id in set(metro["metrocast_location_id"]))
    ):
        rows = metro[metro["metrocast_location_id"] == metro_id]
        if rows.empty:
            raise ValueError(f"Invalid metrocast location: {metro_id}")
        row = rows.iloc[0]
        formats = {
            "location_id": metro_id,
            "metrocast_location_id": metro_id,
            "original_location_code": str(row["original_location_code"]),
            "ISO": str(row["state_iso"]),
            "abbreviation": str(row["state_abb"]),
            "state": str(row["state"]),
            "hsa_counties": str(row["hsa_counties"]),
        }
        return ResolvedLocation(
            kind="metrocast_location",
            location_id=metro_id,
            population_name=f"{METROCAST_PREFIX}{metro_id}",
            location_name=str(row["location_name"]),
            location_name_short=str(row["location_name_short"]),
            source_iso=str(row["state_iso"]),
            identifiers=tuple(formats.items()),
            total_population=float(row["population"]),
            flusight_population=float(row["population"]),
        )

    codebook = get_location_codebook()
    rows = codebook[codebook.astype(str).isin([value]).any(axis=1)]
    if not rows.empty:
        from .populations import POPULATION_LOOKUP

        row = rows.iloc[0]
        iso, population_name = str(row["ISO"]), str(row["location_name_epydemix"])
        fips = str(row["location_code"])
        fips = fips.zfill(2) if fips.isdigit() else fips
        hub = get_flusight_locations()
        hub_rows = hub[hub["location"].astype(str) == fips]
        return ResolvedLocation(
            kind="iso",
            location_id=iso,
            population_name=population_name,
            location_name=str(row["location_name"]),
            location_name_short=str(row["location_name"]),
            source_iso=iso,
            identifiers=(
                ("ISO", iso),
                ("FIPS", fips),
                ("location_id", fips),
                ("abbreviation", str(row["location_abbreviation"])),
            ),
            aliases=(str(row["location_name_epydemix_deprecated"]),),
            total_population=POPULATION_LOOKUP.get(population_name),
            flusight_population=float(hub_rows.iloc[0]["population"]) if not hub_rows.empty else None,
        )

    try:
        iso = validate_iso3166(value)
    except ValueError:
        raise ValueError(f"No ISO location mapping for {value} in the bundled codebook") from None
    from .epydemix_data import get_available_locations

    available = get_available_locations()
    rows = available[available["iso_code"] == iso]
    if len(rows) != 1:
        raise ValueError(f"No unique epydemix population mapping for {value}")
    country = pycountry.countries.get(alpha_2=iso) or pycountry.subdivisions.get(code=iso)
    return ResolvedLocation(
        kind="iso",
        location_id=iso,
        population_name=str(rows.iloc[0]["location"]),
        location_name=country.name,
        location_name_short=country.name,
        source_iso=iso,
        identifiers=(("ISO", iso),),
    )


def resolve_locations(
    population_names: list[str | dict[str, str]], custom_populations: dict | None = None
) -> list[ResolvedLocation]:
    """Expand selectors and resolve every kind through the same identity boundary."""
    import warnings

    resolved = []
    for population in population_names:
        if isinstance(population, dict):
            if not population.get("name"):
                raise ValueError("population dict must have 'name' field")
            resolved.append(resolve_location(population["name"], population.get("type", "iso"), custom_populations))
        elif population in ("all", "all-states"):
            if population == "all":
                warnings.warn(
                    "The 'all' keyword is deprecated. Use 'all-states' instead.", DeprecationWarning, stacklevel=2
                )
            resolved.extend(resolve_location(iso, "iso") for iso in get_location_codebook()["ISO"])
        elif population == "all-metrocast":
            resolved.extend(
                resolve_location(name, "metrocast_location")
                for name in get_metrocast_locations()["metrocast_location_id"]
                if name != "nyc"
            )
        else:
            try:
                resolved.append(resolve_location(population, custom_populations=custom_populations))
            except ValueError as exc:
                raise ValueError(f"Invalid population: {population}. {exc}") from exc
    return resolved


def get_hub_location_id(population_name: str | ResolvedLocation) -> str:
    """Adapt a snapshot or a tabular population column to its submission ID."""
    if isinstance(population_name, str):
        for prefix in (CUSTOM_PREFIX, METROCAST_PREFIX):
            if population_name.startswith(prefix):
                return population_name.removeprefix(prefix)
    return resolve_location(population_name).hub_id
