"""Custom population inputs owned by one build and metadata saved with its results."""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .epydemix_data import get_available_locations

if TYPE_CHECKING:
    from ..schema.basemodel import BaseEpiModel

logger = logging.getLogger(__name__)
CUSTOM_PREFIX = "custom_location_"


@dataclass(frozen=True)
class CustomLocation:
    location_id: str
    parent_iso: str
    location_name: str
    location_name_short: str
    total_population: float
    vaccination_scaling: tuple[tuple[str, float], ...] | None = None

    @property
    def population_name(self) -> str:
        return f"{CUSTOM_PREFIX}{self.location_id}"


@dataclass(frozen=True)
class CustomPopulation:
    location: CustomLocation
    population_by_age: tuple[float, ...]
    parent_population_name: str


def resolve_epydemix_name(value: str, available: pd.DataFrame | None = None) -> str:
    """Resolve an ISO code or population name against the pinned epydemix data."""
    if available is None:
        available = get_available_locations()
    matches = available[(available["iso_code"] == value) | (available["location"] == value)]
    if len(matches) != 1:
        raise ValueError(f"No unique epydemix population mapping for {value}")
    return str(matches.iloc[0]["location"])


def load_custom_locations(basemodel: "BaseEpiModel") -> dict[str, CustomPopulation]:
    """Read and validate custom CSVs once for a builder invocation."""
    from .location import get_location_codebook, get_metrocast_locations, validate_custom_location_id, validate_iso3166
    from .populations import aggregate_population_by_age_groups, get_population_codebook

    config = basemodel.population
    if config.locations_path is None and config.population_path is None:
        return {}
    if config.locations_path is None or config.population_path is None:
        raise ValueError("Custom locations require both locations_path and population_path")

    locations = pd.read_csv(
        config.locations_path,
        dtype={"location_id": str, "parent_iso": str, "location_name": str, "location_name_short": str},
    )
    populations = pd.read_csv(config.population_path, dtype={"location_id": str, "age": str})
    for label, table, required in (
        ("locations", locations, {"location_id", "parent_iso", "population", "location_name"}),
        ("population", populations, {"location_id", "age", "population"}),
    ):
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f"Custom {label} CSV missing columns: {sorted(missing)}")
        if table.empty or table[list(required)].isna().any().any():
            raise ValueError(f"Custom {label} CSV contains empty required values")
        table["population"] = pd.to_numeric(table["population"], errors="raise")
        if not np.isfinite(table["population"]).all() or (table["population"] < 0).any():
            raise ValueError(f"Custom {label} populations must be finite and nonnegative")

    if locations["location_id"].duplicated().any():
        raise ValueError("Duplicate custom location_id in locations CSV")
    if locations["location_name"].str.strip().eq("").any():
        raise ValueError("Custom location_name must not be blank")
    if populations.duplicated(["location_id", "age"]).any():
        raise ValueError("Duplicate custom location_id/age in population CSV")
    if set(locations["location_id"]) != set(populations["location_id"]):
        raise ValueError("Custom locations and population CSVs must contain the same location IDs")

    codebook = get_location_codebook()
    reserved = set(codebook.astype(str).to_numpy().ravel())
    reserved.update(get_metrocast_locations()["metrocast_location_id"])
    reserved.update({"all", "all-states", "all-metrocast"})
    for location_id in locations["location_id"]:
        validate_custom_location_id(location_id)
        if location_id in reserved or location_id.startswith((CUSTOM_PREFIX, "metrocast_location_")):
            raise ValueError(f"Custom location ID conflicts with an existing identifier: {location_id}")
        try:
            validate_iso3166(location_id)
        except ValueError:
            pass
        else:
            raise ValueError(f"Custom location ID conflicts with an ISO code: {location_id}")

    available = get_available_locations()
    parent_populations = get_population_codebook() if basemodel.vaccination else None
    expected_ages = {str(age) for age in range(84)} | {"84+"}
    result = {}
    for row in locations.to_dict("records"):
        location_id = row["location_id"]
        parent_iso = validate_iso3166(row["parent_iso"])
        parent_name = resolve_epydemix_name(parent_iso, available)
        age_data = populations[populations["location_id"] == location_id]
        if set(age_data["age"]) != expected_ages:
            raise ValueError(f"Custom location {location_id} must contain every age 0..83 and 84+")
        total = float(row["population"])
        age_total = float(age_data["population"].sum())
        if total <= 0 or age_total <= 0:
            raise ValueError(f"Custom location {location_id} must have positive total population")
        if abs(age_total - total) / total > 0.01:
            logger.warning(
                "Custom location %s total population %s differs from age total %s by more than 1%%",
                location_id,
                total,
                age_total,
            )
        age_data = age_data.set_index("age").loc[[str(age) for age in range(84)] + ["84+"]]
        counts = tuple(float(value) for value in age_data["population"])
        scaling = None
        if parent_populations is not None:
            if parent_name not in parent_populations:
                raise ValueError(f"Vaccination population data unavailable for custom parent {parent_iso}")
            custom_counts = aggregate_population_by_age_groups(age_data.reset_index(), config.age_groups)
            parent_counts = aggregate_population_by_age_groups(parent_populations[parent_name], config.age_groups)
            if any(value <= 0 for value in parent_counts.values()):
                raise ValueError(f"Vaccination parent {parent_iso} has an empty age group")
            scaling = tuple((group, custom_counts[group] / parent_counts[group]) for group in config.age_groups)
        short_name = row.get("location_name_short")
        if pd.isna(short_name) or not short_name:
            short_name = row["location_name"]
        location = CustomLocation(location_id, parent_iso, row["location_name"], short_name, total, scaling)
        result[location_id] = CustomPopulation(location, counts, parent_name)
    return result
