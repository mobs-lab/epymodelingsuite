"""Canonical identity is resolved at input boundaries and survives without catalogs."""

from dataclasses import replace

import pandas as pd
import pytest

from epymodelingsuite.builders.utils import get_data_in_location
from epymodelingsuite.utils.location import (
    convert_location_name_format,
    get_flusight_population,
    get_hub_location_id,
    resolve_location,
    resolve_locations,
)
from epymodelingsuite.utils.populations import get_total_population
from epymodelingsuite.utils.resolved_location import ResolvedLocation


@pytest.mark.parametrize(
    "aliases,key",
    [
        (
            ["US-MA", "MA", "25", "Massachusetts", "United_States_Massachusetts", "United_States__Massachusetts"],
            ("iso", "US-MA"),
        ),
        (["denver", "metrocast_location_denver"], ("metrocast_location", "denver")),
    ],
)
def test_aliases_resolve_to_one_canonical_object(aliases, key):
    locations = [resolve_location(alias) for alias in aliases]
    assert all(location == locations[0] for location in locations)
    assert locations[0].key == key
    assert all(locations[0].matches(alias) for alias in aliases)


def test_every_kind_uses_saved_formats_without_source_access(monkeypatch):
    custom = ResolvedLocation(
        kind="custom",
        location_id="test-city",
        population_name="custom_location_test-city",
        location_name="Test City",
        location_name_short="City",
        source_iso="US-MA",
        identifiers=(("location_id", "test-city"),),
        total_population=8500,
        flusight_population=9000,
    )
    catalog = {"test-city": custom}
    locations = resolve_locations(["US-MA", "denver", "test-city"], catalog)
    assert resolve_location("custom_location_test-city", custom_populations=catalog) is custom

    def unavailable(*args, **kwargs):
        raise AssertionError("Resolved identities must not access source catalogs")

    for name in ("get_location_codebook", "get_metrocast_locations", "get_flusight_locations"):
        monkeypatch.setattr(f"epymodelingsuite.utils.location.{name}", unavailable)
    data = pd.DataFrame({"location": ["25", "denver", "test-city"], "value": [1, 2, 3]})
    for i, location in enumerate(locations, 1):
        assert resolve_location(location) is location
        assert get_hub_location_id(location) == data.location[i - 1]
        assert convert_location_name_format(location, "name") == location.location_name
        assert get_data_in_location(data, location, "location", "location_id").value.tolist() == [i]
    assert get_total_population(custom) == 8500
    assert get_flusight_population(custom) == 9000
    assert custom.source_iso == "US-MA"
    assert not custom.matches("US-MA")
    with pytest.raises(ValueError, match="Unsupported output format"):
        custom.format("ISO")


def test_explicit_kind_cannot_fall_back_to_a_different_source():
    with pytest.raises(ValueError):
        resolve_location("denver", "iso")
    with pytest.raises(ValueError):
        resolve_location("US-MA", "custom")
    assert not resolve_location("denver").matches("US-CO")


def test_international_iso_uses_pinned_mapping_but_has_no_invented_hub_id(monkeypatch):
    monkeypatch.setattr(
        "epymodelingsuite.utils.epydemix_data.get_available_locations",
        lambda: pd.DataFrame({"iso_code": ["JP-13"], "location": ["Japan__Tokyo-to"]}),
    )
    location = resolve_location("JP-13")
    assert location.key == ("iso", "JP-13")
    assert location.population_name == "Japan__Tokyo-to"
    assert location.format("ISO") == "JP-13"
    with pytest.raises(ValueError, match="Unsupported output format"):
        _ = location.hub_id
    with pytest.raises(ValueError, match="No unique epydemix"):
        resolve_location("GB-ENG")
    loaded = replace(location, total_population=123)
    assert get_total_population(loaded) == 123


def test_inherited_codes_and_labels_are_not_location_aliases():
    assert not resolve_location("colorado").matches("All")
    assert not resolve_location("colorado").matches("Colorado")
    assert resolve_location("US-CO").matches("Colorado")
    assert not resolve_location("boston").matches("22")


def test_observed_population_column_accepts_deprecated_alias_without_lookup(monkeypatch):
    location = resolve_location("US-CA")
    data = pd.DataFrame({"location": ["United_States_California", "United_States__California", "United_States_Texas"]})
    monkeypatch.setattr("epymodelingsuite.utils.location.get_location_codebook", lambda: pytest.fail("lookup"))
    assert len(get_data_in_location(data, location, "location", "epydemix_population")) == 2
