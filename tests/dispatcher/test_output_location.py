"""Tests for location helper functions."""

import pytest

from epymodelingsuite.utils.location import convert_location_name_format, parse_population_name
from epymodelingsuite.utils.populations import get_total_population


class TestParsePopulationName:
    """Tests for parse_population_name function."""

    def test_iso_location(self):
        """Test parsing ISO location population name (epydemix format)."""
        name, loc_type = parse_population_name("United_States_Massachusetts")
        assert name == "United_States_Massachusetts"
        assert loc_type == "iso"

    def test_metrocast_location(self):
        """Test parsing metrocast location population name."""
        name, loc_type = parse_population_name("metrocast_location_denver")
        assert name == "denver"
        assert loc_type == "metrocast_location"

    def test_metrocast_with_underscores(self):
        """Test parsing metrocast location with underscores in name."""
        name, loc_type = parse_population_name("metrocast_location_colorado-springs")
        assert name == "colorado-springs"
        assert loc_type == "metrocast_location"


class TestConvertLocationNameFormatISO:
    """Tests for convert_location_name_format with ISO locations."""

    # ISO has 5 formats: ISO, epydemix_population, name, abbreviation, FIPS

    # From ISO format
    def test_iso_to_iso(self):
        """Test ISO to ISO (identity)."""
        result = convert_location_name_format("US-MA", "ISO")
        assert result == "US-MA"

    def test_iso_to_epydemix_population(self):
        """Test ISO to epydemix_population."""
        result = convert_location_name_format("US-MA", "epydemix_population")
        assert result == "United_States__Massachusetts"

    def test_iso_to_name(self):
        """Test ISO to name."""
        result = convert_location_name_format("US-MA", "name")
        assert result == "Massachusetts"

    def test_iso_to_abbreviation(self):
        """Test ISO to abbreviation."""
        result = convert_location_name_format("US-MA", "abbreviation")
        assert result == "MA"

    def test_iso_to_fips(self):
        """Test ISO to FIPS."""
        result = convert_location_name_format("US-MA", "FIPS")
        assert result == "25"

    # From epydemix_population format
    def test_epydemix_population_to_iso(self):
        """Test epydemix_population to ISO."""
        result = convert_location_name_format("United_States_Massachusetts", "ISO")
        assert result == "US-MA"

    def test_epydemix_population_to_epydemix_population(self):
        """Test epydemix_population (legacy single-underscore) to epydemix_population (canonical double-underscore)."""
        result = convert_location_name_format("United_States_Massachusetts", "epydemix_population")
        assert result == "United_States__Massachusetts"

    def test_epydemix_population_canonical_to_canonical(self):
        """Test that the canonical double-underscore name is also accepted as input."""
        result = convert_location_name_format("United_States__Massachusetts", "epydemix_population")
        assert result == "United_States__Massachusetts"

    def test_epydemix_population_to_name(self):
        """Test epydemix_population to name."""
        result = convert_location_name_format("United_States_Massachusetts", "name")
        assert result == "Massachusetts"

    def test_epydemix_population_to_abbreviation(self):
        """Test epydemix_population to abbreviation."""
        result = convert_location_name_format("United_States_Massachusetts", "abbreviation")
        assert result == "MA"

    def test_epydemix_population_to_fips(self):
        """Test epydemix_population to FIPS."""
        result = convert_location_name_format("United_States_Massachusetts", "FIPS")
        assert result == "25"

    # From name format
    def test_name_to_iso(self):
        """Test name to ISO."""
        result = convert_location_name_format("Massachusetts", "ISO")
        assert result == "US-MA"

    def test_name_to_epydemix_population(self):
        """Test name to epydemix_population."""
        result = convert_location_name_format("Massachusetts", "epydemix_population")
        assert result == "United_States__Massachusetts"

    def test_name_to_name(self):
        """Test name to name (identity)."""
        result = convert_location_name_format("Massachusetts", "name")
        assert result == "Massachusetts"

    def test_name_to_abbreviation(self):
        """Test name to abbreviation."""
        result = convert_location_name_format("Massachusetts", "abbreviation")
        assert result == "MA"

    def test_name_to_fips(self):
        """Test name to FIPS."""
        result = convert_location_name_format("Massachusetts", "FIPS")
        assert result == "25"

    # From abbreviation format
    def test_abbreviation_to_iso(self):
        """Test abbreviation to ISO."""
        result = convert_location_name_format("MA", "ISO")
        assert result == "US-MA"

    def test_abbreviation_to_epydemix_population(self):
        """Test abbreviation to epydemix_population."""
        result = convert_location_name_format("MA", "epydemix_population")
        assert result == "United_States__Massachusetts"

    def test_abbreviation_to_name(self):
        """Test abbreviation to name."""
        result = convert_location_name_format("MA", "name")
        assert result == "Massachusetts"

    def test_abbreviation_to_abbreviation(self):
        """Test abbreviation to abbreviation (identity)."""
        result = convert_location_name_format("MA", "abbreviation")
        assert result == "MA"

    def test_abbreviation_to_fips(self):
        """Test abbreviation to FIPS."""
        result = convert_location_name_format("MA", "FIPS")
        assert result == "25"

    # From FIPS format
    def test_fips_to_iso(self):
        """Test FIPS to ISO."""
        result = convert_location_name_format("25", "ISO")
        assert result == "US-MA"

    def test_fips_to_epydemix_population(self):
        """Test FIPS to epydemix_population."""
        result = convert_location_name_format("25", "epydemix_population")
        assert result == "United_States__Massachusetts"

    def test_fips_to_name(self):
        """Test FIPS to name."""
        result = convert_location_name_format("25", "name")
        assert result == "Massachusetts"

    def test_fips_to_abbreviation(self):
        """Test FIPS to abbreviation."""
        result = convert_location_name_format("25", "abbreviation")
        assert result == "MA"

    def test_fips_to_fips(self):
        """Test FIPS to FIPS (identity)."""
        result = convert_location_name_format("25", "FIPS")
        assert result == "25"

    # Special cases
    def test_national_iso(self):
        """Test national level ISO code."""
        result = convert_location_name_format("US", "ISO")
        assert result == "US"

    def test_national_to_name(self):
        """Test national level to name."""
        result = convert_location_name_format("US", "name")
        assert result == "United States"

    def test_fips_zero_padding(self):
        """Test FIPS codes are zero-padded to 2 characters."""
        # Alabama has FIPS code 1, should be zero-padded to "01"
        result = convert_location_name_format("US-AL", "FIPS")
        assert result == "01"

    def test_explicit_input_format_iso(self):
        """Test with explicit input_format for efficiency."""
        result = convert_location_name_format("US-MA", "name", input_format="ISO")
        assert result == "Massachusetts"

    def test_explicit_location_type_iso(self):
        """Test with explicit location_type."""
        result = convert_location_name_format("US-MA", "name", location_type="iso")
        assert result == "Massachusetts"


class TestConvertLocationNameFormatMetrocast:
    """Tests for convert_location_name_format with metrocast locations."""

    # Metrocast has formats: epydemix_population, name, hsa_counties, state, abbreviation, ISO
    # Other/unknown formats return location ID

    # From location ID (auto-detected as metrocast)
    def test_location_id_to_epydemix_population(self):
        """Test location ID to epydemix_population."""
        result = convert_location_name_format("denver", "epydemix_population")
        assert result == "metrocast_location_denver"

    def test_location_id_to_name(self):
        """Test location ID to name."""
        result = convert_location_name_format("denver", "name")
        assert result == "Denver, CO"

    def test_location_id_to_hsa_counties(self):
        """Test location ID to hsa_counties."""
        result = convert_location_name_format("denver", "hsa_counties")
        assert "Denver" in result
        assert "Adams" in result

    def test_location_id_to_state(self):
        """Test location ID to state."""
        result = convert_location_name_format("denver", "state")
        assert result == "Colorado"

    def test_location_id_to_abbreviation(self):
        """Test location ID to abbreviation (state abbreviation)."""
        result = convert_location_name_format("denver", "abbreviation")
        assert result == "CO"

    def test_location_id_to_iso(self):
        """Test location ID to ISO (parent state ISO)."""
        result = convert_location_name_format("denver", "ISO")
        assert result == "US-CO"

    def test_location_id_to_metrocast_location_id(self):
        """Test location ID to metrocast_location_id."""
        result = convert_location_name_format("denver", "metrocast_location_id")
        assert result == "denver"

    def test_location_id_to_original_location_code(self):
        """Test location ID to original_location_code (HSA NCI ID)."""
        result = convert_location_name_format("denver", "original_location_code")
        assert result == "688"

    def test_location_id_to_fips_raises_error(self):
        """Test location ID to FIPS raises ValueError (no FIPS for metrocast)."""
        with pytest.raises(ValueError, match="Unknown output format 'FIPS' for metrocast"):
            convert_location_name_format("denver", "FIPS")

    def test_location_id_to_unknown_format_raises_error(self):
        """Test location ID to unknown format raises ValueError."""
        with pytest.raises(ValueError, match="Unknown output format"):
            convert_location_name_format("denver", "unknown_format")

    # From epydemix_population format (prefixed)
    def test_prefixed_to_epydemix_population(self):
        """Test prefixed to epydemix_population (identity)."""
        result = convert_location_name_format("metrocast_location_denver", "epydemix_population")
        assert result == "metrocast_location_denver"

    def test_prefixed_to_name(self):
        """Test prefixed to name."""
        result = convert_location_name_format("metrocast_location_denver", "name")
        assert result == "Denver, CO"

    def test_prefixed_to_hsa_counties(self):
        """Test prefixed to hsa_counties."""
        result = convert_location_name_format("metrocast_location_denver", "hsa_counties")
        assert "Denver" in result

    def test_prefixed_to_state(self):
        """Test prefixed to state."""
        result = convert_location_name_format("metrocast_location_denver", "state")
        assert result == "Colorado"

    def test_prefixed_to_abbreviation(self):
        """Test prefixed to abbreviation."""
        result = convert_location_name_format("metrocast_location_denver", "abbreviation")
        assert result == "CO"

    def test_prefixed_to_iso(self):
        """Test prefixed to ISO."""
        result = convert_location_name_format("metrocast_location_denver", "ISO")
        assert result == "US-CO"

    def test_prefixed_to_metrocast_location_id(self):
        """Test prefixed to metrocast_location_id."""
        result = convert_location_name_format("metrocast_location_denver", "metrocast_location_id")
        assert result == "denver"

    def test_prefixed_to_fips_raises_error(self):
        """Test prefixed to FIPS raises ValueError."""
        with pytest.raises(ValueError, match="Unknown output format 'FIPS' for metrocast"):
            convert_location_name_format("metrocast_location_denver", "FIPS")

    # NC flu regions (different location type in same system)
    def test_nc_region_to_epydemix_population(self):
        """Test NC flu region to epydemix_population."""
        result = convert_location_name_format("nenc", "epydemix_population")
        assert result == "metrocast_location_nenc"

    def test_nc_region_to_name(self):
        """Test NC flu region to name."""
        result = convert_location_name_format("nenc", "name")
        # NC regions have descriptive names
        assert result is not None

    def test_nc_region_to_state(self):
        """Test NC flu region to state."""
        result = convert_location_name_format("nenc", "state")
        assert result == "North Carolina"

    def test_nc_region_to_iso(self):
        """Test NC flu region to ISO."""
        result = convert_location_name_format("nenc", "ISO")
        assert result == "US-NC"

    # Location with hyphen in name
    def test_hyphenated_location_to_name(self):
        """Test location with hyphen to name."""
        result = convert_location_name_format("colorado-springs", "name")
        assert result == "Colorado Springs, CO"

    def test_hyphenated_prefixed_to_name(self):
        """Test prefixed location with hyphen to name."""
        result = convert_location_name_format("metrocast_location_colorado-springs", "name")
        assert result == "Colorado Springs, CO"

    # Explicit parameters
    def test_explicit_location_type_metrocast(self):
        """Test with explicit location_type."""
        result = convert_location_name_format("denver", "name", location_type="metrocast_location")
        assert result == "Denver, CO"

    def test_explicit_input_format_epydemix(self):
        """Test with explicit input_format for prefixed input."""
        result = convert_location_name_format(
            "metrocast_location_denver",
            "name",
            input_format="epydemix_population",
            location_type="metrocast_location",
        )
        assert result == "Denver, CO"


class TestConvertLocationNameFormatAutoDetection:
    """Tests for auto-detection behavior in convert_location_name_format."""

    def test_auto_detects_metrocast_by_prefix(self):
        """Test auto-detection of metrocast by prefix."""
        # Prefixed names should be detected as metrocast
        result = convert_location_name_format("metrocast_location_denver", "state")
        assert result == "Colorado"

    def test_auto_detects_metrocast_by_location_list(self):
        """Test auto-detection of metrocast by checking location list."""
        # Non-prefixed metrocast names should be detected by looking up in location list
        result = convert_location_name_format("denver", "state")
        assert result == "Colorado"

    def test_auto_detects_iso_for_non_metrocast(self):
        """Test auto-detection falls back to ISO for non-metrocast names."""
        result = convert_location_name_format("US-MA", "name")
        assert result == "Massachusetts"

    def test_iso_code_not_confused_with_metrocast(self):
        """Test that ISO codes are not confused with metrocast locations."""
        # "CO" is both Colorado abbreviation and could be confused
        result = convert_location_name_format("US-CO", "name")
        assert result == "Colorado"


class TestGetTotalPopulation:
    """Tests for get_total_population function."""

    def test_iso_location(self):
        """Test getting population for ISO location."""
        result = get_total_population("United_States_Massachusetts")
        assert result > 0
        assert isinstance(result, int)

    def test_metrocast_location(self):
        """Test getting population for metrocast location."""
        result = get_total_population("metrocast_location_denver")
        assert result > 0
        assert isinstance(result, int)

    def test_metrocast_population_value(self):
        """Test that Denver population is approximately correct."""
        result = get_total_population("metrocast_location_denver")
        # Denver HSA population should be around 2.9 million
        assert 2_000_000 < result < 4_000_000

    def test_invalid_metrocast_raises_error(self):
        """Test that invalid metrocast location raises ValueError."""
        with pytest.raises(ValueError, match="not found in metrocast locations"):
            get_total_population("metrocast_location_invalid_location")
