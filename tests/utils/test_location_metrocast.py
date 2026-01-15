"""Tests for metrocast location utilities in utils/location.py."""

import pandas as pd
import pytest

from epymodelingsuite.utils.location import (
    get_metrocast_locations,
    get_metrocast_population_data,
    get_parent_region,
    is_state_level_metrocast_location,
    validate_iso3166,
    validate_location_by_type,
    validate_metrocast_location,
)


class TestGetMetrocastLocations:
    """Tests for get_metrocast_locations function."""

    def test_returns_dataframe(self):
        """Test that function returns a DataFrame."""
        result = get_metrocast_locations()
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self):
        """Test that DataFrame has all required columns."""
        result = get_metrocast_locations()
        required_columns = [
            "metrocast_location_id",
            "state",
            "state_abb",
            "location_name",
            "location_name_short",
            "population",
            "location_type",
            "state_iso",
        ]
        for col in required_columns:
            assert col in result.columns, f"Missing column: {col}"

    def test_contains_known_locations(self):
        """Test that DataFrame contains expected metrocast locations."""
        result = get_metrocast_locations()
        known_locations = ["denver", "mesa", "boston", "nenc", "houston"]
        for loc in known_locations:
            assert loc in result["metrocast_location_id"].values, f"Missing location: {loc}"


class TestGetMetrocastPopulationData:
    """Tests for get_metrocast_population_data function."""

    def test_returns_dataframe(self):
        """Test that function returns a DataFrame."""
        result = get_metrocast_population_data()
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self):
        """Test that DataFrame has required columns."""
        result = get_metrocast_population_data()
        required_columns = ["metrocast_location_id", "age", "population"]
        for col in required_columns:
            assert col in result.columns, f"Missing column: {col}"

    def test_contains_known_locations(self):
        """Test that DataFrame contains expected locations."""
        result = get_metrocast_population_data()
        assert "denver" in result["metrocast_location_id"].values
        assert "boston" in result["metrocast_location_id"].values

    def test_age_groups_present(self):
        """Test that multiple age groups are present for each location."""
        result = get_metrocast_population_data()
        denver_data = result[result["metrocast_location_id"] == "denver"]
        assert len(denver_data) > 1  # Should have multiple age groups


class TestGetParentRegion:
    """Tests for get_parent_region function."""

    def test_metrocast_location_returns_state_iso(self):
        """Test that metrocast locations return their parent state in ISO format."""
        result = get_parent_region("denver")
        assert result == "US-CO"

    def test_metrocast_location_returns_state_abbreviation(self):
        """Test getting parent state as abbreviation."""
        result = get_parent_region("denver", output_format="abbreviation")
        assert result == "CO"

    def test_metrocast_location_returns_state_name(self):
        """Test getting parent state as full name."""
        result = get_parent_region("denver", output_format="name")
        assert result == "Colorado"

    def test_metrocast_location_returns_country_level(self):
        """Test getting country level for metrocast location."""
        result = get_parent_region("denver", granularity="country")
        assert result == "US"

    def test_nc_flu_region_returns_state_iso(self):
        """Test NC flu region returns NC state ISO."""
        result = get_parent_region("nenc")
        assert result == "US-NC"

    def test_prefixed_population_name(self):
        """Test that prefixed population names work."""
        result = get_parent_region("metrocast_location_denver")
        assert result == "US-CO"

    def test_invalid_location_raises_error(self):
        """Test that invalid location raises ValueError."""
        with pytest.raises(ValueError, match="Unknown location"):
            get_parent_region("invalid_location")

    def test_iso_state_returns_country(self):
        """Test that ISO 3166-2 state returns country."""
        result = get_parent_region("US-MA", granularity="country")
        assert result == "US"

    def test_iso_state_returns_itself(self):
        """Test that ISO 3166-2 state returns itself for state granularity."""
        result = get_parent_region("US-MA", granularity="state")
        assert result == "US-MA"

    def test_iso_state_returns_abbreviation(self):
        """Test that ISO 3166-2 state can return abbreviation."""
        result = get_parent_region("US-MA", output_format="abbreviation")
        assert result == "MA"


class TestValidateMetrocastLocation:
    """Tests for validate_metrocast_location function."""

    def test_valid_location_returns_name(self):
        """Test that valid location name is returned."""
        result = validate_metrocast_location("denver")
        assert result == "denver"

    def test_valid_nc_region_returns_name(self):
        """Test that valid NC flu region is returned."""
        result = validate_metrocast_location("nenc")
        assert result == "nenc"

    def test_invalid_location_raises_error(self):
        """Test that invalid location raises ValueError."""
        with pytest.raises(ValueError, match="Invalid metrocast location"):
            validate_metrocast_location("invalid_location")


class TestValidateLocationByType:
    """Tests for validate_location_by_type function."""

    def test_iso_location_type(self):
        """Test validation of ISO location type."""
        result = validate_location_by_type("US-MA", "iso")
        assert result == "US-MA"

    def test_metrocast_location_type(self):
        """Test validation of metrocast location type."""
        result = validate_location_by_type("denver", "metrocast_location")
        assert result == "denver"

    def test_invalid_iso_raises_error(self):
        """Test that invalid ISO location raises error."""
        # ISO validation expects ISO 3166-2 format (e.g., US-XX)
        with pytest.raises(ValueError):
            validate_location_by_type("US-ZZ", "iso")  # Invalid state code

    def test_invalid_metrocast_raises_error(self):
        """Test that invalid metrocast location raises ValueError."""
        with pytest.raises(ValueError):
            validate_location_by_type("invalid_metro", "metrocast_location")

    def test_unknown_type_raises_error(self):
        """Test that unknown location type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown location type"):
            validate_location_by_type("some_location", "unknown_type")


class TestValidateISO3166:
    """Tests for validate_iso3166 function."""

    def test_valid_us_state_returns_code(self):
        """Test that valid US state code is returned."""
        result = validate_iso3166("US-MA")
        assert result == "US-MA"

    def test_valid_country_code_returns_code(self):
        """Test that valid country code is returned."""
        result = validate_iso3166("US")
        assert result == "US"

    def test_valid_non_us_subdivision_returns_code(self):
        """Test that valid non-US subdivision is returned."""
        result = validate_iso3166("CA-ON")  # Canada - Ontario
        assert result == "CA-ON"

    def test_location_without_hyphen_raises_valueerror(self):
        """Test that location without hyphen raises ValueError, not IndexError.

        This is a regression test for the fix where metrocast locations like
        'denver' were causing IndexError when validate_iso3166 tried to split
        by hyphen and access index 1.
        """
        with pytest.raises(ValueError, match="Invalid ISO 3166 code"):
            validate_iso3166("denver")

    def test_single_word_location_raises_valueerror(self):
        """Test that single-word non-ISO locations raise ValueError."""
        with pytest.raises(ValueError, match="Invalid ISO 3166 code"):
            validate_iso3166("boston")

    def test_invalid_us_state_raises_valueerror(self):
        """Test that invalid US state code raises ValueError."""
        with pytest.raises(ValueError, match="Invalid ISO 3166 code"):
            validate_iso3166("US-ZZ")  # ZZ is not a valid US state

    def test_invalid_country_code_raises_valueerror(self):
        """Test that invalid country code raises ValueError."""
        with pytest.raises(ValueError, match="Invalid ISO 3166 code"):
            validate_iso3166("XX")  # XX is not a valid country


class TestIsStateLevelMetrocastLocation:
    """Tests for is_state_level_metrocast_location function."""

    def test_state_level_location_returns_true(self):
        """Test that state-level locations return True."""
        # These locations have original_location_code == "All" in metrocast_locations.csv
        assert is_state_level_metrocast_location("colorado") is True
        assert is_state_level_metrocast_location("georgia") is True
        assert is_state_level_metrocast_location("massachusetts") is True
        assert is_state_level_metrocast_location("north-carolina") is True

    def test_sub_state_location_returns_false(self):
        """Test that sub-state locations return False."""
        # These are HSA or NC flu region locations with specific codes
        assert is_state_level_metrocast_location("denver") is False
        assert is_state_level_metrocast_location("boston") is False
        assert is_state_level_metrocast_location("nenc") is False
        assert is_state_level_metrocast_location("houston") is False

    def test_unknown_location_returns_false(self):
        """Test that unknown locations return False."""
        assert is_state_level_metrocast_location("nonexistent") is False
        assert is_state_level_metrocast_location("invalid_location") is False
