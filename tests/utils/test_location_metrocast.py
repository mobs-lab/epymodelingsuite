"""Tests for metrocast location utilities in utils/location.py."""

import pandas as pd
import pytest

from epymodelingsuite.utils.location import (
    get_metrocast_locations,
    get_metrocast_population_data,
    get_parent_region,
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
            "location",
            "state",
            "state_abb",
            "location_name",
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
            assert loc in result["location"].values, f"Missing location: {loc}"


class TestGetMetrocastPopulationData:
    """Tests for get_metrocast_population_data function."""

    def test_returns_dataframe(self):
        """Test that function returns a DataFrame."""
        result = get_metrocast_population_data()
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self):
        """Test that DataFrame has required columns."""
        result = get_metrocast_population_data()
        required_columns = ["location_name", "age", "population"]
        for col in required_columns:
            assert col in result.columns, f"Missing column: {col}"

    def test_contains_known_locations(self):
        """Test that DataFrame contains expected locations."""
        result = get_metrocast_population_data()
        assert "denver" in result["location_name"].values
        assert "boston" in result["location_name"].values

    def test_age_groups_present(self):
        """Test that multiple age groups are present for each location."""
        result = get_metrocast_population_data()
        denver_data = result[result["location_name"] == "denver"]
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
        # Input without hyphen causes IndexError, input with invalid code raises ValueError
        with pytest.raises((ValueError, IndexError, AssertionError)):
            validate_location_by_type("US-ZZ", "iso")  # Invalid state code

    def test_invalid_metrocast_raises_error(self):
        """Test that invalid metrocast location raises ValueError."""
        with pytest.raises(ValueError):
            validate_location_by_type("invalid_metro", "metrocast_location")

    def test_unknown_type_raises_error(self):
        """Test that unknown location type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown location type"):
            validate_location_by_type("some_location", "unknown_type")
