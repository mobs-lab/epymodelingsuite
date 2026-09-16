"""Tests for builders.utils module."""

from datetime import date
from unittest.mock import Mock

import pandas as pd
import pytest

from epymodelingsuite.builders.utils import get_data_in_location, get_data_in_window


class TestGetDataInWindow:
    """Test the get_data_in_window function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with unsorted dates."""
        return pd.DataFrame(
            {
                "date": ["2024-01-05", "2024-01-01", "2024-01-10", "2024-01-03", "2024-01-08"],
                "value": [50, 10, 100, 30, 80],
                "location": ["US", "US", "US", "US", "US"],
            }
        )

    @pytest.fixture
    def calibration_config(self):
        """Create a mock calibration config."""
        config = Mock()
        config.fitting_window.start_date = date(2024, 1, 3)
        config.fitting_window.end_date = date(2024, 1, 8)
        config.comparison = [Mock()]
        config.comparison[0].observed_date_column = "date"
        return config

    def test_filters_by_date_window(self, sample_data, calibration_config):
        """Test that data is filtered to the correct date window."""
        result = get_data_in_window(sample_data, calibration_config)

        # Should include dates from 2024-01-03 to 2024-01-08
        expected_values = {30, 50, 80}  # Values for dates 03, 05, 08
        assert set(result["value"].values) == expected_values
        assert len(result) == 3

    def test_sorts_by_date_ascending_by_default(self, sample_data, calibration_config):
        """Test that data is sorted by date in ascending order (oldest to newest)."""
        result = get_data_in_window(sample_data, calibration_config)

        # Should be sorted: 2024-01-03, 2024-01-05, 2024-01-08
        expected_values = [30, 50, 80]
        assert list(result["value"].values) == expected_values

        # Verify dates are in ascending order
        dates = pd.to_datetime(result["date"]).dt.date.tolist()
        assert dates == sorted(dates)

    def test_sort_false_preserves_original_order(self, sample_data, calibration_config):
        """Test that sort=False preserves the original order from the filtered data."""
        result = get_data_in_window(sample_data, calibration_config, sort=False)

        # Should preserve original order: indices 0, 3, 4 from sample_data
        # Values: 50 (2024-01-05), 30 (2024-01-03), 80 (2024-01-08)
        expected_values = [50, 30, 80]
        assert list(result["value"].values) == expected_values

    def test_resets_index_when_sorting(self, sample_data, calibration_config):
        """Test that index is reset when sorting."""
        result = get_data_in_window(sample_data, calibration_config, sort=True)

        # Index should be reset to 0, 1, 2
        assert list(result.index) == [0, 1, 2]

    def test_empty_result_when_no_data_in_window(self, sample_data):
        """Test that empty DataFrame is returned when no data is in the window."""
        config = Mock()
        config.fitting_window.start_date = date(2025, 1, 1)
        config.fitting_window.end_date = date(2025, 1, 10)
        config.comparison = [Mock()]
        config.comparison[0].observed_date_column = "date"

        result = get_data_in_window(sample_data, config)
        assert len(result) == 0


class TestGetDataInLocation:
    """Test the get_data_in_location function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with multiple locations."""
        return pd.DataFrame(
            {
                "location": ["US-CA", "US-NY", "US-CA", "US-TX", "US-CA"],
                "value": [100, 200, 150, 300, 120],
            }
        )

    def test_filters_by_location_iso_format(self, sample_data):
        """Test filtering by location using ISO format (epydemix population name)."""
        result = get_data_in_location(sample_data, "United_States_California", "location", data_location_format="ISO")

        assert len(result) == 3
        assert all(result["location"] == "US-CA")
        assert list(result["value"].values) == [100, 150, 120]

    def test_empty_result_when_location_not_found(self, sample_data):
        """Test that empty DataFrame is returned when location is not found."""
        result = get_data_in_location(sample_data, "United_States_Florida", "location", data_location_format="ISO")

        assert len(result) == 0

    def test_filters_by_metrocast_location_id(self):
        """Test filtering by metrocast location using metrocast_location_id format."""
        data = pd.DataFrame(
            {
                "location": ["denver", "mesa", "denver", "colorado-springs"],
                "value": [100, 200, 150, 300],
            }
        )
        result = get_data_in_location(
            data, "metrocast_location_denver", "location", data_location_format="metrocast_location_id"
        )

        assert len(result) == 2
        assert all(result["location"] == "denver")
        assert list(result["value"].values) == [100, 150]

    def test_filters_by_metrocast_original_location_code(self):
        """Test filtering by metrocast location using original_location_code format."""
        data = pd.DataFrame(
            {
                "location": ["688", "711", "688", "754"],
                "value": [100, 200, 150, 300],
            }
        )
        result = get_data_in_location(
            data, "metrocast_location_denver", "location", data_location_format="original_location_code"
        )

        assert len(result) == 2
        assert all(result["location"] == "688")
        assert list(result["value"].values) == [100, 150]

    def test_filters_by_metrocast_name(self):
        """Test filtering by metrocast location using name format."""
        data = pd.DataFrame(
            {
                "location": ["Denver, CO", "Mesa, CO", "Denver, CO", "Colorado Springs, CO"],
                "value": [100, 200, 150, 300],
            }
        )
        result = get_data_in_location(data, "metrocast_location_denver", "location", data_location_format="name")

        assert len(result) == 2
        assert all(result["location"] == "Denver, CO")
        assert list(result["value"].values) == [100, 150]
