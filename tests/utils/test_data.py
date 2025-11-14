"""Tests for data fetching utilities."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from epymodelingsuite.utils.data import fetch_hhs_hospitalizations


@pytest.fixture
def mock_socrata_data():
    """Mock Socrata API response data."""
    return [
        {
            "weekendingdate": "2024-10-19T00:00:00.000",
            "jurisdiction": "CA",
            "totalconfflunewadm": "150",
        },
        {
            "weekendingdate": "2024-10-26T00:00:00.000",
            "jurisdiction": "CA",
            "totalconfflunewadm": "175",
        },
        {
            "weekendingdate": "2024-10-19T00:00:00.000",
            "jurisdiction": "TX",
            "totalconfflunewadm": "200",
        },
        {
            "weekendingdate": "2024-10-26T00:00:00.000",
            "jurisdiction": "TX",
            "totalconfflunewadm": None,  # Test NaN handling
        },
        # Territory that should be filtered
        {
            "weekendingdate": "2024-10-19T00:00:00.000",
            "jurisdiction": "PR",
            "totalconfflunewadm": "100",
        },
        # HHS Region that should be filtered
        {
            "weekendingdate": "2024-10-19T00:00:00.000",
            "jurisdiction": "Region 1",
            "totalconfflunewadm": "500",
        },
        # USA national that should be included
        {
            "weekendingdate": "2024-10-19T00:00:00.000",
            "jurisdiction": "USA",
            "totalconfflunewadm": "1000",
        },
    ]


@pytest.fixture
def mock_locations_data():
    """Mock flusight locations data."""
    return pd.DataFrame(
        {
            "abbreviation": ["CA", "TX", "NY", "PR"],
            "location": ["06", "48", "36", "72"],
            "location_name": ["California", "Texas", "New York", "Puerto Rico"],
        }
    )


class TestFetchHHSHospitalizations:
    """Test suite for fetch_hhs_hospitalizations function."""

    @patch("epymodelingsuite.utils.data.Socrata")
    @patch("epymodelingsuite.utils.data.get_flusight_locations")
    def test_fetch_all_data(self, mock_get_locations, mock_socrata_class, mock_socrata_data, mock_locations_data):
        """Test fetching all data without filters."""
        # Setup mocks
        mock_client = MagicMock()
        mock_client.get.return_value = mock_socrata_data
        mock_socrata_class.return_value = mock_client
        mock_get_locations.return_value = mock_locations_data

        # Call function
        result = fetch_hhs_hospitalizations()

        # Verify Socrata client initialization
        mock_socrata_class.assert_called_once_with("data.cdc.gov", None)

        # Verify get was called without where clause
        mock_client.get.assert_called_once_with("mpgq-jmmr", limit=100000)

        # Verify result structure
        assert isinstance(result, pd.DataFrame)
        # Should be 5 rows: 2 CA + 2 TX + 1 USA (PR and Region 1 filtered out)
        assert len(result) == 5
        assert list(result.columns) == [
            "location_iso",
            "location_code",
            "target_end_date",
            "epiweek",
            "hospitalizations",
        ]

        # Verify included locations
        assert "US-CA" in result["location_iso"].values
        assert "US-TX" in result["location_iso"].values
        assert "US" in result["location_iso"].values  # USA national
        assert "06" in result["location_code"].values
        assert "48" in result["location_code"].values

        # Verify filtered locations are NOT present
        assert "US-PR" not in result["location_iso"].values  # Territory filtered
        assert "Region 1" not in result["jurisdiction"] if "jurisdiction" in result.columns else True

        # Verify USA national data
        usa_data = result[result["location_iso"] == "US"]
        assert len(usa_data) == 1
        assert usa_data["location_code"].iloc[0] == "US"
        assert usa_data["hospitalizations"].iloc[0] == 1000.0

        # Verify hospitalizations (NaN should be converted to 0)
        ca_data = result[result["location_iso"] == "US-CA"]
        assert ca_data["hospitalizations"].iloc[0] == 150.0
        tx_data = result[result["location_iso"] == "US-TX"]
        assert tx_data[tx_data["hospitalizations"] == 0.0].shape[0] == 1  # NaN converted to 0

    @patch("epymodelingsuite.utils.data.Socrata")
    @patch("epymodelingsuite.utils.data.get_flusight_locations")
    def test_fetch_with_date_filter(
        self, mock_get_locations, mock_socrata_class, mock_socrata_data, mock_locations_data
    ):
        """Test fetching data with date filter."""
        # Setup mocks
        mock_client = MagicMock()
        mock_client.get.return_value = mock_socrata_data
        mock_socrata_class.return_value = mock_client
        mock_get_locations.return_value = mock_locations_data

        # Call function with date filter
        result = fetch_hhs_hospitalizations(query_start_date="2024-10-01")

        # Verify get was called with where clause
        mock_client.get.assert_called_once_with(
            "mpgq-jmmr",
            where="weekendingdate >= '2024-10-01'",
            limit=100000,
        )

        # Verify result structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5  # 2 CA + 2 TX + 1 USA

    @patch("epymodelingsuite.utils.data.Socrata")
    @patch("epymodelingsuite.utils.data.get_flusight_locations")
    def test_epiweek_conversion(self, mock_get_locations, mock_socrata_class, mock_socrata_data, mock_locations_data):
        """Test that epiweeks are correctly calculated."""
        # Setup mocks
        mock_client = MagicMock()
        mock_client.get.return_value = mock_socrata_data
        mock_socrata_class.return_value = mock_client
        mock_get_locations.return_value = mock_locations_data

        # Call function
        result = fetch_hhs_hospitalizations()

        # Verify epiweek values are integers in CDC format (YYYYWW)
        assert result["epiweek"].dtype in [int, "int64"]
        # 2024-10-19 should be epiweek 202442
        oct_19_data = result[result["target_end_date"] == pd.Timestamp("2024-10-19")]
        assert all(oct_19_data["epiweek"] == 202442)

    @patch("epymodelingsuite.utils.data.Socrata")
    @patch("epymodelingsuite.utils.data.get_flusight_locations")
    def test_save_to_file(
        self,
        mock_get_locations,
        mock_socrata_class,
        mock_socrata_data,
        mock_locations_data,
        tmp_path,
    ):
        """Test saving data to file."""
        # Setup mocks
        mock_client = MagicMock()
        mock_client.get.return_value = mock_socrata_data
        mock_socrata_class.return_value = mock_client
        mock_get_locations.return_value = mock_locations_data

        # Create temporary file path
        save_path = tmp_path / "test_hhs_data.csv"

        # Call function with save_path
        result = fetch_hhs_hospitalizations(save_path=str(save_path))

        # Verify file was created
        assert save_path.exists()

        # Verify we can read the file and it matches the result
        # (Reading happens outside the patch context, so it uses real read_csv)
        saved_data = pd.read_csv(save_path, parse_dates=["target_end_date"])

        # Check that the saved file has the correct columns
        assert list(saved_data.columns) == list(result.columns)
        assert len(saved_data) == len(result)

    @patch("epymodelingsuite.utils.data.Socrata")
    @patch("epymodelingsuite.utils.data.get_flusight_locations")
    def test_location_mapping(self, mock_get_locations, mock_socrata_class, mock_socrata_data, mock_locations_data):
        """Test that location mapping works correctly."""
        # Setup mocks
        mock_client = MagicMock()
        mock_client.get.return_value = mock_socrata_data
        mock_socrata_class.return_value = mock_client
        mock_get_locations.return_value = mock_locations_data

        # Call function
        result = fetch_hhs_hospitalizations()

        # Verify get_flusight_locations was called
        mock_get_locations.assert_called_once()

        # Verify location mapping
        ca_row = result[result["location_iso"] == "US-CA"].iloc[0]
        assert ca_row["location_code"] == "06"

        tx_row = result[result["location_iso"] == "US-TX"].iloc[0]
        assert tx_row["location_code"] == "48"

    @patch("epymodelingsuite.utils.data.Socrata")
    @patch("epymodelingsuite.utils.data.get_flusight_locations")
    def test_territory_and_region_filtering(
        self, mock_get_locations, mock_socrata_class, mock_socrata_data, mock_locations_data
    ):
        """Test that territories and HHS regions are filtered out."""
        # Setup mocks
        mock_client = MagicMock()
        mock_client.get.return_value = mock_socrata_data
        mock_socrata_class.return_value = mock_client
        mock_get_locations.return_value = mock_locations_data

        # Call function
        result = fetch_hhs_hospitalizations()

        # Verify territories are filtered
        territories = ["US-AS", "US-GU", "US-MP", "US-PR", "US-VI"]
        for territory in territories:
            assert territory not in result["location_iso"].values, f"Territory {territory} should be filtered"

        # Verify HHS regions are filtered (they wouldn't have location_iso starting with US-)
        # All location_iso should be either "US" or "US-XX" format
        for loc in result["location_iso"]:
            assert loc == "US" or loc.startswith("US-"), f"Invalid location format: {loc}"
            assert "Region" not in loc, "HHS regions should be filtered"

        # Verify USA national is included
        assert "US" in result["location_iso"].values

        # Verify we have the expected locations (CA, TX, USA)
        unique_locations = set(result["location_iso"].unique())
        assert unique_locations == {"US-CA", "US-TX", "US"}

    @patch("epymodelingsuite.utils.data.Socrata")
    @patch("epymodelingsuite.utils.data.get_flusight_locations")
    def test_sorting(self, mock_get_locations, mock_socrata_class, mock_socrata_data, mock_locations_data):
        """Test that data is sorted by location_code then target_end_date."""
        # Setup mocks
        mock_client = MagicMock()
        mock_client.get.return_value = mock_socrata_data
        mock_socrata_class.return_value = mock_client
        mock_get_locations.return_value = mock_locations_data

        # Call function
        result = fetch_hhs_hospitalizations()

        # Verify sorting by location_code first
        location_codes = result["location_code"].tolist()
        # Should be sorted: 06 (CA), 06 (CA), 48 (TX), 48 (TX), US
        assert location_codes[0] == "06"  # CA first (FIPS 06)
        assert location_codes[1] == "06"  # CA second date
        assert location_codes[2] == "48"  # TX first (FIPS 48)
        assert location_codes[3] == "48"  # TX second date
        assert location_codes[4] == "US"  # US last (alphabetically after numbers)

        # Verify dates are sorted within each location
        ca_data = result[result["location_code"] == "06"]
        ca_dates = ca_data["target_end_date"].tolist()
        assert ca_dates[0] < ca_dates[1], "CA dates should be in ascending order"

        tx_data = result[result["location_code"] == "48"]
        tx_dates = tx_data["target_end_date"].tolist()
        assert tx_dates[0] < tx_dates[1], "TX dates should be in ascending order"
