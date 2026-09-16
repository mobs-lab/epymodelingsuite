"""Tests for data validation module."""

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from epymodelingsuite.schema.data_validation import (
    validate_calibration_data,
    validate_calibration_data_for_config_set,
)


class TestValidateCalibrationData:
    """Tests for validate_calibration_data function."""

    @pytest.fixture
    def mock_calibration_config(self):
        """Create a mock calibration config."""
        config = MagicMock()
        config.modelset.calibration.fitting_window.start_date = date(2024, 1, 1)
        config.modelset.calibration.fitting_window.end_date = date(2024, 3, 1)
        config.modelset.calibration.fitting_window.epiweek_start_date = None
        config.modelset.calibration.fitting_window.epiweek_end_date = None
        config.modelset.calibration.comparison = [MagicMock()]
        config.modelset.calibration.comparison[0].observed_location_column = "location"
        config.modelset.calibration.comparison[0].observed_location_format = "ISO"
        config.modelset.calibration.comparison[0].observed_date_column = "date"
        config.modelset.calibration.observed_data_path = "/path/to/data.csv"
        return config

    def test_all_locations_valid(self, mock_calibration_config):
        """Test when all locations have data."""
        observed_data = pd.DataFrame(
            {
                "location": ["US-CA", "US-CA", "US-NY", "US-NY"],
                "value": [100, 110, 200, 210],
                "date": ["2024-01-01", "2024-01-08", "2024-01-01", "2024-01-08"],
            }
        )

        with patch("epymodelingsuite.schema.data_validation.get_data_in_location") as mock_get_data:
            # Return non-empty data for both locations
            mock_get_data.side_effect = [
                observed_data[observed_data["location"] == "US-CA"],
                observed_data[observed_data["location"] == "US-NY"],
            ]

            valid, invalid = validate_calibration_data(
                calibration_config=mock_calibration_config,
                population_names=["US-CA", "US-NY"],
                observed_data=observed_data,
            )

        assert valid == ["US-CA", "US-NY"]
        assert invalid == []

    def test_some_locations_invalid(self, mock_calibration_config):
        """Test when some locations have no data."""
        observed_data = pd.DataFrame(
            {
                "location": ["US-CA", "US-CA"],
                "value": [100, 110],
                "date": ["2024-01-01", "2024-01-08"],
            }
        )

        with patch("epymodelingsuite.schema.data_validation.get_data_in_location") as mock_get_data:
            # Return data for US-CA, empty for US-NY
            mock_get_data.side_effect = [
                observed_data,  # US-CA has data
                pd.DataFrame(),  # US-NY has no data
            ]

            valid, invalid = validate_calibration_data(
                calibration_config=mock_calibration_config,
                population_names=["US-CA", "US-NY"],
                observed_data=observed_data,
            )

        assert valid == ["US-CA"]
        assert invalid == ["US-NY"]

    def test_all_locations_invalid(self, mock_calibration_config):
        """Test when all locations have no data."""
        observed_data = pd.DataFrame()

        with patch("epymodelingsuite.schema.data_validation.get_data_in_location") as mock_get_data:
            # Return empty for all locations
            mock_get_data.return_value = pd.DataFrame()

            valid, invalid = validate_calibration_data(
                calibration_config=mock_calibration_config,
                population_names=["US-CA", "US-NY"],
                observed_data=observed_data,
            )

        assert valid == []
        assert invalid == ["US-CA", "US-NY"]

    def test_logs_error_for_invalid_location(self, mock_calibration_config, caplog):
        """Test that errors are logged for invalid locations."""
        import logging

        with patch("epymodelingsuite.schema.data_validation.get_data_in_location") as mock_get_data:
            mock_get_data.return_value = pd.DataFrame()

            with caplog.at_level(logging.ERROR):
                validate_calibration_data(
                    calibration_config=mock_calibration_config,
                    population_names=["US-CA"],
                    observed_data=pd.DataFrame(),
                )

        assert "DATA VALIDATION FAILED" in caplog.text
        assert "US-CA" in caplog.text

    def test_logs_debug_for_valid_location(self, mock_calibration_config, caplog):
        """Test that debug logs are created for valid locations."""
        import logging

        observed_data = pd.DataFrame(
            {
                "location": ["US-CA", "US-CA"],
                "value": [100, 110],
                "date": ["2024-01-01", "2024-03-01"],
            }
        )

        with patch("epymodelingsuite.schema.data_validation.get_data_in_location") as mock_get_data:
            mock_get_data.return_value = observed_data

            with caplog.at_level(logging.DEBUG):
                validate_calibration_data(
                    calibration_config=mock_calibration_config,
                    population_names=["US-CA"],
                    observed_data=observed_data,
                )

        assert "Data validation passed" in caplog.text
        assert "US-CA" in caplog.text

    def test_logs_warning_for_incomplete_coverage(self, mock_calibration_config, caplog):
        """Test that warning is logged when data doesn't cover full fitting window."""
        import logging

        # Data only covers part of the fitting window (Jan 15 - Feb 15, not Jan 1 - Mar 1)
        observed_data = pd.DataFrame(
            {
                "location": ["US-CA", "US-CA"],
                "value": [100, 110],
                "date": ["2024-01-15", "2024-02-15"],
            }
        )

        with patch("epymodelingsuite.schema.data_validation.get_data_in_location") as mock_get_data:
            mock_get_data.return_value = observed_data

            with caplog.at_level(logging.WARNING):
                valid, invalid = validate_calibration_data(
                    calibration_config=mock_calibration_config,
                    population_names=["US-CA"],
                    observed_data=observed_data,
                )

        # Location should still be valid (has data)
        assert valid == ["US-CA"]
        assert invalid == []
        # But warning should be logged about incomplete coverage
        assert "Incomplete data coverage" in caplog.text
        assert "US-CA" in caplog.text

    def test_no_warning_for_full_coverage(self, mock_calibration_config, caplog):
        """Test that no warning is logged when data covers full fitting window."""
        import logging

        # Data covers the full fitting window (Jan 1 - Mar 1)
        observed_data = pd.DataFrame(
            {
                "location": ["US-CA", "US-CA"],
                "value": [100, 110],
                "date": ["2024-01-01", "2024-03-01"],
            }
        )

        with patch("epymodelingsuite.schema.data_validation.get_data_in_location") as mock_get_data:
            mock_get_data.return_value = observed_data

            with caplog.at_level(logging.WARNING):
                valid, invalid = validate_calibration_data(
                    calibration_config=mock_calibration_config,
                    population_names=["US-CA"],
                    observed_data=observed_data,
                )

        assert valid == ["US-CA"]
        assert invalid == []
        assert "Incomplete data coverage" not in caplog.text

    def test_uses_epiweek_dates_when_start_date_is_none(self):
        """Test that epiweek dates are used when start_date is None."""
        config = MagicMock()
        config.modelset.calibration.fitting_window.start_date = None
        config.modelset.calibration.fitting_window.end_date = None
        config.modelset.calibration.fitting_window.epiweek_start_date = date(2024, 1, 7)
        config.modelset.calibration.fitting_window.epiweek_end_date = date(2024, 3, 2)
        config.modelset.calibration.comparison = [MagicMock()]
        config.modelset.calibration.comparison[0].observed_location_column = "location"
        config.modelset.calibration.comparison[0].observed_location_format = "ISO"
        config.modelset.calibration.comparison[0].observed_date_column = "date"
        config.modelset.calibration.observed_data_path = "/path/to/data.csv"

        with patch("epymodelingsuite.schema.data_validation.get_data_in_location") as mock_get_data:
            mock_get_data.return_value = pd.DataFrame()

            _, invalid = validate_calibration_data(
                calibration_config=config,
                population_names=["US-CA"],
                observed_data=pd.DataFrame(),
            )

        assert invalid == ["US-CA"]


class TestValidateCalibrationDataForConfigSet:
    """Tests for validate_calibration_data_for_config_set function."""

    def test_returns_tuple(self):
        """Test that function returns a tuple of (bool, str|None)."""
        # Non-existent file should return error
        success, error = validate_calibration_data_for_config_set(
            "/nonexistent/basemodel.yml",
            "/nonexistent/calibration.yml",
        )

        assert isinstance(success, bool)
        assert success is False
        assert error is not None
        assert isinstance(error, str)

    def test_file_not_found_error(self):
        """Test that FileNotFoundError is handled gracefully."""
        success, error = validate_calibration_data_for_config_set(
            "/nonexistent/basemodel.yml",
            "/nonexistent/calibration.yml",
        )

        assert success is False
        assert error is not None
        assert "not found" in error.lower() or "no such file" in error.lower()

    @patch("epymodelingsuite.config_loader.load_basemodel_config_from_file")
    @patch("epymodelingsuite.config_loader.load_calibration_config_from_file")
    @patch("epymodelingsuite.builders.utils.get_data_in_window")
    @patch("pandas.read_csv")
    def test_returns_true_when_all_valid(
        self,
        mock_read_csv,
        mock_get_window,
        mock_load_calib,
        mock_load_base,
    ):
        """Test that function returns True when all locations are valid."""
        # Setup mocks
        mock_load_base.return_value = MagicMock()
        mock_load_base.return_value.model.population.name = "US-CA"

        mock_load_calib.return_value = MagicMock()
        mock_load_calib.return_value.modelset.population_names = ["US-CA"]
        mock_load_calib.return_value.modelset.calibration.observed_data_path = "data.csv"
        mock_load_calib.return_value.modelset.calibration.fitting_window.start_date = date(2024, 1, 1)
        mock_load_calib.return_value.modelset.calibration.fitting_window.end_date = date(2024, 3, 1)
        mock_load_calib.return_value.modelset.calibration.comparison = [MagicMock()]
        mock_load_calib.return_value.modelset.calibration.comparison[0].observed_location_column = "location"
        mock_load_calib.return_value.modelset.calibration.comparison[0].observed_location_format = "ISO"
        mock_load_calib.return_value.modelset.calibration.comparison[0].observed_date_column = "date"

        mock_read_csv.return_value = pd.DataFrame({"col": [1, 2, 3], "location": ["US-CA", "US-CA", "US-CA"]})
        mock_get_window.return_value = pd.DataFrame({"col": [1, 2], "location": ["US-CA", "US-CA"]})

        with patch("epymodelingsuite.schema.data_validation.get_data_in_location") as mock_get_location:
            mock_get_location.return_value = pd.DataFrame({"col": [1, 2], "date": ["2024-01-01", "2024-03-01"]})

            success, error = validate_calibration_data_for_config_set(
                "basemodel.yml",
                "calibration.yml",
            )

        assert success is True
        assert error is None

    @patch("epymodelingsuite.config_loader.load_basemodel_config_from_file")
    @patch("epymodelingsuite.config_loader.load_calibration_config_from_file")
    @patch("epymodelingsuite.builders.utils.get_data_in_window")
    @patch("pandas.read_csv")
    def test_returns_false_when_invalid_locations(
        self,
        mock_read_csv,
        mock_get_window,
        mock_load_calib,
        mock_load_base,
    ):
        """Test that function returns False when some locations are invalid."""
        mock_load_base.return_value = MagicMock()
        mock_load_calib.return_value = MagicMock()
        mock_load_calib.return_value.modelset.population_names = ["US-CA", "US-NY"]
        mock_load_calib.return_value.modelset.calibration.observed_data_path = "data.csv"
        mock_load_calib.return_value.modelset.calibration.fitting_window.start_date = date(2024, 1, 1)
        mock_load_calib.return_value.modelset.calibration.fitting_window.end_date = date(2024, 3, 1)
        mock_load_calib.return_value.modelset.calibration.comparison = [MagicMock()]
        mock_load_calib.return_value.modelset.calibration.comparison[0].observed_location_column = "location"
        mock_load_calib.return_value.modelset.calibration.comparison[0].observed_location_format = "ISO"
        mock_load_calib.return_value.modelset.calibration.comparison[0].observed_date_column = "date"

        mock_read_csv.return_value = pd.DataFrame({"col": [1], "location": ["US-CA"]})
        mock_get_window.return_value = pd.DataFrame({"col": [1], "location": ["US-CA"]})

        with patch("epymodelingsuite.schema.data_validation.get_data_in_location") as mock_get_location:
            # US-CA has data, US-NY has no data
            mock_get_location.side_effect = [
                pd.DataFrame({"col": [1], "date": ["2024-01-01"]}),  # US-CA
                pd.DataFrame(),  # US-NY - empty
            ]

            success, error = validate_calibration_data_for_config_set(
                "basemodel.yml",
                "calibration.yml",
            )

        assert success is False
        assert error is not None
        assert "US-NY" in error

    @patch("epymodelingsuite.config_loader.load_basemodel_config_from_file")
    @patch("epymodelingsuite.config_loader.load_calibration_config_from_file")
    @patch("epymodelingsuite.builders.utils.get_data_in_window")
    @patch("pandas.read_csv")
    def test_returns_false_when_empty_window(
        self,
        mock_read_csv,
        mock_get_window,
        mock_load_calib,
        mock_load_base,
    ):
        """Test that function returns False when no data in fitting window."""
        mock_load_base.return_value = MagicMock()
        mock_load_calib.return_value = MagicMock()
        mock_load_calib.return_value.modelset.population_names = ["US-CA"]
        mock_load_calib.return_value.modelset.calibration.observed_data_path = "data.csv"
        mock_load_calib.return_value.modelset.calibration.fitting_window.start_date = date(2024, 1, 1)
        mock_load_calib.return_value.modelset.calibration.fitting_window.end_date = date(2024, 3, 1)

        mock_read_csv.return_value = pd.DataFrame({"col": [1]})
        mock_get_window.return_value = pd.DataFrame()  # Empty window

        success, error = validate_calibration_data_for_config_set(
            "basemodel.yml",
            "calibration.yml",
        )

        assert success is False
        assert error is not None
        assert "fitting window" in error.lower()

    @patch("epymodelingsuite.config_loader.load_basemodel_config_from_file")
    @patch("epymodelingsuite.config_loader.load_calibration_config_from_file")
    @patch("epymodelingsuite.utils.get_location_codebook")
    @patch("epymodelingsuite.builders.utils.get_data_in_window")
    @patch("pandas.read_csv")
    def test_handles_all_population(
        self,
        mock_read_csv,
        mock_get_window,
        mock_codebook,
        mock_load_calib,
        mock_load_base,
    ):
        """Test that 'all' population is expanded from codebook."""
        mock_load_base.return_value = MagicMock()
        mock_load_calib.return_value = MagicMock()
        mock_load_calib.return_value.modelset.population_names = ["all"]
        mock_load_calib.return_value.modelset.calibration.observed_data_path = "data.csv"
        mock_load_calib.return_value.modelset.calibration.fitting_window.start_date = date(2024, 1, 1)
        mock_load_calib.return_value.modelset.calibration.fitting_window.end_date = date(2024, 3, 1)
        mock_load_calib.return_value.modelset.calibration.comparison = [MagicMock()]
        mock_load_calib.return_value.modelset.calibration.comparison[0].observed_location_column = "location"
        mock_load_calib.return_value.modelset.calibration.comparison[0].observed_location_format = "ISO"
        mock_load_calib.return_value.modelset.calibration.comparison[0].observed_date_column = "date"

        mock_codebook.return_value = pd.DataFrame({"location_name_epydemix": ["California", "Texas", "New York"]})

        mock_read_csv.return_value = pd.DataFrame({"col": [1]})
        mock_get_window.return_value = pd.DataFrame({"col": [1]})

        with patch("epymodelingsuite.schema.data_validation.get_data_in_location") as mock_get_location:
            mock_get_location.return_value = pd.DataFrame({"col": [1], "date": ["2024-01-01"]})

            success, error = validate_calibration_data_for_config_set(
                "basemodel.yml",
                "calibration.yml",
            )

        assert success is True
        assert error is None
        # Verify codebook was called to expand 'all'
        mock_codebook.assert_called_once()
