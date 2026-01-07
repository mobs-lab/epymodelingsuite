"""Tests for visualization generators module."""

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from epymodelingsuite.schema.dispatcher import CalibrationOutput
from epymodelingsuite.schema.output import ObservedValuesConfig, PlotsConfig, QuantilesPlotConfig
from epymodelingsuite.visualization.generators import (
    _prepare_surveillance_for_location,
    _rename_value_column,
    generate_single_quantile_plots,
)


class TestSurveillanceDataFiltering:
    """Tests for surveillance data filtering in quantile plots."""

    @pytest.fixture
    def surveillance_csv_data(self, tmp_path):
        """Create a temporary surveillance CSV file."""
        csv_file = tmp_path / "surveillance.csv"
        # Create surveillance data spanning from 2023-12-01 to 2024-02-15
        dates = pd.date_range("2023-12-01", "2024-02-15", freq="D")
        df = pd.DataFrame(
            {
                "date": dates,
                "location": ["US-CA"] * len(dates),
                "hospitalizations": range(100, 100 + len(dates)),
            }
        )
        df.to_csv(csv_file, index=False)
        return csv_file

    @pytest.fixture
    def calibration_quantiles(self):
        """Create calibration quantiles starting from 2024-01-15."""
        dates = pd.date_range("2024-01-15", "2024-01-31", freq="D")
        quantiles = [0.025, 0.5, 0.975]
        data = []
        for d in dates:
            for q in quantiles:
                data.append({"date": d, "quantile": q, "data": 100 + q * 20})
        return pd.DataFrame(data)

    @pytest.fixture
    def projection_quantiles(self):
        """Create projection quantiles starting from 2024-01-01 (full timespan)."""
        dates = pd.date_range("2024-01-01", "2024-02-29", freq="D")
        quantiles = [0.025, 0.5, 0.975]
        data = []
        for d in dates:
            for q in quantiles:
                data.append({"date": d, "quantile": q, "hospitalizations": 100 + q * 20})
        return pd.DataFrame(data)

    @pytest.fixture
    def mock_calibration_output(self, calibration_quantiles, projection_quantiles):
        """Create a mock CalibrationOutput object."""
        calibration = MagicMock(spec=CalibrationOutput)
        calibration.population = "US-CA"

        # Mock the results object with get_calibration_quantiles method
        calibration.results = MagicMock()
        calibration.results.get_calibration_quantiles.return_value = calibration_quantiles

        # Mock projection quantiles (this has the full timespan)
        calibration.results.get_projection_quantiles.return_value = projection_quantiles

        return calibration

    @pytest.fixture
    def plots_config_with_surveillance(self, surveillance_csv_data):
        """Create a PlotsConfig with surveillance data enabled."""
        surveillance_config = QuantilesSurveillanceConfig(
            show=True,
            data_path=str(surveillance_csv_data),
            value_column="hospitalizations",
            date_column="date",
            location_column="location",
        )

        quantiles_config = QuantilesPlotConfig(
            single=True,  # Enable single plots
            surveillance=surveillance_config,
        )

        return PlotsConfig(
            reference_date=date(2024, 1, 15),
            quantiles=quantiles_config,
        )

    def test_surveillance_filtered_to_timespan_start(self, mock_calibration_output, plots_config_with_surveillance):
        """Test that surveillance data is filtered to start from projection timespan start in filtered plot."""
        out_dict = {}

        # Mock plot_calibration_projection to capture the surveillance data passed to it
        with patch("epymodelingsuite.visualization.generators.plot_calibration_projection") as mock_plot:
            mock_plot.return_value = (MagicMock(), MagicMock())

            generate_single_quantile_plots(
                calibrations=[mock_calibration_output],
                plots_config=plots_config_with_surveillance,
                out_dict=out_dict,
            )

            # Verify plot_calibration_projection was called twice (filtered and full)
            assert mock_plot.call_count == 2

            # Extract the surveillance data passed to both plot calls
            first_call_kwargs = mock_plot.call_args_list[0].kwargs
            second_call_kwargs = mock_plot.call_args_list[1].kwargs

            df_surv_filtered = first_call_kwargs["df_surveillance"]
            df_surv_full = second_call_kwargs["df_surveillance"]

            # Verify filtered surveillance data was provided and filtered correctly
            assert df_surv_filtered is not None
            assert not df_surv_filtered.empty

            # Filtered version: Projection quantiles start from 2024-01-01 (which represents the full timespan)
            # Surveillance data before this date should be filtered out
            min_date_filtered = pd.to_datetime(df_surv_filtered["date"]).min()
            assert min_date_filtered.date() >= date(2024, 1, 1)

            # Filtered version: Verify no upper bound filtering (all dates after start should be included)
            # Original surveillance data goes until 2024-02-15
            max_date_filtered = pd.to_datetime(df_surv_filtered["date"]).max()
            assert max_date_filtered.date() == date(2024, 2, 15)

            # Verify full surveillance data shows all data (no filtering)
            assert df_surv_full is not None
            assert not df_surv_full.empty
            min_date_full = pd.to_datetime(df_surv_full["date"]).min()
            max_date_full = pd.to_datetime(df_surv_full["date"]).max()
            # Full surveillance should start from the beginning of the CSV data
            assert min_date_full.date() == date(2023, 12, 1)
            assert max_date_full.date() == date(2024, 2, 15)

    def test_surveillance_filtering_when_no_calibration_quantiles(
        self, surveillance_csv_data, plots_config_with_surveillance
    ):
        """Test that surveillance is filtered by surveillance_points even when calibration quantiles are not available."""
        # Create calibration output with no calibration quantiles
        calibration = MagicMock(spec=CalibrationOutput)
        calibration.population = "US-CA"
        calibration.results = MagicMock()
        calibration.results.get_calibration_quantiles.return_value = None
        calibration.results.get_projection_quantiles.return_value = None

        out_dict = {}

        with patch("epymodelingsuite.visualization.generators.plot_calibration_projection") as mock_plot:
            mock_plot.return_value = (MagicMock(), MagicMock())

            generate_single_quantile_plots(
                calibrations=[calibration],
                plots_config=plots_config_with_surveillance,
                out_dict=out_dict,
            )

            # Verify plot was called twice (filtered and full)
            assert mock_plot.call_count == 2

            # Extract surveillance data from both calls
            first_call_kwargs = mock_plot.call_args_list[0].kwargs
            second_call_kwargs = mock_plot.call_args_list[1].kwargs

            df_surv_filtered = first_call_kwargs["df_surveillance"]
            df_surv_full = second_call_kwargs["df_surveillance"]

            # When no calibration/projection quantiles, filtered surveillance should still be loaded
            # and filtered to the 8 most recent points (default surveillance_points value)
            assert df_surv_filtered is not None
            # Should have exactly 8 points (the most recent ones)
            assert len(df_surv_filtered) == 8
            # Check that we have the most recent dates
            max_date_filtered = pd.to_datetime(df_surv_filtered["date"]).max()
            assert max_date_filtered.date() == date(2024, 2, 15)

            # Full surveillance should show all data (no filtering)
            assert df_surv_full is not None
            assert len(df_surv_full) == 77  # All dates from 2023-12-01 to 2024-02-15

    def test_surveillance_filtering_falls_back_to_calibration_quantiles(
        self, calibration_quantiles, surveillance_csv_data, plots_config_with_surveillance
    ):
        """Test that surveillance falls back to calibration quantiles when no projection quantiles available."""
        # Create calibration output with only calibration quantiles (no projection)
        calibration = MagicMock(spec=CalibrationOutput)
        calibration.population = "US-CA"
        calibration.results = MagicMock()
        calibration.results.get_calibration_quantiles.return_value = calibration_quantiles
        calibration.results.get_projection_quantiles.return_value = None

        out_dict = {}

        with patch("epymodelingsuite.visualization.generators.plot_calibration_projection") as mock_plot:
            mock_plot.return_value = (MagicMock(), MagicMock())

            generate_single_quantile_plots(
                calibrations=[calibration],
                plots_config=plots_config_with_surveillance,
                out_dict=out_dict,
            )

            # Verify plot was called twice (filtered and full)
            assert mock_plot.call_count == 2

            # Extract surveillance data from both calls
            first_call_kwargs = mock_plot.call_args_list[0].kwargs
            second_call_kwargs = mock_plot.call_args_list[1].kwargs

            df_surv_filtered = first_call_kwargs["df_surveillance"]
            df_surv_full = second_call_kwargs["df_surveillance"]

            # Filtered surveillance should be filtered based on calibration quantiles
            assert df_surv_filtered is not None
            # Calibration quantiles start from 2024-01-15
            min_date_filtered = pd.to_datetime(df_surv_filtered["date"]).min()
            assert min_date_filtered.date() >= date(2024, 1, 15)

            # Full surveillance should show all data (no filtering)
            assert df_surv_full is not None
            min_date_full = pd.to_datetime(df_surv_full["date"]).min()
            assert min_date_full.date() == date(2023, 12, 1)

    def test_surveillance_filtering_with_empty_location_data(
        self, mock_calibration_output, plots_config_with_surveillance
    ):
        """Test behavior when surveillance data has no matching location."""
        # Create calibration for a different location
        mock_calibration_output.population = "US-TX"

        out_dict = {}

        with patch("epymodelingsuite.visualization.generators.plot_calibration_projection") as mock_plot:
            mock_plot.return_value = (MagicMock(), MagicMock())

            generate_single_quantile_plots(
                calibrations=[mock_calibration_output],
                plots_config=plots_config_with_surveillance,
                out_dict=out_dict,
            )

            # Verify plot was called twice (filtered and full)
            assert mock_plot.call_count == 2

            # Extract surveillance data from both calls
            first_call_kwargs = mock_plot.call_args_list[0].kwargs
            second_call_kwargs = mock_plot.call_args_list[1].kwargs

            df_surv_filtered = first_call_kwargs["df_surveillance"]
            df_surv_full = second_call_kwargs["df_surveillance"]

            # Surveillance should be None when no matching location (for both plots)
            assert df_surv_filtered is None
            assert df_surv_full is None


class TestPrepareSurveillanceForLocation:
    """Test _prepare_surveillance_for_location helper function."""

    def test_handles_duplicate_value_column(self):
        """Test that preparing surveillance data avoids duplicate 'value' columns."""
        # Create surveillance data with both 'value' and 'original_value' columns
        # This simulates real-world data where both columns exist
        surveillance_df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=5, freq="D"),
                "location_iso": ["US-AL"] * 5,
                "value": [100.0, 200.0, 300.0, 400.0, 500.0],  # Existing 'value' column
                "original_value": [0.01, 0.02, 0.03, 0.04, 0.05],  # Column we want to rename to 'value'
            }
        )

        # Configure to use 'original_value' as the value column
        config = ObservedValuesConfig(
            data_path="dummy.csv",
            date_column="date",
            value_column="original_value",
            location_column="location_iso",
            location_format="ISO",
        )

        # Call the function
        df_surv_full, df_surv_filtered, _ = _prepare_surveillance_for_location(
            surveillance=surveillance_df,
            location="US-AL",
            proj_quant=None,
            cal_quant=None,
            surveillance_config=config,
        )

        # Verify df_surv_full has exactly 2 columns: date and value
        assert df_surv_full is not None
        assert len(df_surv_full.columns) == 2
        assert list(df_surv_full.columns) == ["date", "value"]

        # Verify df['value'] returns a Series, not a DataFrame
        assert isinstance(df_surv_full["value"], pd.Series)

        # Verify the values are from 'original_value', not the pre-existing 'value' column
        assert df_surv_full["value"].iloc[0] == 0.01
        assert df_surv_full["value"].iloc[4] == 0.05

        # Same checks for filtered surveillance
        assert df_surv_filtered is not None
        assert len(df_surv_filtered.columns) == 2
        assert list(df_surv_filtered.columns) == ["date", "value"]
        assert isinstance(df_surv_filtered["value"], pd.Series)


class TestRenameValueColumn:
    """Test _rename_value_column helper function."""

    def test_rename_existing_column(self):
        """Test renaming when column exists."""
        df = pd.DataFrame({"date": ["2024-01-01"], "ed_signal": [100], "quantile": [0.5]})
        result = _rename_value_column(df, "ed_signal")
        assert "value" in result.columns
        assert "ed_signal" not in result.columns
        assert result["value"].iloc[0] == 100

    def test_handles_duplicate_value_column(self):
        """Test that renaming avoids duplicate 'value' columns when source has pre-existing 'value'."""
        # Simulates projection quantiles with 269 columns including both 'value' and 'ed_signal'
        df = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02"],
                "quantile": [0.5, 0.5],
                "population": ["US-AL", "US-AL"],
                "value": [999.0, 999.0],  # Pre-existing 'value' column from epydemix
                "ed_signal": [100.0, 200.0],  # Column we want to rename to 'value'
                "other_column": [1, 2],  # Extra column that should be filtered out
            }
        )

        result = _rename_value_column(df, "ed_signal")

        # Should have exactly 4 columns: date, quantile, population, value
        assert len(result.columns) == 4
        assert list(result.columns) == ["date", "quantile", "population", "value"]

        # result['value'] should be a Series, not a DataFrame
        assert isinstance(result["value"], pd.Series)

        # Values should be from 'ed_signal', not the pre-existing 'value' column
        assert result["value"].iloc[0] == 100.0
        assert result["value"].iloc[1] == 200.0

    def test_rename_none_input(self):
        """Test with None input."""
        result = _rename_value_column(None, "hospitalizations")
        assert result is None

    def test_rename_missing_column_raises_error(self):
        """Test error when column doesn't exist."""
        df = pd.DataFrame({"date": ["2024-01-01"], "ed_signal": [100], "quantile": [0.5]})
        with pytest.raises(ValueError, match="Column 'hospitalizations' not found"):
            _rename_value_column(df, "hospitalizations")

    def test_error_message_suggests_available_columns(self):
        """Test error message lists available columns."""
        df = pd.DataFrame({"date": ["2024-01-01"], "ed_signal": [100], "other_transition": [200], "quantile": [0.5]})
        with pytest.raises(ValueError, match=r"Available value columns:.*ed_signal.*other_transition"):
            _rename_value_column(df, "hospitalizations")
