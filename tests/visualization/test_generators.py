"""Tests for visualization generators module."""

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from epymodelingsuite.schema.dispatcher import CalibrationOutput
from epymodelingsuite.schema.output import PlotsConfig, QuantilesPlotConfig, QuantilesSurveillanceConfig
from epymodelingsuite.visualization.generators import generate_single_quantile_plots


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
        """Test that surveillance data is filtered to start from projection timespan start."""
        out_dict = {}

        # Mock plot_calibration_projection to capture the surveillance data passed to it
        with patch("epymodelingsuite.visualization.generators.plot_calibration_projection") as mock_plot:
            mock_plot.return_value = (MagicMock(), MagicMock())

            generate_single_quantile_plots(
                calibrations=[mock_calibration_output],
                plots_config=plots_config_with_surveillance,
                out_dict=out_dict,
            )

            # Verify plot_calibration_projection was called
            assert mock_plot.called

            # Extract the surveillance data passed to the plot function
            call_kwargs = mock_plot.call_args.kwargs
            df_surv = call_kwargs["df_surveillance"]

            # Verify surveillance data was provided
            assert df_surv is not None
            assert not df_surv.empty

            # Verify dates are filtered correctly
            # Projection quantiles start from 2024-01-01 (which represents the full timespan)
            # Surveillance data before this date should be filtered out
            min_date = pd.to_datetime(df_surv["date"]).min()
            assert min_date.date() >= date(2024, 1, 1)

            # Verify no upper bound filtering (all dates after start should be included)
            # Original surveillance data goes until 2024-02-15
            max_date = pd.to_datetime(df_surv["date"]).max()
            assert max_date.date() == date(2024, 2, 15)

    def test_surveillance_filtering_when_no_calibration_quantiles(
        self, surveillance_csv_data, plots_config_with_surveillance
    ):
        """Test that surveillance is not date-filtered when calibration quantiles are not available."""
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

            # Verify plot was called
            assert mock_plot.called

            # Extract surveillance data
            call_kwargs = mock_plot.call_args.kwargs
            df_surv = call_kwargs["df_surveillance"]

            # When no calibration quantiles, surveillance should still be loaded
            # but not date-filtered
            assert df_surv is not None
            # Should include all original dates from 2023-12-01 to 2024-02-15
            min_date = pd.to_datetime(df_surv["date"]).min()
            max_date = pd.to_datetime(df_surv["date"]).max()
            assert min_date.date() == date(2023, 12, 1)
            assert max_date.date() == date(2024, 2, 15)

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

            # Verify plot was called
            assert mock_plot.called

            # Extract surveillance data
            call_kwargs = mock_plot.call_args.kwargs
            df_surv = call_kwargs["df_surveillance"]

            # Surveillance should be filtered based on calibration quantiles
            assert df_surv is not None
            # Calibration quantiles start from 2024-01-15
            min_date = pd.to_datetime(df_surv["date"]).min()
            assert min_date.date() >= date(2024, 1, 15)

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

            # Verify plot was called
            assert mock_plot.called

            # Extract surveillance data
            call_kwargs = mock_plot.call_args.kwargs
            df_surv = call_kwargs["df_surveillance"]

            # Surveillance should be None when no matching location
            assert df_surv is None
