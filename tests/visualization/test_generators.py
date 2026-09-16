"""Tests for visualization generators module."""

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from epymodelingsuite.schema.calibration import CalibrationStrategy
from epymodelingsuite.schema.dispatcher import CalibrationOutput
from epymodelingsuite.schema.output import ObservedValuesConfig, PlotsConfig, QuantilesPlotConfig
from epymodelingsuite.visualization.generators import (
    _check_incomplete_generations,
    _clip_surveillance,
    _clip_to_horizon,
    _clip_to_surveillance_start,
    _format_plot_notes,
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
    def surveillance_sources(self, surveillance_csv_data):
        """Create surveillance sources dict with ObservedValuesConfig."""
        return {
            "hosp": ObservedValuesConfig(
                data_path=str(surveillance_csv_data),
                value_column="hospitalizations",
                date_column="date",
                location_column="location",
                location_format="ISO",
            )
        }

    @pytest.fixture
    def plots_config_with_surveillance(self):
        """Create a PlotsConfig with surveillance data enabled."""
        quantiles_config = QuantilesPlotConfig(
            single=True,  # Enable single plots
        )

        return PlotsConfig(
            reference_date=date(2024, 1, 15),
            quantiles=quantiles_config,
        )

    def test_surveillance_filtered_to_timespan_start(
        self, mock_calibration_output, plots_config_with_surveillance, surveillance_sources
    ):
        """Test that surveillance data is filtered to start from projection timespan start in filtered plot."""
        out_dict = {}

        # Mock plot_calibration_projection to capture the surveillance data passed to it
        with patch("epymodelingsuite.visualization.generators.plot_calibration_projection") as mock_plot:
            mock_plot.return_value = (MagicMock(), MagicMock())

            generate_single_quantile_plots(
                calibrations=[mock_calibration_output],
                plots_config=plots_config_with_surveillance,
                out_dict=out_dict,
                surveillance_sources=surveillance_sources,
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
        self, plots_config_with_surveillance, surveillance_sources
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
                surveillance_sources=surveillance_sources,
            )

            # Verify plot was called twice (filtered and full)
            assert mock_plot.call_count == 2

            # Extract surveillance data from both calls
            first_call_kwargs = mock_plot.call_args_list[0].kwargs
            second_call_kwargs = mock_plot.call_args_list[1].kwargs

            df_surv_filtered = first_call_kwargs["df_surveillance"]
            df_surv_full = second_call_kwargs["df_surveillance"]

            # When no calibration/projection quantiles, filtered surveillance should still be loaded
            # and filtered to 8 points before reference_date (2024-01-15) plus all after
            assert df_surv_filtered is not None
            dates_filtered = pd.to_datetime(df_surv_filtered["date"]).dt.date
            before_ref = dates_filtered[dates_filtered <= date(2024, 1, 15)]
            after_ref = dates_filtered[dates_filtered > date(2024, 1, 15)]
            assert len(before_ref) == 8
            assert len(after_ref) > 0
            max_date_filtered = dates_filtered.max()
            assert max_date_filtered == date(2024, 2, 15)

            # Full surveillance should show all data (no filtering)
            assert df_surv_full is not None
            assert len(df_surv_full) == 77  # All dates from 2023-12-01 to 2024-02-15

    def test_surveillance_filtering_falls_back_to_calibration_quantiles(
        self, calibration_quantiles, plots_config_with_surveillance, surveillance_sources
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
                surveillance_sources=surveillance_sources,
            )

            # Verify plot was called twice (filtered and full)
            assert mock_plot.call_count == 2

            # Extract surveillance data from both calls
            first_call_kwargs = mock_plot.call_args_list[0].kwargs
            second_call_kwargs = mock_plot.call_args_list[1].kwargs

            df_surv_filtered = first_call_kwargs["df_surveillance"]
            df_surv_full = second_call_kwargs["df_surveillance"]

            # Filtered surveillance should be clipped to calibration quantile start (2024-01-15)
            # then further filtered by surveillance_points (8 before reference_date + all after)
            assert df_surv_filtered is not None
            dates_filtered = pd.to_datetime(df_surv_filtered["date"]).dt.date
            # Calibration quantiles start from 2024-01-15, so data is clipped there first
            assert dates_filtered.min() >= date(2024, 1, 8)

            # Full surveillance should show all data (no filtering)
            assert df_surv_full is not None
            min_date_full = pd.to_datetime(df_surv_full["date"]).min()
            assert min_date_full.date() == date(2023, 12, 1)

    def test_surveillance_filtering_with_empty_location_data(
        self, mock_calibration_output, plots_config_with_surveillance, surveillance_sources
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
                surveillance_sources=surveillance_sources,
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


class TestCheckIncompleteGenerations:
    """Test _check_incomplete_generations helper."""

    def test_incomplete_returns_note(self):
        """Test returns note when completed < requested."""
        strategy = CalibrationStrategy(name="SMC", options={"num_generations": 7})
        mock_results = MagicMock()
        mock_results.posterior_distributions = [MagicMock() for _ in range(5)]

        calibration = CalibrationOutput.model_construct(
            primary_id=0,
            seed=42,
            delta_t=1.0,
            population="US-CA",
            results=mock_results,
            calibration_strategy=strategy,
        )

        result = _check_incomplete_generations(calibration)
        assert result == "Completed 5 of 7 requested generations"

    def test_complete_returns_none(self):
        """Test returns None when all generations completed."""
        strategy = CalibrationStrategy(name="SMC", options={"num_generations": 7})
        mock_results = MagicMock()
        mock_results.posterior_distributions = [MagicMock() for _ in range(7)]

        calibration = CalibrationOutput.model_construct(
            primary_id=0,
            seed=42,
            delta_t=1.0,
            population="US-CA",
            results=mock_results,
            calibration_strategy=strategy,
        )

        result = _check_incomplete_generations(calibration)
        assert result is None

    def test_no_strategy_returns_none(self):
        """Test returns None when no calibration_strategy is set."""
        mock_results = MagicMock()
        mock_results.posterior_distributions = [MagicMock() for _ in range(5)]

        calibration = CalibrationOutput.model_construct(
            primary_id=0,
            seed=42,
            delta_t=1.0,
            population="US-CA",
            results=mock_results,
            calibration_strategy=None,
        )

        result = _check_incomplete_generations(calibration)
        assert result is None

    def test_no_num_generations_in_options_returns_none(self):
        """Test returns None when num_generations not in strategy options."""
        strategy = CalibrationStrategy(name="rejection", options={"num_particles": 100})
        mock_results = MagicMock()
        mock_results.posterior_distributions = [MagicMock() for _ in range(5)]

        calibration = CalibrationOutput.model_construct(
            primary_id=0,
            seed=42,
            delta_t=1.0,
            population="US-CA",
            results=mock_results,
            calibration_strategy=strategy,
        )

        result = _check_incomplete_generations(calibration)
        assert result is None

    def test_none_results_returns_none(self):
        """Test returns None when results is None."""
        strategy = CalibrationStrategy(name="SMC", options={"num_generations": 7})

        calibration = CalibrationOutput.model_construct(
            primary_id=0,
            seed=42,
            delta_t=1.0,
            population="US-CA",
            results=None,
            calibration_strategy=strategy,
        )

        result = _check_incomplete_generations(calibration)
        assert result is None

    def test_none_posterior_distributions_returns_none(self):
        """Test returns None when posterior_distributions is None."""
        strategy = CalibrationStrategy(name="SMC", options={"num_generations": 7})
        mock_results = MagicMock()
        mock_results.posterior_distributions = None

        calibration = CalibrationOutput.model_construct(
            primary_id=0,
            seed=42,
            delta_t=1.0,
            population="US-CA",
            results=mock_results,
            calibration_strategy=strategy,
        )

        result = _check_incomplete_generations(calibration)
        assert result is None


class TestFormatPlotNotes:
    """Test _format_plot_notes helper."""

    def test_empty_notes(self):
        """Test empty notes returns empty strings."""
        suffix, footnote = _format_plot_notes([])
        assert suffix == ""
        assert footnote == ""

    def test_single_note(self):
        """Test single note returns asterisk and footnote."""
        suffix, footnote = _format_plot_notes(["Completed 5 of 7 requested generations"])
        assert suffix == "*"
        assert footnote == "* Completed 5 of 7 requested generations"

    def test_multiple_notes(self):
        """Test multiple notes are joined with semicolons."""
        suffix, footnote = _format_plot_notes(["Note one", "Note two"])
        assert suffix == "*"
        assert footnote == "* Note one; Note two"


class TestClipToSurveillanceStart:
    """Tests for _clip_to_surveillance_start helper."""

    @pytest.fixture()
    def df(self):
        """Quantile-like DataFrame spanning Jan 1–10."""
        return pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=10, freq="D"),
                "quantile": [0.5] * 10,
                "value": range(10),
            }
        )

    def test_clips_df_to_earliest_surveillance_date(self, df):
        """Rows before the earliest surveillance date are removed."""
        surv = pd.DataFrame({"date": pd.date_range("2024-01-05", periods=3, freq="D"), "value": [1, 2, 3]})
        result = _clip_to_surveillance_start(df, surv)
        assert result is not None
        assert len(result) == 6  # Jan 5–10
        assert pd.to_datetime(result["date"]).dt.date.min() == date(2024, 1, 5)

    def test_returns_df_unchanged_when_surv_is_none(self, df):
        """When surv is None, df is returned as-is."""
        result = _clip_to_surveillance_start(df, None)
        assert result is not None
        assert len(result) == 10

    def test_returns_df_unchanged_when_surv_is_empty(self, df):
        """When surv is an empty DataFrame, df is returned as-is."""
        empty_surv = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "value": pd.Series(dtype="float64")})
        result = _clip_to_surveillance_start(df, empty_surv)
        assert result is not None
        assert len(result) == 10

    def test_returns_none_when_df_is_none(self):
        """When df is None, None is returned."""
        surv = pd.DataFrame({"date": ["2024-01-05"], "value": [1]})
        assert _clip_to_surveillance_start(None, surv) is None

    def test_returns_none_when_all_rows_clipped(self, df):
        """When surveillance starts after all df dates, None is returned."""
        surv = pd.DataFrame({"date": ["2024-02-01"], "value": [1]})
        assert _clip_to_surveillance_start(df, surv) is None


class TestClipSurveillance:
    """Tests for _clip_surveillance helper."""

    @pytest.fixture()
    def surv(self):
        """Weekly surveillance DataFrame spanning 8 weeks."""
        return pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=8, freq="W-SAT"),
                "value": range(8),
            }
        )

    def test_filter_by_start_date(self, surv):
        """Rows before surveillance_start_date are removed."""
        result = _clip_surveillance(surv, surveillance_start_date="2024-01-20")
        assert result is not None
        assert pd.to_datetime(result["date"]).dt.date.min() >= date(2024, 1, 20)

    def test_filter_by_points_with_reference_date(self, surv):
        """Keep N points before reference_date plus all points after."""
        # reference_date in the middle of the series
        ref = date(2024, 2, 3)  # between week 4 and 5
        result = _clip_surveillance(surv, surveillance_points=3, reference_date=ref)
        assert result is not None
        dates = pd.to_datetime(result["date"]).dt.date
        before = dates[dates <= ref]
        after = dates[dates > ref]
        assert len(before) == 3
        assert len(after) > 0

    def test_filter_by_points_without_reference_date(self, surv):
        """Without reference_date, keep the last N rows."""
        result = _clip_surveillance(surv, surveillance_points=3, reference_date=None)
        assert result is not None
        assert len(result) == 3
        # Should be the last 3 rows
        assert list(result["value"]) == [5, 6, 7]

    def test_start_date_takes_precedence_over_points(self, surv):
        """When both surveillance_start_date and surveillance_points are provided, start_date wins."""
        result = _clip_surveillance(
            surv,
            surveillance_start_date="2024-01-20",
            surveillance_points=2,
            reference_date=date(2024, 2, 3),
        )
        assert result is not None
        # start_date filter keeps everything >= Jan 20
        dates = pd.to_datetime(result["date"]).dt.date
        assert all(d >= date(2024, 1, 20) for d in dates)
        # Should NOT be limited to 2 points
        assert len(result) > 2

    def test_returns_none_when_surv_is_none(self):
        """When surv is None, None is returned."""
        assert _clip_surveillance(None) is None

    def test_returns_empty_when_surv_is_empty(self):
        """When surv is empty, empty DataFrame is returned."""
        empty = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "value": pd.Series(dtype="float64")})
        result = _clip_surveillance(empty)
        assert result is not None
        assert result.empty

    def test_reference_date_point_included_in_before(self, surv):
        """A point exactly on reference_date counts in the 'before' group."""
        # Pick a date that matches an actual row
        exact_date = pd.to_datetime(surv["date"]).dt.date.iloc[4]
        result = _clip_surveillance(surv, surveillance_points=2, reference_date=exact_date)
        assert result is not None
        dates = pd.to_datetime(result["date"]).dt.date
        before = dates[dates <= exact_date]
        # The point on exact_date should be in the 'before' group
        assert exact_date in before.values
        assert len(before) == 2


class TestClipToHorizon:
    """Tests for _clip_to_horizon helper."""

    @pytest.fixture()
    def proj(self):
        """Weekly projection DataFrame spanning 10 weeks from reference_date."""
        return pd.DataFrame(
            {
                "date": pd.date_range("2024-01-07", periods=10, freq="W-SUN"),
                "quantile": [0.5] * 10,
                "value": range(10),
            }
        )

    @pytest.fixture()
    def reference_date(self):
        return date(2024, 1, 7)

    def test_clips_to_max_horizon_weeks(self, proj, reference_date):
        """Only rows within horizon_max weeks of reference_date are kept."""
        result = _clip_to_horizon(proj, horizon_max=4, reference_date=reference_date)
        assert result is not None
        dates = pd.to_datetime(result["date"]).dt.date
        assert len(result) == 5  # week 0, 1, 2, 3, 4
        assert dates.max() <= date(2024, 2, 4)  # reference + 4 weeks

    def test_returns_proj_unchanged_when_horizon_max_is_none(self, proj, reference_date):
        """When horizon_max is None, proj is returned as-is."""
        result = _clip_to_horizon(proj, horizon_max=None, reference_date=reference_date)
        assert result is not None
        assert len(result) == 10

    def test_returns_none_when_proj_is_none(self, reference_date):
        """When proj is None, None is returned."""
        assert _clip_to_horizon(None, horizon_max=4, reference_date=reference_date) is None

    def test_does_not_mutate_original(self, proj, reference_date):
        """Clipping returns a copy; the original DataFrame is unchanged."""
        original_len = len(proj)
        result = _clip_to_horizon(proj, horizon_max=2, reference_date=reference_date)
        assert len(proj) == original_len
        assert result is not None
        assert len(result) < original_len
