"""Tests for epymodelingsuite.dispatcher.output module."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from epymodelingsuite.dispatcher.output import filter_failed_projections
from epymodelingsuite.schema.output import CategoricalPlotConfig, FigureOutputTypeEnum, PlotsConfig
from epymodelingsuite.visualization.generators import generate_categorical_plots


class TestFilterFailedProjections:
    """Tests for filter_failed_projections function."""

    @pytest.fixture
    def mock_calibration_results(self):
        """Create a mock calibration results object with projections using realistic data structures."""
        results = MagicMock()
        # Realistic projection structure: dict with "date" (list of dates) and compartment/transition keys
        results.projections = {
            "baseline": [
                {
                    "date": [date(2024, 1, 1), date(2024, 1, 8), date(2024, 1, 15)],
                    "S_0-4": np.array([1000, 990, 980]),
                    "I_0-4": np.array([10, 15, 18]),
                    "R_0-4": np.array([0, 5, 12]),
                    "S_to_I_0-4": np.array([5, 6, 4]),
                },
                {
                    "date": [date(2024, 1, 1), date(2024, 1, 8), date(2024, 1, 15)],
                    "S_0-4": np.array([1000, 985, 975]),
                    "I_0-4": np.array([10, 18, 22]),
                    "R_0-4": np.array([0, 7, 13]),
                    "S_to_I_0-4": np.array([6, 7, 5]),
                },
                {
                    "date": [date(2024, 1, 1), date(2024, 1, 8), date(2024, 1, 15)],
                    "S_0-4": np.array([1000, 988, 978]),
                    "I_0-4": np.array([10, 16, 20]),
                    "R_0-4": np.array([0, 6, 12]),
                    "S_to_I_0-4": np.array([5, 6, 5]),
                },
            ]
        }
        return results

    @pytest.fixture
    def mock_calibration_results_with_failures(self):
        """Create a mock calibration results object with some failed projections (empty dicts)."""
        results = MagicMock()
        results.projections = {
            "baseline": [
                {
                    "date": [date(2024, 1, 1), date(2024, 1, 8), date(2024, 1, 15)],
                    "S_0-4": np.array([1000, 990, 980]),
                    "I_0-4": np.array([10, 15, 18]),
                    "S_to_I_0-4": np.array([5, 6, 4]),
                },
                {},  # Failed projection returns empty dict
                {
                    "date": [date(2024, 1, 1), date(2024, 1, 8), date(2024, 1, 15)],
                    "S_0-4": np.array([1000, 988, 978]),
                    "I_0-4": np.array([10, 16, 20]),
                    "S_to_I_0-4": np.array([5, 6, 5]),
                },
                {},  # Another failed projection
                {
                    "date": [date(2024, 1, 1), date(2024, 1, 8), date(2024, 1, 15)],
                    "S_0-4": np.array([1000, 985, 975]),
                    "I_0-4": np.array([10, 18, 22]),
                    "S_to_I_0-4": np.array([6, 7, 5]),
                },
            ]
        }
        return results

    @pytest.fixture
    def mock_calibration_results_multiple_scenarios(self):
        """Create a mock calibration results object with multiple scenarios and failures."""
        results = MagicMock()
        results.projections = {
            "baseline": [
                {
                    "date": [date(2024, 1, 1), date(2024, 1, 8), date(2024, 1, 15)],
                    "S_0-4": np.array([1000, 990, 980]),
                    "I_0-4": np.array([10, 15, 18]),
                },
                {},  # Failed projection
                {
                    "date": [date(2024, 1, 1), date(2024, 1, 8), date(2024, 1, 15)],
                    "S_0-4": np.array([1000, 988, 978]),
                    "I_0-4": np.array([10, 16, 20]),
                },
            ],
            "intervention": [
                {
                    "date": [date(2024, 1, 1), date(2024, 1, 8), date(2024, 1, 15)],
                    "S_0-4": np.array([1000, 995, 990]),
                    "I_0-4": np.array([10, 12, 14]),
                },
                {
                    "date": [date(2024, 1, 1), date(2024, 1, 8), date(2024, 1, 15)],
                    "S_0-4": np.array([1000, 993, 988]),
                    "I_0-4": np.array([10, 13, 15]),
                },
                {},  # Failed projection
                {},  # Another failed projection
            ],
        }
        return results

    def test_no_projections_attribute_returns_unchanged(self):
        """Test that objects without projections attribute are returned unchanged."""
        results = MagicMock(spec=[])  # No projections attribute
        filtered = filter_failed_projections(results)
        assert filtered is results
        assert filtered._filtered_count == 0

    def test_none_projections_returns_unchanged(self):
        """Test that objects with None projections are returned unchanged."""
        results = MagicMock()
        results.projections = None
        filtered = filter_failed_projections(results)
        assert filtered is results
        assert filtered._filtered_count == 0

    def test_empty_projections_dict_returns_unchanged(self):
        """Test that empty projections dictionary is handled correctly."""
        results = MagicMock()
        results.projections = {}
        filtered = filter_failed_projections(results)
        assert filtered is results
        assert filtered.projections == {}
        assert filtered._filtered_count == 0

    def test_all_valid_simulations_unchanged(self, mock_calibration_results):
        """Test that valid simulations are not modified."""
        original_count = len(mock_calibration_results.projections["baseline"])
        filtered = filter_failed_projections(mock_calibration_results)

        assert filtered is mock_calibration_results
        assert len(filtered.projections["baseline"]) == original_count
        assert all(sim for sim in filtered.projections["baseline"])  # All non-empty
        assert filtered._filtered_count == 0

    def test_filters_out_empty_dicts(self, mock_calibration_results_with_failures):
        """Test that empty dictionaries are filtered out."""
        filtered = filter_failed_projections(mock_calibration_results_with_failures)

        assert len(filtered.projections["baseline"]) == 3
        assert all(sim for sim in filtered.projections["baseline"])  # All non-empty
        assert all("date" in sim for sim in filtered.projections["baseline"])
        assert filtered._filtered_count == 2

    def test_filters_multiple_scenarios(self, mock_calibration_results_multiple_scenarios):
        """Test that filtering works across multiple scenarios."""
        filtered = filter_failed_projections(mock_calibration_results_multiple_scenarios)

        # Baseline should have 2 valid simulations
        assert len(filtered.projections["baseline"]) == 2
        assert all(sim for sim in filtered.projections["baseline"])

        # Intervention should have 2 valid simulations
        assert len(filtered.projections["intervention"]) == 2
        assert all(sim for sim in filtered.projections["intervention"])
        assert filtered._filtered_count == 3  # 1 from baseline + 2 from intervention

    def test_all_failed_simulations_returns_empty_list(self):
        """Test that scenario with all failed projections becomes empty list."""
        results = MagicMock()
        results.projections = {
            "baseline": [{}, {}, {}]  # All failed
        }

        filtered = filter_failed_projections(results)

        assert len(filtered.projections["baseline"]) == 0
        assert filtered.projections["baseline"] == []
        assert filtered._filtered_count == 3

    @patch("epymodelingsuite.dispatcher.output.logger")
    def test_logs_warning_when_filtering(self, mock_logger, mock_calibration_results_with_failures):
        """Test that warning is logged when failed projections are filtered."""
        filter_failed_projections(mock_calibration_results_with_failures)

        # Should have logged a warning
        assert mock_logger.warning.called
        call_args = mock_logger.warning.call_args[0]

        # Check the log message format
        assert "Filtered out" in call_args[0]
        assert "failed projection(s)" in call_args[0]
        assert call_args[1] == 2  # Number filtered
        assert call_args[2] == "baseline"  # Scenario ID
        assert call_args[3] == 3  # Number kept
        assert call_args[4] == 5  # Total original

    @patch("epymodelingsuite.dispatcher.output.logger")
    def test_no_warning_when_all_valid(self, mock_logger, mock_calibration_results):
        """Test that no warning is logged when all simulations are valid."""
        filter_failed_projections(mock_calibration_results)

        # Should not have logged a warning
        assert not mock_logger.warning.called

    @patch("epymodelingsuite.dispatcher.output.logger")
    def test_logs_warning_for_each_scenario_with_failures(
        self, mock_logger, mock_calibration_results_multiple_scenarios
    ):
        """Test that warning is logged for each scenario that has failures."""
        filter_failed_projections(mock_calibration_results_multiple_scenarios)

        # Should have logged two warnings (one per scenario with failures)
        assert mock_logger.warning.call_count == 2

    def test_preserves_simulation_data(self, mock_calibration_results_with_failures):
        """Test that valid projection data is preserved exactly."""
        original_valid_sims = [sim for sim in mock_calibration_results_with_failures.projections["baseline"] if sim]

        filtered = filter_failed_projections(mock_calibration_results_with_failures)
        filtered_sims = filtered.projections["baseline"]

        assert len(filtered_sims) == len(original_valid_sims)
        for original, filtered_sim in zip(original_valid_sims, filtered_sims, strict=False):
            assert original == filtered_sim
        assert filtered._filtered_count == 2

    def test_empty_scenario_list_handled(self):
        """Test that empty list for a scenario is handled correctly."""
        results = MagicMock()
        results.projections = {"baseline": []}

        filtered = filter_failed_projections(results)

        assert filtered.projections["baseline"] == []
        assert filtered._filtered_count == 0

    def test_none_scenario_value_handled(self):
        """Test that None value for a scenario is handled correctly."""
        results = MagicMock()
        results.projections = {"baseline": None}

        filtered = filter_failed_projections(results)

        # Should skip None scenario and not crash
        assert filtered is results
        assert filtered._filtered_count == 0

    def test_stores_filtered_count_on_results(self, mock_calibration_results_with_failures):
        """Test that _filtered_count is stored on the results object."""
        filtered = filter_failed_projections(mock_calibration_results_with_failures)

        # Should store the count on the results object
        assert hasattr(filtered, "_filtered_count")
        assert filtered._filtered_count == 2

    def test_stores_filtered_count_zero_when_no_failures(self, mock_calibration_results):
        """Test that _filtered_count is 0 when there are no failures."""
        filtered = filter_failed_projections(mock_calibration_results)

        # Should store 0 when no failures
        assert hasattr(filtered, "_filtered_count")
        assert filtered._filtered_count == 0

    def test_stores_filtered_count_multiple_scenarios(self, mock_calibration_results_multiple_scenarios):
        """Test that _filtered_count aggregates across all scenarios."""
        filtered = filter_failed_projections(mock_calibration_results_multiple_scenarios)

        # Should aggregate count across scenarios (1 + 2 = 3)
        assert hasattr(filtered, "_filtered_count")
        assert filtered._filtered_count == 3


class TestGenerateCategoricalPlots:
    """Tests for generate_categorical_plots function."""

    @pytest.fixture
    def mock_hub_format_data(self):
        """Create mock FluSight hub format data with rate-trend forecasts."""
        data = []
        for horizon in [0, 1, 2, 3]:
            for location in ["06", "48"]:  # FIPS codes for CA and TX
                for category in ["large_decrease", "decrease", "stable", "increase", "large_increase"]:
                    data.append(
                        {
                            "reference_date": date(2025, 11, 26),
                            "location": location,
                            "target": "wk flu hosp rate change",
                            "horizon": horizon,
                            "target_end_date": date(2025, 12, 7 + horizon * 7),
                            "output_type": "pmf",
                            "output_type_id": category,
                            "value": 0.2,  # Equal probabilities for simplicity
                        }
                    )
        return pd.DataFrame(data)

    @pytest.fixture
    def mock_plots_config_enabled(self):
        """Create PlotsConfig with categorical plots enabled."""
        return PlotsConfig(
            reference_date=date(2025, 11, 26),
            figure_output_types=[FigureOutputTypeEnum.PNG],
            dpi=150,
            categorical=CategoricalPlotConfig(),
        )

    @pytest.fixture
    def mock_plots_config_disabled(self):
        """Create PlotsConfig with categorical plots disabled."""
        return PlotsConfig(
            reference_date=date(2025, 11, 26),
            figure_output_types=[FigureOutputTypeEnum.PNG],
            dpi=150,
            categorical=None,
        )

    def test_categorical_plots_from_flusight_data(self, mock_hub_format_data, mock_plots_config_enabled):
        """Test categorical plot generation from FluSight rate-trend data."""
        out_dict = {}

        # Call generate_categorical_plots
        generate_categorical_plots(mock_plots_config_enabled, out_dict, mock_hub_format_data)

        # Assert: "categorical_rate_trends" in out_dict
        assert "categorical_rate_trends" in out_dict

        # Assert: OutputObject has correct types
        assert len(out_dict["categorical_rate_trends"]) == 1  # One output type (PNG)
        output_obj = out_dict["categorical_rate_trends"][0]
        assert output_obj.name == "categorical_rate_trends.png"
        assert output_obj.output_type == FigureOutputTypeEnum.PNG

    def test_categorical_plots_skip_when_disabled(self, mock_hub_format_data, mock_plots_config_disabled):
        """Test categorical plots are skipped when config.enabled=False."""
        out_dict = {}

        # Call generate_categorical_plots with disabled config
        generate_categorical_plots(mock_plots_config_disabled, out_dict, mock_hub_format_data)

        # Assert: "categorical_rate_trends" NOT in out_dict
        assert "categorical_rate_trends" not in out_dict

    def test_categorical_plots_skip_when_no_flusight_format(self, mock_plots_config_enabled, caplog):
        """Test categorical plots are skipped when FluSight format not configured."""
        out_dict = {}

        # Call generate_categorical_plots with hub_format_data=None
        generate_categorical_plots(mock_plots_config_enabled, out_dict, hub_format_data=None)

        # Assert: categorical plots skipped
        assert "categorical_rate_trends" not in out_dict

        # Assert: warning mentions "flusight_format.rate_trends"
        assert any(
            "Categorical plots enabled but no FluSight format data available" in rec.message for rec in caplog.records
        )
        assert any("flusight_format.rate_trends" in rec.message for rec in caplog.records)

    def test_categorical_plots_skip_when_no_rate_trends(self, mock_plots_config_enabled, caplog):
        """Test categorical plots are skipped when rate-trend data missing."""
        out_dict = {}

        # Create hub format data WITHOUT rate-trend forecasts (only hospitalizations)
        hub_format_data = pd.DataFrame(
            {
                "reference_date": [date(2025, 11, 26)],
                "location": ["06"],
                "target": ["wk ahead inc flu hosp"],  # Different target, not rate-trend
                "horizon": [0],
                "target_end_date": [date(2025, 12, 7)],
                "output_type": ["quantile"],
                "output_type_id": ["0.5"],
                "value": [100.0],
            }
        )

        # Call generate_categorical_plots
        generate_categorical_plots(mock_plots_config_enabled, out_dict, hub_format_data)

        # Assert: categorical plots skipped
        assert "categorical_rate_trends" not in out_dict

        # Assert: warning mentions "rate-trend categorical forecasts"
        assert any("No rate-trend categorical forecasts found" in rec.message for rec in caplog.records)
        assert any("flusight_format.rate_trends" in rec.message for rec in caplog.records)


class TestGetHubLocationId:
    """Tests for get_hub_location_id function."""

    def test_iso_location_returns_fips(self):
        """Test that ISO locations return FIPS codes."""
        from epymodelingsuite.dispatcher.output import get_hub_location_id

        # California
        result = get_hub_location_id("United_States_California")
        assert result == "06"

        # Texas
        result = get_hub_location_id("United_States_Texas")
        assert result == "48"

    def test_metrocast_location_returns_location_id(self):
        """Test that metrocast locations return metrocast_location_id."""
        from epymodelingsuite.dispatcher.output import get_hub_location_id

        # Denver
        result = get_hub_location_id("metrocast_location_denver")
        assert result == "denver"

        # Boston
        result = get_hub_location_id("metrocast_location_boston")
        assert result == "boston"

        # NC flu region
        result = get_hub_location_id("metrocast_location_nenc")
        assert result == "nenc"


class TestGetPlotLocationLabel:
    """Tests for get_plot_location_label function."""

    def test_iso_location_returns_name(self):
        """Test that ISO locations return human-readable names."""
        from epymodelingsuite.dispatcher.output import get_plot_location_label

        # California
        result = get_plot_location_label("United_States_California")
        assert result == "California"

        # Massachusetts
        result = get_plot_location_label("United_States_Massachusetts")
        assert result == "Massachusetts"

    def test_metrocast_location_returns_short_name(self):
        """Test that metrocast locations return short name from CSV."""
        from epymodelingsuite.dispatcher.output import get_plot_location_label

        # Denver - same as full name
        result = get_plot_location_label("metrocast_location_denver")
        assert result == "Denver, CO"

        # Boston - shortened from "Boston Metro, South Shore, Cape & Islands, MA"
        result = get_plot_location_label("metrocast_location_boston")
        assert result == "Greater Boston, MA"

        # NC flu region
        result = get_plot_location_label("metrocast_location_nenc")
        assert result == "Northeastern, NC"
