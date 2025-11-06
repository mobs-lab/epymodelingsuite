"""Tests for epymodelingsuite.dispatcher.output module."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from epymodelingsuite.dispatcher.output import filter_failed_projections


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
