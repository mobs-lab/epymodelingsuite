"""Tests for helper functions extracted from make_simulate_wrapper."""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from flumodelingsuite.builders.orchestrators import (
    align_simulation_to_observed_dates,
    apply_calibrated_parameters,
    apply_seasonality_with_sampled_min,
    apply_vaccination_for_sampled_start,
    compute_simulation_start_date,
    flatten_simulation_results,
)
from flumodelingsuite.schema.basemodel import BaseEpiModel, Parameter, Seasonality, Timespan, Vaccination


class TestComputeSimulationStartDate:
    """Tests for compute_simulation_start_date function."""

    def test_no_sampling_returns_basemodel_start_date(self):
        """Test that without sampling, returns basemodel start date."""
        params = {"projection": False}
        basemodel_timespan = Timespan(start_date=date(2024, 1, 1), end_date=date(2024, 12, 31), delta_t=1.0)
        sampled_start_timespan = None

        result = compute_simulation_start_date(params, basemodel_timespan, sampled_start_timespan)

        assert result == date(2024, 1, 1)

    def test_projection_mode_returns_earliest_start(self):
        """Test that projection mode uses earliest start date for consistent trajectory lengths."""
        params = {"projection": True, "start_date": 10}
        basemodel_timespan = Timespan(start_date=date(2024, 1, 1), end_date=date(2024, 12, 31), delta_t=1.0)
        sampled_start_timespan = Timespan(start_date=date(2024, 1, 5), end_date=date(2024, 12, 31), delta_t=1.0)

        result = compute_simulation_start_date(params, basemodel_timespan, sampled_start_timespan)

        # Should use earliest start, not the offset
        assert result == date(2024, 1, 5)

    def test_calibration_mode_uses_sampled_offset(self):
        """Test that calibration mode applies sampled offset to earliest start."""
        params = {"projection": False, "start_date": 10}
        basemodel_timespan = Timespan(start_date=date(2024, 1, 1), end_date=date(2024, 12, 31), delta_t=1.0)
        sampled_start_timespan = Timespan(start_date=date(2024, 1, 5), end_date=date(2024, 12, 31), delta_t=1.0)

        result = compute_simulation_start_date(params, basemodel_timespan, sampled_start_timespan)

        # Should use earliest + offset (Jan 5 + 10 days = Jan 15)
        assert result == date(2024, 1, 15)

    def test_negative_offset(self):
        """Test that negative offset works correctly."""
        params = {"projection": False, "start_date": -5}
        basemodel_timespan = Timespan(start_date=date(2024, 1, 1), end_date=date(2024, 12, 31), delta_t=1.0)
        sampled_start_timespan = Timespan(start_date=date(2024, 1, 10), end_date=date(2024, 12, 31), delta_t=1.0)

        result = compute_simulation_start_date(params, basemodel_timespan, sampled_start_timespan)

        # Jan 10 - 5 days = Jan 5
        assert result == date(2024, 1, 5)


class TestApplyCalibratedParameters:
    """Tests for apply_calibrated_parameters function."""

    def test_applies_calibrated_parameters(self):
        """Test that calibrated parameters are extracted and applied to model."""
        model = Mock()
        params = {"beta": 0.5, "gamma": 0.1, "epimodel": "something", "projection": False}
        parameter_config = {
            "beta": Parameter(type="calibrated"),
            "gamma": Parameter(type="calibrated"),
            "alpha": Parameter(type="scalar", value=0.1),
        }

        with patch("flumodelingsuite.builders.orchestrators.add_model_parameters_from_config") as mock_add:
            apply_calibrated_parameters(model, params, parameter_config)

            # Should only extract beta and gamma (calibrated params)
            mock_add.assert_called_once()
            call_args = mock_add.call_args
            assert call_args[0][0] == model
            added_params = call_args[0][1]
            assert "beta" in added_params
            assert "gamma" in added_params
            assert "alpha" not in added_params
            assert added_params["beta"].value == 0.5
            assert added_params["gamma"].value == 0.1

    def test_no_calibrated_parameters_skips_add(self):
        """Test that if no calibrated parameters, nothing is added."""
        model = Mock()
        params = {"epimodel": "something", "projection": False}
        parameter_config = {"alpha": Parameter(type="scalar", value=0.1)}

        with patch("flumodelingsuite.builders.orchestrators.add_model_parameters_from_config") as mock_add:
            apply_calibrated_parameters(model, params, parameter_config)

            # Should not call add since no calibrated params
            mock_add.assert_not_called()

    def test_recalculates_derived_parameters(self):
        """Test that calculated parameters trigger recalculation."""
        model = Mock()
        params = {"beta": 0.5}
        parameter_config = {
            "beta": Parameter(type="calibrated"),
            "derived": Parameter(type="calculated", value="beta * 2"),
        }

        with patch("flumodelingsuite.builders.orchestrators.add_model_parameters_from_config") as mock_add, patch(
            "flumodelingsuite.builders.orchestrators.calculate_parameters_from_config"
        ) as mock_calc:
            apply_calibrated_parameters(model, params, parameter_config)

            # Should call both add and calculate
            mock_add.assert_called_once()
            mock_calc.assert_called_once_with(model, parameter_config)


class TestApplyVaccinationForSampledStart:
    """Tests for apply_vaccination_for_sampled_start function."""

    def test_no_vaccination_returns_early(self):
        """Test that without vaccination config, function returns early."""
        model = Mock()
        basemodel = Mock(vaccination=None)
        timespan = Timespan(start_date=date(2024, 1, 1), end_date=date(2024, 12, 31), delta_t=1.0)

        with patch("flumodelingsuite.builders.orchestrators.reaggregate_vaccines") as mock_reagg, patch(
            "flumodelingsuite.builders.orchestrators.add_vaccination_schedules_from_config"
        ) as mock_add:
            apply_vaccination_for_sampled_start(model, basemodel, timespan, None, None)

            # Should not call vaccination functions
            mock_reagg.assert_not_called()
            mock_add.assert_not_called()

    def test_no_sampling_returns_early(self):
        """Test that without start_date sampling, vaccination already applied so returns early."""
        model = Mock()
        basemodel = Mock(vaccination=Mock())
        timespan = Timespan(start_date=date(2024, 1, 1), end_date=date(2024, 12, 31), delta_t=1.0)

        with patch("flumodelingsuite.builders.orchestrators.reaggregate_vaccines") as mock_reagg, patch(
            "flumodelingsuite.builders.orchestrators.add_vaccination_schedules_from_config"
        ) as mock_add:
            apply_vaccination_for_sampled_start(model, basemodel, timespan, None, sampled_start_timespan=None)

            # Should not reaggregate since start_date not sampled
            mock_reagg.assert_not_called()
            mock_add.assert_not_called()

    def test_reaggregates_vaccination_when_start_date_sampled(self):
        """Test that vaccination is reaggregated when start_date is sampled."""
        model = Mock()
        basemodel = Mock(vaccination=Mock(), transitions=[])
        timespan = Timespan(start_date=date(2024, 1, 15), end_date=date(2024, 12, 31), delta_t=1.0)
        earliest_vax = {"vax_data": "some_data"}
        sampled_start_timespan = Timespan(start_date=date(2024, 1, 1), end_date=date(2024, 12, 31), delta_t=1.0)

        with patch("flumodelingsuite.builders.orchestrators.reaggregate_vaccines") as mock_reagg, patch(
            "flumodelingsuite.builders.orchestrators.add_vaccination_schedules_from_config"
        ) as mock_add:
            mock_reagg.return_value = {"reaggregated": "data"}

            apply_vaccination_for_sampled_start(model, basemodel, timespan, earliest_vax, sampled_start_timespan)

            # Should reaggregate and add
            mock_reagg.assert_called_once_with(earliest_vax, date(2024, 1, 15))
            mock_add.assert_called_once()


class TestApplySeasonalityWithSampledMin:
    """Tests for apply_seasonality_with_sampled_min function."""

    def test_no_seasonality_returns_early(self):
        """Test that without seasonality config, function returns early."""
        model = Mock()
        basemodel = Mock(seasonality=None)
        timespan = Timespan(start_date=date(2024, 1, 1), end_date=date(2024, 12, 31), delta_t=1.0)
        params = {}

        with patch("flumodelingsuite.builders.orchestrators.add_seasonality_from_config") as mock_add:
            apply_seasonality_with_sampled_min(model, basemodel, timespan, params)

            # Should not call seasonality functions
            mock_add.assert_not_called()

    def test_applies_seasonality_without_sampled_min(self):
        """Test that seasonality is applied without modifying min_value."""
        model = Mock()
        seasonality_config = Seasonality(
            method="balcan",
            latitude=40.0,
            min_value=0.5,
            max_value=1.5,
            target_parameter="beta",
            seasonality_min_date=date(2024, 7, 1),
            seasonality_max_date=date(2024, 1, 1),
        )
        basemodel = Mock(seasonality=seasonality_config)
        timespan = Timespan(start_date=date(2024, 1, 1), end_date=date(2024, 12, 31), delta_t=1.0)
        params = {}

        with patch("flumodelingsuite.builders.orchestrators.add_seasonality_from_config") as mock_add:
            apply_seasonality_with_sampled_min(model, basemodel, timespan, params)

            # Should call with copied config
            mock_add.assert_called_once()
            call_args = mock_add.call_args
            applied_config = call_args[0][1]
            assert applied_config.min_value == 0.5  # Original value

    def test_applies_seasonality_with_sampled_min(self):
        """Test that seasonality min_value is overridden when sampled."""
        model = Mock()
        seasonality_config = Seasonality(
            method="balcan",
            latitude=40.0,
            min_value=0.5,
            max_value=1.5,
            target_parameter="beta",
            seasonality_min_date=date(2024, 7, 1),
            seasonality_max_date=date(2024, 1, 1),
        )
        basemodel = Mock(seasonality=seasonality_config)
        timespan = Timespan(start_date=date(2024, 1, 1), end_date=date(2024, 12, 31), delta_t=1.0)
        params = {"seasonality_min": 0.3}

        with patch("flumodelingsuite.builders.orchestrators.add_seasonality_from_config") as mock_add:
            apply_seasonality_with_sampled_min(model, basemodel, timespan, params)

            # Should call with modified min_value
            mock_add.assert_called_once()
            call_args = mock_add.call_args
            applied_config = call_args[0][1]
            assert applied_config.min_value == 0.3  # Sampled value

    def test_does_not_mutate_original_seasonality(self):
        """Test that original basemodel seasonality is not mutated."""
        model = Mock()
        seasonality_config = Seasonality(
            method="balcan",
            latitude=40.0,
            min_value=0.5,
            max_value=1.5,
            target_parameter="beta",
            seasonality_min_date=date(2024, 7, 1),
            seasonality_max_date=date(2024, 1, 1),
        )
        basemodel = Mock(seasonality=seasonality_config)
        timespan = Timespan(start_date=date(2024, 1, 1), end_date=date(2024, 12, 31), delta_t=1.0)
        params = {"seasonality_min": 0.3}

        with patch("flumodelingsuite.builders.orchestrators.add_seasonality_from_config"):
            apply_seasonality_with_sampled_min(model, basemodel, timespan, params)

            # Original should not be mutated
            assert basemodel.seasonality.min_value == 0.5


class TestAlignSimulationToObservedDates:
    """Tests for align_simulation_to_observed_dates function."""

    def test_aligns_simulation_to_observed_dates(self):
        """Test that simulation results are aligned to observation dates."""
        # Mock simulation results
        results = Mock()
        results.dates = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)]
        results.transitions = {
            "Hosp_vax": np.array([10, 20, 30, 40, 50]),
            "Hosp_unvax": np.array([5, 10, 15, 20, 25]),
        }

        comparison_transitions = ["Hosp_vax", "Hosp_unvax"]
        data_dates = [date(2024, 1, 2), date(2024, 1, 4)]

        result = align_simulation_to_observed_dates(results, comparison_transitions, data_dates)

        # Should sum transitions and extract only dates matching data_dates
        expected = np.array([30, 60])  # [20+10, 40+20] for dates Jan 2 and Jan 4
        np.testing.assert_array_equal(result, expected)

    def test_pads_with_zeros_when_simulation_shorter(self):
        """Test that zeros are padded when simulation is shorter than observations."""
        results = Mock()
        results.dates = [date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)]
        results.transitions = {
            "Hosp": np.array([30, 40, 50]),
        }

        comparison_transitions = ["Hosp"]
        data_dates = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)]

        result = align_simulation_to_observed_dates(results, comparison_transitions, data_dates)

        # Should pad with 2 zeros at beginning
        expected = np.array([0, 0, 30, 40, 50])
        np.testing.assert_array_equal(result, expected)

    def test_handles_multiple_transitions(self):
        """Test that multiple transitions are summed correctly."""
        results = Mock()
        results.dates = [date(2024, 1, 1), date(2024, 1, 2)]
        results.transitions = {
            "Hosp_vax_0_4": np.array([1, 2]),
            "Hosp_vax_5_17": np.array([3, 4]),
            "Hosp_unvax_0_4": np.array([5, 6]),
            "Hosp_unvax_5_17": np.array([7, 8]),
        }

        comparison_transitions = ["Hosp_vax_0_4", "Hosp_vax_5_17", "Hosp_unvax_0_4", "Hosp_unvax_5_17"]
        data_dates = [date(2024, 1, 1), date(2024, 1, 2)]

        result = align_simulation_to_observed_dates(results, comparison_transitions, data_dates)

        # Should sum all transitions: [1+3+5+7, 2+4+6+8]
        expected = np.array([16, 20])
        np.testing.assert_array_equal(result, expected)


class TestFlattenSimulationResults:
    """Tests for flatten_simulation_results function."""

    def test_flattens_results_structure(self):
        """Test that results are flattened into single dict."""
        results = Mock()
        results.dates = [date(2024, 1, 1), date(2024, 1, 2)]
        results.transitions = {
            "Hosp_vax": np.array([10, 20]),
            "Hosp_unvax": np.array([5, 10]),
        }
        results.compartments = {
            "S": np.array([1000, 990]),
            "I": np.array([10, 20]),
            "R": np.array([0, 0]),
        }

        result = flatten_simulation_results(results)

        # Should have dates at top level plus all transitions and compartments
        assert "dates" in result
        assert result["dates"] == results.dates
        assert "Hosp_vax" in result
        assert "Hosp_unvax" in result
        assert "S" in result
        assert "I" in result
        assert "R" in result
        np.testing.assert_array_equal(result["Hosp_vax"], np.array([10, 20]))
        np.testing.assert_array_equal(result["S"], np.array([1000, 990]))

    def test_handles_empty_transitions_and_compartments(self):
        """Test that empty transitions/compartments are handled."""
        results = Mock()
        results.dates = [date(2024, 1, 1)]
        results.transitions = {}
        results.compartments = {}

        result = flatten_simulation_results(results)

        # Should only have dates
        assert result == {"dates": [date(2024, 1, 1)]}
