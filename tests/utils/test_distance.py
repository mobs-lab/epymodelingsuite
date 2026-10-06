"""Tests for distance function utilities."""

from __future__ import annotations

import numpy as np
from epydemix.calibration import rmse

from epymodelingsuite.utils.distance import wrmse


class TestWrmse:
    """Tests for wrmse (weighted RMSE) distance function."""

    def test_same_interface_as_rmse(self):
        """Verify wrmse has same interface as epydemix distance functions.

        Distance functions take two dicts with "data" keys containing arrays.
        """
        data = {"data": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        simulation = {"data": np.array([1.1, 2.1, 3.1, 4.1, 5.1])}

        # Both should work with same input format
        rmse_result = rmse(data, simulation)
        wrmse_result = wrmse(data, simulation)

        assert isinstance(rmse_result, float)
        assert isinstance(wrmse_result, float)

    def test_perfect_match_returns_zero(self):
        """Verify wrmse returns 0 when simulation matches data exactly."""
        data = {"data": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        simulation = {"data": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}

        result = wrmse(data, simulation)

        assert result == 0.0

    def test_weights_recent_points_more_heavily(self):
        """Verify wrmse weights recent (later) data points more than earlier ones.

        Weight formula: w(t) = 1/((t_n+1)-t) where t is 1-indexed
        For 5 points: w = [1/5, 1/4, 1/3, 1/2, 1/1] = [0.2, 0.25, 0.33, 0.5, 1.0]
        Later points have higher weights.
        """
        # Error only at first point (early, low weight)
        data_early_error = {"data": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        sim_early_error = {"data": np.array([2.0, 2.0, 3.0, 4.0, 5.0])}  # Error at t=1

        # Error only at last point (recent, high weight)
        data_late_error = {"data": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        sim_late_error = {"data": np.array([1.0, 2.0, 3.0, 4.0, 6.0])}  # Error at t=5

        wrmse_early = wrmse(data_early_error, sim_early_error)
        wrmse_late = wrmse(data_late_error, sim_late_error)

        # Same absolute error (1.0) but late error should have higher wrmse
        assert wrmse_late > wrmse_early, "Recent errors should be weighted more heavily"

    def test_handles_nan_values(self):
        """Verify wrmse handles NaN values correctly using nanmean."""
        data = {"data": np.array([1.0, np.nan, 3.0, 4.0, 5.0])}
        simulation = {"data": np.array([1.1, 2.1, 3.1, 4.1, 5.1])}

        # Should not raise, should compute ignoring NaN
        result = wrmse(data, simulation)

        assert np.isfinite(result)
        assert result > 0

    def test_single_point(self):
        """Verify wrmse works with single data point."""
        data = {"data": np.array([5.0])}
        simulation = {"data": np.array([6.0])}

        result = wrmse(data, simulation)

        # With single point, weight is 1/(1+1-1) = 1
        # wrmse = sqrt(1 * (5-6)^2) / 1 = 1.0
        assert np.isfinite(result)
        assert result > 0

    def test_all_zeros(self):
        """Verify wrmse handles all-zero data."""
        data = {"data": np.array([0.0, 0.0, 0.0])}
        simulation = {"data": np.array([0.0, 0.0, 0.0])}

        result = wrmse(data, simulation)

        assert result == 0.0

    def test_weight_calculation_correctness(self):
        """Verify the weight calculation is correct.

        For n=4 points, weights should be:
        w(1) = 1/(4+1-1) = 1/4 = 0.25
        w(2) = 1/(4+1-2) = 1/3 ≈ 0.333
        w(3) = 1/(4+1-3) = 1/2 = 0.5
        w(4) = 1/(4+1-4) = 1/1 = 1.0
        """
        # Create data where we can verify the weighted calculation manually
        data = {"data": np.array([0.0, 0.0, 0.0, 0.0])}
        simulation = {"data": np.array([1.0, 1.0, 1.0, 1.0])}  # All errors = 1.0

        result = wrmse(data, simulation)

        w = np.array([1 / 4, 1 / 3, 1 / 2, 1])
        expected = np.sqrt(np.mean(w * 1.0)) / np.sum(w)

        assert np.isclose(result, expected, rtol=1e-5)


class TestWrmseInConfig:
    """Tests for wrmse integration with calibration config."""

    def test_wrmse_recognized_in_dist_func_dict(self):
        """Verify wrmse is available in the dispatcher's distance function dict."""
        from epymodelingsuite.dispatcher.builder import dist_func_dict

        assert "wrmse" in dist_func_dict
        assert dist_func_dict["wrmse"] is wrmse

    def test_wrmse_valid_in_calibration_schema(self):
        """Verify wrmse is accepted as a valid distance_function in CalibrationConfiguration."""
        from epymodelingsuite.schema.calibration import CalibrationConfiguration

        # Should not raise validation error
        config = CalibrationConfiguration(
            strategy={
                "name": "SMC",
                "options": {"num_particles": 10, "num_generations": 2},
            },
            distance_function="wrmse",
            observed_data_path="data/test.csv",
            comparison=[
                {
                    "observed_date_column": "date",
                    "observed_value_column": "value",
                    "simulation": ["S_to_I"],
                }
            ],
            fitting_window={"start_date": "2024-01-01", "end_date": "2024-03-01"},
            parameters={"beta": {"prior": {"type": "scipy", "name": "uniform", "args": [0.1, 0.5]}}},
        )

        assert config.distance_function == "wrmse"

    def test_all_builtin_distance_functions_in_dict(self):
        """Verify all built-in distance functions are available."""
        from epymodelingsuite.dispatcher.builder import dist_func_dict

        expected_functions = ["rmse", "wmape", "ae", "mae", "mape", "wrmse"]

        for func_name in expected_functions:
            assert func_name in dist_func_dict, f"{func_name} not in dist_func_dict"
