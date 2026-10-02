"""Compare batching against epydemix and independently known quantiles."""

import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from epydemix.calibration import CalibrationResults
from epydemix.model.simulation_output import Trajectory
from epydemix.model.simulation_results import SimulationResults

from epymodelingsuite.utils.quantiles import compute_quantiles, get_calibration_quantiles, get_projection_quantiles


def make_results(values):
    dates = pd.date_range("2024-01-06", periods=values.shape[1], freq="W-SAT")
    draws = [{"date": dates, "data": row.copy(), "other": row.copy() * 2, "random_state": {}} for row in values]
    return CalibrationResults(selected_trajectories={0: draws}, projections={"baseline": draws}), dates


@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64])
@pytest.mark.parametrize("levels", [[0.05, 0.5, 0.95], [0.75, 0.25, 0.75], [0.5], []])
@pytest.mark.parametrize("ignore_nan", [False, True])
@pytest.mark.parametrize("draws", [1, 4])
def test_calibration_and_projection_match_epydemix(dtype, levels, ignore_nan, draws):
    values = np.arange(draws * 3).reshape(draws, 3).astype(dtype)
    results, dates = make_results(values)
    for kind, batched in [("calibration", get_calibration_quantiles), ("projection", get_projection_quantiles)]:
        expected = getattr(results, f"get_{kind}_quantiles")(dates=dates, quantiles=levels, ignore_nan=ignore_nan)
        actual = batched(results, dates=dates, quantiles=levels, ignore_nan=ignore_nan)
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    for i, draw in enumerate(results.projections["baseline"]):
        np.testing.assert_array_equal(draw["data"], values[i])


@pytest.mark.parametrize("ignore_nan", [False, True])
def test_nan_values_and_warning_match(ignore_nan):
    values = np.array([[np.nan, np.nan, 1.0], [np.nan, 2.0, 3.0], [np.nan, 4.0, 5.0]])
    results, dates = make_results(values)
    with warnings.catch_warnings(record=True) as expected_warnings:
        warnings.simplefilter("always")
        expected = results.get_projection_quantiles(dates=dates, quantiles=[0.25, 0.5, 0.75], ignore_nan=ignore_nan)
    with warnings.catch_warnings(record=True) as actual_warnings:
        warnings.simplefilter("always")
        actual = get_projection_quantiles(results, dates=dates, quantiles=[0.25, 0.5, 0.75], ignore_nan=ignore_nan)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert {str(w.message) for w in actual_warnings} == {str(w.message) for w in expected_warnings}


@pytest.mark.parametrize("nan_count, warns", [(2, False), (3, True)])
def test_high_nan_warning_boundary(nan_count, warns):
    values = np.ones((4, 1))
    values[:nan_count] = np.nan
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compute_quantiles({"value": values}, [0], [0.5], ignore_nan=True)
    assert bool(caught) is warns


def test_known_quantiles_and_single_call_per_variable():
    values = np.array([[0, 10], [10, 30], [20, 50]])
    with patch("epymodelingsuite.utils.quantiles.np.quantile", wraps=np.quantile) as quantile:
        actual = compute_quantiles({"value": values}, ["first", "second"], [0.25, 0.5, 0.75])
    assert quantile.call_count == 1
    assert actual.to_dict("list") == {
        "date": ["first", "second"] * 3,
        "quantile": [0.25, 0.25, 0.5, 0.5, 0.75, 0.75],
        "value": [5.0, 20.0, 10.0, 30.0, 15.0, 40.0],
    }


def test_variable_generation_and_scenario_selection():
    results, dates = make_results(np.arange(12).reshape(4, 3))
    later, _ = make_results(np.arange(12).reshape(4, 3) + 100)
    results.selected_trajectories[2] = later.selected_trajectories[0]
    results.projections["intervention"] = later.projections["baseline"]
    for generation in [None, 0, 2]:
        kwargs = dict(dates=dates, quantiles=[0.25, 0.5], generation=generation, variables=["other", "data"])
        pd.testing.assert_frame_equal(
            get_calibration_quantiles(results, **kwargs), results.get_calibration_quantiles(**kwargs), check_exact=True
        )
    for scenario in ["baseline", "intervention"]:
        kwargs = dict(dates=dates, quantiles=[0.5], scenario_id=scenario, variables=["other"])
        with patch.object(results, "get_projection_trajectories", wraps=results.get_projection_trajectories) as stack:
            actual = get_projection_quantiles(results, **kwargs)
        stack.assert_called_once_with(scenario, variables=["other"])
        pd.testing.assert_frame_equal(actual, results.get_projection_quantiles(**kwargs), check_exact=True)


@pytest.mark.parametrize("ignore_nan", [False, True])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_random_interpolation_preserves_scalar_rounding(ignore_nan, dtype):
    values = np.random.default_rng(3).normal(size=(5, 4)).astype(dtype)
    values[0, 0] = np.nan
    results, dates = make_results(values)
    kwargs = dict(dates=dates, quantiles=[0.025, 0.5, 0.975], ignore_nan=ignore_nan)
    pd.testing.assert_frame_equal(
        get_projection_quantiles(results, **kwargs), results.get_projection_quantiles(**kwargs), check_exact=True
    )


@pytest.mark.parametrize("variables", [None, [], ["missing"], ["data", "missing"]])
def test_default_dates_and_missing_variables_match(variables):
    results, _ = make_results(np.arange(12).reshape(4, 3))
    kwargs = dict(quantiles=[0.5], variables=variables)
    if variables == ["missing"]:
        with pytest.raises(IndexError):
            get_projection_quantiles(results, **kwargs)
        with pytest.raises(IndexError):
            results.get_projection_quantiles(**kwargs)
    else:
        pd.testing.assert_frame_equal(
            get_projection_quantiles(results, **kwargs), results.get_projection_quantiles(**kwargs), check_exact=True
        )


def test_empty_and_invalid_inputs():
    results = CalibrationResults(projections={"baseline": []})
    with pytest.raises(IndexError):
        get_projection_quantiles(results, quantiles=[0.5])
    pd.testing.assert_frame_equal(
        get_projection_quantiles(results, dates=[0, 1], quantiles=[0.5]),
        results.get_projection_quantiles(dates=[0, 1], quantiles=[0.5]),
        check_exact=True,
    )
    with pytest.raises(ValueError, match="No projections"):
        get_projection_quantiles(results, scenario_id="missing")
    with pytest.raises(ValueError, match="shape"):
        compute_quantiles({"value": np.zeros((2, 3, 4))}, [0, 1, 2], [0.5])
    with pytest.raises(ValueError):
        compute_quantiles({"value": np.zeros((2, 3))}, [0, 1], [0.5])


@pytest.mark.parametrize("ignore_nan", [False, True])
def test_simulation_matches_epydemix(ignore_nan):
    dates = pd.date_range("2024-01-06", periods=3, freq="W-SAT")
    trajectories = [
        Trajectory(
            compartments={"S": np.array([10.0 + i, np.nan, 20.0])},
            transitions={"S_to_I": np.array([i, i + 1, i + 2])},
            dates=dates,
            compartment_idx={},
            transitions_idx={},
            parameters={},
        )
        for i in range(3)
    ]
    results = SimulationResults(trajectories, {})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        for kind in ["compartments", "transitions"]:
            expected = getattr(results, f"get_quantiles_{kind}")(quantiles=[0.75, 0.25], ignore_nan=ignore_nan)
            stacked = getattr(results, f"get_stacked_{kind}")()
            actual = compute_quantiles(stacked, results.dates, [0.75, 0.25], ignore_nan=ignore_nan)
            pd.testing.assert_frame_equal(actual, expected, check_exact=True)
