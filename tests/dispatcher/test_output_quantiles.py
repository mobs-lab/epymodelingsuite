"""Exercise selection, fallbacks and date-only reads through real output generators."""

from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from epydemix.calibration import CalibrationResults
from epydemix.model.simulation_output import Trajectory
from epydemix.model.simulation_results import SimulationResults

from epymodelingsuite.dispatcher.output import generate_calibration_outputs, generate_simulation_outputs
from epymodelingsuite.schema.dispatcher import CalibrationOutput, SimulationOutput
from epymodelingsuite.schema.output import OutputConfig, PlotsConfig
from epymodelingsuite.visualization.generators import _compute_fitting_window, _fetch_quantiles_for_location


def calibration():
    """Build a deterministic California result with compartments, transitions and calibration data."""
    dates = pd.date_range("2024-01-06", periods=6, freq="W-SAT").to_list()
    draws = [
        {
            "date": dates,
            "S": np.arange(6) + i,
            "I": np.arange(6) + 10 + i,
            "S_to_I": np.arange(6) + 20 + i,
            "I_to_R": np.arange(6) + 30 + i,
            "hospitalizations": np.arange(6) + 40 + i,
        }
        for i in range(4)
    ]
    selected = [{"date": dates, "data": d["hospitalizations"]} for d in draws]
    results = CalibrationResults(selected_trajectories={0: selected}, projections={"baseline": draws})
    return CalibrationOutput(primary_id=7, seed=13, population="United_States_California", results=results)


@pytest.mark.parametrize(
    "compartments, transitions, expected_variables",
    [
        (["I", "S"], ["I_to_R"], ["S", "I", "I_to_R"]),
        (False, True, ["S_to_I", "I_to_R"]),
        (True, False, ["S", "I", "hospitalizations"]),
        (["missing"], ["I_to_R"], None),
        (["S"], ["missing"], None),
    ],
)
def test_calibration_selection_matches_upstream_and_preserves_fallback(compartments, transitions, expected_variables):
    """Check selective stacking and exact output layout; missing requested names retain all-column fallback."""
    item = calibration()
    config = OutputConfig.model_validate(
        {
            "output": {
                "quantiles": {
                    "selections": [0.75, 0.25],
                    "compartments": compartments,
                    "transitions": transitions,
                }
            }
        }
    )
    upstream = item.results.get_projection_quantiles(
        dates=item.results.projections["baseline"][0]["date"],
        quantiles=[0.75, 0.25],
        ignore_nan=True,
    )
    with patch.object(
        item.results, "get_projection_trajectories", wraps=item.results.get_projection_trajectories
    ) as stack:
        output = generate_calibration_outputs(calibrations=[item], output_config=config)
    stack.assert_called_once_with("baseline", variables=expected_variables)
    for kind, selected in [("compartments", compartments), ("transitions", transitions)]:
        if not selected:
            continue
        if isinstance(selected, list) and all(name in upstream.columns for name in selected):
            columns = selected
        else:
            columns = [
                name
                for name in upstream
                if name not in ("date", "quantile") and ("_to_" in name) == (kind == "transitions")
            ]
        expected = upstream[["date", "quantile", *columns]].copy()
        expected.insert(0, "primary_id", 7)
        expected.insert(1, "seed", 13)
        expected.insert(2, "population", item.population)
        pd.testing.assert_frame_equal(output[f"quantiles_projection_{kind}"][0].data, expected, check_exact=True)


def test_calibration_only_and_metadata_do_not_stack_projections():
    """Ensure calibration-only output and metadata never stack projections, while retaining correct date windows."""
    item = calibration()
    config = OutputConfig.model_validate({"output": {"quantiles": {"calibration": True}}})
    with (
        patch.object(item.results, "get_projection_trajectories", side_effect=AssertionError("unnecessary stack")),
        patch.object(
            item.results, "get_calibration_trajectories", wraps=item.results.get_calibration_trajectories
        ) as stack,
    ):
        output = generate_calibration_outputs(calibrations=[item], output_config=config)
    stack.assert_called_once_with(None, variables=["data"])
    assert set(output) == {"quantiles_calibration", "model_metadata"}
    row = output["model_metadata"][0].data.iloc[0]
    assert (row.fitting_start, row.fitting_end, row.start_date, row.end_date) == (
        "2024-01-06",
        "2024-02-10",
        "2024-01-06",
        "2024-02-10",
    )


def test_plot_fetch_selects_target_and_fitting_window_needs_no_quantiles():
    """Stack only the plotted projection target and read fitting dates without stacking calibration samples."""
    item = calibration()
    config = PlotsConfig(reference_date=date(2024, 1, 13))
    expected = item.results.get_projection_quantiles(
        dates=item.results.projections["baseline"][0]["date"],
        quantiles=config.quantiles.quantiles,
        variables=["hospitalizations"],
        ignore_nan=True,
    )
    with patch.object(
        item.results, "get_projection_trajectories", wraps=item.results.get_projection_trajectories
    ) as stack:
        _, actual = _fetch_quantiles_for_location(item, config, False, True)
    stack.assert_called_once_with("baseline", variables=["hospitalizations"])
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    with patch.object(item.results, "get_calibration_trajectories", side_effect=AssertionError("unnecessary stack")):
        assert _compute_fitting_window(item, None) == (date(2024, 1, 6), date(2024, 2, 10))


def test_hub_uses_batched_projection_quantiles():
    """Compute Hub projection levels once while preserving California FIPS and all five forecast horizons."""
    item = calibration()
    config = OutputConfig.model_validate(
        {
            "output": {
                "flusight_format": {
                    "reference_date": "2024-01-13",
                    "quantiles": [0.25, 0.5, 0.75],
                }
            }
        }
    )
    with patch("epymodelingsuite.utils.quantiles.np.nanquantile", wraps=np.nanquantile) as compute:
        output = generate_calibration_outputs(calibrations=[item], output_config=config)
    assert compute.call_count == 1
    hub = output["output_hub_formatted"][0].data
    assert len(hub) == 15
    assert hub.location.unique().tolist() == ["06"]
    assert hub.horizon.unique().tolist() == [-1, 0, 1, 2, 3]


@pytest.mark.parametrize("selected", [["I", "S"], ["missing"], True])
def test_simulation_selection_and_fallback(selected):
    """Check simulation variable selection before stacking and exact tables for selected/all/missing names."""
    dates = pd.date_range("2024-01-06", periods=3, freq="W-SAT")
    trajectories = [
        Trajectory(
            compartments={"S": np.arange(3) + i, "I": np.arange(3) + i + 10},
            transitions={"S_to_I": np.arange(3) + i + 20},
            dates=dates,
            compartment_idx={},
            transitions_idx={},
            parameters={},
        )
        for i in range(3)
    ]
    results = SimulationResults(trajectories, {})
    item = SimulationOutput(primary_id=8, population="United_States_California", results=results)
    config = OutputConfig.model_validate(
        {
            "output": {
                "quantiles": {"selections": [0.25, 0.75], "compartments": selected, "transitions": True},
            }
        }
    )
    with patch.object(results, "get_stacked_compartments", wraps=results.get_stacked_compartments) as stack:
        output = generate_simulation_outputs(simulations=[item], output_config=config)
    variables = selected if selected == ["I", "S"] else None
    assert stack.call_args_list[0].kwargs == {"variables": variables}
    for kind in ["compartments", "transitions"]:
        expected = getattr(results, f"get_quantiles_{kind}")(quantiles=[0.25, 0.75], ignore_nan=True)
        if kind == "compartments" and variables:
            expected = expected[["date", "quantile", *variables]]
        expected.insert(0, "primary_id", 8)
        expected.insert(1, "seed", None)
        expected.insert(2, "population", item.population)
        pd.testing.assert_frame_equal(output[f"quantiles_{kind}"][0].data, expected, check_exact=True)
