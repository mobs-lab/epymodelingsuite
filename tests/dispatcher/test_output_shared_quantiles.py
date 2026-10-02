"""Sharing must preserve requests and must never mix distinct result objects."""

from unittest.mock import patch

import numpy as np
import pandas as pd

from epymodelingsuite.dispatcher.output import generate_calibration_outputs
from epymodelingsuite.schema.output import OutputConfig
from epymodelingsuite.utils.quantiles import SharedQuantiles, get_projection_quantiles
from epymodelingsuite.visualization import generators

from .test_output_quantiles import calibration


def test_shared_levels_and_variables_preserve_order_duplicates_and_copies():
    """Share compatible computation while preserving duplicate/unsorted levels and independent consumer frames.
    Distinct result objects must not share quantile values.
    """
    item = calibration()
    other = calibration()
    other.results.projections["baseline"][0]["S"] += 100
    prepared = SharedQuantiles()
    requests = [
        (item.results, dict(quantiles=[0.75, 0.25, 0.75], variables=["I", "S"])),
        (item.results, dict(quantiles=[0.5, 0.25], variables=["S"])),
        (other.results, dict(quantiles=[0.25], variables=["S"])),
    ]
    for results, kwargs in requests:
        prepared.add(results, "projection", **kwargs)
    with patch("epymodelingsuite.utils.quantiles.np.quantile", wraps=np.quantile) as compute:
        actual = [prepared.projection(results, **kwargs) for results, kwargs in requests]
    assert compute.call_count == 3  # Two variables for item, one for other.
    for frame, (results, kwargs) in zip(actual, requests, strict=True):
        pd.testing.assert_frame_equal(frame, get_projection_quantiles(results, **kwargs), check_exact=True)
    actual[0].loc[:, "S"] = -1
    pd.testing.assert_frame_equal(prepared.projection(item.results, **requests[1][1]), actual[1])


def test_tabular_hub_single_and_grid_share_one_stack_and_preparation():
    """Require tables, Hub, single plots and grid plots to share one stack, computation and plot-data preparation."""
    item = calibration()
    config = OutputConfig.model_validate(
        {
            "output": {
                "quantiles": {"compartments": ["hospitalizations"], "selections": [0.75, 0.25]},
                "flusight_format": {"reference_date": "2024-01-13", "quantiles": [0.25, 0.5, 0.75]},
                "plots": {
                    "reference_date": "2024-01-13",
                    "quantiles": {
                        "single": True,
                        "grid": True,
                        "quantiles": [0.25, 0.5, 0.75],
                        "outputs": [{"type": "full", "show_surveillance": False}],
                    },
                },
            }
        }
    )
    with (
        patch.object(
            item.results, "get_projection_trajectories", wraps=item.results.get_projection_trajectories
        ) as stack,
        patch("epymodelingsuite.utils.quantiles.np.nanquantile", wraps=np.nanquantile) as compute,
        patch(
            "epymodelingsuite.visualization.generators._collect_location_plot_data",
            wraps=generators._collect_location_plot_data,
        ) as prepare,
    ):
        output = generate_calibration_outputs(calibrations=[item], output_config=config)
    assert stack.call_count == compute.call_count == prepare.call_count == 1
    assert {"quantiles_United_States_California_full", "quantiles_grid_full", "output_hub_formatted"} <= output.keys()
    quantiles = output["quantiles_projection_compartments"][0].data
    assert quantiles["quantile"].drop_duplicates().tolist() == [0.75, 0.25]


def test_iso_single_without_grid_uses_shared_preparation():
    """Resolve an ISO single-location selection when the grid is disabled and stack its target exactly once."""
    item = calibration()
    config = OutputConfig.model_validate(
        {
            "output": {
                "plots": {
                    "reference_date": "2024-01-13",
                    "quantiles": {
                        "single": ["US-CA"],
                        "grid": False,
                        "outputs": [{"type": "full", "show_surveillance": False}],
                    },
                },
            }
        }
    )
    with patch.object(
        item.results, "get_projection_trajectories", wraps=item.results.get_projection_trajectories
    ) as stack:
        outputs = generate_calibration_outputs(calibrations=[item], output_config=config)
    assert "quantiles_United_States_California_full" in outputs
    stack.assert_called_once_with("baseline", variables=["hospitalizations"])
