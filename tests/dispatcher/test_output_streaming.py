"""Saving uses the legacy kernels while bounding raw result/trajectory memory."""

import gzip
import subprocess
import sys
import weakref
from copy import deepcopy
from unittest.mock import patch

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from epymodelingsuite.dispatcher.output import generate_calibration_outputs
from epymodelingsuite.output.streaming import OutputWriteError, write_outputs
from epymodelingsuite.schema.output import OutputConfig

from .test_output_quantiles import calibration


def config(plots=False, prop=False):
    """Build the requested file/table/plot configuration for save-path comparisons."""
    output = {
        "tabular_output_types": ["CSVBytes"],
        "quantiles": {"compartments": True, "transitions": True, "calibration": True, "selections": [0.75, 0.25]},
        "trajectories": {"compartments": True, "transitions": True},
        "flusight_format": {"reference_date": "2024-01-13", "quantiles": [0.25, 0.5, 0.75]},
    }
    if prop:
        output["flusight_format"]["prop_ed"] = {"strategy": "transition", "transition_name": "I_to_R"}
    if plots:
        output["plots"] = {
            "reference_date": "2024-01-13",
            "figure_output_types": ["PNG", "PDF", "SVG"],
            "quantiles": {"single": True, "grid": True, "outputs": [{"type": "full", "show_surveillance": False}]},
        }
    return OutputConfig.model_validate({"output": output})


def compare_saved(expected, saved):
    """Compare saved CSV contents and figure bytes with the legacy dictionary-return API."""
    assert set(expected) == set(saved)
    for name, objects in expected.items():
        for obj in objects:
            path = next(p for p in saved[name] if p.name == obj.name)
            if obj.name.endswith(".csv.gz"):
                # CSV text comparison also detects identifier coercion and ordering.
                assert gzip.decompress(path.read_bytes()) == gzip.decompress(obj.data), name
            else:
                assert path.stat().st_size > 100
                if obj.name.endswith(".png"):
                    assert path.read_bytes() == obj.data, name


@pytest.mark.parametrize("prop", [False, True])
def test_streaming_matches_all_tables_and_preserves_heterogeneous_columns(tmp_path, prop):
    """Match legacy CSV contents across heterogeneous columns and prop ED strategies, then remove staging files."""
    items = [calibration(), calibration()]
    items[1].primary_id = 8
    items[1].seed = None
    items[1].population = "United_States_Texas"
    for draw in items[1].results.projections["baseline"]:
        draw["extra"] = draw["S"] + 100
    expected = generate_calibration_outputs(calibrations=deepcopy(items), output_config=config(prop=prop))
    saved = write_outputs(results=iter(items), output_config=config(prop=prop), directory=tmp_path, chunk_rows=5)
    compare_saved(expected, saved)
    assert not list(tmp_path.glob(".output-*"))


def test_figures_grid_formats_duplicate_locations_and_no_live_figures(tmp_path):
    """Preserve single/grid images in each format and duplicate-location overwrite behavior without leaking figures."""
    items = [calibration(), calibration()]
    items[1].primary_id = 8
    expected = generate_calibration_outputs(calibrations=deepcopy(items), output_config=config(plots=True))
    before = set(plt.get_fignums())
    saved = write_outputs(results=iter(items), output_config=config(plots=True), directory=tmp_path)
    compare_saved(expected, saved)
    assert set(plt.get_fignums()) == before


def test_input_is_lazy_and_processed_results_are_released(tmp_path):
    """Consume one raw result at a time and release processed objects before requesting the next input."""
    refs = []

    def inputs():
        for _ in range(3):
            assert all(ref() is None for ref in refs)
            item = calibration()
            refs.append(weakref.ref(item.results))
            yield item
            del item

    write_outputs(results=inputs(), output_config=config(), directory=tmp_path, chunk_rows=5)
    assert all(ref() is None for ref in refs)


def test_trajectories_do_not_use_stacks(tmp_path):
    """Write all trajectory rows without stack APIs and keep every staged frame within the row-chunk limit."""
    item = calibration()
    settings = OutputConfig.model_validate(
        {
            "output": {
                "tabular_output_types": ["CSVBytes"],
                "trajectories": {"compartments": True, "transitions": True},
            }
        }
    )
    with patch.object(item.results, "get_projection_trajectories", side_effect=AssertionError("no stack")):
        saved = write_outputs(results=[item], output_config=settings, directory=tmp_path, chunk_rows=3)
    assert len(pd.read_csv(saved["trajectories_projection_compartments"][0])) == 24


def test_invalid_format_is_rejected_before_consuming_input(tmp_path):
    """Reject unsupported table/figure formats before advancing the input iterator."""

    def inputs():
        raise AssertionError("must not consume")
        yield

    with pytest.raises(ValueError, match="CSVBytes"):
        write_outputs(results=inputs(), output_config=OutputConfig.model_validate({"output": {}}), directory=tmp_path)


def test_publish_failure_reports_completed_files_and_cleans_staging(tmp_path):
    """Expose only successfully published files after a later replacement fails and clean all temporary shards."""
    from epymodelingsuite.output import streaming

    replace = streaming.os.replace
    calls = []

    def fail_second(source, target):
        calls.append(target)
        if len(calls) == 2:
            raise OSError("disk failed")
        replace(source, target)

    with patch.object(streaming.os, "replace", side_effect=fail_second), pytest.raises(OutputWriteError) as caught:
        write_outputs(results=[calibration()], output_config=config(), directory=tmp_path)
    assert "disk failed" in str(caught.value)
    assert sum(map(len, caught.value.completed.values())) == 1
    assert list(tmp_path.iterdir()) == [calls[0]]


def test_plot_write_failure_is_not_swallowed(tmp_path):
    """Propagate figure-file write errors despite legacy warning handlers, and clean staged files and figures."""
    from pathlib import Path

    with (
        patch.object(Path, "write_bytes", side_effect=OSError("image disk failed")),
        pytest.raises(OutputWriteError, match="image disk failed"),
    ):
        write_outputs(results=[calibration()], output_config=config(plots=True), directory=tmp_path)
    assert not list(tmp_path.iterdir())
    assert not plt.get_fignums()


@pytest.mark.parametrize("strategy", ["surveillance_window", "calibration_window"])
def test_global_prop_ed_trends_metadata_and_surveillance_reuse(tmp_path, strategy):
    """Match global prop ED, rate trends and merged metadata while sharing parser-compatible surveillance reads."""
    settings = config(plots=True).model_dump()
    surveillance_path = tmp_path / "observations.csv"
    observations = pd.DataFrame(
        {
            "location": ["06"] * 6 + ["48"] * 6,
            "date": list(pd.date_range("2024-01-06", periods=6, freq="W-SAT")) * 2,
            "hospitalizations": list(range(40, 46)) * 2,
            "ed": [0.02] * 12,
        }
    )
    observations.to_csv(surveillance_path, index=False)
    common = dict(
        data_path=str(surveillance_path), date_column="date", location_column="location", location_format="FIPS"
    )
    settings["output"]["options"] = {
        "surveillance": {
            "hosp": dict(**common, value_column="hospitalizations"),
            "ed": dict(**common, value_column="ed"),
        }
    }
    hub = settings["output"]["flusight_format"]
    hub["rate_trends_source"] = "hosp"
    prop = {"strategy": strategy, "ed_source": "ed"}
    if strategy == "surveillance_window":
        prop.update(hosp_source="hosp", fit_start="2024-01-06", fit_end="2024-02-11")
    else:
        prop.update(num_fit_weeks=4)
    hub["prop_ed"] = prop
    # Plot read options deliberately differ from hub parsing; each representation
    # is read once, rather than silently changing inference for compatibility.
    settings["output"]["plots"]["quantiles"]["outputs"][0].update(show_surveillance=True, surveillance_source="hosp")
    settings["output"]["plots"]["categorical"] = {}
    settings = OutputConfig.model_validate(settings)
    items = [calibration(), calibration(), calibration()]
    for item in items:
        item.population = "United_States__California"
    items[1].population = "United_States__Texas"
    items[1].primary_id = 8
    items[2].primary_id = 9  # Duplicate population must share the global rescaling fit.
    expected = generate_calibration_outputs(calibrations=deepcopy(items), output_config=settings)
    with patch("epymodelingsuite.utils.surveillance.pd.read_csv", wraps=pd.read_csv) as read:
        saved = write_outputs(results=iter(items), output_config=settings, directory=tmp_path / "saved")
    compare_saved(expected, saved)
    assert sum(str(call.args[0]) == str(surveillance_path) for call in read.call_args_list) == 2
    assert "rescaling_factor" in pd.read_csv(saved["model_metadata"][0])


def test_posteriors_generations_failed_projections_and_lazy_grid(tmp_path):
    """Match filtered posterior outputs and release each grid sample frame before loading the next location."""
    import numpy as np

    from epymodelingsuite.output.streaming import _PosteriorFiles

    items = [calibration(), calibration()]
    items[1].population = "United_States_Texas"
    for item in items:
        item.results.posterior_distributions = {
            0: pd.DataFrame({"R0": np.linspace(1, 2, 4), "start_date": [0, 1, 3, 4]}),
            1: pd.DataFrame({"R0": np.linspace(2, 3, 4), "start_date": [4, 6, 7, 8]}),
        }
        item.results.selected_trajectories[1] = deepcopy(item.results.selected_trajectories[0])
        item.start_date_reference = pd.Timestamp("2024-01-01").date()
        item.results.projections["baseline"].insert(1, {})
        item.results.projection_parameters = {"baseline": pd.DataFrame({"beta": [1, 2, 3, 4, 5]})}
    settings = config(plots=True)
    from epymodelingsuite.schema.output import PosteriorsOutput

    settings.output.posteriors = PosteriorsOutput(generations=[0, 1])
    settings.output.model_meta.projection_parameters = True
    settings.output.plots.posterior.single = True
    settings.output.plots.posterior.grid = True
    settings.output.quantiles.calibration = [0, 1]
    expected = generate_calibration_outputs(calibrations=deepcopy(items), output_config=settings)
    seen = []
    samples = []
    load = _PosteriorFiles.__getitem__

    def record(mapping, key):
        assert all(ref() is None for ref in samples)
        seen.append(key)
        frame = load(mapping, key)
        samples.append(weakref.ref(frame))
        return frame

    with patch.object(_PosteriorFiles, "__getitem__", record):
        saved = write_outputs(results=iter(items), output_config=settings, directory=tmp_path)
    compare_saved(expected, saved)
    assert seen == [item.population for item in items]
    assert "posterior_grid" in saved


def test_simulation_tables_match_with_missing_columns_and_seeds(tmp_path):
    """Match legacy simulation CSV schema and metadata inference for missing columns and mixed seed values."""
    import numpy as np
    from epydemix.model.simulation_output import Trajectory
    from epydemix.model.simulation_results import SimulationResults

    from epymodelingsuite.dispatcher.output import generate_simulation_outputs
    from epymodelingsuite.schema.dispatcher import SimulationOutput

    items = []
    for i in range(2):
        dates = pd.date_range("2024-01-06", periods=3)
        trajectories = [
            Trajectory(
                compartments={"S": np.arange(3) + j},
                transitions={"S_to_I": np.arange(3) + 10 + j},
                dates=dates,
                parameters={},
                compartment_idx={},
                transitions_idx={},
            )
            for j in range(3)
        ]
        items.append(
            SimulationOutput(
                primary_id=i,
                seed=13 if i == 0 else None,
                population="United_States_California",
                results=SimulationResults(trajectories, {}),
            )
        )
    settings = OutputConfig.model_validate(
        {
            "output": {
                "tabular_output_types": ["CSVBytes"],
                "quantiles": {"compartments": True, "transitions": True},
                "trajectories": {"compartments": True, "transitions": True},
            }
        }
    )
    expected = generate_simulation_outputs(simulations=deepcopy(items), output_config=settings)
    saved = write_outputs(results=iter(items), output_config=settings, directory=tmp_path, chunk_rows=2)
    compare_saved(expected, saved)


def test_empty_input_has_no_outputs(tmp_path):
    """Return no paths and leave no temporary files for an empty result iterator."""
    assert write_outputs(results=iter(()), output_config=config(plots=True), directory=tmp_path) == {}
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("first", ["output", "dispatcher"])
def test_output_import_order_preserves_dispatcher_exports(first):
    """Import either package first in a fresh process and preserve dispatcher API aliases."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            f"""
import importlib
importlib.import_module("epymodelingsuite." + {first!r})
from epymodelingsuite.output.streaming import OutputWriteError, write_outputs
from epymodelingsuite import dispatcher
assert dispatcher.write_outputs is write_outputs
assert dispatcher.OutputWriteError is OutputWriteError
assert write_outputs.__module__ == "epymodelingsuite.output.streaming"
""",
        ],
        check=True,
    )
