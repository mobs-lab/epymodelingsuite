"""Regression tests for shared, non-mutating quantile plot preparation."""

from collections import Counter
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from quantile_plot_test_utils import make_calibration, make_plots_config, write_surveillance_sources

from epymodelingsuite.visualization import generators, preparation

CA = "United_States_California"
NY = "United_States_New_York"
TX = "United_States_Texas"


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_single_and_grid_share_preparation_and_leave_inputs_unchanged(tmp_path, monkeypatch):
    """Each source/location/panel is prepared once; both layouts reuse it without changing inputs."""
    calibrations = [make_calibration(CA), make_calibration(NY)]
    results = calibrations[0].results
    valid = results.projections["baseline"]
    trajectories = [valid[0], {}, *valid[1:], {}]
    results.projections["baseline"] = trajectories
    parameters = pd.DataFrame({"R0": np.arange(7.0)})
    results.projection_parameters = {"baseline": parameters}
    results._filtered_count = 9  # Existing metadata must not be reset by plotting.
    arrays_before = [{key: np.array(value, copy=True) for key, value in traj.items()} for traj in trajectories]
    parameters_before = parameters.copy()
    sources = write_surveillance_sources(tmp_path)
    config = make_plots_config(
        single=True,
        grid=True,
        outputs=[
            {"type": variant, "show_calibration": True, "surveillance_source": "hosp"}
            for variant in ("filtered", "full", "side_by_side")
        ],
    )
    config_before = config.model_dump()
    frames_before = {}
    panels = {}
    calls = Counter()
    draw_panel = generators._draw_quantile_panel

    def capture_draw(panel, ax, ylabel):
        key = id(panel)
        calls[key] += 1
        if key not in panels:
            panels[key] = panel
            frames_before[key] = [
                None if frame is None else frame.copy(deep=True)
                for frame in (panel.calibration_quantiles, panel.projection_quantiles, panel.surveillance)
            ]
        draw_panel(panel, ax, ylabel)

    monkeypatch.setattr(generators, "_draw_quantile_panel", capture_draw)
    with (
        patch.object(preparation.pd, "read_csv", wraps=pd.read_csv) as read_csv,
        patch.object(
            preparation, "_collect_location_plot_data", wraps=preparation._collect_location_plot_data
        ) as collect,
        patch.object(
            preparation, "prepare_panel_plot_data", wraps=preparation.prepare_panel_plot_data
        ) as prepare_panel,
    ):
        outputs = {}
        generators.generate_quantile_plots(calibrations, config, outputs, sources)
    source_paths = {source.data_path for source in sources.values()}
    # Location-name conversion also reads a codebook; count only the configured sources.
    source_reads = Counter(str(call.args[0]) for call in read_csv.call_args_list if str(call.args[0]) in source_paths)
    assert source_reads == dict.fromkeys(source_paths, 1)
    assert collect.call_count == 2  # Two locations, shared by all variants and layouts.
    assert prepare_panel.call_count == 8  # Two locations * (filtered + full + two side-by-side panels).
    assert len(outputs) == 9  # Six single figures and three grids.
    assert len(calls) == 8 and set(calls.values()) == {2}
    assert calibrations[0].results is results
    assert results.projections["baseline"] is trajectories
    assert len(trajectories) == 7
    assert results.projection_parameters["baseline"] is parameters
    pd.testing.assert_frame_equal(parameters, parameters_before)
    assert results._filtered_count == 9
    for before, after in zip(arrays_before, trajectories, strict=True):
        assert before.keys() == after.keys()
        for name in before:
            np.testing.assert_array_equal(before[name], after[name])
    assert config.model_dump() == config_before
    for key, panel in panels.items():
        for before, after in zip(
            frames_before[key],
            (panel.calibration_quantiles, panel.projection_quantiles, panel.surveillance),
            strict=True,
        ):
            if before is None:
                assert after is None
            else:
                pd.testing.assert_frame_equal(before, after)


@pytest.mark.parametrize("grid", [False, {"enabled": False}])
def test_disabled_layouts_skip_all_preparation(grid):
    with patch.object(generators, "prepare_quantile_plots") as prepare:
        outputs = {}
        generators.generate_quantile_plots([make_calibration(CA)], make_plots_config(single=False, grid=grid), outputs)
    prepare.assert_not_called()
    assert outputs == {}


@pytest.mark.parametrize("columns", [2, 4, 6])
def test_grid_keeps_every_prepared_pair_visible(columns):
    config = make_plots_config(grid={"panels_per_row": columns}, outputs=[{"type": "side_by_side"}])
    outputs = {}
    generators.generate_quantile_plots([make_calibration(p) for p in (CA, NY, TX)], config, outputs)
    fig = outputs["quantiles_grid_sidebyside"][0].data
    assert [ax.get_title() for ax in fig.axes if ax.axison] == [
        "California",
        "California",
        "New York",
        "New York",
        "Texas",
        "Texas",
    ]
    assert all(not ax.axison for ax in fig.axes[6:])


@pytest.mark.parametrize(
    ("generate", "expected"),
    [
        (generators.generate_single_quantile_plots, {f"quantiles_{CA}_full"}),
        (generators.generate_quantile_grid_plot, {"quantiles_grid_full"}),
    ],
)
def test_layout_entrypoints_keep_their_scope_and_do_not_modify_config(generate, expected):
    config = make_plots_config(single=True, grid=True, outputs=[{"type": "full"}])
    before = config.model_dump()
    outputs = {}
    generate([make_calibration(CA)], config, outputs)
    assert set(outputs) == expected
    assert config.model_dump() == before


def test_drawing_failure_closes_figure_and_other_outputs_continue(monkeypatch):
    draw = generators.plot_quantile_panel
    calls = 0

    def fail_first(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("Cannot draw first panel")
        return draw(**kwargs)

    monkeypatch.setattr(generators, "plot_quantile_panel", fail_first)
    outputs = {}
    generators.generate_quantile_plots([make_calibration(CA)], make_plots_config(single=True, grid=True), outputs)
    assert f"quantiles_{CA}_filtered" not in outputs
    assert len(outputs) == 5
    assert plt.get_fignums() == []
