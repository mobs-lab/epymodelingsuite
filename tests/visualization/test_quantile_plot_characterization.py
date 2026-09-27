"""Characterization tests for quantile plots: {single, grid} x {filtered, full, side_by_side}.

These pin down what each drawn axis receives today, so the override refactor can be checked against them.
Assertions marked ``CURRENT BEHAVIOR`` capture drift that the refactor changes on purpose
(see plot-override-refactor-plan.md); everything else should stay the same.

Frames are summarized with ``span()`` as (first date, last date, number of dates, location tag).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from quantile_plot_harness import (
    PlotCapture,
    make_calibration,
    make_plots_config,
    span,
    write_surveillance_sources,
)

from epymodelingsuite.dispatcher.output import generate_calibration_outputs
from epymodelingsuite.schema.output import (
    ModelMetaOutput,
    OutputConfig,
    OutputConfiguration,
    OutputOptions,
    QuantilesOutputConfig,
    TabularOutputTypeEnum,
)
from epymodelingsuite.visualization.generators import generate_quantile_grid_plot, generate_single_quantile_plots

CA = "United_States_California"
NY = "United_States_New_York"
TX = "United_States_Texas"

FITTING_WINDOW = ("2023-11-04", "2024-01-13")

# Frames shared by many panels (tag 1 = California)
CAL_FULL = ("2023-11-04", "2024-01-13", 11, 1)
CAL_FROM_PROJECTION_START = ("2023-12-02", "2024-01-13", 7, 1)
PROJ_BASE_HORIZON = ("2023-12-02", "2024-02-03", 10, 1)  # reference_date + 3 weeks
SURV_ALL = ("2023-10-07", "2024-01-27", 17, 1)
SURV_FROM_PROJECTION_START = ("2023-12-02", "2024-01-27", 9, 1)


def with_tag(summary: tuple, tag: int) -> tuple:
    return (*summary[:3], tag)


def panel(title, calibration, projection, surveillance):
    return {"title": title, "calibration": calibration, "projection": projection, "surveillance": surveillance}


def run_generators(calibrations, plots_config, surveillance_sources):
    out_dict = {}
    generate_single_quantile_plots(calibrations, plots_config, out_dict, surveillance_sources)
    generate_quantile_grid_plot(calibrations, plots_config, out_dict, surveillance_sources)
    return out_dict


def visible_axes(fig) -> int:
    return sum(ax.axison for ax in fig.axes)


@pytest.fixture
def capture(monkeypatch):
    yield PlotCapture(monkeypatch)
    plt.close("all")


@pytest.fixture
def sources(tmp_path):
    return write_surveillance_sources(tmp_path)


@pytest.fixture
def hosp_only(sources):
    return {"hosp": sources["hosp"]}


def all_outputs_with_calibration():
    return [
        QuantilesOutputConfig(type="filtered", surveillance_points=8, show_calibration=True),
        QuantilesOutputConfig(type="full", show_calibration=True),
        QuantilesOutputConfig(type="side_by_side", show_calibration=True),
    ]


class TestSingleDefaultOutputs:
    @pytest.fixture
    def out_dict(self, capture, hosp_only):
        config = make_plots_config(single=True, grid=False, outputs=all_outputs_with_calibration(), ylabel="Hosp")
        return run_generators([make_calibration(CA), make_calibration(NY)], config, hosp_only)

    def test_output_keys_and_panel_counts(self, out_dict, capture):
        assert sorted(out_dict) == sorted(
            f"quantiles_{location}_{kind}" for location in (CA, NY) for kind in ("filtered", "full", "side_by_side")
        )
        for location in (CA, NY):
            assert len(capture.panels_for(f"quantiles_{location}_filtered")) == 1
            assert len(capture.panels_for(f"quantiles_{location}_full")) == 1
            assert len(capture.panels_for(f"quantiles_{location}_side_by_side")) == 2
        assert all(panel.output is not None for panel in capture.panels)

    def test_filtered(self, out_dict, capture):
        # CURRENT BEHAVIOR (decision 8): surveillance_points=8 is counted after the projection start,
        # so only 7 points on or before reference_date are shown.
        assert capture.summary(f"quantiles_{CA}_filtered") == [
            panel("California", CAL_FROM_PROJECTION_START, PROJ_BASE_HORIZON, SURV_FROM_PROJECTION_START)
        ]
        assert capture.summary(f"quantiles_{NY}_filtered") == [
            panel(
                "New York",
                with_tag(CAL_FROM_PROJECTION_START, 2),
                with_tag(PROJ_BASE_HORIZON, 2),
                with_tag(SURV_FROM_PROJECTION_START, 2),
            )
        ]

    def test_full(self, out_dict, capture):
        assert capture.summary(f"quantiles_{CA}_full") == [panel("California", CAL_FULL, PROJ_BASE_HORIZON, SURV_ALL)]

    def test_side_by_side(self, out_dict, capture):
        assert capture.summary(f"quantiles_{CA}_side_by_side") == [
            panel("California", CAL_FULL, PROJ_BASE_HORIZON, SURV_ALL),
            panel("California", CAL_FROM_PROJECTION_START, PROJ_BASE_HORIZON, SURV_FROM_PROJECTION_START),
        ]

    def test_styling(self, out_dict, capture):
        for drawn in capture.panels:
            assert drawn.fitting_window == FITTING_WINDOW
            assert (drawn.calibration_color, drawn.projection_color) == ("C0", "C1")
            assert drawn.xlabel_interval is None
        # side_by_side only labels the left panel
        assert [drawn.ylabel for drawn in capture.panels_for(f"quantiles_{CA}_side_by_side")] == ["Hosp", None]
        assert capture.panels_for(f"quantiles_{CA}_filtered")[0].ylabel == "Hosp"


class TestGridDefaultOutputs:
    """Three locations, one calibration-only, on a partly filled 4-column grid."""

    @pytest.fixture
    def out_dict(self, capture, hosp_only):
        config = make_plots_config(
            single=False, grid={"panels_per_row": 4}, outputs=all_outputs_with_calibration(), ylabel="Hosp"
        )
        calibrations = [make_calibration(CA), make_calibration(NY), make_calibration(TX, with_projection=False)]
        return run_generators(calibrations, config, hosp_only)

    def test_output_keys_and_layout(self, out_dict, capture):
        assert sorted(out_dict) == ["quantiles_grid_filtered", "quantiles_grid_full", "quantiles_grid_sidebyside"]
        for name, n_panels, n_axes in (
            ("quantiles_grid_filtered", 3, 4),
            ("quantiles_grid_full", 3, 4),
            ("quantiles_grid_sidebyside", 6, 8),
        ):
            fig = capture.packaged[name]
            assert len(capture.panels_for(name)) == n_panels
            assert len(fig.axes) == n_axes
            assert visible_axes(fig) == n_panels

    def test_filtered(self, out_dict, capture):
        # CURRENT BEHAVIOR (decision 6): Texas has no projection, so it gets no surveillance.
        assert capture.summary("quantiles_grid_filtered") == [
            panel("California", CAL_FROM_PROJECTION_START, PROJ_BASE_HORIZON, SURV_FROM_PROJECTION_START),
            panel(
                "New York",
                with_tag(CAL_FROM_PROJECTION_START, 2),
                with_tag(PROJ_BASE_HORIZON, 2),
                with_tag(SURV_FROM_PROJECTION_START, 2),
            ),
            panel("Texas", with_tag(CAL_FULL, 3), None, None),
        ]

    def test_full(self, out_dict, capture):
        assert capture.summary("quantiles_grid_full") == [
            panel("California", CAL_FULL, PROJ_BASE_HORIZON, SURV_ALL),
            panel("New York", with_tag(CAL_FULL, 2), with_tag(PROJ_BASE_HORIZON, 2), with_tag(SURV_ALL, 2)),
            panel("Texas", with_tag(CAL_FULL, 3), None, None),
        ]

    def test_side_by_side(self, out_dict, capture):
        assert capture.summary("quantiles_grid_sidebyside") == [
            panel("California", CAL_FULL, PROJ_BASE_HORIZON, SURV_ALL),
            panel("California", CAL_FROM_PROJECTION_START, PROJ_BASE_HORIZON, SURV_FROM_PROJECTION_START),
            panel("New York", with_tag(CAL_FULL, 2), with_tag(PROJ_BASE_HORIZON, 2), with_tag(SURV_ALL, 2)),
            panel(
                "New York",
                with_tag(CAL_FROM_PROJECTION_START, 2),
                with_tag(PROJ_BASE_HORIZON, 2),
                with_tag(SURV_FROM_PROJECTION_START, 2),
            ),
            panel("Texas", with_tag(CAL_FULL, 3), None, None),
            panel("Texas", with_tag(CAL_FULL, 3), None, None),
        ]

    def test_styling(self, out_dict, capture):
        for drawn in capture.panels:
            assert drawn.fitting_window == FITTING_WINDOW
            assert (drawn.calibration_color, drawn.projection_color) == ("C0", "C1")
        # ylabel only on the leftmost column
        assert [drawn.ylabel for drawn in capture.panels_for("quantiles_grid_filtered")] == ["Hosp", None, None]
        # two location pairs per row: Texas starts the second row
        assert [drawn.ylabel for drawn in capture.panels_for("quantiles_grid_sidebyside")] == [
            "Hosp",
            None,
            None,
            None,
            "Hosp",
            None,
        ]


def override_config(**kwargs):
    outputs = [
        QuantilesOutputConfig(
            type="filtered",
            show_calibration=True,
            surveillance_start_date="2023-12-16",
            horizon_max=1,
            xlabel_interval="2W-SAT",
        ),
        QuantilesOutputConfig(type="full", show_calibration=True, surveillance_points=4, horizon_max=6),
        QuantilesOutputConfig(
            type="side_by_side",
            show_calibration=True,
            surveillance_points=3,
            horizon_max=1,
            full_panel={"xlabel_interval": "MS", "surveillance_start_date": "2023-11-01"},
            filtered_panel={"surveillance_points": 2, "xlabel_interval": "W-SAT"},
        ),
    ]
    return make_plots_config(
        outputs=outputs, calibration={"color": "tab:green"}, projection={"color": "tab:red"}, ylabel="Hosp", **kwargs
    )


# filtered output: surveillance_start_date=2023-12-16, horizon_max=1
OVERRIDE_FILTERED = panel(
    "California",
    ("2023-12-16", "2024-01-13", 5, 1),
    ("2023-12-16", "2024-01-20", 6, 1),
    ("2023-12-16", "2024-01-27", 7, 1),
)
# full output: surveillance_points=4 (4 on or before reference_date + 2 after)
# CURRENT BEHAVIOR (R24-2): horizon_max=6 is capped at the base horizon_max=3.
OVERRIDE_FULL = panel("California", CAL_FULL, PROJ_BASE_HORIZON, ("2023-12-23", "2024-01-27", 6, 1))


class TestSingleOverrides:
    @pytest.fixture
    def out_dict(self, capture, hosp_only):
        return run_generators([make_calibration(CA)], override_config(single=True, grid=False), hosp_only)

    def test_filtered_and_full(self, out_dict, capture):
        assert capture.summary(f"quantiles_{CA}_filtered") == [OVERRIDE_FILTERED]
        assert capture.summary(f"quantiles_{CA}_full") == [OVERRIDE_FULL]

    def test_side_by_side(self, out_dict, capture):
        # CURRENT BEHAVIOR: single side_by_side ignores output and panel surveillance limits
        # and the output horizon_max; it only uses the panel xlabel_interval.
        assert capture.summary(f"quantiles_{CA}_side_by_side") == [
            panel("California", CAL_FULL, PROJ_BASE_HORIZON, SURV_ALL),
            panel("California", CAL_FROM_PROJECTION_START, PROJ_BASE_HORIZON, SURV_FROM_PROJECTION_START),
        ]

    def test_styling(self, out_dict, capture):
        assert [drawn.xlabel_interval for drawn in capture.panels] == ["2W-SAT", None, "MS", "W-SAT"]
        for drawn in capture.panels:
            assert (drawn.calibration_color, drawn.projection_color) == ("tab:green", "tab:red")


class TestGridOverrides:
    @pytest.fixture
    def out_dict(self, capture, hosp_only):
        calibrations = [make_calibration(CA), make_calibration(NY)]
        return run_generators(calibrations, override_config(single=False, grid={"panels_per_row": 4}), hosp_only)

    def test_filtered_and_full(self, out_dict, capture):
        assert capture.summary("quantiles_grid_filtered")[0] == OVERRIDE_FILTERED
        assert capture.summary("quantiles_grid_full")[0] == OVERRIDE_FULL
        assert capture.summary("quantiles_grid_full")[1]["surveillance"] == ("2023-12-23", "2024-01-27", 6, 2)

    def test_side_by_side(self, out_dict, capture):
        # CURRENT BEHAVIOR: grid side_by_side uses panel limits only; the output surveillance_points=3
        # and output horizon_max=1 are ignored.
        assert capture.summary("quantiles_grid_sidebyside")[:2] == [
            panel("California", CAL_FULL, PROJ_BASE_HORIZON, ("2023-11-04", "2024-01-27", 13, 1)),
            panel(
                "California",
                ("2024-01-06", "2024-01-13", 2, 1),
                ("2024-01-06", "2024-02-03", 5, 1),
                ("2024-01-06", "2024-01-27", 4, 1),
            ),
        ]

    def test_styling(self, out_dict, capture):
        assert [drawn.xlabel_interval for drawn in capture.panels_for("quantiles_grid_filtered")] == ["2W-SAT"] * 2
        assert [drawn.xlabel_interval for drawn in capture.panels_for("quantiles_grid_full")] == [None] * 2
        assert [drawn.xlabel_interval for drawn in capture.panels_for("quantiles_grid_sidebyside")] == [
            "MS",
            "W-SAT",
        ] * 2
        for drawn in capture.panels:
            assert (drawn.calibration_color, drawn.projection_color) == ("tab:green", "tab:red")


class TestSurveillanceSourceSelection:
    """Two sources loaded; every output selects the second one ("alt", tag + 5)."""

    @pytest.fixture
    def config(self):
        outputs = [
            QuantilesOutputConfig(type=kind, surveillance_source="alt") for kind in ("filtered", "full", "side_by_side")
        ]
        return make_plots_config(single=True, grid=True, outputs=outputs)

    def test_single_uses_first_loaded_source(self, capture, sources, config):
        run_generators([make_calibration(CA)], config, sources)
        single_panels = [drawn for drawn in capture.panels if not drawn.output.startswith("quantiles_grid")]
        assert len(single_panels) == 4
        # CURRENT BEHAVIOR (R23): single plots ignore surveillance_source and use "hosp" (tag 1).
        assert {drawn.surveillance[3] for drawn in single_panels} == {1}

    def test_grid_uses_selected_source(self, capture, sources, config):
        run_generators([make_calibration(CA)], config, sources)
        grid_panels = [drawn for drawn in capture.panels if drawn.output.startswith("quantiles_grid")]
        assert len(grid_panels) == 4
        assert {drawn.surveillance[3] for drawn in grid_panels} == {6}


class TestHiddenSurveillance:
    """filtered hides surveillance while full shows it."""

    @pytest.fixture
    def config(self):
        outputs = [
            QuantilesOutputConfig(type="filtered", show_calibration=True, show_surveillance=False),
            QuantilesOutputConfig(type="full", show_calibration=True),
        ]
        return make_plots_config(single=True, grid=True, outputs=outputs)

    def test_single_filtered_clipped_by_hidden_surveillance(self, capture, hosp_only, config):
        run_generators([make_calibration(CA)], config, hosp_only)
        # CURRENT BEHAVIOR (decision 5): the ribbons are clipped to surveillance that is not shown.
        assert capture.summary(f"quantiles_{CA}_filtered") == [
            panel("California", CAL_FROM_PROJECTION_START, PROJ_BASE_HORIZON, None)
        ]

    def test_grid_filtered_not_clipped(self, capture, hosp_only, config):
        run_generators([make_calibration(CA)], config, hosp_only)
        assert capture.summary("quantiles_grid_filtered") == [panel("California", CAL_FULL, PROJ_BASE_HORIZON, None)]
        assert capture.summary("quantiles_grid_full") == [panel("California", CAL_FULL, PROJ_BASE_HORIZON, SURV_ALL)]


class TestSideBySideAllCalibrationClipped:
    """filtered_panel starts after the calibration ends, so the filtered panel should have no calibration."""

    def test_grid_redraws_clipped_calibration(self, capture, hosp_only):
        outputs = [
            QuantilesOutputConfig(
                type="side_by_side", show_calibration=True, filtered_panel={"surveillance_start_date": "2024-01-20"}
            )
        ]
        run_generators([make_calibration(CA)], make_plots_config(single=False, grid=True, outputs=outputs), hosp_only)
        # CURRENT BEHAVIOR (decision 7): the fully clipped calibration comes back as the full calibration.
        assert capture.summary("quantiles_grid_sidebyside")[1] == panel(
            "California", CAL_FULL, ("2024-01-20", "2024-02-03", 3, 1), ("2024-01-20", "2024-01-27", 2, 1)
        )


class TestColorBool:
    @pytest.mark.parametrize("layer", ["calibration", "projection"])
    def test_false_skips_every_plot(self, capture, hosp_only, layer):
        outputs = [QuantilesOutputConfig(type="full", show_calibration=True, show_projection=layer != "projection")]
        config = make_plots_config(single=True, grid=True, outputs=outputs, **{layer: False})
        # CURRENT BEHAVIOR (R24-3): reading `.color` on False raises, and the plot is silently skipped.
        assert run_generators([make_calibration(CA)], config, hosp_only) == {}


class TestGenerationNotice:
    def test_title_suffix_and_footnote(self, capture, hosp_only):
        config = make_plots_config(single=True, grid=True, outputs=[QuantilesOutputConfig(type="full")])
        run_generators([make_calibration(CA, incomplete=True), make_calibration(NY)], config, hosp_only)
        footnote = "* Completed 1 of 3 requested generations"
        for name in (f"quantiles_{CA}_full", "quantiles_grid_full"):
            titles = [drawn.axis.get_title() for drawn in capture.panels_for(name)]
            assert titles[0] == "California*"
            assert footnote in [text.get_text() for text in capture.packaged[name].texts]
        assert capture.panels_for("quantiles_grid_full")[1].axis.get_title() == "New York"
        assert capture.panels_for(f"quantiles_{NY}_full")[0].axis.get_title() == "New York"


class TestDispatcherFiltersFailedProjections:
    """Failed projections ({}) with aligned projection parameters, through generate_calibration_outputs()."""

    def test_plots_and_filtered_count(self, capture, hosp_only):
        calibration = make_calibration(CA)
        valid = calibration.results.projections["baseline"]
        calibration.results.projections["baseline"] = [valid[0], {}, *valid[1:], {}]
        calibration.results.projection_parameters = {"baseline": pd.DataFrame({"R0": np.arange(7.0)})}
        output_config = OutputConfig(
            output=OutputConfiguration(
                tabular_output_types=[TabularOutputTypeEnum.DataFrame],
                quantiles=None,
                trajectories=None,
                posteriors=False,
                model_meta=ModelMetaOutput(projection_parameters=True),
                options=OutputOptions(surveillance=hosp_only),
                plots=make_plots_config(single=True, grid=True, outputs=[QuantilesOutputConfig(type="full")]),
            )
        )

        outputs = generate_calibration_outputs(calibrations=[calibration], output_config=output_config)

        assert {f"quantiles_{CA}_full", "quantiles_grid_full"} <= set(outputs)
        for name in (f"quantiles_{CA}_full", "quantiles_grid_full"):
            assert capture.summary(name) == [panel("California", None, PROJ_BASE_HORIZON, SURV_ALL)]
        assert len(calibration.results.projections["baseline"]) == 5
        assert calibration.results.projection_parameters["baseline"]["R0"].tolist() == [0.0, 2.0, 3.0, 4.0, 5.0]
        assert len(outputs["projection_parameters_long"][0].data) == 5
        # CURRENT BEHAVIOR (A02): the plot generators filter again, which resets the count to 0.
        assert calibration.results._filtered_count == 0


def test_span_helper():
    df = pd.DataFrame({"date": ["2024-01-06", "2024-01-13", "2024-01-13"], "value": [1500.0, 1600.0, 1700.0]})
    assert span(df) == ("2024-01-06", "2024-01-13", 2, 1)
    assert span(df.iloc[0:0]) == ()
    assert span(None) is None
