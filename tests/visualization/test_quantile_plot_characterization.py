"""Characterization tests for quantile plots: {single, grid} x {filtered, full, side_by_side}.

These pin down what each drawn axis receives today, so the override refactor can be checked against them.
Assertions marked ``TODO: CURRENT BEHAVIOR`` capture inconsistencies that the refactor fixes on purpose;
everything else should stay the same.

Frames are summarized with ``summarize_frame()`` as (first date, last date, number of unique dates, location label, e.g. "CA hosp_aug").
Quantile levels are captured separately for calibration and projection frames.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from quantile_plot_test_utils import (
    QUANTILES,
    PlotCapture,
    make_calibration,
    make_plots_config,
    summarize_frame,
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

# Shared frame summaries; frame_with_location() returns a copy with an explicit location label.
# (first date, last date, number of unique dates, location label)
CALIB_FULL = ("2023-11-04", "2024-01-13", 11, None)
CALIB_FROM_PROJECTION_START = ("2023-12-02", "2024-01-13", 7, None)
PROJ_BASE_HORIZON = ("2023-12-02", "2024-02-03", 10, None)  # reference_date + 3 weeks
SURV_ALL = ("2023-10-07", "2024-01-27", 17, None)
SURV_FROM_PROJECTION_START = ("2023-12-02", "2024-01-27", 9, None)


def frame_with_location(summary: tuple, label: str) -> tuple:
    return (*summary[:3], label)


def make_panel_summary(title, calibration, projection, surveillance):
    return {"title": title, "calibration": calibration, "projection": projection, "surveillance": surveillance}


def run_generators(calibrations, plots_config, surveillance_sources):
    out_dict = {}
    generate_single_quantile_plots(calibrations, plots_config, out_dict, surveillance_sources)
    generate_quantile_grid_plot(calibrations, plots_config, out_dict, surveillance_sources)
    return out_dict


def count_visible_axes(fig) -> int:
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


def make_outputs_with_calibration():
    return [
        QuantilesOutputConfig(type="filtered", surveillance_points=8, show_calibration=True),
        QuantilesOutputConfig(type="full", show_calibration=True),
        QuantilesOutputConfig(type="side_by_side", show_calibration=True),
    ]


class TestSingleDefaultOutputs:
    @pytest.fixture
    def out_dict(self, capture, hosp_only):
        config = make_plots_config(single=True, grid=False, outputs=make_outputs_with_calibration(), ylabel="Hosp")
        return run_generators([make_calibration(CA), make_calibration(NY)], config, hosp_only)

    def test_output_keys_and_panel_counts(self, out_dict, capture):
        """Each location gets filtered, full and side_by_side outputs; side_by_side draws two panels."""
        assert sorted(out_dict) == sorted(
            f"quantiles_{location}_{kind}" for location in (CA, NY) for kind in ("filtered", "full", "side_by_side")
        )
        for location in (CA, NY):
            assert len(capture.get_panels(f"quantiles_{location}_filtered")) == 1
            assert len(capture.get_panels(f"quantiles_{location}_full")) == 1
            assert len(capture.get_panels(f"quantiles_{location}_side_by_side")) == 2
        assert all(panel.output is not None for panel in capture.panels)

    def test_filtered(self, out_dict, capture):
        # TODO: CURRENT BEHAVIOR — surveillance_points=8 is counted after the projection start,
        # so only 7 points on or before reference_date are shown.
        """The filtered plot shows the last 8 surveillance points and clips the ribbons to the first one."""
        assert capture.summarize_output(f"quantiles_{CA}_filtered") == [
            make_panel_summary(
                "California",
                frame_with_location(CALIB_FROM_PROJECTION_START, "CA"),
                frame_with_location(PROJ_BASE_HORIZON, "CA"),
                frame_with_location(SURV_FROM_PROJECTION_START, "CA"),
            )
        ]
        assert capture.summarize_output(f"quantiles_{NY}_filtered") == [
            make_panel_summary(
                "New York",
                frame_with_location(CALIB_FROM_PROJECTION_START, "NY"),
                frame_with_location(PROJ_BASE_HORIZON, "NY"),
                frame_with_location(SURV_FROM_PROJECTION_START, "NY"),
            )
        ]

    def test_full(self, out_dict, capture):
        """The full plot shows all calibration and surveillance, and the projection up to the base horizon."""
        assert capture.summarize_output(f"quantiles_{CA}_full") == [
            make_panel_summary(
                "California",
                frame_with_location(CALIB_FULL, "CA"),
                frame_with_location(PROJ_BASE_HORIZON, "CA"),
                frame_with_location(SURV_ALL, "CA"),
            )
        ]

    def test_side_by_side(self, out_dict, capture):
        """side_by_side draws the full view, then a filtered view starting at the projection start."""
        assert capture.summarize_output(f"quantiles_{CA}_side_by_side") == [
            make_panel_summary(
                "California",
                frame_with_location(CALIB_FULL, "CA"),
                frame_with_location(PROJ_BASE_HORIZON, "CA"),
                frame_with_location(SURV_ALL, "CA"),
            ),
            make_panel_summary(
                "California",
                frame_with_location(CALIB_FROM_PROJECTION_START, "CA"),
                frame_with_location(PROJ_BASE_HORIZON, "CA"),
                frame_with_location(SURV_FROM_PROJECTION_START, "CA"),
            ),
        ]

    def test_quantile_levels_and_styling(self, out_dict, capture):
        """All single panels keep every quantile level and the default styling."""
        for drawn in capture.panels:
            assert drawn.quantile_levels["calibration_quantiles"] == QUANTILES
            assert drawn.quantile_levels["projection_quantiles"] == QUANTILES
            assert drawn.fitting_window == FITTING_WINDOW
            assert (drawn.calibration_color, drawn.projection_color) == ("C0", "C1")
            assert drawn.xlabel_interval is None
        # side_by_side only labels the left panel
        assert [drawn.ylabel for drawn in capture.get_panels(f"quantiles_{CA}_side_by_side")] == ["Hosp", None]
        assert capture.get_panels(f"quantiles_{CA}_filtered")[0].ylabel == "Hosp"


class TestGridDefaultOutputs:
    """Three locations, one calibration-only, on a partly filled 4-column grid."""

    @pytest.fixture
    def out_dict(self, capture, hosp_only):
        config = make_plots_config(
            single=False, grid={"panels_per_row": 4}, outputs=make_outputs_with_calibration(), ylabel="Hosp"
        )
        calibrations = [make_calibration(CA), make_calibration(NY), make_calibration(TX, with_projection=False)]
        return run_generators(calibrations, config, hosp_only)

    def test_output_keys_and_layout(self, out_dict, capture):
        """Each grid has one panel per location (two for side_by_side) and hides the unused axes."""
        assert sorted(out_dict) == ["quantiles_grid_filtered", "quantiles_grid_full", "quantiles_grid_sidebyside"]
        for name, n_panels, n_axes in (
            ("quantiles_grid_filtered", 3, 4),
            ("quantiles_grid_full", 3, 4),
            ("quantiles_grid_sidebyside", 6, 8),
        ):
            fig = capture.packaged[name]
            assert len(capture.get_panels(name)) == n_panels
            assert len(fig.axes) == n_axes
            assert count_visible_axes(fig) == n_panels

    def test_filtered(self, out_dict, capture):
        # TODO: CURRENT BEHAVIOR — Texas has no projection, so it gets no surveillance.
        """Grid filtered panels with surveillance_points=8, including calibration-only Texas."""
        assert capture.summarize_output("quantiles_grid_filtered") == [
            make_panel_summary(
                "California",
                frame_with_location(CALIB_FROM_PROJECTION_START, "CA"),
                frame_with_location(PROJ_BASE_HORIZON, "CA"),
                frame_with_location(SURV_FROM_PROJECTION_START, "CA"),
            ),
            make_panel_summary(
                "New York",
                frame_with_location(CALIB_FROM_PROJECTION_START, "NY"),
                frame_with_location(PROJ_BASE_HORIZON, "NY"),
                frame_with_location(SURV_FROM_PROJECTION_START, "NY"),
            ),
            make_panel_summary("Texas", frame_with_location(CALIB_FULL, "TX"), None, None),
        ]

    def test_full(self, out_dict, capture):
        """Grid full panels for every location, including calibration-only Texas."""
        assert capture.summarize_output("quantiles_grid_full") == [
            make_panel_summary(
                "California",
                frame_with_location(CALIB_FULL, "CA"),
                frame_with_location(PROJ_BASE_HORIZON, "CA"),
                frame_with_location(SURV_ALL, "CA"),
            ),
            make_panel_summary(
                "New York",
                frame_with_location(CALIB_FULL, "NY"),
                frame_with_location(PROJ_BASE_HORIZON, "NY"),
                frame_with_location(SURV_ALL, "NY"),
            ),
            make_panel_summary("Texas", frame_with_location(CALIB_FULL, "TX"), None, None),
        ]

    def test_side_by_side(self, out_dict, capture):
        """Grid side_by_side draws a (full, filtered) pair per location, including calibration-only Texas."""
        assert capture.summarize_output("quantiles_grid_sidebyside") == [
            make_panel_summary(
                "California",
                frame_with_location(CALIB_FULL, "CA"),
                frame_with_location(PROJ_BASE_HORIZON, "CA"),
                frame_with_location(SURV_ALL, "CA"),
            ),
            make_panel_summary(
                "California",
                frame_with_location(CALIB_FROM_PROJECTION_START, "CA"),
                frame_with_location(PROJ_BASE_HORIZON, "CA"),
                frame_with_location(SURV_FROM_PROJECTION_START, "CA"),
            ),
            make_panel_summary(
                "New York",
                frame_with_location(CALIB_FULL, "NY"),
                frame_with_location(PROJ_BASE_HORIZON, "NY"),
                frame_with_location(SURV_ALL, "NY"),
            ),
            make_panel_summary(
                "New York",
                frame_with_location(CALIB_FROM_PROJECTION_START, "NY"),
                frame_with_location(PROJ_BASE_HORIZON, "NY"),
                frame_with_location(SURV_FROM_PROJECTION_START, "NY"),
            ),
            make_panel_summary("Texas", frame_with_location(CALIB_FULL, "TX"), None, None),
            make_panel_summary("Texas", frame_with_location(CALIB_FULL, "TX"), None, None),
        ]

    def test_quantile_levels_and_styling(self, out_dict, capture):
        """All grid panels keep every quantile level and the default styling."""
        for drawn in capture.panels:
            assert drawn.quantile_levels["calibration_quantiles"] == QUANTILES
            if drawn.projection is not None:
                assert drawn.quantile_levels["projection_quantiles"] == QUANTILES
            assert drawn.fitting_window == FITTING_WINDOW
            assert (drawn.calibration_color, drawn.projection_color) == ("C0", "C1")
        # ylabel only on the leftmost column
        assert [drawn.ylabel for drawn in capture.get_panels("quantiles_grid_filtered")] == ["Hosp", None, None]
        # Four columns per row; each location occupies a (full, filtered) pair.
        assert [drawn.ylabel for drawn in capture.get_panels("quantiles_grid_sidebyside")] == [
            "Hosp",  # Row 1, column 1: California full (leftmost column)
            None,  # Row 1, column 2: California filtered
            None,  # Row 1, column 3: New York full
            None,  # Row 1, column 4: New York filtered
            "Hosp",  # Row 2, column 1: Texas full (leftmost column)
            None,  # Row 2, column 2: Texas filtered
        ]


def make_override_config(**kwargs):
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
OVERRIDE_FILTERED = make_panel_summary(
    "California",
    ("2023-12-16", "2024-01-13", 5, "CA"),
    ("2023-12-16", "2024-01-20", 6, "CA"),
    ("2023-12-16", "2024-01-27", 7, "CA"),
)
# full output: surveillance_points=4 (4 on or before reference_date + 2 after)
# TODO: CURRENT BEHAVIOR — horizon_max=6 is capped at the base horizon_max=3.
OVERRIDE_FULL = make_panel_summary(
    "California",
    frame_with_location(CALIB_FULL, "CA"),
    frame_with_location(PROJ_BASE_HORIZON, "CA"),
    ("2023-12-23", "2024-01-27", 6, "CA"),
)


class TestSingleOverrides:
    @pytest.fixture
    def out_dict(self, capture, hosp_only):
        return run_generators([make_calibration(CA)], make_override_config(single=True, grid=False), hosp_only)

    def test_filtered_and_full(self, out_dict, capture):
        """Output surveillance limits and horizon_max on single filtered and full plots."""
        assert capture.summarize_output(f"quantiles_{CA}_filtered") == [OVERRIDE_FILTERED]
        assert capture.summarize_output(f"quantiles_{CA}_full") == [OVERRIDE_FULL]

    def test_side_by_side(self, out_dict, capture):
        # TODO: CURRENT BEHAVIOR — single side_by_side ignores output and panel surveillance limits
        # and the output horizon_max; it only uses the panel xlabel_interval.
        """Output and panel overrides on a single side_by_side plot."""
        assert capture.summarize_output(f"quantiles_{CA}_side_by_side") == [
            make_panel_summary(
                "California",
                frame_with_location(CALIB_FULL, "CA"),
                frame_with_location(PROJ_BASE_HORIZON, "CA"),
                frame_with_location(SURV_ALL, "CA"),
            ),
            make_panel_summary(
                "California",
                frame_with_location(CALIB_FROM_PROJECTION_START, "CA"),
                frame_with_location(PROJ_BASE_HORIZON, "CA"),
                frame_with_location(SURV_FROM_PROJECTION_START, "CA"),
            ),
        ]

    def test_styling(self, out_dict, capture):
        """Output and panel xlabel_interval and the configured colors reach every single-plot panel."""
        assert [drawn.xlabel_interval for drawn in capture.panels] == ["2W-SAT", None, "MS", "W-SAT"]
        for drawn in capture.panels:
            assert (drawn.calibration_color, drawn.projection_color) == ("tab:green", "tab:red")


class TestGridOverrides:
    @pytest.fixture
    def out_dict(self, capture, hosp_only):
        calibrations = [make_calibration(CA), make_calibration(NY)]
        return run_generators(calibrations, make_override_config(single=False, grid={"panels_per_row": 4}), hosp_only)

    def test_filtered_and_full(self, out_dict, capture):
        """Output surveillance limits and horizon_max apply to every grid panel."""
        assert capture.summarize_output("quantiles_grid_filtered")[0] == OVERRIDE_FILTERED
        assert capture.summarize_output("quantiles_grid_full")[0] == OVERRIDE_FULL
        assert capture.summarize_output("quantiles_grid_full")[1]["surveillance"] == (
            "2023-12-23",
            "2024-01-27",
            6,
            "NY",
        )

    def test_side_by_side(self, out_dict, capture):
        # TODO: CURRENT BEHAVIOR — grid side_by_side uses panel limits only; the output surveillance_points=3
        # and output horizon_max=1 are ignored.
        """Output and panel overrides on a grid side_by_side plot."""
        assert capture.summarize_output("quantiles_grid_sidebyside")[:2] == [
            make_panel_summary(
                "California",
                frame_with_location(CALIB_FULL, "CA"),
                frame_with_location(PROJ_BASE_HORIZON, "CA"),
                ("2023-11-04", "2024-01-27", 13, "CA"),
            ),
            make_panel_summary(
                "California",
                ("2024-01-06", "2024-01-13", 2, "CA"),
                ("2024-01-06", "2024-02-03", 5, "CA"),
                ("2024-01-06", "2024-01-27", 4, "CA"),
            ),
        ]

    def test_styling(self, out_dict, capture):
        """Output and panel xlabel_interval and the configured colors reach every grid panel."""
        assert [drawn.xlabel_interval for drawn in capture.get_panels("quantiles_grid_filtered")] == ["2W-SAT"] * 2
        assert [drawn.xlabel_interval for drawn in capture.get_panels("quantiles_grid_full")] == [None] * 2
        assert [drawn.xlabel_interval for drawn in capture.get_panels("quantiles_grid_sidebyside")] == [
            "MS",
            "W-SAT",
        ] * 2
        for drawn in capture.panels:
            assert (drawn.calibration_color, drawn.projection_color) == ("tab:green", "tab:red")


class TestSurveillanceSourceSelection:
    """Outputs alternate between augmented and original hospitalization sources."""

    @pytest.fixture
    def config(self):
        outputs = [
            QuantilesOutputConfig(type="filtered", surveillance_source="hosp_aug"),
            QuantilesOutputConfig(type="full", surveillance_source="hosp"),
            QuantilesOutputConfig(type="side_by_side", surveillance_source="hosp_aug"),
        ]
        return make_plots_config(single=True, grid=True, outputs=outputs)

    def test_single_uses_first_loaded_source(self, capture, sources, config):
        """Single plots use the first loaded source even when outputs select different sources."""
        run_generators([make_calibration(CA)], config, sources)
        single_panels = [drawn for drawn in capture.panels if not drawn.output.startswith("quantiles_grid")]
        assert len(single_panels) == 4
        # TODO: CURRENT BEHAVIOR — single plots ignore surveillance_source and use "hosp" (reported as "CA").
        assert {drawn.surveillance[3] for drawn in single_panels} == {"CA"}

    def test_grid_uses_selected_source(self, capture, sources, config):
        """Each grid output uses its own source, including switching back to "hosp_aug"."""
        run_generators([make_calibration(CA)], config, sources)
        for name, expected in (
            ("quantiles_grid_filtered", ["CA hosp_aug"]),
            ("quantiles_grid_full", ["CA"]),
            ("quantiles_grid_sidebyside", ["CA hosp_aug", "CA hosp_aug"]),
        ):
            assert [drawn.surveillance[3] for drawn in capture.get_panels(name)] == expected


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
        """Whether hidden surveillance clips the ribbons of a single filtered plot."""
        run_generators([make_calibration(CA)], config, hosp_only)
        # TODO: CURRENT BEHAVIOR — the ribbons are clipped to surveillance that is not shown.
        assert capture.summarize_output(f"quantiles_{CA}_filtered") == [
            make_panel_summary(
                "California",
                frame_with_location(CALIB_FROM_PROJECTION_START, "CA"),
                frame_with_location(PROJ_BASE_HORIZON, "CA"),
                None,
            )
        ]

    def test_grid_filtered_not_clipped(self, capture, hosp_only, config):
        """Hidden surveillance does not clip the grid filtered ribbons; the full grid still shows it."""
        run_generators([make_calibration(CA)], config, hosp_only)
        assert capture.summarize_output("quantiles_grid_filtered") == [
            make_panel_summary(
                "California", frame_with_location(CALIB_FULL, "CA"), frame_with_location(PROJ_BASE_HORIZON, "CA"), None
            )
        ]
        assert capture.summarize_output("quantiles_grid_full") == [
            make_panel_summary(
                "California",
                frame_with_location(CALIB_FULL, "CA"),
                frame_with_location(PROJ_BASE_HORIZON, "CA"),
                frame_with_location(SURV_ALL, "CA"),
            )
        ]


class TestSideBySideAllCalibrationClipped:
    """filtered_panel starts after the calibration ends, so the filtered panel should have no calibration."""

    def test_grid_redraws_clipped_calibration(self, capture, hosp_only):
        """What the filtered panel draws when its start date clips away all calibration."""
        outputs = [
            QuantilesOutputConfig(
                type="side_by_side", show_calibration=True, filtered_panel={"surveillance_start_date": "2024-01-20"}
            )
        ]
        run_generators([make_calibration(CA)], make_plots_config(single=False, grid=True, outputs=outputs), hosp_only)
        # TODO: CURRENT BEHAVIOR — the fully clipped calibration comes back as the full calibration.
        assert capture.summarize_output("quantiles_grid_sidebyside")[1] == make_panel_summary(
            "California",
            frame_with_location(CALIB_FULL, "CA"),
            ("2024-01-20", "2024-02-03", 3, "CA"),
            ("2024-01-20", "2024-01-27", 2, "CA"),
        )


class TestColorBool:
    @pytest.mark.parametrize("layer", ["calibration", "projection"])
    def test_false_skips_every_plot(self, capture, hosp_only, layer):
        """What calibration: false / projection: false in the styling section do."""
        outputs = [QuantilesOutputConfig(type="full", show_calibration=True, show_projection=layer != "projection")]
        config = make_plots_config(single=True, grid=True, outputs=outputs, **{layer: False})
        # TODO: CURRENT BEHAVIOR — reading `.color` on False raises, and the plot is silently skipped.
        assert run_generators([make_calibration(CA)], config, hosp_only) == {}


class TestGenerationNotice:
    def test_title_suffix_and_footnote(self, capture, hosp_only):
        """Incomplete generations add "*" to the title and a footnote, in single and grid plots."""
        config = make_plots_config(single=True, grid=True, outputs=[QuantilesOutputConfig(type="full")])
        run_generators([make_calibration(CA, incomplete=True), make_calibration(NY)], config, hosp_only)
        footnote = "* Completed 1 of 3 requested generations"
        for name in (f"quantiles_{CA}_full", "quantiles_grid_full"):
            titles = [drawn.axis.get_title() for drawn in capture.get_panels(name)]
            assert titles[0] == "California*"
            assert footnote in [text.get_text() for text in capture.packaged[name].texts]
        assert capture.get_panels("quantiles_grid_full")[1].axis.get_title() == "New York"
        assert capture.get_panels(f"quantiles_{NY}_full")[0].axis.get_title() == "New York"


class TestDispatcherFiltersFailedProjections:
    """Failed projections ({}) with aligned projection parameters, through generate_calibration_outputs()."""

    def test_plots_and_filtered_count(self, capture, hosp_only):
        """generate_calibration_outputs() drops failed projections and their parameters before plotting."""
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
            assert capture.summarize_output(name) == [
                make_panel_summary(
                    "California",
                    None,
                    frame_with_location(PROJ_BASE_HORIZON, "CA"),
                    frame_with_location(SURV_ALL, "CA"),
                )
            ]
        assert len(calibration.results.projections["baseline"]) == 5
        assert calibration.results.projection_parameters["baseline"]["R0"].tolist() == [0.0, 2.0, 3.0, 4.0, 5.0]
        assert len(outputs["projection_parameters_long"][0].data) == 5
        # TODO: CURRENT BEHAVIOR — the plot generators filter again, which resets the count to 0.
        assert calibration.results._filtered_count == 0


def test_summarize_frame():
    """summarize_frame() summarizes dates and the location label, and keeps empty and None apart."""
    df = pd.DataFrame({"date": ["2024-01-06", "2024-01-13", "2024-01-13"], "value": [1500.0, 1600.0, 1700.0]})
    assert summarize_frame(df) == ("2024-01-06", "2024-01-13", 2, "CA")
    assert summarize_frame(df.iloc[0:0]) == ()
    assert summarize_frame(None) is None
