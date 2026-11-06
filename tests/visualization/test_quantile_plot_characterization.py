"""Characterization tests for quantile plots: {single, grid} x {filtered, full, side_by_side}.

These pin down what each drawn axis receives. Assertions marked ``CHANGED`` were updated on purpose by the
override refactor.

Frames are summarized with ``summarize_frame()`` as (first date, last date, number of unique dates, location label, e.g. "CA hosp_aug").
Quantile levels are captured separately for calibration and projection frames.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
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
from epymodelingsuite.visualization.generators import generate_quantile_plots

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
SURV_FROM_CALIBRATION_START = ("2023-11-04", "2024-01-27", 13, None)
# surveillance_points=8: the last 8 points on or before reference_date, plus the 2 after it
SURV_LAST_8 = ("2023-11-25", "2024-01-27", 10, None)
CALIB_FROM_LAST_8 = ("2023-11-25", "2024-01-13", 8, None)


def frame_with_location(summary: tuple, label: str) -> tuple:
    """Return a new frame summary with its location field replaced by ``label``.

    Parameters
    ----------
    summary : tuple
        ``(first_date, last_date, unique_date_count, location)``. Dates are ISO
        date strings, the count is the number of distinct dates, and location
        may be ``None`` in a shared template.
    label : str
        Location label to use in the returned summary, such as ``"CA"`` or ``"NY"``.

    Returns
    -------
    tuple
        ``(first_date, last_date, unique_date_count, label)`` with the first
        three fields copied from ``summary``. The input tuple is unchanged.
    """
    return (*summary[:3], label)


def make_expected_panel_summary(title, calibration, projection, surveillance):
    """Build the expected dictionary for one drawn axis in a plot assertion.

    Parameters
    ----------
    title : str
        Expected axis title, such as ``"California"``.
    calibration : tuple or None
        Expected calibration frame as ``(first_date, last_date,
        unique_date_count, location)``: ISO date bounds, number of distinct
        dates, and location label. ``()`` means the frame is empty;
        ``None`` means the layer is absent.
    projection : tuple or None
        Expected projection frame in the same tuple format as ``calibration``,
        or ``None`` when the layer is absent.
    surveillance : tuple or None
        Expected surveillance frame in the same tuple format as ``calibration``,
        or ``None`` when the layer is absent.

    Returns
    -------
    dict
        The supplied values grouped under ``title``, ``calibration``,
        ``projection`` and ``surveillance``, for comparison with one item
        from ``PlotCapture.summarize_output()``.
    """
    return {"title": title, "calibration": calibration, "projection": projection, "surveillance": surveillance}


def run_generators(calibrations, plots_config, surveillance_sources):
    out_dict = {}
    generate_quantile_plots(calibrations, plots_config, out_dict, surveillance_sources)
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


def make_variant_configs_with_calibration():
    return [
        QuantilesOutputConfig(type="filtered", surveillance_points=8, show_calibration=True),
        QuantilesOutputConfig(type="full", show_calibration=True),
        QuantilesOutputConfig(type="side_by_side", show_calibration=True),
    ]


class TestSingleDefaultOutputs:
    """Default outputs for California and New York.

    Check output keys, panel counts, location-specific data ranges, quantile
    levels and styling across filtered, full and side_by_side single plots.
    """

    @pytest.fixture
    def out_dict(self, capture, hosp_only):
        config = make_plots_config(
            single=True, grid=False, outputs=make_variant_configs_with_calibration(), ylabel="Hosp"
        )
        return run_generators([make_calibration(CA), make_calibration(NY)], config, hosp_only)

    def test_output_keys_and_panel_counts(self, out_dict, capture):
        """Default outputs for two locations: check output keys and panel counts.

        Each location gets filtered and full outputs with one panel each, and a
        side_by_side output with two panels. Every captured panel has an output key.
        """
        assert sorted(out_dict) == sorted(
            f"quantiles_{location}_{kind}" for location in (CA, NY) for kind in ("filtered", "full", "side_by_side")
        )
        for location in (CA, NY):
            assert len(capture.get_panels(f"quantiles_{location}_filtered")) == 1
            assert len(capture.get_panels(f"quantiles_{location}_full")) == 1
            assert len(capture.get_panels(f"quantiles_{location}_side_by_side")) == 2
        assert all(panel.output is not None for panel in capture.panels)

    def test_filtered(self, out_dict, capture):
        """Filtered single plots with eight requested surveillance points.

        Surveillance starts at November 25, keeping eight observations on or before
        January 13 and two later ones. Calibration starts there too; projection
        retains its December 2 start.
        """
        # CHANGED: count observations before clipping to the projection start, which used to leave only 7.
        assert capture.summarize_output(f"quantiles_{CA}_filtered") == [
            make_expected_panel_summary(
                "California",
                frame_with_location(CALIB_FROM_LAST_8, "CA"),
                frame_with_location(PROJ_BASE_HORIZON, "CA"),
                frame_with_location(SURV_LAST_8, "CA"),
            )
        ]
        assert capture.summarize_output(f"quantiles_{NY}_filtered") == [
            make_expected_panel_summary(
                "New York",
                frame_with_location(CALIB_FROM_LAST_8, "NY"),
                frame_with_location(PROJ_BASE_HORIZON, "NY"),
                frame_with_location(SURV_LAST_8, "NY"),
            )
        ]

    def test_full(self, out_dict, capture):
        """Full single plot for California with the default three-week horizon.

        Keep all calibration and surveillance dates, and end the projection at
        February 3, three weeks after the January 13 reference date.
        """
        assert capture.summarize_output(f"quantiles_{CA}_full") == [
            make_expected_panel_summary(
                "California",
                frame_with_location(CALIB_FULL, "CA"),
                frame_with_location(PROJ_BASE_HORIZON, "CA"),
                frame_with_location(SURV_ALL, "CA"),
            )
        ]

    def test_side_by_side(self, out_dict, capture):
        """Default side_by_side output for California: full panel, then filtered panel.

        The full panel keeps all calibration and surveillance dates. The filtered
        panel starts all three layers at the December 2 projection start.
        """
        assert capture.summarize_output(f"quantiles_{CA}_side_by_side") == [
            make_expected_panel_summary(
                "California",
                frame_with_location(CALIB_FULL, "CA"),
                frame_with_location(PROJ_BASE_HORIZON, "CA"),
                frame_with_location(SURV_ALL, "CA"),
            ),
            make_expected_panel_summary(
                "California",
                frame_with_location(CALIB_FROM_PROJECTION_START, "CA"),
                frame_with_location(PROJ_BASE_HORIZON, "CA"),
                frame_with_location(SURV_FROM_PROJECTION_START, "CA"),
            ),
        ]

    def test_quantile_levels_and_styling(self, out_dict, capture):
        """Calibration and projection quantiles and styling in all single variants.

        Keep all five quantile levels, the fitting-window dates, default colors
        and automatic tick intervals. The side-by-side ylabel goes on the left panel.
        """
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
    """A four-column grid with California, New York and calibration-only Texas.

    Check location order, data ranges, hidden unused axes, quantile levels and
    labels. Each side-by-side location uses two columns, putting Texas on row 2.
    """

    @pytest.fixture
    def out_dict(self, capture, hosp_only):
        config = make_plots_config(
            single=False, grid={"panels_per_row": 4}, outputs=make_variant_configs_with_calibration(), ylabel="Hosp"
        )
        calibrations = [make_calibration(CA), make_calibration(NY), make_calibration(TX, with_projection=False)]
        return run_generators(calibrations, config, hosp_only)

    def test_output_keys_and_layout(self, out_dict, capture):
        """Partly filled grids with three locations and four columns per row.

        Filtered and full grids draw three panels in four slots. The side-by-side
        grid draws six panels in eight slots. All unused axes are hidden.
        """
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
        """Filtered grid with eight requested surveillance points and calibration-only Texas.

        All three locations start calibration and surveillance at November 25.
        California and New York keep their December 2 projection start; Texas
        receives surveillance despite having no projection.
        """
        # CHANGED: count all observations and include surveillance for calibration-only locations.
        assert capture.summarize_output("quantiles_grid_filtered") == [
            make_expected_panel_summary(
                "California",
                frame_with_location(CALIB_FROM_LAST_8, "CA"),
                frame_with_location(PROJ_BASE_HORIZON, "CA"),
                frame_with_location(SURV_LAST_8, "CA"),
            ),
            make_expected_panel_summary(
                "New York",
                frame_with_location(CALIB_FROM_LAST_8, "NY"),
                frame_with_location(PROJ_BASE_HORIZON, "NY"),
                frame_with_location(SURV_LAST_8, "NY"),
            ),
            make_expected_panel_summary(
                "Texas", frame_with_location(CALIB_FROM_LAST_8, "TX"), None, frame_with_location(SURV_LAST_8, "TX")
            ),
        ]

    def test_full(self, out_dict, capture):
        """Full grid with projections for California and New York, and calibration-only Texas.

        Check all three locations in order. Every location keeps full calibration
        and surveillance ranges; only California and New York have projections.
        """
        assert capture.summarize_output("quantiles_grid_full") == [
            make_expected_panel_summary(
                "California",
                frame_with_location(CALIB_FULL, "CA"),
                frame_with_location(PROJ_BASE_HORIZON, "CA"),
                frame_with_location(SURV_ALL, "CA"),
            ),
            make_expected_panel_summary(
                "New York",
                frame_with_location(CALIB_FULL, "NY"),
                frame_with_location(PROJ_BASE_HORIZON, "NY"),
                frame_with_location(SURV_ALL, "NY"),
            ),
            make_expected_panel_summary(
                "Texas", frame_with_location(CALIB_FULL, "TX"), None, frame_with_location(SURV_ALL, "TX")
            ),
        ]

    def test_side_by_side(self, out_dict, capture):
        """Side-by-side grid with a full/filtered pair for each of three locations.

        Check each pair's location and date ranges in drawing order. Texas keeps
        full calibration in both panels; its filtered surveillance starts at the
        November 4 calibration start because it has no projection.
        """
        assert capture.summarize_output("quantiles_grid_sidebyside") == [
            make_expected_panel_summary(
                "California",
                frame_with_location(CALIB_FULL, "CA"),
                frame_with_location(PROJ_BASE_HORIZON, "CA"),
                frame_with_location(SURV_ALL, "CA"),
            ),
            make_expected_panel_summary(
                "California",
                frame_with_location(CALIB_FROM_PROJECTION_START, "CA"),
                frame_with_location(PROJ_BASE_HORIZON, "CA"),
                frame_with_location(SURV_FROM_PROJECTION_START, "CA"),
            ),
            make_expected_panel_summary(
                "New York",
                frame_with_location(CALIB_FULL, "NY"),
                frame_with_location(PROJ_BASE_HORIZON, "NY"),
                frame_with_location(SURV_ALL, "NY"),
            ),
            make_expected_panel_summary(
                "New York",
                frame_with_location(CALIB_FROM_PROJECTION_START, "NY"),
                frame_with_location(PROJ_BASE_HORIZON, "NY"),
                frame_with_location(SURV_FROM_PROJECTION_START, "NY"),
            ),
            make_expected_panel_summary(
                "Texas", frame_with_location(CALIB_FULL, "TX"), None, frame_with_location(SURV_ALL, "TX")
            ),
            make_expected_panel_summary(
                "Texas",
                frame_with_location(CALIB_FULL, "TX"),
                None,
                frame_with_location(SURV_FROM_CALIBRATION_START, "TX"),
            ),
        ]

    def test_quantile_levels_and_styling(self, out_dict, capture):
        """Calibration and projection quantiles and styling in all grid variants.

        Each supplied layer keeps all five quantile levels. Check fitting-window
        dates and default colors. Filtered and side-by-side grids label only the
        first column of each row, including the second row occupied by Texas.
        """
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
    variant_configs = [
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
        outputs=variant_configs,
        calibration={"color": "tab:green"},
        projection={"color": "tab:red"},
        ylabel="Hosp",
        **kwargs,
    )


# filtered output: surveillance_start_date=2023-12-16, horizon_max=1
OVERRIDE_FILTERED = make_expected_panel_summary(
    "California",
    ("2023-12-16", "2024-01-13", 5, "CA"),
    ("2023-12-16", "2024-01-20", 6, "CA"),
    ("2023-12-16", "2024-01-27", 7, "CA"),
)
# full output: surveillance_points=4 (4 on or before reference_date + 2 after)
# CHANGED: horizon_max=6 extends past the base horizon_max=3.
OVERRIDE_FULL = make_expected_panel_summary(
    "California",
    frame_with_location(CALIB_FULL, "CA"),
    ("2023-12-02", "2024-02-24", 13, "CA"),
    ("2023-12-23", "2024-01-27", 6, "CA"),
)
# side_by_side output: horizon_max=1; full_panel starts 2023-11-01, filtered_panel keeps 2 points.
# CHANGED: side_by_side used to ignore the output horizon_max (single and grid) and the panel limits (single).
OVERRIDE_SIDE_BY_SIDE = [
    make_expected_panel_summary(
        "California",
        frame_with_location(CALIB_FULL, "CA"),
        ("2023-12-02", "2024-01-20", 8, "CA"),
        frame_with_location(SURV_FROM_CALIBRATION_START, "CA"),
    ),
    make_expected_panel_summary(
        "California",
        ("2024-01-06", "2024-01-13", 2, "CA"),
        ("2024-01-06", "2024-01-20", 3, "CA"),
        ("2024-01-06", "2024-01-27", 4, "CA"),
    ),
]


class TestSingleOverrides:
    """Date limits and styling overrides on filtered, full and side_by_side single outputs."""

    @pytest.fixture
    def out_dict(self, capture, hosp_only):
        return run_generators([make_calibration(CA)], make_override_config(single=True, grid=False), hosp_only)

    def test_filtered_and_full(self, out_dict, capture):
        """Date limits on filtered and full single outputs.

        A December 16 start and one-week horizon clip the filtered projection to
        December 16–January 20. The full output keeps four observations up to the
        reference date plus two later ones; its six-week horizon extends to February 24.
        """
        assert capture.summarize_output(f"quantiles_{CA}_filtered") == [OVERRIDE_FILTERED]
        assert capture.summarize_output(f"quantiles_{CA}_full") == [OVERRIDE_FULL]

    def test_side_by_side(self, out_dict, capture):
        """Single side_by_side output with date limits on both the output and its panels.

        Both panels inherit the one-week horizon. The full panel starts surveillance
        at November 4, the first observation after its November 1 limit; the filtered
        panel keeps two observations up to the reference date plus two later ones.
        """
        assert capture.summarize_output(f"quantiles_{CA}_side_by_side") == OVERRIDE_SIDE_BY_SIDE

    def test_styling(self, out_dict, capture):
        """Custom colors and output/panel tick intervals on single plots.

        Check green calibration and red projection ribbons, a two-week Saturday
        interval for filtered output, and monthly/weekly intervals for the side-by-side pair.
        """
        assert [drawn.xlabel_interval for drawn in capture.panels] == ["2W-SAT", None, "MS", "W-SAT"]
        for drawn in capture.panels:
            assert (drawn.calibration_color, drawn.projection_color) == ("tab:green", "tab:red")


class TestGridOverrides:
    """Date limits and styling overrides on grids containing California and New York."""

    @pytest.fixture
    def out_dict(self, capture, hosp_only):
        calibrations = [make_calibration(CA), make_calibration(NY)]
        return run_generators(calibrations, make_override_config(single=False, grid={"panels_per_row": 4}), hosp_only)

    def test_filtered_and_full(self, out_dict, capture):
        """Date limits on filtered and full grid outputs.

        Check California's filtered and full data ranges, including the full
        output's six-week horizon. Check New York's four observations
        up to the reference date plus two later observations in the full grid.
        """
        assert capture.summarize_output("quantiles_grid_filtered")[0] == OVERRIDE_FILTERED
        assert capture.summarize_output("quantiles_grid_full")[0] == OVERRIDE_FULL
        assert capture.summarize_output("quantiles_grid_full")[1]["surveillance"] == (
            "2023-12-23",
            "2024-01-27",
            6,
            "NY",
        )

    def test_side_by_side(self, out_dict, capture):
        """Grid side_by_side output with competing output-level and panel-level date limits.

        California's full panel applies its November 1 surveillance start; its
        filtered panel keeps two observations up to the reference date plus two
        later ones. These panel limits override the output's three-point limit,
        while both panels inherit its one-week horizon.
        """
        assert capture.summarize_output("quantiles_grid_sidebyside")[:2] == OVERRIDE_SIDE_BY_SIDE

    def test_styling(self, out_dict, capture):
        """Custom colors and output/panel tick intervals on both locations in a grid.

        Check the filtered output's two-week Saturday interval, automatic ticks
        for full output, monthly/weekly intervals for each side-by-side pair,
        and green calibration/red projection colors throughout.
        """
        assert [drawn.xlabel_interval for drawn in capture.get_panels("quantiles_grid_filtered")] == ["2W-SAT"] * 2
        assert [drawn.xlabel_interval for drawn in capture.get_panels("quantiles_grid_full")] == [None] * 2
        assert [drawn.xlabel_interval for drawn in capture.get_panels("quantiles_grid_sidebyside")] == [
            "MS",
            "W-SAT",
        ] * 2
        for drawn in capture.panels:
            assert (drawn.calibration_color, drawn.projection_color) == ("tab:green", "tab:red")


class TestSurveillanceSourceSelection:
    """Different surveillance sources for successive outputs.

    Request hosp_aug for filtered, hosp for full, then hosp_aug for side_by_side.
    Check the selected source for each output in both single and grid plots.
    """

    @pytest.fixture
    def config(self):
        variant_configs = [
            QuantilesOutputConfig(type="filtered", surveillance_source="hosp_aug"),
            QuantilesOutputConfig(type="full", surveillance_source="hosp"),
            QuantilesOutputConfig(type="side_by_side", surveillance_source="hosp_aug"),
        ]
        return make_plots_config(single=True, grid=True, outputs=variant_configs)

    def test_single_uses_selected_source(self, capture, sources, config):
        """Single outputs requesting hosp_aug, hosp and hosp_aug in succession.

        Check each output's selected source, including both side-by-side panels.
        Reusing the first loaded source or the first output's source must fail.
        """
        run_generators([make_calibration(CA)], config, sources)
        # CHANGED: single plots used to ignore surveillance_source and use the first loaded source.
        for name, expected in (
            (f"quantiles_{CA}_filtered", ["CA hosp_aug"]),
            (f"quantiles_{CA}_full", ["CA"]),
            (f"quantiles_{CA}_side_by_side", ["CA hosp_aug", "CA hosp_aug"]),
        ):
            assert [drawn.surveillance[3] for drawn in capture.get_panels(name)] == expected

    def test_grid_uses_selected_source(self, capture, sources, config):
        """Grid outputs switching from hosp_aug to hosp and back to hosp_aug.

        Check the selected source for each output, including both side-by-side
        panels. Reusing the first output's source must fail this assertion.
        """
        run_generators([make_calibration(CA)], config, sources)
        for name, expected in (
            ("quantiles_grid_filtered", ["CA hosp_aug"]),
            ("quantiles_grid_full", ["CA"]),
            ("quantiles_grid_sidebyside", ["CA hosp_aug", "CA hosp_aug"]),
        ):
            assert [drawn.surveillance[3] for drawn in capture.get_panels(name)] == expected


class TestHiddenSurveillance:
    """Hidden surveillance in filtered output while full output keeps it visible.

    Check the layers and date ranges passed to the single and grid panels.
    """

    @pytest.fixture
    def config(self):
        variant_configs = [
            QuantilesOutputConfig(type="filtered", show_calibration=True, show_surveillance=False),
            QuantilesOutputConfig(type="full", show_calibration=True),
        ]
        return make_plots_config(single=True, grid=True, outputs=variant_configs)

    def test_single_filtered_not_clipped(self, capture, hosp_only, config):
        """Single filtered output with surveillance hidden.

        Keep the full calibration range and the base projection horizon.
        No surveillance is passed to the panel or used to clip the ribbons.
        """
        run_generators([make_calibration(CA)], config, hosp_only)
        # CHANGED: the ribbons used to be clipped to surveillance that was not shown.
        assert capture.summarize_output(f"quantiles_{CA}_filtered") == [
            make_expected_panel_summary(
                "California",
                frame_with_location(CALIB_FULL, "CA"),
                frame_with_location(PROJ_BASE_HORIZON, "CA"),
                None,
            )
        ]

    def test_grid_filtered_not_clipped(self, capture, hosp_only, config):
        """Grid outputs with surveillance hidden in filtered and visible in full.

        The filtered panel keeps the full calibration range and the base projection
        horizon. The full panel keeps those ranges and includes all surveillance dates.
        """
        run_generators([make_calibration(CA)], config, hosp_only)
        assert capture.summarize_output("quantiles_grid_filtered") == [
            make_expected_panel_summary(
                "California", frame_with_location(CALIB_FULL, "CA"), frame_with_location(PROJ_BASE_HORIZON, "CA"), None
            )
        ]
        assert capture.summarize_output("quantiles_grid_full") == [
            make_expected_panel_summary(
                "California",
                frame_with_location(CALIB_FULL, "CA"),
                frame_with_location(PROJ_BASE_HORIZON, "CA"),
                frame_with_location(SURV_ALL, "CA"),
            )
        ]


class TestSideBySideAllCalibrationClipped:
    """Filtered side-by-side panels starting after the last calibration date."""

    def test_clipped_calibration_stays_empty(self, capture, hosp_only):
        """Filtered panel starting January 20, after calibration ends on January 13.

        In both single and grid outputs, calibration stays empty while projection
        and surveillance start at January 20. The grid axis draws only projection
        ribbons and surveillance points.
        """
        variant_configs = [
            QuantilesOutputConfig(
                type="side_by_side", show_calibration=True, filtered_panel={"surveillance_start_date": "2024-01-20"}
            )
        ]
        run_generators(
            [make_calibration(CA)], make_plots_config(single=True, grid=True, outputs=variant_configs), hosp_only
        )
        # CHANGED: the fully clipped calibration used to come back as the full calibration.
        expected = make_expected_panel_summary(
            "California",
            (),
            ("2024-01-20", "2024-02-03", 3, "CA"),
            ("2024-01-20", "2024-01-27", 2, "CA"),
        )
        assert capture.summarize_output(f"quantiles_{CA}_side_by_side")[1] == expected
        assert capture.summarize_output("quantiles_grid_sidebyside")[1] == expected
        filtered_axis = capture.get_panels("quantiles_grid_sidebyside")[1].axis
        assert len(filtered_axis.collections) == 3  # projection CrI + IQR ribbons, surveillance scatter


class TestSideBySideOutputLimits:
    """Output-level surveillance limits inherited by side_by_side panels without their own limits."""

    def test_output_limit_inherited_by_both_panels(self, capture, hosp_only):
        """Single and grid side_by_side outputs requesting three surveillance points.

        Both panels keep three observations up to the reference date plus two
        later ones. Only the filtered panel clips projection to December 30,
        the first retained surveillance date; calibration is hidden in both.
        """
        variant_configs = [QuantilesOutputConfig(type="side_by_side", surveillance_points=3)]
        run_generators(
            [make_calibration(CA)], make_plots_config(single=True, grid=True, outputs=variant_configs), hosp_only
        )
        last_3 = ("2023-12-30", "2024-01-27", 5, "CA")
        for name in (f"quantiles_{CA}_side_by_side", "quantiles_grid_sidebyside"):
            assert capture.summarize_output(name) == [
                make_expected_panel_summary("California", None, frame_with_location(PROJ_BASE_HORIZON, "CA"), last_3),
                make_expected_panel_summary("California", None, ("2023-12-30", "2024-02-03", 6, "CA"), last_3),
            ]


class TestColorBool:
    """Reject boolean calibration or projection styling with a migration hint."""

    @pytest.mark.parametrize("layer", ["calibration", "projection"])
    def test_false_fails_validation(self, layer):
        """Setting either styling section to false points to outputs[].show_<layer>."""
        with pytest.raises(ValidationError, match=f"outputs\\[\\].show_{layer}: false"):
            make_plots_config(single=True, grid=True, **{layer: False})


class TestGenerationNotice:
    """Incomplete calibration generations in single and grid full plots."""

    def test_title_suffix_and_footnote(self, capture, hosp_only):
        """California completes one of three requested generations; New York has no notice.

        Check California's asterisk and explanatory footnote in single and grid
        full plots, while New York's title stays unmarked.
        """
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
    """Failed projections through generate_calibration_outputs().

    Insert two failed projections among five valid ones, with seven aligned
    parameter rows, and check the plots and data retained after filtering.
    """

    def test_plots_and_filtered_count(self, capture, hosp_only):
        """Dispatcher output generation with two failed projections among five valid ones.

        Check single and grid full plots, the five retained projections and their
        matching parameter rows, and a five-row parameter table. Plot generation
        must preserve the failure count recorded by dispatcher preprocessing.
        """
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
                make_expected_panel_summary(
                    "California",
                    None,
                    frame_with_location(PROJ_BASE_HORIZON, "CA"),
                    frame_with_location(SURV_ALL, "CA"),
                )
            ]
        assert len(calibration.results.projections["baseline"]) == 5
        assert calibration.results.projection_parameters["baseline"]["R0"].tolist() == [0.0, 2.0, 3.0, 4.0, 5.0]
        assert len(outputs["projection_parameters_long"][0].data) == 5
        # Plotting does not filter the shared results again or reset their failure count.
        assert calibration.results._filtered_count == 2


def test_summarize_frame():
    """Frame-summary helper with duplicate dates, an empty frame, and a missing frame.

    Check date bounds, the count of distinct dates and the location label.
    An empty frame produces (), while a missing frame stays None.
    """
    df = pd.DataFrame({"date": ["2024-01-06", "2024-01-13", "2024-01-13"], "value": [1500.0, 1600.0, 1700.0]})
    assert summarize_frame(df) == ("2024-01-06", "2024-01-13", 2, "CA")
    assert summarize_frame(df.iloc[0:0]) == ()
    assert summarize_frame(None) is None
