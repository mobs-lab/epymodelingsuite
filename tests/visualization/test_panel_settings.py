"""Tests for quantile plot override resolution and per-panel data preparation (no matplotlib)."""

from dataclasses import replace
from datetime import date

import pandas as pd
import pytest

from epymodelingsuite.schema.output import QuantilesOutputConfig, QuantilesPlotConfig
from epymodelingsuite.visualization.generators import (
    LocationPlotData,
    ResolvedPanelSettings,
    prepare_panel_plot_data,
    resolve_panel_settings,
)


def resolve(output: dict, base: dict | None = None, loaded: list[str] | None = None) -> list[ResolvedPanelSettings]:
    return resolve_panel_settings(
        QuantilesPlotConfig(**(base or {})), QuantilesOutputConfig(**output), loaded if loaded is not None else []
    )


def limits(settings: ResolvedPanelSettings) -> tuple:
    return (settings.surveillance_points, settings.surveillance_start_date, settings.xlabel_interval)


JAN_1 = date(2024, 1, 1)
DEC_1 = date(2023, 12, 1)


class TestResolvePanelSettings:
    @pytest.mark.parametrize(
        ("output", "expected"),
        [
            # filtered / full: output > default
            ({"type": "filtered"}, [(None, None, None)]),
            ({"type": "full", "surveillance_points": 8, "xlabel_interval": "MS"}, [(8, None, "MS")]),
            # both kept at one level; prepare_panel_plot_data gives surveillance_start_date priority
            (
                {"type": "filtered", "surveillance_points": 8, "surveillance_start_date": "2024-01-01"},
                [(8, JAN_1, None)],
            ),
            # side_by_side: output values are inherited by panels that set nothing
            ({"type": "side_by_side", "surveillance_points": 8}, [(8, None, None), (8, None, None)]),
            # panel > output
            (
                {"type": "side_by_side", "surveillance_points": 8, "filtered_panel": {"surveillance_points": 4}},
                [(8, None, None), (4, None, None)],
            ),
            # points / start_date are inherited as a pair
            (
                {
                    "type": "side_by_side",
                    "surveillance_points": 8,
                    "filtered_panel": {"surveillance_start_date": "2024-01-01"},
                },
                [(8, None, None), (None, JAN_1, None)],
            ),
            (
                {
                    "type": "side_by_side",
                    "surveillance_start_date": "2023-12-01",
                    "full_panel": {"surveillance_points": 4},
                },
                [(4, None, None), (None, DEC_1, None)],
            ),
            # xlabel_interval is inherited on its own; an xlabel-only panel still inherits the limits
            (
                {
                    "type": "side_by_side",
                    "surveillance_points": 8,
                    "xlabel_interval": "MS",
                    "full_panel": {"xlabel_interval": "W-SAT"},
                },
                [(8, None, "W-SAT"), (8, None, "MS")],
            ),
            # null on the panel = inherit, not "show all"
            (
                {"type": "side_by_side", "surveillance_points": 8, "full_panel": {"surveillance_points": None}},
                [(8, None, None), (8, None, None)],
            ),
        ],
    )
    def test_surveillance_limits_and_xlabel(self, output, expected):
        """Surveillance limits and xlabel_interval follow panel > output > default."""
        assert [limits(settings) for settings in resolve(output)] == expected

    @pytest.mark.parametrize(
        ("base_horizon", "output_horizon", "expected"),
        [(3, None, 3), (3, 1, 1), (3, 6, 6), (3, 0, 0), (None, None, None), (None, 2, 2)],
    )
    def test_horizon_max(self, base_horizon, output_horizon, expected):
        """horizon_max is the output value, else the base value, for every output type."""
        for kind in ("filtered", "full", "side_by_side"):
            settings = resolve({"type": kind, "horizon_max": output_horizon}, base={"horizon_max": base_horizon})
            assert {panel.horizon_max for panel in settings} == {expected}

    @pytest.mark.parametrize(
        ("loaded", "selected", "expected"),
        [
            (["hosp"], "hosp", "hosp"),
            (["hosp", "alt"], "alt", "alt"),
            (["hosp"], None, "hosp"),  # the only source
            (["hosp"], "alt", "hosp"),  # selected source not loaded; the only loaded source is used
            (["hosp", "alt"], None, None),  # several sources, none selected
            (["hosp", "alt"], "missing", None),
            ([], None, None),
            ([], "hosp", None),
        ],
    )
    def test_surveillance_source(self, loaded, selected, expected):
        """The selected source if loaded, else the only loaded source, else none."""
        (settings,) = resolve({"type": "full", "surveillance_source": selected}, loaded=loaded)
        assert settings.surveillance_source == expected

    def test_views(self):
        """Filtered and full resolve to one panel; side_by_side to (full, filtered)."""
        assert [settings.view for settings in resolve({"type": "filtered"})] == ["filtered"]
        assert [settings.view for settings in resolve({"type": "full"})] == ["full"]
        assert [settings.view for settings in resolve({"type": "side_by_side"})] == ["full", "filtered"]

    def test_show_flags_and_colors(self):
        """Show flags come from the output and colors from the base config, for every panel."""
        output = {
            "type": "side_by_side",
            "show_calibration": True,
            "show_projection": False,
            "show_surveillance": False,
            "show_fitting_window_line": False,
        }
        base = {"calibration": {"color": "red"}, "projection": {"color": "blue"}}
        for settings in resolve(output, base=base):
            assert (settings.show_calibration, settings.show_projection) == (True, False)
            assert (settings.show_surveillance, settings.show_fitting_window_line) == (False, False)
            assert (settings.calibration_color, settings.projection_color) == ("red", "blue")


REFERENCE_DATE = date(2024, 1, 13)


def weekly_frame(start: str, end: str, *, quantiles: bool = True) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="W-SAT")
    if not quantiles:
        return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "value": range(len(dates))})
    return pd.DataFrame(
        [{"date": d.date(), "quantile": q, "value": float(i)} for i, d in enumerate(dates) for q in (0.025, 0.5, 0.975)]
    )


def date_range_of(df: pd.DataFrame | None) -> tuple | None:
    if df is None:
        return None
    if df.empty:
        return ()
    dates = pd.to_datetime(df["date"]).dt.date
    return (dates.min().isoformat(), dates.max().isoformat(), dates.nunique())


@pytest.fixture
def location_data() -> LocationPlotData:
    return LocationPlotData(
        location="United_States_California",
        calibration_quantiles=weekly_frame("2023-11-04", "2024-01-13"),  # 11 weeks
        projection_quantiles=weekly_frame("2023-12-02", "2024-03-02"),  # 14 weeks
        surveillance={"hosp": weekly_frame("2023-10-07", "2024-01-27", quantiles=False)},  # 17 weeks
        fitting_window_start=date(2023, 11, 4),
        fitting_window_end=date(2024, 1, 13),
        title_suffix="",
        footnote="",
    )


def settings_for(view: str, **overrides) -> ResolvedPanelSettings:
    defaults = ResolvedPanelSettings(
        view=view,
        surveillance_source="hosp",
        surveillance_points=None,
        surveillance_start_date=None,
        horizon_max=3,
        xlabel_interval=None,
        show_calibration=True,
        show_projection=True,
        show_surveillance=True,
        show_fitting_window_line=True,
        calibration_color="C0",
        projection_color="C1",
    )
    return replace(defaults, **overrides)


def prepared(location_data, view, **overrides) -> dict:
    panel = prepare_panel_plot_data(location_data, settings_for(view, **overrides), REFERENCE_DATE)
    return {
        "calibration": date_range_of(panel.calibration),
        "projection": date_range_of(panel.projection),
        "surveillance": date_range_of(panel.surveillance),
    }


CAL_ALL = ("2023-11-04", "2024-01-13", 11)
PROJ_TO_HORIZON = ("2023-12-02", "2024-02-03", 10)  # reference_date + 3 weeks
SURV_ALL = ("2023-10-07", "2024-01-27", 17)


class TestPreparePanelPlotData:
    def test_full_without_limits_keeps_everything(self, location_data):
        """Without limits, the full view keeps everything except projection past the horizon."""
        assert prepared(location_data, "full") == {
            "calibration": CAL_ALL,
            "projection": PROJ_TO_HORIZON,
            "surveillance": SURV_ALL,
        }

    def test_full_with_limit_does_not_clip_ribbons(self, location_data):
        """A surveillance limit in the full view trims surveillance only."""
        assert prepared(location_data, "full", surveillance_points=2) == {
            "calibration": CAL_ALL,
            "projection": PROJ_TO_HORIZON,
            "surveillance": ("2024-01-06", "2024-01-27", 4),
        }

    def test_filtered_default_starts_at_projection(self, location_data):
        """Without limits, the filtered view starts at the projection start."""
        assert prepared(location_data, "filtered") == {
            "calibration": ("2023-12-02", "2024-01-13", 7),
            "projection": PROJ_TO_HORIZON,
            "surveillance": ("2023-12-02", "2024-01-27", 9),
        }

    def test_filtered_default_falls_back_to_calibration_start(self, location_data):
        """Without a projection, the filtered view starts at the calibration start."""
        without_projection = replace(location_data, projection_quantiles=None)
        assert prepared(without_projection, "filtered")["surveillance"] == ("2023-11-04", "2024-01-27", 13)

    def test_filtered_points_count_all_observations(self, location_data):
        # 8 points on or before reference_date, even though the projection starts later; plus 2 after it
        """surveillance_points counts every observation up to reference_date."""
        assert prepared(location_data, "filtered", surveillance_points=8) == {
            "calibration": ("2023-11-25", "2024-01-13", 8),
            "projection": PROJ_TO_HORIZON,
            "surveillance": ("2023-11-25", "2024-01-27", 10),
        }

    def test_filtered_start_date_before_projection_start(self, location_data):
        """surveillance_start_date can start the filtered view before the projection."""
        assert prepared(location_data, "filtered", surveillance_start_date=date(2023, 11, 11)) == {
            "calibration": ("2023-11-11", "2024-01-13", 10),
            "projection": PROJ_TO_HORIZON,
            "surveillance": ("2023-11-11", "2024-01-27", 12),
        }

    def test_start_date_wins_over_points(self, location_data):
        """surveillance_start_date takes priority over surveillance_points."""
        result = prepared(location_data, "filtered", surveillance_points=2, surveillance_start_date=date(2023, 12, 16))
        assert result["surveillance"] == ("2023-12-16", "2024-01-27", 7)

    def test_filtered_clips_projection_to_first_surveillance_date(self, location_data):
        """The filtered projection starts at the first visible surveillance date."""
        result = prepared(location_data, "filtered", surveillance_points=1)
        assert result["projection"] == ("2024-01-13", "2024-02-03", 4)

    @pytest.mark.parametrize(
        ("horizon_max", "expected"),
        [
            (None, ("2023-12-02", "2024-03-02", 14)),
            (0, ("2023-12-02", "2024-01-13", 7)),
            (6, ("2023-12-02", "2024-02-24", 13)),
        ],
    )
    def test_horizon(self, location_data, horizon_max, expected):
        """horizon_max clips the projection; None means no limit."""
        assert prepared(location_data, "full", horizon_max=horizon_max)["projection"] == expected

    def test_fully_clipped_frames_stay_empty(self, location_data):
        """A frame clipped away entirely stays empty instead of falling back to unclipped data."""
        result = prepared(location_data, "filtered", surveillance_start_date=date(2024, 1, 20))
        assert result == {
            "calibration": (),
            "projection": ("2024-01-20", "2024-02-03", 3),
            "surveillance": ("2024-01-20", "2024-01-27", 2),
        }

    def test_hidden_layers_are_none(self, location_data):
        """Hidden calibration and projection are returned as None."""
        result = prepared(location_data, "full", show_calibration=False, show_projection=False)
        assert result == {"calibration": None, "projection": None, "surveillance": SURV_ALL}

    @pytest.mark.parametrize(
        "overrides",
        [{"show_surveillance": False}, {"surveillance_source": None}, {"surveillance_source": "not_for_location"}],
    )
    def test_no_surveillance_means_no_clipping(self, location_data, overrides):
        """Without shown surveillance, the filtered ribbons are not clipped."""
        assert prepared(location_data, "filtered", surveillance_points=2, **overrides) == {
            "calibration": CAL_ALL,
            "projection": PROJ_TO_HORIZON,
            "surveillance": None,
        }

    def test_does_not_mutate_location_data(self, location_data):
        """Preparing a panel does not modify the collected location data."""
        before = location_data.projection_quantiles.copy()
        prepared(location_data, "filtered", surveillance_points=1, horizon_max=0)
        pd.testing.assert_frame_equal(location_data.projection_quantiles, before)
