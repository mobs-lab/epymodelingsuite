"""Shared data and capture helpers for the quantile plot characterization tests.

Builds real ``CalibrationResults`` for a few locations and captures every call to
``plot_calibration_projection`` while still drawing real (Agg) figures.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import matplotlib as mpl
import numpy as np
import pandas as pd
from epydemix.calibration import CalibrationResults

from epymodelingsuite.schema.calibration import CalibrationStrategy
from epymodelingsuite.schema.dispatcher import CalibrationOutput
from epymodelingsuite.schema.output import ObservedValuesConfig, PlotsConfig, QuantilesPlotConfig
from epymodelingsuite.visualization import core, generators

mpl.use("Agg")

REFERENCE_DATE = date(2024, 1, 13)
QUANTILES = [0.025, 0.25, 0.5, 0.75, 0.975]

# Each location's values are offset by a location number * 1000, so a captured frame tells which location it
# came from. The second surveillance source ("hosp_aug") adds 5 to the location number.
LOCATION_NUMBERS = {
    "United_States_California": 1,
    "United_States_New_York": 2,
    "United_States_Texas": 3,
}
ISO_CODES = {
    "United_States_California": "US-CA",
    "United_States_New_York": "US-NY",
    "United_States_Texas": "US-TX",
}
HOSP_AUG_SOURCE_OFFSET = 5
# Label that span() reports for each value range
VALUE_RANGE_LABELS = {1: "CA", 2: "NY", 3: "TX", 6: "CA hosp_aug", 7: "NY hosp_aug", 8: "TX hosp_aug"}

SURVEILLANCE_DATES = pd.date_range("2023-10-07", "2024-01-27", freq="W-SAT")  # 17 weeks
CALIBRATION_DATES = pd.date_range("2023-11-04", "2024-01-13", freq="W-SAT")  # 11 weeks
PROJECTION_DATES = pd.date_range("2023-12-02", "2024-03-02", freq="W-SAT")  # 14 weeks


def make_calibration(population: str, *, with_projection: bool = True, incomplete: bool = False) -> CalibrationOutput:
    """Build a CalibrationOutput with real CalibrationResults for one location."""
    location_number = LOCATION_NUMBERS[population]
    n_draws = 5
    calibration_dates = [d.date() for d in CALIBRATION_DATES]
    projection_dates = [d.date() for d in PROJECTION_DATES]
    selected = [
        {"date": calibration_dates, "data": location_number * 1000.0 + np.arange(len(calibration_dates)) + draw}
        for draw in range(n_draws)
    ]
    projections = {}
    if with_projection:
        projections["baseline"] = [
            {
                "date": projection_dates,
                "hospitalizations": location_number * 1000.0 + 500 + np.arange(len(projection_dates)) + draw,
            }
            for draw in range(n_draws)
        ]
    results = CalibrationResults(
        selected_trajectories={0: selected},
        posterior_distributions={0: pd.DataFrame({"R0": np.linspace(1.1, 1.5, n_draws)})},
        projections=projections,
    )
    strategy = CalibrationStrategy(name="SMC", options={"num_generations": 3}) if incomplete else None
    return CalibrationOutput(
        primary_id=location_number,
        seed=location_number,
        population=population,
        results=results,
        calibration_strategy=strategy,
    )


def write_surveillance_sources(tmp_path) -> dict[str, ObservedValuesConfig]:
    """Write synthetic "hosp" and "hosp_aug" CSVs covering every location.

    "hosp_aug" stands in for an augmented hospitalizations source; the offset only
    distinguishes source selection and does not perform real augmentation.
    """
    sources = {}
    for name, source_offset in (("hosp", 0), ("hosp_aug", HOSP_AUG_SOURCE_OFFSET)):
        rows = [
            {
                "week_end": d.date().isoformat(),
                "location": ISO_CODES[population],
                "observed": (location_number + source_offset) * 1000 + i,
            }
            for population, location_number in LOCATION_NUMBERS.items()
            for i, d in enumerate(SURVEILLANCE_DATES)
        ]
        path = tmp_path / f"{name}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        sources[name] = ObservedValuesConfig(
            data_path=str(path), value_column="observed", date_column="week_end", location_column="location"
        )
    return sources


def make_plots_config(**quantiles_kwargs) -> PlotsConfig:
    return PlotsConfig(
        reference_date=REFERENCE_DATE,
        figure_output_types=["MPLFigure"],
        quantiles=QuantilesPlotConfig(**quantiles_kwargs),
    )


def span(df: pd.DataFrame | None) -> tuple | None:
    """Summarize a captured frame as (first date, last date, number of dates, location label such as "CA")."""
    if df is None:
        return None
    if df.empty:
        return ()
    dates = pd.to_datetime(df["date"]).dt.date
    return (
        dates.min().isoformat(),
        dates.max().isoformat(),
        dates.nunique(),
        VALUE_RANGE_LABELS[int(df["value"].min() // 1000)],
    )


@dataclass
class Panel:
    """One call to plot_calibration_projection, i.e. one drawn axis."""

    output: str | None  # output key the figure was packaged under
    title: str
    calibration: tuple | None
    projection: tuple | None
    surveillance: tuple | None
    quantile_levels: dict[str, list[float]]
    fitting_window: tuple | None
    calibration_color: str
    projection_color: str
    xlabel_interval: str | None
    ylabel: str | None
    axis: object


class PlotCapture:
    """Wrap plot_calibration_projection in both namespaces and record every panel drawn."""

    def __init__(self, monkeypatch):
        self.panels: list[Panel] = []
        self.packaged: dict[str, object] = {}  # output name -> figure
        real_plot = core.plot_calibration_projection
        real_package = generators._package_figure_outputs

        def capture_plot(**kwargs):
            fig, ax = real_plot(**kwargs)
            start, end = kwargs.get("fitting_window_start"), kwargs.get("fitting_window_end")
            self.panels.append(
                Panel(
                    output=None,
                    title=kwargs.get("title"),
                    calibration=span(kwargs.get("calibration_quantiles")),
                    projection=span(kwargs.get("projection_quantiles")),
                    surveillance=span(kwargs.get("df_surveillance")),
                    quantile_levels={
                        name: sorted(df["quantile"].unique())
                        for name, df in kwargs.items()
                        if name in ("calibration_quantiles", "projection_quantiles") and df is not None
                    },
                    fitting_window=None if start is None else (str(start), str(end)),
                    calibration_color=kwargs.get("calibration_color"),
                    projection_color=kwargs.get("projection_color"),
                    xlabel_interval=kwargs.get("xlabel_interval"),
                    ylabel=kwargs.get("ylabel"),
                    axis=ax,
                )
            )
            return fig, ax

        def capture_package(fig, name, plots_config):
            self.packaged[name] = fig
            for panel in self.panels:
                if panel.output is None and panel.axis.figure is fig:
                    panel.output = name
            return real_package(fig, name, plots_config)

        monkeypatch.setattr(core, "plot_calibration_projection", capture_plot)
        monkeypatch.setattr(generators, "plot_calibration_projection", capture_plot)
        monkeypatch.setattr(generators, "_package_figure_outputs", capture_package)

    def panels_for(self, output: str) -> list[Panel]:
        return [panel for panel in self.panels if panel.output == output]

    def summary(self, output: str) -> list[dict]:
        """Per-panel data summary for one output, in drawing order."""
        return [
            {
                "title": panel.title,
                "calibration": panel.calibration,
                "projection": panel.projection,
                "surveillance": panel.surveillance,
            }
            for panel in self.panels_for(output)
        ]
