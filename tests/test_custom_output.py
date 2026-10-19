"""Custom output keeps location identity and labels without reopening its CSVs."""

import pickle
from dataclasses import replace
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from epydemix.calibration import CalibrationResults

from epymodelingsuite.dispatcher.output import _validate_custom_output_locations, generate_calibration_outputs
from epymodelingsuite.multistrain.formatter import create_submission
from epymodelingsuite.schema.dispatcher import CalibrationOutput
from epymodelingsuite.schema.output import (
    FlusightTrajectorySamples,
    ObservedValuesConfig,
    OutputConfig,
    QuantilesPlotConfig,
)
from epymodelingsuite.utils.custom_locations import CustomLocation
from epymodelingsuite.visualization.core import format_location_name, sort_locations_by_state
from epymodelingsuite.visualization.generators import _draw_quantile_panel, get_locations_to_plot
from epymodelingsuite.visualization.preparation import PanelPlotData, _select_surveillance


@pytest.fixture
def custom() -> CustomLocation:
    return CustomLocation("boston-all", "US-MA", "City of Boston", "Boston", 675647)


def test_custom_plot_selection_and_metadata_validation(custom: CustomLocation) -> None:
    result = SimpleNamespace(population=custom.population_name, custom_location=custom)
    assert QuantilesPlotConfig(single=[custom.location_id]).single == [custom.location_id]
    assert get_locations_to_plot([result], [custom.location_id]) == {custom.population_name}
    assert get_locations_to_plot([result], [custom.population_name]) == {custom.population_name}
    with pytest.raises(ValueError, match="not found in results"):
        get_locations_to_plot([result], ["missing-id"])
    output = SimpleNamespace(flusight_format=None)
    _validate_custom_output_locations([result], output)
    result.custom_location = replace(custom, location_id="another-id")
    with pytest.raises(ValueError, match="inconsistent"):
        _validate_custom_output_locations([result], output)


def test_custom_official_flusight_is_rejected(custom: CustomLocation) -> None:
    result = SimpleNamespace(population=custom.population_name, custom_location=custom)
    output = SimpleNamespace(flusight_format=SimpleNamespace(metrocast=False))
    with pytest.raises(ValueError, match="official FluSight"):
        _validate_custom_output_locations([result], output)
    output.flusight_format = SimpleNamespace(metrocast=True, prop_ed=SimpleNamespace(strategy="surveillance_window"))
    with pytest.raises(ValueError, match="window-based prop ED"):
        _validate_custom_output_locations([result], output)


def test_custom_surveillance_and_quantile_panel(custom: CustomLocation) -> None:
    surveillance = pd.DataFrame({"location": ["boston-all", "other"], "date": ["2026-10-10"] * 2, "cases": [5, 999]})
    source = ObservedValuesConfig(
        data_path="unused.csv",
        location_column="location",
        location_format="location_id",
        date_column="date",
        value_column="cases",
    )
    selected = _select_surveillance(surveillance, custom.population_name, source, custom)
    assert selected["value"].tolist() == [5]
    quantiles = pd.DataFrame({"date": pd.to_datetime(["2026-10-10"]), "quantile": [0.5], "value": [5]})
    panel = PanelPlotData(
        location=custom.population_name,
        calibration_quantiles=quantiles,
        projection_quantiles=None,
        surveillance=selected,
        calibration_color="C0",
        projection_color="C1",
        xlabel_interval=None,
        fitting_window_start=None,
        fitting_window_end=None,
        title_suffix="",
        footnote="",
        custom_location=custom,
    )
    fig, ax = plt.subplots()
    try:
        _draw_quantile_panel(panel, ax, None)
        assert ax.get_title() == "Boston"
    finally:
        plt.close(fig)
    assert format_location_name(custom.population_name, custom) == "Boston"
    with pytest.raises(ValueError, match="metadata"):
        format_location_name(custom.population_name)
    assert sort_locations_by_state([custom.population_name, "US-CA"], {custom.population_name: custom}) == [
        "US-CA",
        custom.population_name,
    ]


@pytest.mark.parametrize("profile", ["bphc_ed", "metrocast"])
def test_custom_multistrain_quantiles_and_samples_use_original_id(custom: CustomLocation, profile: str) -> None:
    dates = pd.date_range("2026-09-26", periods=7, freq="7D")
    trajectories = pd.DataFrame(
        [(custom.population_name, date, sample, sample + 1.0) for date in dates for sample in range(3)],
        columns=["location", "date", "sample_id", "target_total"],
    )
    submission = create_submission(
        trajectories,
        "2026-10-10",
        profile,
        trajectory_samples=FlusightTrajectorySamples(n_samples=2, seed=1),
    )
    assert set(submission["location"]) == {custom.location_id}
    assert set(submission["output_type"]) == {"quantile", "sample"}
    samples = submission[submission.output_type == "sample"]
    assert set(samples.output_type_id) == {"1", "2"}
    assert np.isfinite(submission.value).all()


def test_saved_custom_result_generates_hub_quantiles_and_samples(custom: CustomLocation) -> None:
    dates = pd.date_range("2026-09-26", periods=7, freq="7D").to_pydatetime().tolist()
    results = CalibrationResults(
        projections={"baseline": [{"date": dates, "hospitalizations": np.arange(7) + sample} for sample in range(3)]}
    )
    calibration = CalibrationOutput(
        primary_id=1, population=custom.population_name, results=results, custom_location=custom
    )
    restored = pickle.loads(pickle.dumps(calibration))  # noqa: S301 -- bytes created by this test
    config = OutputConfig.model_validate(
        {
            "output": {
                "flusight_format": {
                    "reference_date": "2026-10-10",
                    "metrocast": True,
                    "horizons": [-1, 0, 1, 2, 3],
                    "samples": {"n_samples": 2, "seed": 1},
                    "hospitalizations": {"target": "wk inc ed signal"},
                },
                "tabular_output_types": ["DataFrame"],
            }
        }
    )
    output = generate_calibration_outputs(calibrations=[restored], output_config=config)
    hub = output["output_hub_formatted"][0]
    table = hub.data
    assert set(table.location) == {custom.location_id}
    assert set(table.output_type) == {"sample"}
    assert set(table.horizon) == {-1, 0, 1, 2, 3}
    config.output.flusight_format.samples = None
    quantiles = generate_calibration_outputs(calibrations=[restored], output_config=config)["output_hub_formatted"][
        0
    ].data
    assert set(quantiles.location) == {custom.location_id}
    assert set(quantiles.output_type) == {"quantile"}


def test_metadata_accepts_numpy_selected_trajectories(custom: CustomLocation) -> None:
    dates = pd.date_range("2026-10-03", periods=2, freq="7D").to_pydatetime()
    selected = np.array([{"date": dates, "data": np.array([1, 2])}] * 2, dtype=object)
    projections = np.array([{"date": dates, "hospitalizations": np.array([1, 2])}] * 2, dtype=object)
    results = CalibrationResults(selected_trajectories={0: selected}, projections={"baseline": projections})
    calibration = CalibrationOutput(
        primary_id=1, population=custom.population_name, results=results, custom_location=custom
    )
    config = OutputConfig.model_validate({"output": {"model_meta": {}, "tabular_output_types": ["DataFrame"]}})
    output = generate_calibration_outputs(calibrations=[calibration], output_config=config)
    metadata = output["model_metadata"][0].data.iloc[0]
    assert metadata.fitting_start == "2026-10-03"
    assert metadata.fitting_end == "2026-10-10"
    assert metadata.start_date == "2026-10-03"
    assert metadata.end_date == "2026-10-10"
