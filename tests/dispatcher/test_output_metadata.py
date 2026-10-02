"""Regression checks for metadata on wide, fragmented result frames."""

import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from epymodelingsuite.dispatcher.output import (
    format_quantiles_flusightforecast,
    generate_calibration_outputs,
    generate_simulation_outputs,
    make_prop_ed_flusightforecast,
    prepend_metadata_columns,
)
from epymodelingsuite.schema.output import FlusightPropED, OutputConfig


@pytest.mark.parametrize("workflow", ["simulation", "calibration"])
@pytest.mark.parametrize("generations", [True, [0, 1]])
def test_wide_output_metadata(workflow: str, *, generations: bool | list[int]) -> None:
    dates = pd.date_range("2025-11-29", periods=2, freq="7D")
    quantiles = pd.DataFrame({"date": dates, "quantile": [0.5, 0.5]}, index=[4, 9])
    # Reproduce the fragmented frames returned by upstream quantile builders.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        for i in range(110):
            quantiles[f"I_{i}"] = [i, i + 1]
            quantiles[f"S_to_I_{i}"] = [i + 2, i + 3]
    original = quantiles.copy()
    compartments = {c: quantiles[c].to_numpy() for c in quantiles if c.startswith("I_")}
    transitions = {c: quantiles[c].to_numpy() for c in quantiles if "_to_" in c}
    result = MagicMock()
    result.projections = {}
    result.dates = dates
    result.Nsim = 1
    result.parameters = {}
    result.get_stacked_compartments.return_value = {key: [value] for key, value in compartments.items()}
    result.get_quantiles_compartments.return_value = quantiles
    result.get_quantiles_transitions.return_value = quantiles
    result.get_calibration_quantiles.return_value = quantiles
    result.get_projection_quantiles.return_value = quantiles
    result.get_posterior_distribution.return_value = quantiles
    result.get_projection_trajectories.return_value = {
        key: [value] for key, value in {"date": dates, **compartments, **transitions}.items()
    }
    result.trajectories = [SimpleNamespace(dates=dates, compartments=compartments, transitions=transitions)]
    model = SimpleNamespace(primary_id=7, seed=42, delta_t=1, population="United_States_California", results=result)
    config = OutputConfig.model_validate(
        {
            "output": {
                "tabular_output_types": ["DataFrame"],
                "quantiles": {"compartments": True, "transitions": True, "calibration": generations},
                "trajectories": {"compartments": True, "transitions": True},
                "posteriors": True if generations is True else {"generations": generations},
            }
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.PerformanceWarning)
        if workflow == "simulation":
            outputs = generate_simulation_outputs(simulations=[model], output_config=config)
            expected_keys = {
                "quantiles_compartments",
                "quantiles_transitions",
                "trajectories_compartments",
                "trajectories_transitions",
            }
        else:
            outputs = generate_calibration_outputs(calibrations=[model], output_config=config)
            expected_keys = {
                "quantiles_projection_compartments",
                "quantiles_projection_transitions",
                "quantiles_calibration",
                "trajectories_projection_compartments",
                "trajectories_projection_transitions",
                "posteriors",
            }

    assert expected_keys <= outputs.keys()
    for name in expected_keys:
        frame = outputs[name][0].data
        prefix = ["primary_id", "seed", "population"]
        expected_data = original
        if "trajectories" in name:
            prefix = (
                ["primary_id", "sim_id", "date", "seed", "population"]
                if workflow == "simulation"
                else ["primary_id", "sim_id", "seed", "population", "date"]
            )
            values = compartments if name.endswith("compartments") else transitions
            expected_data = pd.DataFrame(values)
            assert frame["sim_id"].tolist() == [0, 0]
        elif workflow == "calibration":
            if name.endswith("compartments"):
                expected_data = original[["date", "quantile", *compartments]]
            elif name.endswith("transitions"):
                expected_data = original[["date", "quantile", *transitions]]
            elif generations is not True:
                prefix.append("generation")
                expected_data = pd.concat([original, original], ignore_index=True)
                assert frame["generation"].tolist() == [0, 0, 1, 1]
        assert list(frame.columns) == prefix + list(expected_data.columns)
        assert frame["primary_id"].tolist() == [7] * len(frame)
        assert frame["seed"].tolist() == [42] * len(frame)
        assert frame["population"].tolist() == [model.population] * len(frame)
        assert frame["date"].tolist() == list(dates) * (len(frame) // len(dates))
        pd.testing.assert_frame_equal(
            frame[list(expected_data)].reset_index(drop=True), expected_data.reset_index(drop=True)
        )
    pd.testing.assert_frame_equal(quantiles, original)


@pytest.mark.parametrize("index", [[], [4, 9], [4, 4]])
def test_prepend_metadata_preserves_frame(index: list[int]) -> None:
    frame = pd.DataFrame({"value": pd.array(range(len(index)), dtype="Int64")}, index=index)
    original = frame.copy()
    metadata = {"seed": None, "date": pd.date_range("2025-11-29", periods=len(index))}
    metadata["aligned"] = pd.Series(range(len(index)), index=index, dtype="int64").iloc[::-1]
    metadata.update({f"field_{i}": i for i in range(110)})
    expected = frame.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        for position, (name, value) in enumerate(metadata.items()):
            expected.insert(position, name, value)
    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.PerformanceWarning)
        actual = prepend_metadata_columns(frame, **metadata)
    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(frame, original)
    with pytest.raises(ValueError, match="overlapping"):
        prepend_metadata_columns(frame, value=0)


@pytest.mark.parametrize("metrocast", [False, True])
@pytest.mark.parametrize("forecast", ["hospitalizations", "prop_ed"])
def test_forecast_metadata_order_and_values(forecast: str, *, metrocast: bool) -> None:
    dates = pd.date_range("2025-11-15", periods=4, freq="7D")
    reference_date = dates[2].date()
    frame = pd.DataFrame(
        {"date": dates, "quantile": 0.5, "hospitalizations": [10.1, 20.2, 30.3, 40.4]}, index=[9, 3, 9, 3]
    )
    if forecast == "prop_ed":
        frame["population"] = "United_States_California"
    original = frame.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.PerformanceWarning)
        if forecast == "hospitalizations":
            actual = format_quantiles_flusightforecast(frame, reference_date, metrocast=metrocast)
        else:
            actual, factors = make_prop_ed_flusightforecast(
                pd.DataFrame(),
                FlusightPropED(strategy="transition", transition_name="hospitalizations"),
                {},
                projection_quantiles=frame,
                reference_date=reference_date,
                metrocast=metrocast,
            )
            assert factors.empty
    start = 2 if metrocast else 1
    expected = pd.DataFrame(
        {
            "horizon": list(range(-2, 2))[start:],
            "target_end_date": [d.date() for d in dates[start:]],
            "target": "wk inc flu hosp" if forecast == "hospitalizations" else "wk inc flu prop ed visits",
            "output_type": "quantile",
            "output_type_id": 0.5,
            "value": pd.array([10, 20, 30, 40][start:], dtype="Int64")
            if forecast == "hospitalizations"
            else [10.1, 20.2, 30.3, 40.4][start:],
        },
        index=frame.index[start:],
    )
    if forecast == "prop_ed":
        expected.insert(0, "reference_date", reference_date)
        expected.insert(0, "location", "06")
    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(frame, original)
