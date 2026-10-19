"""Custom snapshots stay with the runtime that owns them."""

from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from epymodelingsuite.builders.base import LocatedModel
from epymodelingsuite.builders.orchestrators import apply_vaccination_for_sampled_start
from epymodelingsuite.school_closures import add_school_closure_interventions
from epymodelingsuite.utils.resolved_location import ResolvedLocation, VaccinationContext
from epymodelingsuite.vaccinations import add_vaccination_schedule


def location():
    return ResolvedLocation(
        kind="custom",
        location_id="test-city",
        population_name="custom_location_test-city",
        source_iso="US-MA",
        location_name="Test city",
        location_name_short="Test",
        total_population=100,
        identifiers=(("location_id", "test-city"),),
    )


def vaccination():
    return VaccinationContext("US-MA", (("0+", 0.25),))


def test_custom_vaccination_scales_parent_doses_and_keeps_schedule(monkeypatch):
    model = MagicMock()
    model.population.name = "custom_location_test-city"
    model.population.Nk_names = ["0+"]
    schedule = pd.DataFrame({"location": ["US-MA", "US-MA"], "0+": [5.0, 7.0]})
    monkeypatch.setattr("epymodelingsuite.vaccinations.remove_vaccination_transitions", lambda model, *_: model)
    add_vaccination_schedule(model, lambda *_: None, "S", "V", schedule, context=vaccination())
    np.testing.assert_array_equal(model.add_transition.call_args.kwargs["params"][0], [[1], [2]])
    assert schedule["0+"].tolist() == [5.0, 7.0]


def test_vaccination_rejects_scaling_for_wrong_age_groups():
    model = MagicMock()
    model.population.Nk_names = ["0+"]
    with pytest.raises(ValueError, match="match model age groups"):
        add_vaccination_schedule(
            model,
            lambda *_: None,
            "S",
            "V",
            pd.DataFrame({"location": ["US-MA"], "0+": [4]}),
            context=VaccinationContext("US-MA", (("5+", 0.25),)),
        )


def test_sampled_start_reaggregates_before_passing_custom_snapshot(monkeypatch):
    snapshot = vaccination()
    original = pd.DataFrame({"location": ["US-MA"], "0+": [7]})
    reaggregated = pd.DataFrame({"location": ["US-MA"], "0+": [11]})
    timespan = SimpleNamespace(start_date=date(2024, 1, 2))
    reaggregate = MagicMock(return_value=reaggregated)
    add = MagicMock()
    monkeypatch.setattr("epymodelingsuite.builders.orchestrators.reaggregate_vaccines", reaggregate)
    monkeypatch.setattr("epymodelingsuite.builders.orchestrators.add_vaccination_schedules_from_config", add)
    basemodel = SimpleNamespace(vaccination=object(), transitions=[])
    apply_vaccination_for_sampled_start(object(), basemodel, timespan, original, object(), snapshot)
    reaggregate.assert_called_once_with(original, timespan.start_date)
    assert add.call_args.kwargs["use_schedule"] is reaggregated
    assert add.call_args.kwargs["context"] is snapshot


def test_custom_school_closure_uses_parent_region(monkeypatch):
    import epymodelingsuite.utils

    convert = MagicMock(return_value="MA")
    monkeypatch.setattr(epymodelingsuite.utils, "convert_location_name_format", convert)
    closure = SimpleNamespace(start_date=date(2024, 1, 1), end_date=date(2024, 1, 2), name="Winter")
    model = MagicMock()
    add_school_closure_interventions(model, {"MA": [closure]}, 0.2, location=location())
    convert.assert_called_once_with("US-MA", "abbreviation")
    model.add_intervention.assert_called_once_with(
        layer_name="school",
        start_date=closure.start_date,
        end_date=closure.end_date,
        reduction_factor=0.2,
        name="Winter",
    )
    with pytest.raises(ValueError, match="No school closure data"):
        add_school_closure_interventions(model, {}, 0.2, location=location())


@pytest.mark.parametrize(
    "mode",
    [
        "simulation",
        "simulation_error",
        "calibration",
        "calibration_error",
        "projection",
        "projection_error",
        "projection_calibration_error",
    ],
)
def test_runner_preserves_custom_snapshot_on_success_and_failure(monkeypatch, mode):
    from epymodelingsuite.dispatcher import runner

    snapshot = location()
    output = MagicMock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(runner, "SimulationOutput", output)
    monkeypatch.setattr(runner, "CalibrationOutput", output)
    monkeypatch.setattr(runner.ExecutionTelemetry, "get_current", lambda: None)
    model = MagicMock()
    model.population.name = snapshot.population_name
    calibrator = MagicMock()
    calibrator.parameters = {"epimodel": model}
    configs = SimpleNamespace(
        primary_id=1,
        seed=123,
        delta_t=1.0,
        model=model,
        simulation={},
        location=snapshot,
        calibrator=calibrator,
        start_date_reference=None,
        calibration=SimpleNamespace(name="test", options={}),
        projection=SimpleNamespace(end_date=date(2024, 1, 31), n_trajectories=1, generation_number=None),
    )
    if mode == "simulation_error":
        model.run_simulations.side_effect = ValueError("failed")
    elif mode in {"calibration_error", "projection_calibration_error"}:
        calibrator.calibrate.side_effect = ValueError("failed")
    elif mode == "projection_error":
        calibrator.run_projections.side_effect = ValueError("failed")
    function = (
        runner.run_simulation
        if mode.startswith("simulation")
        else (runner.run_calibration if mode.startswith("calibration") else runner.run_calibration_with_projection)
    )
    if mode in {"simulation_error", "calibration_error", "projection_calibration_error"}:
        with pytest.raises(RuntimeError, match="failed"):
            function(configs)
    else:
        result = function(configs)
        assert result.location is snapshot
    output.assert_called_once()
    assert output.call_args.kwargs["location"] is snapshot


@pytest.mark.parametrize("schedule_parent", ["US-MA", "US-CA"])
def test_sampled_custom_setup_uses_parent_and_rejects_missing_schedule(monkeypatch, schedule_parent):
    from epymodelingsuite.builders.orchestrators import setup_vaccination_schedules

    snapshot = location()
    model = MagicMock()
    model.population.name = snapshot.population_name
    schedule = pd.DataFrame({"location": [schedule_parent], "0+": [11]})
    scenario = MagicMock(return_value=schedule)
    monkeypatch.setattr("epymodelingsuite.builders.orchestrators.scenario_to_epydemix", scenario)
    timespan = SimpleNamespace(start_date=date(2024, 1, 1), end_date=date(2024, 1, 31))
    basemodel = SimpleNamespace(
        vaccination=SimpleNamespace(preprocessed_vaccination_data_path=None, scenario_data_path="ignored.csv"),
        population=SimpleNamespace(age_groups=["0+"]),
    )
    monkeypatch.setattr("epymodelingsuite.builders.orchestrators.prepare_vaccination_context", lambda *_: vaccination())
    args = (basemodel, [LocatedModel(model, snapshot)], timespan)
    if schedule_parent == "US-MA":
        _, earliest = setup_vaccination_schedules(*args)
        assert earliest is schedule
        assert earliest["0+"].tolist() == [11]
    else:
        with pytest.raises(ValueError, match="US-MA.*not found"):
            setup_vaccination_schedules(*args)
    assert scenario.call_args.kwargs["states"] == ["US-MA"]


def test_sampled_vaccination_rejects_unprepared_runtime():
    from epymodelingsuite.builders.orchestrators import make_simulate_wrapper

    with pytest.raises(ValueError, match="prepared vaccination context"):
        make_simulate_wrapper(
            basemodel=SimpleNamespace(vaccination=object()),
            calibration=None,
            observed_data=pd.DataFrame(),
            intervention_types=[],
            sampled_start_timespan=object(),
            location=location(),
        )
