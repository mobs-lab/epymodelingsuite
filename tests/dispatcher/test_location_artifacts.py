"""Canonical ownership and migration of real pre-location pickle states."""

import pickle
from dataclasses import replace

import numpy as np
import pytest
from epydemix.calibration import ABCSampler
from epydemix.model import EpiModel
from epydemix.population import Population
from pydantic import BaseModel, ValidationError

from epymodelingsuite.schema.dispatcher import BuilderOutput, CalibrationOutput, SimulationOutput
from epymodelingsuite.utils.location import resolve_location
from epymodelingsuite.utils.resolved_location import ResolvedLocation


def make_model(name):
    population = Population(name=name)
    population.add_population(Nk=np.array([100.0]), Nk_names=["0+"])
    model = EpiModel()
    model.set_population(population)
    return model


def legacy_pickle(output_type, fields):
    """Create the exact Pydantic v2 pickle state emitted before location existed."""
    legacy = output_type.__new__(output_type)
    BaseModel.__setstate__(
        legacy,
        {
            "__dict__": fields,
            "__pydantic_fields_set__": set(fields),
            "__pydantic_extra__": None,
            "__pydantic_private__": None,
        },
    )
    assert "location" not in legacy.__getstate__()["__dict__"]
    return pickle.dumps(legacy)


@pytest.mark.parametrize("output_type", [SimulationOutput, CalibrationOutput])
@pytest.mark.parametrize("population", ["United_States_Massachusetts", "metrocast_location_denver"])
def test_legacy_result_constructor_owns_one_canonical_location(output_type, population):
    output = output_type(primary_id=1, population=population, results=None)
    assert output.location == resolve_location(population)
    assert output.population == output.location.population_name
    assert "population" not in output.__dict__
    assert "population" not in output.model_dump()
    with pytest.raises(AttributeError):
        output.population = "US-CA"


@pytest.mark.parametrize("output_type", [SimulationOutput, CalibrationOutput])
@pytest.mark.parametrize("population", ["United_States_Massachusetts", "metrocast_location_denver"])
def test_actual_old_result_pickle_migrates_without_source_data(output_type, population, monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail("Builtin legacy artifacts must resolve from bundled data")

    monkeypatch.setattr("epymodelingsuite.utils.epydemix_data.get_available_locations", no_network)
    artifact = legacy_pickle(
        output_type,
        {"primary_id": 1, "seed": 123, "delta_t": 1.0, "population": population, "results": None},
    )
    restored = pickle.loads(artifact)
    assert restored.location == resolve_location(population)
    assert restored.population == restored.location.population_name
    assert "population" not in restored.__dict__
    assert "population" not in restored.model_fields_set
    assert "location" in restored.model_fields_set
    assert pickle.loads(pickle.dumps(restored)).location == restored.location


def test_legacy_builder_constructor_and_pickle_infer_builtin_model_location():
    model = make_model("United_States_Massachusetts")
    output = BuilderOutput(primary_id=1, model=model)
    assert output.location == resolve_location("US-MA")
    artifact = legacy_pickle(BuilderOutput, {"primary_id": 1, "model": model, "calibrator": None})
    restored = pickle.loads(artifact)
    assert restored.location == output.location
    assert restored.model.population.name == "United_States_Massachusetts"


def test_builder_checks_both_model_and_calibrator_location():
    location = resolve_location("US-MA")
    model = make_model(location.population_name)
    other_model = make_model(resolve_location("US-CA").population_name)
    calibrator = ABCSampler(
        simulation_function=lambda params: {"data": np.array([1])},
        priors={},
        parameters={"epimodel": other_model},
        observed_data=np.array([1]),
        distance_function=lambda observed, simulated: 0,
    )
    with pytest.raises(ValidationError, match="model and calibrator"):
        BuilderOutput(primary_id=1, location=location, model=other_model)
    with pytest.raises(ValidationError, match="model and calibrator"):
        BuilderOutput(primary_id=1, location=location, model=model, calibrator=calibrator)
    inferred = BuilderOutput(primary_id=1, calibrator=calibrator)
    assert inferred.location == resolve_location("US-CA")


@pytest.mark.parametrize("output_type", [SimulationOutput, CalibrationOutput])
def test_explicit_location_accepts_alias_and_rejects_conflicting_population(output_type):
    location = resolve_location("US-MA")
    assert (
        output_type(primary_id=1, location=location, population="United_States_Massachusetts", results=None).location
        is location
    )
    with pytest.raises(ValidationError, match="Population must match"):
        output_type(primary_id=1, location=location, population="US-CA", results=None)


@pytest.mark.parametrize("output_type", [SimulationOutput, CalibrationOutput])
def test_custom_result_requires_snapshot_and_survives_pickle(output_type):
    location = ResolvedLocation(
        kind="custom",
        location_id="city",
        population_name="custom_location_city",
        location_name="City",
        location_name_short="City",
        source_iso="US-MA",
        identifiers=(("location_id", "city"),),
        total_population=100,
    )
    result = output_type(primary_id=1, location=location, results=None)
    assert pickle.loads(pickle.dumps(result)).location == location
    assert output_type.model_validate(result.model_dump()).location == location
    with pytest.raises(ValidationError, match="explicit resolved location"):
        output_type(primary_id=1, population="custom_location_city", results=None)
    with pytest.raises(ValueError, match="explicit resolved location"):
        pickle.loads(
            legacy_pickle(output_type, {"primary_id": 1, "population": "custom_location_city", "results": None})
        )
    with pytest.raises(ValidationError, match="Population must match"):
        output_type(
            primary_id=1,
            location=replace(location, location_id="other", population_name="custom_location_other"),
            population="custom_location_city",
            results=None,
        )


def test_actual_pickle_rejects_snapshot_mismatched_to_model():
    builder = BuilderOutput(primary_id=1, model=make_model("United_States_Massachusetts"))
    builder.model = make_model("United_States_California")
    with pytest.raises(ValueError, match="model and calibrator"):
        pickle.loads(pickle.dumps(builder))


@pytest.mark.parametrize("model_name", ["US-MA", "Massachusetts", "25"])
def test_builder_rejects_display_or_reporting_alias_as_model_population(model_name):
    with pytest.raises(ValidationError, match="model and calibrator"):
        BuilderOutput(primary_id=1, location=resolve_location("US-MA"), model=make_model(model_name))
