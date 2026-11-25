"""Custom CSV validation, population isolation and standalone result execution."""

import pickle
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from epydemix.population import Population as EpiPopulation

from epymodelingsuite.builders.orchestrators import create_model_collection
from epymodelingsuite.builders.utils import get_data_in_location
from epymodelingsuite.config_loader import load_basemodel_config_from_file, load_sampling_config_from_file
from epymodelingsuite.dispatcher import dispatch_builder
from epymodelingsuite.schema.basemodel import BasemodelConfig, Population
from epymodelingsuite.schema.data_validation import find_missing_data_files
from epymodelingsuite.utils.custom_locations import load_custom_locations
from epymodelingsuite.utils.location import (
    convert_location_name_format,
    get_hub_location_id,
    get_parent_region,
    resolve_locations,
)
from epymodelingsuite.utils.populations import get_total_population
from tests.conftest import AGE_GROUPS, make_sir_config


@pytest.fixture
def custom_config(tmp_path, monkeypatch):
    locations = tmp_path / "locations.csv"
    populations = tmp_path / "population.csv"
    pd.DataFrame(
        [{"location_id": "test-city", "parent_iso": "US-MA", "location_name": "Test City", "population": 8500}]
    ).to_csv(locations, index=False)
    pd.DataFrame(
        {"location_id": "test-city", "age": [str(age) for age in range(84)] + ["84+"], "population": 100}
    ).to_csv(populations, index=False)
    config = make_sir_config(end_date=date(2024, 1, 13), n_sims=1, random_seed=123)
    config.population = Population(
        name="test-city",
        location_type="custom",
        locations_path=str(locations),
        population_path=str(populations),
        age_groups=AGE_GROUPS,
    )
    available = pd.DataFrame(
        {"iso_code": ["US-MA", "JP-13"], "location": ["United_States__Massachusetts", "Japan__Tokyo-to"]}
    )
    monkeypatch.setattr("epymodelingsuite.utils.custom_locations.get_available_locations", lambda: available)

    def load_parent(population_name, age_group_mapping):
        parent = EpiPopulation(name=population_name)
        groups = list(age_group_mapping)
        parent.add_population(Nk=np.repeat(100000.0, len(groups)), Nk_names=groups)
        parent.add_contact_matrix(np.ones((len(groups), len(groups))), layer_name="home")
        return parent

    monkeypatch.setattr("epymodelingsuite.builders.base.load_epydemix_population", load_parent)
    return config


def test_config_loading_needs_no_csv_and_order_is_independent(custom_config, tmp_path):
    base_path = tmp_path / "base.yaml"
    sampling_path = tmp_path / "sampling.yaml"
    base_path.write_text(
        yaml.safe_dump(BasemodelConfig(model=custom_config).model_dump(mode="json", exclude_none=True))
    )
    sampling_path.write_text(yaml.safe_dump({"modelset": {"population_names": ["test-city"]}}))
    Path(custom_config.population.locations_path).unlink()
    Path(custom_config.population.population_path).unlink()
    sampling = load_sampling_config_from_file(str(sampling_path))
    base = load_basemodel_config_from_file(str(base_path))
    assert sampling.modelset.population_names == ["test-city"]
    assert set(find_missing_data_files(base)) == {
        custom_config.population.locations_path,
        custom_config.population.population_path,
    }
    with pytest.raises(FileNotFoundError):
        dispatch_builder(basemodel_config=base, sampling_config=sampling)


def test_custom_population_builds_multiple_models_and_resolves_helpers(custom_config):
    custom = load_custom_locations(custom_config)
    models = create_model_collection(custom_config, ["test-city", {"name": "test-city", "type": "custom"}], custom)
    assert [item.location.location_id for item in models] == ["test-city", "test-city"]
    np.testing.assert_array_equal(models[0].model.population.Nk, [500, 1300, 3200, 1500, 2000])
    np.testing.assert_array_equal(models[0].model.population.contact_matrices["home"], np.ones((5, 5)))
    snapshot = custom["test-city"].location
    assert get_hub_location_id(models[0].model.population.name) == "test-city"
    assert get_parent_region(snapshot) == "US-MA"
    assert convert_location_name_format(snapshot, "name_short") == "Test City"
    assert get_total_population(snapshot) == 8500
    assert all(loc.kind != "custom" for loc in resolve_locations(["all-metrocast"], custom))
    data = pd.DataFrame({"location": ["test-city", "other"], "value": [7, 999]})
    assert get_data_in_location(data, snapshot, "location", "location_id").value.tolist() == [7]
    with pytest.raises(ValueError, match="Invalid population"):
        resolve_locations(["missing-city"], custom)


@pytest.mark.parametrize("parent", ["US-MA", "JP-13"])
def test_non_us_parent_does_not_require_us_codebook(custom_config, parent):
    path = Path(custom_config.population.locations_path)
    frame = pd.read_csv(path)
    frame["parent_iso"] = parent
    frame.to_csv(path, index=False)
    bundle = dispatch_builder(basemodel_config=BasemodelConfig(model=custom_config))
    assert bundle.location.source_iso == parent
    assert bundle.model.population.name == "custom_location_test-city"
    assert not hasattr(bundle.location, "vaccination_scaling")


def test_same_id_from_two_builds_keeps_its_own_population(custom_config):
    first = dispatch_builder(basemodel_config=BasemodelConfig(model=custom_config))
    path = Path(custom_config.population.population_path)
    frame = pd.read_csv(path)
    frame["population"] *= 2
    frame.to_csv(path, index=False)
    metadata_path = Path(custom_config.population.locations_path)
    metadata = pd.read_csv(metadata_path)
    metadata["population"] *= 2
    metadata["location_name"] = "Second City"
    metadata.to_csv(metadata_path, index=False)
    second = dispatch_builder(basemodel_config=BasemodelConfig(model=custom_config))
    assert first.location.total_population == 8500
    assert second.location.total_population == 17000
    assert first.location.location_name == "Test City"
    assert second.location.location_name == "Second City"
    assert first.model.population.Nk.sum() == 8500
    assert second.model.population.Nk.sum() == 17000


@pytest.mark.parametrize(
    "invalid",
    [
        "duplicate_id",
        "duplicate_age",
        "missing_age",
        "unknown_age",
        "negative",
        "infinite",
        "missing_value",
        "unknown_id",
        "missing_column",
        "iso_collision",
        "metro_collision",
        "unknown_parent",
    ],
)
def test_invalid_csv_is_rejected(custom_config, invalid):
    location_path = Path(custom_config.population.locations_path)
    population_path = Path(custom_config.population.population_path)
    locations = pd.read_csv(location_path)
    populations = pd.read_csv(population_path, dtype={"age": str})
    if invalid == "duplicate_id":
        locations = pd.concat([locations, locations])
    elif invalid == "duplicate_age":
        populations = pd.concat([populations, populations.iloc[[0]]])
    elif invalid == "missing_age":
        populations = populations.iloc[1:]
    elif invalid == "unknown_age":
        populations.loc[0, "age"] = "99"
    elif invalid in {"negative", "infinite", "missing_value"}:
        populations["population"] = populations["population"].astype(float)
        populations.loc[0, "population"] = {"negative": -1, "infinite": float("inf"), "missing_value": float("nan")}[
            invalid
        ]
    elif invalid == "unknown_id":
        populations.loc[0, "location_id"] = "not-defined"
    elif invalid == "missing_column":
        locations = locations.drop(columns="parent_iso")
    elif invalid in {"iso_collision", "metro_collision"}:
        value = "JP-13" if invalid == "iso_collision" else "denver"
        locations["location_id"] = value
        populations["location_id"] = value
    elif invalid == "unknown_parent":
        locations["parent_iso"] = "US-ZZ"
    locations.to_csv(location_path, index=False)
    populations.to_csv(population_path, index=False)
    with pytest.raises(ValueError):
        load_custom_locations(custom_config)


def test_result_execution_and_output_in_fresh_processes_without_csv(custom_config, tmp_path):
    bundle = dispatch_builder(basemodel_config=BasemodelConfig(model=custom_config))
    input_path = tmp_path / "input.pkl"
    result_path = tmp_path / "result.pkl"
    input_path.write_bytes(pickle.dumps(bundle))
    Path(custom_config.population.locations_path).unlink()
    Path(custom_config.population.population_path).unlink()
    run = """
import pickle, sys
from pathlib import Path
from epymodelingsuite.dispatcher import dispatch_runner
workload = pickle.loads(Path(sys.argv[1]).read_bytes())
result = dispatch_runner(workload)
assert result.location.total_population == 8500
Path(sys.argv[2]).write_bytes(pickle.dumps(result))
"""
    subprocess.run(
        [sys.executable, "-c", run, str(input_path), str(result_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = """
import pickle, sys
from pathlib import Path
from epymodelingsuite.dispatcher import dispatch_output_generator
from epymodelingsuite.schema.output import OutputConfig
from epymodelingsuite.utils.populations import get_total_population
config = OutputConfig(output={'quantiles': {'compartments': True}, 'tabular_output_types': ['DataFrame']})
result = pickle.loads(Path(sys.argv[1]).read_bytes())
assert get_total_population(result.location) == 8500
artifacts = dispatch_output_generator(simulations=[result], output_config=config)
assert artifacts
frames = [item.data for items in artifacts.values() for item in items if hasattr(item.data, 'columns')]
assert any('population' in frame and set(frame.population) == {'custom_location_test-city'} for frame in frames)
"""
    subprocess.run(
        [sys.executable, "-c", output, str(result_path)], check=True, capture_output=True, text=True, timeout=60
    )


def test_display_names_are_preserved_as_text(custom_config):
    path = Path(custom_config.population.locations_path)
    frame = pd.read_csv(path)
    frame["location_name"] = "001"
    frame["location_name_short"] = "007"
    frame.to_csv(path, index=False)
    location = load_custom_locations(custom_config)["test-city"].location
    assert (location.location_name, location.location_name_short) == ("001", "007")
    frame["location_name"] = "  "
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="blank"):
        load_custom_locations(custom_config)


def test_mixed_location_id_data_uses_custom_id_fips_and_metrocast_id(custom_config):
    location = load_custom_locations(custom_config)["test-city"].location
    data = pd.DataFrame({"location": ["test-city", "25", "denver"], "value": [1, 2, 3]})
    assert get_data_in_location(data, location, "location", "location_id").value.tolist() == [1]
    assert get_data_in_location(data, "United_States__Massachusetts", "location", "location_id").value.tolist() == [2]
    assert get_data_in_location(data, "denver", "location", "location_id").value.tolist() == [3]


def test_bphc_population_contacts_and_vaccine_scaling_match_legacy(custom_config, monkeypatch):
    from epydemix.model import EpiModel

    from epymodelingsuite.builders.base import load_custom_population, load_metrocast_population
    from epymodelingsuite.builders.vaccination import prepare_vaccination_context
    from epymodelingsuite.schema.basemodel import Vaccination
    from epymodelingsuite.utils.populations import aggregate_population_by_age_groups, get_population_codebook

    fixture = Path(__file__).parent / "fixtures" / "custom_locations"
    custom_config.population = Population(
        name="boston-all",
        location_type="custom",
        age_groups=AGE_GROUPS,
        locations_path=str(fixture / "boston_locations.csv"),
        population_path=str(fixture / "boston_population.csv"),
    )
    custom_config.vaccination = Vaccination(
        scenario_data_path="unused.csv", origin_compartment="S", eligible_compartments=["S"]
    )
    population = load_custom_locations(custom_config)["boston-all"]
    actual = load_custom_population(population, AGE_GROUPS)
    legacy_data = pd.read_csv(fixture / "boston_population.csv").rename(
        columns={"location_id": "metrocast_location_id"}
    )
    monkeypatch.setattr("epymodelingsuite.builders.base.get_metrocast_population_data", lambda: legacy_data)
    from dataclasses import replace

    legacy_location = replace(
        population.location,
        kind="metrocast_location",
        population_name="metrocast_location_boston-all",
        identifiers=(("location_id", "boston-all"), ("original_location_code", "BPHC")),
    )
    legacy = load_metrocast_population(legacy_location, AGE_GROUPS)
    np.testing.assert_array_equal(actual.Nk, legacy.Nk)
    assert actual.Nk.sum() == population.location.total_population == 675647
    for layer in legacy.contact_matrices:
        np.testing.assert_array_equal(actual.contact_matrices[layer], legacy.contact_matrices[layer])
    model = EpiModel()
    model.set_population(actual)
    context = prepare_vaccination_context(model, population.location)
    parent_counts = aggregate_population_by_age_groups(
        get_population_codebook()["United_States__Massachusetts"], AGE_GROUPS
    )
    assert dict(context.scaling) == {group: count / parent_counts[group] for group, count in zip(AGE_GROUPS, legacy.Nk)}


def test_calibration_build_and_run_captures_custom_snapshot(custom_config, tmp_path):
    from epymodelingsuite.dispatcher import dispatch_runner
    from epymodelingsuite.schema.calibration import CalibrationConfig

    observed = tmp_path / "observed.csv"
    pd.DataFrame({"location": "test-city", "date": ["2024-01-06", "2024-01-13"], "value": [1, 2]}).to_csv(
        observed, index=False
    )
    config = CalibrationConfig(
        modelset={
            "population_names": ["test-city"],
            "calibration": {
                "strategy": {"name": "top_fraction", "options": {"Nsim": 2, "top_fraction": 1.0}},
                "distance_function": "rmse",
                "observed_data_path": str(observed),
                "comparison": [
                    {
                        "observed_date_column": "date",
                        "observed_value_column": "value",
                        "observed_location_column": "location",
                        "observed_location_format": "location_id",
                        "simulation": ["I_to_R_total"],
                    }
                ],
                "fitting_window": {"start_date": "2024-01-06", "end_date": "2024-01-13"},
                "parameters": {"beta": {"prior": {"type": "scipy", "name": "uniform", "args": [0.1, 0.1]}}},
            },
        }
    )
    bundles = dispatch_builder(basemodel_config=BasemodelConfig(model=custom_config), calibration_config=config)
    assert len(bundles) == 1
    bundle = bundles[0]
    Path(custom_config.population.locations_path).unlink()
    Path(custom_config.population.population_path).unlink()
    result = dispatch_runner(bundle)
    assert result.location is bundle.location
    assert result.results is not None


def test_custom_paths_must_be_paired():
    with pytest.raises(ValueError, match="both locations_path and population_path"):
        Population(name="test-city", location_type="custom", age_groups=AGE_GROUPS, locations_path="locations.csv")


def test_declared_total_is_preserved_and_difference_warns(custom_config, caplog):
    path = Path(custom_config.population.locations_path)
    frame = pd.read_csv(path)
    frame["population"] = 9000
    frame.to_csv(path, index=False)
    location = load_custom_locations(custom_config)["test-city"].location
    assert location.total_population == 9000
    assert "more than 1%" in caplog.text


def test_builder_rejects_metadata_from_another_population(custom_config):
    from dataclasses import replace

    from epymodelingsuite.schema.dispatcher import BuilderOutput

    bundle = dispatch_builder(basemodel_config=BasemodelConfig(model=custom_config))
    values = {name: getattr(bundle, name) for name in type(bundle).model_fields}
    values["location"] = replace(
        bundle.location,
        location_id="another-city",
        population_name="custom_location_another-city",
        identifiers=(("location_id", "another-city"),),
    )
    with pytest.raises(ValueError, match="must match the model"):
        BuilderOutput(**values)


def test_mixed_population_build_run_and_output_share_one_identity_contract(custom_config):
    from epymodelingsuite.dispatcher import dispatch_output_generator, dispatch_runner
    from epymodelingsuite.schema.output import OutputConfig
    from epymodelingsuite.schema.sampling import SamplingConfig

    sampling = SamplingConfig(modelset={"population_names": ["US-MA", "denver", "test-city"]})
    bundles = dispatch_builder(basemodel_config=BasemodelConfig(model=custom_config), sampling_config=sampling)
    Path(custom_config.population.locations_path).unlink()
    Path(custom_config.population.population_path).unlink()
    results = [dispatch_runner(bundle) for bundle in bundles]
    assert [result.location.key for result in results] == [
        ("iso", "US-MA"),
        ("metrocast_location", "denver"),
        ("custom", "test-city"),
    ]
    for result, bundle in zip(results, bundles, strict=True):
        assert result.location is bundle.location
        assert result.population == result.location.population_name
        assert "population" not in type(result).model_fields
        assert "custom_location" not in type(result).model_fields
    artifacts = dispatch_output_generator(
        simulations=results,
        output_config=OutputConfig(output={"quantiles": {"compartments": True}, "tabular_output_types": ["DataFrame"]}),
    )
    populations = set()
    for items in artifacts.values():
        for item in items:
            if isinstance(item.data, pd.DataFrame) and "population" in item.data:
                populations.update(item.data.population)
    assert populations == {result.location.population_name for result in results}
