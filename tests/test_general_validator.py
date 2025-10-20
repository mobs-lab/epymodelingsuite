from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from flumodelingsuite.validation.general_validator import (
    _ensure_compartments_valid,
    _ensure_parameters_present,
    _ensure_populations_valid,
    _ensure_transitions_valid,
    _to_set,
    validate_modelset_consistency,
)


@dataclass
class DummyCompartment:
    id: str


@dataclass
class DummyTransition:
    id: str
    source: str
    target: str


@dataclass
class DummyPopulation:
    name: str | None = None


@dataclass
class DummyModel:
    parameters: dict[str, Any]
    compartments: list[DummyCompartment]
    transitions: list[DummyTransition]
    population: DummyPopulation | None = None


@dataclass
class DummyBaseConfig:
    model: DummyModel | None


@dataclass
class DummySampling:
    parameters: dict[str, Any]
    compartments: dict[str, Any] | None = None


@dataclass
class DummyComparison:
    simulation: list[str]


@dataclass
class DummyCalibration:
    parameters: dict[str, Any]
    comparison: list[DummyComparison] | None = None


@dataclass
class DummyModelset:
    population_names: list[str] | None = None
    sampling: DummySampling | None = None
    calibration: DummyCalibration | None = None


@dataclass
class DummyModelsetConfig:
    modelset: DummyModelset | None


class TestToSet:
    def test_none_input(self):
        assert _to_set(None) == set()

    def test_list_input(self):
        assert _to_set([1, 2, 3]) == {1, 2, 3}

    def test_set_input(self):
        assert _to_set({1, 2}) == {1, 2}

    def test_empty_list(self):
        assert _to_set([]) == set()


class TestEnsureParametersPresent:
    def test_all_present(self):
        base_params = {"a", "b", "c"}
        modelset_params = {"a", "b"}
        _ensure_parameters_present(base_params, modelset_params)  # Should not raise

    def test_missing_params(self):
        base_params = {"a", "b"}
        modelset_params = {"a", "c"}
        with pytest.raises(ValueError, match="Parameters in modelset not defined in base model: \\['c'\\]"):
            _ensure_parameters_present(base_params, modelset_params)


class TestEnsureCompartmentsValid:
    def test_no_sampling(self):
        base_compartments = {"S", "I", "R"}
        _ensure_compartments_valid(base_compartments, None)

    def test_no_compartments_in_sampling(self):
        base_compartments = {"S", "I", "R"}
        sampling = SimpleNamespace(compartments=None)
        _ensure_compartments_valid(base_compartments, sampling)

    def test_valid_compartments(self):
        base_compartments = {"S", "I", "R"}
        sampling = SimpleNamespace(compartments={"S": object(), "I": object()})
        _ensure_compartments_valid(base_compartments, sampling)

    def test_invalid_compartments(self):
        base_compartments = {"S", "I", "R"}
        sampling = SimpleNamespace(compartments={"X": object()})
        with pytest.raises(ValueError, match="Compartments in modelset not defined in base model: \\['X'\\]"):
            _ensure_compartments_valid(base_compartments, sampling)


class TestEnsurePopulationsValid:
    def test_no_base_population(self):
        modelset_populations = {"US", "all"}
        _ensure_populations_valid(None, modelset_populations)  # Should not raise

    def test_no_modelset_populations(self):
        base_population_name = "US"
        _ensure_populations_valid(base_population_name, set())  # Should not raise

    def test_valid_populations(self):
        base_population_name = "US"
        modelset_populations = {"US", "all"}
        _ensure_populations_valid(base_population_name, modelset_populations)  # Should not raise

    def test_invalid_populations(self):
        base_population_name = "US"
        modelset_populations = {"US", "CA"}
        with pytest.raises(ValueError, match="Populations in modelset not matching base model: \\['CA'\\]"):
            _ensure_populations_valid(base_population_name, modelset_populations)


class TestEnsureTransitionsValid:
    def test_no_calibration(self):
        base_transitions = {"inf", "rec"}
        _ensure_transitions_valid(base_transitions, None)

    def test_no_comparisons(self):
        base_transitions = {"inf", "rec"}
        calibration = SimpleNamespace(comparison=None)
        _ensure_transitions_valid(base_transitions, calibration)

    def test_valid_transitions(self):
        base_transitions = {"inf", "rec"}
        comparison = SimpleNamespace(simulation=["inf", "rec"])
        calibration = SimpleNamespace(comparison=[comparison])
        _ensure_transitions_valid(base_transitions, calibration)

    def test_invalid_transitions(self):
        base_transitions = {"inf", "rec"}
        comparison = SimpleNamespace(simulation=["inf", "death"])
        calibration = SimpleNamespace(comparison=[comparison])
        with pytest.raises(
            ValueError, match="Transitions in calibration comparison not defined in base model: \\['death'\\]"
        ):
            _ensure_transitions_valid(base_transitions, calibration)


class TestValidateModelsetConsistency:
    def create_base_config(self) -> DummyBaseConfig:
        compartments = [
            DummyCompartment(id="S"),
            DummyCompartment(id="I"),
            DummyCompartment(id="R"),
        ]
        transitions = [
            DummyTransition(id="inf", source="S", target="I"),
            DummyTransition(id="rec", source="I", target="R"),
        ]
        parameters = {"beta": object(), "gamma": object()}
        population = DummyPopulation(name="US")
        model = DummyModel(
            parameters=parameters, compartments=compartments, transitions=transitions, population=population
        )
        return DummyBaseConfig(model=model)

    def create_sampling_config(
        self,
        population_names: list[str] | None = None,
        sampling_params: dict[str, Any] | None = None,
        compartments: dict[str, Any] | None = None,
    ) -> DummyModelsetConfig:
        sampling = DummySampling(parameters=sampling_params or {"beta": object()}, compartments=compartments)
        modelset = DummyModelset(population_names=population_names or ["US"], sampling=sampling)
        return DummyModelsetConfig(modelset=modelset)

    def create_calibration_config(
        self,
        population_names: list[str] | None = None,
        calibration_params: dict[str, Any] | None = None,
        comparisons: list[DummyComparison] | None = None,
    ) -> DummyModelsetConfig:
        calibration = DummyCalibration(
            parameters=calibration_params or {"beta": object()},
            comparison=comparisons or [DummyComparison(simulation=["S_to_I_total"])],
        )
        modelset = DummyModelset(population_names=population_names or ["US"], calibration=calibration)
        return DummyModelsetConfig(modelset=modelset)

    def test_valid_sampling(self):
        base_config = self.create_base_config()
        sampling_config = self.create_sampling_config()
        validate_modelset_consistency(base_config, sampling_config)

    def test_valid_calibration(self):
        base_config = self.create_base_config()
        calibration_config = self.create_calibration_config()
        validate_modelset_consistency(base_config, calibration_config)

    def test_missing_base_model(self):
        base_config = DummyBaseConfig(model=None)
        sampling_config = self.create_sampling_config()
        with pytest.raises(ValueError, match="Both base model and modelset must be defined"):
            validate_modelset_consistency(base_config, sampling_config)

    def test_missing_modelset(self):
        base_config = self.create_base_config()
        sampling_config = DummyModelsetConfig(modelset=None)
        with pytest.raises(ValueError, match="Both base model and modelset must be defined"):
            validate_modelset_consistency(base_config, sampling_config)

    def test_no_sampling_or_calibration(self):
        base_config = self.create_base_config()
        modelset = DummyModelset(population_names=["US"], sampling=None, calibration=None)
        sampling_config = DummyModelsetConfig(modelset=modelset)
        with pytest.raises(ValueError, match="Modelset must provide a 'sampling' or 'calibration' section"):
            validate_modelset_consistency(base_config, sampling_config)

    def test_missing_parameters(self):
        base_config = self.create_base_config()
        sampling_config = self.create_sampling_config(sampling_params={"delta": object()})
        with pytest.raises(ValueError, match="Parameters in modelset not defined in base model"):
            validate_modelset_consistency(base_config, sampling_config)

    def test_invalid_compartments(self):
        base_config = self.create_base_config()
        sampling_config = self.create_sampling_config(compartments={"X": object()})
        with pytest.raises(ValueError, match="Compartments in modelset not defined in base model"):
            validate_modelset_consistency(base_config, sampling_config)

    def test_invalid_populations(self):
        base_config = self.create_base_config()
        sampling_config = self.create_sampling_config(population_names=["CA"])
        with pytest.raises(ValueError, match="Populations in modelset not matching base model"):
            validate_modelset_consistency(base_config, sampling_config)

    def test_invalid_transitions_in_calibration(self):
        base_config = self.create_base_config()
        comparison = DummyComparison(simulation=["death"])
        calibration_config = self.create_calibration_config(comparisons=[comparison])
        with pytest.raises(ValueError, match="Transitions in calibration comparison not defined in base model"):
            validate_modelset_consistency(base_config, calibration_config)
