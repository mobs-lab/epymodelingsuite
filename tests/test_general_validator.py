from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from flumodelingsuite.schema.general import (
    _ensure_compartments_valid,
    _ensure_output_references_valid,
    _ensure_parameters_present,
    _ensure_populations_valid,
    _ensure_transitions_valid,
    _validate_compartment_list,
    _validate_transition_list,
    validate_cross_config_consistency,
)
from flumodelingsuite.schema.output import OutputConfig, OutputConfiguration, QuantilesOutput, TrajectoriesOutput


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
        validate_cross_config_consistency(base_config, sampling_config)

    def test_valid_calibration(self):
        base_config = self.create_base_config()
        calibration_config = self.create_calibration_config()
        validate_cross_config_consistency(base_config, calibration_config)

    def test_missing_base_model(self):
        base_config = DummyBaseConfig(model=None)
        sampling_config = self.create_sampling_config()
        with pytest.raises(ValueError, match="Both base model and modelset must be defined"):
            validate_cross_config_consistency(base_config, sampling_config)

    def test_missing_modelset(self):
        base_config = self.create_base_config()
        sampling_config = DummyModelsetConfig(modelset=None)
        with pytest.raises(ValueError, match="Both base model and modelset must be defined"):
            validate_cross_config_consistency(base_config, sampling_config)

    def test_no_sampling_or_calibration(self):
        base_config = self.create_base_config()
        modelset = DummyModelset(population_names=["US"], sampling=None, calibration=None)
        sampling_config = DummyModelsetConfig(modelset=modelset)
        with pytest.raises(ValueError, match="Modelset must provide a 'sampling' or 'calibration' section"):
            validate_cross_config_consistency(base_config, sampling_config)

    def test_missing_parameters(self):
        base_config = self.create_base_config()
        sampling_config = self.create_sampling_config(sampling_params={"delta": object()})
        with pytest.raises(ValueError, match="Parameters in modelset not defined in base model"):
            validate_cross_config_consistency(base_config, sampling_config)

    def test_invalid_compartments(self):
        base_config = self.create_base_config()
        sampling_config = self.create_sampling_config(compartments={"X": object()})
        with pytest.raises(ValueError, match="Compartments in modelset not defined in base model"):
            validate_cross_config_consistency(base_config, sampling_config)

    def test_invalid_populations(self):
        base_config = self.create_base_config()
        sampling_config = self.create_sampling_config(population_names=["CA"])
        with pytest.raises(ValueError, match="Populations in modelset not matching base model"):
            validate_cross_config_consistency(base_config, sampling_config)

    def test_invalid_transitions_in_calibration(self):
        base_config = self.create_base_config()
        comparison = DummyComparison(simulation=["death"])
        calibration_config = self.create_calibration_config(comparisons=[comparison])
        with pytest.raises(ValueError, match="Transitions in calibration comparison not defined in base model"):
            validate_cross_config_consistency(base_config, calibration_config)

    def test_with_valid_output_config(self):
        base_config = self.create_base_config()
        sampling_config = self.create_sampling_config()
        output_config = OutputConfig(
            output=OutputConfiguration(quantiles=QuantilesOutput(compartments=["S_total", "I_total"]))
        )
        validate_cross_config_consistency(base_config, sampling_config, output_config)

    def test_with_invalid_output_config(self):
        base_config = self.create_base_config()
        sampling_config = self.create_sampling_config()
        output_config = OutputConfig(output=OutputConfiguration(quantiles=QuantilesOutput(compartments=["Hosp_total"])))
        with pytest.raises(ValueError, match="Compartments in .* not defined"):
            validate_cross_config_consistency(base_config, sampling_config, output_config)


class TestValidateCompartmentList:
    def test_valid_compartments(self):
        base_compartments = {"S", "I", "R", "Hosp"}
        names = ["S_total", "Hosp_total"]
        _validate_compartment_list(names, base_compartments, "test.compartments")

    def test_valid_without_suffix(self):
        base_compartments = {"S", "I", "R"}
        names = ["S", "I"]
        _validate_compartment_list(names, base_compartments, "test.compartments")

    def test_invalid_compartments(self):
        base_compartments = {"S", "I", "R"}
        names = ["S_total", "Hosp_total"]
        with pytest.raises(
            ValueError, match="Compartments in test.compartments not defined in basemodel: \\['Hosp'\\]"
        ):
            _validate_compartment_list(names, base_compartments, "test.compartments")

    def test_empty_list(self):
        base_compartments = {"S", "I", "R"}
        names = []
        _validate_compartment_list(names, base_compartments, "test.compartments")


class TestValidateTransitionList:
    def test_valid_transitions(self):
        base_transitions = {"S_to_I", "I_to_R"}
        names = ["S_to_I_total", "I_to_R_total"]
        _validate_transition_list(names, base_transitions, "test.transitions")

    def test_valid_without_suffix(self):
        base_transitions = {"S_to_I", "I_to_R"}
        names = ["S_to_I", "I_to_R"]
        _validate_transition_list(names, base_transitions, "test.transitions")

    def test_invalid_transitions(self):
        base_transitions = {"S_to_I"}
        names = ["S_to_I_total", "I_to_R_total"]
        with pytest.raises(
            ValueError, match="Transitions in test.transitions not defined in basemodel: \\['I_to_R'\\]"
        ):
            _validate_transition_list(names, base_transitions, "test.transitions")

    def test_invalid_format(self):
        base_transitions = {"S_to_I"}
        names = ["invalid_name"]
        with pytest.raises(ValueError, match="Transitions in test.transitions not defined in basemodel"):
            _validate_transition_list(names, base_transitions, "test.transitions")

    def test_empty_list(self):
        base_transitions = {"S_to_I"}
        names = []
        _validate_transition_list(names, base_transitions, "test.transitions")


class TestEnsureOutputReferencesValid:
    def test_valid_quantiles_compartments_list(self):
        base_compartments = {"S", "I", "R", "Hosp"}
        base_transitions = set()
        output = OutputConfig(
            output=OutputConfiguration(quantiles=QuantilesOutput(compartments=["S_total", "Hosp_total"]))
        )
        _ensure_output_references_valid(base_compartments, base_transitions, output)

    def test_invalid_quantiles_compartments(self):
        base_compartments = {"S", "I", "R"}
        base_transitions = set()
        output = OutputConfig(
            output=OutputConfiguration(quantiles=QuantilesOutput(compartments=["S_total", "Hosp_total"]))
        )
        with pytest.raises(ValueError, match="Compartments in quantiles.compartments not defined in basemodel"):
            _ensure_output_references_valid(base_compartments, base_transitions, output)

    def test_quantiles_compartments_boolean_true(self):
        base_compartments = {"S", "I", "R"}
        base_transitions = set()
        output = OutputConfig(output=OutputConfiguration(quantiles=QuantilesOutput(compartments=True)))
        _ensure_output_references_valid(base_compartments, base_transitions, output)

    def test_valid_quantiles_transitions_list(self):
        base_compartments = set()
        base_transitions = {"S_to_I", "I_to_R"}
        output = OutputConfig(
            output=OutputConfiguration(quantiles=QuantilesOutput(transitions=["S_to_I_total", "I_to_R_total"]))
        )
        _ensure_output_references_valid(base_compartments, base_transitions, output)

    def test_invalid_quantiles_transitions(self):
        base_compartments = set()
        base_transitions = {"S_to_I"}
        output = OutputConfig(
            output=OutputConfiguration(quantiles=QuantilesOutput(transitions=["S_to_I_total", "I_to_R_total"]))
        )
        with pytest.raises(ValueError, match="Transitions in quantiles.transitions not defined in basemodel"):
            _ensure_output_references_valid(base_compartments, base_transitions, output)

    def test_quantiles_transitions_boolean_true(self):
        base_compartments = set()
        base_transitions = {"S_to_I"}
        output = OutputConfig(output=OutputConfiguration(quantiles=QuantilesOutput(transitions=True)))
        _ensure_output_references_valid(base_compartments, base_transitions, output)

    def test_valid_trajectories_compartments_list(self):
        base_compartments = {"S", "I", "R", "Hosp"}
        base_transitions = set()
        output = OutputConfig(
            output=OutputConfiguration(trajectories=TrajectoriesOutput(compartments=["S_total", "Hosp_total"]))
        )
        _ensure_output_references_valid(base_compartments, base_transitions, output)

    def test_invalid_trajectories_compartments(self):
        base_compartments = {"S", "I", "R"}
        base_transitions = set()
        output = OutputConfig(
            output=OutputConfiguration(trajectories=TrajectoriesOutput(compartments=["S_total", "Hosp_total"]))
        )
        with pytest.raises(ValueError, match="Compartments in trajectories.compartments not defined in basemodel"):
            _ensure_output_references_valid(base_compartments, base_transitions, output)

    def test_trajectories_compartments_boolean_true(self):
        base_compartments = {"S", "I", "R"}
        base_transitions = set()
        output = OutputConfig(output=OutputConfiguration(trajectories=TrajectoriesOutput(compartments=True)))
        _ensure_output_references_valid(base_compartments, base_transitions, output)

    def test_valid_trajectories_transitions_list(self):
        base_compartments = set()
        base_transitions = {"S_to_I", "I_to_R"}
        output = OutputConfig(
            output=OutputConfiguration(trajectories=TrajectoriesOutput(transitions=["S_to_I_total", "I_to_R_total"]))
        )
        _ensure_output_references_valid(base_compartments, base_transitions, output)

    def test_invalid_trajectories_transitions(self):
        base_compartments = set()
        base_transitions = {"S_to_I"}
        output = OutputConfig(
            output=OutputConfiguration(trajectories=TrajectoriesOutput(transitions=["S_to_I_total", "I_to_R_total"]))
        )
        with pytest.raises(ValueError, match="Transitions in trajectories.transitions not defined in basemodel"):
            _ensure_output_references_valid(base_compartments, base_transitions, output)

    def test_trajectories_transitions_boolean_true(self):
        base_compartments = set()
        base_transitions = {"S_to_I"}
        output = OutputConfig(output=OutputConfiguration(trajectories=TrajectoriesOutput(transitions=True)))
        _ensure_output_references_valid(base_compartments, base_transitions, output)

    def test_no_quantiles_or_trajectories(self):
        base_compartments = {"S", "I", "R"}
        base_transitions = set()
        output = OutputConfig(output=OutputConfiguration(quantiles=None, trajectories=None))
        _ensure_output_references_valid(base_compartments, base_transitions, output)

    def test_both_quantiles_and_trajectories(self):
        base_compartments = {"S", "I", "R", "Hosp"}
        base_transitions = {"S_to_I", "I_to_R"}
        output = OutputConfig(
            output=OutputConfiguration(
                quantiles=QuantilesOutput(compartments=["S_total", "I_total"]),
                trajectories=TrajectoriesOutput(transitions=["S_to_I_total"]),
            )
        )
        _ensure_output_references_valid(base_compartments, base_transitions, output)
