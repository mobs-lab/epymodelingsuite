from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from typing import Any

import pytest

from epymodelingsuite.schema.general import (
    _ensure_compartments_valid,
    _ensure_fitting_window_within_timespan,
    _ensure_output_references_valid,
    _ensure_parameters_present,
    _ensure_populations_valid,
    _ensure_transitions_valid,
    _validate_compartment_list,
    _validate_transition_list,
    _warn_mismatched_observed_data_paths,
    validate_cross_config_consistency,
)
from epymodelingsuite.schema.output import (
    FlusightForecastOutput,
    ObservedValuesConfig,
    OutputConfig,
    OutputConfiguration,
    OutputOptions,
    QuantilesOutput,
    TrajectoriesOutput,
)


class TestEnsureParametersPresent:
    def test_all_present(self):
        base_params = {"a", "b", "c"}
        modelset_params = {"a", "b"}
        _ensure_parameters_present(base_params, modelset_params)  # Should not raise

    def test_missing_params_warns(self, caplog):
        base_params = {"a", "b"}
        modelset_params = {"a", "c"}
        with caplog.at_level("WARNING"):
            _ensure_parameters_present(base_params, modelset_params)
        assert "Parameters in modelset not defined in base model: ['c']" in caplog.text


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

    def test_invalid_transitions_warns(self, caplog):
        base_transitions = {"inf", "rec"}
        comparison = SimpleNamespace(simulation=["inf", "death"])
        calibration = SimpleNamespace(comparison=[comparison])
        with caplog.at_level("WARNING"):
            _ensure_transitions_valid(base_transitions, calibration)
        assert "Transitions in calibration comparison not defined in base model: ['death']" in caplog.text


class TestValidateModelsetConsistency:
    def _create_base_config(
        self,
        compartment_ids: list[str] | None = None,
        parameters: dict[str, Any] | None = None,
        population_name: str | None = "US",
        start_date: date | str = date(2024, 1, 1),
        end_date: date = date(2024, 12, 31),
    ) -> SimpleNamespace:
        """Create a mock base config with model."""
        compartment_ids = compartment_ids or ["S", "I", "R"]
        compartments = [SimpleNamespace(id=cid) for cid in compartment_ids]
        transitions = [
            SimpleNamespace(id="inf", source="S", target="I"),
            SimpleNamespace(id="rec", source="I", target="R"),
        ]
        parameters = parameters or {"beta": object(), "gamma": object()}
        population = SimpleNamespace(name=population_name)
        timespan = SimpleNamespace(start_date=start_date, end_date=end_date)
        model = SimpleNamespace(
            parameters=parameters,
            compartments=compartments,
            transitions=transitions,
            population=population,
            timespan=timespan,
        )
        return SimpleNamespace(model=model)

    def _create_sampling_config(
        self,
        population_names: list[str] | None = None,
        sampling_params: dict[str, Any] | None = None,
        compartments: dict[str, Any] | None = None,
    ) -> SimpleNamespace:
        """Create a mock modelset config with sampling."""
        sampling = SimpleNamespace(
            parameters=sampling_params or {"beta": object()},
            compartments=compartments,
        )
        modelset = SimpleNamespace(
            population_names=population_names or ["US"],
            sampling=sampling,
            calibration=None,
        )
        return SimpleNamespace(modelset=modelset)

    def _create_calibration_config(
        self,
        population_names: list[str] | None = None,
        calibration_params: dict[str, Any] | None = None,
        comparisons: list[SimpleNamespace] | None = None,
    ) -> SimpleNamespace:
        """Create a mock modelset config with calibration."""
        fitting_window = SimpleNamespace(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 30),
            epiweek_start_date=date(2024, 1, 1),
            epiweek_end_date=date(2024, 6, 30),
        )
        calibration = SimpleNamespace(
            parameters=calibration_params or {"beta": object()},
            comparison=comparisons or [SimpleNamespace(simulation=["S_to_I_total"])],
            fitting_window=fitting_window,
        )
        modelset = SimpleNamespace(
            population_names=population_names or ["US"],
            sampling=None,
            calibration=calibration,
        )
        return SimpleNamespace(modelset=modelset)

    def test_valid_sampling(self):
        base_config = self._create_base_config()
        sampling_config = self._create_sampling_config()
        validate_cross_config_consistency(base_config, sampling_config)

    def test_valid_calibration(self):
        base_config = self._create_base_config()
        calibration_config = self._create_calibration_config()
        validate_cross_config_consistency(base_config, calibration_config)

    def test_missing_base_model(self):
        base_config = SimpleNamespace(model=None)
        sampling_config = self._create_sampling_config()
        with pytest.raises(ValueError, match="Both base model and modelset must be defined"):
            validate_cross_config_consistency(base_config, sampling_config)

    def test_missing_modelset(self):
        base_config = self._create_base_config()
        sampling_config = SimpleNamespace(modelset=None)
        with pytest.raises(ValueError, match="Both base model and modelset must be defined"):
            validate_cross_config_consistency(base_config, sampling_config)

    def test_no_sampling_or_calibration(self):
        base_config = self._create_base_config()
        modelset = SimpleNamespace(population_names=["US"], sampling=None, calibration=None)
        modelset_config = SimpleNamespace(modelset=modelset)
        with pytest.raises(ValueError, match="Modelset must provide a 'sampling' or 'calibration' section"):
            validate_cross_config_consistency(base_config, modelset_config)

    def test_missing_parameters_warns(self, caplog):
        base_config = self._create_base_config()
        sampling_config = self._create_sampling_config(sampling_params={"delta": object()})
        with caplog.at_level("WARNING"):
            validate_cross_config_consistency(base_config, sampling_config)
        assert "Parameters in modelset not defined in base model" in caplog.text

    def test_invalid_compartments(self):
        base_config = self._create_base_config()
        sampling_config = self._create_sampling_config(compartments={"X": object()})
        with pytest.raises(ValueError, match="Compartments in modelset not defined in base model"):
            validate_cross_config_consistency(base_config, sampling_config)

    def test_invalid_populations(self):
        base_config = self._create_base_config()
        sampling_config = self._create_sampling_config(population_names=["CA"])
        with pytest.raises(ValueError, match="Populations in modelset not matching base model"):
            validate_cross_config_consistency(base_config, sampling_config)

    def test_invalid_transitions_in_calibration_warns(self, caplog):
        base_config = self._create_base_config()
        comparison = SimpleNamespace(simulation=["death"])
        calibration_config = self._create_calibration_config(comparisons=[comparison])
        with caplog.at_level("WARNING"):
            validate_cross_config_consistency(base_config, calibration_config)
        assert "Transitions in calibration comparison not defined in base model" in caplog.text

    def test_with_valid_output_config(self):
        base_config = self._create_base_config()
        sampling_config = self._create_sampling_config()
        output_config = OutputConfig(
            output=OutputConfiguration(quantiles=QuantilesOutput(compartments=["S_total", "I_total"]))
        )
        validate_cross_config_consistency(base_config, sampling_config, output_config)

    def test_with_invalid_output_config(self):
        base_config = self._create_base_config()
        sampling_config = self._create_sampling_config()
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

    def test_invalid_transitions_with_warn_only(self, caplog):
        import logging

        caplog.set_level(logging.WARNING)

        base_transitions = {"S_to_I"}
        names = ["S_to_I_total", "I_to_R_total"]

        # Should not raise when warn_only=True, but should log a warning
        _validate_transition_list(names, base_transitions, "test.transitions", warn_only=True)
        assert "Transitions in test.transitions not defined in basemodel" in caplog.text
        assert "I_to_R" in caplog.text


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

    def test_invalid_quantiles_transitions_warns_only(self, caplog):
        import logging

        caplog.set_level(logging.WARNING)

        base_compartments = set()
        base_transitions = {"S_to_I"}
        output = OutputConfig(
            output=OutputConfiguration(quantiles=QuantilesOutput(transitions=["S_to_I_total", "I_to_R_total"]))
        )
        # Should not raise, but should log a warning
        _ensure_output_references_valid(base_compartments, base_transitions, output)
        assert "Transitions in quantiles.transitions not defined in basemodel" in caplog.text
        assert "I_to_R" in caplog.text

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

    def test_invalid_trajectories_transitions_warns_only(self, caplog):
        import logging

        caplog.set_level(logging.WARNING)

        base_compartments = set()
        base_transitions = {"S_to_I"}
        output = OutputConfig(
            output=OutputConfiguration(trajectories=TrajectoriesOutput(transitions=["S_to_I_total", "I_to_R_total"]))
        )
        # Should not raise, but should log a warning
        _ensure_output_references_valid(base_compartments, base_transitions, output)
        assert "Transitions in trajectories.transitions not defined in basemodel" in caplog.text
        assert "I_to_R" in caplog.text

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

    def test_aggregated_transitions_allowed_in_quantiles(self, caplog):
        """Test that aggregated transitions (not in basemodel) are allowed with warnings."""
        import logging

        caplog.set_level(logging.WARNING)

        # Base model has transitions for both vaccinated and unvaccinated hospitalization
        base_compartments = set()
        base_transitions = {"Home_sev_to_Hosp", "Home_sev_vax_to_Hosp_vax"}

        # Output config references an aggregated "hospitalization" transition that doesn't exist in basemodel
        # This is valid because it will be created by summing multiple transitions during calibration
        output = OutputConfig(
            output=OutputConfiguration(quantiles=QuantilesOutput(transitions=["hospitalization_total"]))
        )

        # Should not raise, but should log a warning about the aggregated transition
        _ensure_output_references_valid(base_compartments, base_transitions, output)
        assert "Transitions in quantiles.transitions not defined in basemodel" in caplog.text
        assert "hospitalization" in caplog.text

    def test_aggregated_transitions_allowed_in_trajectories(self, caplog):
        """Test that aggregated transitions (not in basemodel) are allowed with warnings."""
        import logging

        caplog.set_level(logging.WARNING)

        # Base model has individual transitions
        base_compartments = set()
        base_transitions = {"I_to_R", "I_vax_to_R_vax"}

        # Output config references an aggregated transition for both vaccinated and unvaccinated recovery
        output = OutputConfig(
            output=OutputConfiguration(trajectories=TrajectoriesOutput(transitions=["recovery_total"]))
        )

        # Should not raise, but should log a warning
        _ensure_output_references_valid(base_compartments, base_transitions, output)
        assert "Transitions in trajectories.transitions not defined in basemodel" in caplog.text
        assert "recovery" in caplog.text


class TestWarnMismatchedObservedDataPaths:
    def test_no_warning_when_calibration_is_none(self, caplog):
        output_config = OutputConfig(
            output=OutputConfiguration(
                options=OutputOptions(
                    surveillance={
                        "hosp": ObservedValuesConfig(
                            data_path="data/output.csv",
                            value_column="value",
                            date_column="date",
                            location_column="location",
                        )
                    }
                ),
                flusight_format=FlusightForecastOutput(
                    reference_date="2024-01-01",
                    rate_trends_source="hosp",
                ),
            )
        )
        _warn_mismatched_observed_data_paths(None, output_config)
        assert "Observed data paths differ" not in caplog.text

    def test_no_warning_when_flusight_format_is_none(self, caplog):
        calibration = SimpleNamespace(observed_data_path="data/calibration.csv")
        output_config = OutputConfig(output=OutputConfiguration(flusight_format=None))
        _warn_mismatched_observed_data_paths(calibration, output_config)
        assert "Observed data paths differ" not in caplog.text

    def test_no_warning_when_rate_trends_source_is_none(self, caplog):
        calibration = SimpleNamespace(observed_data_path="data/calibration.csv")
        output_config = OutputConfig(
            output=OutputConfiguration(
                flusight_format=FlusightForecastOutput(reference_date="2024-01-01", rate_trends_source=None)
            )
        )
        _warn_mismatched_observed_data_paths(calibration, output_config)
        assert "Observed data paths differ" not in caplog.text

    def test_no_warning_when_paths_match(self, caplog):
        calibration = SimpleNamespace(observed_data_path="data/same.csv")
        output_config = OutputConfig(
            output=OutputConfiguration(
                options=OutputOptions(
                    surveillance={
                        "hosp": ObservedValuesConfig(
                            data_path="data/same.csv",
                            value_column="value",
                            date_column="date",
                            location_column="location",
                        )
                    }
                ),
                flusight_format=FlusightForecastOutput(
                    reference_date="2024-01-01",
                    rate_trends_source="hosp",
                ),
            )
        )
        _warn_mismatched_observed_data_paths(calibration, output_config)
        assert "Observed data paths differ" not in caplog.text

    def test_warning_when_paths_differ(self, caplog):
        import logging

        caplog.set_level(logging.WARNING)

        calibration = SimpleNamespace(observed_data_path="data/calibration.csv")
        output_config = OutputConfig(
            output=OutputConfiguration(
                options=OutputOptions(
                    surveillance={
                        "hosp": ObservedValuesConfig(
                            data_path="data/output.csv",
                            value_column="value",
                            date_column="date",
                            location_column="location",
                        )
                    }
                ),
                flusight_format=FlusightForecastOutput(
                    reference_date="2024-01-01",
                    rate_trends_source="hosp",
                ),
            )
        )
        _warn_mismatched_observed_data_paths(calibration, output_config)
        assert "Observed data paths differ between configs" in caplog.text
        assert "calibration='data/calibration.csv'" in caplog.text
        assert "rate_trends_source ('hosp')='data/output.csv'" in caplog.text


class TestEnsureFittingWindowWithinTimespan:
    """Test validation of fitting window against simulation timespan."""

    def _create_basemodel_with_timespan(
        self,
        start_date: date | str,
        end_date: date,
    ) -> SimpleNamespace:
        """Create a mock base config with timespan."""
        timespan = SimpleNamespace(start_date=start_date, end_date=end_date)
        model = SimpleNamespace(timespan=timespan)
        return SimpleNamespace(model=model)

    def _create_calibration_with_fitting_window(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        epiweek_start_date: date | None = None,
        epiweek_end_date: date | None = None,
    ) -> SimpleNamespace:
        """Create a mock calibration with fitting window."""
        # Mimic the computed fields behavior from epiweek_fitting_window
        fitting_window = SimpleNamespace(
            start_date=start_date,
            end_date=end_date,
            epiweek_start_date=epiweek_start_date if epiweek_start_date else start_date,
            epiweek_end_date=epiweek_end_date if epiweek_end_date else end_date,
        )
        return SimpleNamespace(fitting_window=fitting_window)

    def test_fitting_window_within_timespan_valid(self):
        """Test that fitting window within timespan is valid."""
        basemodel = self._create_basemodel_with_timespan(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
        )
        calibration = self._create_calibration_with_fitting_window(
            start_date=date(2024, 3, 1),
            end_date=date(2024, 6, 30),
        )
        # Should not raise
        _ensure_fitting_window_within_timespan(basemodel, calibration)

    def test_fitting_window_end_exceeds_timespan_end_invalid(self):
        """Test that fitting window end exceeding timespan raises error."""
        basemodel = self._create_basemodel_with_timespan(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 30),
        )
        calibration = self._create_calibration_with_fitting_window(
            start_date=date(2024, 3, 1),
            end_date=date(2024, 12, 31),  # Exceeds timespan end
        )
        with pytest.raises(ValueError, match="exceeds simulation timespan end_date"):
            _ensure_fitting_window_within_timespan(basemodel, calibration)

    def test_fitting_window_start_before_timespan_start_invalid(self):
        """Test that fitting window start before timespan raises error."""
        basemodel = self._create_basemodel_with_timespan(
            start_date=date(2024, 3, 1),
            end_date=date(2024, 12, 31),
        )
        calibration = self._create_calibration_with_fitting_window(
            start_date=date(2024, 1, 1),  # Before timespan start
            end_date=date(2024, 6, 30),
        )
        with pytest.raises(ValueError, match="is before simulation timespan start_date"):
            _ensure_fitting_window_within_timespan(basemodel, calibration)

    def test_sampled_start_date_skips_start_validation(self):
        """Test that 'sampled' start_date skips start date validation."""
        basemodel = self._create_basemodel_with_timespan(
            start_date="sampled",  # Not a concrete date
            end_date=date(2024, 12, 31),
        )
        calibration = self._create_calibration_with_fitting_window(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 30),
        )
        # Should not raise - start date validation skipped
        _ensure_fitting_window_within_timespan(basemodel, calibration)

    def test_calibrated_start_date_skips_start_validation(self):
        """Test that 'calibrated' start_date skips start date validation."""
        basemodel = self._create_basemodel_with_timespan(
            start_date="calibrated",  # Not a concrete date
            end_date=date(2024, 12, 31),
        )
        calibration = self._create_calibration_with_fitting_window(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 30),
        )
        # Should not raise - start date validation skipped
        _ensure_fitting_window_within_timespan(basemodel, calibration)

    def test_sampled_start_still_validates_end_date(self):
        """Test that 'sampled' start_date still validates end date."""
        basemodel = self._create_basemodel_with_timespan(
            start_date="sampled",
            end_date=date(2024, 6, 30),
        )
        calibration = self._create_calibration_with_fitting_window(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),  # Exceeds timespan end
        )
        with pytest.raises(ValueError, match="exceeds simulation timespan end_date"):
            _ensure_fitting_window_within_timespan(basemodel, calibration)

    def test_no_calibration_skips_validation(self):
        """Test that None calibration skips validation."""
        basemodel = self._create_basemodel_with_timespan(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
        )
        # Should not raise
        _ensure_fitting_window_within_timespan(basemodel, None)

    def test_exact_boundary_match_valid(self):
        """Test that fitting window exactly matching timespan is valid."""
        basemodel = self._create_basemodel_with_timespan(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
        )
        calibration = self._create_calibration_with_fitting_window(
            start_date=date(2024, 1, 1),  # Exact match
            end_date=date(2024, 12, 31),  # Exact match
        )
        # Should not raise
        _ensure_fitting_window_within_timespan(basemodel, calibration)

    def test_epiweek_fitting_window_within_timespan_valid(self):
        """Test that fitting window via epiweeks within timespan is valid."""
        basemodel = self._create_basemodel_with_timespan(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
        )
        # Simulate epiweek-based fitting window
        calibration = self._create_calibration_with_fitting_window(
            start_date=None,  # Not using dates directly
            end_date=None,
            epiweek_start_date=date(2024, 3, 3),  # Computed from epiweeks
            epiweek_end_date=date(2024, 6, 29),
        )
        # Should not raise
        _ensure_fitting_window_within_timespan(basemodel, calibration)

    def test_epiweek_fitting_window_end_exceeds_timespan_invalid(self):
        """Test that epiweek-based fitting window end exceeding timespan raises error."""
        basemodel = self._create_basemodel_with_timespan(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 30),
        )
        # Simulate epiweek-based fitting window
        calibration = self._create_calibration_with_fitting_window(
            start_date=None,
            end_date=None,
            epiweek_start_date=date(2024, 3, 3),
            epiweek_end_date=date(2024, 12, 28),  # Exceeds timespan end
        )
        with pytest.raises(ValueError, match="exceeds simulation timespan end_date"):
            _ensure_fitting_window_within_timespan(basemodel, calibration)
