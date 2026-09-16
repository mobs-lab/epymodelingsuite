"""Tests for epymodelingsuite.dispatcher.output module."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from epymodelingsuite.dispatcher.output import filter_failed_projections, generate_calibration_outputs
from epymodelingsuite.schema.output import (
    CategoricalPlotConfig,
    FigureOutputTypeEnum,
    ModelMetaOutput,
    OutputConfig,
    OutputConfiguration,
    PlotsConfig,
    TabularOutputTypeEnum,
)
from epymodelingsuite.visualization.generators import generate_categorical_plots


class TestFilterFailedProjections:
    """Tests for filter_failed_projections function."""

    @pytest.fixture
    def mock_calibration_results(self):
        """Create a mock calibration results object with projections using realistic data structures."""
        results = MagicMock()
        # Realistic projection structure: dict with "date" (list of dates) and compartment/transition keys
        results.projections = {
            "baseline": [
                {
                    "date": [date(2024, 1, 1), date(2024, 1, 8), date(2024, 1, 15)],
                    "S_0-4": np.array([1000, 990, 980]),
                    "I_0-4": np.array([10, 15, 18]),
                    "R_0-4": np.array([0, 5, 12]),
                    "S_to_I_0-4": np.array([5, 6, 4]),
                },
                {
                    "date": [date(2024, 1, 1), date(2024, 1, 8), date(2024, 1, 15)],
                    "S_0-4": np.array([1000, 985, 975]),
                    "I_0-4": np.array([10, 18, 22]),
                    "R_0-4": np.array([0, 7, 13]),
                    "S_to_I_0-4": np.array([6, 7, 5]),
                },
                {
                    "date": [date(2024, 1, 1), date(2024, 1, 8), date(2024, 1, 15)],
                    "S_0-4": np.array([1000, 988, 978]),
                    "I_0-4": np.array([10, 16, 20]),
                    "R_0-4": np.array([0, 6, 12]),
                    "S_to_I_0-4": np.array([5, 6, 5]),
                },
            ]
        }
        return results

    @pytest.fixture
    def mock_calibration_results_with_failures(self):
        """Create a mock calibration results object with some failed projections (empty dicts)."""
        results = MagicMock()
        results.projections = {
            "baseline": [
                {
                    "date": [date(2024, 1, 1), date(2024, 1, 8), date(2024, 1, 15)],
                    "S_0-4": np.array([1000, 990, 980]),
                    "I_0-4": np.array([10, 15, 18]),
                    "S_to_I_0-4": np.array([5, 6, 4]),
                },
                {},  # Failed projection returns empty dict
                {
                    "date": [date(2024, 1, 1), date(2024, 1, 8), date(2024, 1, 15)],
                    "S_0-4": np.array([1000, 988, 978]),
                    "I_0-4": np.array([10, 16, 20]),
                    "S_to_I_0-4": np.array([5, 6, 5]),
                },
                {},  # Another failed projection
                {
                    "date": [date(2024, 1, 1), date(2024, 1, 8), date(2024, 1, 15)],
                    "S_0-4": np.array([1000, 985, 975]),
                    "I_0-4": np.array([10, 18, 22]),
                    "S_to_I_0-4": np.array([6, 7, 5]),
                },
            ]
        }
        return results

    @pytest.fixture
    def mock_calibration_results_multiple_scenarios(self):
        """Create a mock calibration results object with multiple scenarios and failures."""
        results = MagicMock()
        results.projections = {
            "baseline": [
                {
                    "date": [date(2024, 1, 1), date(2024, 1, 8), date(2024, 1, 15)],
                    "S_0-4": np.array([1000, 990, 980]),
                    "I_0-4": np.array([10, 15, 18]),
                },
                {},  # Failed projection
                {
                    "date": [date(2024, 1, 1), date(2024, 1, 8), date(2024, 1, 15)],
                    "S_0-4": np.array([1000, 988, 978]),
                    "I_0-4": np.array([10, 16, 20]),
                },
            ],
            "intervention": [
                {
                    "date": [date(2024, 1, 1), date(2024, 1, 8), date(2024, 1, 15)],
                    "S_0-4": np.array([1000, 995, 990]),
                    "I_0-4": np.array([10, 12, 14]),
                },
                {
                    "date": [date(2024, 1, 1), date(2024, 1, 8), date(2024, 1, 15)],
                    "S_0-4": np.array([1000, 993, 988]),
                    "I_0-4": np.array([10, 13, 15]),
                },
                {},  # Failed projection
                {},  # Another failed projection
            ],
        }
        return results

    def test_no_projections_attribute_returns_unchanged(self):
        """Test that objects without projections attribute are returned unchanged."""
        results = MagicMock(spec=[])  # No projections attribute
        filtered = filter_failed_projections(results)
        assert filtered is results
        assert filtered._filtered_count == 0

    def test_none_projections_returns_unchanged(self):
        """Test that objects with None projections are returned unchanged."""
        results = MagicMock()
        results.projections = None
        filtered = filter_failed_projections(results)
        assert filtered is results
        assert filtered._filtered_count == 0

    def test_empty_projections_dict_returns_unchanged(self):
        """Test that empty projections dictionary is handled correctly."""
        results = MagicMock()
        results.projections = {}
        filtered = filter_failed_projections(results)
        assert filtered is results
        assert filtered.projections == {}
        assert filtered._filtered_count == 0

    def test_all_valid_simulations_unchanged(self, mock_calibration_results):
        """Test that valid simulations are not modified."""
        original_count = len(mock_calibration_results.projections["baseline"])
        filtered = filter_failed_projections(mock_calibration_results)

        assert filtered is mock_calibration_results
        assert len(filtered.projections["baseline"]) == original_count
        assert all(sim for sim in filtered.projections["baseline"])  # All non-empty
        assert filtered._filtered_count == 0

    def test_filters_out_empty_dicts(self, mock_calibration_results_with_failures):
        """Test that empty dictionaries are filtered out."""
        filtered = filter_failed_projections(mock_calibration_results_with_failures)

        assert len(filtered.projections["baseline"]) == 3
        assert all(sim for sim in filtered.projections["baseline"])  # All non-empty
        assert all("date" in sim for sim in filtered.projections["baseline"])
        assert filtered._filtered_count == 2

    def test_filters_multiple_scenarios(self, mock_calibration_results_multiple_scenarios):
        """Test that filtering works across multiple scenarios."""
        filtered = filter_failed_projections(mock_calibration_results_multiple_scenarios)

        # Baseline should have 2 valid simulations
        assert len(filtered.projections["baseline"]) == 2
        assert all(sim for sim in filtered.projections["baseline"])

        # Intervention should have 2 valid simulations
        assert len(filtered.projections["intervention"]) == 2
        assert all(sim for sim in filtered.projections["intervention"])
        assert filtered._filtered_count == 3  # 1 from baseline + 2 from intervention

    def test_all_failed_simulations_returns_empty_list(self):
        """Test that scenario with all failed projections becomes empty list."""
        results = MagicMock()
        results.projections = {
            "baseline": [{}, {}, {}]  # All failed
        }

        filtered = filter_failed_projections(results)

        assert len(filtered.projections["baseline"]) == 0
        assert filtered.projections["baseline"] == []
        assert filtered._filtered_count == 3

    @patch("epymodelingsuite.dispatcher.output.logger")
    def test_logs_warning_when_filtering(self, mock_logger, mock_calibration_results_with_failures):
        """Test that warning is logged when failed projections are filtered."""
        filter_failed_projections(mock_calibration_results_with_failures)

        # Should have logged a warning
        assert mock_logger.warning.called
        call_args = mock_logger.warning.call_args[0]

        # Check the log message format
        assert "Filtered out" in call_args[0]
        assert "failed projection(s)" in call_args[0]
        assert call_args[1] == 2  # Number filtered
        assert call_args[2] == "baseline"  # Scenario ID
        assert call_args[3] == 3  # Number kept
        assert call_args[4] == 5  # Total original

    @patch("epymodelingsuite.dispatcher.output.logger")
    def test_no_warning_when_all_valid(self, mock_logger, mock_calibration_results):
        """Test that no warning is logged when all simulations are valid."""
        filter_failed_projections(mock_calibration_results)

        # Should not have logged a warning
        assert not mock_logger.warning.called

    @patch("epymodelingsuite.dispatcher.output.logger")
    def test_logs_warning_for_each_scenario_with_failures(
        self, mock_logger, mock_calibration_results_multiple_scenarios
    ):
        """Test that warning is logged for each scenario that has failures."""
        filter_failed_projections(mock_calibration_results_multiple_scenarios)

        # Should have logged two warnings (one per scenario with failures)
        assert mock_logger.warning.call_count == 2

    def test_preserves_simulation_data(self, mock_calibration_results_with_failures):
        """Test that valid projection data is preserved exactly."""
        original_valid_sims = [sim for sim in mock_calibration_results_with_failures.projections["baseline"] if sim]

        filtered = filter_failed_projections(mock_calibration_results_with_failures)
        filtered_sims = filtered.projections["baseline"]

        assert len(filtered_sims) == len(original_valid_sims)
        for original, filtered_sim in zip(original_valid_sims, filtered_sims, strict=False):
            assert original == filtered_sim
        assert filtered._filtered_count == 2

    def test_empty_scenario_list_handled(self):
        """Test that empty list for a scenario is handled correctly."""
        results = MagicMock()
        results.projections = {"baseline": []}

        filtered = filter_failed_projections(results)

        assert filtered.projections["baseline"] == []
        assert filtered._filtered_count == 0

    def test_none_scenario_value_handled(self):
        """Test that None value for a scenario is handled correctly."""
        results = MagicMock()
        results.projections = {"baseline": None}

        filtered = filter_failed_projections(results)

        # Should skip None scenario and not crash
        assert filtered is results
        assert filtered._filtered_count == 0

    def test_stores_filtered_count_on_results(self, mock_calibration_results_with_failures):
        """Test that _filtered_count is stored on the results object."""
        filtered = filter_failed_projections(mock_calibration_results_with_failures)

        # Should store the count on the results object
        assert hasattr(filtered, "_filtered_count")
        assert filtered._filtered_count == 2

    def test_stores_filtered_count_zero_when_no_failures(self, mock_calibration_results):
        """Test that _filtered_count is 0 when there are no failures."""
        filtered = filter_failed_projections(mock_calibration_results)

        # Should store 0 when no failures
        assert hasattr(filtered, "_filtered_count")
        assert filtered._filtered_count == 0

    def test_stores_filtered_count_multiple_scenarios(self, mock_calibration_results_multiple_scenarios):
        """Test that _filtered_count aggregates across all scenarios."""
        filtered = filter_failed_projections(mock_calibration_results_multiple_scenarios)

        # Should aggregate count across scenarios (1 + 2 = 3)
        assert hasattr(filtered, "_filtered_count")
        assert filtered._filtered_count == 3


class TestFilterFailedProjectionsAlignsProjectionParameters:
    """filter_failed_projections must filter projection_parameters with the same mask."""

    @staticmethod
    def _make_projection(seed_value: int) -> dict:
        return {
            "date": [date(2024, 1, 1), date(2024, 1, 8)],
            "S_0-4": np.array([1000, 990 - seed_value]),
        }

    def test_mask_applied_to_projection_parameters(self):
        results = MagicMock()
        results.projections = {
            "baseline": [
                self._make_projection(0),
                {},
                self._make_projection(2),
                {},
                self._make_projection(4),
            ]
        }
        results.projection_parameters = {
            "baseline": pd.DataFrame({"Reff": [1.0, 2.0, 3.0, 4.0, 5.0]}),
        }

        filter_failed_projections(results)

        assert len(results.projections["baseline"]) == 3
        assert results.projection_parameters["baseline"]["Reff"].tolist() == [1.0, 3.0, 5.0]
        assert results.projection_parameters["baseline"].index.tolist() == [0, 1, 2]

    def test_missing_projection_parameters_does_not_raise(self):
        # spec=["projections"] -> no projection_parameters attribute
        results = MagicMock(spec=["projections"])
        results.projections = {
            "baseline": [self._make_projection(0), {}, self._make_projection(2)],
        }

        filter_failed_projections(results)

        assert len(results.projections["baseline"]) == 2

    def test_length_mismatch_logs_and_skips(self, caplog):
        results = MagicMock()
        results.projections = {
            "baseline": [
                self._make_projection(0),
                {},
                self._make_projection(2),
                {},
                self._make_projection(4),
            ]
        }
        # Mismatched length (3 vs projections length 5).
        results.projection_parameters = {
            "baseline": pd.DataFrame({"Reff": [1.0, 2.0, 3.0]}),
        }

        with caplog.at_level("WARNING"):
            filter_failed_projections(results)

        # Projections still filtered.
        assert len(results.projections["baseline"]) == 3
        # projection_parameters left untouched on mismatch.
        assert results.projection_parameters["baseline"]["Reff"].tolist() == [1.0, 2.0, 3.0]
        assert any("differs from projections length" in rec.message for rec in caplog.records)

    def test_multi_scenario_filtered_independently(self):
        results = MagicMock()
        results.projections = {
            "baseline": [self._make_projection(0), {}, self._make_projection(2)],
            "intervention": [{}, self._make_projection(1), self._make_projection(3), {}],
        }
        results.projection_parameters = {
            "baseline": pd.DataFrame({"Reff": [0.1, 0.2, 0.3]}),
            "intervention": pd.DataFrame({"Reff": [1.0, 1.1, 1.2, 1.3]}),
        }

        filter_failed_projections(results)

        assert len(results.projections["baseline"]) == 2
        assert results.projection_parameters["baseline"]["Reff"].tolist() == [0.1, 0.3]
        assert len(results.projections["intervention"]) == 2
        assert results.projection_parameters["intervention"]["Reff"].tolist() == [1.1, 1.2]


class TestProjectionParametersLongFile:
    """generate_calibration_outputs must emit projection_parameters_long as a tidy per-draw table."""

    @staticmethod
    def _build_output_config(*, projection_parameters: bool = True) -> OutputConfig:
        return OutputConfig(
            output=OutputConfiguration(
                tabular_output_types=[TabularOutputTypeEnum.DataFrame],
                quantiles=None,
                trajectories=None,
                posteriors=False,
                flusight_format=None,
                covid19_format=None,
                flusmh_format=None,
                model_meta=ModelMetaOutput(projection_parameters=projection_parameters),
                plots=None,
            )
        )

    @staticmethod
    def _valid_projection() -> dict:
        return {
            "date": [pd.Timestamp("2024-05-01"), pd.Timestamp("2024-05-08")],
            "S_0-4": np.array([1000, 990]),
        }

    def _make_calibration(
        self,
        *,
        primary_id: int = 1,
        seed: int = 42,
        population: str = "United_States_California",
        projections: dict | None = None,
        projection_parameters: dict | None = None,
    ) -> MagicMock:
        calibration = MagicMock()
        calibration.primary_id = primary_id
        calibration.seed = seed
        calibration.delta_t = 1.0
        calibration.population = population
        calibration.start_date_reference = None
        calibration.results.projections = projections if projections is not None else {}
        if projection_parameters is not None:
            calibration.results.projection_parameters = projection_parameters
        # Empty dicts make the fitting/projection-window branches skip to the None-append branch
        # without triggering the exception-handling warning path.
        calibration.results.get_calibration_trajectories.return_value = {}
        calibration.results.get_projection_trajectories.return_value = {}
        return calibration

    def test_long_file_shape_and_columns(self):
        projections = {"baseline": [self._valid_projection() for _ in range(1000)]}
        proj_params = pd.DataFrame(
            {
                "Reff": np.linspace(1.0, 2.0, 1000),
                "alpha": np.linspace(3.0, 9.0, 1000),
            }
        )
        calibration = self._make_calibration(
            projections=projections,
            projection_parameters={"baseline": proj_params},
        )

        outputs = generate_calibration_outputs(calibrations=[calibration], output_config=self._build_output_config())

        assert "projection_parameters_long" in outputs
        long_df = outputs["projection_parameters_long"][0].data
        assert long_df.shape == (1000, 7)
        assert list(long_df.columns) == [
            "primary_id",
            "sim_id",
            "scenario_id",
            "seed",
            "population",
            "Reff",
            "alpha",
        ]

    def test_long_file_content_matches_source(self):
        projections = {"baseline": [self._valid_projection() for _ in range(1000)]}
        proj_params = pd.DataFrame(
            {
                "Reff": np.linspace(1.0, 2.0, 1000),
                "alpha": np.linspace(3.0, 9.0, 1000),
            }
        )
        calibration = self._make_calibration(
            projections=projections,
            projection_parameters={"baseline": proj_params},
        )

        outputs = generate_calibration_outputs(calibrations=[calibration], output_config=self._build_output_config())
        long_df = outputs["projection_parameters_long"][0].data

        assert long_df.iloc[0]["Reff"] == proj_params.iloc[0]["Reff"]
        assert long_df.iloc[999]["Reff"] == proj_params.iloc[999]["Reff"]
        assert long_df["sim_id"].tolist() == list(range(1000))
        assert (long_df["scenario_id"] == "baseline").all()
        # Guard against regression to pandas-Series repr text (which truncates with "...").
        for column in ("Reff", "alpha"):
            assert long_df[column].dtype.kind == "f"
            assert not long_df[column].astype(str).str.contains(r"\.\.\.", regex=True).any()

    def test_sim_id_aligns_after_filter(self):
        projections = [self._valid_projection() for _ in range(1000)]
        for failed_index in (5, 17, 999):
            projections[failed_index] = {}
        proj_params = pd.DataFrame({"Reff": np.linspace(1.0, 2.0, 1000)})
        calibration = self._make_calibration(
            projections={"baseline": projections},
            projection_parameters={"baseline": proj_params.copy()},
        )

        outputs = generate_calibration_outputs(calibrations=[calibration], output_config=self._build_output_config())
        long_df = outputs["projection_parameters_long"][0].data

        assert len(long_df) == 997
        assert long_df["sim_id"].tolist() == list(range(997))
        assert long_df.loc[0, "Reff"] == proj_params.iloc[0]["Reff"]
        # Original row 5 was dropped, so long-file row 5 comes from original row 6.
        assert long_df.loc[5, "Reff"] == proj_params.iloc[6]["Reff"]
        # Original row 999 dropped; last kept row in params is original 998.
        assert long_df.loc[996, "Reff"] == proj_params.iloc[998]["Reff"]

    def test_multi_scenario_long_file(self):
        projections = {
            "baseline": [self._valid_projection() for _ in range(100)],
            "counterfactual": [self._valid_projection() for _ in range(100)],
        }
        projection_parameters = {
            "baseline": pd.DataFrame({"Reff": np.linspace(1.0, 2.0, 100)}),
            "counterfactual": pd.DataFrame({"Reff": np.linspace(0.5, 1.5, 100)}),
        }
        calibration = self._make_calibration(projections=projections, projection_parameters=projection_parameters)

        outputs = generate_calibration_outputs(calibrations=[calibration], output_config=self._build_output_config())
        long_df = outputs["projection_parameters_long"][0].data

        assert len(long_df) == 200
        assert set(long_df["scenario_id"].unique()) == {"baseline", "counterfactual"}
        assert set(long_df.columns) == {"primary_id", "sim_id", "scenario_id", "seed", "population", "Reff"}

    def test_missing_projection_parameters_key(self):
        # projections has "baseline" but projection_parameters is empty -> no long file,
        # and no KeyError from the per-model proj_* block.
        calibration = self._make_calibration(
            projections={"baseline": [self._valid_projection()]},
            projection_parameters={},
        )

        outputs = generate_calibration_outputs(calibrations=[calibration], output_config=self._build_output_config())

        assert "projection_parameters_long" not in outputs

    def test_flag_off_emits_nothing(self):
        projections = {"baseline": [self._valid_projection() for _ in range(10)]}
        proj_params = pd.DataFrame({"Reff": np.linspace(1.0, 2.0, 10)})
        calibration = self._make_calibration(
            projections=projections,
            projection_parameters={"baseline": proj_params},
        )

        outputs = generate_calibration_outputs(
            calibrations=[calibration],
            output_config=self._build_output_config(projection_parameters=False),
        )

        assert "projection_parameters_long" not in outputs

    def test_multi_calibration_concat(self):
        params_a = pd.DataFrame({"Reff": np.linspace(1.0, 2.0, 50)})
        params_b = pd.DataFrame({"Reff": np.linspace(0.5, 1.0, 30)})
        calibration_a = self._make_calibration(
            primary_id=101,
            projections={"baseline": [self._valid_projection() for _ in range(50)]},
            projection_parameters={"baseline": params_a},
        )
        calibration_b = self._make_calibration(
            primary_id=202,
            projections={"baseline": [self._valid_projection() for _ in range(30)]},
            projection_parameters={"baseline": params_b},
        )

        outputs = generate_calibration_outputs(
            calibrations=[calibration_a, calibration_b], output_config=self._build_output_config()
        )
        long_df = outputs["projection_parameters_long"][0].data

        assert len(long_df) == 80
        assert set(long_df["primary_id"].unique()) == {101, 202}
        assert (long_df.loc[long_df["primary_id"] == 101, "sim_id"] == np.arange(50)).all()
        assert (long_df.loc[long_df["primary_id"] == 202, "sim_id"] == np.arange(30)).all()


class TestGenerateCategoricalPlots:
    """Tests for generate_categorical_plots function."""

    @pytest.fixture
    def mock_hub_format_data(self):
        """Create mock FluSight hub format data with rate-trend forecasts."""
        data = []
        for horizon in [0, 1, 2, 3]:
            for location in ["06", "48"]:  # FIPS codes for CA and TX
                for category in ["large_decrease", "decrease", "stable", "increase", "large_increase"]:
                    data.append(
                        {
                            "reference_date": date(2025, 11, 26),
                            "location": location,
                            "target": "wk flu hosp rate change",
                            "horizon": horizon,
                            "target_end_date": date(2025, 12, 7 + horizon * 7),
                            "output_type": "pmf",
                            "output_type_id": category,
                            "value": 0.2,  # Equal probabilities for simplicity
                        }
                    )
        return pd.DataFrame(data)

    @pytest.fixture
    def mock_plots_config_enabled(self):
        """Create PlotsConfig with categorical plots enabled."""
        return PlotsConfig(
            reference_date=date(2025, 11, 26),
            figure_output_types=[FigureOutputTypeEnum.PNG],
            dpi=150,
            categorical=CategoricalPlotConfig(),
        )

    @pytest.fixture
    def mock_plots_config_disabled(self):
        """Create PlotsConfig with categorical plots disabled."""
        return PlotsConfig(
            reference_date=date(2025, 11, 26),
            figure_output_types=[FigureOutputTypeEnum.PNG],
            dpi=150,
            categorical=None,
        )

    def test_categorical_plots_from_flusight_data(self, mock_hub_format_data, mock_plots_config_enabled):
        """Test categorical plot generation from FluSight rate-trend data."""
        out_dict = {}

        # Call generate_categorical_plots
        generate_categorical_plots(mock_plots_config_enabled, out_dict, mock_hub_format_data)

        # Assert: "categorical_rate_trends" in out_dict
        assert "categorical_rate_trends" in out_dict

        # Assert: OutputObject has correct types
        assert len(out_dict["categorical_rate_trends"]) == 1  # One output type (PNG)
        output_obj = out_dict["categorical_rate_trends"][0]
        assert output_obj.name == "categorical_rate_trends.png"
        assert output_obj.output_type == FigureOutputTypeEnum.PNG

    def test_categorical_plots_skip_when_disabled(self, mock_hub_format_data, mock_plots_config_disabled):
        """Test categorical plots are skipped when config.enabled=False."""
        out_dict = {}

        # Call generate_categorical_plots with disabled config
        generate_categorical_plots(mock_plots_config_disabled, out_dict, mock_hub_format_data)

        # Assert: "categorical_rate_trends" NOT in out_dict
        assert "categorical_rate_trends" not in out_dict

    def test_categorical_plots_skip_when_no_flusight_format(self, mock_plots_config_enabled, caplog):
        """Test categorical plots are skipped when FluSight format not configured."""
        out_dict = {}

        # Call generate_categorical_plots with hub_format_data=None
        generate_categorical_plots(mock_plots_config_enabled, out_dict, hub_format_data=None)

        # Assert: categorical plots skipped
        assert "categorical_rate_trends" not in out_dict

        # Assert: warning mentions "flusight_format.rate_trends"
        assert any(
            "Categorical plots enabled but no FluSight format data available" in rec.message for rec in caplog.records
        )
        assert any("flusight_format.rate_trends" in rec.message for rec in caplog.records)

    def test_categorical_plots_skip_when_no_rate_trends(self, mock_plots_config_enabled, caplog):
        """Test categorical plots are skipped when rate-trend data missing."""
        out_dict = {}

        # Create hub format data WITHOUT rate-trend forecasts (only hospitalizations)
        hub_format_data = pd.DataFrame(
            {
                "reference_date": [date(2025, 11, 26)],
                "location": ["06"],
                "target": ["wk ahead inc flu hosp"],  # Different target, not rate-trend
                "horizon": [0],
                "target_end_date": [date(2025, 12, 7)],
                "output_type": ["quantile"],
                "output_type_id": ["0.5"],
                "value": [100.0],
            }
        )

        # Call generate_categorical_plots
        generate_categorical_plots(mock_plots_config_enabled, out_dict, hub_format_data)

        # Assert: categorical plots skipped
        assert "categorical_rate_trends" not in out_dict

        # Assert: warning mentions "rate-trend categorical forecasts"
        assert any("No rate-trend categorical forecasts found" in rec.message for rec in caplog.records)
        assert any("flusight_format.rate_trends" in rec.message for rec in caplog.records)


class TestGetHubLocationId:
    """Tests for get_hub_location_id function."""

    def test_iso_location_returns_fips(self):
        """Test that ISO locations return FIPS codes."""
        from epymodelingsuite.dispatcher.output import get_hub_location_id

        # California
        result = get_hub_location_id("United_States_California")
        assert result == "06"

        # Texas
        result = get_hub_location_id("United_States_Texas")
        assert result == "48"

    def test_metrocast_location_returns_location_id(self):
        """Test that metrocast locations return metrocast_location_id."""
        from epymodelingsuite.dispatcher.output import get_hub_location_id

        # Denver
        result = get_hub_location_id("metrocast_location_denver")
        assert result == "denver"

        # Boston
        result = get_hub_location_id("metrocast_location_boston")
        assert result == "boston"

        # NC flu region
        result = get_hub_location_id("metrocast_location_nenc")
        assert result == "nenc"


class TestGetPlotLocationLabel:
    """Tests for get_plot_location_label function."""

    def test_iso_location_returns_name(self):
        """Test that ISO locations return human-readable names."""
        from epymodelingsuite.dispatcher.output import get_plot_location_label

        # California
        result = get_plot_location_label("United_States_California")
        assert result == "California"

        # Massachusetts
        result = get_plot_location_label("United_States_Massachusetts")
        assert result == "Massachusetts"

    def test_metrocast_location_returns_short_name(self):
        """Test that metrocast locations return short name from CSV."""
        from epymodelingsuite.dispatcher.output import get_plot_location_label

        # Denver - same as full name
        result = get_plot_location_label("metrocast_location_denver")
        assert result == "Denver, CO"

        # Boston - shortened from "Boston Metro, South Shore, Cape & Islands, MA"
        result = get_plot_location_label("metrocast_location_boston")
        assert result == "Greater Boston, MA"

        # NC flu region
        result = get_plot_location_label("metrocast_location_nenc")
        assert result == "Northeastern, NC"


class TestFormatQuantilesFlusightforecast:
    """Tests for format_quantiles_flusightforecast function."""

    @pytest.fixture
    def sample_quantiles_df(self):
        """Create sample quantiles DataFrame spanning horizons -1 to 3."""
        reference_date = date(2025, 11, 29)  # Saturday
        # Create dates for horizons -1, 0, 1, 2, 3 (each 7 days apart)
        dates = [
            pd.Timestamp("2025-11-22"),  # horizon -1
            pd.Timestamp("2025-11-29"),  # horizon 0
            pd.Timestamp("2025-12-06"),  # horizon 1
            pd.Timestamp("2025-12-13"),  # horizon 2
            pd.Timestamp("2025-12-20"),  # horizon 3
        ]
        quantiles = [0.25, 0.5, 0.75]
        data = []
        for d in dates:
            for q in quantiles:
                data.append({"date": d, "quantile": q, "hospitalizations": 100.0})
        return pd.DataFrame(data), reference_date

    def test_standard_flusight_includes_horizon_minus_one(self, sample_quantiles_df):
        """Test that standard FluSight (metrocast=False) includes horizon -1."""
        from epymodelingsuite.dispatcher.output import format_quantiles_flusightforecast

        df, reference_date = sample_quantiles_df
        result = format_quantiles_flusightforecast(df, reference_date, metrocast=False)

        horizons = result["horizon"].unique()
        assert -1 in horizons
        assert set(horizons) == {-1, 0, 1, 2, 3}

    def test_metrocast_excludes_horizon_minus_one(self, sample_quantiles_df):
        """Test that metrocast=True excludes horizon -1."""
        from epymodelingsuite.dispatcher.output import format_quantiles_flusightforecast

        df, reference_date = sample_quantiles_df
        result = format_quantiles_flusightforecast(df, reference_date, metrocast=True)

        horizons = result["horizon"].unique()
        assert -1 not in horizons
        assert set(horizons) == {0, 1, 2, 3}

    def test_output_has_correct_columns(self, sample_quantiles_df):
        """Test that output DataFrame has correct FluSight columns."""
        from epymodelingsuite.dispatcher.output import format_quantiles_flusightforecast

        df, reference_date = sample_quantiles_df
        result = format_quantiles_flusightforecast(df, reference_date)

        expected_columns = {"horizon", "target", "output_type", "output_type_id", "target_end_date", "value"}
        assert set(result.columns) == expected_columns

    def test_output_type_is_quantile(self, sample_quantiles_df):
        """Test that output_type column is 'quantile' for all rows."""
        from epymodelingsuite.dispatcher.output import format_quantiles_flusightforecast

        df, reference_date = sample_quantiles_df
        result = format_quantiles_flusightforecast(df, reference_date)

        assert (result["output_type"] == "quantile").all()

    def test_custom_target_name(self, sample_quantiles_df):
        """Test that custom target name is used."""
        from epymodelingsuite.dispatcher.output import format_quantiles_flusightforecast

        df, reference_date = sample_quantiles_df
        custom_target = "custom target name"
        result = format_quantiles_flusightforecast(df, reference_date, target=custom_target)

        assert (result["target"] == custom_target).all()

    def test_default_target_name(self, sample_quantiles_df):
        """Test that default target name is 'wk inc flu hosp'."""
        from epymodelingsuite.dispatcher.output import format_quantiles_flusightforecast

        df, reference_date = sample_quantiles_df
        result = format_quantiles_flusightforecast(df, reference_date)

        assert (result["target"] == "wk inc flu hosp").all()

    def test_hospitalizations_rounded_to_integer(self, sample_quantiles_df):
        """Test that hospitalization values are rounded to integers."""
        from epymodelingsuite.dispatcher.output import format_quantiles_flusightforecast

        df, reference_date = sample_quantiles_df
        # Modify to have non-integer values
        df["hospitalizations"] = 100.7
        result = format_quantiles_flusightforecast(df, reference_date)

        assert (result["value"] == 101).all()
