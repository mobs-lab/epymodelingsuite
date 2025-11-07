"""Tests for the telemetry module."""

import json
import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

from epymodelingsuite.schema.calibration import CalibrationStrategy, FittingWindow
from epymodelingsuite.schema.dispatcher import CalibrationOutput, SimulationOutput
from epymodelingsuite.telemetry import (
    ExecutionTelemetry,
    create_workflow_telemetry,
    extract_builder_metadata,
)


def make_mock_simulation_output(primary_id: int, population: str, n_sims: int = 100):
    """Create a mock SimulationOutput for testing."""
    mock_results = MagicMock()
    mock_results.Nsim = n_sims

    # Use model_construct to bypass Pydantic validation for testing
    return SimulationOutput.model_construct(
        primary_id=primary_id,
        seed=42,
        delta_t=1.0,
        population=population,
        results=mock_results,
    )


def make_mock_calibration_output(
    primary_id: int,
    population: str,
    particles_accepted: int | None = None,
):
    """Create a mock CalibrationOutput for testing."""
    mock_results = MagicMock()

    if particles_accepted is not None:
        mock_results.accepted = [MagicMock() for _ in range(particles_accepted)]
    else:
        mock_results.accepted = None

    # No projections for pure calibration
    mock_results.projections = {}

    # Use model_construct to bypass Pydantic validation for testing
    return CalibrationOutput.model_construct(
        primary_id=primary_id,
        seed=42,
        delta_t=1.0,
        population=population,
        results=mock_results,
    )


def make_mock_projection_output(
    primary_id: int,
    population: str,
    particles_accepted: int | None = None,
    successful_trajectories: int = 987,
    failed_trajectories: int = 0,
):
    """Create a mock CalibrationOutput with projection results for testing."""
    mock_results = MagicMock()

    if particles_accepted is not None:
        mock_results.accepted = [MagicMock() for _ in range(particles_accepted)]
    else:
        mock_results.accepted = None

    # Add projections (dict mapping scenario_id to list of projection dicts)
    # Non-empty dict = successful projection, empty dict {} = failed projection
    projections_list = []
    for _ in range(successful_trajectories):
        projections_list.append({"date": "2024-01-01", "data": 100})  # Non-empty = successful
    for _ in range(failed_trajectories):
        projections_list.append({})  # Empty dict = failed

    mock_results.projections = {"baseline": projections_list}

    # Use model_construct to bypass Pydantic validation for testing
    return CalibrationOutput.model_construct(
        primary_id=primary_id,
        seed=42,
        delta_t=1.0,
        population=population,
        results=mock_results,
    )


class TestExecutionTelemetry:
    """Tests for ExecutionTelemetry class."""

    def test_initialization(self):
        """Test ExecutionTelemetry initialization."""
        telemetry = ExecutionTelemetry()
        assert telemetry.status == "running"
        assert telemetry.warnings == []
        assert telemetry.metadata["process_id"] is not None
        assert "python_version" in telemetry.metadata

    def test_builder_stage(self):
        """Test builder stage tracking."""
        telemetry = ExecutionTelemetry()
        telemetry.enter_builder("simulation")

        assert "start_time" in telemetry.builder
        assert telemetry.configuration["workflow_type"] == "simulation"

        telemetry.exit_builder(
            n_models=3,
            populations=["US-CA", "US-NY"],
            start_date="2024-01-01",
            end_date="2024-12-31",
            delta_t=1.0,
            random_seed=42,
        )

        assert telemetry.builder["n_models"] == 3
        assert telemetry.configuration["populations"] == ["US-CA", "US-NY"]
        assert telemetry.configuration["n_populations"] == 2
        assert telemetry.configuration["start_date"] == "2024-01-01"
        assert telemetry.configuration["end_date"] == "2024-12-31"
        assert telemetry.configuration["delta_t"] == 1.0
        assert telemetry.configuration["random_seed"] == 42
        assert "duration_seconds" in telemetry.builder
        assert "peak_memory_mb" in telemetry.builder

    def test_builder_stage_with_fitting_window(self):
        """Test builder stage tracking with fitting window."""
        telemetry = ExecutionTelemetry()
        telemetry.enter_builder("calibration")

        telemetry.exit_builder(
            n_models=1,
            populations=["US-CA"],
            start_date="2024-01-01",
            end_date="2024-12-31",
            delta_t=1.0,
            random_seed=42,
            fitting_window=("2024-03-01", "2024-06-01"),
        )

        assert telemetry.configuration["fitting_window"]["start_date"] == "2024-03-01"
        assert telemetry.configuration["fitting_window"]["end_date"] == "2024-06-01"

    def test_runner_stage(self):
        """Test runner stage tracking."""
        telemetry = ExecutionTelemetry()
        telemetry.enter_runner()

        assert "start_time" in telemetry.runner

        # Add simulation metrics
        sim_output = make_mock_simulation_output(primary_id=0, population="US-CA", n_sims=100)
        telemetry.capture_simulation(sim_output, duration=10.5)

        assert len(telemetry.runner["models"]) == 1
        model = telemetry.runner["models"][0]
        assert model["primary_id"] == 0
        assert model["population"] == "US-CA"
        assert model["calibration"]["duration_seconds"] == 10.5
        assert model["calibration"]["n_sims"] == 100

        # Add calibration with projection metrics
        proj_output = make_mock_projection_output(
            primary_id=1, population="US-NY", particles_accepted=4500, successful_trajectories=987
        )
        telemetry.capture_projection(proj_output, calib_duration=120.0, proj_duration=30.0, n_trajectories=1000)

        assert len(telemetry.runner["models"]) == 2
        model = telemetry.runner["models"][1]
        assert model["calibration"]["particles_accepted"] == 4500
        assert model["projection"]["successful_trajectories"] == 987
        assert model["projection"]["failed_trajectories"] == 13

        telemetry.exit_runner()
        assert "duration_seconds" in telemetry.runner

    def test_output_stage(self):
        """Test output stage tracking."""
        telemetry = ExecutionTelemetry()
        telemetry.enter_output()

        assert "start_time" in telemetry.output
        assert telemetry.output["files"] == []

        telemetry.capture_file("quantiles.csv.gz", 1024000)
        telemetry.capture_file("trajectories.csv.gz", 5120000)

        assert len(telemetry.output["files"]) == 2
        assert telemetry.output["files"][0]["name"] == "quantiles.csv.gz"
        assert telemetry.output["files"][0]["size_bytes"] == 1024000

        telemetry.exit_output()

        assert "duration_seconds" in telemetry.output
        assert "total_size_bytes" in telemetry.output
        assert telemetry.output["total_size_bytes"] == 1024000 + 5120000  # Sum of file sizes
        assert telemetry.status == "completed"

    def test_error_tracking(self):
        """Test error tracking."""
        telemetry = ExecutionTelemetry()
        telemetry.enter_runner()

        sim_output = make_mock_simulation_output(primary_id=0, population="US-CA")
        telemetry.capture_simulation(sim_output, duration=5.0, error="Simulation failed")

        assert len(telemetry.runner["errors"]) == 1
        assert telemetry.runner["errors"][0]["error"] == "Simulation failed"

    def test_warning_messages(self):
        """Test warning message tracking."""
        telemetry = ExecutionTelemetry()
        telemetry.record_warning("Test warning 1")
        telemetry.record_warning("Test warning 2")

        assert len(telemetry.warnings) == 2
        assert "Test warning 1" in telemetry.warnings
        assert "Test warning 2" in telemetry.warnings

    def test_to_dict(self):
        """Test conversion to dictionary."""
        telemetry = ExecutionTelemetry()
        telemetry.enter_builder("calibration")
        telemetry.exit_builder(n_models=1, populations=["US-CA"])
        telemetry.enter_runner()
        telemetry.exit_runner()
        telemetry.enter_output()
        telemetry.exit_output()

        data = telemetry.to_dict()

        assert "metadata" in data
        assert "configuration" in data
        assert "builder" in data
        assert "runner" in data
        assert "output" in data
        assert "resources" in data
        assert "status" in data
        assert "warnings" in data
        assert data["status"] == "completed"


class TestSummaryFormatting:
    """Tests for summary formatting functions."""

    def test_to_text(self):
        """Test text summary generation."""
        telemetry = ExecutionTelemetry()
        telemetry.enter_builder("simulation")
        telemetry.exit_builder(
            n_models=2,
            populations=["US-CA", "US-NY"],
            start_date="2024-01-01",
            end_date="2024-12-31",
            delta_t=1.0,
            random_seed=42,
        )
        telemetry.enter_runner()
        sim_output = make_mock_simulation_output(primary_id=0, population="US-CA", n_sims=100)
        telemetry.capture_simulation(sim_output, duration=5.0)
        telemetry.exit_runner()
        telemetry.enter_output()
        telemetry.capture_file("test.csv.gz", 1024)
        telemetry.exit_output()

        text = telemetry.to_text()

        assert "Telemetry Summary" in text
        assert "CONFIGURATION" in text
        assert "BUILDER STAGE" in text
        assert "RUNNER STAGE" in text
        assert "OUTPUT STAGE" in text
        assert "SUMMARY" in text
        assert "US-CA" in text
        assert "2024-01-01" in text

    def test_to_json(self):
        """Test JSON summary generation."""
        telemetry = ExecutionTelemetry()
        telemetry.enter_builder("calibration")
        telemetry.exit_builder(n_models=1, populations=["US-CA"])
        telemetry.enter_runner()
        telemetry.exit_runner()
        telemetry.enter_output()
        telemetry.exit_output()

        json_str = telemetry.to_json()
        data = json.loads(json_str)

        assert "metadata" in data
        assert "configuration" in data
        assert "builder" in data
        assert "runner" in data
        assert "output" in data
        assert data["status"] == "completed"

    def test_str(self):
        """Test __str__ method returns full text summary."""
        telemetry = ExecutionTelemetry()
        telemetry.enter_builder("calibration_projection")
        telemetry.exit_builder(n_models=2, populations=["US-CA", "US-NY"])
        telemetry.enter_runner()
        sim_output_1 = make_mock_simulation_output(primary_id=0, population="US-CA")
        telemetry.capture_simulation(sim_output_1, duration=5.0)
        sim_output_2 = make_mock_simulation_output(primary_id=1, population="US-NY")
        telemetry.capture_simulation(sim_output_2, duration=6.0)
        telemetry.exit_runner()
        telemetry.enter_output()
        telemetry.exit_output()

        summary_str = str(telemetry)

        # __str__ should return the same as to_text()
        assert summary_str == telemetry.to_text()
        assert "Telemetry Summary" in summary_str
        assert "Calibration Projection" in summary_str

    def test_repr(self):
        """Test __repr__ method for developer-friendly representation."""
        telemetry = ExecutionTelemetry()
        telemetry.enter_builder("simulation")
        telemetry.exit_builder(n_models=1, populations=["US-CA"])
        telemetry.enter_runner()
        telemetry.exit_runner()
        telemetry.enter_output()
        telemetry.exit_output()

        repr_str = repr(telemetry)

        assert "ExecutionTelemetry" in repr_str
        assert "workflow='simulation'" in repr_str
        assert "duration=" in repr_str

    def test_to_text_with_path(self):
        """Test to_text() writing to file."""
        telemetry = ExecutionTelemetry()
        telemetry.enter_builder("simulation")
        telemetry.exit_builder(n_models=1, populations=["US-CA"])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.txt"
            result = telemetry.to_text(path)

            # Should return None when writing to file
            assert result is None
            # File should exist with content
            assert path.exists()
            content = path.read_text()
            assert "Telemetry Summary" in content
            assert "Stage: Builder" in content

    def test_to_json_with_path(self):
        """Test to_json() writing to file."""
        telemetry = ExecutionTelemetry()
        telemetry.enter_builder("calibration")
        telemetry.exit_builder(n_models=2, populations=["US-CA", "US-NY"])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            result = telemetry.to_json(path)

            # Should return None when writing to file
            assert result is None
            # File should exist with valid JSON
            assert path.exists()
            data = json.loads(path.read_text())
            assert "builder" in data
            assert data["builder"]["n_models"] == 2


class TestCompleteWorkflow:
    """Test complete workflow tracking."""

    def test_simulation_workflow(self):
        """Test tracking a complete simulation workflow."""
        telemetry = ExecutionTelemetry()

        # Builder stage
        telemetry.enter_builder("simulation")
        telemetry.exit_builder(
            n_models=2,
            populations=["US-CA", "US-NY"],
            start_date="2024-01-01",
            end_date="2024-12-31",
            delta_t=1.0,
            random_seed=42,
        )

        # Runner stage
        telemetry.enter_runner()
        sim_output_1 = make_mock_simulation_output(0, "US-CA", n_sims=100)
        telemetry.capture_simulation(sim_output_1, duration=5.0)
        sim_output_2 = make_mock_simulation_output(1, "US-NY", n_sims=100)
        telemetry.capture_simulation(sim_output_2, duration=6.0)
        telemetry.exit_runner()

        # Output stage
        telemetry.enter_output()
        telemetry.capture_file("quantiles.csv.gz", 1024)
        telemetry.capture_file("trajectories.csv.gz", 5120)
        telemetry.exit_output()

        # Verify
        assert telemetry.status == "completed"
        assert telemetry.configuration["workflow_type"] == "simulation"
        assert len(telemetry.runner["models"]) == 2
        assert len(telemetry.output["files"]) == 2
        assert "total_duration_seconds" in telemetry.metadata

    def test_calibration_with_projection_workflow(self):
        """Test tracking a calibration with projection workflow."""
        telemetry = ExecutionTelemetry()

        # Builder stage
        telemetry.enter_builder("calibration_projection")
        telemetry.exit_builder(
            n_models=1,
            populations=["US-CA"],
            start_date="2024-01-01",
            end_date="2024-12-31",
            delta_t=1.0,
            random_seed=42,
        )

        # Runner stage
        telemetry.enter_runner()
        proj_output = make_mock_projection_output(
            primary_id=0, population="US-CA", particles_accepted=4500, successful_trajectories=987
        )
        telemetry.capture_projection(proj_output, calib_duration=120.0, proj_duration=30.0, n_trajectories=1000)
        telemetry.exit_runner()

        # Output stage
        telemetry.enter_output()
        telemetry.capture_file("quantiles_compartments.csv.gz", 1024000)
        telemetry.capture_file("trajectories_compartments.csv.gz", 5120000)
        telemetry.capture_file("posteriors.csv.gz", 512000)
        telemetry.exit_output()

        # Verify
        assert telemetry.status == "completed"
        assert telemetry.configuration["workflow_type"] == "calibration_projection"
        assert len(telemetry.runner["models"]) == 1
        model = telemetry.runner["models"][0]
        assert "calibration" in model
        assert "projection" in model
        assert model["projection"]["failed_trajectories"] == 13


class TestAggregation:
    """Test summary aggregation functionality."""

    def test_aggregate_with_all_stages(self):
        """Test aggregating summaries from all stages."""
        # Builder
        builder_telemetry = ExecutionTelemetry()
        builder_telemetry.enter_builder("calibration")
        builder_telemetry.exit_builder(n_models=2, populations=["US-CA", "US-NY"])

        # Runner (2 parallel tasks)
        runner_telemetry_1 = ExecutionTelemetry()
        runner_telemetry_1.enter_runner()
        calib_output_1 = make_mock_calibration_output(0, "US-CA")
        runner_telemetry_1.capture_calibration(calib_output_1, duration=10.0)
        runner_telemetry_1.exit_runner()

        runner_telemetry_2 = ExecutionTelemetry()
        runner_telemetry_2.enter_runner()
        calib_output_2 = make_mock_calibration_output(1, "US-NY")
        runner_telemetry_2.capture_calibration(calib_output_2, duration=15.0)
        runner_telemetry_2.exit_runner()

        # Output
        output_telemetry = ExecutionTelemetry()
        output_telemetry.enter_output()
        output_telemetry.capture_file("test.csv.gz", 1024)
        output_telemetry.exit_output()

        # Create workflow telemetry
        workflow = create_workflow_telemetry(
            builder_telemetry=builder_telemetry,
            runner_telemetries=[runner_telemetry_1, runner_telemetry_2],
            output_telemetry=output_telemetry,
        )

        # Verify workflow summary exists and contains all stages
        assert workflow is not None
        assert workflow.builder  # Has builder data
        assert workflow.runner  # Has runner data
        assert workflow.output  # Has output data

        # Verify runner aggregation (2 parallel tasks combined)
        assert len(workflow.runner["models"]) == 2

        # Test saving to files
        with tempfile.TemporaryDirectory() as tmpdir:
            workflow.to_text(Path(tmpdir) / "workflow.txt")
            workflow.to_json(Path(tmpdir) / "workflow.json")

            assert (Path(tmpdir) / "workflow.txt").exists()
            assert (Path(tmpdir) / "workflow.json").exists()

    def test_aggregate_with_missing_stages(self):
        """Test aggregation when some stages are missing."""
        # Only runner summary available
        runner_telemetry = ExecutionTelemetry()
        runner_telemetry.enter_runner()
        calib_output = make_mock_calibration_output(0, "US-CA")
        runner_telemetry.capture_calibration(calib_output, duration=10.0)
        runner_telemetry.exit_runner()

        workflow = create_workflow_telemetry(
            builder_telemetry=None,
            runner_telemetries=[runner_telemetry],
            output_telemetry=None,
        )

        # Workflow should exist and have only runner data
        assert workflow is not None
        assert not workflow.builder  # No builder data
        assert workflow.runner  # Has runner data
        assert not workflow.output  # No output data

    def test_aggregate_with_empty_runner_list(self):
        """Test aggregation with empty runner summaries list."""
        builder_telemetry = ExecutionTelemetry()
        builder_telemetry.enter_builder("simulation")
        builder_telemetry.exit_builder(n_models=1, populations=["US-CA"])

        workflow = create_workflow_telemetry(
            builder_telemetry=builder_telemetry,
            runner_telemetries=[],
            output_telemetry=None,
        )

        # Workflow should exist and have only builder data
        assert workflow is not None
        assert workflow.builder  # Has builder data
        assert not workflow.runner.get("models")  # No runner models
        assert not workflow.output  # No output data


class TestPartialWorkflows:
    """Test duration calculation for partial workflows."""

    def test_runner_only_workflow(self):
        """Test total duration calculation when only runner stage runs."""
        telemetry = ExecutionTelemetry()
        telemetry.enter_runner()
        sim_output = make_mock_simulation_output(0, "US-CA")
        telemetry.capture_simulation(sim_output, duration=5.0)
        telemetry.exit_runner()

        # Manually call _calculate_total_duration (normally called by exit_output)
        telemetry._calculate_total_duration()

        # Should calculate duration from runner start to runner end
        assert "total_duration_seconds" in telemetry.metadata
        assert telemetry.metadata["total_duration_seconds"] > 0

    def test_builder_and_runner_only(self):
        """Test total duration when builder and runner run but output doesn't."""
        telemetry = ExecutionTelemetry()

        # Builder stage
        telemetry.enter_builder("simulation")
        telemetry.exit_builder(n_models=1, populations=["US-CA"])

        # Runner stage
        telemetry.enter_runner()
        sim_output = make_mock_simulation_output(0, "US-CA")
        telemetry.capture_simulation(sim_output, duration=5.0)
        telemetry.exit_runner()

        # Manually call _calculate_total_duration
        telemetry._calculate_total_duration()

        # Should calculate from builder start to runner end
        assert "total_duration_seconds" in telemetry.metadata
        assert telemetry.metadata["total_duration_seconds"] > 0

    def test_no_stages_run(self):
        """Test that no duration is calculated when no stages run."""
        telemetry = ExecutionTelemetry()

        # Don't run any stages
        telemetry._calculate_total_duration()

        # Should not have total duration
        assert "total_duration_seconds" not in telemetry.metadata

    def test_only_builder_run(self):
        """Test total duration when only builder runs."""
        telemetry = ExecutionTelemetry()
        telemetry.enter_builder("simulation")
        telemetry.exit_builder(n_models=1, populations=["US-CA"])

        # Manually call _calculate_total_duration
        telemetry._calculate_total_duration()

        # Should calculate from builder start to builder end
        assert "total_duration_seconds" in telemetry.metadata
        assert telemetry.metadata["total_duration_seconds"] > 0


class TestFittingWindow:
    """Test fitting window tracking in telemetry."""

    def test_extract_builder_metadata_with_fitting_window(self):
        """Test that fitting window is extracted from calibration config."""
        # Create mock calibration config with fitting window
        calibration_config = MagicMock()
        calibration_config.modelset.calibration.fitting_window = FittingWindow(
            start_date=date(2024, 3, 1), end_date=date(2024, 6, 1)
        )

        # Create mock basemodel config
        basemodel_config = MagicMock()
        basemodel_config.model.population.name = "US-CA"
        basemodel_config.model.timespan.start_date = date(2024, 1, 1)
        basemodel_config.model.timespan.end_date = date(2024, 12, 31)
        basemodel_config.model.timespan.delta_t = 1.0
        basemodel_config.model.random_seed = 42

        # Create mock builder output
        builder_output = MagicMock()
        builder_output.model = None
        builder_output.calibrator.parameters = {"epimodel": MagicMock(population=MagicMock(name="US-CA"))}

        # Extract metadata
        configs = {"basemodel_config": basemodel_config, "calibration_config": calibration_config}
        metadata = extract_builder_metadata([builder_output], configs)

        # Verify fitting window is present
        assert "fitting_window" in metadata
        assert metadata["fitting_window"] == ("2024-03-01", "2024-06-01")

    def test_extract_builder_metadata_without_calibration_config(self):
        """Test that fitting window is not present for simulation workflow."""
        # Create mock basemodel config only (no calibration config)
        basemodel_config = MagicMock()
        basemodel_config.model.population.name = "US-CA"
        basemodel_config.model.timespan.start_date = date(2024, 1, 1)
        basemodel_config.model.timespan.end_date = date(2024, 12, 31)
        basemodel_config.model.timespan.delta_t = 1.0
        basemodel_config.model.random_seed = 42

        # Create mock builder output
        builder_output = MagicMock()
        builder_output.model.population.name = "US-CA"

        # Extract metadata (no calibration config)
        configs = {"basemodel_config": basemodel_config}
        metadata = extract_builder_metadata(builder_output, configs)

        # Verify fitting window is NOT present
        assert "fitting_window" not in metadata

    def test_fitting_window_in_text_output(self):
        """Test that fitting window appears in text output."""
        telemetry = ExecutionTelemetry()
        telemetry.enter_builder("calibration")
        telemetry.exit_builder(
            n_models=1,
            populations=["US-CA"],
            start_date="2024-01-01",
            end_date="2024-12-31",
            delta_t=1.0,
            random_seed=42,
            fitting_window=("2024-03-01", "2024-06-01"),
        )

        text_output = telemetry.to_text()

        # Verify fitting window appears in text output
        assert "CONFIGURATION" in text_output
        assert "Fitting window: 2024-03-01 to 2024-06-01" in text_output
        # Verify it appears after timespan
        assert text_output.index("Timespan:") < text_output.index("Fitting window:")

    def test_fitting_window_in_json_output(self):
        """Test that fitting window appears in JSON output."""
        telemetry = ExecutionTelemetry()
        telemetry.enter_builder("calibration")
        telemetry.exit_builder(
            n_models=1,
            populations=["US-CA"],
            start_date="2024-01-01",
            end_date="2024-12-31",
            delta_t=1.0,
            random_seed=42,
            fitting_window=("2024-03-01", "2024-06-01"),
        )

        json_data = telemetry.to_dict()

        # Verify fitting window is in configuration
        assert "fitting_window" in json_data["configuration"]
        assert json_data["configuration"]["fitting_window"]["start_date"] == "2024-03-01"
        assert json_data["configuration"]["fitting_window"]["end_date"] == "2024-06-01"

    def test_no_fitting_window_for_simulation(self):
        """Test that fitting window is not present for simulation workflow."""
        telemetry = ExecutionTelemetry()
        telemetry.enter_builder("simulation")
        telemetry.exit_builder(
            n_models=1,
            populations=["US-CA"],
            start_date="2024-01-01",
            end_date="2024-12-31",
            delta_t=1.0,
            random_seed=42,
        )

        text_output = telemetry.to_text()
        json_data = telemetry.to_dict()

        # Verify fitting window is NOT present
        assert "Fitting window:" not in text_output
        assert "fitting_window" not in json_data["configuration"]


class TestCalibrationSubsection:
    """Test calibration subsection and age groups in telemetry."""

    def test_age_groups_display(self):
        """Test that age groups appear inline with populations."""
        telemetry = ExecutionTelemetry()
        telemetry.enter_builder("simulation")
        telemetry.exit_builder(
            n_models=1,
            populations=["US-CA"],
            start_date="2024-01-01",
            end_date="2024-12-31",
            delta_t=1.0,
            random_seed=42,
            age_groups=["0-4", "5-17", "18-49", "50-64", "65+"],
        )

        text_output = telemetry.to_text()
        json_data = telemetry.to_dict()

        # Verify age groups appear in text
        assert "Age groups: 5 (0-4, 5-17, 18-49, 50-64, 65+)" in text_output
        # Verify it appears after populations line
        assert text_output.index("Populations:") < text_output.index("Age groups:")

        # Verify in JSON
        assert "age_groups" in json_data["configuration"]
        assert json_data["configuration"]["age_groups"] == ["0-4", "5-17", "18-49", "50-64", "65+"]

    def test_calibration_subsection_with_all_fields(self):
        """Test calibration subsection with fitting window and distance function."""
        telemetry = ExecutionTelemetry()
        telemetry.enter_builder("calibration")
        telemetry.exit_builder(
            n_models=1,
            populations=["US-CA"],
            start_date="2024-01-01",
            end_date="2024-12-31",
            delta_t=1.0,
            random_seed=42,
            fitting_window=("2024-03-01", "2024-06-01"),
            distance_function="rmse",
        )

        text_output = telemetry.to_text()
        json_data = telemetry.to_dict()

        # Verify calibration subsection appears
        assert "Calibration:" in text_output
        assert "  Fitting window: 2024-03-01 to 2024-06-01" in text_output
        assert "  Distance function: rmse" in text_output

        # Verify ordering: Calibration subsection after main config items
        assert text_output.index("Random seed:") < text_output.index("Calibration:")
        assert text_output.index("Calibration:") < text_output.index("Fitting window:")
        assert text_output.index("Fitting window:") < text_output.index("Distance function:")

        # Verify in JSON
        assert json_data["configuration"]["distance_function"] == "rmse"
        assert json_data["configuration"]["fitting_window"]["start_date"] == "2024-03-01"

    def test_calibration_subsection_fitting_window_only(self):
        """Test calibration subsection with only fitting window."""
        telemetry = ExecutionTelemetry()
        telemetry.enter_builder("calibration")
        telemetry.exit_builder(
            n_models=1,
            populations=["US-CA"],
            fitting_window=("2024-03-01", "2024-06-01"),
        )

        text_output = telemetry.to_text()

        # Verify calibration subsection appears with only fitting window
        assert "Calibration:" in text_output
        assert "  Fitting window: 2024-03-01 to 2024-06-01" in text_output
        assert "Distance function:" not in text_output

    def test_calibration_subsection_distance_only(self):
        """Test calibration subsection with only distance function."""
        telemetry = ExecutionTelemetry()
        telemetry.enter_builder("calibration")
        telemetry.exit_builder(
            n_models=1,
            populations=["US-CA"],
            distance_function="mae",
        )

        text_output = telemetry.to_text()

        # Verify calibration subsection appears with only distance function
        assert "Calibration:" in text_output
        assert "  Distance function: mae" in text_output
        assert "Fitting window:" not in text_output

    def test_no_calibration_subsection_for_simulation(self):
        """Test that calibration subsection doesn't appear for simulation workflow."""
        telemetry = ExecutionTelemetry()
        telemetry.enter_builder("simulation")
        telemetry.exit_builder(
            n_models=1,
            populations=["US-CA"],
            start_date="2024-01-01",
            end_date="2024-12-31",
        )

        text_output = telemetry.to_text()

        # Verify no calibration subsection
        assert "Calibration:" not in text_output
        assert "Fitting window:" not in text_output
        assert "Distance function:" not in text_output

    def test_extract_builder_metadata_with_age_groups_and_distance(self):
        """Test extraction of age groups and distance function from configs."""
        # Create mock calibration config
        calibration_config = MagicMock()
        calibration_config.modelset.calibration.fitting_window = FittingWindow(
            start_date=date(2024, 3, 1), end_date=date(2024, 6, 1)
        )
        calibration_config.modelset.calibration.distance_function = "rmse"

        # Create mock basemodel config
        basemodel_config = MagicMock()
        basemodel_config.model.population.name = "US-CA"
        basemodel_config.model.population.age_groups = ["0-4", "5-17", "18-49", "50-64", "65+"]
        basemodel_config.model.timespan.start_date = date(2024, 1, 1)
        basemodel_config.model.timespan.end_date = date(2024, 12, 31)
        basemodel_config.model.timespan.delta_t = 1.0
        basemodel_config.model.random_seed = 42

        # Create mock builder output
        builder_output = MagicMock()
        builder_output.model = None
        builder_output.calibrator.parameters = {"epimodel": MagicMock(population=MagicMock(name="US-CA"))}

        # Extract metadata
        configs = {"basemodel_config": basemodel_config, "calibration_config": calibration_config}
        metadata = extract_builder_metadata([builder_output], configs)

        # Verify all fields extracted
        assert "age_groups" in metadata
        assert metadata["age_groups"] == ["0-4", "5-17", "18-49", "50-64", "65+"]
        assert "distance_function" in metadata
        assert metadata["distance_function"] == "rmse"
        assert "fitting_window" in metadata
        assert metadata["fitting_window"] == ("2024-03-01", "2024-06-01")

    def test_distance_function_not_in_runner_output(self):
        """Test that distance function no longer appears in runner stage."""
        telemetry = ExecutionTelemetry()
        telemetry.enter_builder("calibration")
        telemetry.exit_builder(
            n_models=1,
            populations=["US-CA"],
            distance_function="rmse",
        )

        telemetry.enter_runner()
        calib_output = make_mock_calibration_output(0, "US-CA", particles_accepted=50)
        strategy = CalibrationStrategy(name="SMC", options={"num_particles": 100, "distance_function": "rmse"})
        telemetry.capture_calibration(calib_output, duration=120.0, calibration_strategy=strategy)
        telemetry.exit_runner()

        text_output = telemetry.to_text()

        # Distance function should be in CONFIGURATION, not in runner
        config_section = text_output[: text_output.index("BUILDER STAGE")]
        runner_section = text_output[text_output.index("RUNNER STAGE") :]

        assert "Distance function: rmse" in config_section
        assert "Distance:" not in runner_section
        assert "Distance function:" not in runner_section


class TestStrategyInfoCapture:
    """Test capturing ABC strategy information in telemetry."""

    def test_capture_calibration_with_strategy(self):
        """Test that ABC strategy info is captured for calibration."""
        telemetry = ExecutionTelemetry()
        telemetry.enter_runner()

        calib_output = make_mock_calibration_output(0, "US-CA", particles_accepted=50)
        strategy = CalibrationStrategy(
            name="SMC", options={"num_particles": 100, "num_generations": 5, "max_time": "30m"}
        )

        telemetry.capture_calibration(calib_output, duration=120.0, calibration_strategy=strategy)

        assert len(telemetry.runner["models"]) == 1
        model = telemetry.runner["models"][0]
        assert model["calibration"]["strategy"] == "SMC"
        assert model["calibration"]["num_particles"] == 100
        assert model["calibration"]["num_generations"] == 5
        # max_time should not be included (only particles, generations, distance_function)
        assert "max_time" not in model["calibration"]

    def test_capture_projection_with_strategy(self):
        """Test that ABC strategy info is captured for projection."""
        telemetry = ExecutionTelemetry()
        telemetry.enter_runner()

        proj_output = make_mock_projection_output(0, "US-CA", particles_accepted=50, successful_trajectories=95)
        strategy = CalibrationStrategy(name="SMC", options={"num_particles": 100, "num_generations": 5})

        telemetry.capture_projection(
            proj_output,
            calib_duration=120.0,
            proj_duration=30.0,
            n_trajectories=100,
            calibration_strategy=strategy,
        )

        assert len(telemetry.runner["models"]) == 1
        model = telemetry.runner["models"][0]
        assert model["calibration"]["strategy"] == "SMC"
        assert model["calibration"]["num_particles"] == 100
        assert model["calibration"]["num_generations"] == 5

    def test_capture_without_strategy_info(self):
        """Test that capture works without strategy info (backward compatibility)."""
        telemetry = ExecutionTelemetry()
        telemetry.enter_runner()

        calib_output = make_mock_calibration_output(0, "US-CA", particles_accepted=50)
        telemetry.capture_calibration(calib_output, duration=120.0)

        assert len(telemetry.runner["models"]) == 1
        model = telemetry.runner["models"][0]
        # Should not have strategy info
        assert "strategy" not in model["calibration"]
        assert "num_particles" not in model["calibration"]
        assert "num_generations" not in model["calibration"]
