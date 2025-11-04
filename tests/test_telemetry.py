"""Tests for the telemetry module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from flumodelingsuite.schema.dispatcher import CalibrationOutput, SimulationOutput
from flumodelingsuite.telemetry import (
    ExecutionTelemetry,
    create_workflow_telemetry,
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

    # No simulations for pure calibration
    mock_results.simulations = []

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
):
    """Create a mock CalibrationOutput with projection results for testing."""
    mock_results = MagicMock()

    if particles_accepted is not None:
        mock_results.accepted = [MagicMock() for _ in range(particles_accepted)]
    else:
        mock_results.accepted = None

    # Add simulations for projection
    mock_results.simulations = [MagicMock() for _ in range(successful_trajectories)]

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
