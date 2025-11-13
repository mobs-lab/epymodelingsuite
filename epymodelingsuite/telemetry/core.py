"""Execution telemetry tracking and reporting for epymodelingsuite workflows.

This module provides tools to track execution metrics throughout the builder,
runner, and output stages of a workflow. It generates both human-readable text
summaries and structured JSON data for automated analysis.
"""

import json
import os
import sys
import warnings
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..utils.formatting import format_data_size, format_duration
from .extractors import extract_builder_metadata, extract_calibration_info, extract_projection_info
from .formatters import CsvFormatter, JsonFormatter, TextFormatter
from .resource_tracker import ResourceTracker

if TYPE_CHECKING:
    from ..schema.calibration import CalibrationStrategy
    from ..schema.dispatcher import BuilderOutput, CalibrationOutput, SimulationOutput


def _get_package_version() -> str:
    """Get epymodelingsuite version."""
    try:
        import epymodelingsuite

        return getattr(epymodelingsuite, "__version__", "unknown")
    except (ImportError, AttributeError):
        return "unknown"


class ExecutionTelemetry:
    """Track execution metrics and telemetry for dispatcher workflows.

    This class captures operational metrics throughout the builder, runner,
    and output stages of a workflow execution. It tracks timing, resource
    usage, and execution metadata for observability and debugging.

    Supports optional context-based access via ContextVar for cleaner function
    signatures without explicit parameter threading.

    Attributes
    ----------
    metadata : dict
        Process and environment metadata
    configuration : dict
        Workflow configuration details
    builder : dict
        Builder stage metrics
    runner : dict
        Runner stage metrics (including per-model details)
    output : dict
        Output stage metrics
    resources : dict
        Overall resource usage metrics
    status : str
        Workflow status ("running", "completed", "failed")
    warnings : list
        List of warning messages
    """

    # ContextVar for optional thread-safe/async-safe context access
    _current: ContextVar["ExecutionTelemetry | None"] = ContextVar("execution_telemetry", default=None)

    @classmethod
    def get_current(cls) -> "ExecutionTelemetry | None":
        """Get the current ExecutionTelemetry from context, if any.

        Returns
        -------
        ExecutionTelemetry | None
            The currently active ExecutionTelemetry, or None if no telemetry is set
        """
        return cls._current.get()

    @classmethod
    def set_current(cls, telemetry: "ExecutionTelemetry | None") -> None:
        """Set the current ExecutionTelemetry in context.

        Parameters
        ----------
        telemetry : ExecutionTelemetry | None
            The telemetry to set as current (or None to clear)
        """
        cls._current.set(telemetry)

    def __enter__(self) -> "ExecutionTelemetry":
        """Enter context manager - set this telemetry as current.

        Returns
        -------
        ExecutionTelemetry
            This telemetry instance
        """
        self.set_current(self)
        return self

    def __exit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any) -> None:
        """Exit context manager - clear current telemetry.

        Parameters
        ----------
        exc_type : type | None
            Exception type if an exception occurred
        exc_val : Exception | None
            Exception value if an exception occurred
        exc_tb : Any
            Exception traceback if an exception occurred
        """
        self.set_current(None)

    def __init__(self) -> None:
        """Initialize a new ExecutionTelemetry."""
        self.metadata: dict[str, Any] = {
            "process_id": os.getpid(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "epymodelingsuite_version": _get_package_version(),
        }
        self.configuration: dict[str, Any] = {}
        self.builder: dict[str, Any] = {}
        self.runner: dict[str, Any] = {"models": [], "errors": []}
        self.output: dict[str, Any] = {}
        self.resources: dict[str, Any] = {}
        self.status = "running"
        self.warnings: list[str] = []

        # Initialize resource tracking
        self._resource_tracker = ResourceTracker()

    def _get_current_memory_mb(self) -> float:
        """Get current process memory usage in MB."""
        return self._resource_tracker.get_current_memory_mb()

    def _update_peak_memory(self) -> None:
        """Update peak memory if current usage is higher."""
        self._resource_tracker.update_peak_memory()

    def _record_stage_end(self, stage_dict: dict[str, Any]) -> None:
        """Record end time and calculate duration for a stage.

        Parameters
        ----------
        stage_dict : dict
            Stage dictionary to update with end_time and duration_seconds
        """
        end_time = datetime.now()
        stage_dict["end_time"] = end_time.isoformat()
        start_time = datetime.fromisoformat(stage_dict["start_time"])
        stage_dict["duration_seconds"] = (end_time - start_time).total_seconds()

    def enter_builder(self, workflow_type: str) -> None:
        """Enter the builder stage.

        Parameters
        ----------
        workflow_type : str
            Type of workflow (simulation/sampling/calibration/calibration_projection)
        """
        self.builder["start_time"] = datetime.now().isoformat()
        self.configuration["workflow_type"] = workflow_type
        self._update_peak_memory()

    def exit_builder(
        self,
        n_models: int,
        populations: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        delta_t: float | None = None,
        random_seed: int | None = None,
        age_groups: list[str] | None = None,
        fitting_window: tuple[str, str] | None = None,
        distance_function: str | None = None,
    ) -> None:
        """Exit the builder stage and record metrics.

        Parameters
        ----------
        n_models : int
            Number of models/calibrators created
        populations : list[str]
            List of population names
        start_date : str | None, optional
            Simulation start date
        end_date : str | None, optional
            Simulation end date
        delta_t : float | None, optional
            Simulation time step
        random_seed : int | None, optional
            Random seed used
        age_groups : list[str] | None, optional
            List of age groups in population
        fitting_window : tuple[str, str] | None, optional
            Fitting window for calibration (start_date, end_date)
        distance_function : str | None, optional
            Distance function for calibration
        """
        self._record_stage_end(self.builder)
        self.builder["n_models"] = n_models

        self._update_peak_memory()
        self.builder["peak_memory_mb"] = self._resource_tracker.get_peak_memory_mb()

        # Update configuration
        self.configuration["populations"] = populations
        self.configuration["n_populations"] = len(populations)
        if start_date:
            self.configuration["start_date"] = start_date
        if end_date:
            self.configuration["end_date"] = end_date
        if delta_t is not None:
            self.configuration["delta_t"] = delta_t
        if random_seed is not None:
            self.configuration["random_seed"] = random_seed
        if age_groups is not None:
            self.configuration["age_groups"] = age_groups
        if fitting_window is not None:
            self.configuration["fitting_window"] = {
                "start_date": fitting_window[0],
                "end_date": fitting_window[1],
            }
        if distance_function is not None:
            self.configuration["distance_function"] = distance_function

        # Finalize telemetry for builder stage
        self.status = "completed"
        self._finalize_resources()
        self._calculate_total_duration()

    def enter_runner(self) -> None:
        """Enter the runner stage."""
        self.runner["start_time"] = datetime.now().isoformat()
        self._update_peak_memory()

    def _extract_calibration_info(self, results: Any) -> dict[str, Any]:
        """Extract calibration metrics from epydemix CalibrationResults.

        Parameters
        ----------
        results : Any
            Calibration results object (epydemix.calibration.CalibrationResults)

        Returns
        -------
        dict
            Calibration info dict with particles_accepted, etc.
        """
        return extract_calibration_info(results)

    def _extract_projection_info(self, results: Any, n_trajectories: int) -> dict[str, Any]:
        """Extract projection metrics from epydemix CalibrationResults.

        Parameters
        ----------
        results : Any
            Projection results object (epydemix.calibration.CalibrationResults)
        n_trajectories : int
            Requested number of trajectories

        Returns
        -------
        dict
            Projection info dict with requested/successful/failed trajectories
        """
        return extract_projection_info(results, n_trajectories)

    def capture_simulation(
        self,
        output: "SimulationOutput",
        duration: float,
        error: str | None = None,
    ) -> None:
        """Capture metrics from a simulation.

        Parameters
        ----------
        output : SimulationOutput
            Simulation output object from runner
        duration : float
            Simulation execution time in seconds
        error : str | None, optional
            Error message if simulation failed
        """
        model_data: dict[str, Any] = {
            "primary_id": output.primary_id,
            "population": output.population,
        }

        # Record simulation info
        model_data["simulation"] = {
            "duration_seconds": duration,
            "n_sims": output.results.Nsim if hasattr(output.results, "Nsim") else None,
        }

        if error:
            model_data["error"] = error
            self.runner["errors"].append(
                {
                    "primary_id": output.primary_id,
                    "population": output.population,
                    "error": error,
                }
            )

        self.runner["models"].append(model_data)
        self._update_peak_memory()

    def capture_calibration(
        self,
        output: "CalibrationOutput",
        duration: float,
        error: str | None = None,
        builder_output: "BuilderOutput | None" = None,
    ) -> None:
        """Capture metrics from a calibration.

        Parameters
        ----------
        output : CalibrationOutput
            Calibration output object from runner
        duration : float
            Calibration execution time in seconds
        error : str | None, optional
            Error message if calibration failed
        builder_output : BuilderOutput | None, optional
            Builder output containing calibration strategy and other metadata
        """
        model_data: dict[str, Any] = {
            "primary_id": output.primary_id,
            "population": output.population,
        }

        # Record calibration info
        calibration_info = self._extract_calibration_info(output.results)
        model_data["calibration"] = {
            "duration_seconds": duration,
            **calibration_info,
        }

        # Extract info from builder_output if provided
        if builder_output and builder_output.calibration:
            calibration_strategy = builder_output.calibration
            model_data["calibration"]["strategy"] = str(calibration_strategy.name)
            # Extract key metrics from options
            if "num_particles" in calibration_strategy.options:
                model_data["calibration"]["num_particles"] = calibration_strategy.options["num_particles"]
            if "num_generations" in calibration_strategy.options:
                model_data["calibration"]["num_generations"] = calibration_strategy.options["num_generations"]
            if "distance_function" in calibration_strategy.options:
                model_data["calibration"]["distance_function"] = calibration_strategy.options["distance_function"]

        if error:
            model_data["error"] = error
            self.runner["errors"].append(
                {
                    "primary_id": output.primary_id,
                    "population": output.population,
                    "error": error,
                }
            )

        self.runner["models"].append(model_data)
        self._update_peak_memory()

    def capture_projection(
        self,
        output: "CalibrationOutput",
        calib_duration: float,
        proj_duration: float,
        n_trajectories: int,
        error: str | None = None,
        builder_output: "BuilderOutput | None" = None,
    ) -> None:
        """Capture metrics from a calibration with projection.

        Parameters
        ----------
        output : CalibrationOutput
            Calibration output object from runner (contains projection results)
        calib_duration : float
            Calibration execution time in seconds
        proj_duration : float
            Projection execution time in seconds
        n_trajectories : int
            Number of requested projection trajectories
        error : str | None, optional
            Error message if calibration or projection failed
        builder_output : BuilderOutput | None, optional
            Builder output containing calibration strategy and other metadata
        """
        model_data: dict[str, Any] = {
            "primary_id": output.primary_id,
            "population": output.population,
        }

        # Record calibration info
        calibration_info = self._extract_calibration_info(output.results)
        model_data["calibration"] = {
            "duration_seconds": calib_duration,
            **calibration_info,
        }

        # Extract info from builder_output if provided
        if builder_output and builder_output.calibration:
            calibration_strategy = builder_output.calibration
            model_data["calibration"]["strategy"] = str(calibration_strategy.name)
            # Extract key metrics from options
            if "num_particles" in calibration_strategy.options:
                model_data["calibration"]["num_particles"] = calibration_strategy.options["num_particles"]
            if "num_generations" in calibration_strategy.options:
                model_data["calibration"]["num_generations"] = calibration_strategy.options["num_generations"]
            if "distance_function" in calibration_strategy.options:
                model_data["calibration"]["distance_function"] = calibration_strategy.options["distance_function"]

        # Record projection info
        projection_info = self._extract_projection_info(output.results, n_trajectories)
        model_data["projection"] = {
            "duration_seconds": proj_duration,
            **projection_info,
        }

        if error:
            model_data["error"] = error
            self.runner["errors"].append(
                {
                    "primary_id": output.primary_id,
                    "population": output.population,
                    "error": error,
                }
            )

        self.runner["models"].append(model_data)
        self._update_peak_memory()

    def exit_runner(self) -> None:
        """Exit the runner stage and record metrics."""
        self._record_stage_end(self.runner)

        self._update_peak_memory()
        peak_memory = self._resource_tracker.get_peak_memory_mb()
        if peak_memory > self.builder.get("peak_memory_mb", 0):
            self.runner["peak_memory_mb"] = peak_memory

        # Finalize telemetry for runner stage
        self.status = "completed"
        self._finalize_resources()
        self._calculate_total_duration()

    def enter_output(self) -> None:
        """Enter the output stage."""
        self.output["start_time"] = datetime.now().isoformat()
        self.output["files"] = []
        self._update_peak_memory()

    def capture_file(self, filename: str, size_bytes: int) -> None:
        """Capture an output file.

        Parameters
        ----------
        filename : str
            Name of the output file
        size_bytes : int
            Size of the file in bytes
        """
        self.output["files"].append({"name": filename, "size_bytes": size_bytes})

    def exit_output(self) -> None:
        """Exit the output stage and finalize the telemetry."""
        self._record_stage_end(self.output)

        # Calculate total file size
        if self.output.get("files"):
            total_size = sum(f["size_bytes"] for f in self.output["files"])
            self.output["total_size_bytes"] = total_size

        self._update_peak_memory()
        peak_memory = self._resource_tracker.get_peak_memory_mb()
        if peak_memory > self.runner.get("peak_memory_mb", 0):
            self.output["peak_memory_mb"] = peak_memory

        # Finalize telemetry
        self.status = "completed"
        self._finalize_resources()
        self._calculate_total_duration()

    def _finalize_resources(self) -> None:
        """Calculate final resource usage metrics."""
        self.resources = self._resource_tracker.finalize()

    def _calculate_total_duration(self) -> None:
        """Calculate total workflow duration from earliest start to latest end.

        This handles partial workflows where not all stages (builder/runner/output) ran.
        """
        start_times = []
        end_times = []

        # Collect all start times
        if "start_time" in self.builder:
            start_times.append(datetime.fromisoformat(self.builder["start_time"]))
        if "start_time" in self.runner:
            start_times.append(datetime.fromisoformat(self.runner["start_time"]))
        if "start_time" in self.output:
            start_times.append(datetime.fromisoformat(self.output["start_time"]))

        # Collect all end times
        if "end_time" in self.builder:
            end_times.append(datetime.fromisoformat(self.builder["end_time"]))
        if "end_time" in self.runner:
            end_times.append(datetime.fromisoformat(self.runner["end_time"]))
        if "end_time" in self.output:
            end_times.append(datetime.fromisoformat(self.output["end_time"]))

        # Calculate total duration if we have any times
        if start_times and end_times:
            self.metadata["total_duration_seconds"] = (max(end_times) - min(start_times)).total_seconds()

    def record_warning(self, message: str) -> None:
        """Add a warning message to the summary.

        Parameters
        ----------
        message : str
            Warning message
        """
        self.warnings.append(message)

    def to_dict(self) -> dict[str, Any]:
        """Export telemetry as a dictionary.

        Returns
        -------
        dict
            Complete telemetry data
        """
        data = {
            "metadata": self.metadata,
            "configuration": self.configuration,
            "builder": self.builder,
            "runner": self.runner,
            "output": self.output,
            "resources": self.resources,
            "status": self.status,
            "warnings": self.warnings,
        }

        # Add total duration if available
        if "total_duration_seconds" in self.metadata:
            data["total_duration_seconds"] = self.metadata["total_duration_seconds"]

        return data

    def to_text(self, path: str | Path | None = None) -> str | None:
        """Generate human-readable text summary.

        Parameters
        ----------
        path : str | Path | None, optional
            If provided, write summary to this file path and return None.
            If None (default), return the summary as a string.

        Returns
        -------
        str | None
            Formatted text summary if path is None, otherwise None
        """
        formatter = TextFormatter()
        data = self.to_dict()

        if path is not None:
            formatter.write(data, path)
            return None
        return formatter.format(data)

    def to_json(self, path: str | Path | None = None) -> str | None:
        """Generate structured JSON summary.

        Parameters
        ----------
        path : str | Path | None, optional
            If provided, write summary to this file path and return None.
            If None (default), return the summary as a string.

        Returns
        -------
        str | None
            JSON-formatted summary if path is None, otherwise None
        """
        formatter = JsonFormatter()
        data = self.to_dict()

        if path is not None:
            formatter.write(data, path)
            return None
        return formatter.format(data)

    def to_csv(self, path: str | Path | None = None, format: str = "readable") -> str | None:
        """Generate CSV summary with one row per model.

        Parameters
        ----------
        path : str | Path | None, optional
            If provided, write CSV to this file path and return None.
            If None (default), return the CSV as a string.
        format : str, optional
            Output format: 'readable' for human-friendly formatting (default),
            or 'raw' for machine-readable numbers.

        Returns
        -------
        str | None
            CSV string if path is None, otherwise None

        Raises
        ------
        ValueError
            If format is not 'readable' or 'raw'

        Notes
        -----
        The total_output_size column reports the same total size for all models
        (shared across models), as per-model output size tracking is not currently
        implemented.
        """
        formatter = CsvFormatter(format=format)
        data = self.to_dict()

        if path is not None:
            formatter.write(data, path)
            return None
        return formatter.format(data)

    def __str__(self) -> str:
        """Return full text summary when printing.

        Returns
        -------
        str
            Complete formatted text summary
        """
        return self.to_text()

    def __repr__(self) -> str:
        """Return developer-friendly representation.

        Returns
        -------
        str
            Compact representation for debugging
        """
        # Determine stage or workflow name
        workflow_type = self.configuration.get("workflow_type")
        has_builder = bool(self.builder)
        has_runner = bool(self.runner and self.runner.get("models"))
        has_output = bool(self.output)
        num_stages = sum([has_builder, has_runner, has_output])

        if num_stages > 1 and workflow_type:
            stage_info = f"workflow='{workflow_type}'"
        elif has_builder:
            stage_info = "stage='builder'"
        elif has_runner:
            stage_info = "stage='runner'"
        elif has_output:
            stage_info = "stage='output'"
        else:
            stage_info = "stage='unknown'"

        duration = self.metadata.get("total_duration_seconds")
        if duration is not None:
            duration_str = format_duration(duration)
            return f"ExecutionTelemetry({stage_info}, duration='{duration_str}')"
        return f"ExecutionTelemetry({stage_info})"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionTelemetry":
        """Reconstruct an ExecutionTelemetry from a dictionary.

        This is the inverse of to_dict(). Internal attributes (_process,
        _baseline_memory, _peak_memory_mb) are not restored as they are
        not needed for loaded telemetry.

        Parameters
        ----------
        data : dict
            Dictionary containing telemetry data (from to_dict())

        Returns
        -------
        ExecutionTelemetry
            Reconstructed telemetry object
        """
        telemetry = cls.__new__(cls)  # Create instance without calling __init__

        # Restore all public attributes
        telemetry.metadata = data.get("metadata", {})
        telemetry.configuration = data.get("configuration", {})
        telemetry.builder = data.get("builder", {})
        telemetry.runner = data.get("runner", {"models": [], "errors": []})
        telemetry.output = data.get("output", {})
        telemetry.resources = data.get("resources", {})
        telemetry.status = data.get("status", "unknown")
        telemetry.warnings = data.get("warnings", [])

        # Set internal attributes to None/defaults (not needed for loaded telemetry)
        telemetry._process = None
        telemetry._baseline_memory = 0.0
        telemetry._peak_memory_mb = 0.0

        return telemetry

    @classmethod
    def load_from_json(cls, json_path: str | Path) -> "ExecutionTelemetry":
        """Load an ExecutionTelemetry from a JSON file.

        Parameters
        ----------
        json_path : str | Path
            Path to the JSON file containing telemetry data

        Returns
        -------
        ExecutionTelemetry
            Reconstructed telemetry object

        Raises
        ------
        FileNotFoundError
            If the JSON file does not exist
        json.JSONDecodeError
            If the file contains invalid JSON
        """
        json_path = Path(json_path)

        if not json_path.exists():
            msg = f"Telemetry file not found: {json_path}"
            raise FileNotFoundError(msg)

        try:
            data = json.loads(json_path.read_text())
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON in telemetry file {json_path}: {e}"
            raise json.JSONDecodeError(msg, e.doc, e.pos) from e

        return cls.from_dict(data)


def create_workflow_telemetry(
    builder_telemetry: ExecutionTelemetry | None,
    runner_telemetries: list[ExecutionTelemetry] | None,
    output_telemetry: ExecutionTelemetry | None,
) -> ExecutionTelemetry:
    """Create a unified workflow telemetry by merging telemetry from all stages.

    This function aggregates telemetry from builder, runner, and output stages
    into a single unified workflow telemetry. Runner telemetries from parallel
    tasks are combined before merging with other stages.

    Parameters
    ----------
    builder_telemetry : ExecutionTelemetry | None
        Telemetry from the builder stage (may be None if builder failed)
    runner_telemetries : list[ExecutionTelemetry] | None
        List of telemetry from runner stage (one per parallel task, may be None/empty)
    output_telemetry : ExecutionTelemetry | None
        Telemetry from the output stage (may be None if output failed)

    Returns
    -------
    ExecutionTelemetry
        Combined workflow telemetry containing data from all stages

    Examples
    --------
    >>> workflow = create_workflow_telemetry(builder, runners, output)
    >>> workflow.to_text("workflow_telemetry.txt")
    >>> workflow.to_json("workflow_telemetry.json")
    >>> print(workflow)  # Display full telemetry
    """
    # Aggregate runner telemetries if any
    if runner_telemetries:
        aggregated_runner = _aggregate_runner_telemetries(runner_telemetries)
    else:
        aggregated_runner = None

    # Merge all stage telemetries
    return _merge_stage_telemetries(builder_telemetry, aggregated_runner, output_telemetry)


def _aggregate_runner_telemetries(runner_telemetries: list[ExecutionTelemetry]) -> ExecutionTelemetry:
    """Aggregate multiple runner telemetries from parallel tasks.

    Parameters
    ----------
    runner_telemetries : list[ExecutionTelemetry]
        List of runner telemetries from parallel tasks

    Returns
    -------
    ExecutionTelemetry
        Aggregated runner telemetry
    """
    if not runner_telemetries:
        return ExecutionTelemetry()

    # Use first telemetry as base
    aggregated = ExecutionTelemetry()

    # Combine runner data from all tasks
    aggregated.runner = {
        "models": [],
        "errors": [],
    }

    # Find earliest start and latest end time
    start_times = [
        datetime.fromisoformat(t.runner["start_time"]) for t in runner_telemetries if "start_time" in t.runner
    ]
    end_times = [datetime.fromisoformat(t.runner["end_time"]) for t in runner_telemetries if "end_time" in t.runner]

    if start_times:
        aggregated.runner["start_time"] = min(start_times).isoformat()
    if end_times:
        aggregated.runner["end_time"] = max(end_times).isoformat()

    # Calculate total duration (latest end - earliest start)
    if start_times and end_times:
        aggregated.runner["duration_seconds"] = (max(end_times) - min(start_times)).total_seconds()

    # Combine all models and errors
    for telemetry in runner_telemetries:
        aggregated.runner["models"].extend(telemetry.runner.get("models", []))
        aggregated.runner["errors"].extend(telemetry.runner.get("errors", []))

    # Track peak memory across all tasks
    peak_memories = [t.runner.get("peak_memory_mb", 0) for t in runner_telemetries]
    if peak_memories:
        aggregated.runner["peak_memory_mb"] = max(peak_memories)

    # Sum CPU times across all parallel runner tasks
    cpu_time_user = sum(t.resources.get("cpu_time_user_seconds", 0.0) for t in runner_telemetries)
    cpu_time_system = sum(t.resources.get("cpu_time_system_seconds", 0.0) for t in runner_telemetries)

    if cpu_time_user > 0:
        aggregated.resources["cpu_time_user_seconds"] = cpu_time_user
    if cpu_time_system > 0:
        aggregated.resources["cpu_time_system_seconds"] = cpu_time_system

    return aggregated


def _merge_stage_telemetries(
    builder_telemetry: ExecutionTelemetry | None,
    runner_telemetry: ExecutionTelemetry | None,
    output_telemetry: ExecutionTelemetry | None,
) -> ExecutionTelemetry:
    """Merge telemetries from all stages into a unified workflow telemetry.

    Parameters
    ----------
    builder_telemetry : ExecutionTelemetry | None
        Telemetry from builder stage
    runner_telemetry : ExecutionTelemetry | None
        Telemetry from runner stage (possibly aggregated)
    output_telemetry : ExecutionTelemetry | None
        Telemetry from output stage

    Returns
    -------
    ExecutionTelemetry
        Complete workflow telemetry
    """
    workflow = ExecutionTelemetry()

    # Copy data from each stage
    if builder_telemetry:
        workflow.metadata.update(builder_telemetry.metadata)
        workflow.configuration = builder_telemetry.configuration.copy()
        workflow.builder = builder_telemetry.builder.copy()

    if runner_telemetry:
        workflow.runner = runner_telemetry.runner.copy()

    if output_telemetry:
        workflow.output = output_telemetry.output.copy()
        # Combine warnings
        for warning in output_telemetry.warnings:
            if warning not in workflow.warnings:
                workflow.warnings.append(warning)

    # Calculate total workflow metrics
    if builder_telemetry and output_telemetry:
        if "start_time" in builder_telemetry.builder and "end_time" in output_telemetry.output:
            start_time = datetime.fromisoformat(builder_telemetry.builder["start_time"])
            end_time = datetime.fromisoformat(output_telemetry.output["end_time"])
            workflow.metadata["total_duration_seconds"] = (end_time - start_time).total_seconds()

    # Track overall peak memory (max across all stages)
    peak_memories = []
    if builder_telemetry and "peak_memory_mb" in builder_telemetry.builder:
        peak_memories.append(builder_telemetry.builder["peak_memory_mb"])
    if runner_telemetry and "peak_memory_mb" in runner_telemetry.runner:
        peak_memories.append(runner_telemetry.runner["peak_memory_mb"])
    if output_telemetry and "peak_memory_mb" in output_telemetry.output:
        peak_memories.append(output_telemetry.output["peak_memory_mb"])

    if peak_memories:
        workflow.resources["peak_memory_mb"] = max(peak_memories)

    # Sum CPU time across all stages (each stage is a separate process)
    cpu_time_user = 0.0
    cpu_time_system = 0.0

    for telemetry in [builder_telemetry, runner_telemetry, output_telemetry]:
        if telemetry and telemetry.resources:
            cpu_time_user += telemetry.resources.get("cpu_time_user_seconds", 0.0)
            cpu_time_system += telemetry.resources.get("cpu_time_system_seconds", 0.0)

    if cpu_time_user > 0:
        workflow.resources["cpu_time_user_seconds"] = cpu_time_user
    if cpu_time_system > 0:
        workflow.resources["cpu_time_system_seconds"] = cpu_time_system

    # Set status
    workflow.status = "completed"

    return workflow
