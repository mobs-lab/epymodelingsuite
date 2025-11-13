"""Text formatter for telemetry output."""

from datetime import datetime
from typing import Any

from ...utils.formatting import format_data_size, format_duration
from .base import TelemetryFormatter

# Text formatting constants
HEADER_WIDTH = 60
HEADER_SEPARATOR = "=" * HEADER_WIDTH
SECTION_SEPARATOR = "-" * HEADER_WIDTH


class TextFormatter(TelemetryFormatter):
    """Human-readable text summary formatter."""

    def format(self, telemetry_data: dict[str, Any]) -> str:
        """Generate human-readable text summary.

        Parameters
        ----------
        telemetry_data : dict
            Telemetry data from ExecutionTelemetry.to_dict()

        Returns
        -------
        str
            Formatted text summary
        """
        lines = []
        self._add_header(lines, telemetry_data)
        self._add_configuration(lines, telemetry_data)
        self._add_builder(lines, telemetry_data)
        self._add_runner(lines, telemetry_data)
        self._add_output(lines, telemetry_data)
        self._add_summary(lines, telemetry_data)
        self._add_warnings(lines, telemetry_data)
        return "\n".join(lines)

    def _add_header(self, lines: list[str], data: dict[str, Any]) -> None:
        """Add header section."""
        lines.append(HEADER_SEPARATOR)
        lines.append("Telemetry Summary")
        lines.append(HEADER_SEPARATOR)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Determine what to show: stage name or workflow type
        configuration = data.get("configuration", {})
        builder = data.get("builder", {})
        runner = data.get("runner", {})
        output = data.get("output", {})

        workflow_type = configuration.get("workflow_type")
        has_builder = bool(builder)
        has_runner = bool(runner and runner.get("models"))
        has_output = bool(output)
        num_stages = sum([has_builder, has_runner, has_output])

        if num_stages > 1 and workflow_type:
            # Multi-stage workflow - show workflow type
            lines.append(f"Workflow: {workflow_type.replace('_', ' ').title()}")
        elif has_builder:
            lines.append("Stage: Builder")
        elif has_runner:
            lines.append("Stage: Runner")
        elif has_output:
            lines.append("Stage: Output")

        lines.append("")

    def _add_configuration(self, lines: list[str], data: dict[str, Any]) -> None:
        """Add configuration section."""
        config = data.get("configuration", {})

        # Check if there's actual configuration data
        has_config = (
            "populations" in config or ("start_date" in config and "end_date" in config) or "random_seed" in config
        )

        if not has_config:
            return

        lines.append("CONFIGURATION")
        lines.append(SECTION_SEPARATOR)
        if "populations" in config:
            pop_list = ", ".join(config["populations"])
            lines.append(f"Populations: {config['n_populations']} ({pop_list})")
            # Show age groups inline with populations
            if "age_groups" in config:
                age_groups = config["age_groups"]
                age_list = ", ".join(age_groups)
                lines.append(f"  Age groups: {len(age_groups)} ({age_list})")
        if "start_date" in config and "end_date" in config:
            delta_t = config.get("delta_t", "unknown")
            lines.append(f"Timespan: {config['start_date']} to {config['end_date']} (dt={delta_t})")
        if "random_seed" in config:
            lines.append(f"Random seed: {config['random_seed']}")

        # Calibration subsection
        if "fitting_window" in config or "distance_function" in config:
            lines.append("")
            lines.append("Calibration:")
            if "fitting_window" in config:
                fw = config["fitting_window"]
                lines.append(f"  Fitting window: {fw['start_date']} to {fw['end_date']}")
            if "distance_function" in config:
                lines.append(f"  Distance function: {config['distance_function']}")

        lines.append("")

    def _add_builder(self, lines: list[str], data: dict[str, Any]) -> None:
        """Add builder section."""
        builder = data.get("builder", {})

        if not builder:
            return

        lines.append("BUILDER STAGE")
        lines.append(SECTION_SEPARATOR)
        if "duration_seconds" in builder:
            lines.append(f"Duration: {format_duration(builder['duration_seconds'])}")
        if "n_models" in builder:
            lines.append(f"Models created: {builder['n_models']}")
        if "peak_memory_mb" in builder:
            lines.append(f"Peak memory: {format_data_size(builder['peak_memory_mb'], 'MB')}")
        lines.append("")

    def _add_runner(self, lines: list[str], data: dict[str, Any]) -> None:
        """Add runner section."""
        runner = data.get("runner", {})

        if not runner or not runner.get("models"):
            return

        lines.append("RUNNER STAGE")
        lines.append(SECTION_SEPARATOR)
        if "duration_seconds" in runner:
            lines.append(f"Total duration: {format_duration(runner['duration_seconds'])}")
        lines.append("")

        for model in runner["models"]:
            pop = model["population"]
            pid = model["primary_id"]
            lines.append(f"{pop} (primary_id: {pid}):")

            if "simulation" in model:
                sim = model["simulation"]
                dur = format_duration(sim["duration_seconds"])
                lines.append(f"  Simulation: {dur}")

            if "calibration" in model:
                cal = model["calibration"]
                dur = format_duration(cal["duration_seconds"])
                lines.append(f"  Calibration: {dur}")

                if "strategy" in cal:
                    strategy = cal["strategy"]
                    particles = cal.get("num_particles", "?")
                    generations = cal.get("num_generations", "?")
                    lines.append(f"    Strategy: {strategy} ({particles} particles, {generations} generations)")

                if "particles_accepted" in cal:
                    lines.append(f"    Particles accepted: {cal['particles_accepted']}")

            if "projection" in model:
                proj = model["projection"]
                dur = format_duration(proj["duration_seconds"])
                lines.append(f"  Projection: {dur}")

                requested = proj.get("requested_trajectories", 0)
                successful = proj.get("successful_trajectories", 0)
                failed = proj.get("failed_trajectories", 0)

                if failed > 0:
                    lines.append(f"    Trajectories: {successful}/{requested} successful ({failed} failed)")
                else:
                    lines.append(f"    Trajectories: {successful}/{requested} successful")

            if "error" in model:
                lines.append(f"  ERROR: {model['error']}")

            lines.append("")

        if "peak_memory_mb" in runner:
            lines.append(f"Peak memory: {format_data_size(runner['peak_memory_mb'], 'MB')}")
        lines.append("")

    def _add_output(self, lines: list[str], data: dict[str, Any]) -> None:
        """Add output section."""
        output = data.get("output", {})

        if not output:
            return

        lines.append("OUTPUT STAGE")
        lines.append(SECTION_SEPARATOR)
        if "duration_seconds" in output:
            lines.append(f"Duration: {format_duration(output['duration_seconds'])}")
        if "files" in output:
            lines.append(f"Files generated: {len(output['files'])}")
            for file_info in output["files"]:
                name = file_info["name"]
                size = format_data_size(file_info["size_bytes"])
                lines.append(f"  - {name} ({size})")
            # Show total size if available
            if "total_size_bytes" in output:
                total_size = format_data_size(output["total_size_bytes"])
                lines.append(f"Total size: {total_size}")
        if output.get("filtered_projections", 0) > 0:
            lines.append(f"Filtered projections: {output['filtered_projections']}")
        if "peak_memory_mb" in output:
            lines.append(f"Peak memory: {format_data_size(output['peak_memory_mb'], 'MB')}")
        lines.append("")

    def _add_summary(self, lines: list[str], data: dict[str, Any]) -> None:
        """Add summary section."""
        metadata = data.get("metadata", {})
        resources = data.get("resources", {})

        lines.append("SUMMARY")
        lines.append(SECTION_SEPARATOR)
        if "total_duration_seconds" in metadata:
            lines.append(f"Total duration: {format_duration(metadata['total_duration_seconds'])}")
        if "peak_memory_mb" in resources:
            lines.append(f"Peak memory: {format_data_size(resources['peak_memory_mb'], 'MB')}")
        if "cpu_time_user_seconds" in resources:
            user_time = format_duration(resources["cpu_time_user_seconds"])
            system_time = format_duration(resources["cpu_time_system_seconds"])
            lines.append(f"CPU time: {user_time} (user), {system_time} (system)")
        lines.append("")

    def _add_warnings(self, lines: list[str], data: dict[str, Any]) -> None:
        """Add warnings section."""
        warnings = data.get("warnings", [])

        if not warnings:
            return

        lines.append(f"WARNINGS ({len(warnings)})")
        lines.append(SECTION_SEPARATOR)
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")
