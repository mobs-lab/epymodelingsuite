"""CSV formatter for telemetry output."""

from typing import Any

from ...utils.formatting import format_data_size, format_duration
from .base import TelemetryFormatter


class CsvFormatter(TelemetryFormatter):
    """CSV export formatter with readable and raw modes."""

    def __init__(self, format: str = "readable"):
        """Initialize CSV formatter.

        Parameters
        ----------
        format : str
            Output format: 'readable' or 'raw'

        Raises
        ------
        ValueError
            If format is not 'readable' or 'raw'
        """
        if format not in ("readable", "raw"):
            msg = f"Invalid format: {format}. Must be 'readable' or 'raw'"
            raise ValueError(msg)
        self._format_mode = format

    def format(self, telemetry_data: dict[str, Any]) -> str:
        """Generate CSV summary.

        Parameters
        ----------
        telemetry_data : dict
            Telemetry data from ExecutionTelemetry.to_dict()

        Returns
        -------
        str
            CSV-formatted summary
        """
        df = self._build_dataframe(telemetry_data)
        return df.to_csv(index=False)

    def _build_dataframe(self, data: dict[str, Any]) -> Any:
        """Build pandas DataFrame for CSV export.

        Parameters
        ----------
        data : dict
            Telemetry data dictionary

        Returns
        -------
        pandas.DataFrame
            DataFrame with one row per model
        """
        import pandas as pd

        runner = data.get("runner", {})
        builder = data.get("builder", {})
        output = data.get("output", {})
        configuration = data.get("configuration", {})

        # If no runner data, return empty DataFrame with headers
        if not runner or not runner.get("models"):
            columns = [
                "primary_id",
                "population",
                "workflow_type",
                "start_date",
                "end_date",
                "delta_t",
                "fitting_window_start",
                "fitting_window_end",
                "random_seed",
                "simulation_duration",
                "calibration_duration",
                "calibration_strategy",
                "calibration_particles",
                "calibration_generations",
                "calibration_accepted",
                "projection_duration",
                "projection_success",
                "projection_failed",
                "builder_duration",
                "output_duration",
                "total_output_size",
                "total_duration",
                "error",
            ]
            return pd.DataFrame(columns=columns)

        # Extract data for each model
        rows = []
        models = runner["models"]
        n_models = len(models)

        # Get shared durations
        builder_duration = builder.get("duration_seconds", 0.0) / n_models if builder else 0.0
        output_duration = output.get("duration_seconds", 0.0) / n_models if output else 0.0
        total_output_size = output.get("total_size_bytes", 0) if output else 0

        for model in models:
            # Basic identifiers
            row = {
                "primary_id": model.get("primary_id", ""),
                "population": model.get("population", ""),
            }

            # Configuration
            row["workflow_type"] = configuration.get("workflow_type", "")
            row["start_date"] = configuration.get("start_date", "")
            row["end_date"] = configuration.get("end_date", "")
            row["delta_t"] = configuration.get("delta_t", "")

            # Fitting window (for calibration workflows)
            fitting_window = configuration.get("fitting_window")
            if fitting_window:
                row["fitting_window_start"] = fitting_window.get("start_date", "")
                row["fitting_window_end"] = fitting_window.get("end_date", "")
            else:
                row["fitting_window_start"] = ""
                row["fitting_window_end"] = ""

            row["random_seed"] = configuration.get("random_seed", "")

            # Simulation metrics
            simulation = model.get("simulation", {})
            if simulation:
                sim_duration = simulation.get("duration_seconds", 0.0)
                row["simulation_duration"] = (
                    format_duration(sim_duration) if self._format_mode == "readable" else sim_duration
                )
            else:
                row["simulation_duration"] = "" if self._format_mode == "readable" else 0.0

            # Calibration metrics
            calibration = model.get("calibration", {})
            if calibration:
                cal_duration = calibration.get("duration_seconds", 0.0)
                row["calibration_duration"] = (
                    format_duration(cal_duration) if self._format_mode == "readable" else cal_duration
                )
                row["calibration_strategy"] = calibration.get("strategy", "")
                row["calibration_particles"] = calibration.get("num_particles", "")
                row["calibration_generations"] = calibration.get("num_generations", "")
                row["calibration_accepted"] = calibration.get("particles_accepted", 0)
            else:
                row["calibration_duration"] = "" if self._format_mode == "readable" else 0.0
                row["calibration_strategy"] = ""
                row["calibration_particles"] = ""
                row["calibration_generations"] = ""
                row["calibration_accepted"] = 0

            # Projection metrics
            projection = model.get("projection", {})
            if projection:
                proj_duration = projection.get("duration_seconds", 0.0)
                row["projection_duration"] = (
                    format_duration(proj_duration) if self._format_mode == "readable" else proj_duration
                )
                row["projection_success"] = projection.get("successful_trajectories", 0)
                row["projection_failed"] = projection.get("failed_trajectories", 0)
            else:
                row["projection_duration"] = "" if self._format_mode == "readable" else 0.0
                row["projection_success"] = 0
                row["projection_failed"] = 0

            # Stage durations
            row["builder_duration"] = (
                format_duration(builder_duration) if self._format_mode == "readable" else builder_duration
            )
            row["output_duration"] = (
                format_duration(output_duration) if self._format_mode == "readable" else output_duration
            )

            # Total output size (shared across all models)
            if self._format_mode == "readable":
                row["total_output_size"] = format_data_size(total_output_size) if total_output_size > 0 else ""
            else:
                row["total_output_size"] = total_output_size

            # Total duration
            total_duration = builder_duration
            if simulation:
                total_duration += simulation.get("duration_seconds", 0.0)
            if calibration:
                total_duration += calibration.get("duration_seconds", 0.0)
            if projection:
                total_duration += projection.get("duration_seconds", 0.0)
            total_duration += output_duration
            row["total_duration"] = (
                format_duration(total_duration) if self._format_mode == "readable" else total_duration
            )

            # Error message
            row["error"] = model.get("error", "")

            rows.append(row)

        return pd.DataFrame(rows)
