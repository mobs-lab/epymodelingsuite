"""Base class for telemetry output formatters."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class TelemetryFormatter(ABC):
    """Abstract base class for telemetry output formatters.

    Formatters convert telemetry data dictionaries into various output formats
    (text, JSON, CSV, etc.). Each formatter implements the format() method to
    generate string output from telemetry data.
    """

    @abstractmethod
    def format(self, telemetry_data: dict[str, Any]) -> str:
        """Format telemetry data to string output.

        Parameters
        ----------
        telemetry_data : dict
            Telemetry data from ExecutionTelemetry.to_dict()

        Returns
        -------
        str
            Formatted output string
        """

    def write(self, telemetry_data: dict[str, Any], path: str | Path) -> None:
        """Write formatted telemetry to file.

        Parameters
        ----------
        telemetry_data : dict
            Telemetry data from ExecutionTelemetry.to_dict()
        path : str | Path
            Output file path
        """
        output = self.format(telemetry_data)
        Path(path).write_text(output)
