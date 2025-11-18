"""JSON formatter for telemetry output."""

import json
from typing import Any

from .base import TelemetryFormatter


class JsonFormatter(TelemetryFormatter):
    """Structured JSON output formatter."""

    def format(self, telemetry_data: dict[str, Any]) -> str:
        """Generate structured JSON summary.

        Parameters
        ----------
        telemetry_data : dict
            Telemetry data from ExecutionTelemetry.to_dict()

        Returns
        -------
        str
            JSON-formatted summary
        """
        return json.dumps(telemetry_data, indent=2)
