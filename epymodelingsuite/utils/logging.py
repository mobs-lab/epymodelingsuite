"""
Logging configuration utilities.

This module provides dual-mode logging infrastructure:
- Local mode: Human-readable console logs for development
- Cloud mode: JSON structured logs for Google Cloud Logging

Following Python library best practices:
- Uses NullHandler by default (libraries should not force logging configuration)
- Provides convenience function for users to enable logging
- Supports structured logging via the 'extra' parameter
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone


def setup_logger() -> None:
    """
    Initialize the epymodelingsuite logger with a NullHandler.

    This follows Python logging best practices for libraries:
    libraries should add a NullHandler to prevent "No handlers found" warnings
    while allowing users to configure logging as they wish.

    This function is called automatically when epymodelingsuite is imported.
    """
    logger = logging.getLogger("epymodelingsuite")
    logger.addHandler(logging.NullHandler())


def configure_logging(
    mode: str | None = None,
    level: int | str | None = None,
    format_string: str | None = None,
) -> None:
    """
    Configure logging for epymodelingsuite.

    Parameters
    ----------
    mode : str | None, optional
        Logging mode: "local" for human-readable output, "cloud" for JSON structured
        logs (Google Cloud Logging compatible). If None, falls back to EXECUTION_MODE
        environment variable, then defaults to "local".
    level : int | str | None, optional
        Logging level. Can be int (e.g., logging.INFO) or string (e.g., "INFO").
        If None, uses LOG_LEVEL environment variable (default: INFO).
    format_string : str | None, optional
        Custom format string for local mode. If None, uses default format.
        Ignored in cloud mode.

    Examples
    --------
    >>> # Local development (human-readable logs)
    >>> from epymodelingsuite import configure_logging
    >>> configure_logging()

    >>> # Cloud deployment (JSON structured logs)
    >>> configure_logging(mode="cloud")

    >>> # Debug logging
    >>> configure_logging(level="DEBUG")

    >>> # Using environment variable (for deployment configs)
    >>> # Set EXECUTION_MODE=cloud in environment, then:
    >>> configure_logging()  # Will use cloud mode
    """
    logger = logging.getLogger("epymodelingsuite")

    # Resolve log level
    if level is None:
        level_str = os.getenv("LOG_LEVEL", "INFO").upper()
        resolved_level: int = getattr(logging, level_str, logging.INFO)
    elif isinstance(level, str):
        resolved_level = getattr(logging, level.upper(), logging.INFO)
    else:
        resolved_level = level

    # Remove existing handlers to avoid duplicates
    logger.handlers = []

    # Resolve mode: explicit parameter > env var > default
    if mode is None:
        mode = os.getenv("EXECUTION_MODE", "local")
    mode = mode.lower()

    # Create handler (stdout for Cloud Logging compatibility)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(resolved_level)

    # Set formatter based on mode
    if mode == "cloud":
        formatter = StructuredFormatter()
    else:
        if format_string is None:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(resolved_level)


class StructuredFormatter(logging.Formatter):
    """
    JSON formatter for structured logging compatible with Google Cloud Logging.

    This formatter converts log records to JSON format for machine-readable logs,
    making them easy to parse by log aggregation tools. All fields passed via
    the 'extra' parameter become JSON fields that can be queried and filtered.

    Output Format
    -------------
    {
        "timestamp": "2024-01-08T10:30:45.123456+00:00",
        "severity": "INFO",
        "logger": "epymodelingsuite.dispatcher.runner",
        "message": "Starting simulation",
        "population": "US-MA",
        "stage": "runner"
    }

    Notes
    -----
    - Uses ISO 8601 timestamps
    - Uses 'severity' field for GCP compatibility (instead of 'level')
    - All extra fields from logger.info(..., extra={}) are included

    Examples
    --------
    >>> import logging
    >>> from epymodelingsuite.utils.logging import StructuredFormatter
    >>> logger = logging.getLogger("epymodelingsuite.test")
    >>> handler = logging.StreamHandler()
    >>> handler.setFormatter(StructuredFormatter())
    >>> logger.addHandler(handler)
    >>> logger.setLevel(logging.INFO)
    >>> logger.info("Processing population", extra={"population": "US-MA", "n_models": 10})
    """

    # Standard LogRecord fields that should not be included in extra
    RESERVED_FIELDS = frozenset(
        {
            "name",
            "msg",
            "args",
            "created",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "message",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "thread",
            "threadName",
            "exc_info",
            "exc_text",
            "stack_info",
            "taskName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record as a JSON string.

        Parameters
        ----------
        record : logging.LogRecord
            The log record to format.

        Returns
        -------
        str
            JSON-formatted log message.
        """
        # Build the base log entry
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add location information for non-INFO levels (useful for debugging)
        if record.levelno != logging.INFO:
            log_data["location"] = f"{record.filename}:{record.lineno}"

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add any extra fields passed via the 'extra' parameter
        extra_fields = {
            key: value
            for key, value in record.__dict__.items()
            if key not in self.RESERVED_FIELDS and not key.startswith("_")
        }
        log_data.update(extra_fields)

        return json.dumps(log_data)
