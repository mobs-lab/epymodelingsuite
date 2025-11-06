"""Formatting utilities for human-readable output.

This module provides utilities for formatting various types of data
(durations, memory sizes, file sizes) into human-readable strings.
"""


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string.

    Parameters
    ----------
    seconds : float
        Duration in seconds

    Returns
    -------
    str
        Formatted duration string

    Examples
    --------
    >>> format_duration(3.5)
    '3.5s'
    >>> format_duration(125)
    '2m 5s'
    >>> format_duration(3665)
    '1h 1m 5s'
    """
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)

    if minutes < 60:
        return f"{minutes}m {remaining_seconds}s"

    hours = minutes // 60
    remaining_minutes = minutes % 60
    return f"{hours}h {remaining_minutes}m {remaining_seconds}s"


def format_data_size(
    value: float,
    input_unit: str = "B",
    output_unit: str | None = None,
    precision: int = 1,
) -> str:
    """Format data size with automatic or specified unit conversion.

    Parameters
    ----------
    value : float
        The size value in the specified input unit
    input_unit : str, default="B"
        The unit of the input value. One of: "B", "KB", "MB", "GB", "TB"
    output_unit : str | None, default=None
        The desired output unit. If None, automatically selects the most
        appropriate unit. One of: "B", "KB", "MB", "GB", "TB"
    precision : int, default=1
        Number of decimal places in the output

    Returns
    -------
    str
        Formatted data size string with unit label

    Examples
    --------
    >>> format_data_size(1536, "B")
    '1.5 KB'
    >>> format_data_size(2048, "MB")
    '2.0 GB'
    >>> format_data_size(1024, "MB", output_unit="GB")
    '1.0 GB'
    >>> format_data_size(512, "B")
    '512 B'
    """
    # Define unit conversions (relative to bytes)
    units = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
    }

    # Validate input unit
    if input_unit not in units:
        msg = f"Invalid input_unit: {input_unit}. Must be one of {list(units.keys())}"
        raise ValueError(msg)

    # Convert input to bytes
    bytes_value = value * units[input_unit]

    # If output_unit is specified, convert directly to that unit
    if output_unit is not None:
        if output_unit not in units:
            msg = f"Invalid output_unit: {output_unit}. Must be one of {list(units.keys())}"
            raise ValueError(msg)

        converted_value = bytes_value / units[output_unit]

        # Format bytes as integer, others with precision
        if output_unit == "B":
            return f"{int(converted_value)} B"
        return f"{converted_value:.{precision}f} {output_unit}"

    # Auto-select appropriate unit
    # Find the largest unit where the value is >= 1
    for unit in ["TB", "GB", "MB", "KB"]:
        if bytes_value >= units[unit]:
            converted_value = bytes_value / units[unit]
            return f"{converted_value:.{precision}f} {unit}"

    # Value is less than 1 KB, display in bytes
    return f"{int(bytes_value)} B"
