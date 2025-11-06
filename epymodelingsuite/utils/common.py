"""Common utility functions."""

from datetime import timedelta

import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import Tick, Week


def parse_timedelta(text: str) -> timedelta:
    """
    Convert a pandas-like frequency string to a Python datetime.timedelta.

    This function accepts various time duration formats and converts them to
    a fixed-length timedelta object. It supports both pandas Timedelta-style
    strings (e.g., '30m', '1h30m') and pandas frequency aliases (e.g., 'W', '2H').

    For more details on supported formats, see:
    https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases

    Parameters
    ----------
    text : str
        A time duration string in one of the following formats:
        - Pandas Timedelta format: '30m', '2H', '1h30m', '45s', '3D'
        - Pandas frequency alias: 'W' (week), '2H' (2 hours), '15T' (15 minutes)

    Returns
    -------
    timedelta
        A Python datetime.timedelta object representing the duration.

    Raises
    ------
    ValueError
        If the string is not recognized as a valid duration format, or if it
        represents a variable-length duration (e.g., months, quarters, years,
        business days) that cannot be converted to a fixed timedelta.

    Examples
    --------
    >>> parse_timedelta('30m')
    datetime.timedelta(seconds=1800)

    >>> parse_timedelta('2H')
    datetime.timedelta(seconds=7200)

    >>> parse_timedelta('1h30m')
    datetime.timedelta(seconds=5400)

    >>> parse_timedelta('W')  # 1 week
    datetime.timedelta(days=7)

    >>> parse_timedelta('3D')
    datetime.timedelta(days=3)

    Variable-length durations are not supported:

    >>> parse_timedelta('M')  # Raises ValueError
    Traceback (most recent call last):
        ...
    ValueError: Frequency 'M' is not a fixed-length duration...

    Notes
    -----
    Fixed-length durations supported:
    - Seconds: 's', 'S'
    - Minutes: 'm', 'T' (or 'min')
    - Hours: 'h', 'H'
    - Days: 'd', 'D'
    - Weeks: 'W', 'W-SUN', 'W-MON', etc.

    Variable-length durations NOT supported:
    - Months: 'M', 'MS', 'ME', 'BM', 'BMS'
    - Quarters: 'Q', 'QS', 'BQ'
    - Years: 'A', 'AS', 'BA', 'Y', 'YS'
    - Business days: 'B', 'C'
    """
    s = text.strip()

    # 1) First try Timedelta-style strings (e.g., '30m', '1h30m', '2D', '45s')
    try:
        td = pd.to_timedelta(s)
        # pandas returns NaT for empty/invalid strings
        if pd.isna(td):
            msg = f"Unrecognized duration/frequency: {text!r}"
            raise ValueError(msg)
        # pandas.Timedelta -> python datetime.timedelta
        return td.to_pytimedelta()
    except (ValueError, TypeError):
        # Not a valid Timedelta string, try frequency alias next
        pass

    # 2) Next try frequency aliases (e.g., 'W', '2H', '30T', '3S')
    try:
        off = to_offset(s)
    except Exception as e:
        msg = f"Unrecognized duration/frequency: {text!r}"
        raise ValueError(msg) from e

    # Fixed-length tick-like offsets -> convertible
    if isinstance(off, Tick):
        # off.nanos is the exact length in nanoseconds
        return pd.Timedelta(off.nanos, unit="ns").to_pytimedelta()

    # Weeks are also fixed-length (7 days each)
    if isinstance(off, Week):
        return timedelta(weeks=off.n)

    # Otherwise it's calendar/anchored/variable; cannot be a pure timedelta
    msg = f"Frequency {text!r} is not a fixed-length duration and cannot be represented as a datetime.timedelta."
    raise ValueError(msg)


def to_set(values: object | None) -> set:
    """
    Normalize an optional iterable into a set.

    Parameters
    ----------
    values : Iterable or None
        Input iterable (or ``None``) to convert.

    Returns
    -------
    set
        Set containing the iterable values, or an empty set when ``None``.
    """
    from collections.abc import Iterable

    if values is None:
        return set()
    if isinstance(values, Iterable) and not isinstance(values, (str, bytes)):
        return set(values)
    return set()


def strip_agegroup_suffix(name: str, age_group: str = "total") -> str:
    """
    Strip age group suffix from compartment/transition names.

    Parameters
    ----------
    name : str
        Name potentially ending with age group suffix.
    age_group : str, default="total"
        Age group suffix to remove (without underscore prefix).

    Returns
    -------
    str
        Name with age group suffix removed.

    Examples
    --------
    >>> strip_agegroup_suffix("S_total")
    'S'
    >>> strip_agegroup_suffix("I_child", age_group="child")
    'I'
    """
    return name.removesuffix(f"_{age_group}")


def parse_transition_name(name: str, age_group: str = "total") -> tuple[str, str]:
    """
    Parse transition name from format: {source}_to_{target}_{age_group}.

    Parameters
    ----------
    name : str
        Transition name in format {source}_to_{target}_{age_group}.
    age_group : str, default="total"
        Age group suffix to remove (without underscore prefix).

    Returns
    -------
    tuple[str, str]
        Source and target compartment names.

    Raises
    ------
    ValueError
        If name does not match expected format.

    Examples
    --------
    >>> parse_transition_name("S_to_I_total")
    ('S', 'I')
    >>> parse_transition_name("S_to_I_child", age_group="child")
    ('S', 'I')
    """
    # Strip age group suffix first
    name_no_suffix = strip_agegroup_suffix(name, age_group=age_group)
    # Split on '_to_'
    parts = name_no_suffix.split("_to_")
    if len(parts) != 2:  # noqa: PLR2004
        msg = f"Invalid transition name format: {name}"
        raise ValueError(msg)
    return parts[0], parts[1]
