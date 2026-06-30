### seasonality.py
# Functions for generating seasonal transmission rates.
import datetime as dt
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClimateSeries:
    """Daily temperature and relative humidity indexed by calendar date."""

    temp_by_date: dict[dt.date, float]
    rh_by_date: dict[dt.date, float]
    min_temp: float
    date_min: dt.date
    date_max: dt.date


def _calc_seasonality_balcan_at_t(
    t: float, t_max: float, val_min: float, val_max: float, period: float = 365.0
) -> float:
    """
    Calculate the seasonal transmission factor based on the time unit. This is a scaling factor that ranges between 0 and 1. To obtain the actual transmission rate at time t, this factor must be multiplied by the baseline parameter value.
    Implementation of eq25 from https://www.sciencedirect.com/science/article/pii/S1877750310000438 .

    Parameters
    ----------
    t : float
        Time in units defined by delta_t (e.g., days if delta_t=1, hours if delta_t=1/24).
    t_max : float
        Time unit when the scaling factor is at its maximum (1.0).
    val_min : float
        Together with val_max, determines the trough as val_min/val_max.
        When val_max=1.0, this directly equals the trough factor (e.g., 0.2 = 20% of peak).
    val_max : float
        Together with val_min, determines the trough as val_min/val_max.
        Typically set to 1.0. The output always peaks at 1.0 regardless of this value.
    period : float, default 365.0
        The period of the seasonality in time units (default=365 for daily units).

    Returns
    -------
    float
        Seasonal scaling factor at time t, ranging from val_min/val_max (at trough) to 1.0 (at peak).
    """
    import numpy as np

    return (
        (1 - (val_min / val_max)) * np.sin((2 * np.pi / period) * (t - t_max) + (np.pi / 2)) + 1 + (val_min / val_max)
    ) / 2


def calc_seasonality_balcan_at_date(
    date_t: dt.date | dt.datetime,
    date_start: dt.date | dt.datetime,
    date_tmax: dt.date | dt.datetime,
    val_min: float,
    val_max: float,
    date_tmin: dt.date | dt.datetime | None = None,
    period: float | None = None,
    delta_t: float = 1.0,
) -> float:
    """
    Compute the seasonal scaling factor for a given date using the Balcan model.

    The scaling factor ranges from (val_min/val_max) at trough to 1.0 at peak.
    When val_max=1.0, this simplifies to ranging from val_min to 1.0.

    Parameters
    ----------
    date_t : date or datetime
        Target date/datetime.
    date_start : date or datetime
        Reference start date/datetime where t=0.
    date_tmax : date or datetime
        Date/datetime when the scaling factor is at its maximum (1.0).
    val_min : float
        Together with val_max, determines the trough as val_min/val_max.
        When val_max=1.0, this directly equals the trough factor (e.g., 0.2 = 20% of peak).
    val_max : float
        Together with val_min, determines the trough as val_min/val_max.
        Typically set to 1.0. The output always peaks at 1.0 regardless of this value.
    date_tmin : date or datetime, optional
        Date/datetime of seasonal trough. Used to derive the period as
        2 * |date_tmin - date_tmax| when period is not specified.
    period : float, optional
        The period of the seasonality in days. If None, derives from date_tmin or defaults to 365.
    delta_t : float, default 1.0
        Time step in days.

    Returns
    -------
    float
        Seasonal scaling factor at date_t, ranging from val_min/val_max (at trough) to 1.0 (at peak).
    """
    # Convert dates to datetime if needed for consistent calculation
    if isinstance(date_t, dt.date) and not isinstance(date_t, dt.datetime):
        date_t = dt.datetime.combine(date_t, dt.time())
    if isinstance(date_start, dt.date) and not isinstance(date_start, dt.datetime):
        date_start = dt.datetime.combine(date_start, dt.time())
    if isinstance(date_tmax, dt.date) and not isinstance(date_tmax, dt.datetime):
        date_tmax = dt.datetime.combine(date_tmax, dt.time())
    if date_tmin is not None and isinstance(date_tmin, dt.date) and not isinstance(date_tmin, dt.datetime):
        date_tmin = dt.datetime.combine(date_tmin, dt.time())

    # Calculate time differences in days
    t_days = (date_t - date_start).total_seconds() / 86400
    t_max_days = (date_tmax - date_start).total_seconds() / 86400

    # Convert to time units based on delta_t
    t_units = t_days / delta_t
    t_max_units = t_max_days / delta_t

    # Determine period in days
    if period is not None:
        period_days = period
    elif date_tmin is not None:
        t_min_days = (date_tmin - date_start).total_seconds() / 86400
        period_days = 2 * abs(t_min_days - t_max_days)
    else:
        period_days = 365

    # Convert period to time units
    period_units = period_days / delta_t

    return _calc_seasonality_balcan_at_t(t_units, t_max_units, val_min, val_max, period_units)


def calc_scaling_at_date(
    date_t: dt.date | dt.datetime,
    scaling_start: dt.date | dt.datetime,
    scaling_stop: dt.date | dt.datetime,
    scaling_factor: float,
) -> float:
    """
    Return the scaling factor if target date is within intervention period, otherwise 1.0. Used for parameter intervention.

    Parameters
    ----------
        date_t: Target date/datetime.
        scaling_start: Start date/datetime of intervention period.
        scaling_stop: Stop date/datetime of intervention period.
        scaling_factor: Scaling factor for parameter intervention.

    Returns
    -------
        float: Scaling factor at date_t.
    """
    # Convert dates to datetime if needed for consistent calculation
    if isinstance(date_t, dt.date) and not isinstance(date_t, dt.datetime):
        date_t = dt.datetime.combine(date_t, dt.time())
    if isinstance(scaling_start, dt.date) and not isinstance(scaling_start, dt.datetime):
        scaling_start = dt.datetime.combine(scaling_start, dt.time())
    if isinstance(scaling_stop, dt.date) and not isinstance(scaling_stop, dt.datetime):
        scaling_stop = dt.datetime.combine(scaling_stop, dt.time())

    # Return the scaling factor
    if scaling_start.date() <= date_t.date() <= scaling_stop.date():
        return scaling_factor
    return 1.0


def generate_seasonal_values(
    date_start: dt.date | dt.datetime,
    date_stop: dt.date | dt.datetime,
    seasonality_func: Callable[[dt.date | dt.datetime], float],
    delta_t: float = 1.0,
) -> tuple[list[dt.date | dt.datetime], list[float]]:
    """
    Generate values over a date range using any seasonality function.

    Parameters
    ----------
        date_start: Start date/datetime.
        date_stop: End date/datetime.
        seasonality_func: A function that computes the seasonal rate for a given date/datetime.
        delta_t: Time step in days (default=1.0). For example, 0.25 for 6-hour intervals, 1/24 for hourly.

    Returns
    -------
        Tuple of (dates/datetimes, values).
    """
    import datetime as dt

    # Convert to datetime for consistent calculation
    if isinstance(date_start, dt.date) and not isinstance(date_start, dt.datetime):
        date_start = dt.datetime.combine(date_start, dt.time())
    if isinstance(date_stop, dt.date) and not isinstance(date_stop, dt.datetime):
        date_stop = dt.datetime.combine(date_stop, dt.time())

    # Calculate number of steps
    total_days = (date_stop - date_start).total_seconds() / 86400
    n_steps = int(total_days / delta_t) + 1

    # Generate dates/datetimes at delta_t intervals
    dates = []
    for i in range(n_steps):
        current_datetime = date_start + dt.timedelta(days=i * delta_t)
        # If delta_t is 1 or greater and we started with dates, keep as date
        if delta_t >= 1.0 and i * delta_t == int(i * delta_t):
            dates.append(current_datetime.date())
        else:
            dates.append(current_datetime)

    # Ensure the last date is included if it's not already
    # Convert both to comparable types for comparison
    last_date = dates[-1]
    if isinstance(last_date, dt.date) and not isinstance(last_date, dt.datetime):
        last_date = dt.datetime.combine(last_date, dt.time())
    if isinstance(date_stop, dt.date) and not isinstance(date_stop, dt.datetime):
        date_stop_compare = dt.datetime.combine(date_stop, dt.time())
    else:
        date_stop_compare = date_stop

    if last_date < date_stop_compare:
        if delta_t >= 1.0:
            dates.append(date_stop.date() if isinstance(date_stop, dt.datetime) else date_stop)
        else:
            dates.append(date_stop)

    # Calculate values for each date
    values = [seasonality_func(d) for d in dates]

    return dates, values


def get_scaled_parameter(
    date_start: dt.date | dt.datetime,
    date_stop: dt.date | dt.datetime,
    scaling_start: dt.date | dt.datetime,
    scaling_stop: dt.date | dt.datetime,
    scaling_factor: float,
    delta_t: float = 1.0,
) -> tuple[list[dt.date | dt.datetime], list[float]]:
    """
    Return scaled parameter values for the specified simulation and intervention periods.
    This is a wrapper for calc_scaling_at_date and generate_seasonal_values().

    Parameters
    ----------
        date_start: Reference start date/datetime where t=0 (start of simulation period).
        date_stop: End date/datetime.
        scaling_start: Start date/datetime of intervention period.
        scaling_stop: Stop date/datetime of intervention period.
        scaling_factor: Scaling factor for parameter intervention.
        delta_t : float, default 1.0
            Time step in days for calculating seasonality. Default 1.0 means daily.
            Examples: 0.25 for 6-hour intervals, 1/24 for hourly, 7 for weekly.

    Returns
    -------
        Tuple of (dates/datetimes, values).
    """
    from functools import partial

    scaling_calculator = partial(
        calc_scaling_at_date, scaling_start=scaling_start, scaling_stop=scaling_stop, scaling_factor=scaling_factor
    )

    dates, values = generate_seasonal_values(
        date_start=date_start, date_stop=date_stop, seasonality_func=scaling_calculator, delta_t=delta_t
    )

    return dates, values


def get_seasonal_transmission_balcan(
    date_start: dt.date | dt.datetime,
    date_stop: dt.date | dt.datetime,
    date_tmax: dt.date | dt.datetime,
    val_min: float,
    val_max: float,
    date_tmin: dt.date | dt.datetime | None = None,
    delta_t: float = 1.0,
) -> tuple[list[dt.date | dt.datetime], list[float]]:
    """
    Return seasonal scaling factors for the specified simulation period using the Balcan model.

    The scaling factors range from (val_min/val_max) at trough to 1.0 at peak.
    When val_max=1.0, this simplifies to ranging from val_min to 1.0.
    This is a wrapper for calc_seasonality_balcan_at_date() and generate_seasonal_values().

    Parameters
    ----------
    date_start : date or datetime
        Reference start date/datetime where t=0 (start of simulation period).
    date_stop : date or datetime
        End date/datetime.
    date_tmax : date or datetime
        Date/datetime when the scaling factor is at its maximum (1.0).
    val_min : float
        Together with val_max, determines the trough as val_min/val_max.
        When val_max=1.0, this directly equals the trough factor (e.g., 0.2 = 20% of peak).
    val_max : float
        Together with val_min, determines the trough as val_min/val_max.
        Typically set to 1.0. The output always peaks at 1.0 regardless of this value.
    date_tmin : date or datetime, optional
        Date/datetime of seasonal trough. Used to derive the period as
        2 * |date_tmin - date_tmax| when period is not specified.
    delta_t : float, default 1.0
        Time step in days.

    Returns
    -------
    tuple[list, list]
        Tuple of (dates/datetimes, scaling_factors) where scaling_factors range
        from val_min/val_max (at trough) to 1.0 (at peak).
    """
    from functools import partial

    balcan_calculator = partial(
        calc_seasonality_balcan_at_date,
        date_start=date_start,
        date_tmax=date_tmax,
        date_tmin=date_tmin,
        val_min=val_min,
        val_max=val_max,
        delta_t=delta_t,
    )

    dates, values = generate_seasonal_values(
        date_start=date_start,
        date_stop=date_stop,
        seasonality_func=balcan_calculator,
        delta_t=delta_t,
    )

    return dates, values


def _to_calendar_date(date_t: dt.date | dt.datetime) -> dt.date:
    if isinstance(date_t, dt.datetime):
        return date_t.date()
    return date_t


def load_climate_series(
    climate_data_path: str | Path,
    date_column: str = "date",
    temp_column: str = "temp",
    rh_column: str = "humid_mean",
    location: str | None = None,
    location_column: str = "Location",
) -> ClimateSeries:
    """
    Load daily climate observations from CSV.

    Parameters
    ----------
    climate_data_path : str or Path
        Path to daily climate CSV (one row per calendar day).
    date_column : str, default "date"
        Column name for observation date.
    temp_column : str, default "temp"
        Column name for temperature in degrees Celsius.
    rh_column : str, default "humid_mean"
        Column name for relative humidity in percent.
    location : str, optional
        If set, filter rows where ``location_column`` equals this value.
    location_column : str, default "Location"
        Column used for optional location filtering.

    Returns
    -------
    ClimateSeries
        Daily temp/RH indexed by date with global min temperature over the series.
    """
    path = Path(climate_data_path)
    if not path.exists():
        raise FileNotFoundError(f"Climate data file not found: {path}")

    df = pd.read_csv(path)
    required = {date_column, temp_column, rh_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Climate CSV missing required columns: {sorted(missing)}")

    if location is not None:
        if location_column not in df.columns:
            raise ValueError(
                f"location={location!r} requested but column {location_column!r} not in climate CSV"
            )
        df = df[df[location_column] == location]
        if df.empty:
            raise ValueError(f"No climate rows found for location={location!r}")

    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column]).dt.date
    df = df.sort_values(date_column).drop_duplicates(subset=[date_column], keep="last")

    if len(df) > 1:
        gaps = pd.Series(list(df[date_column])).diff().dropna()
        max_gap_days = max(g.days for g in gaps)
        if max_gap_days > 1:
            logger.warning(
                "Climate data in %s has gaps up to %d days; missing dates will raise at lookup time",
                path,
                max_gap_days,
            )

    temp_by_date = dict(zip(df[date_column], df[temp_column].astype(float), strict=True))
    rh_by_date = dict(zip(df[date_column], df[rh_column].astype(float), strict=True))
    min_temp = float(df[temp_column].min())
    dates = sorted(temp_by_date.keys())

    return ClimateSeries(
        temp_by_date=temp_by_date,
        rh_by_date=rh_by_date,
        min_temp=min_temp,
        date_min=dates[0],
        date_max=dates[-1],
    )


def _calc_seasonality_climate(
    temp: float,
    rh: float,
    min_temp: float,
    b1: float,
    b3: float,
    rh_optimum: float = 40.0,
) -> float:
    """Raw climate signal (un-normalised): b1*(RH - rh_optimum)^2 + b3*(min_T - T)."""
    return b1 * (rh - rh_optimum) ** 2 + b3 * (min_temp - temp)


def calc_seasonality_climate_at_date(
    date_t: dt.date | dt.datetime,
    climate: ClimateSeries,
    b1: float,
    b3: float,
    s_min: float,
    raw_min: float,
    raw_max: float,
    rh_optimum: float = 40.0,
) -> float:
    """
    Compute the normalised climate scaling factor for a simulation date.

    The multiplier is always in [s_min, 1]:
        multiplier = s_min + (1 - s_min) * (raw - raw_min) / (raw_max - raw_min)

    Parameters
    ----------
    date_t : date or datetime
        Target simulation date/datetime.
    climate : ClimateSeries
        Loaded daily climate observations.
    b1, b3 : float
        Humidity curvature and temperature coefficients.
    s_min : float
        Minimum seasonality multiplier (in [0, 1)).
    raw_min, raw_max : float
        Pre-computed minimum and maximum of the raw signal over the full simulation
        period; used to normalise the series to [0, 1] before rescaling to [s_min, 1].
    rh_optimum : float, default 40.0
        RH (%) at which the parabolic humidity term is minimized.

    Returns
    -------
    float
        Seasonal scaling factor at ``date_t``, guaranteed to be in [s_min, 1].

    Raises
    ------
    ValueError
        If the date is outside the climate file range.
    """
    cal_date = _to_calendar_date(date_t)
    if cal_date < climate.date_min or cal_date > climate.date_max:
        raise ValueError(
            f"Simulation date {cal_date} is outside climate data range "
            f"[{climate.date_min}, {climate.date_max}]. Extend the climate CSV."
        )
    if cal_date not in climate.temp_by_date:
        raise ValueError(
            f"No climate observation for date {cal_date}. "
            f"Climate file covers [{climate.date_min}, {climate.date_max}] but has a gap on this date."
        )

    temp = climate.temp_by_date[cal_date]
    rh = climate.rh_by_date[cal_date]
    raw = _calc_seasonality_climate(temp, rh, climate.min_temp, b1, b3, rh_optimum)
    if raw_max == raw_min:
        return 1.0
    return s_min + (1.0 - s_min) * (raw - raw_min) / (raw_max - raw_min)


def get_seasonal_transmission_climate(
    date_start: dt.date | dt.datetime,
    date_stop: dt.date | dt.datetime,
    climate: ClimateSeries,
    b1: float,
    b3: float,
    s_min: float,
    rh_optimum: float = 40.0,
    delta_t: float = 1.0,
) -> tuple[list[dt.date | dt.datetime], list[float]]:
    """
    Return normalised climate scaling factors for the simulation period.

    The multiplier series is rescaled so the maximum equals 1 and the minimum
    equals ``s_min``:

        raw(t)       = b1*(RH(t) - rh_optimum)^2 + b3*(min_T - T(t))
        multiplier(t) = s_min + (1 - s_min) * (raw(t) - raw_min) / (raw_max - raw_min)

    Parameters
    ----------
    date_start, date_stop : date or datetime
        Simulation period (inclusive).
    climate : ClimateSeries
        Loaded daily climate observations.
    b1, b3 : float
        Humidity curvature and temperature coefficients.
    s_min : float
        Minimum seasonality multiplier (in [0, 1)).
    rh_optimum : float, default 40.0
        RH (%) at which the parabolic humidity term is minimized.
    delta_t : float, default 1.0
        Timestep in days.

    Returns
    -------
    tuple[list, list]
        (dates, multipliers) where every multiplier is in [s_min, 1].
    """
    import numpy as np

    # Use a dummy function to obtain the date grid from generate_seasonal_values.
    dates, _ = generate_seasonal_values(
        date_start=date_start,
        date_stop=date_stop,
        seasonality_func=lambda _: 0.0,
        delta_t=delta_t,
    )

    # First pass: compute raw values for every date.
    raw = []
    for d in dates:
        cal_date = _to_calendar_date(d)
        if cal_date < climate.date_min or cal_date > climate.date_max:
            raise ValueError(
                f"Simulation date {cal_date} is outside climate data range "
                f"[{climate.date_min}, {climate.date_max}]. Extend the climate CSV."
            )
        if cal_date not in climate.temp_by_date:
            raise ValueError(
                f"No climate observation for date {cal_date}. "
                f"Climate file covers [{climate.date_min}, {climate.date_max}] but has a gap on this date."
            )
        temp = climate.temp_by_date[cal_date]
        rh = climate.rh_by_date[cal_date]
        raw.append(_calc_seasonality_climate(temp, rh, climate.min_temp, b1, b3, rh_optimum))

    raw_arr = np.array(raw, dtype=float)
    raw_min = float(raw_arr.min())
    raw_max = float(raw_arr.max())

    if raw_max == raw_min:
        st = [1.0] * len(raw)
    else:
        st = (s_min + (1.0 - s_min) * (raw_arr - raw_min) / (raw_max - raw_min)).tolist()

    return dates, st
