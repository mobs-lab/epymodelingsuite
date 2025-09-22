### seasonality.py
# Functions for generating seasonal transmission rates.
import datetime as dt
from collections.abc import Callable


def _calc_seasonality_balcan_at_t(
    t: float, t_max: float, val_min: float, val_max: float, period: float = 365.0
) -> float:
    """
    Calculate the seasonal transmission factor based on the time unit. This is a scaling factor that ranges between 0 and 1. To obtain the actual transmission rate at time t, this factor must be multiplied by the baseline parameter value.
    Implementation of eq25 from https://www.sciencedirect.com/science/article/pii/S1877750310000438 .

    Parameters
    ----------
        t (float): Time in units defined by delta_t (e.g., days if delta_t=1, hours if delta_t=1/24).
        t_max (float): Time unit when the transmission rate is at its maximum.
        val_min (float): The minimum value that the parameter can take after scaling.
        val_max (float): The maximum value that the parameter can take after scaling.
        period (float): The period of the seasonality in time units (default=365 for daily units).

    Returns
    -------
        float: The seasonal transmission factor at time t.
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
    Compute the seasonal rate for a given date using the Balcan model.

    Parameters
    ----------
        date_t: Target date/datetime.
        date_start: Reference start date/datetime where t=0.
        date_tmax: Date/datetime when the transmission rate is at its maximum.
        date_tmin: Date/datetime when the transmission rate is at its minimum (optional).
        val_min: The minimum value that the parameter can take after scaling.
        val_max: The maximum value that the parameter can take after scaling.
        period: The period of the seasonality in days (default=365). If None, derives using date_tmin and date_start or defaults to 365.
        delta_t: Time step in days (default=1.0). For example, 0.25 for 6-hour intervals, 1/24 for hourly.

    Returns
    -------
        float: Seasonality factor at date_t, a scaling factor between 0 and 1.
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
    scaling_factor: float
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
    if scaling_start.date() <= date_t.date() <= scaling_stop.date():
        return scaling_factor
    else:
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
    delta_t: float = 1.0
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
        calc_scaling_at_date,
        scaling_start=scaling_start,
        scaling_stop=scaling_stop,
        scaling_factor=scaling_factor
    )

    dates, values = generate_seasonal_values(
        date_start=date_start,
        date_stop=date_stop,
        seasonality_func=scaling_calculator,
        delta_t=delta_t
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
    Return seasonal transmission rates for the specified simulation period using the Balcan model.
    This is a wrapper for calc_seasonality_balcan_at_date() and generate_seasonal_values().

    Parameters
    ----------
        date_start: Reference start date/datetime where t=0 (start of simulation period).
        date_stop: End date/datetime.
        date_tmax: Date/datetime when the transmission rate is at its maximum.
        date_tmin: Date/datetime when the transmission rate is at its minimum (optional).
        val_min: The minimum transmission rate.
        val_max: The maximum transmission rate.
        delta_t : float, default 1.0
            Time step in days for calculating seasonality. Default 1.0 means daily.
            Examples: 0.25 for 6-hour intervals, 1/24 for hourly, 7 for weekly.

    Returns
    -------
        Tuple of (dates/datetimes, values).
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
