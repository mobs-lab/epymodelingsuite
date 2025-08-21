### seasonality.py
# Functions for generating seasonal transmission rates.
import datetime as dt
from collections.abc import Callable


def _calc_seasonality_balcan_at_t(t: int, t_max: int, R_min: float, R_max: float, period: int = 365) -> float:
    """
    Calculate the seasonal transmission rate based on the time of year. Period can be customized.
    Implementation of eq25 from https://www.sciencedirect.com/science/article/pii/S1877750310000438 .

    Parameters
    ----------
        t (int): Day of year (0-365).
        t_max (int): Day of year when the transmission rate is at its maximum.
        R_min (float): The minimum transmission rate.
        R_max (float): The maximum transmission rate.
        period (int): The period of the seasonality in days (default=365).

    Returns
    -------
        float: The seasonal transmission rate at time t.
    """
    import numpy as np

    return ((1 - (R_min / R_max)) * np.sin((2 * np.pi / period) * (t - t_max) + (np.pi / 2)) + 1 + (R_min / R_max)) / 2


def calc_seasonality_balcan_at_date(
    date_t: dt.date,
    date_start: dt.date,
    date_tmax: dt.date,
    R_min: float,
    R_max: float,
    date_tmin: dt.date | None = None,
    period: int | None = None,
) -> float:
    """
    Compute the seasonal rate for a given date using the Balcan model.

    Parameters
    ----------
        date_t: Target date.
        date_start: Reference start date where t=0.
        date_tmax: Date when the transmission rate is at its maximum.
        date_tmin: Date when the transmission rate is at its minimum (optional).
        R_min: The minimum transmission rate.
        R_max: The maximum transmission rate.
        period: The period of the seasonality in days (default=365). If None, derives using date_tmin and date_start or defaults to 365.

    Returns
    -------
        Seasonality factor at date_t.
    """
    t_days = (date_t - date_start).days
    t_max_days = (date_tmax - date_start).days

    if period is not None:
        p = period
    elif date_tmin is not None:
        t_min_days = (date_tmin - date_start).days
        p = 2 * abs(t_min_days - t_max_days)
    else:
        p = 365

    return _calc_seasonality_balcan_at_t(t_days, t_max_days, R_min, R_max, p)


def generate_seasonal_values(
    date_start: dt.date, date_stop: dt.date, seasonality_func: Callable[[dt.date], float]
) -> tuple[list[dt.date], list[float]]:
    """
    Generate values over a date range using any seasonality function.

    Parameters
    ----------
        date_start: Start date.
        date_stop: End date.
        seasonality_func: A function that computes the seasonal rate for a given date (e.g. calc_seasonality_balcan_at_date).

    Returns
    -------
        Tuple of (dates, values).
    """
    import datetime as dt

    dates = [date_start + dt.timedelta(days=i) for i in range((date_stop - date_start).days + 1)]
    values = [seasonality_func(d) for d in dates]
    return dates, values


def get_seasonal_transmission_balcan(
    date_start: dt.date,
    date_stop: dt.date,
    date_tmax: dt.date,
    R_min: float,
    R_max: float,
    date_tmin: dt.date | None = None,
) -> tuple[list[dt.date], list[float]]:
    """
    Return seasonal transmission rates for the specified simulation period using the Balcan model.
    This is a wrapper for calc_seasonality_balcan_at_date() and generate_seasonal_values().

    Parameters
    ----------
        date_start: Reference start date where t=0 (start of simulation period).
        date_stop: End date.
        date_tmax: Date when the transmission rate is at its maximum.
        date_tmin: Date when the transmission rate is at its minimum (optional).
        R_min: The minimum transmission rate.
        R_max: The maximum transmission rate.

    Returns
    -------
        Tuple of (dates, values).
    """
    from functools import partial

    balcan_calculator = partial(
        calc_seasonality_balcan_at_date,
        date_start=date_start,
        date_tmax=date_tmax,
        date_tmin=date_tmin,
        R_min=R_min,
        R_max=R_max,
    )

    dates, values = generate_seasonal_values(
        date_start=date_start, date_stop=date_stop, seasonality_func=balcan_calculator
    )

    return dates, values
