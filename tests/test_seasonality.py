# test_seasonality.py
# Pytest unit tests for seasonality.py
import datetime as dt
import math
import numpy as np
import pytest

from flumodelingsuite.seasonality import (
    _calc_seasonality_balcan_at_t,
    calc_seasonality_balcan_at_date,
    generate_seasonal_values,
    get_seasonal_transmission_balcan,
)

def test__calc_seasonality_balcan_at_t_peak_and_min():
    """Check that the peak at t_max is 1.0 and the trough at half-period is R_min/R_max."""
    R_min, R_max = 0.6, 1.4
    period = 365
    t_max = 120

    # Peak exactly at t_max -> should be 1.0
    vals = [
        _calc_seasonality_balcan_at_t(t, t_max, R_min, R_max, period) for t in range(0, period+1)
    ]

    # Peak date should correspond to t_max
    t_max_sim = np.argmax(vals)
    assert t_max_sim == t_max, "Peak date does not match t_max"

    # Peak value should be 1.0 at t_max
    val_peak = _calc_seasonality_balcan_at_t(t_max, t_max, R_min, R_max, period)
    assert math.isclose(val_peak, 1.0, rel_tol=0, abs_tol=1e-8), "Peak value does not match 1.0"

    # Minimum date should be at half-period away from t_max
    t_min_sim = np.argmin(vals)
    expected_t_min = t_max + period // 2
    assert t_min_sim == expected_t_min, "Minimum date does not match expected"

    # Minimum value should be R_min/R_max at t_min
    val_min = min(vals)
    val_max = max(vals)
    expected_min = R_min / R_max
    assert math.isclose(val_min, expected_min, rel_tol=0, abs_tol=1e-4), "Minimum value does not match expected"

    # All values over a full period lie within [R_min/R_max, 1.0]
    assert val_min >= expected_min - 1e-12, "Minimum value is below expected range [R_min/R_max, 1.0]"
    assert val_max <= 1.0 + 1e-12, "Maximum value is above expected range [R_min/R_max, 1.0]"

@pytest.mark.parametrize(
    "R_min,R_max,period,t_offset",
    [
        (0.2, 1.0, 365, 0),
        (0.5, 1.5, 360, 17),
        (0.9, 1.1, 180, 73),
    ],
)
def test_calc_seasonality_balcan_at_date_matches__calc(R_min, R_max, period, t_offset):
    """calc_seasonality_balcan_at_date should match the private _calc_* when period is explicit."""
    date_start = dt.date(2020, 1, 1)
    date_tmax = date_start + dt.timedelta(days=50)
    date_t = date_start + dt.timedelta(days=50 + t_offset)

    v_date = calc_seasonality_balcan_at_date(
        date_t=date_t,
        date_start=date_start,
        date_tmax=date_tmax,
        R_min=R_min,
        R_max=R_max,
        period=period,
    )

    # Expected via the t-based function
    t_days = (date_t - date_start).days
    t_max_days = (date_tmax - date_start).days
    v_expected = _calc_seasonality_balcan_at_t(t_days, t_max_days, R_min, R_max, period)

    assert math.isclose(v_date, v_expected, rel_tol=0, abs_tol=1e-12), "calc_seasonality_balcan_at_date did not match expected"

def test_calc_seasonality_balcan_at_date_derives_period_from_tmin():
    """When date_tmin is given and period is None, period should be 2*|tmin - tmax|."""
    R_min, R_max = 0.6, 1.2
    date_start = dt.date(2023, 1, 1)
    date_tmax = dt.date(2023, 3, 1)
    date_tmin = dt.date(2023, 9, 1)  # roughly 184 days after tmax -> expected period ~ 368
    date_t = dt.date(2023, 6, 1)

    # Calculate seasonality without period (automatic derivation with 2*|t_min - t_max|)
    v = calc_seasonality_balcan_at_date(
        date_t=date_t,
        date_start=date_start,
        date_tmax=date_tmax,
        date_tmin=date_tmin,
        R_min=R_min,
        R_max=R_max,
        period=None,
    )

    # Manually compute expected with derived period
    t_days = (date_t - date_start).days
    t_max_days = (date_tmax - date_start).days
    t_min_days = (date_tmin - date_start).days
    derived_period = 2 * abs(t_min_days - t_max_days)

    v_expected = _calc_seasonality_balcan_at_t(t_days, t_max_days, R_min, R_max, derived_period)
    assert math.isclose(v, v_expected, rel_tol=0, abs_tol=1e-12), "calc_seasonality_balcan_at_date did not match expected"

def test_calc_seasonality_balcan_at_date_period_overrides_tmin():
    """If both period and date_tmin are provided, the explicit period must be used (override)."""
    R_min, R_max = 0.4, 1.6
    date_start = dt.date(2022, 1, 1)
    date_tmax = dt.date(2022, 2, 1)
    date_tmin = dt.date(2022, 8, 1)  # would imply its own period if used
    forced_period = 200
    date_t = dt.date(2022, 5, 1)

    v = calc_seasonality_balcan_at_date(
        date_t=date_t,
        date_start=date_start,
        date_tmax=date_tmax,
        date_tmin=date_tmin,
        R_min=R_min,
        R_max=R_max,
        period=forced_period,
    )

    t_days = (date_t - date_start).days
    t_max_days = (date_tmax - date_start).days
    v_expected = _calc_seasonality_balcan_at_t(t_days, t_max_days, R_min, R_max, forced_period)
    assert math.isclose(v, v_expected, rel_tol=0, abs_tol=1e-12), "calc_seasonality_balcan_at_date did not match expected"

def test_generate_seasonal_values_inclusive_range_and_calls_func():
    """generate_seasonal_values should produce inclusive dates and call the provided function per date."""
    date_start = dt.date(2021, 1, 1)
    date_stop = dt.date(2021, 1, 10)

    # A simple seasonality function that returns day index (for testability)
    def dummy(date: dt.date) -> float:
        return float((date - date_start).days)

    dates, values = generate_seasonal_values(date_start, date_stop, dummy)

    assert dates[0] == date_start
    assert dates[-1] == date_stop
    assert len(dates) == len(values) == (date_stop - date_start).days + 1
    assert values == [float(i) for i in range(len(values))], "Generated values do not match expected range"

def test_get_seasonal_transmission_balcan_end_to_end():
    """End-to-end test: ensure wrapper yields expected length and peak near date_tmax is the maximum."""
    date_start = dt.date(2020, 1, 1)
    date_stop = dt.date(2020, 12, 31)
    date_tmax = dt.date(2020, 6, 1)
    R_min, R_max = 0.7, 1.3

    dates, values = get_seasonal_transmission_balcan(
        date_start=date_start,
        date_stop=date_stop,
        date_tmax=date_tmax,
        R_min=R_min,
        R_max=R_max,
        date_tmin=None,
    )

    # Basic shape
    assert dates[0] == date_start and dates[-1] == date_stop, "Date range is not as expected"
    assert len(dates) == len(values) == (date_stop - date_start).days + 1, "Date and value arrays are not the same length"

    # Peak check: the value at tmax index should be (almost) the global maximum ~ 1.0
    idx_tmax = (date_tmax - date_start).days
    assert math.isclose(values[idx_tmax], 1.0, rel_tol=0, abs_tol=1e-12), "Value at tmax is not as expected"
    assert values[idx_tmax] == pytest.approx(max(values), abs=1e-12), "Value at tmax is not the maximum"

    # Trough check at half-period away (default period=365)
    idx_tmin = idx_tmax + 365 // 2
    if idx_tmin < len(values):
        expected_min = R_min / R_max
        assert values[idx_tmin] == pytest.approx(expected_min, abs=1e-4), "Value at tmin is not as expected"
