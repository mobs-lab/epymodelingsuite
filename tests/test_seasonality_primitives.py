# Unit tests for seasonality.py primitives
import datetime as dt

import numpy as np
import pytest

from epymodelingsuite.seasonality import (
    _calc_seasonality_balcan_at_t,
    calc_seasonality_balcan_at_date,
    generate_seasonal_values,
    get_scaled_parameter,
    get_seasonal_transmission_balcan,
)


def test__calc_seasonality_balcan_at_t_peak_and_min():
    """Check that the peak at t_max is 1.0 and the trough at half-period is val_min/val_max."""
    val_min, val_max = 0.6, 1.4
    period = 365
    t_max = 120

    # Peak exactly at t_max -> should be 1.0
    vals = [_calc_seasonality_balcan_at_t(t, t_max, val_min, val_max, period) for t in range(period + 1)]

    # Peak date should correspond to t_max
    t_max_sim = np.argmax(vals)
    assert t_max_sim == t_max, "Peak date does not match t_max"

    # Peak value should be 1.0 at t_max
    val_peak = _calc_seasonality_balcan_at_t(t_max, t_max, val_min, val_max, period)
    assert np.isclose(val_peak, 1.0, rtol=0, atol=1e-8), "Peak value does not match 1.0"

    # Minimum date should be at half-period away from t_max
    t_min_sim = np.argmin(vals)
    expected_t_min = t_max + period // 2
    assert t_min_sim == expected_t_min, "Minimum date does not match expected"

    # Minimum value should be val_min/val_max at t_min
    output_min = min(vals)
    output_max = max(vals)
    expected_min = val_min / val_max
    assert np.isclose(output_min, expected_min, rtol=0, atol=1e-4), "Minimum value does not match expected"

    # All values over a full period lie within [val_min/val_max, 1.0]
    assert output_min >= expected_min - 1e-12, "Minimum value is below expected range [val_min/val_max, 1.0]"
    assert output_max <= 1.0 + 1e-12, "Maximum value is above expected range [val_min/val_max, 1.0]"


@pytest.mark.parametrize(
    "val_min,val_max,period,t_offset",
    [
        (0.2, 1.0, 365, 0),
        (0.5, 1.5, 360, 17),
        (0.9, 1.1, 180, 73),
    ],
)
def test_calc_seasonality_balcan_at_date_matches__calc(val_min, val_max, period, t_offset):
    """calc_seasonality_balcan_at_date should match the private _calc_* when period is explicit."""
    date_start = dt.date(2020, 1, 1)
    date_tmax = date_start + dt.timedelta(days=50)
    date_t = date_start + dt.timedelta(days=50 + t_offset)

    v_date = calc_seasonality_balcan_at_date(
        date_t=date_t,
        date_start=date_start,
        date_tmax=date_tmax,
        val_min=val_min,
        val_max=val_max,
        period=period,
    )

    # Expected via the t-based function
    t_days = (date_t - date_start).days
    t_max_days = (date_tmax - date_start).days
    v_expected = _calc_seasonality_balcan_at_t(t_days, t_max_days, val_min, val_max, period)

    assert np.isclose(v_date, v_expected, rtol=0, atol=1e-12), "calc_seasonality_balcan_at_date did not match expected"


def test_calc_seasonality_balcan_at_date_derives_period_from_tmin():
    """When date_tmin is given and period is None, period should be 2*|tmin - tmax|."""
    val_min, val_max = 0.6, 1.2
    date_start = dt.date(2025, 10, 1)
    date_tmax = dt.date(2025, 12, 31)  # Peak in late December
    date_tmin = dt.date(2026, 6, 15)  # Trough in mid-June -> period ~ 332 days
    date_t = dt.date(2026, 3, 1)

    # Calculate seasonality without period (automatic derivation with 2*|t_min - t_max|)
    v = calc_seasonality_balcan_at_date(
        date_t=date_t,
        date_start=date_start,
        date_tmax=date_tmax,
        date_tmin=date_tmin,
        val_min=val_min,
        val_max=val_max,
        period=None,
    )

    # Manually compute expected with derived period
    t_days = (date_t - date_start).days
    t_max_days = (date_tmax - date_start).days
    t_min_days = (date_tmin - date_start).days
    derived_period = 2 * abs(t_min_days - t_max_days)

    v_expected = _calc_seasonality_balcan_at_t(t_days, t_max_days, val_min, val_max, derived_period)
    assert np.isclose(v, v_expected, rtol=0, atol=1e-12), "calc_seasonality_balcan_at_date did not match expected"


def test_calc_seasonality_balcan_at_date_period_overrides_tmin():
    """If both period and date_tmin are provided, the explicit period must be used (override)."""
    val_min, val_max = 0.4, 1.6
    date_start = dt.date(2025, 10, 1)
    date_tmax = dt.date(2025, 12, 31)  # Peak in late December
    date_tmin = dt.date(2026, 6, 15)  # Trough in mid-June (would imply its own period if used)
    forced_period = 200
    date_t = dt.date(2026, 2, 1)

    v = calc_seasonality_balcan_at_date(
        date_t=date_t,
        date_start=date_start,
        date_tmax=date_tmax,
        date_tmin=date_tmin,
        val_min=val_min,
        val_max=val_max,
        period=forced_period,
    )

    t_days = (date_t - date_start).days
    t_max_days = (date_tmax - date_start).days
    v_expected = _calc_seasonality_balcan_at_t(t_days, t_max_days, val_min, val_max, forced_period)
    assert np.isclose(v, v_expected, rtol=0, atol=1e-12), "calc_seasonality_balcan_at_date did not match expected"


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
    """End-to-end test with realistic flu season dates.

    Flu transmission peaks in winter (late December) and troughs in summer (mid-June).
    """
    # Flu season timespan: Oct 2025 - May 2026
    date_start = dt.date(2025, 10, 1)
    date_stop = dt.date(2026, 5, 31)
    date_tmax = dt.date(2025, 12, 31)  # Peak transmission in late December
    date_tmin = dt.date(2026, 6, 15)  # Trough in mid-June (outside range, used for period calc)
    val_min, val_max = 0.7, 1.3

    dates, values = get_seasonal_transmission_balcan(
        date_start=date_start,
        date_stop=date_stop,
        date_tmax=date_tmax,
        val_min=val_min,
        val_max=val_max,
        date_tmin=date_tmin,
    )

    # Basic shape
    assert dates[0] == date_start and dates[-1] == date_stop, "Date range is not as expected"
    assert len(dates) == len(values) == (date_stop - date_start).days + 1, (
        "Date and value arrays are not the same length"
    )

    # Peak check: the value at tmax index should be (almost) the global maximum ~ 1.0
    idx_tmax = (date_tmax - date_start).days
    assert np.isclose(values[idx_tmax], 1.0, rtol=0, atol=1e-12), "Value at tmax is not as expected"
    assert np.isclose(values[idx_tmax], max(values), rtol=0, atol=1e-12), "Value at tmax is not the maximum"

    # Verify transmission is lower at start of season (October) than at peak (December)
    assert values[0] < values[idx_tmax], "October transmission should be lower than December peak"


class TestSeasonalityPrimitivesSubdaily:
    """Tests verifying seasonality functions work correctly with subdaily delta_t values."""

    @pytest.mark.parametrize("delta_t", [1.0, 0.5, 0.25])
    def test_calc_seasonality_balcan_at_date_consistent_across_dt(self, delta_t):
        """Same date should produce the same seasonal factor regardless of delta_t."""
        date_start = dt.date(2025, 10, 1)
        date_tmax = dt.date(2025, 12, 31)
        date_t = dt.date(2025, 11, 15)
        val_min, val_max = 0.5, 1.0

        value = calc_seasonality_balcan_at_date(
            date_t=date_t,
            date_start=date_start,
            date_tmax=date_tmax,
            val_min=val_min,
            val_max=val_max,
            delta_t=delta_t,
        )

        # Reference value at dt=1.0
        reference = calc_seasonality_balcan_at_date(
            date_t=date_t,
            date_start=date_start,
            date_tmax=date_tmax,
            val_min=val_min,
            val_max=val_max,
            delta_t=1.0,
        )

        assert np.isclose(value, reference, rtol=0, atol=1e-12), (
            f"Seasonal factor at delta_t={delta_t} ({value}) differs from delta_t=1.0 ({reference})"
        )

    @pytest.mark.parametrize(
        ("delta_t", "expected_len"),
        [
            (1.0, 11),
            (0.5, 21),
            (0.25, 41),
        ],
    )
    def test_generate_seasonal_values_correct_length(self, delta_t, expected_len):
        """Verify generated array length matches expected timestep count."""
        date_start = dt.date(2025, 1, 1)
        date_stop = dt.date(2025, 1, 11)  # 10-day range

        def dummy(d):
            return 1.0

        dates, values = generate_seasonal_values(date_start, date_stop, dummy, delta_t=delta_t)

        assert len(dates) == expected_len, f"Expected {expected_len} dates for delta_t={delta_t}, got {len(dates)}"
        assert len(values) == expected_len

    def test_generate_seasonal_values_subdaily_dates_are_datetimes(self):
        """With subdaily delta_t, dates should be datetime objects (not date)."""
        date_start = dt.date(2025, 1, 1)
        date_stop = dt.date(2025, 1, 3)

        def dummy(d):
            return 1.0

        dates, _ = generate_seasonal_values(date_start, date_stop, dummy, delta_t=0.5)

        # Sub-day entries should be datetime objects
        for d in dates:
            assert isinstance(d, (dt.datetime, dt.date))
        # At least some entries should be datetime (the half-day points)
        # The half-day entries (12:00:00) should be present
        has_non_midnight = any(isinstance(d, dt.datetime) and d.hour != 0 for d in dates)
        assert has_non_midnight, "Subdaily delta_t should produce datetime entries with non-midnight times"

    def test_get_seasonal_transmission_balcan_subdaily(self):
        """Full season with delta_t=0.5: correct length, peak at tmax, values in range."""
        date_start = dt.date(2025, 10, 1)
        date_stop = dt.date(2026, 5, 31)
        date_tmax = dt.date(2025, 12, 31)
        date_tmin = dt.date(2026, 6, 15)
        val_min, val_max = 0.5, 1.0
        delta_t = 0.5

        dates, values = get_seasonal_transmission_balcan(
            date_start=date_start,
            date_stop=date_stop,
            date_tmax=date_tmax,
            date_tmin=date_tmin,
            val_min=val_min,
            val_max=val_max,
            delta_t=delta_t,
        )

        # Check length
        total_days = (date_stop - date_start).days
        expected_len = int(total_days / delta_t) + 1
        assert len(dates) == expected_len, f"Expected {expected_len} entries, got {len(dates)}"
        assert len(values) == expected_len

        # Values should be in range [val_min/val_max, 1.0]
        min_expected = val_min / val_max
        assert min(values) >= min_expected - 1e-10
        assert max(values) <= 1.0 + 1e-10

        # Peak should be near 1.0
        assert np.isclose(max(values), 1.0, atol=1e-6)

    def test_get_scaled_parameter_subdaily(self):
        """get_scaled_parameter with delta_t=0.5 produces correct length and scaling."""
        date_start = dt.date(2025, 1, 1)
        date_stop = dt.date(2025, 1, 11)  # 10-day range
        scaling_start = dt.date(2025, 1, 3)
        scaling_stop = dt.date(2025, 1, 7)
        scaling_factor = 0.5
        delta_t = 0.5

        dates, values = get_scaled_parameter(
            date_start=date_start,
            date_stop=date_stop,
            scaling_start=scaling_start,
            scaling_stop=scaling_stop,
            scaling_factor=scaling_factor,
            delta_t=delta_t,
        )

        # Check length
        total_days = (date_stop - date_start).days
        expected_len = int(total_days / delta_t) + 1
        assert len(dates) == expected_len
        assert len(values) == expected_len

        # Values outside intervention period should be 1.0, inside should be scaling_factor
        for d, v in zip(dates, values, strict=True):
            d_date = d.date() if isinstance(d, dt.datetime) else d
            if scaling_start <= d_date <= scaling_stop:
                assert v == scaling_factor, f"Expected {scaling_factor} at {d}, got {v}"
            else:
                assert v == 1.0, f"Expected 1.0 at {d}, got {v}"
