# Unit tests for seasonality.py
import datetime as dt

import numpy as np
import pytest

from epymodelingsuite.seasonality import (
    _calc_seasonality_balcan_at_t,
    calc_seasonality_balcan_at_date,
    generate_seasonal_values,
    get_seasonal_transmission_balcan,
)


def test__calc_seasonality_balcan_at_t_peak_and_min():
    """Check that the peak at t_max is 1.0 and the trough at half-period is val_min/val_max."""
    val_min, val_max = 0.6, 1.4
    period = 365
    t_max = 120

    # Peak exactly at t_max -> should be 1.0
    vals = [_calc_seasonality_balcan_at_t(t, t_max, val_min, val_max, period) for t in range(0, period + 1)]

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


# =============================================================================
# E2E tests for seasonality dynamics
# =============================================================================


def _create_sir_model_for_seasonality(location: str = "United_States_Massachusetts"):
    """Create a basic SIR model with age structure for seasonality E2E tests.

    Parameters
    ----------
    location : str
        Population name for epydemix.

    Returns
    -------
    EpiModel
        Basic SIR model with population set.
    """
    from epydemix.model import EpiModel
    from epydemix.population import load_epydemix_population

    model = EpiModel()

    age_group_mapping = {
        "0-4": [str(i) for i in range(5)],
        "5-17": [str(i) for i in range(5, 18)],
        "18-49": [str(i) for i in range(18, 50)],
        "50-64": [str(i) for i in range(50, 65)],
        "65+": [str(i) for i in range(65, 84)] + ["84+"],
    }
    population = load_epydemix_population(
        population_name=location,
        age_group_mapping=age_group_mapping,
    )
    model.set_population(population)

    # Add compartments
    model.add_compartments(["S", "I", "R"])

    # Add transitions
    model.add_transition("S", "I", params=("beta", "I"), kind="mediated")
    model.add_transition("I", "R", params="gamma", kind="spontaneous")

    return model


def _create_initial_conditions_for_seasonality(model, seed_infections: int = 100) -> dict:
    """Create initial conditions for seasonality E2E simulation.

    Parameters
    ----------
    model : EpiModel
        Model with population set.
    seed_infections : int
        Number of initial infections in 18-49 age group.

    Returns
    -------
    dict
        Initial conditions dictionary.
    """
    n_age = len(model.population.Nk)
    i_init = np.zeros(n_age)
    i_init[2] = seed_infections  # Seed in 18-49 age group
    return {
        "S": model.population.Nk - i_init,
        "I": i_init,
        "R": np.zeros(n_age),
    }


@pytest.mark.dynamics
class TestSeasonalityE2E:
    """End-to-end tests verifying seasonality affects transmission dynamics."""

    def test_seasonality_affects_transmission(self):
        """Test that seasonal transmission differs from constant transmission.

        Creates two models:
        1. Constant beta (no seasonality)
        2. Seasonal beta (using add_seasonality_from_config)

        Verifies that the infection dynamics differ between the two models.
        """
        from epymodelingsuite.builders.seasonality import add_seasonality_from_config
        from epymodelingsuite.schema.basemodel import Seasonality, Timespan

        # Simulation parameters
        start_date = dt.date(2025, 10, 1)
        end_date = dt.date(2025, 12, 31)
        baseline_beta = 0.2
        gamma = 0.1

        # Model 1: Constant transmission
        model_constant = _create_sir_model_for_seasonality()
        model_constant.add_parameter(parameters_dict={"beta": baseline_beta, "gamma": gamma})
        init_constant = _create_initial_conditions_for_seasonality(model_constant)

        # Model 2: Seasonal transmission
        model_seasonal = _create_sir_model_for_seasonality()
        model_seasonal.add_parameter(parameters_dict={"beta": baseline_beta, "gamma": gamma})

        # Apply seasonality - winter peak on Dec 31, summer trough on June 15
        seasonality_config = Seasonality(
            method="balcan",
            min_value=0.3,  # Summer trough at 30% of baseline
            max_value=1.0,  # Winter peak at 100% of baseline
            target_parameter="beta",
            seasonality_max_date=dt.date(2025, 12, 31),
            seasonality_min_date=dt.date(2026, 6, 15),
        )
        timespan = Timespan(start_date=start_date, end_date=end_date, delta_t=1.0)
        add_seasonality_from_config(model_seasonal, seasonality_config, timespan)
        init_seasonal = _create_initial_conditions_for_seasonality(model_seasonal)

        # Run simulations
        rng1 = np.random.default_rng(42)
        results_constant = model_constant.run_simulations(
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            initial_conditions_dict=init_constant,
            Nsim=10,
            dt=1.0,
            rng=rng1,
        )

        rng2 = np.random.default_rng(42)
        results_seasonal = model_seasonal.run_simulations(
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            initial_conditions_dict=init_seasonal,
            Nsim=10,
            dt=1.0,
            rng=rng2,
        )

        # Get total infections
        transitions_constant = results_constant.get_stacked_transitions()
        transitions_seasonal = results_seasonal.get_stacked_transitions()

        infections_constant = np.sum(transitions_constant["S_to_I_total"], axis=1)
        infections_seasonal = np.sum(transitions_seasonal["S_to_I_total"], axis=1)

        avg_constant = np.mean(infections_constant)
        avg_seasonal = np.mean(infections_seasonal)

        # Verify the models produce different results
        assert avg_seasonal != avg_constant, (
            f"Seasonal model should differ from constant: constant={avg_constant:.0f}, seasonal={avg_seasonal:.0f}"
        )

    def test_winter_higher_transmission(self):
        """Test that winter simulations have more infections than summer.

        Creates two simulations with identical parameters and seasonality,
        but running at different times of year:
        1. Winter simulation (Dec-Feb): High transmission period
        2. Summer simulation (Jun-Aug): Low transmission period

        Verifies that the winter simulation has more total infections.
        """
        from epymodelingsuite.builders.seasonality import add_seasonality_from_config
        from epymodelingsuite.schema.basemodel import Seasonality, Timespan

        # Common parameters
        baseline_beta = 0.2
        gamma = 0.1

        # Seasonality config: peak in winter (Dec 31), trough in summer (June 15)
        seasonality_config = Seasonality(
            method="balcan",
            min_value=0.3,  # Summer at 30% of peak
            max_value=1.0,
            target_parameter="beta",
            seasonality_max_date=dt.date(2025, 12, 31),
            seasonality_min_date=dt.date(2026, 6, 15),
        )

        # Winter simulation (Dec 1 - Feb 1)
        winter_start = dt.date(2025, 12, 1)
        winter_end = dt.date(2026, 2, 1)

        model_winter = _create_sir_model_for_seasonality()
        model_winter.add_parameter(parameters_dict={"beta": baseline_beta, "gamma": gamma})
        timespan_winter = Timespan(start_date=winter_start, end_date=winter_end, delta_t=1.0)
        add_seasonality_from_config(model_winter, seasonality_config, timespan_winter)
        init_winter = _create_initial_conditions_for_seasonality(model_winter)

        # Summer simulation (Jun 1 - Aug 1)
        summer_start = dt.date(2026, 6, 1)
        summer_end = dt.date(2026, 8, 1)

        model_summer = _create_sir_model_for_seasonality()
        model_summer.add_parameter(parameters_dict={"beta": baseline_beta, "gamma": gamma})
        timespan_summer = Timespan(start_date=summer_start, end_date=summer_end, delta_t=1.0)
        add_seasonality_from_config(model_summer, seasonality_config, timespan_summer)
        init_summer = _create_initial_conditions_for_seasonality(model_summer)

        # Run simulations
        rng1 = np.random.default_rng(42)
        results_winter = model_winter.run_simulations(
            start_date=winter_start.isoformat(),
            end_date=winter_end.isoformat(),
            initial_conditions_dict=init_winter,
            Nsim=10,
            dt=1.0,
            rng=rng1,
        )

        rng2 = np.random.default_rng(42)
        results_summer = model_summer.run_simulations(
            start_date=summer_start.isoformat(),
            end_date=summer_end.isoformat(),
            initial_conditions_dict=init_summer,
            Nsim=10,
            dt=1.0,
            rng=rng2,
        )

        # Get total infections
        transitions_winter = results_winter.get_stacked_transitions()
        transitions_summer = results_summer.get_stacked_transitions()

        infections_winter = np.sum(transitions_winter["S_to_I_total"], axis=1)
        infections_summer = np.sum(transitions_summer["S_to_I_total"], axis=1)

        avg_winter = np.mean(infections_winter)
        avg_summer = np.mean(infections_summer)

        # Winter should have more infections due to higher transmission
        assert avg_winter > avg_summer, (
            f"Winter should have more infections than summer: winter={avg_winter:.0f}, summer={avg_summer:.0f}"
        )
