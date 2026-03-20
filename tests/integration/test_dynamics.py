"""End-to-end dynamics tests for epidemic interventions and seasonality.

These tests verify that interventions (school closures, parameter changes,
seasonality) correctly affect transmission dynamics in epidemic simulations.
All tests use real EpiModel objects and run actual simulations.
"""

import datetime as dt

import numpy as np
import pytest
from epydemix.model import EpiModel
from epydemix.population import load_epydemix_population

from epymodelingsuite.builders.interventions import add_parameter_interventions_from_config
from epymodelingsuite.builders.seasonality import add_seasonality_from_config
from epymodelingsuite.schema.basemodel import Intervention, Seasonality, Timespan

# =============================================================================
# Shared helpers
# =============================================================================

_AGE_GROUP_MAPPING = {
    "0-4": [str(i) for i in range(5)],
    "5-17": [str(i) for i in range(5, 18)],
    "18-49": [str(i) for i in range(18, 50)],
    "50-64": [str(i) for i in range(50, 65)],
    "65+": [str(i) for i in range(65, 84)] + ["84+"],
}


def create_sir_model(location: str = "United_States_Massachusetts") -> EpiModel:
    """Create a basic SIR model with age structure.

    Parameters
    ----------
    location : str
        Population name for epydemix.

    Returns
    -------
    EpiModel
        SIR model with population, compartments, and transitions set.
    """
    model = EpiModel()

    population = load_epydemix_population(
        population_name=location,
        age_group_mapping=_AGE_GROUP_MAPPING,
    )
    model.set_population(population)

    model.add_compartments(["S", "I", "R"])
    model.add_transition("S", "I", params=("beta", "I"), kind="mediated")
    model.add_transition("I", "R", params="gamma", kind="spontaneous")

    return model


def create_initial_conditions(model: EpiModel, seed_age_index: int = 2, seed_infections: int = 100) -> dict:
    """Create initial conditions for SIR simulation.

    Parameters
    ----------
    model : EpiModel
        Model with population set.
    seed_age_index : int
        Index of age group to seed infections in (default 2 = 18-49).
    seed_infections : int
        Number of initial infections.

    Returns
    -------
    dict
        Initial conditions dictionary.
    """
    n_age = len(model.population.Nk)
    i_init = np.zeros(n_age)
    i_init[seed_age_index] = seed_infections
    return {
        "S": model.population.Nk - i_init,
        "I": i_init,
        "R": np.zeros(n_age),
    }


def create_sir_model_with_school_contacts(
    reduction_factor: float | None = None,
    closure_start: dt.date | None = None,
    closure_end: dt.date | None = None,
) -> EpiModel:
    """Create an SIR model with school contact layer and optional closure.

    Parameters
    ----------
    reduction_factor : float, optional
        School closure reduction factor. None means no closure.
    closure_start : date, optional
        Start date of school closure.
    closure_end : date, optional
        End date of school closure.

    Returns
    -------
    EpiModel
        Model with school contacts and optional closure intervention.
    """
    model = create_sir_model()

    # Use moderate transmission to avoid epidemic saturation while showing clear effect
    # beta=0.15 with gamma=0.1 gives R0 ~ 1.5 for a slower epidemic
    model.add_parameter(parameters_dict={"beta": 0.15, "gamma": 0.1})

    if reduction_factor is not None and closure_start is not None and closure_end is not None:
        model.add_intervention(
            layer_name="school",
            start_date=closure_start.isoformat(),
            end_date=closure_end.isoformat(),
            reduction_factor=reduction_factor,
            name="School Closure Test",
        )

    return model


# =============================================================================
# Seasonality dynamics tests
# =============================================================================


@pytest.mark.dynamics
class TestSeasonalityDynamics:
    """Tests verifying seasonality affects transmission dynamics."""

    def test_seasonality_affects_transmission(self):
        """Test that seasonal transmission differs from constant transmission.

        Creates two models:
        1. Constant beta (no seasonality)
        2. Seasonal beta (using add_seasonality_from_config)

        Verifies that the infection dynamics differ between the two models.
        """
        start_date = dt.date(2025, 10, 1)
        end_date = dt.date(2025, 12, 31)
        baseline_beta = 0.2
        gamma = 0.1

        # Model 1: Constant transmission
        model_constant = create_sir_model()
        model_constant.add_parameter(parameters_dict={"beta": baseline_beta, "gamma": gamma})
        init_constant = create_initial_conditions(model_constant)

        # Model 2: Seasonal transmission
        model_seasonal = create_sir_model()
        model_seasonal.add_parameter(parameters_dict={"beta": baseline_beta, "gamma": gamma})

        seasonality_config = Seasonality(
            method="balcan",
            min_value=0.3,
            max_value=1.0,
            target_parameter="beta",
            seasonality_max_date=dt.date(2025, 12, 31),
            seasonality_min_date=dt.date(2026, 6, 15),
        )
        timespan = Timespan(start_date=start_date, end_date=end_date, delta_t=1.0)
        add_seasonality_from_config(model_seasonal, seasonality_config, timespan)
        init_seasonal = create_initial_conditions(model_seasonal)

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

        transitions_constant = results_constant.get_stacked_transitions()
        transitions_seasonal = results_seasonal.get_stacked_transitions()

        infections_constant = np.sum(transitions_constant["S_to_I_total"], axis=1)
        infections_seasonal = np.sum(transitions_seasonal["S_to_I_total"], axis=1)

        avg_constant = np.mean(infections_constant)
        avg_seasonal = np.mean(infections_seasonal)

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
        baseline_beta = 0.2
        gamma = 0.1

        seasonality_config = Seasonality(
            method="balcan",
            min_value=0.3,
            max_value=1.0,
            target_parameter="beta",
            seasonality_max_date=dt.date(2025, 12, 31),
            seasonality_min_date=dt.date(2026, 6, 15),
        )

        # Winter simulation (Dec 1 - Feb 1)
        winter_start = dt.date(2025, 12, 1)
        winter_end = dt.date(2026, 2, 1)

        model_winter = create_sir_model()
        model_winter.add_parameter(parameters_dict={"beta": baseline_beta, "gamma": gamma})
        timespan_winter = Timespan(start_date=winter_start, end_date=winter_end, delta_t=1.0)
        add_seasonality_from_config(model_winter, seasonality_config, timespan_winter)
        init_winter = create_initial_conditions(model_winter)

        # Summer simulation (Jun 1 - Aug 1)
        summer_start = dt.date(2026, 6, 1)
        summer_end = dt.date(2026, 8, 1)

        model_summer = create_sir_model()
        model_summer.add_parameter(parameters_dict={"beta": baseline_beta, "gamma": gamma})
        timespan_summer = Timespan(start_date=summer_start, end_date=summer_end, delta_t=1.0)
        add_seasonality_from_config(model_summer, seasonality_config, timespan_summer)
        init_summer = create_initial_conditions(model_summer)

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

        transitions_winter = results_winter.get_stacked_transitions()
        transitions_summer = results_summer.get_stacked_transitions()

        infections_winter = np.sum(transitions_winter["S_to_I_total"], axis=1)
        infections_summer = np.sum(transitions_summer["S_to_I_total"], axis=1)

        avg_winter = np.mean(infections_winter)
        avg_summer = np.mean(infections_summer)

        assert avg_winter > avg_summer, (
            f"Winter should have more infections than summer: winter={avg_winter:.0f}, summer={avg_summer:.0f}"
        )


# =============================================================================
# School closure dynamics tests
# =============================================================================


@pytest.mark.dynamics
class TestSchoolClosureDynamics:
    """Tests verifying school closure reduces infections."""

    @pytest.fixture
    def simulation_dates(self):
        """Provide simulation period spanning school year."""
        return {
            "start_date": "2025-01-01",
            "end_date": "2025-02-01",
            "closure_start": dt.date(2025, 1, 5),
            "closure_end": dt.date(2025, 1, 25),
        }

    @pytest.fixture
    def initial_conditions(self):
        """Create initial conditions with seed infections in school-age group."""

        def _make_init_conditions(model: EpiModel) -> dict:
            return create_initial_conditions(model, seed_age_index=1, seed_infections=200)

        return _make_init_conditions

    def test_school_closure_reduces_infections(self, simulation_dates, initial_conditions):
        """Test that school closures reduce total infections.

        Creates two models:
        1. No school closure (baseline)
        2. With school closure (reduction_factor=0.0 = complete closure)

        Verifies that the school closure model has fewer total S->I transitions.
        """
        model_baseline = create_sir_model_with_school_contacts()
        init_baseline = initial_conditions(model_baseline)

        model_closure = create_sir_model_with_school_contacts(
            reduction_factor=0.0,
            closure_start=simulation_dates["closure_start"],
            closure_end=simulation_dates["closure_end"],
        )
        init_closure = initial_conditions(model_closure)

        rng1 = np.random.default_rng(42)
        results_baseline = model_baseline.run_simulations(
            start_date=simulation_dates["start_date"],
            end_date=simulation_dates["end_date"],
            initial_conditions_dict=init_baseline,
            Nsim=10,
            dt=1.0,
            rng=rng1,
        )

        rng2 = np.random.default_rng(42)
        results_closure = model_closure.run_simulations(
            start_date=simulation_dates["start_date"],
            end_date=simulation_dates["end_date"],
            initial_conditions_dict=init_closure,
            Nsim=10,
            dt=1.0,
            rng=rng2,
        )

        transitions_baseline = results_baseline.get_stacked_transitions()
        transitions_closure = results_closure.get_stacked_transitions()

        infections_baseline = np.sum(transitions_baseline["S_to_I_total"], axis=1)
        infections_closure = np.sum(transitions_closure["S_to_I_total"], axis=1)

        avg_baseline = np.mean(infections_baseline)
        avg_closure = np.mean(infections_closure)

        assert avg_closure < avg_baseline, (
            f"School closure should reduce infections: baseline={avg_baseline:.0f}, with_closure={avg_closure:.0f}"
        )

    def test_complete_vs_partial_closure(self, simulation_dates, initial_conditions):
        """Test that complete closure (factor=0.0) reduces infections more than partial (factor=0.5).

        Creates two models with school closures:
        1. Complete closure (reduction_factor=0.0)
        2. Partial closure (reduction_factor=0.5)

        Verifies that complete closure has fewer total infections.
        """
        model_complete = create_sir_model_with_school_contacts(
            reduction_factor=0.0,
            closure_start=simulation_dates["closure_start"],
            closure_end=simulation_dates["closure_end"],
        )
        init_complete = initial_conditions(model_complete)

        model_partial = create_sir_model_with_school_contacts(
            reduction_factor=0.5,
            closure_start=simulation_dates["closure_start"],
            closure_end=simulation_dates["closure_end"],
        )
        init_partial = initial_conditions(model_partial)

        rng1 = np.random.default_rng(42)
        results_complete = model_complete.run_simulations(
            start_date=simulation_dates["start_date"],
            end_date=simulation_dates["end_date"],
            initial_conditions_dict=init_complete,
            Nsim=10,
            dt=1.0,
            rng=rng1,
        )

        rng2 = np.random.default_rng(42)
        results_partial = model_partial.run_simulations(
            start_date=simulation_dates["start_date"],
            end_date=simulation_dates["end_date"],
            initial_conditions_dict=init_partial,
            Nsim=10,
            dt=1.0,
            rng=rng2,
        )

        transitions_complete = results_complete.get_stacked_transitions()
        transitions_partial = results_partial.get_stacked_transitions()

        infections_complete = np.sum(transitions_complete["S_to_I_total"], axis=1)
        infections_partial = np.sum(transitions_partial["S_to_I_total"], axis=1)

        avg_complete = np.mean(infections_complete)
        avg_partial = np.mean(infections_partial)

        assert avg_complete < avg_partial, (
            f"Complete closure should reduce infections more than partial: "
            f"complete={avg_complete:.0f}, partial={avg_partial:.0f}"
        )


# =============================================================================
# Parameter intervention dynamics tests
# =============================================================================


@pytest.mark.dynamics
class TestParameterInterventionDynamics:
    """Tests verifying parameter interventions affect epidemic dynamics."""

    def test_parameter_reduction_reduces_infections(self):
        """Test that reducing beta mid-simulation reduces subsequent infections.

        Creates two models:
        1. Constant beta (no intervention)
        2. Beta reduced by 50% starting on day 15

        Verifies that the intervention model has fewer total infections.
        """
        start_date = dt.date(2025, 1, 1)
        end_date = dt.date(2025, 2, 1)
        baseline_beta = 0.2
        gamma = 0.1

        # Model 1: No intervention (baseline)
        model_baseline = create_sir_model()
        model_baseline.add_parameter(parameters_dict={"beta": baseline_beta, "gamma": gamma})
        init_baseline = create_initial_conditions(model_baseline)

        # Model 2: With parameter intervention (beta reduced by 50% from day 15)
        model_intervention = create_sir_model()
        model_intervention.add_parameter(parameters_dict={"beta": baseline_beta, "gamma": gamma})

        intervention = Intervention(
            type="parameter",
            target_parameter="beta",
            scaling_factor=0.5,
            start_date=dt.date(2025, 1, 15),
            end_date=end_date,
        )
        timespan = Timespan(start_date=start_date, end_date=end_date, delta_t=1.0)

        add_parameter_interventions_from_config(
            model=model_intervention,
            interventions=[intervention],
            timespan=timespan,
        )
        init_intervention = create_initial_conditions(model_intervention)

        # Run simulations
        rng1 = np.random.default_rng(42)
        results_baseline = model_baseline.run_simulations(
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            initial_conditions_dict=init_baseline,
            Nsim=10,
            dt=1.0,
            rng=rng1,
        )

        rng2 = np.random.default_rng(42)
        results_intervention = model_intervention.run_simulations(
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            initial_conditions_dict=init_intervention,
            Nsim=10,
            dt=1.0,
            rng=rng2,
        )

        transitions_baseline = results_baseline.get_stacked_transitions()
        transitions_intervention = results_intervention.get_stacked_transitions()

        infections_baseline = np.sum(transitions_baseline["S_to_I_total"], axis=1)
        infections_intervention = np.sum(transitions_intervention["S_to_I_total"], axis=1)

        avg_baseline = np.mean(infections_baseline)
        avg_intervention = np.mean(infections_intervention)

        assert avg_intervention < avg_baseline, (
            f"Parameter intervention should reduce infections: "
            f"baseline={avg_baseline:.0f}, with_intervention={avg_intervention:.0f}"
        )
