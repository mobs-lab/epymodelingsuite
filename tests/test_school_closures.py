"""End-to-end tests for school closure dynamics.

These tests verify that school closure interventions correctly reduce
transmission in epidemic simulations.
"""

import datetime as dt

import numpy as np
import pytest
from epydemix.model import EpiModel
from epydemix.population import load_epydemix_population


def create_sir_model_with_school_contacts(
    reduction_factor: float | None = None,
    closure_start: dt.date | None = None,
    closure_end: dt.date | None = None,
) -> EpiModel:
    """Create an SIR model with age structure and school contact layer.

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
    model = EpiModel()

    # Load population with contact matrices
    age_group_mapping = {
        "0-4": [str(i) for i in range(5)],
        "5-17": [str(i) for i in range(5, 18)],
        "18-49": [str(i) for i in range(18, 50)],
        "50-64": [str(i) for i in range(50, 65)],
        "65+": [str(i) for i in range(65, 84)] + ["84+"],
    }
    population = load_epydemix_population(
        population_name="United_States_Massachusetts",
        age_group_mapping=age_group_mapping,
    )
    model.set_population(population)

    # Add compartments
    model.add_compartments(["S", "I", "R"])

    # Add transitions
    model.add_transition("S", "I", params=("beta", "I"), kind="mediated")
    model.add_transition("I", "R", params="gamma", kind="spontaneous")

    # Use moderate transmission to avoid epidemic saturation while showing clear effect
    # beta=0.15 with gamma=0.1 gives R0 ~ 1.5 for a slower epidemic
    model.add_parameter(parameters_dict={"beta": 0.15, "gamma": 0.1})

    # Add school closure intervention if specified
    if reduction_factor is not None and closure_start is not None and closure_end is not None:
        model.add_intervention(
            layer_name="school",
            start_date=closure_start.isoformat(),
            end_date=closure_end.isoformat(),
            reduction_factor=reduction_factor,
            name="School Closure Test",
        )

    return model


@pytest.mark.dynamics
class TestSchoolClosureE2E:
    """End-to-end tests verifying school closure reduces infections."""

    @pytest.fixture
    def simulation_dates(self):
        """Provide simulation period spanning school year."""
        return {
            "start_date": "2025-01-01",
            "end_date": "2025-02-01",  # Shorter period to avoid saturation
            "closure_start": dt.date(2025, 1, 5),
            "closure_end": dt.date(2025, 1, 25),
        }

    @pytest.fixture
    def initial_conditions(self):
        """Create initial conditions with seed infections in school-age group."""

        def _make_init_conditions(model: EpiModel) -> dict:
            n_age = len(model.population.Nk)
            i_init = np.zeros(n_age)
            # Seed infections in 5-17 age group (index 1)
            i_init[1] = 200
            return {
                "S": model.population.Nk - i_init,
                "I": i_init,
                "R": np.zeros(n_age),
            }

        return _make_init_conditions

    def test_school_closure_reduces_infections(self, simulation_dates, initial_conditions):
        """Test that school closures reduce total infections.

        Creates two models:
        1. No school closure (baseline)
        2. With school closure (reduction_factor=0.0 = complete closure)

        Verifies that the school closure model has fewer total S->I transitions.
        """
        # Model without closure
        model_baseline = create_sir_model_with_school_contacts()
        init_baseline = initial_conditions(model_baseline)

        # Model with complete school closure
        model_closure = create_sir_model_with_school_contacts(
            reduction_factor=0.0,
            closure_start=simulation_dates["closure_start"],
            closure_end=simulation_dates["closure_end"],
        )
        init_closure = initial_conditions(model_closure)

        # Run simulations
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

        # Get total infections (S->I transitions)
        transitions_baseline = results_baseline.get_stacked_transitions()
        transitions_closure = results_closure.get_stacked_transitions()

        infections_baseline = np.sum(transitions_baseline["S_to_I_total"], axis=1)
        infections_closure = np.sum(transitions_closure["S_to_I_total"], axis=1)

        avg_baseline = np.mean(infections_baseline)
        avg_closure = np.mean(infections_closure)

        # School closure should reduce infections
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
        # Model with complete closure
        model_complete = create_sir_model_with_school_contacts(
            reduction_factor=0.0,
            closure_start=simulation_dates["closure_start"],
            closure_end=simulation_dates["closure_end"],
        )
        init_complete = initial_conditions(model_complete)

        # Model with partial closure
        model_partial = create_sir_model_with_school_contacts(
            reduction_factor=0.5,
            closure_start=simulation_dates["closure_start"],
            closure_end=simulation_dates["closure_end"],
        )
        init_partial = initial_conditions(model_partial)

        # Run simulations
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

        # Get total infections
        transitions_complete = results_complete.get_stacked_transitions()
        transitions_partial = results_partial.get_stacked_transitions()

        infections_complete = np.sum(transitions_complete["S_to_I_total"], axis=1)
        infections_partial = np.sum(transitions_partial["S_to_I_total"], axis=1)

        avg_complete = np.mean(infections_complete)
        avg_partial = np.mean(infections_partial)

        # Complete closure should have fewer infections than partial
        assert avg_complete < avg_partial, (
            f"Complete closure should reduce infections more than partial: "
            f"complete={avg_complete:.0f}, partial={avg_partial:.0f}"
        )
