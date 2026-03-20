"""Tests for intervention builder functions.

Includes E2E tests for parameter interventions that verify epidemic dynamics.
"""

import datetime as dt

import numpy as np
import pytest
from epydemix.model import EpiModel
from epydemix.population import load_epydemix_population

from epymodelingsuite.builders.interventions import add_parameter_interventions_from_config
from epymodelingsuite.schema.basemodel import Intervention, Timespan


def _create_sir_model_for_interventions() -> EpiModel:
    """Create a basic SIR model for intervention testing.

    Returns
    -------
    EpiModel
        Basic SIR model with population and parameters set.
    """
    model = EpiModel()

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

    return model


def _create_initial_conditions(model: EpiModel, seed_infections: int = 100) -> dict:
    """Create initial conditions for parameter intervention testing.

    Parameters
    ----------
    model : EpiModel
        Model with population set.
    seed_infections : int
        Number of initial infections.

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
class TestParameterInterventionE2E:
    """End-to-end tests verifying parameter interventions affect epidemic dynamics."""

    def test_parameter_reduction_reduces_infections(self):
        """Test that reducing beta mid-simulation reduces subsequent infections.

        Creates two models:
        1. Constant beta (no intervention)
        2. Beta reduced by 50% starting on day 15

        Verifies that the intervention model has fewer total infections.
        """
        # Simulation parameters
        start_date = dt.date(2025, 1, 1)
        end_date = dt.date(2025, 2, 1)
        baseline_beta = 0.2
        gamma = 0.1

        # Model 1: No intervention (baseline)
        model_baseline = _create_sir_model_for_interventions()
        model_baseline.add_parameter(parameters_dict={"beta": baseline_beta, "gamma": gamma})
        init_baseline = _create_initial_conditions(model_baseline)

        # Model 2: With parameter intervention (beta reduced by 50% from day 15)
        model_intervention = _create_sir_model_for_interventions()
        model_intervention.add_parameter(parameters_dict={"beta": baseline_beta, "gamma": gamma})

        # Create intervention that reduces beta by 50% from Jan 15 to end
        intervention = Intervention(
            type="parameter",
            target_parameter="beta",
            scaling_factor=0.5,  # 50% of baseline
            start_date=dt.date(2025, 1, 15),
            end_date=end_date,
        )
        timespan = Timespan(start_date=start_date, end_date=end_date, delta_t=1.0)

        # Apply intervention
        add_parameter_interventions_from_config(
            model=model_intervention,
            interventions=[intervention],
            timespan=timespan,
        )
        init_intervention = _create_initial_conditions(model_intervention)

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

        # Get total infections
        transitions_baseline = results_baseline.get_stacked_transitions()
        transitions_intervention = results_intervention.get_stacked_transitions()

        infections_baseline = np.sum(transitions_baseline["S_to_I_total"], axis=1)
        infections_intervention = np.sum(transitions_intervention["S_to_I_total"], axis=1)

        avg_baseline = np.mean(infections_baseline)
        avg_intervention = np.mean(infections_intervention)

        # Parameter intervention should reduce infections
        assert avg_intervention < avg_baseline, (
            f"Parameter intervention should reduce infections: "
            f"baseline={avg_baseline:.0f}, with_intervention={avg_intervention:.0f}"
        )
