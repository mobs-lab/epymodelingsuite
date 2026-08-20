"""Shared test fixtures and helpers for epymodelingsuite tests."""

from __future__ import annotations

from datetime import date

from epymodelingsuite.schema.basemodel import (
    BaseEpiModel,
    Compartment,
    Parameter,
    Population,
    Simulation,
    Timespan,
    Transition,
)
from epymodelingsuite.schema.dispatcher import BuilderOutput

# =============================================================================
# Shared Constants
# =============================================================================

AGE_GROUPS = ["0-4", "5-17", "18-49", "50-64", "65+"]

AGE_GROUP_MAPPING = {
    "0-4": [str(i) for i in range(5)],
    "5-17": [str(i) for i in range(5, 18)],
    "18-49": [str(i) for i in range(18, 50)],
    "50-64": [str(i) for i in range(50, 65)],
    "65+": [str(i) for i in range(65, 84)] + ["84+"],
}


# =============================================================================
# Factory Functions
# =============================================================================


def make_sir_config(
    population_name="US-CA",
    beta=0.5,
    gamma=0.1,
    initial_infected=10,
    start_date=date(2024, 1, 1),
    end_date=date(2024, 12, 31),
    delta_t=1.0,
    n_sims=10,
    resample_frequency="W-SAT",
    random_seed=None,
) -> BaseEpiModel:
    """Create a minimal SIR BaseEpiModel configuration for testing.

    Parameters
    ----------
    population_name : str
        Population ISO code (e.g., "US-CA").
    beta : float
        Transmission rate.
    gamma : float
        Recovery rate.
    initial_infected : int
        Number of initial infected individuals.
    start_date : date
        Simulation start date.
    end_date : date
        Simulation end date.
    delta_t : float
        Time step size.
    n_sims : int
        Number of simulations.
    resample_frequency : str
        Resampling frequency.
    random_seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    BaseEpiModel
        A minimal SIR model configuration.
    """
    compartments = [
        Compartment(id="S", label="Susceptible", init="default"),
        Compartment(id="I", label="Infected", init=initial_infected),
        Compartment(id="R", label="Recovered", init=0),
    ]

    transitions = [
        Transition(source="S", target="I", type="mediated", rate="beta", mediator="I"),
        Transition(source="I", target="R", type="spontaneous", rate="gamma"),
    ]

    parameters = {
        "beta": Parameter(type="scalar", value=beta),
        "gamma": Parameter(type="scalar", value=gamma),
    }

    population = Population(name=population_name, age_groups=AGE_GROUPS)
    timespan = Timespan(start_date=start_date, end_date=end_date, delta_t=delta_t)
    simulation = Simulation(n_sims=n_sims, resample_frequency=resample_frequency)

    return BaseEpiModel(
        name="test_model",
        compartments=compartments,
        transitions=transitions,
        parameters=parameters,
        population=population,
        timespan=timespan,
        simulation=simulation,
        random_seed=random_seed,
    )


def create_builder_output(**kwargs):
    """Create a BuilderOutput bypassing Pydantic validation for testing with mocks."""
    return BuilderOutput.model_construct(**kwargs)
