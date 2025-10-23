"""Vaccination building functions for EpiModel instances."""

import logging

import pandas as pd
from epydemix.model import EpiModel
from pandas import DataFrame

from ..schema.basemodel import Timespan, Transition, Vaccination
from ..vaccinations import add_vaccination_schedule, make_vaccination_probability_function, scenario_to_epydemix

logger = logging.getLogger(__name__)


def add_vaccination_schedules_from_config(
    model: EpiModel,
    transitions: list[Transition],
    vaccination: Vaccination,
    timespan: Timespan,
    use_schedule: DataFrame | None = None,
) -> EpiModel:
    """
    Add transitions between compartments due to vaccination to the EpiModel instance.

    Parameters
    ----------
        model: The EpiModel instance to which vaccination schedules will be added.
        transitions: List of Transition objects, including vaccination transitions.
        vaccination: Vaccination configuration object.
        timespan: Timespan configuration object with simulation dates.
        use_schedule: Optional pre-loaded vaccination schedule DataFrame.

    Returns
    -------
        EpiModel instance with vaccination schedules added.
    """
    # Extract compartment transitions due to vaccination
    vaccination_transitions = [transition for transition in transitions if transition.type == "vaccination"]

    # Define vaccine probability function
    vaccine_probability_function = make_vaccination_probability_function(
        vaccination.origin_compartment, vaccination.eligible_compartments
    )

    # Ignore provided data path in vaccination input if use_schedule is provided
    if use_schedule is not None:
        vaccination_schedule = use_schedule
    # Preprocessed vaccination schedule
    elif vaccination.preprocessed_vaccination_data_path:
        vaccination_schedule = pd.read_csv(vaccination.preprocessed_vaccination_data_path)
        logger.info(f"Loaded preprocessed vaccination schedule from {vaccination.preprocessed_vaccination_data_path}")
    # Create schedule from SMH scenario
    else:
        try:
            vaccination_schedule = scenario_to_epydemix(
                input_filepath=vaccination.scenario_data_path,
                start_date=timespan.start_date,
                end_date=timespan.end_date,
                target_age_groups=model.population.Nk_names,
                delta_t=timespan.delta_t,
                states=[model.population.name],
            )
            logger.info(f"Created vaccination schedule from scenario data at {vaccination.scenario_data_path}")
        except Exception as e:
            raise ValueError(f"Error creating vaccination schedule from scenario data:\n{e}")

    # Add vaccine transitions to the model
    for transition in vaccination_transitions:
        try:
            model = add_vaccination_schedule(
                model=model,
                vaccine_probability_function=vaccine_probability_function,
                source_comp=transition.source,
                target_comp=transition.target,
                vaccination_schedule=vaccination_schedule,
            )
            logger.info(f"Added vaccination transition: {transition.source} -> {transition.target}")
        except Exception as e:
            raise ValueError(f"Error adding vaccination transition {transition}: {e}")

    return model
