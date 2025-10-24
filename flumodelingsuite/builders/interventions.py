"""Intervention building functions for EpiModel instances."""

import copy
import logging

import numpy as np
from epydemix.model import EpiModel

from ..schema.basemodel import Intervention, Timespan
from ..school_closures import add_school_closure_interventions
from ..seasonality import get_scaled_parameter

logger = logging.getLogger(__name__)


def add_school_closure_intervention_from_config(
    model: EpiModel, interventions: list[Intervention], closure_dict: dict
) -> EpiModel:
    """
    Apply a school closure intervention to the EpiModel instance.

    Parameters
    ----------
        model: The EpiModel instance to which the intervention will be applied.
        interventions: List of Intervention objects.
        closure_dict: Dictionary containing school closure data.

    Returns
    -------
        EpiModel instance with the intervention applied.
    """
    # Extract school_closure intervention
    # Validator enforces only 1 school_closure intervention, so can just take index here
    try:
        intervention = interventions[[i.type for i in interventions].index("school_closure")]
    except ValueError:  # ValueError thrown by index() if there is no school_closure intervention
        return model

    # Apply the intervention
    try:
        model = add_school_closure_interventions(
            model=model, closure_dict=closure_dict, reduction_factor=intervention.scaling_factor
        )
        logger.info(f"Applied school closure intervention with reduction factor: {intervention.scaling_factor}")
    except Exception as e:
        raise ValueError(f"Error applying school closure intervention {intervention}:\n{e}")

    return model


def add_contact_matrix_interventions_from_config(model: EpiModel, interventions: list[Intervention]) -> EpiModel:
    """
    Apply contact matrix interventions.

    Parameters
    ----------
        model: The EpiModel instance to which the intervention will be applied.
        interventions: List of Intervention objects.

    Returns
    -------
        EpiModel instance with contact matrix interventions applied.
    """
    model = copy.deepcopy(model)

    # Extract interventions
    cm_invs = [i for i in interventions if i.type == "contact_matrix"]

    for i in cm_invs:
        # Ensure layer is present
        assert i.contact_matrix_layer in model.population.layers, (
            f"Contact matrix intervention cannot use layer '{i.contact_matrix_layer}'. Available layers are {model.population.layers}."
        )

        # Apply the intervention
        try:
            model.add_intervention(
                layer_name=i.contact_matrix_layer,
                start_date=i.start_date,
                end_date=i.end_date,
                reduction_factor=i.scaling_factor,
            )
            logger.info(
                f"Applied contact matrix intervention to layer '{i.contact_matrix_layer}' with scaling factor {i.scaling_factor}."
            )
        except Exception as e:
            raise ValueError(f"Error applying contact matrix intervention {i}:\n{e}")

    return model


def add_parameter_interventions_from_config(
    model: EpiModel, interventions: list[Intervention], timespan: Timespan
) -> EpiModel:
    """
    Apply parameter interventions to the EpiModel instance.

    Handles both scaling factor interventions and parameter override interventions.
    Override interventions are applied last to ensure they are the final parameter values.

    Parameters
    ----------
        model: The EpiModel instance to apply interventions to.
        interventions: List of Intervention objects.
        timespan: Timespan configuration object with simulation dates.

    Returns
    -------
        EpiModel instance with parameter interventions applied.
    """
    model = copy.deepcopy(model)

    # Extract parameter interventions
    param_invs = [i for i in interventions if i.type == "parameter"]

    # Apply scaling interventions
    for i in [inv for inv in param_invs if inv.scaling_factor]:
        # Target parameter must already exist
        try:
            previous_value = model.get_parameter(i.target_parameter)
        except KeyError:
            raise ValueError(
                f"Attempted to apply scaling factor parameter intervention to undefined parameter {i.target_parameter}"
            )

        # Calculate rescaling vector
        dates, st = get_scaled_parameter(
            date_start=timespan.start_date,
            date_stop=timespan.end_date,
            scaling_start=i.start_date,
            scaling_stop=i.end_date,
            scaling_factor=i.scaling_factor,
            delta_t=timespan.delta_t,
        )

        # Handle possibilities for previous parameter value (expressions should already be evaluated at parameter definition)
        T = len(st)
        N = model.population.num_groups
        # If existing parameter is constant, transform to array of size (T, 1) with time-varying values
        # If existing parameter is time-varying (array of size (T, 1)), do piecewise multiplication
        if (not hasattr(previous_value, "__len__")) or previous_value.shape == (T,):
            new_value = np.array(st) * np.array(previous_value)
        # If existing parameter is age-varying (array of size (1, N)), transform to array of size (T, N) with time-varying and age-varying values
        # If existing parameter is time-varying and age-varying (array of size (T, N)), do piecewise for each age group
        elif previous_value.shape == (T, N) or previous_value.shape == (1, N):
            new_value = np.zeros((T, N))
            for i in range(N):
                new_value[:, i] = np.array(st) * np.array(previous_value[:, i])
        # Uncertain how this will work for priors
        else:
            raise ValueError(
                f"Cannot apply scaling intervention to existing parameter {i.target_parameter} = {previous_value}"
            )

        # Overwrite parameter with new scaled values
        try:
            model.add_parameter(i.target_parameter, new_value)
            logger.info(f"Added scaling intervention to parameter {i.target_parameter}")
        except Exception as e:
            raise ValueError(f"Error adding parameter scaling intervention to model: {e}")

    # Apply override interventions.
    # This must occur at the end to ensure override values are final parameter values.
    for i in [inv for inv in param_invs if inv.override_value]:
        try:
            model.override_parameter(
                start_date=i.start_date, end_date=i.end_date, parameter_name=i.target_parameter, value=i.override_value
            )
            logger.info(f"Added override intervention to parameter {i.target_parameter}")
        except Exception as e:
            raise ValueError(f"Error adding parameter override intervention to model: {e}")

    return model
