"""Core model building functions for EpiModel instances."""

import logging

import numpy as np
from epydemix.model import EpiModel
from epydemix.population import load_epydemix_population
from epydemix.utils import convert_to_2Darray

from ..schema.basemodel import Compartment, Parameter, Transition
from ..utils import convert_location_name_format
from ..utils.expression_eval import RetrieveName, SafeEvalVisitor, safe_eval

logger = logging.getLogger(__name__)


def _parse_age_group(group_str: str) -> list:
    """
    Parse an age group string like "0-4", "65+" into a list of individual age labels.
    For "a-b", returns [str(a), str(a+1), ..., str(b)].
    For "c+", returns [str(c), ..., "84", "84+"].

    Parameters
    ----------
        group_str: Age group string to parse.

    Returns
    -------
        List of individual age labels as strings.
    """
    if group_str.endswith("+"):
        # e.g. "65+" -> start=65, end at 84 then add "84+"
        start = int(group_str[:-1])
        end = 84
        labels = [str(i) for i in range(start, end)] + [f"{end}+"]
    else:
        # e.g. "5-17" -> start=5, end=17
        start, end = map(int, group_str.split("-"))
        labels = [str(i) for i in range(start, end + 1)]
    return labels


def set_population_from_config(model: EpiModel, population_name: str, age_groups: list[str]) -> EpiModel:
    """
    Set the population for the EpiModel instance.

    Parameters
    ----------
        model: The EpiModel instance for which the population will be set.
        population_name: Name of the population to load.
        age_groups: List of age group strings to map.

    Returns
    -------
        EpiModel instance with the population set.
    """
    try:
        # Convert to "epydemix_population" name
        population_name = convert_location_name_format(population_name, "epydemix_population")

        # Create age group mapping
        age_group_mapping = {group: _parse_age_group(group) for group in age_groups}
        population = load_epydemix_population(population_name=population_name, age_group_mapping=age_group_mapping)
        model.set_population(population)
        logger.info(f"Model population set to: {population_name}")
    except Exception as e:
        raise ValueError(f"Error setting population: {e}")
    return model


def add_model_compartments_from_config(model: EpiModel, compartments: list[Compartment]) -> EpiModel:
    """
    Add compartments to the EpiModel instance.

    Parameters
    ----------
        model: The EpiModel instance to which compartments will be added.
        compartments: List of Compartment objects containing compartment definitions.

    Returns
    -------
        EpiModel instance with compartments added.
    """
    # Add compartments to the model
    try:
        compartment_ids = [compartment.id for compartment in compartments]
        model.add_compartments(compartment_ids)
        logger.info(f"Added compartments: {compartment_ids}")
    except Exception as e:
        raise ValueError(f"Error adding compartments: {e}")

    return model


def add_model_transitions_from_config(model: EpiModel, transitions: list[Transition]) -> EpiModel:
    """
    Add transitions between compartments to the EpiModel instance.

    Parameters
    ----------
        model: The EpiModel instance to which compartment transitions will be added.
        transitions: List of Transition objects defining transitions.

    Returns
    -------
        EpiModel instance with compartment transitions added.
    """
    # Check that required attributes of model configuration are not None
    if transitions is None:
        return model

    # Add transitions to the model
    for transition in transitions:
        if transition.type == "mediated":
            try:
                model.add_transition(
                    transition.source,
                    transition.target,
                    params=(transition.rate, transition.mediator),
                    kind=transition.type,
                )
                logger.info(
                    f"Added mediated transition: {transition.source} -> {transition.target} (mediator: {transition.mediator}, rate: {transition.rate})"
                )
            except Exception as e:
                raise ValueError(f"Error adding mediated transition {transition}: {e}")
        elif transition.type == "spontaneous":
            try:
                model.add_transition(transition.source, transition.target, params=transition.rate, kind=transition.type)
                logger.info(
                    f"Added spontaneous transition: {transition.source} -> {transition.target} (rate: {transition.rate})"
                )
            except Exception as e:
                raise ValueError(f"Error adding spontaneous transition {transition}: {e}")

    return model


def add_model_parameters_from_config(model: EpiModel, parameters: dict[str, Parameter]) -> EpiModel:
    """
    Add parameters to the EpiModel instance.

    Parameters
    ----------
        model: The EpiModel instance to which parameters will be added.
        parameters: Dictionary mapping parameter names to Parameter objects.

    Returns
    -------
        EpiModel instance with parameters added.
    """
    # Add parameters to the model
    parameters_dict = {}
    for key, data in parameters.items():
        if data.type == "scalar":
            if type(data.value) is str:
                parameters_dict[key] = safe_eval(data.value)
            else:
                parameters_dict[key] = data.value
        elif data.type == "age_varying":  # Ensure array matches population age structure
            if model.population.num_groups == len(data.values):
                parameters_dict[key] = convert_to_2Darray(
                    [safe_eval(val) if type(val) is str else val for val in data.values]
                )
            else:
                raise ValueError(
                    f"Array values supplied for parameter {key} do not match model population age structure"
                )
        elif data.type in ["sampled", "calibrated", "calculated"]:
            # Skip parameters without values.
            # They will be set later during calibration/sampling or calculated after all other parameters are defined
            pass

    try:
        model.add_parameter(parameters_dict=parameters_dict)
        logger.info(f"Added parameters: {list(parameters_dict.keys())}")

        return model
    except Exception as e:
        raise ValueError(f"Error adding parameters to model: {e}")


def calculate_parameters_from_config(model: EpiModel, parameters: dict[str, Parameter]) -> EpiModel:
    """
    Add calculated parameters to the EpiModel, assuming all non-calculated parameters are already in the model.

    Parameters
    ----------
        model: The EpiModel instance to which calculated parameters will be added.
        parameters: Dictionary mapping parameter names to Parameter objects.

    Returns
    -------
        EpiModel instance with calculated parameters added.
    """
    # Extract parameter names and expressions
    calc_params = {name: param.value for name, param in parameters.items() if param.type == "calculated"}

    # Build a dictionary of calculated values
    parameters_dict = {}
    for name, expr in calc_params.items():
        logger.info(f"Calculating parameter {name} using expression: {expr}")
        try:
            # Parse the expression into a tree
            tree = ast.parse(expr, mode="eval")

            # Substitute retrieved parameter values or contact matrix eigenvalue into the tree
            RetrieveName(model).visit(tree)

            # Validate the expression
            SafeEvalVisitor().visit(tree)

            # Evaluate the expression
            code = compile(tree, filename="<calc_eval>", mode="eval")
            parameters_dict[name] = eval(code, {"__builtins__": None, "np": np, "scipy": scipy}, {})
            logger.info(f"Calculated parameter {name}: {parameters_dict[name]}")
        except Exception as e:
            raise ValueError(f"Error calculating parameter {name}: {e}")

    try:
        model.add_parameter(parameters_dict=parameters_dict)
        logger.info(f"Calculated parameters: {list(parameters_dict.keys())}")

        return model
    except Exception as e:
        raise ValueError(f"Error adding parameters to model: {e}")


def calculate_compartment_initial_conditions(
    compartments: list,
    population_array: np.ndarray,
    sampled_compartments: dict | None = None,
) -> dict | None:
    """
    Calculate initial conditions for compartments based on their initialization values.

    This function handles three types of compartment initialization:
    1. Counts (value >= 1): Distributed proportionally across age groups
    2. Proportions (value < 1): Applied directly to population by age group
    3. Default: Remaining population distributed per age group

    Parameters
    ----------
    compartments : list
        List of Compartment objects from the configuration.
    population_array : np.ndarray
        Array of population counts by age group.
    sampled_compartments : dict | None, optional
        Dictionary of compartment names to sampled values (overrides config values).

    Returns
    -------
    dict | None
        Dictionary mapping compartment names to initial condition arrays,
        or None if no initial conditions are specified.

    Raises
    ------
    ValueError
        If multiple compartments are marked as default, or if initialization
        logic produces invalid values.

    Examples
    --------
    >>> compartments = [
    ...     Compartment(id="S", init="default"),
    ...     Compartment(id="I", init=100),
    ...     Compartment(id="R", init=0.1)
    ... ]
    >>> population = np.array([1000, 2000, 3000])
    >>> calculate_compartment_initial_conditions(compartments, population)
    {'S': array([...]), 'I': array([...]), 'R': array([...])}
    """
    # Initialize tracking variables
    default_compartment_ids = []
    initial_conditions_dict = {}
    remaining_population = population_array.astype(float)

    # First pass: identify default compartments
    for compartment in compartments:
        if compartment.init == "default":
            default_compartment_ids.append(compartment.id)

    # Second pass: calculate non-default compartment initial conditions
    for compartment in compartments:
        # Skip default compartment for now
        if compartment.init == "default":
            continue

        # Get initialization value (from samples if provided, otherwise from config)
        initial_value = (
            sampled_compartments.get(compartment.id, compartment.init)
            if sampled_compartments
            else compartment.init
        )

        # Skip compartments with no initial value
        if initial_value is None:
            continue

        # Case 1: Count-based initialization (value >= 1)
        if initial_value >= 1:
            # Distribute total count proportionally across age groups
            initial_conditions = initial_value * population_array / population_array.sum()
            initial_conditions_dict[compartment.id] = initial_conditions
            remaining_population -= initial_conditions

        # Case 2: Proportion-based initialization (0 < value < 1)
        elif 0 < initial_value < 1:
            # Apply proportion to each age group
            initial_conditions = population_array * initial_value
            initial_conditions_dict[compartment.id] = initial_conditions
            remaining_population -= initial_conditions

        # Case 3: Zero initialization
        elif initial_value == 0:
            initial_conditions_dict[compartment.id] = np.zeros_like(population_array)

        else:
            raise ValueError(f"Invalid initial value for compartment {compartment.id}: {initial_value}")

    # Third pass: assign remaining population to default compartment(s)
    if default_compartment_ids:
        if np.any(remaining_population < 0):
            raise ValueError(
                f"Initial conditions exceed population in some age groups. Remaining population: {remaining_population}"
            )
        # If multiple default compartments, split remaining population equally
        num_defaults = len(default_compartment_ids)
        per_default = remaining_population / num_defaults

        for compartment_id in default_compartment_ids:
            initial_conditions_dict[compartment_id] = per_default

    return initial_conditions_dict if initial_conditions_dict else None
