"""Core model building functions for EpiModel instances."""

import logging
from dataclasses import dataclass, replace
from functools import partial

import numpy as np
import pandas as pd
from epydemix.model import EpiModel
from epydemix.population import Population
from epydemix.utils import convert_to_2Darray

from ..schema.basemodel import Compartment, Parameter, Transition
from ..schema.basemodel import Population as PopulationConfig
from ..utils import convert_location_name_format, load_epydemix_population
from ..utils.custom_locations import CustomPopulation, load_custom_locations, resolve_epydemix_name
from ..utils.expression_eval import resolve_model_name, safe_eval
from ..utils.location import (
    get_metrocast_population_data,
    resolve_location,
)
from ..utils.populations import aggregate_population_by_age_groups
from ..utils.resolved_location import ResolvedLocation, VaccinationContext

logger = logging.getLogger(__name__)


@dataclass
class LocatedModel:
    """Keep a model's regional identity attached through filtering and sampling."""

    model: EpiModel
    location: ResolvedLocation
    vaccination: VaccinationContext | None = None

    def __post_init__(self):
        if not self.location.matches(self.model.population.name):
            raise ValueError("Resolved location must match the model population")
        if self.location.total_population is None:
            self.location = replace(self.location, total_population=float(self.model.population.Nk.sum()))


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


def load_iso_population(
    location_name: str | ResolvedLocation,
    age_groups: list[str],
    contact_matrix_override: str | None = None,
) -> Population:
    """
    Load population for ISO 3166 location using epydemix data.

    Parameters
    ----------
        location_name: ISO 3166 location code (e.g., "US-MA")
        age_groups: List of age group strings to map
        contact_matrix_override: Optional ISO code to use different contact matrix

    Returns
    -------
        epydemix Population object
    """

    location = resolve_location(location_name, "iso")
    age_group_mapping = {group: _parse_age_group(group) for group in age_groups}
    population = load_epydemix_population(
        population_name=location.population_name,
        age_group_mapping=age_group_mapping,
    )

    # Keep location_name's population counts; replace only the contact matrices
    if contact_matrix_override and contact_matrix_override != location.location_id:
        cm_population = load_epydemix_population(
            population_name=convert_location_name_format(contact_matrix_override, "epydemix_population"),
            age_group_mapping=age_group_mapping,
        )
        population.contact_matrices = cm_population.contact_matrices

    return population


def load_metrocast_population(
    location_name: str | ResolvedLocation,
    age_groups: list[str],
    contact_matrix_override: str | None = None,
) -> Population:
    """
    Load population for metrocast location.

    For sub-state metrocast locations (HSAs, NC flu regions), loads population from metrocast_population.csv with parent state's contact matrix.

    For state-level metrocast locations (e.g., "colorado", "georgia"), loads population from epydemix using the state's ISO code but maintains metrocast naming convention for output formatting.

    Parameters
    ----------
        location_name: Metrocast location name (e.g., "denver", "nenc", "colorado")
        age_groups: List of age group strings to map
        contact_matrix_override: Optional ISO code to use different contact matrix

    Returns
    -------
        epydemix Population object with metrocast naming
    """
    location = resolve_location(location_name, "metrocast_location")
    if location.format("original_location_code") == "All":
        parent = load_iso_population(location.source_iso, age_groups, contact_matrix_override=contact_matrix_override)
        parent.name = location.population_name
        return parent

    age_data = get_metrocast_population_data()
    location_data = age_data[age_data["metrocast_location_id"] == location.location_id]
    if location_data.empty:
        raise ValueError(f"No population data found for metrocast location: {location.location_id}")
    counts = aggregate_population_by_age_groups(location_data, age_groups)
    parent = load_iso_population(contact_matrix_override or location.source_iso, age_groups)
    population = Population(name=location.population_name)
    population.add_population(Nk=np.array([counts[group] for group in age_groups], dtype=float), Nk_names=age_groups)
    population.contact_matrices = parent.contact_matrices
    return population


def load_custom_population(
    custom_population: CustomPopulation, age_groups: list[str], contact_matrix_override: str | None = None
) -> Population:
    """Build custom age counts with a parent or explicitly selected contact matrix."""
    age_data = pd.DataFrame(
        {
            "age": [str(age) for age in range(84)] + ["84+"],
            "population": custom_population.population_by_age,
        }
    )
    counts = aggregate_population_by_age_groups(age_data, age_groups)
    source = custom_population.parent_population_name
    if contact_matrix_override:
        source = resolve_epydemix_name(contact_matrix_override)
    parent = load_epydemix_population(
        source, age_group_mapping={group: _parse_age_group(group) for group in age_groups}
    )
    population = Population(name=custom_population.location.population_name)
    population.add_population(Nk=np.array([counts[group] for group in age_groups], dtype=float), Nk_names=age_groups)
    population.contact_matrices = parent.contact_matrices
    return population


def set_population_from_config(
    model: EpiModel,
    population_config: PopulationConfig,
    location: ResolvedLocation | None = None,
    custom_populations: dict[str, CustomPopulation] | None = None,
) -> EpiModel:
    """Load demographics for a canonical region and bind them to an epydemix model."""
    if location is None:
        if custom_populations is None and population_config.locations_path is not None:
            from types import SimpleNamespace

            custom_populations = load_custom_locations(SimpleNamespace(population=population_config))
        location = resolve_location(population_config.name, population_config.location_type, custom_populations)
    age_groups = population_config.age_groups
    contact_matrix = population_config.contact_matrix
    try:
        if location.kind == "iso":
            population = load_iso_population(location, age_groups, contact_matrix)
        elif location.kind == "metrocast_location":
            population = load_metrocast_population(location, age_groups, contact_matrix)
        else:
            definition = (custom_populations or {}).get(location.location_id)
            if definition is None or definition.location != location:
                raise ValueError(f"Matching custom population data required for {location.location_id}")
            population = load_custom_population(definition, age_groups, contact_matrix)
        population.name = location.population_name
        model.set_population(population)
        logger.info("Model population set to: %s (type: %s)", location.location_id, location.kind)
    except Exception as exc:
        raise ValueError(f"Error setting population for {location.location_id}: {exc}") from exc
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
        The same EpiModel instance with compartments added (modified in-place).
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
        The same EpiModel instance with compartment transitions added (modified in-place).
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
        The same EpiModel instance with parameters added (modified in-place).
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
        if parameters_dict:
            model.add_parameter(parameters_dict=parameters_dict)
            logger.info(f"Added parameters: {list(parameters_dict.keys())}")
        else:
            logger.info("No scalar/age_varying parameters to add (all parameters are calibrated/sampled)")

        return model
    except Exception as e:
        raise ValueError(f"Error adding parameters to model: {e}")


def calculate_parameters_from_config(
    model: EpiModel, parameters: dict[str, Parameter], compartment_init: dict[str, np.ndarray] | None
) -> EpiModel:
    """
    Add calculated parameters to the EpiModel, assuming all non-calculated parameters are already in the model.

    Parameters
    ----------
    model: EpiModel
            The EpiModel instance to which calculated parameters will be added.
    parameters: dict[str, Parameter]
            Dictionary mapping parameter names to Parameter objects.
    compartment_init: dict[str, np.ndarray] | None
            Dictionary mapping compartment names to initial condition arrays,
            or None if no initial conditions are specified.

    Returns
    -------
    EpiModel
            EpiModel instance with calculated parameters added.
    """
    # Extract parameter names and expressions
    calc_params = {name: param.value for name, param in parameters.items() if param.type == "calculated"}

    # Build a dictionary of calculated values
    for name, expr in calc_params.items():
        parameter_dict = {}
        logger.info(f"Calculating parameter {name} using expression: {expr}")
        try:
            value = safe_eval(expr, partial(resolve_model_name, model, compartment_init))
            # Age-varying results use the same 2D shape as configured age_varying parameters
            parameter_dict[name] = convert_to_2Darray(value) if np.ndim(value) == 1 else value
            logger.info(f"Calculated parameter {name}: {parameter_dict[name]}")
        except Exception as e:
            raise ValueError(f"Error calculating parameter {name}: {e}")

        try:
            model.add_parameter(parameters_dict=parameter_dict)
            logger.info(f"Added parameter: {list(parameter_dict.keys())}")
        except Exception as e:
            raise ValueError(f"Error adding parameters to model: {e}")

    return model


def _allocate_integer_counts(total: float, weights: np.ndarray) -> np.ndarray:
    """
    Split a total count into integers proportional to weights, preserving the rounded total.

    Uses the largest remainder method: each group receives the floor of its proportional share,
    and the leftover units go to the groups with the largest fractional remainders.

    Parameters
    ----------
    total : float
        Total count to allocate. Rounded to the nearest integer before allocation.
    weights : np.ndarray
        Non-negative weights (e.g., population by age group).

    Returns
    -------
    np.ndarray
        Integer counts per group summing to ``round(total)``.
    """
    total_int = int(np.rint(total))
    shares = total_int * weights / weights.sum()
    counts = np.floor(shares).astype(np.int64)
    leftover = total_int - counts.sum()
    if leftover > 0:
        largest_remainders = np.argsort(-(shares - counts), kind="stable")[:leftover]
        counts[largest_remainders] += 1
    return counts


def calculate_compartment_initial_conditions(
    compartments: list,
    population_array: np.ndarray,
    params_dict: dict | None = None,
) -> dict[str, np.ndarray] | None:
    """
    Calculate initial conditions for compartments based on their initialization values.

    This function handles four types of compartment initialization:
    1. Age-varying (list): Each value applied to corresponding age group
       - Values >= 1: Counts applied directly to that age group
       - Values 0 < v < 1: Proportions applied to that age group's population
    2. Scalar counts (value >= 1): Distributed proportionally across age groups
    3. Scalar proportions (0 < value < 1): Applied to all age groups
    4. Default: Remaining population distributed per age group

    Age-specific counts and proportions are assigned first. Scalar counts are then
    limited to the remaining population in each age group, redistributing any excess
    to groups with available capacity while preserving each compartment's total.

    Parameters
    ----------
    compartments : list
        List of Compartment objects from the configuration.
    population_array : np.ndarray
        Array of population counts by age group.
    params_dict : dict | None, optional
        Dictionary containing sampled/calibrated values. Only compartment names are extracted;
        other keys are ignored (overrides config values for matching compartment names).

    Returns
    -------
    dict[str, np.ndarray] | None
        Dictionary mapping compartment names to initial condition arrays,
        or None if no initial conditions are specified.

    Raises
    ------
    ValueError
        If initialization logic produces invalid values.

    Examples
    --------
    >>> compartments = [
    ...     Compartment(id="S", init="default"),
    ...     Compartment(id="I", init=100),
    ...     Compartment(id="R", init=0.1),
    ...     Compartment(id="M", init=[0.3, 0, 0])  # 30% of first age group only
    ... ]
    >>> population = np.array([1000, 2000, 3000])
    >>> calculate_compartment_initial_conditions(compartments, population)
    {'S': array([...]), 'I': array([...]), 'R': array([...]), 'M': array([300, 0, 0])}
    """
    # Initialize tracking variables
    default_compartment_ids = []
    initial_conditions_dict = {}
    scalar_counts = {}
    remaining_population = np.rint(population_array).astype(np.int64)

    # First pass: identify default compartments
    for compartment in compartments:
        if compartment.init == "default":
            default_compartment_ids.append(compartment.id)

    # Second pass: calculate non-default compartment initial conditions
    for compartment in compartments:
        # Skip default compartment for now
        if compartment.init == "default":
            continue

        # Get initialization value: use params_dict value if available, otherwise use config value
        initial_value = params_dict.get(compartment.id, compartment.init) if params_dict else compartment.init

        # Skip compartments with no initial value
        if initial_value is None:
            continue

        # Skip sampled/calibrated compartments that don't have values in params_dict yet
        # (if initial_value is still the enum, it means no numeric value was provided)
        if isinstance(initial_value, Compartment.InitCompartmentEnum) and initial_value in (
            Compartment.InitCompartmentEnum.sampled,
            Compartment.InitCompartmentEnum.calibrated,
        ):
            continue

        # Case 1: Age-varying initialization (list)
        if isinstance(initial_value, list):
            initial_conditions = np.zeros_like(population_array, dtype=np.int64)
            for age_idx, val in enumerate(initial_value):
                if val >= 1:
                    # Count for this age group (applied directly)
                    initial_conditions[age_idx] = np.rint(val)
                elif 0 < val < 1:
                    # Proportion for this age group
                    initial_conditions[age_idx] = np.rint(population_array[age_idx] * val)
                elif val == 0:
                    initial_conditions[age_idx] = 0
                else:
                    raise ValueError(
                        f"Invalid initial value {val} at age group {age_idx} for compartment {compartment.id}"
                    )
            initial_conditions_dict[compartment.id] = initial_conditions
            remaining_population -= initial_conditions

        # Case 2: Scalar count-based initialization (value >= 1)
        elif initial_value >= 1:
            # Reserve age-specific values before distributing counts across age groups.
            scalar_counts[compartment.id] = int(np.rint(initial_value))

        # Case 3: Scalar proportion-based initialization (0 < value < 1)
        elif 0 < initial_value < 1:
            # Apply proportion to each age group
            initial_conditions = np.rint(population_array * initial_value).astype(np.int64)
            initial_conditions_dict[compartment.id] = initial_conditions
            remaining_population -= initial_conditions

        # Case 4: Zero initialization
        elif initial_value == 0:
            initial_conditions_dict[compartment.id] = np.zeros_like(population_array, dtype=np.int64)

        else:
            raise ValueError(f"Invalid initial value for compartment {compartment.id}: {initial_value}")

    if np.any(remaining_population < 0) or sum(scalar_counts.values()) > remaining_population.sum():
        raise ValueError(f"Initial conditions exceed population. Remaining population: {remaining_population}")

    for compartment_id, total in scalar_counts.items():
        counts = np.minimum(_allocate_integer_counts(total, population_array), remaining_population)
        overflow = total - counts.sum()
        if overflow:
            # Reallocate only the excess, using the unoccupied population as capacity.
            counts += _allocate_integer_counts(overflow, remaining_population - counts)
        initial_conditions_dict[compartment_id] = counts
        remaining_population -= counts

    # Third pass: assign remaining population to default compartment(s)
    if default_compartment_ids:
        # If multiple default compartments, split remaining population equally,
        # giving any leftover individuals to the earlier default compartments
        num_defaults = len(default_compartment_ids)
        per_default, leftover = np.divmod(remaining_population, num_defaults)

        for default_idx, compartment_id in enumerate(default_compartment_ids):
            initial_conditions_dict[compartment_id] = per_default + (leftover > default_idx)

    return initial_conditions_dict or None
