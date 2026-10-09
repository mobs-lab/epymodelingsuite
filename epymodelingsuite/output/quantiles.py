"""Variable selection for output quantiles."""

import numpy as np
import pandas as pd
from epydemix.model.simulation_results import SimulationResults

from ..schema.output import QuantilesOutput
from ..utils.quantiles import compute_quantiles


def select_projection_quantile_variables(projections: list[dict], config: QuantilesOutput) -> list[str] | None:
    """Select projection variables before stacking, preserving fallback behavior.

    Parameters
    ----------
    projections : list of dict
        Raw projection draws; the first draw defines available numeric names.
    config : QuantilesOutput
        Compartment and transition selections.

    Returns
    -------
    list of str or None
        Selected numeric variables, ["date"] for an empty selection, or None for all variables.

    Notes
    -----
    Missing requested names retain the existing all-column fallback. An empty
    list means all variables in epydemix, so ["date"] represents no numeric variables.
    """
    if not projections:
        return None
    available_variables = [
        name for name, values in projections[0].items() if np.issubdtype(np.asarray(values).dtype, np.number)
    ]
    listed_variables = [
        name
        for selection in (config.compartments, config.transitions)
        if isinstance(selection, list)
        for name in selection
    ]
    if any(name not in available_variables for name in listed_variables):
        return None  # Keep the existing all-column fallback.

    requested = set(listed_variables)
    if config.compartments is True:
        # All compartments
        requested.update(name for name in available_variables if "_to_" not in name)
    if config.transitions is True:
        # All transitions
        requested.update(name for name in available_variables if "_to_" in name)
    selected_variables = [name for name in available_variables if name in requested]  # Keep projection order.
    return selected_variables or ["date"]  # An empty list means "all variables" in epydemix.


def get_simulation_quantiles(
    results: SimulationResults,
    kind: str,
    quantiles: list[float],
    selection: bool | list[str],
) -> pd.DataFrame:
    """Select simulation variables before stacking and compute output quantiles.

    Parameters
    ----------
    results : SimulationResults
        Raw simulation trajectories and date labels.
    kind : {'compartments', 'transitions'}
        Trajectory collection to summarize.
    quantiles : list of float
        Requested levels, preserving order and duplicates.
    selection : bool or list of str
        True selects all variables. A list selects its names only when all exist;
        missing names retain the existing all-column fallback.

    Returns
    -------
    pd.DataFrame
        Quantile-major rows with date, quantile and selected numeric variable columns.

    Notes
    -----
    NaNs are excluded using the existing output policy. Upstream stack and
    quantile validation errors propagate to the output generator.
    """
    first_trajectory = results.trajectories[0] if results.trajectories else None
    if first_trajectory is None:
        available_variables = {}
    elif kind == "compartments":
        available_variables = first_trajectory.compartments
    else:
        available_variables = first_trajectory.transitions
    variables = None  # epydemix stacks all variables when variables is None.
    if isinstance(selection, list) and all(name in available_variables for name in selection):
        variables = selection
    # Stack draws into one (draws, dates) array per variable.
    if kind == "compartments":
        trajectories = results.get_stacked_compartments(variables=variables)
    else:
        trajectories = results.get_stacked_transitions(variables=variables)
    return compute_quantiles(trajectories, results.dates, quantiles, ignore_nan=True)
