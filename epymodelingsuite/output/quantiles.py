"""Variable selection for output quantiles."""

import numpy as np
import pandas as pd
from epydemix.model.simulation_results import SimulationResults

from ..schema.output import QuantilesOutput
from ..utils.quantiles import compute_quantiles


def projection_quantile_variables(projections: list[dict], config: QuantilesOutput) -> list[str] | None:
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
    numeric_names = [
        name for name, values in projections[0].items() if np.issubdtype(np.asarray(values).dtype, np.number)
    ]
    selections = (config.compartments, config.transitions)
    if any(
        isinstance(selected, list) and any(name not in numeric_names for name in selected) for selected in selections
    ):
        return None
    return [
        name
        for name in numeric_names
        if (config.compartments is True and "_to_" not in name)
        or (config.transitions is True and "_to_" in name)
        or any(isinstance(selected, list) and name in selected for selected in selections)
    ] or ["date"]  # An empty variables list means "all variables" in epydemix.


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
    available = getattr(results.trajectories[0], kind) if results.trajectories else {}
    variables = selection if isinstance(selection, list) and all(v in available for v in selection) else None
    trajectories = getattr(results, f"get_stacked_{kind}")(variables=variables)
    return compute_quantiles(trajectories, results.dates, quantiles, ignore_nan=True)
