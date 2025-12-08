## udf_examples.py
# Dummy examples for User-Defined Functions
import copy
from typing import Dict

from epydemix.calibration import rmse
from epydemix.model.simulation_output import Trajectory


def udf_distance(data: Dict, simulation: Dict):
    """Dummy wrapper for epydemix rmse."""
    return rmse(data, simulation)


def udf_transform(trajectory: Trajectory, context: dict | None = None):
    """
    Dummy post-hoc transformation function that adds new compartments and transitions.

    Parameters
    ----------
    trajectory : Trajectory
        Simulation results from epydemix.simulate()
    context : dict | None, optional
        Simulation context containing:
        - params: dict of calibrated parameters (e.g., beta, gamma, start_date)
        - basemodel: BaseEpiModel configuration
        - timespan: Timespan with actual simulation dates
        - observed_data: DataFrame of observed data
        - intervention_types: list of intervention type strings
        - projection: bool indicating calibration vs projection mode
        - location: str location/population name

    Returns
    -------
    Trajectory
        Modified trajectory with additional compartments and transitions

    Examples
    --------
    Adding new compartments:
        >>> traj.compartments["Combined"] = traj.compartments["S_total"] + traj.compartments["I_total"]

    Adding new transitions:
        >>> traj.transitions["Total_flow"] = traj.transitions["S_to_I_total"] + traj.transitions["I_to_R_total"]

    Access calibrated parameters:
        >>> if context:
        ...     beta = context['params'].get('beta', 0)
        ...     if beta > 0.5:
        ...         # High transmission - apply different transformation
        ...         pass

    Location-specific transformations:
        >>> if context and context['location'] == 'United_States_California':
        ...     # California-specific transformation
        ...     pass

    Mode-dependent transformations:
        >>> if context and context['projection']:
        ...     # Only add extra outputs in projection mode
        ...     pass
    """
    traj = copy.deepcopy(trajectory)

    # Add new compartment (sum of existing compartments)
    traj.compartments["Dummy_S+L"] = traj.compartments["S_total"] + traj.compartments["L_total"]

    # Add new transition (you can combine or derive from existing transitions)
    # Note: Only add if the source transitions exist
    if "S_to_I_total" in traj.transitions and "I_to_R_total" in traj.transitions:
        traj.transitions["Dummy_total_flow"] = traj.transitions["S_to_I_total"] + traj.transitions["I_to_R_total"]

    # Example: Use context if provided
    if context:
        # Access calibrated parameters
        beta = context["params"].get("beta", 0)

        # Conditional logic based on parameter values
        if beta > 0.5:
            # High transmission - could add extra compartment or modify behavior
            pass

        # Access location
        location = context["location"]

        # Check mode
        if context["projection"]:
            # Projection-specific transformation
            pass

    return traj
