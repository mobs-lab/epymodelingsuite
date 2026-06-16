"""
User-defined function for COVID hospitalizations rescaling.
Computes rescaled hospitalizations from the sum of hosp transitions
multiplied by alpha and strain_severity.
"""

import copy
import logging

from epydemix.model.simulation_output import Trajectory

logger = logging.getLogger(__name__)


def covid_rescaling(
    trajectory: Trajectory, context: dict | None = None
) -> Trajectory:
    """
    Post-hoc transformation for COVID hospitalizations rescaling.

    Adds "hospitalizations" (unscaled sum of hosp transitions) and
    "hospitalizations_rescaled" (sum * alpha * strain_severity).

    Parameters
    ----------
    trajectory : Trajectory
        Simulation results from epydemix.simulate()
    context : dict | None, optional
        Simulation context containing calibrated parameters (alpha, strain_severity)

    Returns
    -------
    Trajectory
        Modified trajectory with "hospitalizations" and "hospitalizations_rescaled" transitions

    Examples
    --------
    Usage in modelset.yml::

        calibration:
            post_hoc_transformation:
                user_script_path: /path/to/functions/covid_rescaling.py
                user_function_name: covid_rescaling
    """
    traj = copy.deepcopy(trajectory)
    STRAIN_SEVERITY = 0.3

    # Get alpha and strain_severity from parameters
    if context and "params" in context:
        params = context["params"]
        if "alpha" in params:
            alpha = params["alpha"]
        else:
            logger.warning("alpha not found in params, using default 1.0")
            alpha = 1.0
        
    # Compute hospitalizations (unscaled) and hospitalizations_rescaled
    if (
        "Home_to_Hosp_total" in traj.transitions
        and "Home_vax_to_Hosp_vax_total" in traj.transitions
    ):
        hosp_sum = (
            traj.transitions["Home_to_Hosp_total"]
            + traj.transitions["Home_vax_to_Hosp_vax_total"]
        )
        traj.transitions["hospitalizations"] = hosp_sum
        traj.transitions["hospitalizations_rescaled"] = (
            hosp_sum * alpha * STRAIN_SEVERITY
        )

    return traj
