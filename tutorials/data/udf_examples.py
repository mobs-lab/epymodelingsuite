## udf_examples.py
# Dummy examples for User-Defined Functions
import copy

from epydemix.calibration import rmse
from epydemix.model.simulation_output import Trajectory


def udf_distance(data: Dict, simulation: Dict):
    """Dummy wrapper for epydemix rmse."""
    return rmse(data, simulation)


def udf_transform(trajectory: Trajectory):
    """Dummy post-hoc transformation function that records a new compartment."""
    traj = copy.deepcopy(trajectory)
    traj.compartments["Dummy_L+I"] = traj.compartments["L_total"] + traj.compartments["I_total"]
    return traj
