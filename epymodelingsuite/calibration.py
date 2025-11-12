from collections.abc import Callable
from datetime import date

from epydemix.calibration import ABCSampler, CalibrationResults


def reproduce_trajectory(
    calibrator: ABCSampler,
    calibration_results: CalibrationResults,
    particle_index: int,
    generation: int,
    simulation_function: Callable,
    end_date: date,
) -> dict:
    """
    Reproduce a specific trajectory using a user provided simulation function.
    If the provided simulation function is the same as the one used during calibration
    (stored in the ABCSampler) an identical trajectory will be created. If the simulation
    function is different, the result will be a "paired" trajectory which uses the
    same sequence of random numbers.

    Parameters
    ----------
    calibrator : ABCSampler
        The ABCSampler instance that provides the fixed model parameters.
    calibration_results : CalibrationResults
        Results from the calibration process, containing posterior parameter samples
        and stored trajectory metadata (including random states).
    particle_index : int
        Index of the particle (trajectory) to reproduce within the specified generation.
    generation : int
        Generation number in the ABC process from which to reproduce the trajectory.
    simulation_function : Callable
        A user-provided function that runs the model simulation. It must accept a
        dictionary of parameters and return a simulation output dictionary.
    end_date : date
        The end date for the reproduced simulation, appended to the parameter set.

    Returns
    -------
    dict
        The reproduced simulation output, as returned by `simulation_function`.

    Raises
    ------
    ValueError
        If the specified generation or particle index is out of range.
    """
    # Validate generation index
    num_generations = len(calibration_results.posterior_distributions)
    if generation < 0 or generation >= num_generations:
        errMsg = f"Invalid generation {generation}. Must be between 0 and {num_generations - 1}."
        raise ValueError(errMsg)

    # Extract the parameters for this particle
    params_df = calibration_results.posterior_distributions[generation]

    # Validate particle index
    num_particles = len(params_df)
    if particle_index < 0 or particle_index >= num_particles:
        errMsg = f"Invalid particle_index {particle_index}. Must be between 0 and {num_particles - 1}."
        raise ValueError(errMsg)

    params = params_df.iloc[particle_index].to_dict()

    # Get fixed parameters from the calibrator
    fixed_params = calibrator.parameters.copy()

    # Merge sampled parameters with fixed parameters
    all_params = {**fixed_params, **params}

    # Extract the random state from the original trajectory
    original_trajectory = calibration_results.selected_trajectories[generation][particle_index]
    random_state = original_trajectory["random_state"]

    # Add random state to params
    all_params["random_state"] = random_state
    all_params["projection"] = True
    all_params["end_date"] = end_date

    # Simulation function used here is user-provided, which may be different from the one embedded in the calibrator (ABCSampler)
    result = simulation_function(all_params)

    return result


def reproduce_trajectories_in_generation(
    calibrator: ABCSampler,
    calibration_results: CalibrationResults,
    simulation_function: Callable,
    end_date: date,
    generation: int | None = None,
) -> list[dict]:
    """
    Reproduce all trajectories in a specified generation.

    Parameters
    ----------
    calibrator : ABCSampler
        The ABCSampler instance that contains the simulation function.
    calibration_results : CalibrationResults
        Calibration results containing trajectories and parameters.
    end_date : date
        End date for the simulations.
    generation : int, optional
        Generation number to reproduce. Defaults to the last generation.

    Returns
    -------
    list of dict
        List of reproduced simulation results for each particle in the generation.
    """
    if generation is None:
        generation = len(calibration_results.selected_trajectories) - 1

    results = []
    for particle_index in range(len(calibration_results.selected_trajectories[generation])):
        reproduced = reproduce_trajectory(
            calibrator=calibrator,
            calibration_results=calibration_results,
            particle_index=particle_index,
            generation=generation,
            simulation_function=simulation_function,
            end_date=end_date,
        )
        results.append(reproduced)

    return results


def reproduce_all_trajectories(
    calibrator: ABCSampler,
    calibration_results: CalibrationResults,
    simulation_function: Callable,
    end_date: date,
) -> dict[int, list[dict]]:
    """
    Reproduce all trajectories across all generations.

    Parameters
    ----------
    calibrator : ABCSampler
        The ABCSampler instance that contains the simulation function.
    calibration_results : CalibrationResults
        Calibration results containing trajectories and parameters.

    Returns
    -------
    dict
        Dictionary mapping generation numbers to lists of reproduced trajectories.
    """
    results = {}
    for generation in range(len(calibration_results.selected_trajectories)):
        results[generation] = reproduce_trajectories_in_generation(
            calibrator=calibrator,
            calibration_results=calibration_results,
            generation=generation,
            simulation_function=simulation_function,
            end_date=end_date,
        )

    return results


def calc_beta(model_pop, Rt, mu, p_asymptomatic, r_beta_asymp):
    """
    Calculate transmission rate (beta) from reproduction number (Rt).

    Parameters
    ----------
    model_pop : Population
        Population object with contact matrices
    Rt : float
        Reproduction number
    mu : float
        Recovery rate
    p_asymptomatic : float
        Proportion of asymptomatic infections
    r_beta_asymp : float
        Relative transmissibility of asymptomatic infections

    Returns
    -------
    float
        Transmission rate beta
    """
    import numpy as np

    C = np.sum([c for _, c in model_pop.contact_matrices.items()], axis=0)
    eigenvalue = np.linalg.eigvals(C).real.max()

    return Rt * mu / (eigenvalue * (1 - p_asymptomatic + p_asymptomatic * r_beta_asymp))
