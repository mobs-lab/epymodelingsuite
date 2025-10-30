"""Reproduce the results of a previous simulation runs."""

from epydemix.calibration import ABCSampler, CalibrationResults

def reproduce_trajectory(
    calibrator: ABCSampler,
    calibration_results: CalibrationResults,
    particle_index: int,
    generation: int,
) -> dict:
    """
    Reproduce a specific trajectory using the stored simulate_wrapper.
    
    Parameters
    ----------
    calibrator : ABCSampler
        The ABCSampler instance that contains the simulation function.
    cal_results : CalibrationResults
        Calibration results containing trajectories and parameters.
    generation : int
        Generation number to reproduce from.
    particle_index : int
        Index of the particle/trajectory to reproduce.
        
    Returns
    -------
    dict
        Reproduced simulation results.
    """

    # Extract the parameters for this particle
    params_df = calibration_results.posterior_distributions[generation]
    params = params_df.iloc[particle_index].to_dict()
    
    # Get fixed parameters from the calibrator
    fixed_params = calibrator.parameters.copy()
    
    # Merge sampled parameters with fixed parameters
    all_params = {**fixed_params, **params}
    
    # Extract the random state from the original trajectory
    original_trajectory = calibration_results.selected_trajectories[generation][particle_index]
    random_state = original_trajectory['random_state']
    
    # Add random state to params (if you modified simulate_wrapper to accept it)
    all_params['random_state'] = random_state
    all_params['projection'] = True
    
    # Call the stored simulate_wrapper
    result = calibrator.simulation_function(all_params)
    
    return result

def reproduce_trajectories_in_generation(
    calibrator: ABCSampler,
    calibration_results: CalibrationResults,
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
        )
        results.append(reproduced)

    return results


def reproduce_all_trajectories(
    calibrator: ABCSampler,
    calibration_results: CalibrationResults,
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
        )

    return results
