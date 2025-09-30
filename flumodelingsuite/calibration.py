import pandas as pd
from epydemix.model import EpiModel


def calc_beta(model_pop, Rt, mu, p_asymptomatic, r_beta_asymp):
    import numpy as np

    C = np.sum([c for _, c in model_pop.contact_matrices.items()], axis=0)
    eigenvalue = np.linalg.eigvals(C).real.max()

    return Rt * mu / (eigenvalue * (1 - p_asymptomatic + p_asymptomatic * r_beta_asymp))


def calibrate(
    model: EpiModel,
    date: str | pd.Timestamp,
    fixed_parameters: dict,
    priors: dict,
    data: pd.DataFrame,
    target_transitions: list[str],
    infectious_seed_compartments: list[str],
    residual_immunity_compartments: list[str],
    vaccine_probability_function: callable,
    calc_beta: callable,
    vax_sched: pd.DataFrame,
    distance_function: callable,
    scenario: str | None = None,
    strategy: str = "top_fraction",
    **strategy_kwargs,
):
    """
    Calibrate an Epydemix model to hospitalization data using ABC methods.

    This function performs Approximate Bayesian Computation (ABC) calibration
    of an epidemic model to observed hospitalization data for a specific location
    and date using the specified calibration strategy.

    Parameters
    ----------
    model : EpiModel
        The epidemic model to calibrate
    date : str | pd.Timestamp
        The "as_of" date for filtering the hospitalization data
    fixed_parameters : dict
        Dictionary of parameters to keep fixed during calibration
    priors : dict
        Dictionary of prior distributions for parameters to be calibrated
    data : pd.DataFrame
        Hospitalization data with 'geo_value', 'as_of', and 'value' columns
    target_transitions : list[str]
        List of transition names to sum for hospitalization output
    infectious_seed_compartments : list[str]
        Compartments to seed with initial infectious individuals
    residual_immunity_compartments : list[str]
        Compartments for individuals with residual immunity
    vaccine_probability_function : callable
        Function to calculate vaccination probabilities
    calc_beta : callable
        Function to calculate transmission rate from R0
    vax_sched : pd.DataFrame
        Vaccination schedule data with 'location' and 'scenario' columns
    scenario : str | None, default None
        Scenario name to filter vaccination schedule
    strategy : str, default 'top_fraction'
        ABC calibration strategy ('top_fraction', 'smc', or 'rejection')
    **strategy_kwargs
        Strategy-specific parameters:
        - For 'top_fraction': top_fraction, Nsim, verbose
        - For 'smc': num_particles, num_generations, epsilon_schedule, etc.
        - For 'rejection': num_particles, epsilon, max_time, etc.

    Returns
    -------
    CalibrationResults
        Results object containing posterior distributions, selected trajectories,
        distances, weights, and other calibration outputs
    """

    from epydemix.calibration import ABCSampler, rmse, wmape
    from .utils import convert_location_name_format

    location = model.population.name
    location_iso = convert_location_name_format(location, "ISO")

    data_state = data[data["geo_value"] == location_iso]
    data_state = data_state[data_state["as_of"] == date]

    sim_wrapper = make_simulate_wrapper(
        model=model,
        priors=priors,
        fixed_parameters=fixed_parameters,
        data=data_state,
        target_transitions=target_transitions,
        infectious_seed_compartments=infectious_seed_compartments,
        residual_immunity_compartments=residual_immunity_compartments,
        vaccine_probability_function=vaccine_probability_function,
        calc_beta=calc_beta,
        vax_sched=vax_sched,
        scenario=scenario,
    )

    abc_sampler = ABCSampler(
        simulation_function=sim_wrapper,
        priors=priors,
        parameters=fixed_parameters,
        observed_data=data_state["value"].values,
        distance_function=distance_function,
    )

    results_abc = abc_sampler.calibrate(strategy=strategy, **strategy_kwargs)

    return results_abc


def make_simulate_wrapper(
    model: EpiModel,
    priors: dict,
    fixed_parameters: dict,
    data: pd.DataFrame,
    target_transitions: list[str],
    infectious_seed_compartments: list[str],
    residual_immunity_compartments: list[str],
    vaccine_probability_function: callable,
    calc_beta: callable,
    vax_sched: pd.DataFrame,
    scenario: str | None = None,
) -> callable:
    """
    Create a simulation wrapper function for ABC calibration of an Epydemix model.

    This function creates a closure that wraps the epidemic simulation with
    parameter sampling, vaccination scheduling, and output formatting suitable
    for ABC calibration methods.

    Parameters
    ----------
    model : EpiModel
        The epidemic model to simulate
    priors : dict
        Dictionary of prior distributions for parameters to be calibrated
    data : pd.DataFrame
        Observed data with 'target_end_date' column for date matching
    target_transitions : list[str]
        List of transition names to sum for hospitalization output
    infectious_seed_compartments : list[str]
        Compartments to seed with initial infectious individuals
    residual_immunity_compartments : list[str]
        Compartments for individuals with residual immunity
    vaccine_probability_function : callable
        Function to calculate vaccination probabilities
    calc_beta : callable
        Function to calculate transmission rate from R0
    vax_sched : pd.DataFrame
        Vaccination schedule data with 'location' and 'scenario' columns
    scenario : str | None, default None
        Scenario name to filter vaccination schedule

    Returns
    -------
    callable
        Simulation wrapper function that takes parameter dict and returns
        dictionary with 'data' key containing hospitalization time series

    Notes
    -----
    The returned wrapper function expects parameters dict containing:
    - 'end_date': simulation end date
    - 'reference_date': simulation reference date
    - 'infectious_seed': fraction of population initially infectious
    - 'resample_frequency': frequency for output resampling
    - 'start_date_offset': optional offset from reference date
    - 'residual_immunity': optional fraction with residual immunity
    - 'Rt': optional reproduction number for transmission calculation
    """
    import copy
    import datetime as dt

    import numpy as np
    import pandas as pd
    from epydemix import simulate

    from .seasonality import get_seasonal_transmission_balcan
    from .utils import convert_location_name_format
    from .vaccinations import add_vaccination_schedule, reaggregate_vaccines

    location = model.population.name
    location_iso = convert_location_name_format(location, "ISO")
    vax_schedule_state = vax_sched.query("location == @location_iso")
    if scenario is not None:
        vax_schedule_state = vax_schedule_state.query("scenario == @scenario")

    def simulate_wrapper(params):
        # Extract the sampled parameters
        start_date_offset = 0  # defaults to 0 (no offset, just starts at reference date)
        date_stop = params["end_date"]
        reference_date = params["reference_date"]
        epi_model = copy.deepcopy(model)

        if "start_date_offset" in params.keys():
            start_date_offset = int(params["start_date_offset"])
            actual_start_date = reference_date + dt.timedelta(days=start_date_offset)
            vax_schedule_reag = reaggregate_vaccines(schedule=vax_schedule_state, actual_start_date=actual_start_date)

            epi_model = add_vaccination_schedule(
                model=epi_model,  # Use the renamed variable
                vaccine_probability_function=vaccine_probability_function,
                source_comp="Susceptible",
                target_comp="Susceptible_vax",
                vaccination_schedule=vax_schedule_reag,
            )

        if "Rt" in params.keys():
            beta = calc_beta(
                epi_model.population,
                Rt=params["Rt"],
                mu=params["mu"],
                p_asymptomatic=params["p_asymptomatic"],
                r_beta_asymp=params["r_beta_asymp"],
            )
            _, st = get_seasonal_transmission_balcan(
                date_start=reference_date + dt.timedelta(days=start_date_offset),
                date_stop=date_stop,
                date_tmax=dt.date(2025, 1, 1),  # peak transmission is January 1
                date_tmin=dt.date(2025, 7, 1),  # lowest transmission is July 1
                val_min=params["val_min"],
                val_max=params["val_max"],
            )
            beta = np.array(beta) * st
            epi_model.beta = beta

        infectious_frac = params["infectious_seed"]  # must be specified either in priors or fixed_parameters
        infectious_frac_distributed = infectious_frac / len(infectious_seed_compartments)
        residual_immunity = (
            params["residual_immunity"] if "residual_immunity" in params.keys() else 0.0
        )  # must be specified either in priors or fixed_parameters
        residual_immunity_distributed = residual_immunity / len(residual_immunity_compartments)

        init_cond = {
            "Susceptible": (1 - infectious_frac - residual_immunity) * epi_model.population.Nk,
        }
        for comp in infectious_seed_compartments:
            init_cond[comp] = infectious_frac_distributed * epi_model.population.Nk

        for comp in residual_immunity_compartments:
            if comp not in epi_model.compartments:
                epi_model.add_compartments(comp)
                init_cond[comp] = residual_immunity_distributed * epi_model.population.Nk

        sim_params = {
            "epimodel": epi_model,
            "start_date": actual_start_date,
            "beta": beta,
            "initial_conditions_dict": init_cond,
            "end_date": date_stop,
            "resample_frequency": params["resample_frequency"],
        }

        # Run simulation
        try:
            results = simulate(**sim_params)
            trajectory_dates = results.dates
            data_dates = list(pd.to_datetime(data["target_end_date"].values))

            mask = [date in data_dates for date in trajectory_dates]

            total_hosp = sum(results.transitions[key] for key in target_transitions)

            total_hosp = total_hosp[mask]

            if len(total_hosp) < len(data_dates):
                pad_len = len(data_dates) - len(total_hosp)
                total_hosp = np.pad(total_hosp, (pad_len, 0), constant_values=0)

        except Exception as e:
            print(f"Simulation failed with parameters {params}: {e}")
            data_dates = list(pd.to_datetime(data["target_end_date"].values))
            total_hosp = np.full(len(data_dates), 0)
        return {"data": total_hosp}

    return simulate_wrapper
