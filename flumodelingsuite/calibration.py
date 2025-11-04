import pandas as pd


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


def resimulate_with_posterior(
    epi_model,
    posterior: pd.DataFrame,
    fixed_parameters: dict,
    infectious_seed_compartments: list[str],
    residual_immunity_compartments: list[str],
    all_scen: pd.DataFrame,
    scenario: str,
    vaccine_probability_function: callable,
    n_trajectories: int,
):
    """
    Resimulate the model with posterior parameter sets by randomly sampling rows.

    Parameters
    ----------
    epi_model : EpiModel
        Base epidemic model
    posterior : pd.DataFrame
        Posterior samples (rows = parameter sets)
    fixed_parameters : dict
        Fixed simulation parameters (end_date, reference_date, resample_frequency, etc.)
    infectious_seed_compartments : list[str]
        Compartments for initial seeding of infection
    residual_immunity_compartments : list[str]
        Compartments for distributing residual immunity
    all_scen : pd.DataFrame
        Vaccination schedules with 'location' and 'scenario' columns
    scenario : str
        Scenario to filter vaccination schedule
    vaccine_probability_function : callable
        Function for vaccination probability
    n_trajectories : int
        Number of posterior draws to simulate

    Returns
    -------
    comp_stacked : pd.DataFrame
        Stacked compartment trajectories
    trans_stacked : pd.DataFrame
        Stacked transition trajectories
    """
    import copy
    import datetime as dt

    import numpy as np
    import pandas as pd
    from epydemix import simulate

    from .seasonality import get_seasonal_transmission_balcan
    from .utils import convert_location_name_format
    from .vaccinations import add_vaccination_schedule, reaggregate_vaccines

    comp_stacked = []
    trans_stacked = []

    state_iso = convert_location_name_format(epi_model.population.name, "ISO")
    vax_schedule_state = all_scen.query("location == @state_iso and scenario == @scenario")

    # sample posterior indices with replacement
    sampled_indices = np.random.choice(posterior.index, size=n_trajectories, replace=True)

    for draw_idx, idx in enumerate(sampled_indices):
        row = posterior.loc[idx]
        posterior_dict = row.to_dict()
        combined_parameters = {**fixed_parameters, **posterior_dict}

        model_copy = copy.deepcopy(epi_model)
        start_date_offset = int(combined_parameters.get("start_date_offset", 0))
        reference_date = combined_parameters["reference_date"]
        actual_start_date = reference_date + dt.timedelta(days=start_date_offset)
        date_stop = combined_parameters["end_date"]

        # vaccination schedule
        vax_schedule_reag = reaggregate_vaccines(schedule=vax_schedule_state, actual_start_date=actual_start_date)
        model_copy = add_vaccination_schedule(
            model=model_copy,
            vaccine_probability_function=vaccine_probability_function,
            source_comp="Susceptible",
            target_comp="Susceptible_vax",
            vaccination_schedule=vax_schedule_reag,
        )

        # transmission rate
        beta = None
        if "Rt" in combined_parameters:
            beta = calc_beta(
                model_copy.population,
                Rt=combined_parameters["Rt"],
                mu=combined_parameters["mu"],
                p_asymptomatic=combined_parameters["p_asymptomatic"],
                r_beta_asymp=combined_parameters["r_beta_asymp"],
            )
            _, st = get_seasonal_transmission_balcan(
                date_start=reference_date + dt.timedelta(days=start_date_offset),
                date_stop=date_stop,
                date_tmax=dt.date(2025, 1, 1),
                date_tmin=dt.date(2025, 7, 1),
                val_min=combined_parameters["val_min"],
                val_max=combined_parameters["val_max"],
            )
            beta = np.array(beta) * st
            model_copy.beta = beta

        # initial conditions
        infectious_frac = combined_parameters["infectious_seed"]
        infectious_frac_distributed = infectious_frac / len(infectious_seed_compartments)
        residual_immunity = combined_parameters.get("residual_immunity", 0.0)
        residual_immunity_distributed = residual_immunity / len(residual_immunity_compartments)

        init_cond = {
            "Susceptible": (1 - infectious_frac - residual_immunity) * model_copy.population.Nk,
        }
        for comp in infectious_seed_compartments:
            init_cond[comp] = infectious_frac_distributed * model_copy.population.Nk

        for comp in residual_immunity_compartments:
            if comp not in model_copy.compartments:
                model_copy.add_compartments(comp)
            init_cond[comp] = residual_immunity_distributed * model_copy.population.Nk

        sim_params = {
            "epimodel": model_copy,
            "start_date": actual_start_date,
            "beta": beta,
            "initial_conditions_dict": init_cond,
            "end_date": date_stop,
            "resample_frequency": combined_parameters["resample_frequency"],
        }

        results = simulate(**sim_params)

        comp = pd.DataFrame(results.compartments)
        comp["draw_idx"] = draw_idx
        comp["posterior_idx"] = idx
        comp["dates"] = results.dates
        comp_stacked.append(comp)

        trans = pd.DataFrame(results.transitions)
        trans["draw_idx"] = draw_idx
        trans["posterior_idx"] = idx
        trans["dates"] = results.dates
        trans_stacked.append(trans)

    comp_stacked = pd.concat(comp_stacked, axis=0).reset_index(drop=True)
    trans_stacked = pd.concat(trans_stacked, axis=0).reset_index(drop=True)

    return comp_stacked, trans_stacked
