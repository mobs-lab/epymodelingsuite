import numpy as np

def compute_multimediated_transition_probability(params: tuple | list[tuple], data) -> np.ndarray:
    """
    Compute the probability of a mediated transition that involves multiple mediator compartments.

    An example of "multi-mediated" transition is:
    - S -> L with beta_1 from I_1, I_2
    - S -> L with beta_2 from I_3, I_4

    Parameters
    ----------
    params : tuple | list[tuple]
        A tuple of the transition parameters (rate, mediators), or a list of such tuples [(rate_1, mediators_1), (rate_2, mediators_2), ...] where
        - rate, rate_i: float | str (parameter name/expression, e.g. "beta * r_a")
        - mediators, mediators_i: str | list[str] of compartment names
    data : The data needed for the transition

    Returns
    -------
    prob : np.ndarray shape (n_age_groups,), transition probabilities per age group.

    Math
    ----
    - lambda_total = sum_i [ rate_i * (C · (sum_m I_m / N)) ]
    - p_total = 1 - exp( - lambda_total * dt )

    Examples
    --------
    >>> model.register_transition_kind(kind="mediated_multi", function=compute_multimediated_transition_probability)
    >>> model.add_transition(
        source="S",
        target="L",
        params= [
            ("beta_1", ["I_1", "I_2"]),
            ("beta_2", ["I_3", "I_4"])
        ],
        kind="mediated_multi"
    )
    """
    import copy

    import numpy as np
    from epydemix.utils.utils import evaluate

    # --- Normalize to a list of (rate, mediators) groups ---
    if isinstance(params, (list, tuple)) and params and isinstance(params[0], (list, tuple, tuple)):
        groups = list(params)  # e.g., [(rate1, [..]), (rate2, [..])]
    else:
        groups = [params]  # e.g., (rate, [..]) or (rate, "I")

    # --- Common handles from data dict (as epydemix provides) ---
    C = data["contact_matrix"]["overall"]  # (G, G)
    pop = data["pop"]  # (n_comp, G)
    pop_sizes = data["pop_sizes"]  # (G,)
    comp_indices = data["comp_indices"]  # name -> row index
    dt = data["dt"]
    t = data["t"]

    lambda_total = np.zeros_like(pop_sizes, dtype=float)

    for rate_param, mediators in groups:
        # 1) Evaluate rate at time t (supports expressions over parameters)
        if isinstance(rate_param, str):
            env = copy.deepcopy(data["parameters"])  # parameters over time
            rate_t = evaluate(expr=rate_param, env=env)[t]
        else:
            rate_t = float(rate_param)

        # 2) Normalize mediators to list[str]
        if isinstance(mediators, str):
            mediators = [mediators]
        else:
            mediators = list(mediators)

        # 3) Validate mediator names
        missing = [m for m in mediators if m not in comp_indices]
        if missing:
            raise KeyError(f"mediated_multi: Unknown mediator compartment(s): {missing}")

        # 4) Sum mediator populations element-wise over age groups
        idxs = [comp_indices[m] for m in mediators]
        agents_sum = np.sum(pop[idxs, :], axis=0)  # (G,)

        # 5) Force of infection (force of infection) for this group
        # interaction_k = sum_j C_{k,j} * (agents_j / N_j)
        interaction = C.dot(agents_sum / pop_sizes)  # (G,)
        lambda_total += rate_t * interaction

    # 6) Convert summed intensity to probability over dt
    prob = 1.0 - np.exp(-lambda_total * dt)  # (G,)
    return prob
