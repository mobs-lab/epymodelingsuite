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
