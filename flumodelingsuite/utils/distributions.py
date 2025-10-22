"""Distribution conversion utilities."""

import scipy.stats

from ..schema.common import Distribution


def distribution_to_scipy(distribution: Distribution):
    """
    Convert a Distribution object to a scipy distribution object.

    Parameters
    ----------
    distribution : Distribution
        A Distribution instance containing name, args, and kwargs for the scipy distribution.

    Returns
    -------
    scipy.stats distribution object
        The scipy distribution object created from the Distribution parameters.

    Examples
    --------
    >>> from flumodelingsuite.schema.common import Distribution
    >>> dist_config = Distribution(name="norm", args=[0, 1])
    >>> scipy_dist = distribution_to_scipy(dist_config)
    >>> scipy_dist.rvs(5)  # Generate 5 random samples

    >>> dist_config = Distribution(name="uniform", args=[0, 1])
    >>> scipy_dist = distribution_to_scipy(dist_config)
    >>> scipy_dist.rvs(10)  # Generate 10 random samples from uniform distribution
    """
    if distribution.type == "scipy":
        # Get the distribution class from scipy.stats
        dist_class = getattr(scipy.stats, distribution.name)

        # Create the distribution object with args and kwargs
        kwargs = distribution.kwargs or {}
        dist = dist_class(*distribution.args, **kwargs)
    return dist
