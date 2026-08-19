"""Distribution validation and conversion utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.stats

if TYPE_CHECKING:
    from epymodelingsuite.schema.common import Distribution

SCALAR_SUPPORT_BOUND_COUNT = 2


def _get_parameter_value(
    distribution: Distribution,
    *,
    name: str,
    position: int,
    default: float,
) -> object:
    """Get a SciPy parameter supplied as either a keyword or positional argument."""
    if name in distribution.kwargs:
        return distribution.kwargs[name]
    if len(distribution.args) > position:
        return distribution.args[position]
    return default


def _is_finite_scalar(value: object) -> bool:
    """Return whether a value is a finite numeric scalar."""
    try:
        array = np.asarray(value)
        return array.ndim == 0 and bool(np.isfinite(array))
    except TypeError:
        return False


def validate_distribution(  # noqa: C901 - validation is intentionally linear
    distribution: Distribution,
    *,
    context: str | None = None,
) -> None:
    """
    Validate a probability distribution configuration.

    SciPy continuous and discrete scalar random variables are supported. Custom
    distributions are left unchanged because they are resolved outside SciPy.

    Parameters
    ----------
    distribution : Distribution
        Distribution configuration to validate.
    context : str, optional
        Description of where the distribution is used, included in errors.

    Raises
    ------
    ValueError
        If the distribution type, name, or parameters are invalid.
    """
    prefix = f"{context}: " if context else ""

    # Custom distributions are validated by their own implementation rather than
    # against SciPy's random-variable API.
    if distribution.type == "custom":
        return
    if distribution.type != "scipy":
        msg = f"{prefix}Unsupported distribution type {distribution.type!r}"
        raise ValueError(msg)

    # Resolve only SciPy's scalar continuous and discrete random-variable
    # generators. Other scipy.stats utilities and multivariate distributions do
    # not match the scalar parameter interface expected by the samplers.
    dist_class = getattr(scipy.stats, distribution.name, None)
    if dist_class is None:
        msg = f"{prefix}Unknown scipy.stats distribution '{distribution.name}'"
        raise ValueError(msg)
    if not isinstance(dist_class, (scipy.stats.rv_continuous, scipy.stats.rv_discrete)):
        msg = (
            f"{prefix}scipy.stats.{distribution.name} is not a supported scalar continuous or discrete random variable"
        )
        raise ValueError(msg)  # noqa: TRY004 - the configured value, rather than a Python type, is invalid

    # Freezing checks argument names, argument counts, and required shape
    # parameters. SciPy still permits some invalid values (such as scale=0), so
    # the explicit checks below remain necessary.
    details = f"args={distribution.args!r}, kwargs={distribution.kwargs!r}"
    try:
        frozen_distribution = dist_class(*distribution.args, **distribution.kwargs)
    except (TypeError, ValueError) as e:
        msg = f"{prefix}Invalid arguments for scipy.stats.{distribution.name} ({details}): {e}"
        raise ValueError(msg) from e

    # Both continuous and discrete SciPy variables accept loc after their shape
    # arguments. Calibration priors must use a finite scalar location.
    loc = _get_parameter_value(
        distribution,
        name="loc",
        position=dist_class.numargs,
        default=0,
    )
    if not _is_finite_scalar(loc):
        msg = (
            f"{prefix}Invalid scipy.stats.{distribution.name} location: loc must be a finite scalar; "
            f"got {loc!r} ({details})"
        )
        raise ValueError(msg)

    # Scale is part of the continuous-variable API only. Its positional index is
    # immediately after loc, following any distribution-specific shape arguments.
    if isinstance(dist_class, scipy.stats.rv_continuous):
        scale = _get_parameter_value(
            distribution,
            name="scale",
            position=dist_class.numargs + 1,
            default=1,
        )
        if not _is_finite_scalar(scale) or scale <= 0:
            help_text = ""
            if distribution.name == "uniform":
                help_text = " For scipy.stats.uniform, arguments are [loc, scale], where upper bound = loc + scale."
            msg = (
                f"{prefix}Invalid scipy.stats.{distribution.name} scale: scale must be a finite scalar "
                f"greater than 0; got {scale!r} ({details}).{help_text}"
            )
            raise ValueError(msg)

    # SciPy's support calculation applies distribution-specific shape checks
    # without sampling. Invalid combinations return NaN bounds; vectorized bounds
    # indicate a non-scalar distribution, which the current samplers do not support.
    try:
        support = np.asarray(frozen_distribution.support(), dtype=float)
    except (TypeError, ValueError) as e:
        msg = f"{prefix}Could not validate scipy.stats.{distribution.name} support ({details}): {e}"
        raise ValueError(msg) from e

    if support.size != SCALAR_SUPPORT_BOUND_COUNT:
        msg = (
            f"{prefix}scipy.stats.{distribution.name} must define a scalar random variable; "
            f"received vectorized parameters ({details})"
        )
        raise ValueError(msg)
    if np.any(np.isnan(support)):
        msg = (
            f"{prefix}Invalid parameters for scipy.stats.{distribution.name}: "
            f"the configured distribution has no valid support ({details})"
        )
        raise ValueError(msg)


def distribution_to_scipy(distribution: Distribution, *, context: str | None = None) -> object:
    """
    Convert a Distribution object to a validated scipy distribution object.

    Parameters
    ----------
    distribution : Distribution
        A Distribution instance containing name, args, and kwargs for the scipy distribution.
    context : str, optional
        Description of where the distribution is used, included in errors.

    Returns
    -------
    scipy.stats distribution object
        The frozen scipy distribution object created from the Distribution parameters.

    Examples
    --------
    >>> from epymodelingsuite.schema.common import Distribution
    >>> dist_config = Distribution(name="norm", args=[0, 1])
    >>> scipy_dist = distribution_to_scipy(dist_config)
    >>> scipy_dist.rvs(5)  # Generate 5 random samples

    >>> dist_config = Distribution(name="uniform", args=[0, 1])
    >>> scipy_dist = distribution_to_scipy(dist_config)
    >>> scipy_dist.rvs(10)  # Generate 10 random samples from uniform distribution
    """
    validate_distribution(distribution, context=context)

    if distribution.type != "scipy":
        prefix = f"{context}: " if context else ""
        msg = f"{prefix}Cannot convert distribution type {distribution.type!r} to scipy"
        raise ValueError(msg)

    dist_class = getattr(scipy.stats, distribution.name)
    return dist_class(*distribution.args, **distribution.kwargs)
