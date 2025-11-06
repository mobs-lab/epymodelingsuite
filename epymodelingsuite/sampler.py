# sampler.py

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import qmc

logger = logging.getLogger(__name__)


"""
Overview
--------
This module generate samples for a set of
- compartment proportions (for initial conditions)
- parameter values
which would be used for parameter scan in simulations.

The steps are:
1) Generate high-dimensional uniform samples [0,1]^D with Latin Hypercube Sampling or Sobol sequences.
2) Deterministically map subsets of U to:
    - "compartments" that must satisfy per-dimension [min, max] box constraints *and* a global sum constraint (sum <= total = 1).  If requested, we can also enforce sum == total via a simple deterministic water-filling step.
    - other parameters (e.g., uniform scalars, dates via uniform days).
"""


# ============================================================
# Uniform samplers for high-dimensional spaces
# ============================================================
def _sample_lhs(n_samples: int, dimension: int, *, seed: int | None = None, scramble: bool = True) -> np.ndarray:
    """
    Generate an (n_samples, dimension) Latin Hypercube sample U on [0,1]^dimension.

    This uses scipy's `qmc.LatinHypercube`, which produces a space-filling design
    via stratification in every dimension. With `scramble=True` (default), Owen
    scrambling is applied to reduce patterning while preserving stratification.

    Parameters
    ----------
    n_samples : int
        Number of rows (design points).
    dimension : int
        Number of columns (dimensions) in the design, i.e., D in [0,1]^D.
    seed : Optional[int]
        Random seed for reproducibility (passed to the SciPy engine).
    scramble : bool
        If True, use Owen scrambling; if False, produce an unscrambled LHS.

    Returns
    -------
    U : (n_samples, dimension) ndarray
        LHS points in [0,1]^dimension. No further randomness is used beyond this
        matrix so that the LHS stratification is preserved downstream.
    """
    engine = qmc.LatinHypercube(d=dimension, scramble=scramble, seed=seed)
    U = engine.random(n_samples)
    return U


def _sample_sobol(n_samples: int, dimension: int, *, seed: int | None = None, scramble: bool = True) -> np.ndarray:
    """
    Generate an (n_samples, dimension) Sobol sequence sample U on [0,1]^dimension.

    This uses scipy's `qmc.Sobol`, which produces a low-discrepancy sequence
    that fills the space more uniformly than random sampling. With `scramble=True`
    (default), Owen scrambling is applied to randomize the sequence while
    preserving its low-discrepancy properties.

    Parameters
    ----------
    n_samples : int
        Number of rows (design points). Note: Sobol sequences perform best when
        n_samples is a power of 2, though this is not strictly required.
    dimension : int
        Number of columns (dimensions) in the design, i.e., D in [0,1]^D.
        Maximum supported dimension is typically 21170 for SciPy's implementation.
    seed : Optional[int]
        Random seed for reproducibility (used for scrambling if enabled).
        Note: unscrambled Sobol sequences are deterministic.
    scramble : bool
        If True, use Owen scrambling to randomize the sequence;
        if False, produce the deterministic Sobol sequence.

    Returns
    -------
    U : (n_samples, dimension) ndarray
        Sobol sequence points in [0,1]^dimension. These points have better
        space-filling properties than random sampling, especially useful for
        numerical integration and global optimization.

    Notes
    -----
    - Sobol sequences are quasi-random (deterministic but well-distributed)
    - They achieve O(log^d(N)/N) discrepancy vs O(1/sqrt(N)) for random sampling
    - Particularly effective for moderate dimensions (e.g., d < 20)
    - The first point is always at the origin [0, 0, ..., 0] unless scrambled

    Examples
    --------
    >>> # Generate 256 points in 3D (power of 2 recommended)
    >>> U = sample_sobol(256, 3, seed=42)
    >>> print(U.shape)
    (256, 3)

    >>> # Unscrambled sequence is deterministic
    >>> U1 = sample_sobol(16, 2, scramble=False)
    >>> U2 = sample_sobol(16, 2, scramble=False)
    >>> np.allclose(U1, U2)
    True
    """
    # Create Sobol engine with specified parameters
    # Note: seed parameter in Sobol only affects scrambling
    engine = qmc.Sobol(d=dimension, scramble=scramble, seed=seed)

    # Generate the Sobol sequence
    U = engine.random(n_samples)

    return U


# ============================================================
# Sampler for the set of parameter and compartment proportions
# ============================================================


# Class for storing sampling results
@dataclass
class SamplingResult:
    compartments: dict[str, np.ndarray]
    parameters: dict[str, np.ndarray]
    uniform_samples: np.ndarray

    def __str__(self) -> str:
        lines = []
        n_samples = self.uniform_samples.shape[0]
        lines.append(f"SamplingResult (n={n_samples:,} samples)")
        lines.append("")

        # Compartments
        if self.compartments:
            lines.append("Compartments:")
            for name in sorted(self.compartments.keys()):
                if name.startswith("_"):
                    continue
                values = self.compartments[name]
                lines.append(f"  {name:<10} [{np.min(values):.4f}, {np.max(values):.4f}]")

            if "_residual" in self.compartments:
                res = self.compartments["_residual"]
                lines.append(f"  {'_residual':<10} [{np.min(res):.4f}, {np.max(res):.4f}]")

        # Parameters
        if self.parameters:
            lines.append("\nParameters:")
            for name in sorted(self.parameters.keys()):
                values = self.parameters[name]
                # Check if dates
                if values.dtype == "object":
                    lines.append(f"  {name:<10} [{min(values)}, {max(values)}]")
                else:
                    lines.append(f"  {name:<10} [{np.min(values):.4f}, {np.max(values):.4f}]")

        return "\n".join(lines)

    def __repr__(self) -> str:
        n = self.uniform_samples.shape[0]
        nc = len([k for k in self.compartments if not k.startswith("_")])
        np_ = len(self.parameters)
        return f"SamplingResult(n={n}, compartments={nc}, parameters={np_})"

    def plot_compartments_distribution(self, figsize=(12, 6), bins=30, show_residual=False):
        """Plot distribution of compartment values."""
        # Filter compartments
        comps = {
            k: v for k, v in self.compartments.items() if not k.startswith("_") or (k == "_residual" and show_residual)
        }

        if not comps:
            print("No compartments to plot")
            return None

        n_comps = len(comps)
        fig, axes = plt.subplots(1, n_comps, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        for ax, (name, values) in zip(axes, comps.items(), strict=False):
            ax.hist(values, bins=bins, alpha=0.7, edgecolor="black")
            ax.set_title(name)
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")

            # Add statistics
            mean = np.mean(values)
            median = np.median(values)
            ax.axvline(mean, color="red", linestyle="--", alpha=0.7, label=f"Mean: {mean:.3f}")
            ax.axvline(median, color="green", linestyle="--", alpha=0.7, label=f"Median: {median:.3f}")
            ax.legend(fontsize=8)

            # Add range info
            ax.text(
                0.02,
                0.98,
                f"[{np.min(values):.3f}, {np.max(values):.3f}]",
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        plt.suptitle(f"Compartment Distributions (n={len(next(iter(comps.values()))):,} samples)")
        plt.tight_layout()
        return fig

    def plot_parameters_distribution(self, figsize=(12, 6), bins=30):
        """Plot distribution of parameter values."""
        if not self.parameters:
            print("No parameters to plot")
            return None

        n_params = len(self.parameters)
        fig, axes = plt.subplots(1, n_params, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        for ax, (name, values) in zip(axes, self.parameters.items(), strict=False):
            # Handle date parameters
            if values.dtype == "object":
                from datetime import date

                if len(values) > 0 and isinstance(values[0], date):
                    # Convert dates to numbers for plotting
                    import matplotlib.dates as mdates
                    from matplotlib.dates import DateFormatter, date2num

                    # Convert to matplotlib date numbers
                    date_nums = [date2num(d) for d in values]

                    ax.hist(date_nums, bins=bins, alpha=0.7, edgecolor="black", color="orange")
                    ax.set_title(name)
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Frequency")

                    # Format x-axis as dates
                    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
                    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

                    # Add range info with dates
                    min_date = min(values)
                    max_date = max(values)
                    ax.text(
                        0.02,
                        0.98,
                        f"[{min_date}, {max_date}]",
                        transform=ax.transAxes,
                        fontsize=8,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                    )
                continue

            # Handle numeric parameters
            ax.hist(values, bins=bins, alpha=0.7, edgecolor="black", color="orange")
            ax.set_title(name)
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")

            mean = np.mean(values)
            ax.axvline(mean, color="red", linestyle="--", alpha=0.7, label=f"Mean: {mean:.3f}")
            ax.legend(fontsize=8)

            ax.text(
                0.02,
                0.98,
                f"[{np.min(values):.3f}, {np.max(values):.3f}]",
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        plt.suptitle(f"Parameter Distributions (n={len(next(iter(self.parameters.values()))):,} samples)")
        plt.tight_layout()
        return fig

    def plot_compartments_3d(self, compartments=None, figsize=(6, 5), alpha=0.6):
        """Plot 3D scatter of selected compartments."""
        # Filter out special compartments
        available = [k for k in self.compartments.keys() if not k.startswith("_")]
        # Default to first 3 compartments if not specified
        if compartments is None:
            compartments = available[:3]
        if len(compartments) < 3:
            print(f"Need at least 3 compartments for 3D plot. Available: {available}")
            return None

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        x = self.compartments[compartments[0]]
        y = self.compartments[compartments[1]]
        z = self.compartments[compartments[2]]

        # Color by sum of all compartments
        comp_sum = sum(self.compartments[k] for k in available)
        scatter = ax.scatter(x, y, z, c=comp_sum, cmap="viridis", alpha=alpha, s=1)

        ax.set_xlabel(compartments[0])
        ax.set_ylabel(compartments[1])
        ax.set_zlabel(compartments[2])
        ax.set_title(f"Sampled compartment proportions  (n={len(x):,} samples)")

        # Adjust colorbar position to avoid overlap
        cbar = plt.colorbar(scatter, ax=ax, label="Sum of all compartments", shrink=0.8, pad=0.1)

        plt.tight_layout()
        return fig

    def plot_parameters_3d(self, parameters=None, figsize=(6, 5), alpha=0.6):
        """Plot 3D scatter of selected parameters."""
        from datetime import date

        import matplotlib.dates as mdates
        from matplotlib.dates import DateFormatter, date2num

        # Helper function to convert to numeric
        def to_numeric(series, param_name):
            """Convert series to numeric, handling datetime types."""
            if pd.api.types.is_numeric_dtype(series):
                return series, False  # Not a date
            if series.dtype == "object" and len(series) > 0 and isinstance(series[0], date):
                # Convert dates to matplotlib date numbers
                return np.array([date2num(d) for d in series]), True  # Is a date
            return None, False

        # Filter plottable parameters (numeric or datetime)
        plottable_params = {}
        date_params = set()  # Track which params are dates
        for k, v in self.parameters.items():
            numeric_v, is_date = to_numeric(v, k)
            if numeric_v is not None:
                plottable_params[k] = numeric_v
                if is_date:
                    date_params.add(k)

        if parameters is None:
            parameters = list(plottable_params.keys())[:3]
        if len(parameters) < 3:
            print(f"Need at least 3 numeric/datetime parameters. Available: {list(plottable_params.keys())}")
            return None

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        # Get numeric values for plotting
        x = plottable_params[parameters[0]]
        y = plottable_params[parameters[1]]
        z = plottable_params[parameters[2]]

        # Simple scatter without colormap
        scatter = ax.scatter(x, y, z, alpha=alpha, s=2)

        # Set labels and format date axes
        for i, (param_name, axis_data, axis_obj) in enumerate(
            [(parameters[0], x, ax.xaxis), (parameters[1], y, ax.yaxis), (parameters[2], z, ax.zaxis)]
        ):
            if param_name in date_params:
                # Format axis for dates
                if i == 0:
                    ax.set_xlabel(param_name, labelpad=10)
                    ax.xaxis.set_major_formatter(DateFormatter("%m/%d"))
                    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                    # Rotate x-axis tick labels
                    for tick in ax.xaxis.get_major_ticks():
                        tick.label1.set_rotation(45)
                        tick.label1.set_ha("right")
                elif i == 1:
                    ax.set_ylabel(param_name, labelpad=10)
                    ax.yaxis.set_major_formatter(DateFormatter("%m/%d"))
                    ax.yaxis.set_major_locator(mdates.AutoDateLocator())
                    # Rotate y-axis tick labels
                    for tick in ax.yaxis.get_major_ticks():
                        tick.label1.set_rotation(-45)
                        tick.label1.set_ha("left")
                else:  # z-axis
                    ax.set_zlabel(param_name, labelpad=10)
                    ax.zaxis.set_major_formatter(DateFormatter("%m/%d"))
                    ax.zaxis.set_major_locator(mdates.AutoDateLocator())
            # Regular numeric axis with padding
            elif i == 0:
                ax.set_xlabel(param_name)
            elif i == 1:
                ax.set_ylabel(param_name)
            else:
                ax.set_zlabel(param_name)

        ax.set_title(f"Sampled parameters (n={len(x):,} samples)")

        plt.tight_layout()
        return fig

    def plot_correlation_matrix(self, figsize=(10, 8)):
        """Plot correlation matrix between all variables."""
        import seaborn as sns

        # Combine all numeric data
        data_dict = {}
        for name, values in self.compartments.items():
            if not name.startswith("_"):
                data_dict[name] = values

        for name, values in self.parameters.items():
            if values.dtype != "object":  # Skip dates
                data_dict[name] = values

        if len(data_dict) < 2:
            print("Need at least 2 variables for correlation matrix")
            return None

        # Create correlation matrix
        import pandas as pd

        df = pd.DataFrame(data_dict)
        corr = df.corr()

        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlation Matrix")
        plt.tight_layout()
        return fig


def generate_parameter_samples(
    n_samples: int,
    compartment_names: list[str],
    mins: list[float],
    maxs: list[float],
    *,
    total: float = 1.0,
    param_specs: list[dict] | None = None,
    seed: int | None = None,
    scramble: bool = True,
    enforce_compartment_sum_equal: bool = False,
    sampling_method: Literal["lhs", "sobol"] | Callable = "lhs",
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """
    Generate samples of parameter and compartment proportion sets. It first constructs a quasi-Monte Carlo sample on [0,1]^(n_compartments + n_parameters), then maps it to:
    - compartments with box constraints (mins/maxs) and a global sum constraint
    - other parameters (uniform scalars, dates via uniform days).

    The sampling method can be specified as either Latin Hypercube Sampling (LHS) or Sobol sequences, or a custom sampling function.

    Parameters
    ----------
    n_samples : int
        Number of design points (rows). For Sobol sequences, powers of 2 (=2^n) are recommended for best performance.
    compartment_names : List[str]
        Names of compartments in order. Length defines n_compartments.
    mins, maxs : List[float]
        Per-compartment lower/upper bounds (same order as names). Must satisfy
        maxs[i] >= mins[i] for all i.
    total : float, default=1.0
        Upper bound on the sum across compartments.
    param_specs : Optional[List[Dict]]
        Parameter specs describing how to map additional columns of U.
        Supported examples:
          {"name": "Rt", "type": "uniform", "args": (0.0, 2.0)}
          {"name": "start_date", "type": "date_uniform",
           "reference_date": date(2024,10,1), "args": (0, 30)}
    seed : Optional[int]
        Random seed for the sampling engine.
    scramble : bool
        Whether to use Owen scrambling in the sampling engine.
    enforce_compartment_sum_equal : bool, default=False
        If True, enforce sum(compartments) == total when under budget via
        a deterministic water-filling pass. If False, allow sum <= total.
    sampling_method : Literal["lhs", "sobol"] or Callable, default="lhs"
        The sampling method to use:
        - "lhs": Latin Hypercube Sampling (default)
        - "sobol": Sobol sequences (low-discrepancy)
        - Callable: Custom sampling function with signature
          (n_samples, dimension, seed, scramble) -> ndarray

    Returns
    -------
    comps_dict : Dict[str, ndarray]
        Each compartment name maps to an (n_samples,) vector of values.
        Also includes a "_residual" entry with total - sum(compartments).
    params_dict : Dict[str, ndarray]
        Each parameter name maps to an (n_samples,) vector of values.
    U_all : (n_samples, n_compartments + n_parameters) ndarray
        The underlying uniform samples. Keep this if you need to audit or reproduce
        the exact design-to-parameter mapping.

    Examples
    --------
    >>> # Using Latin Hypercube Sampling (default)
    >>> comps, params, U = generate_parameter_samples(
    ...     n_samples=100,
    ...     compartment_names=["S", "E", "I", "R"],
    ...     mins=[0.1, 0.0, 0.0, 0.0],
    ...     maxs=[0.9, 0.3, 0.3, 0.5],
    ...     sampling_method="lhs"
    ... )

    >>> # Using Sobol sequences (powers of 2 recommended)
    >>> comps, params, U = generate_parameter_samples(
    ...     n_samples=256,  # 2^8
    ...     compartment_names=["S", "E", "I", "R"],
    ...     mins=[0.1, 0.0, 0.0, 0.0],
    ...     maxs=[0.9, 0.3, 0.3, 0.5],
    ...     sampling_method="sobol",
    ...     scramble=False  # Deterministic sequence
    ... )

    >>> # Using a custom sampling function
    >>> def my_sampler(n_samples, dimension, seed=None, scramble=True):
    ...     # Your custom implementation
    ...     return np.random.random((n_samples, dimension))
    >>> comps, params, U = generate_parameter_samples(
    ...     n_samples=100,
    ...     compartment_names=["S", "E"],
    ...     mins=[0.1, 0.0],
    ...     maxs=[0.9, 0.5],
    ...     sampling_method=my_sampler
    ... )

    Notes
    -----
    - LHS provides good stratification and is generally robust for any n_samples
    - Sobol sequences provide lower discrepancy but work best with powers of 2
    - Unscrambled Sobol sequences are deterministic (same points every run)
    - Custom samplers should return uniform samples on [0,1]^dimension
    """
    # ---- Make the dimensions explicit for readability ----
    n_compartments: int = len(compartment_names)
    n_parameters: int = 0 if param_specs is None else len(param_specs)
    total_dimension: int = n_compartments + n_parameters

    # Validate lengths early to fail fast with a clear error message.
    if not (len(mins) == len(maxs) == n_compartments):
        err_msg = "Length mismatch: mins/maxs must match number of compartments."
        raise ValueError(err_msg)

    # ---- Select and apply the sampling method ----
    if callable(sampling_method):
        # Custom sampling function
        U_all = sampling_method(n_samples=n_samples, dimension=total_dimension, seed=seed, scramble=scramble)
    elif sampling_method == "lhs":
        U_all = _sample_lhs(n_samples=n_samples, dimension=total_dimension, seed=seed, scramble=scramble)
    elif sampling_method == "sobol":
        # Optionally warn if n_samples is not a power of 2
        if n_samples & (n_samples - 1) != 0:
            warn_msg = f"n_samples={n_samples} is not a power of 2. Sobol sequences perform best with powers of 2 (e.g., 64, 128, 256, ...)."
            logger.warning(warn_msg)

        U_all = _sample_sobol(n_samples=n_samples, dimension=total_dimension, seed=seed, scramble=scramble)
    else:
        err_msg = f"Unknown sampling_method: {sampling_method!r}. Use 'lhs', 'sobol', or provide a callable."
        raise ValueError(err_msg)

    # Validate the shape of the returned samples
    if U_all.shape != (n_samples, total_dimension):
        err_msg = f"Sampling method returned wrong shape. Expected ({n_samples}, {total_dimension}), got {U_all.shape}"
        raise ValueError(err_msg)

    # Split U into the compartments block and the parameters block
    U_comp = U_all[:, :n_compartments] if n_compartments > 0 else np.empty((n_samples, 0))
    U_param = U_all[:, n_compartments:] if n_parameters > 0 else np.empty((n_samples, 0))

    # ---- Map uniform samples to compartment proportions, given constraints ----
    P, residual = _map_compartments_from_uniform(
        U_comp,
        mins=np.asarray(mins, dtype=float),
        maxs=np.asarray(maxs, dtype=float),
        total=total,
        enforce_sum_equal=enforce_compartment_sum_equal,
    )
    comps_dict: dict[str, np.ndarray] = {name: P[:, j] for j, name in enumerate(compartment_names)}
    comps_dict["_residual"] = residual  # slack: total - sum(compartments)

    # ---- Map other parameters deterministically (examples: uniform, date) ----
    params_dict: dict[str, np.ndarray] = {}
    if n_parameters > 0:
        for j, spec in enumerate(param_specs):
            u = U_param[:, j]  # Uniform samples for this parameter
            typ = spec["type"]
            name = spec["name"]

            if typ == "uniform":
                lo, hi = spec["args"]
                params_dict[name] = _map_uniform(u, lo, hi)

            elif typ == "randint":
                lo, hi = spec["args"]
                params_dict[name] = _map_randint(u, lo, hi)

            elif typ == "date_uniform":
                ref: date = spec["reference_date"]
                lo_days, hi_days = spec["args"]
                params_dict[name] = _map_date_uniform(u, ref, lo_days, hi_days)

            else:
                err_msg = f"Unknown parameter type: {typ!r}. Add a deterministic mapping function if needed."
                raise ValueError(err_msg)

    return SamplingResult(compartments=comps_dict, parameters=params_dict, uniform_samples=U_all)


# ============================================================
# Util: Mapping functions from U[0,1] to compartment proportions
# ============================================================
def _project_onto_capped_simplex(
    y: np.ndarray, caps: np.ndarray, budget: float, *, tol: float = 1e-12, max_iter: int = 100
) -> np.ndarray:
    """
    L2-project a vector y onto the "capped simplex"
        S = { z : 0 <= z_i <= caps_i,  sum_i z_i = budget }.

    This is used inside `map_compartments_from_uniform` function to project uniform samples onto the simplex with constraints (min/max for each compartment).

    Intuition
    ---------
    - If you simply scale y by (budget / sum(y)), you may violate z_i <= caps_i.
      This function instead finds the *closest* feasible z (in Euclidean distance)
      to y, respecting both the per-dimension caps and the exact sum constraint.
    - The solution can be expressed as:
          z_i = clip(y_i - tau, 0, caps_i)
      for some "waterline" tau chosen so that sum_i z_i == budget.

    Algorithm
    ---------
    - First, clip y to [0, caps]. If the sum is already <= budget (within tol),
      we are done: return the clipped vector.
    - Otherwise, perform a bisection search over tau such that
          sum_i clip(y_i - tau, 0, caps_i) == budget.

    Parameters
    ----------
    y : (k,) array_like
        Proposed increments per dimension (before enforcing budget).
    caps : (k,) array_like
        Per-dimension upper bounds for the increments (i.e., maxs - mins).
    budget : float
        The required total sum of the increments after projection.
    tol : float
        Tolerance on the sum equality (numerical stopping criterion).
    max_iter : int
        Maximum number of bisection iterations.

    Returns
    -------
    z : (k,) ndarray
        The projected increments satisfying 0 <= z_i <= caps_i and sum z == budget
        (within numerical tolerance).

    Complexity
    ----------
    Each bisection step is O(k) due to the elementwise clip and sum.
    Total complexity ~ O(k * log(1/tol)).

    See Also
    --------
    - "Water-filling" interpretations in resource allocation problems.
    - Projection onto a simplex with box constraints (capped simplex).
    """
    y = np.asarray(y, dtype=float)
    caps = np.asarray(caps, dtype=float)

    # First try the trivial case: clip to [0, caps] and check the sum.
    z = np.clip(y, 0.0, caps)
    s = z.sum()
    if s <= budget + tol:
        # Already within budget; no need to reduce further.
        return z

    # Otherwise, find tau such that sum clip(y - tau, 0, caps) = budget.
    # Lower/upper bounds for tau:
    #  - If tau <= min(y - caps), then clip(y - tau, 0, caps) = caps (sum too large).
    #  - If tau >= max(y), then clip(...) = 0 (sum too small).
    lo = np.min(y - caps)
    hi = np.max(y)

    for _ in range(max_iter):
        tau = 0.5 * (lo + hi)
        z = np.clip(y - tau, 0.0, caps)
        s = z.sum()

        # Stop when the total is sufficiently close to the budget.
        if abs(s - budget) <= tol:
            return z

        # If sum is still too large, we need to raise tau (subtract more).
        if s > budget:
            lo = tau
        else:
            hi = tau

    # Fallback: return the last iterate (should be very close within tolerance).
    return z


def _map_compartments_from_uniform(
    U_comp: np.ndarray, mins: np.ndarray, maxs: np.ndarray, *, total: float = 1.0, enforce_sum_equal: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map uniform samples U_comp to compartment proportion values p satisfying:
      mins <= p <= maxs  (elementwise)  and  sum(p) <= total (=1.0 by default).

    Strategy (deterministic)
    --------------------------------------------
    1) Reserve the lower bounds: start from p := mins.
       This guarantees p_i >= mins_i.
    2) Compute per-dimension "capacity" after mins:
          caps := maxs - mins
       and the total "budget" left after mins:
          budget := total - sum(mins)
       (If sum(mins) > total, the constraints are infeasible.)
    3) Propose increments y := caps * u  (where u is one row from U_comp).
       - If sum(y) <= budget (within tolerance):
            set z := clip(y, 0, caps).
            If enforce_sum_equal=True and we still have leftover (budget - sum(z)),
            perform a simple *deterministic* water-filling pass to distribute it
            without exceeding caps.
       - Else (sum(y) > budget):
            z := project_onto_capped_simplex(y, caps, budget)
            to exactly match the total budget while respecting caps.
    4) Return P := mins + z, and residual := total - sum(P) (>= 0).
       If enforce_sum_equal=True and sum(mins) <= total, residual will be 0.

    Parameters
    ----------
    U_comp : (n_samples, n_compartments) ndarray
        Uniform samples for the compartment block (each row is one draw).
    mins, maxs : (n_compartments,) array_like
        Per-compartment lower/upper bounds. Must satisfy maxs_i >= mins_i.
    total : float, default=1.0
        Upper bound on the sum of the compartments: sum(P) <= total.
    enforce_sum_equal : bool, default=False
        If True, enforce sum(P) == total when we are *under* budget by
        water-filling the leftover in a deterministic single pass.
        If False, allow sum(P) <= total.

    Returns
    -------
    P: (n_samples, n_compartments) ndarray
        Feasible compartment proportion values per sample.
    residual : (n_samples,) ndarray
        total - sum(P, axis=1), i.e., nonnegative slack. Becomes zero when
        enforce_sum_equal=True and the problem is feasible.

    Failure Modes
    -------------
    - If sum(mins) > total (beyond a small tolerance), the problem is infeasible
      and a ValueError is raised.

    Notes
    -----
    - We use a single-pass water-filling when under budget and the user requests
      exact equality. This keeps the mapping deterministic and simple. For more
      sophisticated tie-breaking or fairness criteria, implement a custom
      allocator here.
    """
    mins = np.asarray(mins, dtype=float)
    maxs = np.asarray(maxs, dtype=float)
    if np.any(maxs < mins):
        err_msg = "Each element of 'maxs' must be >= the corresponding element of 'mins'."
        raise ValueError(err_msg)

    # Sum of lower bounds: must not exceed the total.
    min_sum = float(np.sum(mins))
    if min_sum > total + 1e-12:
        err_msg = f"Infeasible bounds: sum(mins)={min_sum} exceeds total={total}."
        raise ValueError(err_msg)

    # Capacity after reserving mins, and the remaining budget we are allowed to add.
    caps = maxs - mins
    budget = total - min_sum

    n_samples, n_compartments = U_comp.shape
    P = np.empty((n_samples, n_compartments), dtype=float)
    residual = np.empty(n_samples, dtype=float)

    for i in range(n_samples):
        # Proposed increments per dimension in [0, caps].
        u = U_comp[i]  # one uniform sample row for compartments
        y = caps * u  # elementwise scaling up to caps

        if y.sum() <= budget + 1e-12:
            # Under budget: just clip to [0, caps] to guard roundoff.
            z = np.clip(y, 0.0, caps)

            # Optionally make the sum exactly equal to 'budget' by
            # water-filling the leftover deterministically.
            if enforce_sum_equal:
                leftover = budget - float(np.sum(z))
                if leftover > 0.0:
                    # Deterministic water-filling:
                    # Iterate in a fixed order and fill as much as possible
                    # until caps are reached or the leftover is consumed.
                    for j in range(n_compartments):
                        if leftover <= 0.0:
                            break
                        room = float(caps[j] - z[j])
                        if room <= 0.0:
                            continue
                        add = min(room, leftover)
                        z[j] += add
                        leftover -= add
        else:
            # Over budget: project onto capped simplex to hit the exact budget.
            z = _project_onto_capped_simplex(y, caps, budget)

        # Add mins back to form the final compartment proportions and record residual slack.
        p = mins + z
        P[i, :] = p
        residual[i] = max(0.0, total - float(np.sum(p)))  # nonnegative by construction

    return P, residual


# ============================================================
# Util: Mapping functions from U[0,1]
# ============================================================
def _map_uniform(u: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """
    Map U from [0,1] to Uniform[lo, hi].

    Parameters
    ----------
    u : (n,) ndarray
        Uniform samples for a single parameter.
    lo, hi : float
        Lower/upper endpoints of the target uniform distribution.

    Returns
    -------
    x : (n,) ndarray
        Transformed samples in [lo, hi].
    """
    return lo + (hi - lo) * u


def _map_randint(u: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """
    Map U from [0,1] to randint(lo, hi).

    Parameters
    ----------
    u : (n,) ndarray
        Uniform samples for a single parameter.
    lo, hi : int
        Lower (inclusive) and upper (exclusive) bounds of randint.

    Returns
    -------
    x : (n,) ndarray
        Transformed integer samples in [lo, hi).
    """
    # return (lo + np.floor((hi - lo) * u)).astype(int)
    return lo + np.floor(u * (hi - lo)).astype(int)


def _map_date_uniform(u: np.ndarray, reference_date: date, lo_days: int, hi_days: int) -> np.ndarray:
    """
    Deterministically map U in [0,1] to dates by adding Uniform[lo_days, hi_days]
    days to a reference date.

    Implementation detail:
      We floor the day increments to integers to keep tidy (whole-day) dates.

    Parameters
    ----------
    u : (n,) ndarray
        Uniform samples for a date parameter.
    reference_date : date
        Origin date to which we add sampled day offsets.
    lo_days, hi_days : int
        Bounds for the uniform number of days to add (inclusive lower, exclusive upper
        once floored).

    Returns
    -------
    dates : (n,) ndarray of datetime.date
        Deterministically transformed date samples.
    """
    days = np.floor(lo_days + (hi_days - lo_days) * u).astype(int)
    return np.array([reference_date + timedelta(days=int(k)) for k in days])
