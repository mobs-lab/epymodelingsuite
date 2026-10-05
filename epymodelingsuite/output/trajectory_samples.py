"""Trajectory samples for hubverse (FluSight) submissions."""

from collections.abc import Callable
from datetime import date
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..utils.location import convert_location_name_format, get_hub_location_id

if TYPE_CHECKING:
    from ..schema.dispatcher import CalibrationOutput
    from ..schema.output import FlusightForecastOutput


def select_random_indices(values: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Select trajectory indices uniformly without replacement.

    Parameters
    ----------
    values : np.ndarray
        Complete trajectories with shape ``(trajectories, horizons)``.
    n : int
        Maximum number of trajectories to select.
    rng : np.random.Generator
        Random number generator used for selection.

    Returns
    -------
    np.ndarray
        Sorted integer row indices into ``values``. All indices are returned
        when there are at most ``n`` trajectories.
    """
    if len(values) <= n:
        return np.arange(len(values))
    return np.sort(rng.choice(len(values), size=n, replace=False))


# Name -> selector. A selector gets one location's complete trajectories (rows x horizons), the number of
# samples wanted and a seeded generator, and returns the selected row indices. Register new methods here.
SAMPLE_SELECTORS: dict[str, Callable[[np.ndarray, int, np.random.Generator], np.ndarray]] = {
    "random": select_random_indices,
}


def select_trajectory_indices(
    values: np.ndarray, n: int, method: str = "random", seed: int | None = None
) -> np.ndarray:
    """
    Select row indices of up to `n` complete trajectories.

    Trajectories with a NaN in any horizon can't be submitted and are never selected. If fewer than `n`
    complete trajectories remain, return all their indices without resampling.

    Parameters
    ----------
    values : np.ndarray
        Values with shape (trajectories, horizons).
    n : int
        Number of samples wanted.
    method : str, optional
        Key of ``SAMPLE_SELECTORS``. Defaults to ``"random"``.
    seed : int or None, optional
        Seed for the selector's random generator. Defaults to unseeded selection.

    Returns
    -------
    np.ndarray
        Integer row indices into the original ``values`` array, after excluding
        trajectories with a NaN at any horizon.

    Raises
    ------
    KeyError
        If ``method`` is not registered in ``SAMPLE_SELECTORS``.
    """
    complete = np.flatnonzero(~np.isnan(values).any(axis=1))
    chosen = SAMPLE_SELECTORS[method](values[complete], n, np.random.default_rng(seed))
    return complete[chosen]


def trajectories_to_sample_rows(  # noqa: PLR0913
    values: np.ndarray,
    horizons: list[int],
    reference_date: date,
    location: str,
    target: str,
    id_prefix: str,
    *,
    integer: bool = False,
    upper: float | None = None,
) -> pd.DataFrame:
    """
    Convert selected trajectories into a hub-format sample table.

    Each trajectory becomes one row per horizon, sharing a sample id across
    those rows. Values are clipped and optionally rounded for the target.

    Parameters
    ----------
    values : np.ndarray
        Selected trajectories with shape (samples, horizons).
    horizons : list[int]
        Horizon of each column of `values`.
    reference_date : date
        Submission reference date.
    location : str
        Hub location id (e.g. FIPS code).
    target : str
        Hub target name.
    id_prefix : str
        Prefix of `output_type_id`; samples are numbered `<prefix>00`, `<prefix>01`, ...
    integer : bool, optional
        Round values to integers (counts). Defaults to False.
    upper : float or None, optional
        Upper clipping bound, such as 1 for proportions. Defaults to no upper
        bound. Values are always clipped to be non-negative.

    Returns
    -------
    pd.DataFrame
        One row per sample and horizon with the hub columns.
    """
    values = np.clip(values, 0, upper)
    if integer:
        # NumPy rint rounds exact .5 ties to the nearest even integer (10.5 -> 10, 11.5 -> 12),
        # matching pandas round used for hospitalization quantiles.
        values = np.rint(values)
    n_samples, n_horizons = values.shape
    ref = pd.Timestamp(reference_date)
    return pd.DataFrame(
        {
            "reference_date": ref.date(),
            "horizon": np.tile(horizons, n_samples),
            "target_end_date": [(ref + pd.Timedelta(weeks=h)).date() for h in horizons] * n_samples,
            "location": location,
            "target": target,
            "output_type": "sample",
            "output_type_id": np.repeat([f"{id_prefix}{i:02d}" for i in range(n_samples)], n_horizons),
            "value": values.reshape(-1).astype(float),
        }
    )


def _build_horizon_matrix(
    dates_list: list[np.ndarray], values_list: list[np.ndarray], reference_date: date, horizons: list[int]
) -> np.ndarray:
    """Align trajectory values to the requested weekly forecast horizons.

    Parameters
    ----------
    dates_list : list[np.ndarray]
        Date arrays, one per trajectory.
    values_list : list[np.ndarray]
        Value arrays aligned one-to-one with ``dates_list`` and their dates.
    reference_date : date
        Date corresponding to horizon zero.
    horizons : list[int]
        Week offsets from ``reference_date``, in the desired column order.

    Returns
    -------
    np.ndarray
        Float matrix of shape ``(trajectories, horizons)``. Entries are NaN
        where a trajectory has no value at the exact target date.

    Raises
    ------
    ValueError
        If trajectory counts or corresponding date/value array lengths differ.

    Examples
    --------
    With reference date 2026-10-10, horizons -1, 0 and 1 correspond to
    2026-10-03, 2026-10-10 and 2026-10-17. Rows represent trajectories and
    columns follow the requested horizon order. The second trajectory has no
    value on 2026-10-10, so its horizon-zero entry is NaN. Dates are matched
    exactly, without aggregation or interpolation.

    >>> dates_list = [
    ...     np.array(["2026-10-03", "2026-10-10", "2026-10-17"], dtype="datetime64[D]"),
    ...     np.array(["2026-10-03", "2026-10-17"], dtype="datetime64[D]"),
    ... ]
    >>> values_list = [np.array([10, 20, 30]), np.array([100, 300])]
    >>> _build_horizon_matrix(dates_list, values_list, date(2026, 10, 10), [-1, 0, 1])
    array([[ 10.,  20.,  30.],
           [100.,  nan, 300.]])
    """
    target_dates = [pd.Timestamp(reference_date) + pd.Timedelta(weeks=h) for h in horizons]
    out = np.full((len(values_list), len(horizons)), np.nan)
    for i, (dates, values) in enumerate(zip(dates_list, values_list, strict=True)):
        lookup = dict(zip(pd.to_datetime(dates), values, strict=True))
        out[i] = [lookup.get(d, np.nan) for d in target_dates]
    return out


def build_flusight_trajectory_samples(
    calibrations: list["CalibrationOutput"],
    flusight_format: "FlusightForecastOutput",
    rescaling_factors: pd.DataFrame,
) -> tuple[list[pd.DataFrame], list[str]]:
    """
    Build FluSight trajectory sample tables from calibration results.

    For each location and enabled target, align projections to weekly horizons,
    select trajectories with ``select_trajectory_indices``, and convert them to
    rows with ``trajectories_to_sample_rows``.

    Hospitalization samples come from the 'hospitalizations' projections. Prop ED samples come from the
    `transition_name` projections ('transition' strategy), or are the hospitalization trajectories times the
    location's rescaling factor (window strategies), in which case both targets share sample ids.

    Parameters
    ----------
    calibrations : list[CalibrationOutput]
        One calibration per location.
    flusight_format : FlusightForecastOutput
        FluSight output configuration with `samples` set.
    rescaling_factors : pd.DataFrame
        Prop ED rescaling factors (`population`, `rescaling_factor`) for the window strategies.

    Returns
    -------
    rows : list[pd.DataFrame]
        Sample tables per location and enabled target over horizons -1 through 3.
        Window-based ED outputs use the selected hospitalization trajectories.
    warnings : list[str]
        Diagnostics for skipped models or targets, duplicate locations, and
        locations with fewer complete trajectories than requested.
    """
    cfg = flusight_format.samples
    prop_ed = flusight_format.prop_ed
    # FluSight samples cover the previous week, reference week, and next three weeks.
    horizons = list(range(-1, 4))

    # Window strategies convert hospitalization trajectories to ED proportions.
    # Key the factors by the same hub location IDs used for calibration outputs.
    factors = {}
    if prop_ed and prop_ed.strategy != "transition" and not rescaling_factors.empty:
        factors = dict(
            zip(
                rescaling_factors.population.map(get_hub_location_id),
                rescaling_factors.rescaling_factor,
                strict=True,
            )
        )

    rows, warns, seen = [], [], set()
    for calibration in calibrations:
        location = get_hub_location_id(calibration.population)
        # Keep only the first calibration per location to avoid duplicate sample IDs.
        if location in seen:
            warns.append(f"OUTPUT GENERATOR: more than one model for location {location}; samples kept for the first.")
            continue
        seen.add(location)

        # Read the raw projection trajectories. An extraction failure skips this
        # calibration while allowing the remaining locations to be processed.
        try:
            traj = calibration.results.get_projection_trajectories()
        except Exception:
            warns.append(
                f"OUTPUT GENERATOR: failed to obtain projection trajectories for samples, model primary_id={calibration.primary_id}."
            )
            continue

        # An output-level seed overrides the calibration seed. If both are None,
        # selection is unseeded.
        seed = cfg.seed if cfg.seed is not None else calibration.seed
        id_prefix = convert_location_name_format(calibration.population, "abbreviation")
        ref = flusight_format.reference_date

        # Collect (target, selected values with shape (samples, horizons), formatting options).
        # Keep values unrounded until row formatting so window ED can use the original counts.
        per_target = []
        try:
            # Window ED also needs hospitalization trajectories when hospitalization
            # output is disabled. Align to the requested weeks, then select whole
            # trajectories with no missing horizons. Select once so window ED reuses
            # exactly these rows, including when selection is unseeded.
            if flusight_format.hospitalizations or (prop_ed and prop_ed.strategy != "transition"):
                hosp = _build_horizon_matrix(traj["date"], traj["hospitalizations"], ref, horizons)
                hosp = hosp[select_trajectory_indices(hosp, cfg.n_samples, cfg.method, seed)]
            if flusight_format.hospitalizations:
                per_target.append((flusight_format.hospitalizations.target, hosp, {"integer": True}))
            if prop_ed and prop_ed.strategy == "transition":
                # Transition ED uses its own configured projection variable and
                # selects its trajectories separately from the hospitalization target.
                ed = _build_horizon_matrix(traj["date"], traj[prop_ed.transition_name], ref, horizons)
                ed = ed[select_trajectory_indices(ed, cfg.n_samples, cfg.method, seed)]
                per_target.append((prop_ed.target, ed, {"upper": 1}))
            elif prop_ed and location in factors:
                # Scale the selected, unrounded hospitalization values. Preserving
                # their row order keeps hospitalization and ED sample IDs paired.
                per_target.append((prop_ed.target, hosp * factors[location], {"upper": 1}))
            elif prop_ed:
                # Missing factors skip ED only; any requested hospitalization rows remain.
                warns.append(f"OUTPUT GENERATOR: no prop ED rescaling factor for {location}; skipping ED samples.")
        except (KeyError, ValueError) as e:
            # Invalid projection data skips all sample targets for this location.
            warns.append(f"OUTPUT GENERATOR: failed to create samples for {location}: {e}")
            continue

        for target, values, options in per_target:
            # Rescaling can introduce NaNs even in complete selected trajectories.
            values = values[~np.isnan(values).any(axis=1)]
            # Submit the available complete trajectories without duplicating them
            # to reach the requested count; report the shortfall to the caller.
            if len(values) < cfg.n_samples:
                warns.append(
                    f"OUTPUT GENERATOR: only {len(values)} complete trajectories for '{target}' samples in {location} "
                    f"(requested {cfg.n_samples})."
                )
            # Expand each trajectory into one row per horizon, adding dates and IDs.
            # Formatting clips counts at zero and rounds them, or clips ED to [0, 1].
            rows.append(trajectories_to_sample_rows(values, horizons, ref, location, target, id_prefix, **options))

    # The caller combines these tables with other forecast types and serializes them.
    return rows, warns
