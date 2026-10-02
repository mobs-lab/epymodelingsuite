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


def select_random(values: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Pick `n` rows of `values` (trajectories x horizons) uniformly without replacement, or all if fewer."""
    if len(values) <= n:
        return np.arange(len(values))
    return np.sort(rng.choice(len(values), size=n, replace=False))


# Name -> selector. A selector gets one location's complete trajectories (rows x horizons), the number of
# samples wanted and a seeded generator, and returns the selected row indices. Register new methods here.
SAMPLE_SELECTORS: dict[str, Callable[[np.ndarray, int, np.random.Generator], np.ndarray]] = {
    "random": select_random,
}


def select_samples(values: np.ndarray, n: int, method: str = "random", seed: int | None = None) -> np.ndarray:
    """
    Select up to `n` trajectories to submit as samples.

    Trajectories with a NaN in any horizon can't be submitted and are never selected. If fewer than `n`
    complete trajectories remain, all of them are returned (no resampling).

    Parameters
    ----------
    values : np.ndarray
        Values with shape (trajectories, horizons).
    n : int
        Number of samples wanted.
    method : str
        Key of `SAMPLE_SELECTORS`.
    seed : int | None
        Seed for the selector's random generator.

    Returns
    -------
    np.ndarray
        Row indices into `values`.
    """
    complete = np.flatnonzero(~np.isnan(values).any(axis=1))
    chosen = SAMPLE_SELECTORS[method](values[complete], n, np.random.default_rng(seed))
    return complete[chosen]


def make_sample_rows(  # noqa: PLR0913
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
    Build hub rows for selected sample trajectories.

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
    integer : bool
        Round values to integers (counts).
    upper : float | None
        Clip values to at most this (e.g. 1 for proportions). Values are always clipped to be non-negative.

    Returns
    -------
    pd.DataFrame
        One row per sample and horizon with the hub columns.
    """
    values = np.clip(values, 0, upper)
    if integer:
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
    """Return the values of each trajectory at the horizons' target dates, shape (trajectories, horizons); NaN where missing."""
    target_dates = [pd.Timestamp(reference_date) + pd.Timedelta(weeks=h) for h in horizons]
    out = np.full((len(values_list), len(horizons)), np.nan)
    for i, (dates, values) in enumerate(zip(dates_list, values_list, strict=True)):
        lookup = dict(zip(pd.to_datetime(dates), values, strict=True))
        out[i] = [lookup.get(d, np.nan) for d in target_dates]
    return out


def make_flusight_samples(
    calibrations: list["CalibrationOutput"],
    flusight_format: "FlusightForecastOutput",
    rescaling_factors: pd.DataFrame,
) -> tuple[list[pd.DataFrame], list[str]]:
    """
    Create FluSight trajectory sample rows for the enabled hospitalization and prop ED targets.

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
    tuple[list[pd.DataFrame], list[str]]
        Sample rows per location and target, and warnings.
    """
    cfg = flusight_format.samples
    prop_ed = flusight_format.prop_ed
    horizons = list(range(-1, 4))
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
        if location in seen:
            warns.append(f"OUTPUT GENERATOR: more than one model for location {location}; samples kept for the first.")
            continue
        seen.add(location)
        try:
            traj = calibration.results.get_projection_trajectories()
        except Exception:
            warns.append(
                f"OUTPUT GENERATOR: failed to obtain projection trajectories for samples, model primary_id={calibration.primary_id}."
            )
            continue
        seed = cfg.seed if cfg.seed is not None else calibration.seed
        id_prefix = convert_location_name_format(calibration.population, "abbreviation")
        ref = flusight_format.reference_date

        # (target, selected values with shape (samples, horizons), rounding options)
        per_target = []
        try:
            if flusight_format.hospitalizations or (prop_ed and prop_ed.strategy != "transition"):
                hosp = _build_horizon_matrix(traj["date"], traj["hospitalizations"], ref, horizons)
                hosp = hosp[select_samples(hosp, cfg.n_samples, cfg.method, seed)]
            if flusight_format.hospitalizations:
                per_target.append((flusight_format.hospitalizations.target, hosp, {"integer": True}))
            if prop_ed and prop_ed.strategy == "transition":
                ed = _build_horizon_matrix(traj["date"], traj[prop_ed.transition_name], ref, horizons)
                ed = ed[select_samples(ed, cfg.n_samples, cfg.method, seed)]
                per_target.append((prop_ed.target, ed, {"upper": 1}))
            elif prop_ed and location in factors:
                per_target.append((prop_ed.target, hosp * factors[location], {"upper": 1}))
            elif prop_ed:
                warns.append(f"OUTPUT GENERATOR: no prop ED rescaling factor for {location}; skipping ED samples.")
        except (KeyError, ValueError) as e:
            warns.append(f"OUTPUT GENERATOR: failed to create samples for {location}: {e}")
            continue

        for target, values, options in per_target:
            # Rescaling can introduce NaNs even in complete selected trajectories.
            values = values[~np.isnan(values).any(axis=1)]
            if len(values) < cfg.n_samples:
                warns.append(
                    f"OUTPUT GENERATOR: only {len(values)} complete trajectories for '{target}' samples in {location} "
                    f"(requested {cfg.n_samples})."
                )
            rows.append(make_sample_rows(values, horizons, ref, location, target, id_prefix, **options))
    return rows, warns
