import pandas as pd

from ..output.trajectory_samples import select_trajectory_indices, trajectories_to_sample_rows
from ..schema.output import (
    FlusightTrajectorySamples,
    get_flusight_categorical_horizons,
    get_flusight_horizons,
    get_flusight_quantiles,
    get_metrocast_horizons,
    get_metrocast_quantiles,
)
from ..utils.location import convert_location_name_format

# fmt: off

MIN_COUNT_CHANGE = 10
RATE_TREND_CATEGORIES = ["large_decrease", "decrease", "stable", "increase", "large_increase"]

# Population data (2024 Census estimates)
POPULATION = {
    "US": 340110988,
    "AL": 5157699, "AK": 740133, "AZ": 7582384, "AR": 3088354,
    "CA": 39431263, "CO": 5957493, "CT": 3675069, "DE": 1051917,
    "DC": 702250, "FL": 23372215, "GA": 11180878, "HI": 1446146,
    "ID": 2001619, "IL": 12710158, "IN": 6924275, "IA": 3241488,
    "KS": 2970606, "KY": 4588372, "LA": 4597740, "ME": 1405012,
    "MD": 6263220, "MA": 7136171, "MI": 10140459, "MN": 5793151,
    "MS": 2943045, "MO": 6245466, "MT": 1137233, "NE": 2005465,
    "NV": 3267467, "NH": 1409032, "NJ": 9500851, "NM": 2130256,
    "NY": 19867248, "NC": 11046024, "ND": 796568, "OH": 11883304,
    "OK": 4095393, "OR": 4272371, "PA": 13078751, "RI": 1112308,
    "SC": 5478831, "SD": 924669, "TN": 7227750, "TX": 31290831,
    "UT": 3503613, "VT": 648493, "VA": 8811195, "WA": 7958180,
    "WV": 1769979, "WI": 5960975, "WY": 587618, "PR": 3203295
}

# Rate trend thresholds by horizon (per 100k population)
RATE_TREND_THRESHOLDS = {
    0: (0.3, 1.7),
    1: (0.5, 3.0),
    2: (0.7, 4.0),
    3: (1.0, 5.0),
}
# fmt: on


def cols_to_dt(
    dataframe: pd.DataFrame,
    names: list,
):
    """
    Convert given columns to datetime objects.
    """
    df = dataframe.copy()
    for name in names:
        df[name] = pd.to_datetime(df[name])
    return df


def classify_rate_trend(count_change: float, population: int, horizon: int) -> str:
    """Classify hospitalization count change into rate trend category."""
    stable_thresh, large_thresh = RATE_TREND_THRESHOLDS.get(horizon, RATE_TREND_THRESHOLDS[3])
    rate_change = (count_change / population) * 100000

    if abs(rate_change) < stable_thresh or abs(count_change) < MIN_COUNT_CHANGE:
        return "stable"

    if rate_change >= large_thresh:
        return "large_increase"
    if rate_change > 0:
        return "increase"
    if rate_change <= -large_thresh:
        return "large_decrease"
    return "decrease"


def compute_quantiles(
    df: pd.DataFrame,
    quantiles: list[float] | None = None,
    value_col: str = "target_total",
    date_col: str = "date",
    location_col: str = "location",
) -> pd.DataFrame:
    """Compute quantiles from sampled trajectories."""
    if quantiles is None:
        quantiles = get_flusight_quantiles()

    result = df.groupby([location_col, date_col])[value_col].quantile(quantiles).unstack()
    result.columns = [f"q{int(q * 1000):03d}" for q in quantiles]
    return result.reset_index()


def compute_rate_trend_categories(
    df: pd.DataFrame,
    reference_date: str,
    surveillance_df: pd.DataFrame,
    horizons: list = None,
    value_col: str = "target_total",
    surveillance_date_col: str = "date",
    surveillance_value_col: str = "hospitalizations",
    surveillance_location_col: str = "location",
) -> pd.DataFrame:
    """Compute rate trend category for each trajectory at each horizon. Surveillance is keyed by FIPS code."""
    if horizons is None:
        horizons = get_flusight_categorical_horizons()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    reference_date = pd.to_datetime(reference_date)

    surv = surveillance_df.copy()
    surv[surveillance_date_col] = pd.to_datetime(surv[surveillance_date_col])
    baseline_date = reference_date - pd.Timedelta(weeks=1)

    results = []

    for pop in df["location"].unique():
        abbrev = convert_location_name_format(value=pop, output_format="abbreviation", location_type="iso")
        location = _epydemix_to_fips(pop)
        pop_data = df[df["location"] == pop]
        population_size = POPULATION.get(abbrev, POPULATION.get("US"))

        surv_pop = surv[surv[surveillance_location_col] == location]
        baseline_row = surv_pop[surv_pop[surveillance_date_col] == baseline_date]

        if len(baseline_row) == 0:
            print(f"Warning: No surveillance data for {abbrev} ({location}) on {baseline_date}")
            continue

        baseline_val = baseline_row[surveillance_value_col].values[0]

        for horizon in horizons:
            target_date = reference_date + pd.Timedelta(weeks=horizon)
            target_data = pop_data[pop_data["date"] == target_date]

            for _, row in target_data.iterrows():
                sample_id = row["sample_id"]
                target_val = row[value_col]

                if pd.isna(baseline_val) or pd.isna(target_val):
                    continue

                count_change = target_val - baseline_val
                category = classify_rate_trend(count_change, population_size, horizon)

                results.append(
                    {
                        "reference_date": reference_date,
                        "sample_id": sample_id,
                        "population": pop,
                        "abbreviation": abbrev,
                        "location": location,
                        "baseline_date": baseline_date,
                        "horizon": horizon,
                        "target_date": target_date,
                        "baseline_value": baseline_val,
                        "target_value": target_val,
                        "count_change": count_change,
                        "rate_change": (count_change / population_size) * 100000,
                        "category": category,
                    }
                )

    return pd.DataFrame(results)


def compute_rate_trend_pmf(categories_df: pd.DataFrame) -> pd.DataFrame:
    """Compute probability mass function for rate trend categories."""
    counts = (
        categories_df.groupby(["population", "abbreviation", "location", "horizon", "target_date", "category"])
        .size()
        .unstack(fill_value=0)
    )

    for cat in RATE_TREND_CATEGORIES:
        if cat not in counts.columns:
            counts[cat] = 0

    counts = counts[RATE_TREND_CATEGORIES]
    pmf = counts.div(counts.sum(axis=1), axis=0)
    return pmf.reset_index()


def _epydemix_to_fips(population: str) -> str:
    abbrev = convert_location_name_format(value=population, output_format="abbreviation", location_type="iso")
    return convert_location_name_format(
        value=abbrev, output_format="FIPS", input_format="abbreviation", location_type="iso"
    )


def _epydemix_to_abbreviation(population: str) -> str:
    return convert_location_name_format(value=population, output_format="abbreviation", location_type="iso")


def _strip_metrocast_prefix(population: str) -> str:
    return population.removeprefix("metrocast_location_")


# Everything that differs between hub submission files. `location` maps an epydemix population name to the
# hub location id. `pmf` adds the FluSight rate-trend target, which needs surveillance for the baseline.
# `sample_prefix` maps the population to the prefix of sample ids (`MA00`, ...); None numbers them 1, 2, ...
# per location, as Metrocast does.
SUBMISSION_PROFILES = {
    "flusight_hosp": {
        "location": _epydemix_to_fips,
        "target": "wk inc flu hosp",
        "decimals": 0,
        "horizons": list(get_flusight_horizons()),
        "quantiles": get_flusight_quantiles(),
        "pmf": True,
        "sample_prefix": _epydemix_to_abbreviation,
    },
    "flusight_ed": {
        "location": _epydemix_to_fips,
        "target": "wk inc flu prop ed visits",
        "decimals": 3,
        "horizons": list(get_flusight_horizons()),
        "quantiles": get_flusight_quantiles(),
        "pmf": False,
        "sample_prefix": _epydemix_to_abbreviation,
    },
    "metrocast": {
        "location": _strip_metrocast_prefix,
        "target": "Flu ED visits pct",
        "decimals": 3,
        "horizons": list(get_metrocast_horizons()),
        "quantiles": get_metrocast_quantiles(),
        "pmf": False,
        "sample_prefix": None,
    },
    "bphc_ed": {
        "location": _strip_metrocast_prefix,
        "target": "wk inc ed signal",
        "decimals": 3,
        "horizons": list(get_flusight_horizons()),
        "quantiles": get_flusight_quantiles(),
        "pmf": False,
        "sample_prefix": None,
    },
}


def create_submission(
    trajectories_df: pd.DataFrame,
    reference_date: str,
    profile: str,
    value_col: str = "target_total",
    surveillance_df: pd.DataFrame | None = None,
    trajectory_samples: FlusightTrajectorySamples | None = None,
) -> pd.DataFrame:
    """
    Create a hubverse submission file from sampled trajectories.

    Parameters
    ----------
    trajectories_df : pd.DataFrame
        Trajectories with `location` (epydemix population), `date`, `sample_id` and `value_col` columns.
    reference_date : str
        Reference date of the submission (horizon 0).
    profile : str
        Key of `SUBMISSION_PROFILES`.
    value_col : str
        Column with the values to summarize.
    surveillance_df : pd.DataFrame | None
        Observed `date`, `location`, `target` (see `read_surveillance`) for the rate-trend baseline.
        Required when the profile has `pmf`.
    trajectory_samples : FlusightTrajectorySamples | None
        If given, add up to `trajectory_samples.n_samples` trajectories per location from `value_col` as the
        'sample' output type. Trajectories missing any horizon are never selected. Unseeded unless
        `trajectory_samples.seed` is set.

    Returns
    -------
    pd.DataFrame
        Submission with the 8 hubverse columns, sorted by location, target, horizon, output type and id.
    """
    profile_spec = SUBMISSION_PROFILES[profile]
    df = trajectories_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    reference_date_dt = pd.to_datetime(reference_date)

    results = []
    sample_tables = []
    # `population` is the epydemix name in the trajectories' `location` column (e.g. "United_States__Massachusetts"),
    # `location` is the hub location id written to the submission (e.g. "25")
    populations = df["location"].unique()
    for population in populations:
        location = profile_spec["location"](population)
        # All sample_ids and dates of this population
        population_trajectories = df[df["location"] == population]
        if trajectory_samples is not None:
            sample_tables.append(
                _build_location_trajectory_samples(
                    population_trajectories,
                    reference_date,
                    profile_spec,
                    population,
                    location,
                    value_col,
                    trajectory_samples,
                )
            )

        for horizon in profile_spec["horizons"]:
            target_end_date = reference_date_dt + pd.Timedelta(weeks=horizon)
            target_data = population_trajectories[population_trajectories["date"] == target_end_date][value_col]

            if len(target_data) == 0:
                continue

            for q in profile_spec["quantiles"]:
                results.append(
                    {
                        "reference_date": reference_date,
                        "horizon": horizon,
                        "target_end_date": target_end_date.strftime("%Y-%m-%d"),
                        "location": location,
                        "target": profile_spec["target"],
                        "output_type": "quantile",
                        "output_type_id": str(q),
                        "value": round(target_data.quantile(q), profile_spec["decimals"]),
                    }
                )
    submission = pd.concat([pd.DataFrame(results), *sample_tables], ignore_index=True)

    if profile_spec["pmf"]:
        if surveillance_df is None:
            raise ValueError(f"Profile '{profile}' needs surveillance data for the rate-trend baseline.")
        categories_df = compute_rate_trend_categories(
            df=trajectories_df,
            reference_date=reference_date,
            surveillance_df=surveillance_df,
            value_col=value_col,
            surveillance_value_col="target",
        )
        pmf_df = compute_rate_trend_pmf(categories_df)
        submission = pd.concat([submission, _pmf_rows(pmf_df, reference_date)], ignore_index=True)

    return submission.sort_values(["location", "target", "horizon", "output_type", "output_type_id"]).reset_index(
        drop=True
    )


def _build_location_trajectory_samples(  # noqa: PLR0913
    population_trajectories: pd.DataFrame,
    reference_date: str,
    profile_spec: dict,
    population: str,
    location: str,
    value_col: str,
    trajectory_samples: FlusightTrajectorySamples,
) -> pd.DataFrame:
    """
    Build the hub 'sample' rows of one location from its aggregated trajectories.

    Multistrain counterpart of the per-location body of `build_flusight_trajectory_samples`. Reshapes the long
    trajectories into a (sample_id x horizon) array, then selects and formats them with the shared
    `select_trajectory_indices` and `trajectories_to_sample_rows`. Trajectories missing any horizon are never
    selected; if fewer than `trajectory_samples.n_samples` complete ones remain, all are used and a warning is
    printed.

    Parameters
    ----------
    population_trajectories : pd.DataFrame
        This population's rows of the aggregated trajectories (all sample_ids and dates), with `sample_id`,
        `date` and `value_col` columns.
    reference_date : str
        Reference date of the submission (horizon 0). Written as is to the `reference_date` column.
    profile_spec : dict
        Entry of `SUBMISSION_PROFILES`. Uses its `horizons`, `target` and `sample_prefix`.
    population : str
        Epydemix population name (e.g. "United_States__Massachusetts"). Only used to build the sample id
        prefix via `profile_spec["sample_prefix"]` (e.g. "MA" -> "MA00", "MA01", ...).
    location : str
        Hub location id written to the rows (e.g. "25").
    value_col : str
        Column of `population_trajectories` with the values to sample, the same one the quantiles use.
    trajectory_samples : FlusightTrajectorySamples
        Number of samples, selection method and seed.

    Returns
    -------
    pd.DataFrame
        One row per selected trajectory and horizon with the 8 hubverse columns, `output_type` "sample".
        `target_end_date` is a 'YYYY-MM-DD' string, matching the quantile rows.

    Examples
    --------
    For `flusight_hosp` and Massachusetts, each selected trajectory becomes one row per horizon (-1..3)
    sharing an `output_type_id` (`MA00`, `MA01`, ...). Values are rounded/clipped by target in
    `trajectories_to_sample_rows`:

    ```
    reference_date  horizon target_end_date location           target output_type output_type_id   value
        2026-04-25       -1      2026-04-18       25  wk inc flu hosp      sample           MA00  1729.0
        2026-04-25        0      2026-04-25       25  wk inc flu hosp      sample           MA00   385.0
        2026-04-25        1      2026-05-02       25  wk inc flu hosp      sample           MA00  1842.0
        2026-04-25        2      2026-05-09       25  wk inc flu hosp      sample           MA00   182.0
        2026-04-25        3      2026-05-16       25  wk inc flu hosp      sample           MA00  1238.0
        2026-04-25       -1      2026-04-18       25  wk inc flu hosp      sample           MA01   404.0
        ...
    ```

    For `metrocast` and `bphc_ed`, ids restart at "1", "2", ... per location and horizons are those of the profile.
    """
    target_dates = [pd.to_datetime(reference_date) + pd.Timedelta(weeks=h) for h in profile_spec["horizons"]]
    values = population_trajectories.pivot(index="sample_id", columns="date", values=value_col).reindex(
        columns=target_dates
    )
    values = values.to_numpy(dtype=float)
    selected = values[
        select_trajectory_indices(
            values, trajectory_samples.n_samples, trajectory_samples.method, trajectory_samples.seed
        )
    ]
    if len(selected) < trajectory_samples.n_samples:
        print(
            f"  WARNING: only {len(selected)} complete trajectories for {location} "
            f"(requested {trajectory_samples.n_samples})."
        )
    prefix = profile_spec["sample_prefix"](population) if profile_spec["sample_prefix"] else None
    rows = trajectories_to_sample_rows(
        selected, profile_spec["horizons"], reference_date, location, profile_spec["target"], prefix
    )
    # Match the quantile rows: the given reference_date and 'YYYY-MM-DD' target dates
    rows["reference_date"] = reference_date
    rows["target_end_date"] = [d.strftime("%Y-%m-%d") for d in rows["target_end_date"]]
    return rows


def _pmf_rows(pmf_df: pd.DataFrame, reference_date: str) -> pd.DataFrame:
    """Create PMF forecasts in FluSight submission format."""
    results = []

    for _, row in pmf_df.iterrows():
        location = row["location"]
        horizon = row["horizon"]
        target_date = row["target_date"]

        if isinstance(target_date, str):
            target_end_date = target_date
        else:
            target_end_date = target_date.strftime("%Y-%m-%d")

        for category in RATE_TREND_CATEGORIES:
            results.append(
                {
                    "reference_date": reference_date,
                    "horizon": horizon,
                    "target_end_date": target_end_date,
                    "location": location,
                    "target": "wk flu hosp rate change",
                    "output_type": "pmf",
                    "output_type_id": category,
                    "value": row[category],
                }
            )

    return pd.DataFrame(results)


def read_aggregated(path: str, config) -> pd.DataFrame:
    """Read aggregated trajectories (parquet, or csv as fallback) and keep the columns needed downstream."""
    try:
        aggregated = pd.read_parquet(path)
    except Exception as e1:
        try:
            aggregated = pd.read_csv(path)
        except Exception as e2:
            raise ValueError(
                f"Failed to read {path} as either parquet or csv:\nParquet error:\n{e1}\nCSV error:\n{e2}"
            ) from e2
    aggregated = aggregated[[config.date_column, config.location_column, config.target_column, "sample_id"]]
    return cols_to_dt(aggregated, [config.date_column])


def read_surveillance(config) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Read in-sample (fit) and optional out-of-sample (recent) surveillance as `date`, `location`, `target`.

    `config.location_column` must hold the hub location ids of the submission profile, e.g. FIPS codes
    (`location_code` in hosp files, `location` in ED files) or metrocast ids (`location`).
    """

    def _read(fname: str) -> pd.DataFrame:
        surv = pd.read_csv(f"{config.directory}/{fname}", dtype={config.location_column: str})
        surv = cols_to_dt(surv, [config.date_column])
        surv = surv.rename(
            columns={config.date_column: "date", config.location_column: "location", config.target_column: "target"}
        )
        return surv[["date", "location", "target"]]

    fit = _read(config.fit_fname)
    recent = _read(config.recent_fname) if config.recent_fname is not None else None
    return fit, recent
