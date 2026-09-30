import pandas as pd

from ..schema.output import (
    get_flusight_categorical_horizons,
    get_flusight_quantiles,
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

MIN_COUNT_CHANGE = 10
RATE_TREND_CATEGORIES = ["large_decrease", "decrease", "stable", "increase", "large_increase"]
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
    surveillance_location_col: str = "abbreviation",
) -> pd.DataFrame:
    """Compute rate trend category for each trajectory at each horizon."""
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
        pop_data = df[df["location"] == pop]
        population_size = POPULATION.get(abbrev, POPULATION.get("US"))

        surv_pop = surv[surv[surveillance_location_col] == abbrev]
        baseline_row = surv_pop[surv_pop[surveillance_date_col] == baseline_date]

        if len(baseline_row) == 0:
            print(f"Warning: No surveillance data for {abbrev} on {baseline_date}")
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
        categories_df.groupby(["population", "abbreviation", "horizon", "target_date", "category"])
        .size()
        .unstack(fill_value=0)
    )

    for cat in RATE_TREND_CATEGORIES:
        if cat not in counts.columns:
            counts[cat] = 0

    counts = counts[RATE_TREND_CATEGORIES]
    pmf = counts.div(counts.sum(axis=1), axis=0)
    return pmf.reset_index()


def create_flusight_submission(
    trajectories_df: pd.DataFrame,
    pmf_df: pd.DataFrame,
    reference_date: str,
    horizons: list = None,
    value_col: str = "target_total",
    output_path: str = None,
) -> pd.DataFrame:
    """Create complete FluSight submission file combining quantiles and PMF."""
    quantile_df = create_quantile_submission(
        df=trajectories_df, reference_date=reference_date, horizons=horizons, value_col=value_col
    )

    pmf_submission_df = create_pmf_submission(pmf_df=pmf_df, reference_date=reference_date)

    submission = pd.concat([quantile_df, pmf_submission_df], ignore_index=True)

    submission = submission.sort_values(["location", "target", "horizon", "output_type", "output_type_id"]).reset_index(
        drop=True
    )

    if output_path:
        if output_path.endswith(".parquet"):
            submission.to_parquet(output_path, index=False)
        elif output_path.endswith(".gz"):
            submission.to_csv(output_path, index=False, compression="gzip")
        else:
            submission.to_csv(output_path, index=False)
        print(f"Saved submission to: {output_path}")

    return submission


def create_quantile_submission(
    df: pd.DataFrame, reference_date: str, horizons: list = None, value_col: str = "target_total"
) -> pd.DataFrame:
    """Create quantile forecasts in FluSight submission format."""
    if horizons is None:
        horizons = [-1, 0, 1, 2, 3]
    flusight_quantiles = get_flusight_quantiles()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    reference_date_dt = pd.to_datetime(reference_date)

    results = []

    for pop in df["location"].unique():
        abbrev = convert_location_name_format(value=pop, output_format="abbreviation", location_type="iso")
        location = convert_location_name_format(
            value=abbrev, output_format="FIPS", input_format="abbreviation", location_type="iso"
        )
        pop_data = df[df["location"] == pop]

        for horizon in horizons:
            target_end_date = reference_date_dt + pd.Timedelta(weeks=horizon)
            target_data = pop_data[pop_data["date"] == target_end_date][value_col]

            if len(target_data) == 0:
                continue

            for q in flusight_quantiles:
                value = target_data.quantile(q)
                results.append(
                    {
                        "reference_date": reference_date,
                        "horizon": horizon,
                        "target_end_date": target_end_date.strftime("%Y-%m-%d"),
                        "location": location,
                        "target": "wk inc flu hosp",
                        "output_type": "quantile",
                        "output_type_id": str(q),
                        "value": round(value, 0),
                    }
                )

    return pd.DataFrame(results)


def create_ed_submission(
    trajectories_df: pd.DataFrame,
    reference_date: str,
    horizons: list = None,
    value_col: str = "target_total",
    output_path: str = None,
) -> pd.DataFrame:
    """Create complete FluSight submission file combining quantiles and PMF."""
    quantile_df = create_ed_quantile_submission(
        df=trajectories_df, reference_date=reference_date, horizons=horizons, value_col=value_col
    )

    submission = quantile_df.sort_values(
        ["location", "target", "horizon", "output_type", "output_type_id"]
    ).reset_index(drop=True)

    if output_path:
        if output_path.endswith(".parquet"):
            submission.to_parquet(output_path, index=False)
        elif output_path.endswith(".gz"):
            submission.to_csv(output_path, index=False, compression="gzip")
        else:
            submission.to_csv(output_path, index=False)
        print(f"Saved submission to: {output_path}")

    return submission


def create_ed_quantile_submission(
    df: pd.DataFrame, reference_date: str, horizons: list = None, value_col: str = "target_total"
) -> pd.DataFrame:
    """Create quantile forecasts in FluSight submission format."""
    if horizons is None:
        horizons = [-1, 0, 1, 2, 3]
    flusight_quantiles = get_flusight_quantiles()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    reference_date_dt = pd.to_datetime(reference_date)

    results = []

    for pop in df["location"].unique():
        abbrev = convert_location_name_format(value=pop, output_format="abbreviation", location_type="iso")
        location = convert_location_name_format(
            value=abbrev, output_format="FIPS", input_format="abbreviation", location_type="iso"
        )
        pop_data = df[df["location"] == pop]

        for horizon in horizons:
            target_end_date = reference_date_dt + pd.Timedelta(weeks=horizon)
            target_data = pop_data[pop_data["date"] == target_end_date][value_col]

            if len(target_data) == 0:
                continue

            for q in flusight_quantiles:
                value = target_data.quantile(q)
                results.append(
                    {
                        "reference_date": reference_date,
                        "horizon": horizon,
                        "target_end_date": target_end_date.strftime("%Y-%m-%d"),
                        "location": location,
                        "target": "wk inc flu prop ed visits",
                        "output_type": "quantile",
                        "output_type_id": str(q),
                        "value": round(value, 3),
                    }
                )

    return pd.DataFrame(results)


def create_metro_submission(
    trajectories_df: pd.DataFrame,
    reference_date: str,
    quantiles: list[float],
    horizons: list = None,
    value_col: str = "target_total",
    output_path: str = None,
) -> pd.DataFrame:
    """Create complete FluSight submission file combining quantiles and PMF."""
    quantile_df = create_metro_quantile_submission(
        df=trajectories_df, reference_date=reference_date, quantiles=quantiles, horizons=horizons, value_col=value_col
    )

    submission = quantile_df.sort_values(
        ["location", "target", "horizon", "output_type", "output_type_id"]
    ).reset_index(drop=True)

    if output_path:
        if output_path.endswith(".parquet"):
            submission.to_parquet(output_path, index=False)
        elif output_path.endswith(".gz"):
            submission.to_csv(output_path, index=False, compression="gzip")
        else:
            submission.to_csv(output_path, index=False)
        print(f"Saved submission to: {output_path}")

    return submission


def create_metro_quantile_submission(
    df: pd.DataFrame,
    reference_date: str,
    quantiles: list[float],
    horizons: list = None,
    value_col: str = "target_total",
) -> pd.DataFrame:
    """Create quantile forecasts in Metrocast submission format."""
    if horizons is None:
        horizons = [0, 1, 2, 3]

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    reference_date_dt = pd.to_datetime(reference_date)

    results = []

    for pop in df["location"].unique():
        location = pop.replace("metrocast_location_", "")
        pop_data = df[df["location"] == pop]

        for horizon in horizons:
            target_end_date = reference_date_dt + pd.Timedelta(weeks=horizon)
            target_data = pop_data[pop_data["date"] == target_end_date][value_col]

            if len(target_data) == 0:
                continue

            for q in quantiles:
                value = target_data.quantile(q)
                results.append(
                    {
                        "reference_date": reference_date,
                        "horizon": horizon,
                        "target_end_date": target_end_date.strftime("%Y-%m-%d"),
                        "location": location,
                        "target": "Flu ED visits pct",
                        "output_type": "quantile",
                        "output_type_id": str(q),
                        "value": round(value, 3),
                    }
                )

    return pd.DataFrame(results)


def create_pmf_submission(pmf_df: pd.DataFrame, reference_date: str) -> pd.DataFrame:
    """Create PMF forecasts in FluSight submission format."""
    results = []

    for _, row in pmf_df.iterrows():
        abbrev = row["abbreviation"]
        location = convert_location_name_format(
            value=abbrev, output_format="FIPS", input_format="abbreviation", location_type="iso"
        )
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
