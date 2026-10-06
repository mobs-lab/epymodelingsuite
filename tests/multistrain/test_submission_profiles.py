import numpy as np
import pandas as pd
import pytest

from epymodelingsuite.multistrain.formatter import SUBMISSION_PROFILES, create_submission
from epymodelingsuite.schema.output import FlusightTrajectorySamples
from epymodelingsuite.schema.submission import SubmissionProfile, validate_submission

REFERENCE_DATE = "2026-04-25"
N_SAMPLES = 50

POPULATIONS = {
    "flusight_hosp": (["United_States", "United_States__Massachusetts"], ["US", "25"]),
    "flusight_ed": (["United_States", "United_States__Massachusetts"], ["US", "25"]),
    "metrocast": (["metrocast_location_boston", "metrocast_location_denver"], ["boston", "denver"]),
    "bphc_ed": (["metrocast_location_boston-all"], ["boston-all"]),
}
EXPECTED = {
    "flusight_hosp": ("wk inc flu hosp", [-1, 0, 1, 2, 3], 23),
    "flusight_ed": ("wk inc flu prop ed visits", [-1, 0, 1, 2, 3], 23),
    "metrocast": ("Flu ED visits pct", [0, 1, 2, 3], 9),
    "bphc_ed": ("wk inc ed signal", [-1, 0, 1, 2, 3], 23),
}


def _trajectories(populations: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range(end=pd.Timestamp(REFERENCE_DATE) + pd.Timedelta(weeks=4), periods=8, freq="7D")
    rows = [(pop, d, s) for pop in populations for d in dates for s in range(N_SAMPLES)]
    df = pd.DataFrame(rows, columns=["location", "date", "sample_id"])
    df["target_total"] = rng.gamma(2.0, 500.0, len(df))
    return df


def _surveillance() -> pd.DataFrame:
    baseline_date = pd.Timestamp(REFERENCE_DATE) - pd.Timedelta(weeks=1)
    return pd.DataFrame({"date": [baseline_date] * 2, "location": ["US", "25"], "target": [1000.0, 50.0]})


def test_schema_profiles_match_formatter():
    assert set(SubmissionProfile.__args__) == set(SUBMISSION_PROFILES)


@pytest.mark.parametrize("profile", list(SUBMISSION_PROFILES))
def test_submission_profile(profile):
    populations, locations = POPULATIONS[profile]
    target, horizons, n_quantiles = EXPECTED[profile]
    surveillance = _surveillance() if SUBMISSION_PROFILES[profile]["pmf"] else None

    sub = create_submission(_trajectories(populations), REFERENCE_DATE, profile, surveillance_df=surveillance)

    assert list(sub.columns) == [
        "reference_date",
        "horizon",
        "target_end_date",
        "location",
        "target",
        "output_type",
        "output_type_id",
        "value",
    ]
    quantiles = sub[sub["output_type"] == "quantile"]
    assert set(quantiles["target"]) == {target}
    assert sorted(sub["location"].unique()) == sorted(locations)
    assert sorted(quantiles["horizon"].unique()) == horizons
    assert len(quantiles) == len(locations) * len(horizons) * n_quantiles

    pmf = sub[sub["output_type"] == "pmf"]
    if SUBMISSION_PROFILES[profile]["pmf"]:
        assert set(pmf["target"]) == {"wk flu hosp rate change"}
        assert len(pmf) == len(locations) * 4 * 5
        assert np.allclose(pmf.groupby(["location", "horizon"])["value"].sum(), 1)
    else:
        assert pmf.empty


SAMPLE_IDS = {
    "flusight_hosp": {"US": "US00", "25": "MA00"},
    "flusight_ed": {"US": "US00", "25": "MA00"},
    "metrocast": {"boston": "1", "denver": "1"},
    "bphc_ed": {"boston-all": "1"},
}


@pytest.mark.parametrize("profile", list(SUBMISSION_PROFILES))
def test_submission_samples(profile):
    populations, locations = POPULATIONS[profile]
    target, horizons, _ = EXPECTED[profile]
    surveillance = _surveillance() if SUBMISSION_PROFILES[profile]["pmf"] else None
    samples = FlusightTrajectorySamples(n_samples=10, seed=1)

    sub = create_submission(
        _trajectories(populations), REFERENCE_DATE, profile, surveillance_df=surveillance, samples=samples
    )

    rows = sub[sub["output_type"] == "sample"]
    assert set(rows["target"]) == {target}
    assert len(rows) == len(locations) * len(horizons) * 10
    for location, first_id in SAMPLE_IDS[profile].items():
        loc_rows = rows[rows["location"] == location]
        assert loc_rows["output_type_id"].nunique() == 10
        assert first_id in set(loc_rows["output_type_id"])
        # every sample covers every horizon
        assert all(sorted(h) == horizons for h in loc_rows.groupby("output_type_id")["horizon"].apply(list))
    assert (rows["target_end_date"].map(type) == str).all()


def test_samples_skip_incomplete_trajectories():
    df = _trajectories(["United_States"])
    horizon_0 = pd.Timestamp(REFERENCE_DATE)
    df = df[~((df["sample_id"] < 45) & (df["date"] == horizon_0))]

    sub = create_submission(df, REFERENCE_DATE, "flusight_ed", samples=FlusightTrajectorySamples(n_samples=10, seed=1))

    assert sub[sub["output_type"] == "sample"]["output_type_id"].nunique() == 5


def test_pmf_profile_requires_surveillance():
    with pytest.raises(ValueError, match="surveillance"):
        create_submission(_trajectories(["United_States"]), REFERENCE_DATE, "flusight_hosp")
    with pytest.raises(ValueError, match="surveillance"):
        validate_submission(
            {
                "submission": {
                    "submission_week": 202616,
                    "profile": "flusight_hosp",
                    "model_name": "MOBS-EpyStrain_Flu",
                    "aggregated": {"target_column": "target_total"},
                }
            }
        )
