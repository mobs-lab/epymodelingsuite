# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "pyarrow", "epiweeks"]
# ///
"""Convert a trajectories_aggregated.csv to the epystrain parquet format.

Filters to 4 weekly horizons (0-3) from the given reference_date and
adds reference_date, horizon, and epiweek columns.

Usage:
    uv run trajectories_to_parquet.py --csv PATH --reference-date YYYY-MM-DD [--output PATH]
"""

import argparse
from pathlib import Path

import pandas as pd
from epiweeks import Week


def convert(csv_path: Path, reference_date: pd.Timestamp, n_horizons: int = 4) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])

    horizon_dates = [reference_date + pd.Timedelta(weeks=h) for h in range(n_horizons)]
    missing = [d for d in horizon_dates if d not in set(df["date"])]
    if missing:
        raise ValueError(
            f"CSV is missing dates for horizons: {[d.date().isoformat() for d in missing]}"
        )

    out = df[df["date"].isin(horizon_dates)].copy()
    out["reference_date"] = reference_date
    out["horizon"] = ((out["date"] - reference_date).dt.days // 7).astype(int)
    out["epiweek"] = out["date"].apply(lambda d: Week.fromdate(d.date()).cdcformat())
    return out.reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, required=True, help="Path to trajectories_aggregated.csv")
    p.add_argument(
        "--reference-date",
        type=lambda s: pd.Timestamp(s),
        required=True,
        help="Reference date (YYYY-MM-DD), horizon 0",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output parquet path (default: epystrain_trajectories_<ref>.parquet in CWD)",
    )
    p.add_argument("--n-horizons", type=int, default=4, help="Number of weekly horizons (default 4)")
    args = p.parse_args()

    out = convert(args.csv, args.reference_date, args.n_horizons)
    output = args.output or Path.cwd() / f"epystrain_trajectories_{args.reference_date.date()}.parquet"
    out.to_parquet(output, index=False)
    print(f"Wrote {len(out):,} rows to {output}")


if __name__ == "__main__":
    main()
