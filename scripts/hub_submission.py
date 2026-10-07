#!/usr/bin/env python3
"""
Filter and combine FluSight (hubverse) tables, e.g. hospitalization and ED.

Inputs and output may be CSV or Parquet (chosen by file suffix).

Example usage:
    # Drop locations (abbreviation, FIPS, name or ISO) from one table
    python hub_submission.py filter ed.csv -o ed_filtered.csv --exclude-location MA 06 "Puerto Rico"

    # Concatenate tables into one submission; fails if two inputs share a location/target/output_type
    python hub_submission.py combine hosp.csv ed_filtered.csv -o 2026-10-10-MOBS-GLEAM_FLUH.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from epymodelingsuite.output.hub_files import cast_hub_dtypes, combine_submissions, read_hub_table
from epymodelingsuite.utils.location import convert_location_name_format


def write_table(df: pd.DataFrame, path: Path) -> None:
    if path.suffix.lower() in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    print(f"Wrote {len(df)} rows to {path}")  # noqa: T201


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter and combine FluSight tables")
    subparsers = parser.add_subparsers(dest="command", required=True)

    filter_parser = subparsers.add_parser("filter", help="Drop locations from one table")
    filter_parser.add_argument("input", type=Path, help="Input CSV or Parquet file")
    filter_parser.add_argument("-o", "--output", type=Path, required=True, help="Output file (.csv or .parquet)")
    filter_parser.add_argument(
        "--exclude-location", nargs="+", required=True, metavar="LOC", help="Locations to drop, e.g. MA 06 US-NY Texas"
    )

    combine_parser = subparsers.add_parser("combine", help="Concatenate tables into one submission")
    combine_parser.add_argument("inputs", nargs="+", type=Path, help="Input CSV or Parquet files")
    combine_parser.add_argument("-o", "--output", type=Path, required=True, help="Output file (.csv or .parquet)")

    args = parser.parse_args()

    if args.command == "combine":
        write_table(combine_submissions([read_hub_table(p) for p in args.inputs]), args.output)
        return

    try:
        exclude = {convert_location_name_format(loc, "FIPS", location_type="iso") for loc in args.exclude_location}
    except AssertionError as e:
        filter_parser.error(str(e))
    table = cast_hub_dtypes(read_hub_table(args.input))
    missing = exclude - set(table["location"])
    if missing:
        filter_parser.error(f"--exclude-location not found in {args.input}: {sorted(missing)}")
    write_table(table[~table["location"].isin(exclude)].reset_index(drop=True), args.output)


if __name__ == "__main__":
    main()
