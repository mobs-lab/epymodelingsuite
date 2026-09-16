#!/usr/bin/env python3
"""
Script to download HHS flu hospitalization data from CDC.

Example usage:
    # Download with defaults (starts from 2024-01-01, saves to flu_hosp_25.csv)
    python download_hhs_data.py

    # Download from a specific date
    python download_hhs_data.py --start-date 2024-10-01

    # Specify custom output file
    python download_hhs_data.py --output my_hhs_data.csv

    # Download preliminary data
    python download_hhs_data.py --data-type preliminary

    # All options specified
    python download_hhs_data.py --start-date 2024-01-01 --output flu_hosp_25.csv --data-type preliminary
"""

import argparse
from pathlib import Path

from epymodelingsuite.utils.data import fetch_hhs_hospitalizations


def main():
    """Download HHS hospitalization data."""
    parser = argparse.ArgumentParser(
        description="Download HHS flu hospitalization data from CDC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--start-date",
        type=str,
        default="2024-01-01",
        help="Start date for data download (format: YYYY-MM-DD). Default: 2024-01-01",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="flu_hosp_25.csv",
        help="Output CSV file path. Default: flu_hosp_25.csv",
    )

    parser.add_argument(
        "--data-type",
        type=str,
        choices=["default", "preliminary"],
        default="default",
        help="Type of data to fetch. 'default' for non-preliminary, 'preliminary' for preliminary data. Default: default",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Downloading HHS hospitalization data...")
    print(f"  Start date: {args.start_date}")
    print(f"  Data type: {args.data_type}")
    print(f"  Output file: {args.output}")

    # Fetch data
    df = fetch_hhs_hospitalizations(
        query_start_date=args.start_date,
        save_path=args.output,
        data_type=args.data_type,
    )

    print("\nDownload complete!")
    print(f"  Total records: {len(df)}")
    print(f"  Date range: {df['target_end_date'].min()} to {df['target_end_date'].max()}")
    print(f"  Locations: {df['location_iso'].nunique()}")
    print(f"  Saved to: {args.output}")

    # Show preview
    print("\nFirst few rows:")
    print(df.head())


if __name__ == "__main__":
    main()
