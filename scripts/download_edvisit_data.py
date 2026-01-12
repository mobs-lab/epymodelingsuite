#!/usr/bin/env python3
"""
Script to download NSSP ED visits data from CDC.

Example usage:
    # Download with defaults (starts from 2024-01-01, saves to flu_prop-ed_25.csv)
    python download_edvisit_data.py

    # Download from a specific date
    python download_edvisit_data.py --start-date 2024-10-01

    # Specify custom output file
    python download_edvisit_data.py --output my_edvisit_data.csv

    # All options specified
    python download_edvisit_data.py --start-date 2024-01-01 --output flu_prop-ed_25.csv
"""

import argparse
from pathlib import Path

from epymodelingsuite.utils.data import fetch_nssp_edvisits


def main():
    """Download NSSP ED visits data."""
    parser = argparse.ArgumentParser(
        description="Download NSSP ED visits data from CDC",
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
        default="flu_prop-ed_25.csv",
        help="Output CSV file path. Default: flu_prop-ed_25.csv",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Downloading NSSP ED visits data...")
    print(f"  Start date: {args.start_date}")
    print(f"  Output file: {args.output}")

    # Fetch data
    df = fetch_nssp_edvisits(
        query_start_date=args.start_date,
        save_path=args.output,
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
