#!/usr/bin/env python3
"""
Script to aggregate trajectories from multistrain experiments and create a hubverse submission file.
All options specified via yaml file.

Usage:
    python aggregate_multistrain_experiments.py --config aggregation.yml
"""

import argparse
import tempfile
from pathlib import Path

from epymodelingsuite.config_loader import load_aggregation_config_from_file
from epymodelingsuite.strain_aggregator import pull_trajectory_projections

def main():
    """Aggregate trajectories from multistrain experiments."""
    parser = argparse.ArgumentParser(
        description="Aggregate trajectories from multistrain experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to aggregation config yaml (required)",
    )

    args = parser.parse_args()

    # Load config
    try:
        config = load_aggregation_config_from_file(args.config)
    except Exception as e:
        raise ValueError(f"Error loading aggregation config: {e}")
    aggregation = config.aggregation

    # Work inside temp directory
    outputs = {}
    with tempfile.TemporaryDirectory() as tempdir:
        subtypes = pull_trajectory_projections(aggregation, tempdir)
        

    
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