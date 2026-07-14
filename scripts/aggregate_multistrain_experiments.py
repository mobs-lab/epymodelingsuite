# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "pyarrow", "epiweeks"]
# ///
"""
Script to aggregate trajectories from multistrain experiments and add baseline noise.
Takes a file path to a configuration and, optionally, a directory path for outputs.
All other options specified via yaml file.

Usage:
    python aggregate_multistrain_experiments.py --config aggregation.yml --output ./multistrain
"""

import argparse
import tempfile
from pathlib import Path
from epiweeks import Week
import pandas as pd

from epymodelingsuite.config_loader import load_aggregation_config_from_file
from epymodelingsuite.multistrain.aggregator import (
    dispatch_strain_aggregator,
    dispatch_strain_sampler,
    dispatch_baseline,
    merge_strain_trajectories,
    pull_trajectory_projections,
)


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

    parser.add_argument(
        "--output",
        type=str,
        default="./multistrain_outputs",
        help="Directory for all outputs. Default: './multistrain_outputs'",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load config
    try:
        config = load_aggregation_config_from_file(args.config)
    except Exception as e:
        raise ValueError(f"Error loading aggregation config: {e}")
    aggregation = config.aggregation

    print(f"Downloading strain trajectories from {aggregation.bucket} ...")

    # Work inside temp directory for pulling from bucket
    # Make dict of strain: trajectory df
    strains = {}
    with tempfile.TemporaryDirectory() as tempdir:
        strains.update(pull_trajectory_projections(aggregation, tempdir))

    print("  Obtained trajectories for strains:")
    for strain, traj_df in strains.items():
        print(f"  {strain}: shape {traj_df.shape}")

    print("\nCreating strain sim_id mapping...")

    # Create an n-sample mapping of trajectories from each strain
    mapping_df = dispatch_strain_sampler(strains, aggregation)

    print(f"  Created mapping with {len(mapping_df)} rows:\n{mapping_df.tail()}")

    print("\nMerging strain trajectories...")

    # Merge the strain trajectories based on the mapping
    merged_df = merge_strain_trajectories(strains, mapping_df, aggregation)

    print(f"  Merged trajectories with {len(merged_df)} rows:\n{merged_df.tail()}")

    print("\nCreating aggregated trajectories...")

    # Aggregate the merged trajectories
    aggregated_df = dispatch_strain_aggregator(merged_df, aggregation)

    print(f"  Aggregated trajectories with {len(aggregated_df)} rows:\n{aggregated_df.tail()}")

    # Add baseline noise to the aggregated trajectories
    if aggregation.baseline is not None:
        print("\nAdding baseline to aggregated trajectories...")
        
        aggregated_df = dispatch_baseline(aggregated_df, aggregation)

        print(f"  Added {aggregation.baseline.method} baseline noise with dispersion/variance modifiers {aggregation.baseline.dispersion_values}:\n{aggregated_df.tail()}")

    # Add formatting columns
    ref_date = Week.fromstring(str(aggregation.submission_week)).enddate()
    aggregated_df['reference_date'] = ref_date
    aggregated_df['horizon'] = ((aggregated_df['date'] - pd.Timestamp(ref_date)).dt.days // 7).astype(int)
    aggregated_df['epiweek'] = [Week.fromdate(dat).cdcformat() for dat in aggregated_df.date]
    
    # Write aggregated trajectories to file
    if aggregation.outputs.base_fname:
        fname = f"{args.output}/trajectories_aggregated_{aggregation.outputs.base_fname}.parquet"
    else:
        fname = f"{args.output}/trajectories_aggregated_{ref_date.strftime("%Y-%m-%d")}.parquet"
    aggregated_df.to_parquet(fname, index=False)
    print(f"  Saved aggregated trajectories at {fname}")


if __name__ == "__main__":
    main()
