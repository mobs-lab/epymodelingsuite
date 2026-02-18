#!/usr/bin/env python3
"""
Script to aggregate trajectories from multistrain experiments and create a hubverse submission file.
All options specified via yaml file.

Usage:
    python aggregate_multistrain_experiments.py --config aggregation.yml --output ./multistrain
"""

import argparse
import tempfile
from pathlib import Path

from epymodelingsuite.config_loader import load_aggregation_config_from_file
from epymodelingsuite.multistrain.aggregator import (
    dispatch_strain_aggregator,
    dispatch_strain_sampler,
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

    print(f"  Obtained trajectories for strains:")
    for strain, traj_df in strains.items():
        print(f"  {strain}: shape {traj_df.shape}")

    # Write raw trajectories to file
    if aggregation.outputs.raw_trajectories:
        for strain, traj_df in strains.items():
            fname = f"{args.output}/trajectories_{strain}.csv"
            traj_df.to_csv(fname, index=False)
            print(f"  Saved trajectory file at {fname}")

    print("\nCreating strain sim_id mapping...")

    # Create an n-sample mapping of trajectories from each strain
    mapping_df = dispatch_strain_sampler(strains, aggregation)

    print(f"  Created mapping:\n{mapping_df.head()}")

    print("\nMerging strain trajectories...")

    # Merge the strain trajectories based on the mapping
    merged_df = merge_strain_trajectories(strains, mapping_df, aggregation)

    print(f"  Merged trajectories:\n{merged_df.head()}")

    print("\nCreating aggregated trajectories...")

    # Aggregate the merged trajectories
    aggregated_df = dispatch_strain_aggregator(merged_df, aggregation)

    print(f"  Aggregated trajectories:\n{aggregated_df.head()}")

    # Write aggregated trajectories to file
    if aggregation.outputs.aggregated_trajectories:
        fname = f"{args.output}/trajectories_aggregated.csv"
        aggregated_df.to_csv(fname, index=False)
        print(f"  Saved aggregated trajectories at {fname}")


if __name__ == "__main__":
    main()
