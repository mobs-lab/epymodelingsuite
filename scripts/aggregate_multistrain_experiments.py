#!/usr/bin/env python3
"""
Script to aggregate trajectories from multistrain experiments and create a hubverse submission file.
All options specified via yaml file.

Usage:
    python aggregate_multistrain_experiments.py --config aggregation.yml
"""

import argparse
import tempfile

from epymodelingsuite.config_loader import load_aggregation_config_from_file
from epymodelingsuite.multistrain.aggregator import (
    dispatch_strain_aggregator,
    dispatch_strain_sampler,
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

    args = parser.parse_args()

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

    print(f"  Obtained trajectories for strains {strains.keys()}.")

    print("\nCreating strain sim_id mapping...")

    # Create an n-sample mapping of trajectories from each strain
    mapping_df = dispatch_strain_sampler(strains, aggregation)

    print(f"  Created mapping:\n{mapping_df.head()}")

    print("\nCreating aggregated trajectories...")

    # Aggregate the trajectories based on the mapping
    aggregated_df = dispatch_strain_aggregator(strains, mapping_df, aggregation)

    print(f"  Aggregated trajectories:\n{aggregated_df.head()}")


if __name__ == "__main__":
    main()
