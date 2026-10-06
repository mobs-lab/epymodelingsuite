# /// script
# requires-python = ">=3.11"
# dependencies = ["epymodelingsuite", "pyarrow"]
#
# [tool.uv.sources]
# epymodelingsuite = { path = "../../", editable = true }
# ///
"""
Script to create a hubverse submission file from aggregated trajectories.
Takes a file path to a configuration, a file path to aggregated trajectories, and optionally, a directory path for outputs.
All other options specified via yaml file.

Usage:
    uv run submission.py --config submission.yml \
    --aggregated trajectories_aggregated.parquet --output ./multistrain
"""

import argparse
from pathlib import Path

from epiweeks import Week

from epymodelingsuite.config_loader import load_submission_config_from_file
from epymodelingsuite.multistrain.formatter import create_submission, read_aggregated, read_surveillance


def main():
    """Submission workflow."""
    parser = argparse.ArgumentParser(
        description="Create a hub submission file from aggregated trajectories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True, help="Path to submission config yaml")
    parser.add_argument("--aggregated", type=str, required=True, help="Path to aggregated trajectories")
    parser.add_argument(
        "--output",
        type=str,
        default="./multistrain_outputs",
        help="Directory for all outputs. Default: './multistrain_outputs'",
    )
    args = parser.parse_args()

    try:
        config = load_submission_config_from_file(args.config).submission
    except Exception as e:
        raise ValueError(f"Error loading submission config: {e}")

    reference_date = Week.fromstring(str(config.submission_week)).enddate()
    subs_path = Path(args.output)
    subs_path.mkdir(parents=True, exist_ok=True)

    surv_fit = None
    if config.surveillance is not None:
        print("\nLoading surveillance ...")
        surv_fit, _ = read_surveillance(config.surveillance)

    print(f"\nLoading aggregated trajectories from {args.aggregated} ...")
    aggregated = read_aggregated(args.aggregated, config.aggregated)
    print(aggregated.tail())

    print(f"\nGenerating {config.profile} submission file ...")
    submission = create_submission(
        trajectories_df=aggregated,
        reference_date=reference_date,
        profile=config.profile,
        value_col=config.aggregated.target_column,
        surveillance_df=surv_fit,
        samples=config.samples,
    )
    print(submission.groupby(["target", "output_type"]).size())
    sub_file = subs_path / f"{reference_date}-{config.model_name}.csv"
    submission.to_csv(sub_file, index=False)
    print(f"Saved submission to {sub_file}\n")


if __name__ == "__main__":
    main()
