# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "pyarrow", "epiweeks"]
# ///
"""
Script to create multistrain comparison plots from aggregated trajectories.
Takes a file path to a configuration, a file path to aggregated trajectories, and optionally, a directory path for outputs.
All other options specified via yaml file.

Usage:
    python plot.py --config plot.yml \
    --aggregated trajectories_aggregated.parquet --output ./multistrain
"""

import argparse
from pathlib import Path

import pandas as pd
from epiweeks import Week

from epymodelingsuite.config_loader import load_plot_config_from_file
from epymodelingsuite.multistrain.formatter import (
    cols_to_dt,
    compute_quantiles,
    read_aggregated,
    read_surveillance,
)
from epymodelingsuite.multistrain.plot import (
    make_fit_start_labels,
    make_plotting_windows,
    plot_hosp_multistrain_quantiles,
    pull_single_strain_trajectories,
)
from epymodelingsuite.utils.location import convert_location_name_format


def main():
    """Plot workflow."""
    parser = argparse.ArgumentParser(
        description="Create multistrain comparison plots from aggregated trajectories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True, help="Path to plot config yaml")
    parser.add_argument("--aggregated", type=str, required=True, help="Path to aggregated trajectories")
    parser.add_argument(
        "--output",
        type=str,
        default="./multistrain_outputs",
        help="Directory for all outputs. Default: './multistrain_outputs'",
    )
    args = parser.parse_args()

    try:
        config = load_plot_config_from_file(args.config).plot
    except Exception as e:
        raise ValueError(f"Error loading plot config: {e}")

    reference_dt = pd.Timestamp(Week.fromstring(str(config.submission_week)).enddate())
    plots_path = Path(args.output)
    plots_path.mkdir(parents=True, exist_ok=True)

    print("\nLoading surveillance ...")
    surv_fit, surv_recent = read_surveillance(config.surveillance)

    print(f"\nLoading aggregated trajectories from {args.aggregated} ...")
    aggregated = read_aggregated(args.aggregated, config.aggregated)
    print(aggregated.tail())

    if config.single_strain is None:
        single_strain = None
    else:
        print(f"\nDownloading single-strain trajectories from {config.single_strain.bucket} ...")
        try:
            single_strain = pull_single_strain_trajectories(config)
        except Exception as e:
            raise ValueError(f"Failed to download single-strain:\n{e}")
        single_strain = single_strain[
            [config.single_strain.target_column, config.single_strain.date_column, config.single_strain.location_column]
        ]
        single_strain = cols_to_dt(single_strain, [config.single_strain.date_column])
        print(single_strain.tail())

    print("\nGenerating plots ...")
    agg_quantiles = compute_quantiles(
        aggregated,
        value_col=config.aggregated.target_column,
        date_col=config.aggregated.date_column,
        location_col=config.aggregated.location_column,
    )
    # Add state abbreviations
    agg_quantiles["abbreviation"] = agg_quantiles[config.aggregated.location_column].apply(
        lambda l: convert_location_name_format(
            value=l, output_format="abbreviation", input_format="epydemix_population", location_type="iso"
        )
    )
    # Do the same for single strain
    if single_strain is not None:
        single_quantiles = compute_quantiles(
            single_strain,
            value_col=config.single_strain.target_column,
            date_col=config.single_strain.date_column,
            location_col=config.single_strain.location_column,
        )
        single_quantiles["abbreviation"] = single_quantiles[config.single_strain.location_column].apply(
            lambda l: convert_location_name_format(
                value=l, output_format="abbreviation", input_format="epydemix_population", location_type="iso"
            )
        )
    else:
        single_quantiles = None
    # Get sorted list of populations (US first, then states alphabetically)
    populations = sorted(agg_quantiles[config.aggregated.location_column].unique())
    if "United_States" in populations:
        populations = ["United_States"] + [p for p in populations if p != "United_States"]
    print(f"  Plotting {len(populations)} locations")

    plotting_windows = make_plotting_windows(config)
    fit_start_labels = make_fit_start_labels(config)

    for plot_focus, plotting_window in plotting_windows.items():
        plot_start_date = pd.Timestamp(plotting_window[0])
        plot_end_date = pd.Timestamp(plotting_window[1])
        include_single = "" if config.single_strain is None else " vs Single Strain"
        plot_title = f"{config.submission_week} Multistrain {config.aggregated.target_column}{include_single}"
        plot_fname = f"{plot_focus.lower()}-{config.submission_week}-{config.aggregated.target_column}-multistrain{include_single.lower().replace(' ', '_')}.pdf"
        plot_fpath = f"{plots_path}/{plot_fname}"

        agg_quantiles_filter = (
            agg_quantiles[
                (agg_quantiles[config.aggregated.date_column] >= plot_start_date)
                & (agg_quantiles[config.aggregated.date_column] < plot_end_date)
            ]
            .sort_values(config.aggregated.date_column)
            .rename(columns={config.aggregated.date_column: "date"})
        )
        single_quantiles_filter = None
        if single_quantiles is not None:
            single_quantiles_filter = (
                single_quantiles[
                    (single_quantiles[config.single_strain.date_column] >= plot_start_date)
                    & (single_quantiles[config.single_strain.date_column] < plot_end_date)
                ]
                .sort_values(config.single_strain.date_column)
                .rename(columns={config.single_strain.date_column: "date"})
            )
        surv_fit_filter = surv_fit[
            (surv_fit["date"] >= plot_start_date) & (surv_fit["date"] < plot_end_date)
        ].sort_values("date")
        surv_recent_filter = None
        if surv_recent is not None:
            surv_fit_filter = surv_fit_filter[surv_fit_filter["date"] < reference_dt]
            surv_recent_filter = surv_recent[
                (surv_recent["date"] >= reference_dt) & (surv_recent["date"] < plot_end_date)
            ].sort_values("date")

        plot_hosp_multistrain_quantiles(
            populations=populations,
            reference_date=reference_dt,
            multistrain_quantiles=agg_quantiles_filter,
            single_strain_quantiles=single_quantiles_filter,
            surveillance_fit=surv_fit_filter,
            surveillance_recent=surv_recent_filter,
            fit_start_labels={
                s: l for s, l in fit_start_labels.items() if plot_start_date <= pd.Timestamp(s) <= plot_end_date
            },
            plot_title=plot_title,
            save_path=plot_fpath,
        )


if __name__ == "__main__":
    main()
