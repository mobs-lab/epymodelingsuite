# /// script
# requires-python = ">=3.11"
# dependencies = ["epymodelingsuite", "pyarrow"]
#
# [tool.uv.sources]
# epymodelingsuite = { path = "../../", editable = true }
# ///
"""
Script to create multistrain comparison plots from aggregated trajectories.
Takes a file path to a configuration, a base path for aggregated trajectories, and optionally, a directory path for outputs.
Each labelled entry under `aggregated` is read from `--aggregated` joined with its `trajectory_file`, and all are plotted together.
All other options specified via yaml file.

Usage:
    uv run plot.py --config plot.yml \
    --aggregated outputs/ --output ./multistrain
"""

import argparse
from pathlib import Path

import pandas as pd
from epiweeks import Week

from epymodelingsuite.config_loader import load_plot_config_from_file
from epymodelingsuite.multistrain.formatter import (
    SUBMISSION_PROFILES,
    cols_to_dt,
    compute_quantiles,
    read_aggregated,
    read_surveillance,
)
from epymodelingsuite.multistrain.plot import (
    make_fit_start_labels,
    make_plotting_windows,
    plot_multistrain_quantiles,
    pull_single_strain_trajectories,
)


def _quantiles(df: pd.DataFrame, value_col: str, date_col: str, location_col: str, to_location) -> pd.DataFrame:
    """Quantiles per location and date, with `date` and hub-id `location` columns."""
    q = compute_quantiles(df, value_col=value_col, date_col=date_col, location_col=location_col)
    q = q.rename(columns={date_col: "date"})
    q["location"] = q.pop(location_col).map(to_location)
    return q


def _window(df: pd.DataFrame | None, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame | None:
    if df is None:
        return None
    return df[(df["date"] >= start) & (df["date"] < end)].sort_values("date")


def main():
    """Plot workflow."""
    parser = argparse.ArgumentParser(
        description="Create multistrain comparison plots from aggregated trajectories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True, help="Path to plot config yaml")
    parser.add_argument("--aggregated", type=str, required=True, help="Base path for aggregated trajectories")
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

    surv_fit, surv_recent = None, None
    if config.surveillance is not None:
        print("\nLoading surveillance ...")
        surv_fit, surv_recent = read_surveillance(config.surveillance)

    aggregated = {}
    for label, agg_config in config.aggregated.items():
        path = Path(args.aggregated) / agg_config.trajectory_file
        print(f"\nLoading '{label}' aggregated trajectories from {path} ...")
        aggregated[label] = read_aggregated(str(path), agg_config)
        print(aggregated[label].tail())

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
    to_location = SUBMISSION_PROFILES[config.profile]["location"]
    agg_quantiles = {
        label: _quantiles(
            df,
            config.aggregated[label].target_column,
            config.aggregated[label].date_column,
            config.aggregated[label].location_column,
            to_location,
        )
        for label, df in aggregated.items()
    }
    single_quantiles = None
    if single_strain is not None:
        single_quantiles = _quantiles(
            single_strain,
            config.single_strain.target_column,
            config.single_strain.date_column,
            config.single_strain.location_column,
            to_location,
        )
    # US first, then the rest alphabetically
    all_locations = set().union(*(q["location"] for q in agg_quantiles.values()))
    locations = sorted(all_locations, key=lambda loc: (loc != "US", loc))
    print(f"  Plotting {len(locations)} locations")

    plotting_windows = make_plotting_windows(config)
    fit_start_labels = make_fit_start_labels(config)

    for plot_focus, plotting_window in plotting_windows.items():
        plot_start_date = pd.Timestamp(plotting_window[0])
        plot_end_date = pd.Timestamp(plotting_window[1])
        include_single = "" if config.single_strain is None else " vs Single Strain"
        plot_title = f"{config.submission_week} {' '.join(config.aggregated)}{include_single}"
        plot_fname = (
            f"{plot_focus.lower()}-{config.submission_week}-multistrain{include_single.lower().replace(' ', '_')}.pdf"
        )

        # With out-of-sample data, in-sample points stop at the reference date
        surv_fit_filter = _window(surv_fit, plot_start_date, reference_dt if surv_recent is not None else plot_end_date)
        surv_recent_filter = _window(surv_recent, reference_dt, plot_end_date)

        plot_multistrain_quantiles(
            locations=locations,
            reference_date=reference_dt,
            multistrain_quantiles={
                label: _window(q, plot_start_date, plot_end_date) for label, q in agg_quantiles.items()
            },
            single_strain_quantiles=_window(single_quantiles, plot_start_date, plot_end_date),
            surveillance_fit=surv_fit_filter,
            surveillance_recent=surv_recent_filter,
            fit_start_labels={
                s: l for s, l in fit_start_labels.items() if plot_start_date <= pd.Timestamp(s) <= plot_end_date
            },
            plot_title=plot_title,
            save_path=f"{plots_path}/{plot_fname}",
            subplots_per_row=config.subplots_per_row,
        )


if __name__ == "__main__":
    main()
