# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "pyarrow", "epiweeks"]
# ///
"""
Script to create plots and hubverse submission files from aggregated trajectories.
Takes a file path to a configuration, a file path to aggregated trajectories, and optionally, a directory path for outputs.
Script 
All other options specified via yaml file.

Usage:
    python multistrain_post_aggregation.py --config post_aggregation.yml \
    --aggregated trajectories_aggregated.parquet --output ./multistrain
"""

import argparse
from pathlib import Path

import pandas as pd
from epiweeks import Week

from epymodelingsuite.config_loader import load_post_aggregation_config_from_file
from epymodelingsuite.multistrain.formatter import (
    RATE_TREND_CATEGORIES,
    cols_to_dt,
    compute_quantiles,
    compute_rate_trend_categories,
    compute_rate_trend_pmf,
    create_flusight_submission,
)
from epymodelingsuite.multistrain.plot import (
    make_fit_start_labels,
    make_plotting_windows,
    plot_hosp_multistrain_quantiles,
    pull_single_strain_trajectories,
)
from epymodelingsuite.schema.output import (
    get_flusight_categorical_horizons,
    get_flusight_horizons,
)
from epymodelingsuite.utils.location import convert_location_name_format


def main():
    """Post-aggregation workflow."""
    parser = argparse.ArgumentParser(
        description="Create plots and submissions from aggregated trajectories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to post-aggregation config yaml (required)",
    )

    parser.add_argument(
        "--aggregated",
        type=str,
        default=None,
        help="Path to aggregated trajectories (required)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./multistrain_outputs",
        help="Directory for all outputs. Default: './multistrain_outputs'",
    )

    args = parser.parse_args()

    # Load config
    try:
        configuration = load_post_aggregation_config_from_file(args.config)
    except Exception as e:
        raise ValueError(f"Error loading post-aggregation config: {e}")
    config = configuration.post_aggregation

    reference_date = Week.fromstring(str(config.submission_week)).enddate()
    reference_dt = pd.Timestamp(reference_date)

    # Create output directory if it doesn't exist
    plots_path = Path(f"{args.output}/plots")
    plots_path.mkdir(parents=True, exist_ok=True)
    subs_path = Path(f"{args.output}/submissions")
    subs_path.mkdir(parents=True, exist_ok=True)

    ### Load surveillance

    print("\nLoading surveillance ...")
    try:
        print(f"  Using {config.surveillance.fit_fname} as fit data.")
        surv_fit = pd.read_csv(f"{config.surveillance.directory}/{config.surveillance.fit_fname}")
        surv_recent = None
        if config.surveillance.recent_fname is not None:
            print(f"  Using {config.surveillance.recent_fname} for out-of-sample data.")
            surv_recent = pd.read_csv(f"{config.surveillance.directory}/{config.surveillance.recent_fname}")
    except Exception as e:
        raise ValueError(f"Failed to load surveillance: {e}")
    # Format columns
    surv_fit = cols_to_dt(surv_fit, [config.surveillance.date_column])
    surv_fit["abbreviation"] = surv_fit["location_iso"].apply(lambda k: k.split("-")[-1])
    surv_fit = surv_fit[[config.surveillance.date_column, "abbreviation", config.surveillance.target_column]]
    if surv_recent is not None:
        surv_recent = cols_to_dt(surv_recent, [config.surveillance.date_column])
        surv_recent["abbreviation"] = surv_recent["location_iso"].apply(lambda k: k.split("-")[-1])
        surv_recent = surv_recent[[config.surveillance.date_column, "abbreviation", config.surveillance.target_column]]

    ### Load aggregated trajectories

    print(f"\nLoading aggregated trajectories from {args.aggregated} ...")
    try:
        aggregated = pd.read_parquet(args.aggregated)
    except Exception as e1:
        try:
            aggregated = pd.read_csv(args.aggregated)
        except Exception as e2:
            raise ValueError(
                f"Failed to read {args.aggregated} as either parquet or csv:\n\
                Parquet error: \n{e1}\n\
                CSV error: \n{e2}"
            )
    # Format trajectories
    aggregated = aggregated[
        [
            config.aggregated.date_column,
            config.aggregated.location_column,
            config.aggregated.target_column,
            "sample_id",
        ]
    ]
    aggregated = cols_to_dt(aggregated, [config.aggregated.date_column])
    print(aggregated.tail())

    ### Load single strain trajectories
    if config.single_strain is None:
        single_strain = None
    else:
        print(f"\nDownloading single-strain trajectories from {config.single_strain.bucket} ...")
        try:
            single_strain = pull_single_strain_trajectories(config)
        except Exception as e:
            raise ValueError(f"Failed to download single-strain:\n{e}")
        # Format trajectories
        single_strain = single_strain[
            [config.single_strain.target_column, config.single_strain.date_column, config.single_strain.location_column]
        ]
        single_strain = cols_to_dt(single_strain, [config.single_strain.date_column])
        print(single_strain.tail())

    ### Submission file

    print("\nGenerating submission file ...")

    # Compute rate trend categories for each trajectory
    # Baseline comes from surveillance data, target from trajectories
    print("  Computing rate-trends ...")
    categories_df = compute_rate_trend_categories(
        df=aggregated,
        reference_date=reference_date,
        surveillance_df=surv_fit,
        horizons=get_flusight_categorical_horizons(),
        value_col=config.aggregated.target_column,
        surveillance_date_col=config.surveillance.date_column,
        surveillance_value_col=config.surveillance.target_column,
        surveillance_location_col="abbreviation",
    )
    print(f"    Categories computed: {len(categories_df):,} rows")
    print(f"    Unique populations: {categories_df['population'].nunique()}")
    print(f"    Horizons: {sorted(categories_df['horizon'].unique())}")
    print("\n    Category distribution:")
    print(categories_df["category"].value_counts())
    print("\n    Baseline values come from surveillance (same for all samples per location):")
    print(categories_df.tail())

    # Compute PMF from categories
    pmf_df = compute_rate_trend_pmf(categories_df)
    print(f"    PMF shape: {pmf_df.shape}")
    print(f"    Columns: {pmf_df.columns.tolist()}")
    print(f"\n    Probabilities sum to 1: {(pmf_df[RATE_TREND_CATEGORIES].sum(axis=1).round(6) == 1).all()}")
    # Format PMF for hub submission (long format)
    pmf_long = pmf_df.melt(
        id_vars=["population", "abbreviation", "horizon", "target_date"],
        value_vars=RATE_TREND_CATEGORIES,
        var_name="output_type_id",
        value_name="value",
    )
    # Add required columns for FluSight format
    pmf_long["target"] = "wk flu hosp rate change"
    pmf_long["output_type"] = "pmf"
    pmf_long["location"] = pmf_long["abbreviation"]
    # Reorder columns
    hub_pmf = pmf_long[["location", "target", "horizon", "target_date", "output_type", "output_type_id", "value"]]
    print(f"    PMF hub submission format shape: {hub_pmf.shape}")
    print("\n    Sample rows:")
    print(hub_pmf.head(15))
    # Verify probabilities sum to 1 for each location/horizon
    verification = hub_pmf.groupby(["location", "horizon"])["value"].sum()
    print("    All PMFs sum to 1:", (verification.round(6) == 1).all())
    print(f"\n    Total PMF entries: {len(hub_pmf)}")
    print(f"    Locations: {hub_pmf['location'].nunique()}")
    print(f"    Horizons: {sorted(hub_pmf['horizon'].unique())}")
    print(f"    Categories per location/horizon: {len(RATE_TREND_CATEGORIES)}")

    # Create complete FluSight submission
    print("  Creating submission file ...")
    submission = create_flusight_submission(
        trajectories_df=aggregated,
        pmf_df=pmf_df,
        reference_date=reference_date,
        horizons=get_flusight_horizons(),
        value_col=config.aggregated.target_column,
    )
    print(f"    Submission shape: {submission.shape}")
    print(f"\n    Columns: {submission.columns.tolist()}")
    print(f"\n    Targets: {submission['target'].unique()}")
    print(f"    Output types: {submission['output_type'].unique()}")
    print(f"    Horizons: {sorted(submission['horizon'].unique())}")
    print(f"    Locations: {submission['location'].nunique()}")
    print("\n    Rows per target:")
    print(submission.groupby(["target", "output_type"]).size())
    sub_file = f"{reference_date}-{config.submission.model_name}.parquet"
    submission.to_parquet(f"{subs_path}/{sub_file}")
    print(f"Saved submission to {subs_path}/{sub_file}\n")

    ### Generate plots

    print("\nGenerating plots ...")
    # Compute quantiles
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
    print(f"  Quantiles shape: {agg_quantiles.shape}")
    print(f"  Quantile columns: {[c for c in agg_quantiles.columns if c.startswith('q')]}")
    print(agg_quantiles.tail())
    # Handle dates
    plotting_windows = make_plotting_windows(config)
    fit_start_labels = make_fit_start_labels(config)

    # Create plots
    for plot_focus, plotting_window in plotting_windows.items():
        plot_start_date = pd.Timestamp(plotting_window[0])
        plot_end_date = pd.Timestamp(plotting_window[1])
        include_single = "" if config.single_strain is None else " vs Single Strain"
        plot_title = f"{config.submission_week} Multistrain {config.aggregated.target_column}{include_single}"
        plot_fname = f"{plot_focus.lower()}-{config.submission_week}-{config.aggregated.target_column}-multistrain{include_single.lower().replace(' ', '_')}.pdf"
        plot_fpath = f"{plots_path}/{plot_fname}"

        # Format data
        agg_quantiles_filter = (
            agg_quantiles[
                (agg_quantiles[config.aggregated.date_column] >= plot_start_date)
                & (agg_quantiles[config.aggregated.date_column] < plot_end_date)
            ]
            .sort_values(config.aggregated.date_column)
            .rename(columns={config.aggregated.date_column: "date"})
        )
        single_quantiles_filter = (
            single_quantiles[
                (single_quantiles[config.single_strain.date_column] >= plot_start_date)
                & (single_quantiles[config.single_strain.date_column] < plot_end_date)
            ]
            .sort_values(config.single_strain.date_column)
            .rename(columns={config.single_strain.date_column: "date"})
        )
        surv_fit_filter = (
            surv_fit[
                (surv_fit[config.surveillance.date_column] >= plot_start_date)
                & (surv_fit[config.surveillance.date_column] < plot_end_date)
            ]
            .sort_values(config.surveillance.date_column)
            .rename(columns={config.surveillance.date_column: "date", config.surveillance.target_column: "target"})
        )
        surv_recent_filter = None
        if surv_recent is not None:
            surv_fit_filter = surv_fit_filter[surv_fit_filter["date"] < reference_dt]
            surv_recent_filter = (
                surv_recent[
                    (surv_recent[config.surveillance.date_column] >= reference_dt)
                    & (surv_recent[config.surveillance.date_column] < plot_end_date)
                ]
                .sort_values(config.surveillance.date_column)
                .rename(columns={config.surveillance.date_column: "date", config.surveillance.target_column: "target"})
            )

        # Plotting function
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
