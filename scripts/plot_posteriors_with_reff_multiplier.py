import argparse
import gzip
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from epymodelingsuite.visualization.core import plot_posterior_histogram_grid, sort_locations_by_state

POSTERIOR_METADATA_COLUMNS = {"sim_id", "location", "population", "primary_id", "seed", "generation"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recreate the multi-location posterior histogram grid from a posteriors CSV/CSV.GZ, "
            "after multiplying all Reff values by a user-provided factor."
        )
    )
    parser.add_argument(
        "--posteriors",
        required=True,
        help="Path to posteriors CSV or CSV.GZ (e.g. *_posteriors.csv.gz).",
    )
    parser.add_argument(
        "--multiplier",
        required=True,
        type=float,
        help="Multiplier applied to every value in the Reff column.",
    )
    parser.add_argument(
        "--reff-col",
        default="Reff",
        help="Name of the Reff column to scale (default: Reff).",
    )
    parser.add_argument(
        "--population-col",
        default="population",
        help="Column containing location/population identifiers (default: population).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=25,
        help="Number of histogram bins (default: 25).",
    )
    parser.add_argument(
        "--start-date-reference",
        default=None,
        help="Optional reference date used if start_date is in posterior columns (format: YYYY-MM-DD).",
    )
    parser.add_argument(
        "--output-prefix",
        default="posteriors_grid_reff_scaled",
        help="Prefix for output files (default: posteriors_grid_reff_scaled).",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="If set, do not write a PDF output.",
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="If set, do not write a PNG output.",
    )
    return parser.parse_args()


def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".gz":
        with gzip.open(p, "rt") as f:
            return pd.read_csv(f)
    return pd.read_csv(p)


def main() -> None:
    args = parse_args()
    df = load_csv(args.posteriors)

    if args.population_col not in df.columns:
        raise ValueError(f"Missing required population column '{args.population_col}' in posteriors file.")
    if args.reff_col not in df.columns:
        raise ValueError(f"Missing required Reff column '{args.reff_col}' in posteriors file.")

    # Apply user-requested scaling to Reff.
    df[args.reff_col] = pd.to_numeric(df[args.reff_col], errors="coerce") * args.multiplier

    location_posteriors = {}
    for location in sort_locations_by_state(df[args.population_col].dropna().unique().tolist()):
        location_posteriors[str(location)] = df[df[args.population_col] == location].copy()

    if not location_posteriors:
        raise ValueError("No location-level posterior data found after reading the input file.")

    params = [col for col in df.columns if col not in POSTERIOR_METADATA_COLUMNS]
    if not params:
        raise ValueError("No posterior parameter columns found to plot.")

    fig, _ = plot_posterior_histogram_grid(
        location_posteriors=location_posteriors,
        parameters=sorted(params),
        bins=args.bins,
        start_date_reference=args.start_date_reference,
    )

    fig.suptitle(f"Posterior Distributions (Reff x {args.multiplier:g})", fontsize=14, y=0.995)
    fig.tight_layout()

    prefix = Path(args.output_prefix)
    wrote_any = False

    if not args.no_png:
        fig.savefig(prefix.with_suffix(".png"), bbox_inches="tight", dpi=150)
        wrote_any = True
    if not args.no_pdf:
        fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight", dpi=150)
        wrote_any = True
    if not wrote_any:
        fig.savefig(prefix.with_suffix(".png"), bbox_inches="tight", dpi=150)

    plt.close(fig)


if __name__ == "__main__":
    main()
