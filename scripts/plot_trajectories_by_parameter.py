import argparse
import gzip
import re
from datetime import date, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from epymodelingsuite.visualization.core import plot_trajectories_by_parameter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot individual trajectories colored by a parameter (e.g. Reff) "
            "from a trajectories_projection_transitions CSV/CSV.GZ file."
        )
    )
    parser.add_argument(
        "--trajectories",
        required=True,
        help="Path to trajectories CSV or CSV.GZ (e.g. *_trajectories_projection_transitions.csv.gz).",
    )
    parser.add_argument(
        "--parameters",
        required=True,
        help="Path to projection parameters CSV or CSV.GZ (e.g. *_projection_parameters_long.csv.gz).",
    )
    parser.add_argument(
        "--value-col",
        default="hospitalizations",
        help="Column in the trajectories file to plot on the y-axis (default: hospitalizations).",
    )
    parser.add_argument(
        "--parameter-col",
        default="Reff",
        help="Single column to use for coloring trajectories (default: Reff). Ignored if --parameter-cols is set.",
    )
    parser.add_argument(
        "--parameter-cols",
        nargs="+",
        default=None,
        help=(
            "One or more columns to use for coloring trajectories. "
            "If provided, a separate figure is created for each column."
        ),
    )
    parser.add_argument(
        "--observed",
        default=None,
        help=(
            "Optional observed/ground-truth CSV to overlay as black circles. "
            "Example: tutorials/data/covid-hospital-admissions2.csv"
        ),
    )
    parser.add_argument(
        "--observed-date-col",
        default="date",
        help="Observed CSV date column (default: date).",
    )
    parser.add_argument(
        "--observed-value-col",
        default="hospitalizations",
        help="Observed CSV value column (default: hospitalizations).",
    )
    parser.add_argument(
        "--observed-location-col",
        default="location_iso",
        help="Observed CSV location column (default: location_iso).",
    )
    parser.add_argument(
        "--date-col",
        default="date",
        help="Column containing dates (default: date).",
    )
    parser.add_argument(
        "--id-cols",
        nargs="+",
        default=["primary_id", "seed", "population"],
        help="Identifier columns shared across trajectories (default: primary_id seed population).",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=100,
        help="Maximum number of trajectories (sim_id values) per location to plot (default: 100).",
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=1,
        help="Number of subplot columns in the grid (default: 1).",
    )
    parser.add_argument(
        "--figsize-per-panel",
        type=float,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(6.0, 4.5),
        help="Size of each panel in inches (default: 6.0 4.5).",
    )
    parser.add_argument(
        "--output-prefix",
        default="trajectories_by_parameter",
        help="Prefix for output files (PNG/PDF). Default: trajectories_by_parameter.",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="If set, do not write a PDF output (only PNG).",
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="If set, do not write a PNG output (only PDF).",
    )
    parser.add_argument(
        "--line-alpha",
        type=float,
        default=0.6,
        help="Alpha (transparency) for trajectory lines (default: 0.6).",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Seed used to choose the random subset of trajectories (default: 42).",
    )
    return parser.parse_args()


def load_trajectories(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".gz":
        with gzip.open(p, "rt") as f:
            return pd.read_csv(f)
    return pd.read_csv(p)

def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".gz":
        with gzip.open(p, "rt") as f:
            return pd.read_csv(f)
    return pd.read_csv(p)


_PARAM_JOIN_KEYS = ["primary_id", "sim_id", "seed", "population"]


def epiweek_to_saturday(year: int, week: int) -> date:
    # CDC epiweek 1 starts on the Sunday on or before Jan 4
    jan4 = date(year, 1, 4)
    days_since_sunday = (jan4.weekday() + 1) % 7
    week1_sunday = jan4 - timedelta(days=days_since_sunday)
    week_sunday = week1_sunday + timedelta(weeks=week - 1)
    return week_sunday + timedelta(days=6)


def extract_reference_date(path: str) -> date | None:
    m = re.search(r"(\d{6})", Path(path).name)
    if not m:
        return None
    code = m.group(1)
    year, week = int(code[:4]), int(code[4:])
    return epiweek_to_saturday(year, week)


def merge_parameters(df_traj: pd.DataFrame, df_params: pd.DataFrame) -> pd.DataFrame:
    join_keys = [k for k in _PARAM_JOIN_KEYS if k in df_traj.columns and k in df_params.columns]
    param_cols = [c for c in df_params.columns if c not in join_keys]
    return df_traj.merge(df_params[join_keys + param_cols], on=join_keys, how="left")


def main() -> None:
    args = parse_args()

    df_traj = load_trajectories(args.trajectories)
    df_params = load_csv(args.parameters)
    df_traj = merge_parameters(df_traj, df_params)
    df_obs = load_csv(args.observed) if args.observed else None

    ref_date = extract_reference_date(args.trajectories)

    # Determine which parameter columns to use
    if args.parameter_cols:
        param_cols = args.parameter_cols
    else:
        param_cols = [args.parameter_col]

    for param in param_cols:
        fig, _ = plot_trajectories_by_parameter(
            df_trajectories=df_traj,
            df_posteriors=None,  # parameters already merged into df_traj
            df_observed=df_obs,
            value_col=args.value_col,
            parameter_col=param,
            date_col=args.date_col,
            id_cols=tuple(args.id_cols),
            ncols=args.ncols,
            figsize_per_panel=tuple(args.figsize_per_panel),
            max_trajectories=args.max_trajectories,
            observed_date_col=args.observed_date_col,
            observed_value_col=args.observed_value_col,
            observed_location_col=args.observed_location_col,
            line_alpha=args.line_alpha,
            subsample_seed=args.sample_seed,
        )

        if ref_date is not None:
            ref_ts = pd.Timestamp(ref_date)
            for ax in fig.axes:
                if ax.get_visible():
                    ax.axvline(ref_ts, color="black", linestyle="--", linewidth=1, zorder=10)

        prefix = Path(f"{args.output_prefix}_{param}")
        wrote_any = False

        # Some PDF viewers can be finicky with certain filenames; also write a lowercase copy.
        prefix_lower = Path(f"{args.output_prefix}_{param.lower()}")

        if not args.no_png:
            png_path = prefix.with_suffix(".png")
            fig.savefig(png_path, bbox_inches="tight", dpi=150)
            if prefix_lower != prefix:
                fig.savefig(prefix_lower.with_suffix(".png"), bbox_inches="tight", dpi=150)
            wrote_any = True

        if not args.no_pdf:
            pdf_path = prefix.with_suffix(".pdf")
            fig.savefig(pdf_path, bbox_inches="tight", dpi=150)
            if prefix_lower != prefix:
                fig.savefig(prefix_lower.with_suffix(".pdf"), bbox_inches="tight", dpi=150)
            wrote_any = True

        if not wrote_any:
            # Fallback: write PNG if user disabled both by mistake
            png_path = prefix.with_suffix(".png")
            fig.savefig(png_path, bbox_inches="tight", dpi=150)

        plt.close(fig)


if __name__ == "__main__":
    main()

