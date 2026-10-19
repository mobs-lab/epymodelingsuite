import logging
import os
import subprocess
import tempfile
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from epiweeks import Week
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..schema.plot import PlotConfiguration

logger = logging.getLogger(__name__)

MULTISTRAIN_COLORS = ["blue", "darkorange", "darkred", "aqua", "olive"]


def pull_single_strain_trajectories(config: PlotConfiguration) -> pd.DataFrame:
    """
    Pull a single-strain trajectory file from Google Cloud Storage.
    Requires cloud authorization and gcloud.

    Downloads e.g. trajectories_projection_transitions.csv.gz files (NOT runner artifacts).

    Parameters
    ----------
    config: str
        Loaded plot configuration object.

    Returns
    -------
    pd.DataFrame
        DataFrame with trajectories.
    """
    source = config.single_strain

    trajectories = pd.DataFrame()
    with tempfile.TemporaryDirectory() as tempdir:
        if source.run_id == "latest":
            output = subprocess.run(
                ["gcloud", "storage", "ls", f"{source.bucket}/{source.experiment}"], capture_output=True, text=True
            )
            if output.returncode == 0:
                assert type(output.stdout) is str
                run_id = output.stdout.split("/")[-2]
            else:
                raise ValueError(
                    f"Couldn't find 'latest' trajectories for experiment: {source.experiment}\n\
                        Exit code: {output.returncode}\n\
                        Stdout: {output.stdout}\n\
                        Stderr: {output.stderr} "
                )
        elif source.run_id == "any":
            run_id = "*"
        else:
            run_id = source.run_id
        source_location = f"{source.bucket}/{source.experiment}/{run_id}/outputs/*/{source.trajectory_file}"
        target_location = f"{tempdir}/single_{source.trajectory_file}"

        command = f"gcloud storage cp -r {source_location} {target_location}"
        exit_code = os.system(command)

        if exit_code == 0:
            logger.info(f"Downloaded single-strain: {source.experiment}")
            # Load the downloaded data
            trajectories = pd.read_csv(target_location, parse_dates=[source.date_column])
        else:
            prepend_err = ""
            if source.run_id == "any":
                prepend_err = "Warning: setting single_strain.run_id to 'any' will\
                fail if more than one matching trajectory file exists.\n"
                logger.warning(prepend_err)
            raise ValueError(
                f"{prepend_err}Failed to download: {source.experiment}\n\
                    Exit code: {exit_code}\n\
                    Source: {source_location}\n\
                    Target: {target_location}"
            )

    return trajectories


def make_plotting_windows(config: PlotConfiguration) -> dict[str, tuple[date, date]]:
    """Create a dictionary with plotting windows and identifying names"""
    windows = {}
    season_start = Week.fromstring(str(config.season_start_week))
    season_end = Week.fromstring(str(config.season_end_week))
    focus_start = Week.fromstring(str(config.focus_start_week))
    focus_end = Week.fromstring(str(config.focus_end_week))
    if (season_start == focus_start) and (season_end == focus_end):
        windows["Focus"] = (season_start.startdate(), season_end.enddate())
        return windows
    if season_start == focus_start:
        windows["Focus"] = (season_start.startdate(), focus_end.enddate())
        windows["Season"] = (season_start.startdate(), season_end.enddate())
        return windows
    if season_end == focus_end:
        windows["Focus"] = (focus_start.startdate(), season_end.enddate())
        windows["Season"] = (season_start.startdate(), season_end.enddate())
        return windows
    # weeks should all be different
    msg = "Unexpected input/behavior, check for bug in date logic and validation"
    assert len(set([season_start, season_end, focus_start, focus_end])) == 4, msg
    windows["Focus"] = (focus_start.startdate(), focus_end.enddate())
    windows["Season"] = (season_start.startdate(), season_end.enddate())
    return windows


def make_fit_start_labels(
    config: PlotConfiguration,
) -> dict[date, str]:
    """Resolve names and dates for fitting window starts"""
    fit_start_dates = set([Week.fromstring(str(fit.week)).enddate() for fit in config.strain_fit_starts])
    fit_start_labels = {}
    for fit_start_date in fit_start_dates:
        for fit in config.strain_fit_starts:
            oldlabel = fit_start_labels.setdefault(fit_start_date, None)
            if oldlabel is None:
                label = f"{fit.label}"
            else:
                label = f"{oldlabel}/{fit.label}"
            fit_start_labels[fit_start_date] = label
    return fit_start_labels


def plot_multistrain_quantiles(
    locations: list[str],
    reference_date: date,
    multistrain_quantiles: dict[str, pd.DataFrame],
    single_strain_quantiles: pd.DataFrame | None,
    surveillance_fit: pd.DataFrame | None,
    surveillance_recent: pd.DataFrame | None,
    fit_start_labels: dict[date, str],
    plot_title: str,
    save_path: str | None = None,
    subplots_per_row: int = 4,
) -> (Figure, np.ndarray[Axes]):
    """
    Plot multistrain (and optionally single-strain) quantile ribbons against surveillance, one panel per location.
    Each labelled entry of `multistrain_quantiles` is drawn in its own color.

    All data frames are keyed by a `location` column holding hub location ids (the submission profile's
    location transform), which is also used as the panel title.
    """
    sp_rows = int(np.floor(len(locations) / subplots_per_row)) + int(bool(len(locations) % subplots_per_row))
    figheight = int(np.floor((sp_rows / subplots_per_row) * 15))
    fig, axes = plt.subplots(sp_rows, subplots_per_row, figsize=(20, figheight))
    axes = axes.flatten()

    for idx, location in enumerate(locations):
        ax = axes[idx]

        # Get data for this location
        if surveillance_fit is not None:
            surv_fit_pop = surveillance_fit[surveillance_fit["location"] == location]
        if surveillance_recent is not None:
            surv_recent_pop = surveillance_recent[surveillance_recent["location"] == location]
        if single_strain_quantiles is not None:
            single_pop = single_strain_quantiles[single_strain_quantiles["location"] == location]

        # Plot multistrain, one color per label
        for agg_idx, (agg_label, agg_quantiles) in enumerate(multistrain_quantiles.items()):
            agg_pop = agg_quantiles[agg_quantiles["location"] == location]
            color = MULTISTRAIN_COLORS[agg_idx % len(MULTISTRAIN_COLORS)]
            ax.fill_between(
                agg_pop["date"], agg_pop["q025"], agg_pop["q975"], alpha=0.15, color=color, label=f"{agg_label} 95% PI"
            )
            ax.fill_between(
                agg_pop["date"], agg_pop["q250"], agg_pop["q750"], alpha=0.3, color=color, label=f"{agg_label} 50% PI"
            )
            ax.plot(agg_pop["date"], agg_pop["q500"], color=color, linewidth=1.5)

        # Plot single strain (green), 50% PI only to limit clutter
        if single_strain_quantiles is not None:
            ax.fill_between(
                single_pop["date"],
                single_pop["q250"],
                single_pop["q750"],
                alpha=0.3,
                color="green",
                label="Single strain 50% PI",
            )
            ax.plot(single_pop["date"], single_pop["q500"], "g-", linewidth=1.5, label="Single strain median")

        # Plot out-of-sample surveillance (black markers with white fill)
        if surveillance_recent is not None:
            ax.scatter(
                surv_recent_pop["date"],
                surv_recent_pop["target"],
                color="white",
                edgecolor="black",
                s=15,
                zorder=5,
                label="Out of sample",
            )
        # Plot in-sample surveillance (black markers)
        if surveillance_fit is not None:
            ax.scatter(surv_fit_pop["date"], surv_fit_pop["target"], color="black", s=15, zorder=5, label="Observed")

        # Formatting
        ax.set_title(location, fontsize=14, fontweight="bold")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)

        # Last date of fitting window
        ax.axvline(
            reference_date - pd.Timedelta(weeks=1),
            color="black",
            linestyle="--",
            linewidth=1,
            label="Last in-sample date",
        )

        # Fitting window starts
        if len(fit_start_labels) > 0:
            line_types = ["--", "-.", ":", "-"]
            fit_idx = 0
            for fit_start, fit_label in fit_start_labels.items():
                if fit_idx > 3:
                    logger.warning(
                        f"Cannot display more than 4 distinct strain fit starts. Ignoring fit start for {fit_label}"
                    )
                else:
                    ax.axvline(
                        fit_start, color="red", linestyle=line_types[fit_idx], linewidth=1, label=f"{fit_label} fit"
                    )
                fit_idx += 1

        # Only show legend for first subplot
        if idx == 0:
            ax.legend(fontsize=7)
        # Hide any unused subplots
        for idx in range(len(locations), len(axes)):
            axes[idx].set_visible(False)

    # Finish
    plt.suptitle(plot_title, fontsize=14, y=1.01)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved plot at {save_path}")

    return fig, axes
