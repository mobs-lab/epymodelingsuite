import logging
import os
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import tempfile
from tempfile import TemporaryDirectory
from datetime import date
from epiweeks import Week

from ..schema.output import (
    get_flusight_quantiles,
)
from ..schema.post_aggregation import PostAggregationConfiguration
from ..utils.location import convert_location_name_format


logger = logging.getLogger(__name__)


def pull_single_strain_trajectories(config: PostAggregationConfiguration) -> pd.DataFrame:
    """
    Pull a single-strain trajectory file from Google Cloud Storage.
    Requires cloud authorization and gcloud.

    Downloads e.g. trajectories_projection_transitions.csv.gz files (NOT runner artifacts).

    Parameters
    ----------
    config: str
        Loaded post-aggregation configuration object.

    Returns
    -------
    pd.DataFrame
        DataFrame with trajectories.
    """

    source = config.single_strain

    trajectories = pd.DataFrame()
    with tempfile.TemporaryDirectory() as tempdir:
        if source.run_id == "latest":
            output = subprocess.run(["gcloud","storage","ls",
                                     f"{source.bucket}/{source.experiment}"], 
                                    capture_output=True, text=True)
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
                prepend_err = f"Warning: setting single_strain.run_id to 'any' will\
                fail if more than one matching trajectory file exists.\n"
                logger.warn(prepend_err)
            raise ValueError(
                f"{prepend_err}Failed to download: {source.experiment}\n\
                    Exit code: {exit_code}\n\
                    Source: {source_location}\n\
                    Target: {target_location}"
            )

    return trajectories


def make_plotting_windows(config: PostAggregationConfiguration) -> dict[str,tuple[date,date]]:
    """Create a dictionary with plotting windows and identifying names"""
    windows = {}
    season_start = Week.fromstring(str(config.plot.season_start_week))
    season_end = Week.fromstring(str(config.plot.season_end_week))
    focus_start = Week.fromstring(str(config.plot.focus_start_week))
    focus_end = Week.fromstring(str(config.plot.focus_end_week))
    if (season_start == focus_start) and (season_end == focus_end):
        windows["Focus"] = (season_start.startdate(), season_end.enddate())
        return windows
    elif season_start == focus_start:
        windows["Focus"] = (season_start.startdate(), focus_end.enddate())
        windows["Season"] = (season_start.startdate(), season_end.enddate())
        return windows
    elif season_end == focus_end:
        windows["Focus"] = (focus_start.startdate(), season_end.enddate())
        windows["Season"] = (season_start.startdate(), season_end.enddate())
        return windows
    else:
        # weeks should all be different
        msg = "Unexpected input/behavior, check for bug in date logic and validation"
        assert len(set([season_start,season_end,focus_start,focus_end])) == 4, msg
        windows["Focus"] = (focus_start.startdate(), focus_end.enddate())
        windows["Season"] = (season_start.startdate(), season_end.enddate())
        return windows


def make_fit_start_labels(
    config: PostAggregationConfiguration,
) -> dict[date,str]:
    """Resolve names and dates for fitting window starts"""
    fit_start_dates = set(
        [Week.fromstring(str(fit.week)).enddate()
         for fit in config.plot.strain_fit_starts]
    )
    fit_start_labels = {}
    for fit_start_date in fit_start_dates:
        for fit in config.plot.strain_fit_starts:
            oldlabel = fit_start_labels.setdefault(fit_start_date, None)
            if oldlabel is None:
                label = f"{fit.label}"
            else:
                label = f"{oldlabel}/{fit.label}"
            fit_start_labels[fit_start_date] = label
    return fit_start_labels


def plot_hosp_multistrain_quantiles(
    populations: list[str],
    reference_date: date,
    multistrain_quantiles: pd.DataFrame,
    single_strain_quantiles: pd.DataFrame | None,
    surveillance_fit: pd.DataFrame,
    surveillance_recent: pd.DataFrame | None,
    fit_start_labels: dict[date,str],
    plot_title: str,
    save_path: str | None = None,
    subplots_per_row: int = 4,
) -> (Figure, np.ndarray[Axes]):
    """"""
    sp_rows = int(np.floor(len(populations)/subplots_per_row)) + int(bool(len(populations)%subplots_per_row))
    figheight = int(np.floor((sp_rows/subplots_per_row)*20))
    fig, axes = plt.subplots(sp_rows, subplots_per_row, figsize=(20,figheight))
    axes = axes.flatten()

    for idx, pop in enumerate(populations):
        ax = axes[idx]
        abbrev = convert_location_name_format(
            value=pop,
            output_format="abbreviation",
            location_type="iso"
        )

        # Get data for this population
        agg_pop = multistrain_quantiles[multistrain_quantiles["abbreviation"] == abbrev]
        surv_fit_pop = surveillance_fit[surveillance_fit["abbreviation"] == abbrev]
        if surveillance_recent is not None:
            surv_recent_pop = surveillance_recent[surveillance_recent["abbreviation"] == abbrev]
        if single_strain_quantiles is not None:
            single_pop = single_strain_quantiles[single_strain_quantiles["abbreviation"] == abbrev]
        
        # Plot multistrain (blue)
        ax.fill_between(agg_pop["date"], agg_pop["q025"], agg_pop["q975"],
                        alpha=0.15, color="blue", label="Multistrain 95% PI")
        ax.fill_between(agg_pop["date"], agg_pop["q250"], agg_pop["q750"],
                        alpha=0.3, color="blue", label="Multistrain 50% PI")
        ax.plot(agg_pop["date"], agg_pop["q500"], "b-", linewidth=1.5, label="Multistrain median")

        # Plot single strain (green)
        if single_strain_quantiles is not None:
            ax.fill_between(single_pop["date"], single_pop["q025"], single_pop["q975"], 
                            alpha=0.15, color="green", label="Single strain 95% PI")
            ax.fill_between(single_pop["date"], single_pop["q250"], single_pop["q750"], 
                            alpha=0.3, color="green", label="Single strain 50% PI")
            ax.plot(single_pop["date"], single_pop["q500"], "g-", linewidth=1.5, label="Single strain median")

        # Plot out-of-sample surveillance (black markers with white fill)
        if surveillance_recent is not None:
            ax.scatter(surv_recent_pop["date"], surv_recent_pop["target"],
                       color="white", edgecolor="black", s=15, zorder=5, label="Out of sample")
        # Plot in-sample surveillance (black markers)
        ax.scatter(surv_fit_pop["date"], surv_fit_pop["target"],
                   color="black", s=15, zorder=5, label="Observed")

        # Formatting
        ax.set_title(abbrev, fontsize=14, fontweight="bold")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)

        # Last date of fitting window
        ax.axvline(reference_date - pd.Timedelta(weeks=1), color='black', linestyle='--', linewidth=1, label="Last in-sample date")
        
        # Fitting window starts
        if len(fit_start_labels) > 0:
            line_types = ['--', '-.', ':', '-']
            fit_idx = 0
            for fit_start, fit_label in fit_start_labels.items():
                if fit_idx > 3:
                    logger.warn(
                        f"Cannot display more than 4 distinct strain fit starts. Ignoring fit start for {fit_label}"
                    )
                else:
                    ax.axvline(fit_start, color="red", linestyle=line_types[fit_idx],
                               linewidth=1, label=f"{fit_label} fit")
                fit_idx += 1
        
        # Only show legend for first subplot
        if idx == 0:
            ax.legend(fontsize=8, loc='upper right')
        # Hide any unused subplots
        for idx in range(len(populations), len(axes)):
            axes[idx].set_visible(False)

    # Finish
    plt.suptitle(plot_title, fontsize=14, y=1.01)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved plot at {save_path}")
    
    return fig, axes
        
        
    
        
    