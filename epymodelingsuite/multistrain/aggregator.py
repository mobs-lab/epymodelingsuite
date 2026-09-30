import logging
import os
import subprocess
import sys
import tempfile
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from ..schema.aggregation import (
    AggregationConfiguration,
    AggregationStrategyEnum,
    BaselineStrategyEnum,
    SamplingStrategyEnum,
)

#############
### UTILS ###
#############


def validate_strains(
    strains: dict[str, pd.DataFrame],
    config: AggregationConfiguration,
) -> pd.DataFrame:
    """
    Ensure strain trajectories match sources in config.

    Parameters
    ----------
    strains : dict
        Dictionary mapping strain names to their trajectory DataFrames
        Example: {'h1': df_h1, 'h3': df_h3, 'b': df_b}
        Or: {'a': df_a, 'b': df_b}
    config: AggregationConfiguration
        Config object with settings

    Raises
    ------
    ValueError
        if strains do not match
    """
    source_strains = [source.strain for source in config.sources]
    traj_strains = list(strains.keys())
    if (len(source_strains) != len(traj_strains)) or (set(source_strains) != set(traj_strains)):
        raise ValueError(
            f"Received trajectories for strains {traj_strains}, \
            but configurations for strains {source_strains}."
        )


def pull_single_strain_submission(source_location: str, fname: str) -> pd.DataFrame:
    """
    TO BE DEPRECATED
    Pull single-strain submission files from Google Cloud Storage.
    Requires cloud authorization and gcloud.

    Downloads e.g. output_hub_formatted.csv.gz files (NOT runner artifacts).

    Parameters
    ----------
    source_location: str
        Location of folder in bucket containing desiredfile
    fname: str
        Name of desired file

    Returns
    -------
    pd.DataFrame
        DataFrame with submission file
    """
    # Download pipeline submission file
    submission = pd.DataFrame()
    with tempfile.TemporaryDirectory() as tempdir:
        # Retrieve latest run
        command = f"gcloud storage ls '{source_location}'"
        output = subprocess.run(["gcloud", "storage", "ls", f"{source_location}"], capture_output=True, text=True)
        if output.returncode == 0:
            assert type(output.stdout) is str
            run_id = output.stdout.split("/")[-2]
        else:
            raise ValueError(f"Failed to find runs for experiment {source_location}")
        source_location = f"{source_location}/{run_id}/outputs/*/{fname}"
        target_location = f"{tempdir}/{fname}"

        command = f"gcloud storage cp {source_location} {target_location}"
        exit_code = os.system(command)

        if exit_code == 0:
            print(f"Downloaded: {fname}")
            # Load the downloaded data
            submission = pd.read_csv(target_location, dtype={"output_type_id": str})
        else:
            raise ValueError(f"Failed to download: {source_location}\nExit code: {exit_code}")

    return submission


def pull_trajectory_projections(
    config: AggregationConfiguration, tempdir: TemporaryDirectory
) -> dict[str, pd.DataFrame]:
    """
    Pull trajectory projection files from Google Cloud Storage.
    Requires cloud authorization and gcloud.

    Downloads e.g. trajectories_projection_transitions.csv.gz files (NOT runner artifacts).

    Parameters
    ----------
    config: AggregationConfiguration
        Config object containing source information
    tempdir: tempfile.TemporaryDirectory
        Directory for storing pulled files

    Returns
    -------
    dict[str, pd.DataFrame]
        trajectories: {subtype: DataFrame}
    """
    trajectories = {}

    for source in config.sources:
        # Download trajectory projections (not runner artifacts)
        if source.run_id == "latest":
            output = subprocess.run(
                ["gcloud", "storage", "ls", f"{config.bucket}/{source.experiment}"], capture_output=True, text=True
            )
            if output.returncode == 0:
                assert type(output.stdout) is str
                run_id = output.stdout.split("/")[-2]
            else:
                raise ValueError(
                    f"Failed to find runs for experiment: {source.experiment}\n\
                        Exit code: {output.returncode}\n\
                        Stdout: {output.stdout}\n\
                        Stderr: {output.stderr} "
                )
        elif source.run_id == "any":
            run_id = "*"
        else:
            run_id = source.run_id
        source_location = f"{config.bucket}/{source.experiment}/{run_id}/outputs/*/{source.trajectory_file}"
        target_location = f"{tempdir}/{source.strain}_{source.trajectory_file}"

        command = f"gcloud storage cp -r {source_location} {target_location}"
        exit_code = os.system(command)

        if exit_code == 0:
            logger.info(f"Downloaded: {source.strain}")
            # Load the downloaded data
            trajectories[source.strain] = pd.read_csv(target_location, parse_dates=[source.date_column])
        else:
            prepend_err = ""
            if source.run_id == "any":
                prepend_err = "Warning: setting source.run_id to 'any' will\
                fail if more than one matching trajectory file exists.\n"
                logger.warning(prepend_err)
            raise ValueError(
                f"{prepend_err}Failed to download: {source.experiment}\n\
                    Exit code: {exit_code}\n\
                    Source: {source_location}\n\
                    Target: {target_location}"
            )

    return trajectories


def merge_strain_trajectories(
    strains: dict[str, pd.DataFrame],
    mapping: pd.DataFrame,
    config: AggregationConfiguration,
) -> pd.DataFrame:
    """
    Merge trajectory DataFrames based on a mapping of sim_id.

    Parameters
    ----------
    strains : dict
        Dictionary mapping strain names to their trajectory DataFrames
        Example: {'h1': df_h1, 'h3': df_h3, 'b': df_b}
        Or: {'a': df_a, 'b': df_b}
    mapping: pd.DataFrame
        DataFrame mapping sample_id to strain sim_ids
    config: AggregationConfiguration
        Config object with sampling settings

    Returns
    -------
    pd.DataFrame
        DataFrame with merged trajectories:
        strain trajectories in columns "target_{strain}"
        strain sim_id in columns "{strain}"
        identifying columns "sample_id", "location", and "date"

    """
    validate_strains(strains, config)

    # Standardize strain DFs
    formatted_strains = {}
    for source in config.sources:
        traj_df = strains[source.strain].copy()
        formatted = traj_df.rename(
            columns={
                source.target_column: "target",
                source.date_column: "date",
                source.location_column: "location",
                source.sim_id: "sim_id",
            }
        )[["date", "location", "target", "sim_id"]]
        formatted_strains[source.strain] = formatted

    # Start with first strain
    first_source = config.sources[0]
    combined = (
        mapping.merge(
            formatted_strains[first_source.strain], left_on=f"sim_id_{first_source.strain}", right_on="sim_id"
        )
        .rename(columns={"target": f"target_{first_source.strain}"})
        .drop("sim_id", axis=1)
    )

    # Merge remaining strains
    for source in config.sources[1:]:
        combined = (
            combined.merge(
                formatted_strains[source.strain],
                left_on=[f"sim_id_{source.strain}", "location", "date"],
                right_on=["sim_id", "location", "date"],
                how="outer",
            )
            .rename(columns={"target": f"target_{source.strain}"})
            .drop("sim_id", axis=1)
        )

    return combined


################
### SAMPLING ###
################


def _random_sample_mapping(
    strains: dict[str, pd.DataFrame],
    config: AggregationConfiguration,
) -> pd.DataFrame:
    """
    Random sampling: independently shuffle strain sim_ids and sum trajectories.

    Parameters
    ----------
    strains : dict
        Dictionary mapping strain names to their trajectory DataFrames
        Example: {'h1': df_h1, 'h3': df_h3, 'b': df_b}
        Or: {'a': df_a, 'b': df_b}
    config: AggregationConfiguration
        Config object with sampling settings

    Returns
    -------
    pd.DataFrame
        DataFrame mapping sample_id to strain sim_ids
    """
    rng = np.random.default_rng(seed=config.random_seed)
    n_samples = config.sampling.n_samples

    # Sample sim_ids for each strain and create mapping
    mapping_data = {"sample_id": range(n_samples)}
    for source in config.sources:
        # Sample sim_ids
        sim_ids = strains[source.strain][source.sim_id].unique()
        if len(sim_ids) < n_samples:
            raise ValueError(
                f"Source {source.strain} contains {len(sim_ids)} \
                trajectories but {n_samples} were requested."
            )
        sampled_ids = rng.choice(sim_ids, size=n_samples, replace=False)
        mapping_data[f"sim_id_{source.strain}"] = sampled_ids
    return pd.DataFrame(mapping_data)


def dispatch_strain_sampler(
    strains: dict[str, pd.DataFrame],
    config: AggregationConfiguration,
) -> pd.DataFrame:
    """
    Dispatch multistrain trajectory sampling methods.

    Parameters
    ----------
    strains : dict
        Dictionary mapping strain names to their trajectory DataFrames
        Example: {'h1': df_h1, 'h3': df_h3, 'b': df_b}
        Or: {'a': df_a, 'b': df_b}
    config: AggregationConfiguration
        Config object with sampling settings

    Returns
    -------
    pd.DataFrame
        DataFrame mapping sample_id to strain sim_ids
    """
    validate_strains(strains, config)

    match config.sampling.method:
        case SamplingStrategyEnum.random:
            return _random_sample_mapping(strains, config)
        case _:
            raise NotImplementedError(f"Invalid sampling method: {config.sampling.method}")


###################
### AGGREGATION ###
###################


def _aggregate_sum(
    merged_trajectories: pd.DataFrame,
    config: AggregationConfiguration,
) -> pd.DataFrame:
    """
    Dispatch multistrain trajectory aggregation methods.

    Parameters
    ----------
    merged_trajectories: pd.DataFrame
        DataFrame with merged trajectories in columns "target_{strain}",
        sim_id in columns "{strain}",
        and identifying columns "sample_id", "location", and "date"
    config: AggregationConfiguration
        Config object with sampling settings

    Returns
    -------
    pd.DataFrame
        DataFrame with summed trajectories
    """
    aggregated = merged_trajectories.copy()

    # Sum target values from all strains
    target_cols = [f"target_{source.strain}" for source in config.sources]
    agg_colname = "target_sum"
    aggregated[agg_colname] = aggregated[target_cols].fillna(0).sum(axis=1)

    # Clean up columns
    output_cols = (
        ["sample_id"]
        + [f"sim_id_{source.strain}" for source in config.sources]
        + ["location", "date"]
        + target_cols
        + [agg_colname]
    )

    # Filter to only columns that exist
    output_cols = [col for col in output_cols if col in aggregated.columns]

    return aggregated[output_cols]


def dispatch_strain_aggregator(
    merged_trajectories: pd.DataFrame,
    config: AggregationConfiguration,
) -> pd.DataFrame:
    """
    Dispatch multistrain trajectory aggregation methods.

    Parameters
    ----------
    merged_trajectories: pd.DataFrame
        DataFrame with merged trajectories in columns "target_{strain}"
    config: AggregationConfiguration
        Config object with sampling settings

    Returns
    -------
    pd.DataFrame
        DataFrame with aggregated trajectories
    """
    match config.aggregate_method:
        case AggregationStrategyEnum.sum:
            return _aggregate_sum(merged_trajectories, config)
        case _:
            raise NotImplementedError(f"Invalid aggregation method: {config.aggregate_method}")


################
### BASELINE ###
################


def _baseline_negative_binomial(
    aggregated_trajectories: pd.DataFrame,
    combined: pd.DataFrame,
    kvals: int | list[int],
) -> pd.DataFrame:
    """
    Dispatch post-aggregation baseline addition.

    Parameters
    ----------
    aggregated_trajectories: pd.DataFrame
        DataFrame with aggregated trajectories
    config: AggregationConfiguration
        Config object with settings

    Returns
    -------
    pd.DataFrame
        DataFrame with baseline noise added to aggregated trajectories in columns "target_baseline_k{dispersion_parameter}"
    """
    rng = np.random.default_rng()
    aggregated = aggregated_trajectories.copy()

    if not isinstance(kvals, list):
        kvals = [kvals]

    for k in kvals:
        p_vals = k / (k + combined["baseline"])
        baseline_sample = rng.negative_binomial(k, p_vals)
        aggregated[f"target_baseline_k{k}"] = combined["target_sum"] + baseline_sample

    return aggregated


def dispatch_baseline(
    aggregated_trajectories: pd.DataFrame,
    config: AggregationConfiguration,
) -> pd.DataFrame:
    """
    Dispatch post-aggregation baseline addition.

    Parameters
    ----------
    aggregated_trajectories: pd.DataFrame
        DataFrame with aggregated trajectories in column "target_sum"
    config: AggregationConfiguration
        Config object with settings

    Returns
    -------
    pd.DataFrame
        DataFrame with baseline noise added to aggregated trajectories in columns "target_baseline_k{config.baseline.dispersion_values}"
    """
    try:
        filename = os.path.join(
            os.path.dirname(sys.modules[__name__].__file__), f"../data/{config.baseline.observed_means}"
        )
        baselines_avg = pd.read_csv(filename)
    except Exception as e:
        print(os.getcwd())
        raise ValueError(
            f"Baseline file {config.baseline.observed_means} not found: {e}\n\
            Tried to read location {filename}"
        )

    traj_loc_fmt = (
        "location_name_epydemix"
        if aggregated_trajectories.location.str.contains("__").any()
        else "location_name_epydemix_deprecated"
    )
    combined = aggregated_trajectories.merge(baselines_avg, how="inner", left_on="location", right_on=traj_loc_fmt)[
        ["location", "baseline", "target_sum"]
    ]

    match config.baseline.method:
        case BaselineStrategyEnum.negative_binomial:
            return _baseline_negative_binomial(aggregated_trajectories, combined, config.baseline.dispersion_values)
        case _:
            raise NotImplementedError(f"Invalid baseline method: {config.baseline.method}")
