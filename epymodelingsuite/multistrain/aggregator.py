import logging
import os
import subprocess
import tempfile
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from ..schema.aggregation import AggregationConfiguration, AggregationStrategyEnum, SamplingStrategyEnum


# Weekly baseline hospitalization values per location.
# Keys: epydemix population name (e.g., "United_States_Massachusetts").
BASELINE = {
    "United_States": 889.0833333333334,
    "United_States_Alabama": 21.083333333333332,
    "United_States_Alaska": 3.4166666666666665,
    "United_States_Arizona": 13.0,
    "United_States_Arkansas": 3.25,
    "United_States_California": 81.16666666666667,
    "United_States_Colorado": 3.8333333333333335,
    "United_States_Connecticut": 6.5,
    "United_States_Delaware": 0.5,
    "United_States_District_of_Columbia": 1.0833333333333333,
    "United_States_Florida": 156.5,
    "United_States_Georgia": 18.666666666666668,
    "United_States_Hawaii": 3.5,
    "United_States_Idaho": 1.5833333333333333,
    "United_States_Illinois": 10.333333333333334,
    "United_States_Indiana": 10.083333333333334,
    "United_States_Iowa": 2.0,
    "United_States_Kansas": 3.5833333333333335,
    "United_States_Kentucky": 10.166666666666666,
    "United_States_Louisiana": 23.75,
    "United_States_Maine": 0.6666666666666666,
    "United_States_Maryland": 6.0,
    "United_States_Massachusetts": 9.75,
    "United_States_Michigan": 20.916666666666668,
    "United_States_Minnesota": 2.6666666666666665,
    "United_States_Mississippi": 13.416666666666666,
    "United_States_Missouri": 11.583333333333334,
    "United_States_Montana": 1.5833333333333333,
    "United_States_Nebraska": 2.1666666666666665,
    "United_States_Nevada": 6.416666666666667,
    "United_States_New_Hampshire": 1.8333333333333333,
    "United_States_New_Jersey": 11.833333333333334,
    "United_States_New_Mexico": 1.5833333333333333,
    "United_States_New_York": 17.25,
    "United_States_North_Carolina": 13.166666666666666,
    "United_States_North_Dakota": 1.8333333333333333,
    "United_States_Ohio": 12.0,
    "United_States_Oklahoma": 8.583333333333334,
    "United_States_Oregon": 3.6666666666666665,
    "United_States_Pennsylvania": 105.75,
    "United_States_Puerto_Rico": 89.75,
    "United_States_Rhode_Island": 1.25,
    "United_States_South_Carolina": 8.583333333333334,
    "United_States_South_Dakota": 1.6666666666666667,
    "United_States_Tennessee": 29.166666666666668,
    "United_States_Texas": 100.5,
    "United_States_Utah": 3.1666666666666665,
    "United_States_Vermont": 0.16666666666666666,
    "United_States_Virginia": 10.75,
    "United_States_Washington": 7.75,
    "United_States_West_Virginia": 0.5,
    "United_States_Wisconsin": 5.25,
    "United_States_Wyoming": 0.75,
}


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
        output = subprocess.run(["gcloud","storage","ls",f"{source_location}"], 
                                capture_output=True, text=True)
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
            output = subprocess.run(["gcloud","storage","ls",f"{config.bucket}/{source.experiment}"], 
                                    capture_output=True, text=True)
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
            trajectories[source.strain] = pd.read_csv(target_location)
        else:
            prepend_err = ""
            if source.run_id == "any":
                prepend_err = f"Warning: setting source.run_id to 'any' will\
                fail if more than one matching trajectory file exists.\n"
                logger.warn(prepend_err)
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
                # suffixes=("", f"_{source.strain}"),
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
    aggregated["target_sum"] = aggregated[target_cols].fillna(0).sum(axis=1)

    # Clean up columns
    output_cols = (
        ["sample_id"]
        + [f"sim_id_{source.strain}" for source in config.sources]
        + ["location", "date"]
        + target_cols
        + ["target_sum"]
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
            raise NotImplementedError(f"Invalid aggregation method: {config.aggregation.method}")


################
### BASELINE ###
################


def negbin_baseline_addition(
    aggregated_trajectories: pd.DataFrame,
    config: AggregationConfiguration,
) -> pd.DataFrame:
    """
    Dispatch post-aggregation baseline addition.

    Parameters
    ----------
    aggregated_trajectories: pd.DataFrame
        DataFrame with aggregated trajectories in column "target_total"
    config: AggregationConfiguration
        Config object with settings

    Returns
    -------
    pd.DataFrame
        DataFrame with baseline noise added to aggregated trajectories in column "target_baseline"
    """
    rng = np.random.default_rng()
    