import logging
import os
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from ..schema.aggregation import AggregationConfiguration, AggregationStrategyEnum, SamplingStrategyEnum

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


def pull_trajectory_projections(
    config: AggregationConfiguration, tempdir: TemporaryDirectory
) -> dict[str, pd.DataFrame]:
    """
    Pull trajectory projection files from Google Cloud Storage.
    Requires cloud authorization and gsutil.

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
        # TODO: use source.run_id
        source_location = f"{config.bucket}/{source.experiment}/*/outputs/*/{source.trajectory_file}"
        target_location = f"{tempdir}/{source.strain}"

        command = f"gsutil cp -r '{source_location}' '{target_location}'"
        exit_code = os.system(command)

        if exit_code == 0:
            logger.info(f"Downloaded: {source.strain}")
            # Load the downloaded data
            trajectory_file = f"{target_location}/{source.strain}/{source.trajectory_file}"
            trajectories[source.strain] = pd.read_csv(trajectory_file)
        else:
            raise ValueError(f"Failed to download: {source.experiment}\nExit code: {exit_code}")

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
        )[["target", "date", "location", "sim_id"]]
        formatted_strains[source.strain] = formatted

    # Start with first strain
    first_source = config.sources[0]
    combined = mapping.merge(
        formatted_strains[first_source.strain], left_on=first_source.strain, right_on="sim_id"
    ).rename(columns={"target": f"target_{first_source.strain}"})

    # Merge remaining strains
    for source in config.sources[1:]:
        combined = combined.merge(
            formatted_strains[source.strain],
            left_on=[source.strain, "location", "date"],
            right_on=["sim_id", "location", "date"],
            suffixes=("", f"_{source.strain}"),
            how="outer",
        ).rename(columns={"target": f"target_{source.strain}"})

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
    rng = np.random.default_rng(seed=config.seed)
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
        mapping_data[source.strain] = sampled_ids
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
    aggregated["target_total"] = aggregated[target_cols].fillna(0).sum(axis=1)

    # Clean up columns
    output_cols = (
        ["sample_id"]
        + [source.strain for source in config.sources]
        + ["location", "date"]
        + target_cols
        + ["target_total"]
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
    validate_strains(strains, config)

    match config.aggregate_method:
        case AggregationStrategyEnum.sum:
            return _aggregate_sum(strains, mapping_df, config)
        case _:
            raise NotImplementedError(f"Invalid aggregation method: {config.aggregation.method}")
