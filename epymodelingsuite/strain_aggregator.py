import logging
import os
from tempfile import TemporaryDirectory
import pandas as pd

logger = logging.getLogger(__name__)

from .schema.aggregation import AggregationConfig, AggregationConfiguration, AggregationStrategyEnum, SamplingStrategyEnum
    
    
def pull_trajectory_projections(config: AggregationConfiguration, tempdir: TemporaryDirectory) -> dict[str, pd.DataFrame]
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
        source_location = (
            f"{config.bucket}/{source.experiment}/*/"
            f"outputs/*/{source.trajectory_file}"
        )
        target_location = f"{tempdir}/{source.subtype}"

        command = f"gsutil cp -r '{source_location}' '{target_location}'"
        exit_code = os.system(command)

        if exit_code == 0:
            logger.info(f"Downloaded: {source.subtype}")
            # Load the downloaded data
            trajectory_file = f"{target_location}/{source.subtype}/{source.trajectory_file}"
            trajectories[source.subtype] = pd.read_csv(trajectory_file)
        else:
            raise ValueError(f"Failed to download: {source.experiment}\nExit code: {exit_code}")

    return trajectories


def random_sample_mapping(
    strains: dict[str, pd.DataFrame],
    config: AggregationConfiguration,
    n_samples: int = 1000,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    """
    
def dispatch_strain_sampler(
    strains: dict[str, pd.DataFrame],
    config: AggregationConfiguration,
) -> pd.DataFrame:
    """
    """
    match config.sampling.method:
        case SamplingStrategyEnum.random:
            return random_sample_mapping(strains, config)
        case _:
            raise NotImplementedError(f"Invalid sampling method: {config.sampling.method}")
    

def random_sample_and_sum(
    strain_trajectories: Dict[str, pd.DataFrame],
    n_samples: int = 1000,
    seed: Optional[int] = None
) -> tuple:
    """
    Random sampling: independently shuffle strain sim_ids and sum trajectories.

    Parameters
    ----------
    strain_trajectories : dict
        Dictionary mapping strain names to their trajectory DataFrames
        Example: {'h1': df_h1, 'h3': df_h3, 'b': df_b}
        Or: {'a': df_a, 'b': df_b}
    n_samples : int
        Number of samples to generate (default: 1000)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    tuple
        (combined_df, mapping_df)
        - combined_df: DataFrame with summed trajectories
        - mapping_df: DataFrame mapping sample_id to strain sim_ids
    """
    if seed is not None:
        np.random.seed(seed)

    strain_names = list(strain_trajectories.keys())

    # Sample sim_ids for each strain
    sampled_ids = {}
    for strain in strain_names:
        sim_ids = strain_trajectories[strain]['sim_id'].unique()
        sampled_ids[strain] = np.random.choice(sim_ids, size=n_samples, replace=False)

    # Create mapping DataFrame
    mapping_data = {'sample_id': range(n_samples)}
    for strain in strain_names:
        mapping_data[f'{strain}_sim_id'] = sampled_ids[strain]
    mapping_df = pd.DataFrame(mapping_data)

    # Start with first strain
    first_strain = strain_names[0]
    combined = mapping_df.merge(
        strain_trajectories[first_strain],
        left_on=f'{first_strain}_sim_id',
        right_on='sim_id'
    ).rename(columns={'hospitalizations': f'hospitalizations_{first_strain}'})

    # Merge remaining strains
    for strain in strain_names[1:]:
        combined = combined.merge(
            strain_trajectories[strain],
            left_on=[f'{strain}_sim_id', 'population', 'date'],
            right_on=['sim_id', 'population', 'date'],
            suffixes=('', f'_{strain}'),
            how='outer'
        )
        combined = combined.rename(columns={'hospitalizations': f'hospitalizations_{strain}'})

    # Sum hospitalizations from all strains
    hosp_cols = [f'hospitalizations_{strain}' for strain in strain_names]
    combined['hospitalizations_total'] = combined[hosp_cols].fillna(0).sum(axis=1)

    # Clean up columns
    output_cols = ['sample_id'] + [f'{strain}_sim_id' for strain in strain_names] + \
                  ['population', 'date'] + hosp_cols + ['hospitalizations_total']

    # Filter to only columns that exist
    output_cols = [col for col in output_cols if col in combined.columns]
    combined = combined[output_cols]

    return combined, mapping_df




def dispatch_aggregator(
    source_results: dict[str, list[CalibrationOutput]], config: AggregationConfig
) -> list[CalibrationOutput]:
    """
    Aggregate projection trajectories from multiple experiments.

    Produces N aggregated outputs where N is the number of locations/tasks in source experiments.
    Each aggregated output combines the corresponding results from all sources:
    aggregated[i] = aggregate(source1[i].results, source2[i].results, ...) for location i

    Parameters
    ----------
    source_results : dict[str, list[CalibrationOutput]]
        Dictionary mapping exp_id to list of CalibrationOutput for that experiment.
        Keys are exp_ids from config.sources, values are CalibrationOutput objects from runner-artifacts.
        All sources must have the same number of outputs (one per population/location) and must be aligned.
        This structure preserves source grouping needed for weighted/bootstrap/correlated methods.
    config : AggregationConfig
        Aggregation configuration (method, compartments, transitions, weights, etc.)

    Returns
    -------
    list[CalibrationOutput]
        List of aggregated CalibrationOutput objects, one per location.
        Length matches the input (e.g., 10 outputs for 10 locations).

    Examples
    --------
    >>> source_results = {
    ...     "h1-calibration": [output0, output1, ...],  # 10 CalibrationOutput (one per location)
    ...     "h3-calibration": [output0, output1, ...]   # 10 CalibrationOutput (one per location)
    ... }
    >>> aggregated = dispatch_aggregator(source_results, config)
    >>> len(aggregated)  # 10 aggregated CalibrationOutput
    10
    >>> # aggregated[0] contains h1[0].results + h3[0].results (for location 0)
    >>> # aggregated[1] contains h1[1].results + h3[1].results (for location 1)
    """
    aggregation = config.aggregation

    # Validate that all exp_ids in config have corresponding results
    for source in aggregation.sources:
        if source.exp_id not in source_results:
            raise ValueError(f"Missing results for exp_id: {source.exp_id}")

    # Validate that results sets are of equal length
    if not len(set(len(l) for l in source_results.values())) == 1:
        msg = f"All sources must have the same number of outputs (one per population). \
        Received sources {list(source_results.keys())} \
        with corresponding num outputs {[len(l) for l in source_results.values()]}"
        raise ValueError(msg)

    # Validate that results sets are aligned by population
    results_zip = zip(*source_results.values(), strict=True)
    idx = 0
    for tup in results_zip:
        if not len(set(obj.population for obj in tup)) == 1:
            msg = f"Source results must be aligned such that results objects at equivalent \
            indices all share the same population/location. \
            Encountered differing populations at index {idx}: {[obj.population for obj in tup]}."
            raise ValueError(msg)
        idx += 1

    # Dispatch to method-specific aggregator
    if aggregation.method == AggregationStrategyEnum.sum:
        return aggregate_sum(source_results, aggregation)
    if aggregation.method == AggregationStrategyEnum.weighted_sum:
        return aggregate_weighted_sum(source_results, aggregation)
    if aggregation.method == AggregationStrategyEnum.bootstrap:
        return aggregate_bootstrap(source_results, aggregation)
    if aggregation.method == AggregationStrategyEnum.correlated:
        return aggregate_correlated(source_results, aggregation)
    raise ValueError(f"Unknown aggregation method: {aggregation.method}")


def aggregate_sum(
    source_results: dict[str, list[CalibrationOutput]], config: AggregationConfiguration
) -> list[CalibrationOutput]:
    """
    Simple element-wise summation of trajectories.

    Parameters
    ----------
    source_results : dict[str, list[CalibrationOutput]]
        Dictionary mapping exp_id to list of CalibrationOutput
    config : AggregationConfiguration
        Aggregation configuration

    Returns
    -------
    list[CalibrationOutput]
        List of CalibrationOutput with summed projections (one per location)

    Notes
    -----
    Workflow:
        1. Validate all sources have same number of outputs (N locations)
        2. For each location i in 0..N-1:
            a. Extract .results from each source's CalibrationOutput[i]
            b. Extract trajectories via get_projection_trajectories()
            c. Identify variables to aggregate based on config
            d. Sum arrays element-wise across all sources
            e. Preserve metadata (date, random_state) from first source
            f. Package as CalibrationResults with projections dict
            g. Wrap aggregated CalibrationResults into CalibrationOutput
        3. Return list of N aggregated CalibrationOutput objects
    """
    return []


def aggregate_weighted_sum(
    source_results: dict[str, list[CalibrationOutput]], config: AggregationConfiguration
) -> list[CalibrationOutput]:
    """"""
    raise NotImplementedError("Weighted sum aggregation not yet implemented")


def aggregate_bootstrap(
    source_results: dict[str, list[CalibrationOutput]], config: AggregationConfiguration
) -> list[CalibrationOutput]:
    """"""
    raise NotImplementedError("Bootstrap aggregation not yet implemented")


def aggregate_correlated(
    source_results: dict[str, list[CalibrationOutput]], config: AggregationConfiguration
) -> list[CalibrationOutput]:
    """"""
    raise NotImplementedError("Correlated aggregation not yet implemented")
