import logging

from ..schema.aggregation import AggregationConfig, AggregationConfiguration, AggregationStrategyEnum
from ..schema.dispatcher import CalibrationOutput

logger = logging.getLogger(__name__)


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
