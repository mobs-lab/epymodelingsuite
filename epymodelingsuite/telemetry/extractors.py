"""Data extraction functions for telemetry - extract metrics from epydemix results."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..schema.dispatcher import BuilderOutput


def extract_builder_metadata(
    builder_outputs: "BuilderOutput | list[BuilderOutput]",
    configs: dict[str, Any],
) -> dict[str, Any] | None:
    """Extract metadata from builder outputs and configs for telemetry tracking.

    Parameters
    ----------
    builder_outputs : BuilderOutput | list[BuilderOutput]
        Builder outputs from dispatch_builder
    configs : dict
        Configuration dictionary containing basemodel_config, etc.

    Returns
    -------
    dict | None
        Metadata dict to pass to telemetry.exit_builder() via **kwargs,
        or None if no basemodel_config available
    """
    basemodel_config = configs.get("basemodel_config")
    if not basemodel_config:
        return None

    basemodel = basemodel_config.model

    # Get population names and count
    if isinstance(builder_outputs, list):
        # For sampling/calibration, extract unique population names from models
        populations = list(
            {
                bo.model.population.name if bo.model else bo.calibrator.parameters["epimodel"].population.name
                for bo in builder_outputs
            }
        )
        n_models = len(builder_outputs)
    else:
        # For single model
        populations = [basemodel.population.name]
        n_models = 1

    metadata = {
        "n_models": n_models,
        "populations": populations,
        "start_date": str(basemodel.timespan.start_date),
        "end_date": str(basemodel.timespan.end_date),
        "delta_t": basemodel.timespan.delta_t,
        "random_seed": basemodel.random_seed,
    }

    # Extract age groups from basemodel
    if basemodel.population.age_groups:
        metadata["age_groups"] = basemodel.population.age_groups

    # Extract calibration-specific fields if calibration config is present
    calibration_config = configs.get("calibration_config")
    if calibration_config:
        fitting_window = calibration_config.modelset.calibration.fitting_window
        metadata["fitting_window"] = (
            str(fitting_window.start_date),
            str(fitting_window.end_date),
        )
        # Extract distance function
        metadata["distance_function"] = calibration_config.modelset.calibration.distance_function

    return metadata


def extract_calibration_info(results: Any) -> dict[str, Any]:
    """Extract calibration metrics from epydemix CalibrationResults.

    Parameters
    ----------
    results : Any
        Calibration results object (epydemix.calibration.CalibrationResults)

    Returns
    -------
    dict
        Calibration info dict with particles_accepted, etc.
    """
    calibration_info = {}

    # Extract particles accepted from calibration results
    if hasattr(results, "accepted") and results.accepted is not None:
        calibration_info["particles_accepted"] = len(results.accepted)

    return calibration_info


def extract_projection_info(results: Any, n_trajectories: int) -> dict[str, Any]:
    """Extract projection metrics from epydemix CalibrationResults.

    Parameters
    ----------
    results : Any
        Projection results object (epydemix.calibration.CalibrationResults)
    n_trajectories : int
        Requested number of trajectories

    Returns
    -------
    dict
        Projection info dict with requested/successful/failed trajectories
    """
    projection_info = {
        "requested_trajectories": n_trajectories,
    }

    if results and hasattr(results, "projections") and results.projections:
        # projections is a dict mapping scenario_id to list of projection dicts
        # Count non-empty projection dicts across all scenarios
        successful = 0
        for scenario_id, projections_list in results.projections.items():
            # Count non-empty dicts (empty dict {} means failed projection)
            successful += sum(1 for proj in projections_list if proj)

        projection_info["successful_trajectories"] = successful
        projection_info["failed_trajectories"] = n_trajectories - successful
    else:
        # If no results, all failed
        projection_info["successful_trajectories"] = 0
        projection_info["failed_trajectories"] = n_trajectories

    return projection_info
