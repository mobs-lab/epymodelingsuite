import logging
from datetime import date, timedelta

from ..utils.common import parse_transition_name, strip_agegroup_suffix, to_set
from .basemodel import BasemodelConfig
from .calibration import CalibrationConfig, CalibrationConfiguration
from .output import OutputConfig
from .sampling import SamplingConfig, SamplingConfiguration

logger = logging.getLogger(__name__)


def _ensure_parameters_present(base_params: set, modelset_params: set) -> None:
    """
    Warn if modelset parameters are not declared in the base model.

    Modelset parameters may be referenced only inside user-defined functions
    (custom distance functions, post-hoc transformations, calculated parameter
    expressions) and thus need not appear in `basemodel.parameters`. We emit a
    warning rather than raising so legitimate usages are not blocked.
    typos are still surfaced.

    Parameters
    ----------
    base_params : set
        Parameter names available in the base model.
    modelset_params : set
        Parameter names referenced by the modelset.
    """
    missing = modelset_params - base_params
    if missing:
        logger.warning(
            f"Parameters in modelset not defined in base model: {sorted(missing)}. "
            "If these are referenced only inside user-defined functions this is fine; "
            "otherwise check for a typo."
        )


def _ensure_compartments_valid(base_compartments: set, sampling: SamplingConfiguration | None) -> None:
    """
    Validate that sampling compartments match the base model compartments.

    Parameters
    ----------
    base_compartments : set
        Compartment identifiers defined in the base model.
    sampling : SamplingConfiguration or None
        Sampling section of the modelset, if present.

    Raises
    ------
    ValueError
        Raised when sampling references unknown compartments.
    """
    if not sampling or not sampling.compartments:
        return
    missing = set(sampling.compartments.keys()) - base_compartments
    if missing:
        err_msg = f"Compartments in modelset not defined in base model: {sorted(missing)}"
        raise ValueError(err_msg)


def _ensure_populations_valid(base_population_name: str | None, modelset_populations: set) -> None:
    """
    Validate that modelset population names are compatible with the base model.

    Parameters
    ----------
    base_population_name : str or None
        Name of the population defined in the base model.
    modelset_populations : set
        Population names referenced by the modelset.

    Raises
    ------
    ValueError
        Raised when modelset populations do not match the base population or "all".
    """
    if not base_population_name or not modelset_populations:
        return
    invalid = modelset_populations - {base_population_name, "all"}
    if invalid:
        err_msg = f"Populations in modelset not matching base model: {sorted(invalid)}"
        raise ValueError(err_msg)


def _ensure_transitions_valid(base_transitions: set, calibration: CalibrationConfiguration | None) -> None:
    """
    Warn if calibration comparison transitions are not declared in the base model.

    Comparison transitions may be UDF-computed (e.g. derived signals like `ed_signal`)
    rather than direct basemodel transitions. We warn rather than raise so legitimate
    UDF transitions are not blocked, while typos are still surfaced.

    Parameters
    ----------
    base_transitions : set
        Transition identifiers defined in the base model.
    calibration : CalibrationConfiguration or None
        Calibration section of the modelset, if present.
    """
    if not calibration:
        return
    for comparison in calibration.comparison or []:
        missing = set(comparison.simulation) - base_transitions
        if missing:
            logger.warning(
                f"Transitions in calibration comparison not defined in base model: {sorted(missing)}. "
                "If these are UDF-computed transitions this is fine; otherwise check for a typo."
            )


def _validate_compartment_list(names: list[str], base_compartments: set, context: str) -> None:
    """
    Validate compartment names exist in basemodel.

    Parameters
    ----------
    names : list[str]
        List of compartment names to validate.
    base_compartments : set
        Compartment identifiers defined in base model.
    context : str
        Description of where these compartments are referenced (for error messages).

    Raises
    ------
    ValueError
        If any compartment names are not defined in base model.
    """
    invalid = {strip_agegroup_suffix(name) for name in names} - base_compartments
    if invalid:
        err_msg = f"Compartments in {context} not defined in basemodel: {sorted(invalid)}"
        raise ValueError(err_msg)


def _validate_transition_list(
    names: list[str], base_transitions: set, context: str, *, warn_only: bool = False
) -> None:
    """
    Validate transition names exist in basemodel.

    Parameters
    ----------
    names : list[str]
        List of transition names to validate.
    base_transitions : set
        Transition identifiers defined in base model (format: {source}_to_{target}).
    context : str
        Description of where these transitions are referenced (for error messages).
    warn_only : bool, optional
        If True, log a warning instead of raising an error for invalid transitions.
        Defaults to False (raise error).

    Raises
    ------
    ValueError
        If any transition names are not defined in base model (unless warn_only=True).
    """
    invalid = set()
    for name in names:
        try:
            source, target = parse_transition_name(name)
            transition_id = f"{source}_to_{target}"
            if transition_id not in base_transitions:
                invalid.add(transition_id)
        except ValueError:
            invalid.add(name)

    if invalid:
        msg = f"Transitions in {context} not defined in basemodel: {sorted(invalid)}"
        if warn_only:
            logger.warning(msg)
        else:
            raise ValueError(msg)


def _ensure_output_references_valid(
    base_compartments: set[str], base_transitions: set[str], output_config: OutputConfig
) -> None:
    """
    Validate that output config references exist in basemodel.

    Compartment references are strictly validated (errors raised).
    Transition references are loosely validated (warnings only) to allow aggregated transitions.

    Parameters
    ----------
    base_compartments : set[str]
        Compartment identifiers defined in base model.
    base_transitions : set[str]
        Transition identifiers defined in base model (format: {source}_to_{target}).
    output_config : OutputConfig
        Output configuration to validate.

    Raises
    ------
    ValueError
        If any referenced compartments are not defined in base model.
    """
    output = output_config.output

    # Validate quantiles section
    if output.quantiles is not None:
        quantiles = output.quantiles
        # Validate compartments if it's a list (skip if boolean)
        if isinstance(quantiles.compartments, list):
            _validate_compartment_list(quantiles.compartments, base_compartments, "quantiles.compartments")
        # Validate transitions if it's a list (skip if boolean) - warn only for aggregated transitions
        if isinstance(quantiles.transitions, list):
            _validate_transition_list(quantiles.transitions, base_transitions, "quantiles.transitions", warn_only=True)

    # Validate trajectories section
    if output.trajectories is not None:
        trajectories = output.trajectories
        # Validate compartments if it's a list (skip if boolean)
        if isinstance(trajectories.compartments, list):
            _validate_compartment_list(trajectories.compartments, base_compartments, "trajectories.compartments")
        # Validate transitions if it's a list (skip if boolean) - warn only for aggregated transitions
        if isinstance(trajectories.transitions, list):
            _validate_transition_list(
                trajectories.transitions, base_transitions, "trajectories.transitions", warn_only=True
            )


def _warn_mismatched_observed_data_paths(
    calibration: CalibrationConfiguration | None, output_config: OutputConfig
) -> None:
    """
    Warn if observed data paths differ between calibration and output configs.

    Parameters
    ----------
    calibration : CalibrationConfiguration or None
        Calibration configuration, if present.
    output_config : OutputConfig
        Output configuration to check.
    """
    # Only check if we have both calibration config and FluSight rate trends output
    if calibration is None:
        return

    output = output_config.output
    if output.flusight_format is None or output.flusight_format.rate_trends_source is None:
        return

    # Get surveillance source configuration
    if not output.options or not output.options.surveillance:
        logger.warning(
            "FluSight rate_trends_source specified but no surveillance sources defined in output.options.surveillance"
        )
        return

    source_name = output.flusight_format.rate_trends_source
    if source_name not in output.options.surveillance:
        logger.warning("FluSight rate_trends_source '%s' not found in output.options.surveillance", source_name)
        return

    calibration_path = calibration.observed_data_path
    output_path = output.options.surveillance[source_name].data_path

    if calibration_path != output_path:
        logger.warning(
            "Observed data paths differ between configs: "
            "calibration='%s', "
            "output.flusight_format.rate_trends_source ('%s')='%s'. "
            "This may lead to inconsistent results if the files contain different data.",
            calibration_path,
            source_name,
            output_path,
        )


def _compare_prop_ed_window_against_calibration_window(
    calibration: CalibrationConfiguration | None, output_config: OutputConfig
) -> None:
    """
    Ensure rescaling factor fitting window is within calibration fitting window.

    Parameters
    ----------
    calibration : CalibrationConfiguration or None
        Calibration configuration, if present.
    output_config : OutputConfig
        Output configuration to check.
    """
    # Only check if we have both calibration config and FluSight prop ED output
    if calibration is None:
        return

    output = output_config.output
    if output.flusight_format is None or output.flusight_format.prop_ed is None:
        return

    # Collect calibration window
    calibration_end = calibration.fitting_window.end_date
    calibration_start = calibration.fitting_window.start_date

    # calibration_window strategy
    if output.flusight_format.prop_ed.strategy == "calibration_window":
        rescale_start = calibration_end - timedelta(weeks=output.flusight_format.prop_ed.num_fit_weeks - 1)

        # Validate
        if rescale_start < calibration_start:
            msg = (
                f"Received flusight_format.prop_ed.num_fit_weeks: {output.flusight_format.prop_ed.num_fit_weeks}"
                f"Resulting in a rescaling factor fit start: {rescale_start}"
                f"Rescaling factor fit start cannot be earlier than calibration.fitting_window.start_date: {calibration_start}"
            )
            raise ValueError(msg)

    # surveillance_window strategy
    if output.flusight_format.prop_ed.strategy == "surveillance_window":
        rescale_start = output.flusight_format.prop_ed.fit_start
        rescale_end = output.flusight_format.prop_ed.fit_end

        # Validate
        if (
            (rescale_start < calibration_start)
            or (rescale_start > calibration_end)
            or (rescale_end > calibration_end)
            or (rescale_end < calibration_start)
        ):
            msg = (
                f"Received flusight_format.prop_ed.fit_start: {rescale_start} and fit_end: {rescale_end}"
                f"Received calibration.fitting_window.start_date: {calibration_start} and end_date: {calibration_end}"
                "Prop ED rescaling factor must not extend beyond the calibration fitting window."
            )
            raise ValueError(msg)


def _ensure_fitting_window_within_timespan(
    basemodel: "BasemodelConfig",
    calibration: CalibrationConfiguration | None,
) -> None:
    """
    Validate fitting window is contained within simulation timespan.

    Parameters
    ----------
    basemodel : BasemodelConfig
        Base model configuration.
    calibration : CalibrationConfiguration or None
        Calibration configuration, if present.

    Raises
    ------
    ValueError
        Raised when fitting window extends beyond simulation timespan.
    """
    if calibration is None:
        return

    fitting_window = calibration.fitting_window
    timespan = basemodel.model.timespan

    # Use computed fields that handle both date and epiweek specifications
    fit_start = fitting_window.epiweek_start_date
    fit_end = fitting_window.epiweek_end_date

    # Validate fitting_window end <= timespan.end_date
    if fit_end > timespan.end_date:
        msg = (
            f"Fitting window end_date ({fit_end}) exceeds "
            f"simulation timespan end_date ({timespan.end_date}). "
            "The fitting window must be contained within the simulation timespan."
        )
        raise ValueError(msg)

    # Validate fitting_window start >= timespan.start_date
    # (only when timespan.start_date is a concrete date)
    if isinstance(timespan.start_date, date) and fit_start < timespan.start_date:
        msg = (
            f"Fitting window start_date ({fit_start}) is before "
            f"simulation timespan start_date ({timespan.start_date}). "
            "The fitting window must be contained within the simulation timespan."
        )
        raise ValueError(msg)


def validate_cross_config_consistency(
    base_config: BasemodelConfig,
    modelset_config: SamplingConfig | CalibrationConfig,
    output_config: OutputConfig | None = None,
) -> None:
    """
    Cross-validate modelset and output configs against the base model configuration.

    Ensures that all references (parameters, compartments, transitions) in
    modelset and output configs are defined in the basemodel config.

    Parameters
    ----------
    base_config : BasemodelConfig
        Validated base model configuration.
    modelset_config : SamplingConfig | CalibrationConfig
        Validated modelset configuration (sampling or calibration).
    output_config : OutputConfig | None, optional
        Output configuration to validate.

    Raises
    ------
    ValueError
        Raised when required references are missing or inconsistent between configs.
    """
    basemodel = base_config.model
    modelset = getattr(modelset_config, "modelset", None)

    # Ensure both base model and modelset are provided
    if basemodel is None or modelset is None:
        err_msg = "Both base model and modelset must be defined before running consistency checks."
        raise ValueError(err_msg)

    # Modelset must contain either sampling or calibration section
    sampling = getattr(modelset, "sampling", None)
    calibration = getattr(modelset, "calibration", None)

    # CalibrationConfig must have calibration section
    if isinstance(modelset_config, CalibrationConfig) and not calibration:
        err_msg = "Calibration modelset must provide a 'calibration' section."
        raise ValueError(err_msg)

    # SamplingConfig supports population-only mode (sampling=None), skip remaining validation
    # Example YAML:
    #   modelset:
    #     population_names: ["US-CA", "US-TX"]
    if isinstance(modelset_config, SamplingConfig) and sampling is None:
        logger.info(
            "Sampling modelset received without sampled variables (only populations). Ensure your modelset does not contain any 'sampled' keywords"
        )
        return

    # Require either sampling or calibration for remaining validation
    if sampling is None and calibration is None:
        err_msg = "Modelset must provide a 'sampling' or 'calibration' section."
        raise ValueError(err_msg)

    # Parameter consistency checks
    # - Get sets of parameters for basemodel and modelset
    # - Ensure all modelset parameters exist in basemodel
    base_params = set((basemodel.parameters or {}).keys())
    modelset_params = (
        set((sampling.parameters or {}).keys()) if sampling else set((calibration.parameters or {}).keys())
    )
    _ensure_parameters_present(base_params, modelset_params)

    # Compartment consistency checks
    # - Get set of compartments for basemodel
    # - Ensure all sampling compartments exist in basemodel
    base_compartments = {comp.id for comp in basemodel.compartments or []}
    _ensure_compartments_valid(base_compartments, sampling)

    # Population consistency checks
    # - Get basemodel population name and modelset population(s)
    # - Ensure modelset populations match basemodel population or "all"
    base_population_name = getattr(getattr(basemodel, "population", None), "name", None)
    modelset_populations = to_set(getattr(modelset, "population_names", None))
    _ensure_populations_valid(base_population_name, modelset_populations)

    # Transitions consistency checks (for calibration comparison)
    # - Get set of transitions for basemodel
    # - Ensure all transitions used in calibration comparison exist in basemodel
    base_transitions = {f"{t.source}_to_{t.target}_total" for t in basemodel.transitions or []}
    _ensure_transitions_valid(base_transitions, calibration)

    # Fitting window consistency checks
    # - Ensure fitting window is within simulation timespan
    _ensure_fitting_window_within_timespan(base_config, calibration)

    # Output config consistency checks
    # - Validate output config references if provided
    if output_config is not None:
        base_compartments_output = {comp.id for comp in basemodel.compartments or []}
        base_transitions_output = {f"{t.source}_to_{t.target}" for t in basemodel.transitions or []}
        _ensure_output_references_valid(base_compartments_output, base_transitions_output, output_config)

        # Warn if observed data paths differ between calibration and output configs
        _warn_mismatched_observed_data_paths(calibration, output_config)

        # Ensure prop ed fitting window contained by calibration fitting window
        _compare_prop_ed_window_against_calibration_window(calibration, output_config)

    logger.info("Config consistency validated successfully.")
