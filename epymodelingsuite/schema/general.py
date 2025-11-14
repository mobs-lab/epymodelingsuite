import logging

from ..utils.common import parse_transition_name, strip_agegroup_suffix, to_set
from .basemodel import BasemodelConfig
from .calibration import CalibrationConfig, CalibrationConfiguration
from .output import OutputConfig
from .sampling import SamplingConfig, SamplingConfiguration

logger = logging.getLogger(__name__)


def _ensure_parameters_present(base_params: set, modelset_params: set) -> None:
    """
    Validate that all modelset parameters exist in the base model.

    Parameters
    ----------
    base_params : set
        Parameter names available in the base model.
    modelset_params : set
        Parameter names referenced by the modelset.

    Raises
    ------
    ValueError
        Raised when at least one modelset parameter is missing from the base model.
    """
    missing = modelset_params - base_params
    if missing:
        err_msg = f"Parameters in modelset not defined in base model: {sorted(missing)}"
        raise ValueError(err_msg)


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
    Validate that calibration comparisons reference known transitions.

    Parameters
    ----------
    base_transitions : set
        Transition identifiers defined in the base model.
    calibration : CalibrationConfiguration or None
        Calibration section of the modelset, if present.

    Raises
    ------
    ValueError
        Raised when calibration comparisons contain unknown transitions.
    """
    if not calibration:
        return
    for comparison in calibration.comparison or []:
        missing = set(comparison.simulation) - base_transitions
        if missing:
            err_msg = f"Transitions in calibration comparison not defined in base model: {sorted(missing)}"
            raise ValueError(err_msg)


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


def _validate_transition_list(names: list[str], base_transitions: set, context: str) -> None:
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

    Raises
    ------
    ValueError
        If any transition names are not defined in base model.
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
        err_msg = f"Transitions in {context} not defined in basemodel: {sorted(invalid)}"
        raise ValueError(err_msg)


def _ensure_output_references_valid(
    base_compartments: set[str], base_transitions: set[str], output_config: OutputConfig
) -> None:
    """
    Validate that output config references exist in basemodel.

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
        If any referenced compartments or transitions are not defined in base model.
    """
    output = output_config.output

    # Validate quantiles section
    if output.quantiles is not None:
        quantiles = output.quantiles
        # Validate compartments if it's a list (skip if boolean)
        if isinstance(quantiles.compartments, list):
            _validate_compartment_list(quantiles.compartments, base_compartments, "quantiles.compartments")
        # Validate transitions if it's a list (skip if boolean)
        if isinstance(quantiles.transitions, list):
            _validate_transition_list(quantiles.transitions, base_transitions, "quantiles.transitions")

    # Validate trajectories section
    if output.trajectories is not None:
        trajectories = output.trajectories
        # Validate compartments if it's a list (skip if boolean)
        if isinstance(trajectories.compartments, list):
            _validate_compartment_list(trajectories.compartments, base_compartments, "trajectories.compartments")
        # Validate transitions if it's a list (skip if boolean)
        if isinstance(trajectories.transitions, list):
            _validate_transition_list(trajectories.transitions, base_transitions, "trajectories.transitions")


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
    if output.flusight_format is None or output.flusight_format.rate_trends is None:
        return

    calibration_path = calibration.observed_data_path
    output_path = output.flusight_format.rate_trends.observed_data_path

    if calibration_path != output_path:
        logger.warning(
            "Observed data paths differ between configs: "
            "calibration='%s', "
            "output.flusight_format.rate_trends='%s'. "
            "This may lead to inconsistent results if the files contain different data.",
            calibration_path,
            output_path,
        )


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
    if not sampling and not calibration:
        err_msg = "Modelset must provide a 'sampling' or 'calibration' section."
        raise ValueError(err_msg)

    # End validation if no variables are sampled (modelset is used only for population)
    if sampling == "populations":
        logger.info(
            "Sampling modelset received without sampled variables (only populations). Ensure your modelset does not contain any 'sampled' keywords"
        )
        return

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

    # Output config consistency checks
    # - Validate output config references if provided
    if output_config is not None:
        base_compartments_output = {comp.id for comp in basemodel.compartments or []}
        base_transitions_output = {f"{t.source}_to_{t.target}" for t in basemodel.transitions or []}
        _ensure_output_references_valid(base_compartments_output, base_transitions_output, output_config)

        # Warn if observed data paths differ between calibration and output configs
        _warn_mismatched_observed_data_paths(calibration, output_config)

    logger.info("Config consistency validated successfully.")
