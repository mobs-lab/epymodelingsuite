import logging
from collections.abc import Iterable

from .calibration_validator import CalibrationConfig, CalibrationConfiguration
from .basemodel_validator import BasemodelConfig
from .sampling_validator import SamplingConfig, SamplingConfiguration

logger = logging.getLogger(__name__)


def _to_set(values: Iterable | None) -> set:
    """
    Normalize an optional iterable into a set.

    Parameters
    ----------
    values : Iterable or None
        Input iterable (or ``None``) to convert.

    Returns
    -------
    set
        Set containing the iterable values, or an empty set when ``None``.
    """
    return set(values or [])


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


def validate_modelset_consistency(base_config: BasemodelConfig, modelset_config: SamplingConfig | CalibrationConfig) -> None:
    """
    Cross-validate a modelset against the base model configuration.

    Parameters
    ----------
    base_config : RootConfig
        Validated base model configuration.
    modelset_config : SamplingConfig or CalibrationConfig
        Validated modelset configuration (sampling or calibration).

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
    modelset_populations = _to_set(getattr(modelset, "population_names", None))
    _ensure_populations_valid(base_population_name, modelset_populations)

    # Transitions consistency checks (for calibration comparison)
    # - Get set of transitions for basemodel
    # - Ensure all transitions used in calibration comparison exist in basemodel
    base_transitions = {transition.id for transition in basemodel.transitions or []}
    _ensure_transitions_valid(base_transitions, calibration)

    logger.info("Modelset consistency validated successfully.")
