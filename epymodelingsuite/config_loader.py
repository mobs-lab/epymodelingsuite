### config_loader.py
# Functions for loading and validating configuration files (defined in YAML format).

import logging

from .schema.basemodel import BasemodelConfig, validate_basemodel
from .schema.calibration import CalibrationConfig, validate_calibration
from .schema.output import OutputConfig, validate_output
from .schema.sampling import SamplingConfig, validate_sampling

__all__ = [
    "load_basemodel_config_from_file",
    "load_calibration_config_from_file",
    "load_output_config_from_file",
    "load_sampling_config_from_file",
]

logger = logging.getLogger(__name__)


def load_basemodel_config_from_file(path: str) -> BasemodelConfig:
    """
    Load model configuration YAML from the given path and validate against the schema.

    Parameters
    ----------
        path: The file path to the YAML configuration file.

    Returns
    -------
        The validated configuration object.
    """
    from pathlib import Path

    import yaml

    with Path(path).open() as f:
        raw = yaml.safe_load(f)

    root = validate_basemodel(raw)
    logger.info("Basemodel configuration loaded successfully.")
    return root


def load_sampling_config_from_file(path: str) -> SamplingConfig:
    """
    Load sampling configuration YAML from the given path and validate against the schema.

    Parameters
    ----------
        path: The file path to the YAML configuration file.

    Returns
    -------
        The validated configuration object.
    """
    from pathlib import Path

    import yaml

    with Path(path).open() as f:
        raw = yaml.safe_load(f)

    root = validate_sampling(raw)
    logger.info("Sampling configuration loaded successfully.")
    return root


def load_calibration_config_from_file(path: str) -> CalibrationConfig:
    """
    Load calibration configuration YAML from the given path and validate against the schema.

    Parameters
    ----------
        path: The file path to the YAML configuration file.

    Returns
    -------
        The validated configuration object.
    """
    from pathlib import Path

    import yaml

    with Path(path).open() as f:
        raw = yaml.safe_load(f)

    root = validate_calibration(raw)
    logger.info("Calibration configuration loaded successfully.")
    return root


def load_output_config_from_file(path: str) -> OutputConfig:
    """
    Load output configuration YAML from the given path and validate against the schema.

    Parameters
    ----------
        path: The file path to the YAML configuration file.

    Returns
    -------
        The validated configuration object.
    """
    from pathlib import Path

    import yaml

    with Path(path).open() as f:
        raw = yaml.safe_load(f)

    root = validate_output(raw)
    logger.info("Output configuration loaded successfully.")
    return root
