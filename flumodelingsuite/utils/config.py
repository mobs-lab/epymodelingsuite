"""Configuration file utilities."""

from pathlib import Path


def identify_config_type(file_path: str) -> str | None:
    """
    Identify the config type of a single YAML file by checking its structure.

    Checks the YAML structure to determine config type:
    - Has 'modelset.calibration' -> 'calibration'
    - Has 'modelset.sampling' -> 'sampling'
    - Has 'model' (without 'modelset') -> 'basemodel'
    - Has 'outputs' -> 'output'
    - Otherwise -> None

    Parameters
    ----------
    file_path : str
        Path to the YAML config file (str or Path)

    Returns
    -------
    str | None
        Config type string: 'basemodel', 'sampling', 'calibration', 'output', or None
        Example: 'basemodel', 'sampling', 'output', None

    Raises
    ------
    FileNotFoundError
        If the file doesn't exist
    ValueError
        If the file is not a YAML file (.yml or .yaml)
    """
    import yaml

    path = Path(file_path)

    if not path.exists():
        msg = f"Config file not found: {file_path}"
        raise FileNotFoundError(msg)

    if path.suffix.lower() not in [".yml", ".yaml"]:
        msg = f"File must be a YAML file (.yml or .yaml): {file_path}"
        raise ValueError(msg)

    # Load the YAML file
    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        return None

    # Check for config type based on structure
    # Calibration: has modelset.calibration
    if "modelset" in data and isinstance(data["modelset"], dict):
        if "calibration" in data["modelset"]:
            return "calibration"
        if "sampling" in data["modelset"]:
            return "sampling"

    # Basemodel: has 'model' key (without modelset)
    if "model" in data:
        return "basemodel"

    # Output: has 'outputs' key
    if "outputs" in data:
        return "output"

    return None
