"""Sample generation functionality for flu modeling suite.

This module handles the generation of parameter and compartment samples
based on validated sampling configurations. The refactored approach:

1. Create parameter specs for LHS/other samplers (excluding grid parameters)
2. Generate samples with the primary sampler
3. Apply grid sampling on top if grid sampler exists
"""

import itertools
import logging
from typing import Any

from .config_loader import load_sampling_config_from_file
from .sampler import generate_parameter_samples
from .sampling_validator import Sampler, SamplingConfiguration

logger = logging.getLogger(__name__)


def _build_start_date_spec(sampling_config: SamplingConfiguration) -> dict[str, Any]:
    """Build parameter spec for start_date. Start date is a special kind of parameter."""
    if not sampling_config.modelset.sampling.start_date:
        msg = "start_date parameter specified but not configured"
        raise ValueError(msg)

    start_date_config = sampling_config.modelset.sampling.start_date
    return {
        "name": "start_date",
        "type": "date_uniform",
        "reference_date": start_date_config.reference_date,
        "args": start_date_config.distribution.args,
    }


def _build_regular_param_spec(param_name: str, sampling_config: SamplingConfiguration) -> dict[str, Any]:
    """Build parameter spec for regular parameters."""
    if param_name not in sampling_config.modelset.sampling.parameters:
        msg = f"Parameter {param_name} not found in configuration"
        raise ValueError(msg)

    param_config = sampling_config.modelset.sampling.parameters[param_name]
    if not param_config.distribution:
        msg = f"Sampling requires distribution for parameter {param_name}"
        raise ValueError(msg)

    dist = param_config.distribution
    if dist.name == "uniform":
        return {"name": param_name, "type": "uniform", "args": dist.args}
    if dist.name == "randint":
        return {"name": param_name, "type": "randint", "args": dist.args}
    msg = f"Unsupported distribution {dist.name} for parameter {param_name}"
    raise ValueError(msg)


def build_all_parameter_specs(
    sampler: Sampler, sampling_config: SamplingConfiguration, exclude_params: set[str] | None = None
) -> list[dict[str, Any]]:
    """
    Build parameter specifications for the sampler from configuration.

    Parameters
    ----------
    sampler : Sampler
        The sampler configuration object
    sampling_config : SamplingConfiguration
        The validated sampling configuration
    exclude_params : set[str] | None
        Set of parameter names to exclude (e.g., grid parameters)

    Returns
    -------
    List[Dict[str, Any]]
        List of parameter specifications for generate_parameter_samples
    """
    if exclude_params is None:
        exclude_params = set()

    param_specs = []

    for param_name in sampler.parameters:
        if param_name in exclude_params:
            continue

        if param_name == "start_date":
            param_specs.append(_build_start_date_spec(sampling_config))
        else:
            param_specs.append(_build_regular_param_spec(param_name, sampling_config))

    return param_specs


def build_grid_combinations(sampling_config: SamplingConfiguration, grid_sampler: Sampler) -> list[dict[str, Any]]:
    """Generate all grid parameter combinations."""
    all_param_combinations = []

    for param_name in grid_sampler.parameters:
        if param_name in sampling_config.modelset.sampling.parameters:
            param_spec = sampling_config.modelset.sampling.parameters[param_name]
            if param_spec.values:
                all_param_combinations.append([(param_name, val) for val in param_spec.values])
            else:
                msg = f"Grid sampling requires 'values' for parameter {param_name}"
                raise ValueError(msg)
        else:
            msg = f"Parameter {param_name} not found in configuration"
            raise ValueError(msg)

    if not all_param_combinations:
        return [{}]

    return [dict(combination) for combination in itertools.product(*all_param_combinations)]


def build_compartment_specs(
    sampling_config: SamplingConfiguration, sampler: Sampler
) -> tuple[list[str], list[float], list[float]]:
    """Prepare compartment specifications for sampling."""
    compartment_names = sampler.compartments or []
    mins = []
    maxs = []

    if compartment_names and sampling_config.modelset.sampling.compartments:
        for comp_name in compartment_names:
            if comp_name in sampling_config.modelset.sampling.compartments:
                comp_range = sampling_config.modelset.sampling.compartments[comp_name]
                mins.append(comp_range.min)
                maxs.append(comp_range.max)
            else:
                msg = f"Compartment {comp_name} not found in configuration"
                raise ValueError(msg)

    return compartment_names, mins, maxs


def generate_samples_with_method(
    sampler: Sampler,
    compartment_specs: tuple[list[str], list[float], list[float]],
    param_specs: list[dict[str, Any]],
    seed: int | None,
) -> object:
    """Generate parameter samples using the specified sampling method."""
    compartment_names, mins, maxs = compartment_specs

    # Determine sampling method based on strategy
    sampling_method = "lhs" if sampler.strategy == "LHS" else sampler.strategy.lower()

    if compartment_names:
        return generate_parameter_samples(
            n_samples=sampler.n_samples,
            compartment_names=compartment_names,
            mins=mins,
            maxs=maxs,
            param_specs=param_specs,
            seed=seed,
            sampling_method=sampling_method,
        )

    # Only parameters, no compartments
    return generate_parameter_samples(
        n_samples=sampler.n_samples,
        compartment_names=[],
        mins=[],
        maxs=[],
        param_specs=param_specs,
        seed=seed,
        sampling_method=sampling_method,
    )


def convert_result_to_samples(sampler: Sampler, result: object) -> list[dict[str, Any]]:
    """Convert sampling result to desired format."""
    samples = []
    for i in range(sampler.n_samples):
        sample = {"parameters": {}, "compartments": {}, "start_date": None}

        # Extract parameters
        for param_name, param_values in result.parameters.items():
            if param_name == "start_date":
                sample["start_date"] = param_values[i].strftime("%Y-%m-%d")
            else:
                sample["parameters"][param_name] = param_values[i]

        # Extract compartments
        for comp_name, comp_values in result.compartments.items():
            if not comp_name.startswith("_"):  # Skip internal variables like _residual
                sample["compartments"][comp_name] = comp_values[i]

        samples.append(sample)

    return samples


def generate_primary_samples(
    sampling_config: SamplingConfiguration, sampler: Sampler, grid_params: set[str], seed: int | None = None
) -> list[dict[str, Any]]:
    """Generate samples for the primary (non-grid) sampling strategy."""
    # Prepare compartment specifications
    compartment_specs = build_compartment_specs(sampling_config, sampler)

    # Build parameter specifications (excluding grid parameters)
    param_specs = build_all_parameter_specs(sampler, sampling_config, exclude_params=grid_params)

    # Validate that we have something to sample
    compartment_names, _, _ = compartment_specs
    if not compartment_names and not param_specs:
        msg = f"{sampler.strategy} sampler must specify either compartments or parameters"
        raise ValueError(msg)

    # Generate samples using the sampler
    result = generate_samples_with_method(sampler, compartment_specs, param_specs, seed)

    # Convert to desired format
    return convert_result_to_samples(sampler, result)


def generate_samples(sampling_config: SamplingConfiguration, seed: int | None = None) -> list[dict[str, Any]]:
    """
    Generate samples from a validated sampling configuration.

    This function implements the core sampling logic:
    1. Create parameter specs for LHS/other samplers (excluding grid parameters)
    2. Generate samples with the primary sampler
    3. Apply grid sampling on top if grid sampler exists

    Parameters
    ----------
    sampling_config : SamplingConfiguration
        The validated sampling configuration
    seed : int | None, optional
        Random seed for reproducible sampling

    Returns
    -------
    List[Dict[str, Any]]
        List of sample dictionaries, each containing:
        - 'parameters': dict of parameter names to values
        - 'compartments': dict of compartment names to values
        - 'start_date': date string (if start_date is sampled)
    """
    # Separate grid and non-grid samplers
    grid_sampler = next((s for s in sampling_config.modelset.sampling.samplers if s.strategy == "grid"), None)
    other_sampler = next((s for s in sampling_config.modelset.sampling.samplers if s.strategy != "grid"), None)

    # Validate sampler constraints
    max_samplers = 2
    if len(sampling_config.modelset.sampling.samplers) > max_samplers:
        msg = f"Maximum of {max_samplers} samplers allowed"
        raise ValueError(msg)

    grid_sampler_count = len([s for s in sampling_config.modelset.sampling.samplers if s.strategy == "grid"])
    if grid_sampler_count > 1:
        msg = "Only one grid sampler allowed"
        raise ValueError(msg)

    # Get grid parameters for exclusion from other samplers
    grid_params = set(grid_sampler.parameters) if grid_sampler else set()

    # Generate grid combinations
    grid_combinations = build_grid_combinations(sampling_config, grid_sampler) if grid_sampler else [{}]

    all_samples = []

    if other_sampler:
        # Generate primary samples for each grid combination
        for grid_combo in grid_combinations:
            primary_samples = generate_primary_samples(sampling_config, other_sampler, grid_params, seed=seed)

            # Combine grid parameters with primary sampler results
            for primary_sample in primary_samples:
                combined_sample = {
                    "parameters": {**grid_combo, **primary_sample["parameters"]},
                    "compartments": primary_sample["compartments"],
                    "start_date": primary_sample["start_date"],
                }
                all_samples.append(combined_sample)
    else:
        # Only grid sampler - return grid combinations as samples
        all_samples = [{"parameters": combo, "compartments": {}, "start_date": None} for combo in grid_combinations]

    return all_samples


def generate_samples_from_config(config_path: str, seed: int | None = None) -> list[dict[str, Any]]:
    """
    Read validated sampling configuration from file and generate samples.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file
    seed : int | None, optional
        Random seed for reproducible sampling

    Returns
    -------
    List[Dict[str, Any]]
        List of sample dictionaries, each containing:
        - 'parameters': dict of parameter names to values
        - 'compartments': dict of compartment names to values
        - 'start_date': date string (if start_date is sampled)

    Examples
    --------
    >>> samples = generate_samples_from_config('basic_modelset_sampling.yml')
    >>> len(samples)
    500  # 5 grid values x 100 LHS samples
    >>> samples[0].keys()
    dict_keys(['parameters', 'compartments', 'start_date'])
    """
    # Load and validate configuration
    validated_config = load_sampling_config_from_file(config_path)
    sampling_config = validated_config.modelset.sampling

    return generate_samples(sampling_config, seed=seed)
