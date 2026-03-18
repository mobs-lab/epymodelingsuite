"""Utility functions for epymodelingsuite.

This package contains utility functions organized by category:
- logging: Dual-mode logging (local/cloud) configuration
- expression_eval: Safe expression evaluation for model parameters
- location: Location validation and conversion utilities
- populations: Population-related utilities
- config: Configuration file utilities
- distributions: Distribution conversion utilities
- formatting: Formatting utilities for human-readable output
- common: Common utility functions
- data: External data fetching utilities
"""

# Import all public functions from submodules
from .common import parse_timedelta
from .config import identify_config_type
from .data import fetch_hhs_hospitalizations
from .distance import wrmse
from .distributions import distribution_to_scipy
from .expression_eval import RetrieveName, SafeEvalVisitor, safe_eval
from .formatting import format_data_size, format_duration
from .location import convert_location_name_format, get_location_codebook, validate_iso3166
from .logging import StructuredFormatter, configure_logging, setup_logger
from .populations import get_population_codebook, get_total_population, make_dummy_population

__all__ = [
    "RetrieveName",
    "SafeEvalVisitor",
    "StructuredFormatter",
    "configure_logging",
    "convert_location_name_format",
    "distribution_to_scipy",
    "fetch_hhs_hospitalizations",
    "format_data_size",
    "format_duration",
    "get_location_codebook",
    "get_population_codebook",
    "get_total_population",
    "identify_config_type",
    "make_dummy_population",
    "parse_timedelta",
    "safe_eval",
    "setup_logger",
    "validate_iso3166",
    "wrmse",
]
