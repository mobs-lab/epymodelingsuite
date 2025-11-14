"""Utility functions for epymodelingsuite.

This package contains utility functions organized by category:
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
from .distributions import distribution_to_scipy
from .expression_eval import RetrieveName, SafeEvalVisitor, safe_eval
from .formatting import format_data_size, format_duration
from .location import convert_location_name_format, get_location_codebook, validate_iso3166
from .populations import get_population_codebook, make_dummy_population

__all__ = [
    # Common utilities
    "parse_timedelta",
    # Config utilities
    "identify_config_type",
    # Data utilities
    "fetch_hhs_hospitalizations",
    # Distribution utilities
    "distribution_to_scipy",
    # Expression evaluation
    "RetrieveName",
    "SafeEvalVisitor",
    "safe_eval",
    # Formatting utilities
    "format_data_size",
    "format_duration",
    # Location utilities
    "convert_location_name_format",
    "get_location_codebook",
    "validate_iso3166",
    # Population utilities
    "get_population_codebook",
    "make_dummy_population",
]
