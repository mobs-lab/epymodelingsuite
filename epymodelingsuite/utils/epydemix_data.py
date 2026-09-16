"""Pinned access to the `epydemix-data` GitHub repository.

`epydemix` itself does not ship demographic or contact-matrix data. The data
lives in https://github.com/epistorm/epydemix-data and is fetched at runtime.
An upstream rename of that data (e.g., the switch from single- to
double-underscore US state names) can break our pipeline without any change on
our side.

Starting in epydemix v1.2.0, `load_epydemix_population` accepts a
``data_version`` argument that maps to a git tag of the data repository,
replacing the older ``path_to_data_github`` URL parameter. We wrap the upstream
loader so it always uses a pinned tag. Bumping the data version is a single-line
change to ``EPYDEMIX_DATA_VERSION`` below.
"""

from __future__ import annotations

from epydemix.population import (
    get_available_locations as _upstream_get_available_locations,
)
from epydemix.population import (
    load_epydemix_population as _upstream_load_epydemix_population,
)

EPYDEMIX_DATA_VERSION = "v1.2.0"


def load_epydemix_population(*args, **kwargs):
    """Wrap upstream ``load_epydemix_population`` with a pinned data version.

    Forwards every argument unchanged except ``data_version``, which is forced
    to ``EPYDEMIX_DATA_VERSION``. Callers may still pass ``path_to_data`` for
    fully local data.
    """
    kwargs.setdefault("data_version", EPYDEMIX_DATA_VERSION)
    return _upstream_load_epydemix_population(*args, **kwargs)


def get_available_locations(*args, **kwargs):
    """Wrap upstream ``get_available_locations`` with a pinned data version."""
    kwargs.setdefault("data_version", EPYDEMIX_DATA_VERSION)
    return _upstream_get_available_locations(*args, **kwargs)
