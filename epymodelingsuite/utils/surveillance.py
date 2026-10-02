"""Reuse surveillance reads for the lifetime of an output call."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

import pandas as pd

_reads = ContextVar("output_surveillance_reads", default=None)


@contextmanager
def surveillance_cache():
    """Scope surveillance CSV reuse to one output invocation.

    Yields
    ------
    None
        Control inside the active cache context.

    Notes
    -----
    Nested contexts reuse the active cache. The outer context resets it on exit,
    including exceptional exit; no process-global frames are retained.
    """
    if _reads.get() is not None:
        yield
        return
    token = _reads.set({})
    try:
        yield
    finally:
        _reads.reset(token)


def share_surveillance(function):
    """Wrap a function in an invocation-scoped surveillance cache.

    Parameters
    ----------
    function : callable
        Function whose surveillance reads should share one cache.

    Returns
    -------
    callable
        Wrapper preserving the function's metadata, return value and exceptions.
    """

    @wraps(function)
    def wrapped(*args, **kwargs):
        """Invoke the decorated function inside the surveillance cache.

        Parameters
        ----------
        *args : tuple
            Positional arguments forwarded to the function.
        **kwargs : dict
            Keyword arguments forwarded to the function.

        Returns
        -------
        object
            The decorated function's return value.
        """
        with surveillance_cache():
            return function(*args, **kwargs)

    return wrapped


def read_surveillance_csv(path, **kwargs):
    """Read a surveillance CSV, reusing matching reads in the active context.

    Parameters
    ----------
    path : str or path-like
        CSV path passed to pandas.read_csv.
    **kwargs : dict
        Parser options forwarded to pandas.read_csv and included in the cache key.

    Returns
    -------
    pd.DataFrame
        A copy of a cached frame, or the direct pandas result when no cache is active.

    Notes
    -----
    Copies isolate consumer mutations. File and parsing errors propagate from pandas.
    """
    cache = _reads.get()
    if cache is None:
        return pd.read_csv(path, **kwargs)
    # Parser options are part of the key: plots and hub retain their existing
    # identifier/date inference. Never return the cached frame for mutation.
    key = str(path), repr(sorted(kwargs.items()))
    if key not in cache:
        cache[key] = pd.read_csv(path, **kwargs)
    return cache[key].copy()
