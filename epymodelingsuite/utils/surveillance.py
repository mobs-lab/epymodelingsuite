"""Reuse surveillance reads for the lifetime of an output call."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

import pandas as pd

_reads = ContextVar("output_surveillance_reads", default=None)


@contextmanager
def surveillance_cache():
    if _reads.get() is not None:
        yield
        return
    token = _reads.set({})
    try:
        yield
    finally:
        _reads.reset(token)


def share_surveillance(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with surveillance_cache():
            return function(*args, **kwargs)

    return wrapped


def read_surveillance_csv(path, **kwargs):
    cache = _reads.get()
    if cache is None:
        return pd.read_csv(path, **kwargs)
    # Parser options are part of the key: plots and hub retain their existing
    # identifier/date inference. Never return the cached frame for mutation.
    key = str(path), repr(sorted(kwargs.items()))
    if key not in cache:
        cache[key] = pd.read_csv(path, **kwargs)
    return cache[key].copy()
