import os
import warnings
from contextlib import contextmanager

import numpy as np

from ClearMap.IO.FileUtils import normalize_location_spec
from ClearMap.IO.source import Source as source_mod
from ClearMap.IO.dispatch import source_to_module, as_source

from ClearMap.Utils.exceptions import ClearMapValueError


def read(source_: os.PathLike | np.ndarray | source_mod.Source,
         slicing=None, *args, **kwargs):
    """
    Read data from a data source.

    .. warning::
        For some modules (file types) this does an active read
        for others, it just does an open and returns a Source class
        that can be used to read the data.

    Parameters
    ----------
    source_ : str, pathlib.Path, array, Source class
       The source to read the data from.

    Returns
    -------
    data : array
        The data of the source.
    """
    source_ = normalize_location_spec(source_)

    if isinstance(source_, np.ndarray):  # Already materialised — nothing to do
        return source_
    elif isinstance(source_, source_mod.Source):  # Source-like with .array (Block, Slice, NPY.Source, MMP.Source, ...)
        if not hasattr(source_, 'array'):
            raise ClearMapValueError(f'Source {source_} has no array property and cannot be read directly')
        if args or kwargs:
            warnings.warn(f'Ignoring unsupported read arguments {args=} and {kwargs=}'
                          f' for materialised source {source_!r}.', stacklevel=2)
        return source_.array  if slicing is None else source_[slicing]

    # File path or expression — dispatch to the right module
    mod = source_to_module(source_)
    return mod.read(source_, *args, **kwargs)


def open_ro(source_, **kwargs):
    """Open a source strictly read-only for metadata queries."""
    source_ = normalize_location_spec(source_)

    mod = source_to_module(source_)
    if isinstance(source_, source_mod.Source):
        if source_.mode == 'r':
            return source_
        if hasattr(mod, 'open_ro'):
            return mod.open_ro(source_, **kwargs)
        return source_  # non-MMP Sources are read-only at API level already
    else:  # e.g. string
        if hasattr(mod, 'open_ro'):
            return mod.open_ro(source_, **kwargs)
    # no open_ro available -> fallback: construct with mode='r'
    return module_to_source_cls[mod](source_, mode='r', **kwargs)  # FIXME:


def edit(source_, **kwargs):
    """Open a source for in-place editing (mode='r+')."""
    source_ = normalize_location_spec(source_)

    mod = source_to_module(source_)
    if hasattr(mod, 'edit'):
        return mod.edit(source_, **kwargs)
    return as_source(source_, **kwargs)  # FIXME: explicit mode = r+


def write(sink, data, *args, **kwargs):
    """
    Write data to a data source.

    Parameters
    ----------
    sink : str, pathlib.Path, array, Source class
        The source to write data to.
    data : array
        The data to write to the sink.
    slicing : slice specification or None
        Optional sub-slice to write data to.

    Returns
    -------
    sink : str, array or Source class
        The sink to which the data was written.
    """
    # REFACTOR: if both source_to_module and mod.write implement normalize, we can kick it here (and in create...)
    sink = normalize_location_spec(sink)
    mod = source_to_module(sink)
    return mod.write(sink, open_ro(data), *args, **kwargs)


def create(source_, *args, **kwargs):
    """
    Create a data source on disk.

    Parameters
    ----------
    source_ : str, pathlib.Path, array, Source class
        The source to write data to.

    Returns
    -------
    sink : str, array or Source class
       The sink to which the data was written.
    """
    source_ = normalize_location_spec(source_)
    mod = source_to_module(source_)
    return mod.create(source_, *args, **kwargs)
