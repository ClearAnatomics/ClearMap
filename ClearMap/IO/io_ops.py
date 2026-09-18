import pathlib
import warnings

import numpy as np
import pandas as pd

from ClearMap.IO.source import Source as source_mod
from ClearMap.IO.IO import _is_feather_path, module_to_source_cls
from ClearMap.IO.dispatch import source_to_module, as_source
from ClearMap.Utils.exceptions import ClearMapValueError


def read(source_, slicing=None, *args, **kwargs):
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
    if isinstance(source_, pathlib.Path):
        source_ = str(source_)
    if _is_feather_path(source_):
        return pd.read_feather(source_)
    elif isinstance(source_, np.ndarray):  # Already materialised — nothing to do
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
    if isinstance(source_, pathlib.Path):
        source_ = str(source_)
    if _is_feather_path(source_):
        return pd.read_feather(source_)
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
    return module_to_source_cls[mod](source_, mode='r', **kwargs)


def edit(source_, **kwargs):
    """Open a source for in-place editing (mode='r+')."""
    if isinstance(source_, pathlib.Path):
        source_ = str(source_)
    if _is_feather_path(source_):
        return pd.read_feather(source_)
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
    if isinstance(sink, pathlib.Path):
        sink = str(sink)
    if _is_feather_path(sink):
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)  # backward compat: structured array
        if not isinstance(data.index, pd.RangeIndex) or data.index[0] != 0:
            data = data.reset_index(drop=True)  # feather requires default RangeIndex
        data.to_feather(sink)
        return sink
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
    if isinstance(source_, pathlib.Path):
        source_ = str(source_)
    mod = source_to_module(source_)
    return mod.create(source_, *args, **kwargs)
