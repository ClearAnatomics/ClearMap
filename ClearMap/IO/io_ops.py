import os
import warnings
from contextlib import contextmanager

import numpy as np

from ClearMap.IO.source import Source as source_mod
from ClearMap.IO.dispatch import source_to_module, source_to_class, as_source, normalize_source_spec

from ClearMap.Utils import tag_expression as te
from ClearMap.Utils.exceptions import ClearMapValueError, SourceModuleNotFoundError, SourceNotFoundError


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
    source_ = normalize_source_spec(source_)

    if isinstance(source_, np.ndarray):  # Already materialised — nothing to do
        return source_ # TODO: check if we fwd slice
    elif isinstance(source_, (source_mod.TableSource, source_mod.GraphSource)):  # Whole-file sources read themselves
        return source_.read(slicing=slicing, **kwargs)
    elif isinstance(source_, source_mod.Source):  # Source-like with .array (Block, Slice, NPY.Source, MMP.Source, ...)
        if not hasattr(source_, 'array'):
            raise ClearMapValueError(f'Source {source_} has no array property and cannot be read directly')
        if args or kwargs:
            warnings.warn(f'Ignoring unsupported read arguments {args=} and {kwargs=}'
                          f' for materialised source {source_!r}.', stacklevel=2)
        return source_.array  if slicing is None else source_[slicing]

    # File path or expression — dispatch to the right module
    mod = source_to_module(source_)
    return mod.read(source_, slicing, *args, **kwargs)  # positional order matches the backend read(source, slicing, ...)


def open_ro(source_, **kwargs):
    """Open a source strictly read-only for metadata queries."""
    source_ = normalize_source_spec(source_)

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
    return source_to_class(source_)(source_, mode='r', **kwargs)


def edit(source_, **kwargs):
    """Open a source for in-place editing (mode='r+')."""
    source_ = normalize_source_spec(source_)

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
    data : array, DataFrame or Source
        The data to write to the sink. For table sinks (CSV, Feather), a DataFrame, a
        table source, or an array together with ``columns=[...]``.
    slicing : slice specification or None
        Optional sub-slice to write data to.

    Returns
    -------
    sink : str, array or Source class
        The sink to which the data was written.
    """
    # REFACTOR: if both source_to_module and mod.write implement normalize, we can kick it here (and in create...)
    sink = normalize_source_spec(sink)
    mod = source_to_module(sink)
    if mod.SOURCE_CLASS.data_model == 'array':  # Table and graph backends coerce their own input
        data = open_ro(data)
    return mod.write(sink, data, *args, **kwargs)


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
    source_ = normalize_source_spec(source_)
    mod = source_to_module(source_)
    return mod.create(source_, *args, **kwargs)


@contextmanager
def peek_into(source_, **kwargs):
    """Temporarily open *source_* for metadata inspection.

    Existing Source objects remain owned by the caller. Sources opened from
    paths or other descriptors are closed on exit when possible.
    """
    source = open_ro(source_, **kwargs)
    owns_source = source is not source_

    try:
        yield source
    finally:
        if owns_source and hasattr(source, 'close'):
            source.close()  # FIXME: no `close` API in Source


def is_source(source_, exists=True):
    """
    Check whether *source_* is a valid source specification.

    Parameters
    ----------
    source_ : object
        The source specification to check.
    exists : bool
        If True, locations and Sources must also exist.

    Returns
    -------
    is_source : bool
        True if *source_* is a valid (and, if requested, existing) source.
    """
    source_ = normalize_source_spec(source_)
    if isinstance(source_, source_mod.Source):
        return source_.exists() if exists else True
    if isinstance(source_, (str, te.Expression)):
        try:
            source_to_module(source_)  # a format ClearMap can handle?
        except SourceModuleNotFoundError:
            return False
        if not exists:
            return True
        try:
            return open_ro(source_).exists()
        except (FileNotFoundError, SourceNotFoundError):
            return False
    return isinstance(source_, (np.ndarray, list, tuple))


# TODO: arg memory= to specify which kind of array is created, better use device=
# TODO: arg processes= in order to use ParallelIO -> can combine with buffer=

def buffer(source_):
    """
    Return an IO buffer of the data of a source, e.g. for use with Cython.

    Parameters
    ----------
    source_ : source specification
        The source specification.

    Returns
    -------
    buffer : array or memmap
        A buffer to read and write data.
    """
    try:
        return as_source(source_).as_buffer()
    except Exception as err:
        raise ClearMapValueError(f'Cannot get an IO buffer for {source_!r}: {err}') from err
