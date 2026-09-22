# -*- coding: utf-8 -*-
"""
Feather
=======

Backend for Apache Feather tabular files.
"""

import os
from functools import cached_property

import pandas as pd
import pyarrow  # noqa: F401  # pandas Feather support depends on pyarrow

from ClearMap.IO.FileUtils import normalize_location_spec
from ClearMap.IO.source.Source import Source, TableSource
from ClearMap.IO.source.backends.registry import BackendName

from ClearMap.Utils.exceptions import ClearMapNotImplementedError, SourceNotFoundError


PathLike = str | os.PathLike[str]


class FeatherSource(TableSource):
    """Tabular source backed by a Feather file."""

    backend = BackendName.FEATHER
    _name = 'Feather'
    _CACHED_PROPERTIES = ('frame',)

    def __init__(self, location: PathLike, mode: str = 'r', name: str | None = None):
        super().__init__(name=name, mode=mode)
        self.location = location

    @cached_property
    def frame(self) -> pd.DataFrame:
        """The table stored in this Feather source."""
        if self.location is None:
            raise SourceNotFoundError(
                location=None,
                message='FeatherSource has no location.',
            )

        if not self.exists():
            raise SourceNotFoundError(
                location=self.location,
                message=f'Feather file does not exist: {self.location}',
            )

        return pd.read_feather(self.location)

    def read(self, slicing=None, **kwargs) -> pd.DataFrame:
        """Read this Feather table.

        Parameters
        ----------
        slicing
            Optional positional slicing applied with ``DataFrame.iloc``.
        **kwargs
            Additional arguments passed to :func:`pandas.read_feather`.

        Notes
        -----
        A plain read uses the cached ``frame`` property. Reads with additional
        pandas arguments bypass the cache because they may request only part
        of the file, e.g. selected columns.
        """
        if kwargs:
            frame = pd.read_feather(self.location, **kwargs)
        else:
            frame = self.frame

        if slicing is not None:
            frame = frame.iloc[slicing]

        return frame

    def as_memory(self):
        """Return the in-memory tabular representation."""
        return self.frame


SOURCE_CLASS = FeatherSource


def open_ro(source_, **kwargs) -> FeatherSource:
    """Open a Feather source read-only."""
    if isinstance(source_, FeatherSource):
        return source_

    return FeatherSource(source_, mode='r', **kwargs)


def read(source_, slicing=None, **kwargs) -> pd.DataFrame:
    """Read a Feather file as a pandas DataFrame."""
    if isinstance(source_, FeatherSource):
        return source_.read(slicing=slicing, **kwargs)

    source = FeatherSource(source_, mode='r')
    return source.read(slicing=slicing, **kwargs)


def _as_frame(data) -> pd.DataFrame:
    """Coerce supported ClearMap/tabular inputs to a DataFrame."""
    if isinstance(data, pd.DataFrame):
        return data

    if isinstance(data, TableSource):
        return data.frame

    if isinstance(data, Source):
        data = data.as_memory()

    return pd.DataFrame(data)


def write(sink, data=None, slicing=None, overwrite=True, **kwargs):
    """Write a complete table to a Feather file.

    Parameters
    ----------
    sink : str | os.PathLike | FeatherSource
        Destination Feather file.
    data : DataFrame | TableSource | Source | array-like
        Tabular data to write. Non-DataFrame inputs are converted to a
        DataFrame for backwards compatibility.
    slicing
        Feather does not support partial in-place writes.
    overwrite : bool
        Whether an existing file may be replaced. Defaults to True to preserve
        the historical ClearMap Feather behaviour.
    **kwargs
        Additional arguments passed to :meth:`DataFrame.to_feather`.

    Returns
    -------
    sink
        The original sink specification.
    """
    if slicing is not None:
        raise ClearMapNotImplementedError('Feather does not support sliced/in-place writes; '
                                          'rewrite the complete table instead.',
                                          operation='write', backend='feather')

    if data is None:
        raise ValueError('Feather write() requires data.')

    location = sink.location if isinstance(sink, FeatherSource) else normalize_location_spec(sink)

    if not overwrite and os.path.exists(location):
        raise FileExistsError(f'Feather file already exists: {location}')

    frame = _as_frame(data)

    # Feather expects the default RangeIndex. This also handles empty frames,
    # unlike checking data.index[0].
    expected_index = pd.RangeIndex(len(frame))
    if not frame.index.equals(expected_index):
        frame = frame.reset_index(drop=True)

    frame.to_feather(location, **kwargs)

    # If a FeatherSource was rewritten, make sure a previously cached frame
    # cannot survive the write.
    if isinstance(sink, FeatherSource):
        sink._invalidate_cache()

    return sink


def create(location=None, shape=None, dtype=None, order=None,
           mode=None, array=None, as_source=True, **kwargs):
    """Create a Feather file from tabular data.

    Blank creation from ``shape`` / ``dtype`` is intentionally unsupported:
    Feather tables have named columns and potentially different dtypes per
    column.
    """
    if location is None:
        raise ValueError('Feather create() requires a location.')

    if array is None:
        raise ClearMapNotImplementedError('A blank Feather table cannot be created from shape/dtype alone; '
                                          'provide table data via array= or use write().',
                                          operation='create', backend='feather')

    if shape is not None or dtype is not None or order is not None:
        raise ClearMapValueError('shape, dtype and order are not meaningful creation arguments for a Feather table.')

    write(location, data=array, overwrite=True, **kwargs)

    if as_source:
        return FeatherSource(location, mode='r')

    return pd.read_feather(location)