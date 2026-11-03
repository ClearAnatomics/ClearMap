# -*- coding: utf-8 -*-
"""
CSV
===

Backend for CSV tables, read and written with pandas.

A ClearMap CSV file is a table: its first row names the columns. Headerless
numeric files written by older ClearMap versions are refused on read rather than
silently misparsed (pandas would otherwise take the first data row as the header).
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'https://idisco.info'
__download__  = 'https://www.github.com/ChristophKirst/ClearMap2'

import pandas as pd

from ClearMap.IO.source.Source import TableSource
from ClearMap.IO.source.protocol import Backend
from ClearMap.Utils.exceptions import ClearMapValueError


# Reader options that change how the header row is found; forwarded to the header probe.
_HEADER_PROBE_KWARGS = ('sep', 'delimiter', 'encoding', 'comment', 'skiprows', 'compression', 'quotechar')


class CSVSource(TableSource):
    """Table source backed by a CSV file with a header row."""

    backend = Backend.CSV
    _name = 'CSV-Source'

    @classmethod
    def _load(cls, location, **kwargs):
        if 'header' not in kwargs and 'names' not in kwargs:  # caller took charge of the header explicitly
            _check_header(location, kwargs)
        return pd.read_csv(location, **kwargs)

    @classmethod
    def _dump(cls, frame, location, **kwargs):
        _check_column_names(frame.columns, location, reading=False)
        kwargs.setdefault('index', False)
        frame.to_csv(location, **kwargs)


# class CSVVirtualSource(VirtualSource):
#     _real_class = CSVSource
#
#     def __init__(self, source=None, shape=None, dtype=None, order=None,
#                  location=None, name=None, mode=None):
#         super().__init__(source=source, shape=shape, dtype=dtype, order=order, location=location, name=name, mode=mode)
#         if isinstance(source, CSVSource):
#             self.location = source.location

SOURCE_CLASS = CSVSource

###############################################################################
### IO Interface
###############################################################################

def open_ro(source_, **kwargs):
    return CSVSource.open_ro(source_, **kwargs)


def read(source_, slicing=None, **kwargs):
    return CSVSource.read_table(source_, slicing=slicing, **kwargs)


def write(sink, data=None, slicing=None, overwrite=True, **kwargs):
    return CSVSource.write_table(sink, data, slicing=slicing, overwrite=overwrite, **kwargs)


def create(location=None, shape=None, dtype=None, order=None,
           mode=None, array=None, as_source=True, **kwargs):
    return CSVSource.create_table(location, shape=shape, dtype=dtype, order=order,
                                  mode=mode, array=array, as_source=as_source, **kwargs)


def edit(source_, **kwargs):
    return CSVSource.edit(source_, **kwargs)


def is_csv(source):
    """Checks if this source is a CSV source."""
    return isinstance(source, CSVSource) or (isinstance(source, str) and source.lower().endswith('.csv'))


###############################################################################
### Header validation
###############################################################################

def _is_number(value):
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _check_column_names(columns, location, reading=True):
    """Raise if more than half of *columns* parse as numbers, i.e. look like data, not names."""
    n_numeric = sum(_is_number(column) for column in columns)
    if n_numeric * 2 <= len(columns):
        return
    if reading:
        raise ClearMapValueError(
            f'{location}: the first row looks like data, not a header ({n_numeric}/{len(columns)} fields are '
            f'numbers). ClearMap tables need a header row naming the columns. For a headerless file from an '
            f'older ClearMap version, add a header line, or read it explicitly with header=None, names=[...].',
            value=tuple(columns), expected='a header row of column names')
    raise ClearMapValueError(
        f'Refusing to write {location}: {n_numeric}/{len(columns)} column names look like numbers and would '
        f'be unreadable as a header. Give the columns descriptive names.',
        value=tuple(columns), expected='descriptive column names')


def _check_header(location, read_kwargs):
    """Validate the header row by reading it alone, before paying for a full read."""
    probe = {key: read_kwargs[key] for key in _HEADER_PROBE_KWARGS if key in read_kwargs}
    try:
        columns = pd.read_csv(location, nrows=0, **probe).columns
    except pd.errors.EmptyDataError as err:
        raise ClearMapValueError(f'{location} is empty; expected at least a header row.') from err
    _check_column_names(columns, location, reading=True)
