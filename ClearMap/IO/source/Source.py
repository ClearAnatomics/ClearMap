# -*- coding: utf-8 -*-
"""
Source
======

This module provides the base class for data sources and sinks.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'https://idisco.info'
__download__  = 'https://www.github.com/ChristophKirst/ClearMap2'

import warnings
from functools import cached_property
from typing import ClassVar, TYPE_CHECKING

import numpy as np

import ClearMap.IO.FileUtils as fu
from ClearMap.IO.source import geometry_utils
from ClearMap.IO.source.source_modes import WRITABLE_MODES, PERSISTABLE_MODES

from ClearMap.Utils.Formatting import ensure
from ClearMap.Utils.exceptions import (ClearMapValueError, ClearMapRuntimeError, ClearMapPermissionError,
                                       ClearMapNotImplementedError, SourceNotFoundError)

if TYPE_CHECKING:
    from ClearMap.IO.source.protocol import Backend


def trim_path(path, max_len=100, keep=50):
    """Shorten a path for display, keeping both the root and the filename."""
    text = str(path)
    if len(text) <= max_len:
        return text
    return f'{text[:keep]}...{text[-keep:]}'


class ReprField:
    """One field in a source's string representation.

    Arguments
    ---------
    attr : str
        Attribute to read off the source. A missing attribute, a ``None`` value
        or a failing conversion all render as ``''``.
    delimiters : str
        Two-character open/close pair, e.g. ``'[]'``, ``'{}'``, ``'||'``.
        Empty for an undelimited field.
    convert : callable
        Applied to the value before interpolation.
    cheap : bool
        False if rendering this field may trigger I/O; see ``Source.__str__``.

    Note
    ----
    Instances are shared between classes as module-level constants, so treat
    them as immutable.
    """

    __slots__ = ('attr', 'open', 'close', 'convert', 'cheap')

    def __init__(self, attr, delimiters='', convert=str, cheap=True):
        if delimiters and len(delimiters) != 2:
            raise ClearMapValueError('delimiters must be a two-character open/close pair, or empty.',
                                     value=delimiters, expected="'[]', '{}', '||', ''")
        self.attr = attr
        self.open, self.close = delimiters or ('', '')
        self.convert = convert
        self.cheap = cheap

    def render(self, source):
        try:
            value = getattr(source, self.attr, None)
        except Exception:
            return ''
        if value is None:
            return ''
        try:
            return f'{self.open}{self.convert(value)}{self.close}'
        except Exception:
            return ''

    def __repr__(self):
        return f'{type(self).__name__}({self.attr!r}, {self.open + self.close!r})'

class ReprFields:
    """Shared display fields. See ``Source.__str__``."""
    NAME     = ReprField('name')
    SHAPE    = ReprField('shape', convert=lambda s: repr(tuple(s)))
    DTYPE    = ReprField('dtype', '[]')
    ORDER    = ReprField('order', '||')
    LOCATION = ReprField('location', '{}', convert=trim_path)

###############################################################################
### Source base class
###############################################################################

class Source:
    """Base abstract source class."""
    backend: ClassVar['Backend | None'] = None

    _name: ClassVar[str | None] = None  # override in subclasses as class variable
    _location: ClassVar[str | None] = None
    _CACHED_PROPERTIES: ClassVar[tuple[str, ...]] = ()
    _REPR_FIELDS: ClassVar[tuple[ReprField, ...]] = (ReprFields.NAME, ReprFields.LOCATION)
    _virtual_class: ClassVar[type | None] = None
    data_model: ClassVar[str | None] = None

    # __slots__ = ()

    def __init__(self, name=None, mode=None):
        """Initialization."""
        if name is not None:
            self._name = name
        self._mode = mode

    @property
    def name(self):
        """The name of this source.

        Returns
        -------
        name : str
            Name of this source.
        """
        mod_name = type(self).__module__.split(".")[-1]
        name_fallback = f'{mod_name}-Source'
        cls_name = getattr(self, '_name', name_fallback)
        if cls_name is None:
            cls_name = name_fallback
        return cls_name

    @name.setter
    def name(self, value: str):
        warnings.warn('Setting name on a Source instance is discouraged (reserved for testing/debugging). '
                      'Define _name as a class variable in subclasses instead.', UserWarning, stacklevel=2)
        self._name = ensure(value, str)

    @property
    def location(self):
        """The location where the data of the source is stored or None for memory_only sources."""
        return self._location

    @location.setter
    def location(self, value):
        value = self._coerce_location(value)
        if value == self._location:
            return
        self._location = value
        self._invalidate_cache()
        self._on_location_changed()

    def _coerce_location(self, value):
        """Normalise a location before storing. Override to canonicalise."""
        return fu.normalize_location_spec(value)

    def _on_location_changed(self):
        """Hook for subclasses that must re-derive state eagerly."""

    def _invalidate_cache(self):
        for key in self._CACHED_PROPERTIES:
            self.__dict__.pop(key, None)

    def exists(self):
        return fu.is_file(self.location) if self.location is not None else False

    ### Data

    @property
    def mode(self):
        return self._mode

    @property
    def is_writable(self):
        """True if in-memory mutation via __setitem__ is permitted.

        Note: mode 'c' is writable *in memory only* — changes are never persisted.
        """
        return self.mode is None or self.mode in WRITABLE_MODES

    @property
    def is_persistable(self):
        """True if changes can be written back to disk (mode 'r+' or 'w+')."""
        if self.mode in PERSISTABLE_MODES:
            return True
        if self.mode is None:
            # mode-less is only honest for pure in-memory sources
            return self.location is None
        return False

    def _assert_writable(self):
        if not self.is_writable:
            raise ClearMapPermissionError(f'{self} was opened read-only (mode="r"). Use io.edit() to open for writing.')
        if self.location is not None and not self.is_persistable:
            raise ClearMapPermissionError(f'{self} was opened with mode={self._mode!r}; writes would not be '
                                          f'persisted to {self.location}. Use io.edit() to open for writing.')

    # ## Element access

    def __getitem__(self, slicing):
        return self._getitem(slicing)

    def __setitem__(self, slicing, value):
        self._assert_writable()
        self._setitem(slicing, value)

    def _getitem(self, slicing):
        raise ClearMapNotImplementedError(f'{type(self).__name__} does not support item access.',
                                          operation='getitem', backend=type(self).__name__)

    def _setitem(self, slicing, value):
        raise ClearMapNotImplementedError(f'{type(self).__name__} does not support item assignment.',
                                          operation='setitem', backend=type(self).__name__)

    def read(self, *args, **kwargs):
        raise ClearMapNotImplementedError(f'{type(self).__name__} does not support read().',
                                          operation='read', backend=type(self).__name__)

    def write(self, *args, **kwargs):
        raise ClearMapNotImplementedError(f'{type(self).__name__} does not support write().',
                                          operation='write', backend=type(self).__name__)

    # ## Conversions

    def as_memory(self):  # FIXME
        return np.array(self.as_buffer())

    def as_real(self):
        return self

    def as_virtual(self):
        if self._virtual_class is None:
            raise ClearMapNotImplementedError(f'{type(self).__name__} has no virtual counterpart.',
                                              operation='as_virtual', backend=type(self).__name__)
        return self._virtual_class(source=self)

    def as_buffer(self):
        raise ClearMapNotImplementedError(f'{type(self).__name__} has no buffer representation.',
                                          operation='as_buffer', backend=type(self).__name__)

    ### Formatting
    def __str__(self):
        fields = self._REPR_FIELDS
        # if not REPR_ALLOWS_IO:  # FIXME: add check later for cheap only
        #     fields = tuple(f for f in fields if f.cheap)
        return ''.join(f.render(self) for f in fields)

    __repr__ = __str__


###############################################################################
### Abstract and VirtualSource base class
###############################################################################

# TODO: memory -> device argument
class AbstractSource(Source):
    """Abstract source to handle data sources without data in memory.

    Note
    ----
    This class handles essential info about a source and to how access its data.
    """

    # __slots__ = ('_shape', '_dtype', '_order', '_location')

    def __init__(self, source=None, shape=None, dtype=None,
                 order=None, location =None, name=None, mode=None):
        """Source class constructor.

        Arguments
        ---------
        source :
        shape : tuple of int or None
            Shape of the source, if None try to determine from source.
        dtype : dtype or None
            The data type of the source, if None try to determine from source.
        order : 'C' or 'F' or None
            The order of the source, c or Fortran contiguous.
        location : str or None
            The location of the source.
        name : str | None
        mode : str | None
        """
        super().__init__(name=name, mode=mode)

        if source is not None:
            if shape is None and hasattr(source, 'shape'):
                shape = source.shape
            if dtype is None and hasattr(source, 'dtype'):
                dtype = source.dtype
            if order is None and hasattr(source, 'order'):
                order = source.order
            # if memory is None and hasattr(source, 'memory'):
            #     memory = memory.order
            if location is None and hasattr(source, 'location'):
                location = source.location
            if hasattr(source, 'mode'):  # Don't add to sources that don't have that attr
                if mode is None:
                    mode = source.mode
                self._mode = ensure(mode, str)

        self._shape    = ensure(shape,    tuple)
        self._dtype    = ensure(dtype,    np.dtype)
        self._order    = ensure(order,    str)
        # self._memory   = ensure(memory,   str)
        self._location = self._coerce_location(location)

    @property
    def shape(self):
        """The shape of the source.

        Returns
        -------
        shape : tuple
            The shape of the source.
        """
        return self._shape

    @shape.setter
    def shape(self, value):
        self._shape = ensure(value, tuple)

    @property
    def dtype(self):
        """The data type of the source.

        Returns
        -------
        dtype : dtype
            The data type of the source.
        """
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        self._dtype = ensure(value, np.dtype)

    @property
    def order(self):
        """The contiguous order of the data array of the source.

        Returns
        -------
        order : str
           Returns 'C' for C and 'F' for fortran contiguous arrays, None otherwise.
        """
        return self._order

    @order.setter
    def order(self, value):
        if value not in [None, 'C', 'F']:
            raise ValueError(f"Order {value!r} not in [None, 'C' or 'F']!")
        self._order = ensure(value, str)

    # @location.setter
    # def location(self, value):
    #     self._location = ensure(value, str)

    @property
    def is_writable(self):  # FIXME: check if parent implementation is OK
        return self.mode in WRITABLE_MODES

    def as_virtual(self):
        return self

    def as_real(self):
        raise ClearMapRuntimeError('The abstract source cannot be converted to a real source!')

    def as_buffer(self):
        raise ClearMapRuntimeError('The abstract source cannot be converted to a buffer!')


class VirtualSource(AbstractSource):
    """Virtual source to handle data sources without data in memory.

    Note
    ----
    This class is fast to serialize and useful as a source pointer in parallel processing.
    """
    _real_class: ClassVar[type['Source'] | None] = None  # Subclasses set this to their concrete Source class


    def __init__(self, source=None, shape=None, dtype=None,
                 order=None, location=None, name=None, mode=None):
        AbstractSource.__init__(self, source=source, shape=shape, dtype=dtype,
                                order=order, location=location, name=name, mode=mode)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.backend is None:
            real_cls = cls._real_class
            if real_cls is not None:
                cls.backend = real_cls.backend

    @property
    def name(self):
        if self._name is not None:
            return self._name
        mod_name = type(self).__module__.split(".")[-1]
        return f'Virtual-{mod_name}-Source'

    def __getitem__(self, *args):
        return self.as_real().__getitem__(*args)

    def __setitem__(self, *args):
        self.as_real().__setitem__(*args)

    def read(self, *args, **kwargs):
        return self.as_real().read(*args, **kwargs)

    def write(self, *args, **kwargs):
        self.as_real().write(*args, **kwargs)

    def as_virtual(self):
        return self  # always true for VirtualSource — move to parent

    def as_buffer(self):
        return self.as_real().as_buffer()

    def as_real(self):
        """Default: reopen from location with mode.
        Override in modules that need extra constructor args."""
        if self._real_class is None:
            raise ClearMapRuntimeError(f'{self.__class__.__name__} must set _real_class or override as_real()')
        return self._real_class(location=self.location, mode=self._mode)


def element_strides(array):
    """Strides of *array* in items rather than bytes."""
    return tuple(s // array.itemsize for s in array.strides)


class ArraySource(Source):
    """Source whose data is an n-dimensional array.

    Geometry is answered from ``self.array`` by default, which is correct for
    sources that hold their data in memory. Sources that can determine shape,
    dtype or order more cheaply — from a header, say — should override those
    properties with a ``cached_property`` and list them in
    ``_CACHED_PROPERTIES``.
    """

    data_model: ClassVar[str] = 'array'
    _REPR_FIELDS = (ReprFields.NAME, ReprFields.SHAPE, ReprFields.DTYPE, ReprFields.ORDER, ReprFields.LOCATION)

    # ## Data
    @property
    def array(self):
        """The underlying data array."""
        raise ClearMapNotImplementedError(f'{type(self).__name__} does not expose an array.',
                                          operation='array', backend=type(self).__name__)

    # ## Geometry
    @property
    def shape(self):
        return self.array.shape

    @shape.setter
    def shape(self, value):
        raise ClearMapNotImplementedError(f'Cannot set shape on {type(self).__name__}.',
                                          operation='shape', backend=type(self).__name__)

    @property
    def dtype(self):
        return self.array.dtype

    @dtype.setter
    def dtype(self, value):
        raise ClearMapNotImplementedError(f'Cannot set dtype on {type(self).__name__}.',
                                          operation='dtype', backend=type(self).__name__)

    @property
    def order(self):
        return geometry_utils.order(self.array)

    @order.setter
    def order(self, value):
        raise ClearMapNotImplementedError(f'Cannot set order on {type(self).__name__}.',
                                          operation='order', backend=type(self).__name__)

    # ## Derived geometry
    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return int(np.prod(self.shape))

    @property
    def element_strides(self):
        """Strides of the array elements, in items rather than bytes."""
        return element_strides(self.as_buffer())

    @property
    def offset(self):
        """Offset of this source's data within its backing buffer."""
        array = self.as_buffer()
        base = getattr(array, 'base', None)
        if base is None:
            return 0
        return np.byte_bounds(array)[0] - np.byte_bounds(base)[0]

    # ## Item access
    def _getitem(self, slicing):
        return self.as_buffer()[slicing]

    def _setitem(self, slicing, value):
        self.as_buffer()[slicing] = value

    # ## Conversions
    def as_buffer(self):
        return self.array

    def as_memory(self):
        return np.array(self.as_buffer())  # FIXME: check if we need to check instance (mmemmap) to decide array vs asarray cost


###############################################################################
### Table sources
###############################################################################

# Shape and columns only render once the table is loaded, so printing a source
# never triggers a full read of a large file.
TABLE_SHAPE = ReprField('_loaded_shape', convert=lambda s: repr(tuple(s)))
TABLE_COLUMNS = ReprField('_loaded_columns', convert=lambda c: repr(tuple(c)))


class TableSource(Source):
    """Source whose data is a table of named columns stored as a whole-table file.

    Conventions
    -----------
    * Item access follows pandas ``DataFrame.__getitem__``: a column name gives a
      Series, a list of names gives a DataFrame, a slice or a boolean mask selects rows.
    * Tables are read and written whole. There is no item assignment and no edit mode:
      read the frame, modify it, write it back.
    * Columns must be named with strings. An array is not a table: writing one raises
      unless ``columns=`` names its columns, and so does a DataFrame with default
      integer column labels.
    * ``order`` and a global ``dtype`` are deliberately absent; see ``dtypes``.

    Subclasses implement the two hooks ``_load`` and ``_dump`` and set ``backend``.
    The protocol functions of a table backend module are thin wrappers around the
    classmethods ``open_ro``, ``read_table``, ``write_table``, ``create_table`` and
    ``edit``.
    """

    data_model: ClassVar[str] = 'table'
    _CACHED_PROPERTIES = ('frame',)
    _REPR_FIELDS = (ReprFields.NAME, TABLE_SHAPE, TABLE_COLUMNS, ReprFields.LOCATION)

    def __init__(self, location, mode=None, name=None):
        if location is None:
            raise ClearMapValueError(f'{type(self).__name__} requires a location.')
        if mode not in (None, 'r'):
            raise ClearMapValueError(f'{type(self).__name__} only supports mode="r": tables are written whole '
                                     f'with write(), not edited in place.', value=mode, expected="'r' or None")
        super().__init__(name=name, mode='r')
        self.location = location

    # ## Backend hooks
    @classmethod
    def _load(cls, location, **kwargs):
        """Read the whole table at *location* into a DataFrame."""
        raise ClearMapNotImplementedError(f'{cls.__name__} does not implement _load().',
                                          operation='load', backend=cls.__name__)

    @classmethod
    def _dump(cls, frame, location, **kwargs):
        """Write *frame*, which has a default RangeIndex, to *location*."""
        raise ClearMapNotImplementedError(f'{cls.__name__} does not implement _dump().',
                                          operation='dump', backend=cls.__name__)

    # ## Data
    def _require_existing(self):
        if not self.exists():
            raise SourceNotFoundError(location=self.location,
                                      message=f'{type(self).__name__}: no table file at {self.location}')

    @cached_property
    def frame(self):
        """The table, loaded on first access and cached.

        The cached frame is shared by every access through this source; treat it as
        read-only. Use ``read()`` for a copy you own.
        """
        self._require_existing()
        return self._load(self.location)

    def read(self, slicing=None, **load_kwargs):
        """Load the table from disk, optionally selecting with pandas ``[]`` semantics.

        Always reads the file and never touches the cached ``frame``, so the result
        belongs to the caller. ``load_kwargs`` are passed to the backend reader
        (e.g. ``usecols=`` for CSV, ``columns=`` for Feather).
        """
        self._require_existing()
        frame = self._load(self.location, **load_kwargs)
        return frame if slicing is None else frame[slicing]

    def write(self, data, overwrite=True, **dump_kwargs):
        """Replace the table at this source's location with *data*."""
        self.write_table(self, data, overwrite=overwrite, **dump_kwargs)
        return self

    def _getitem(self, key):
        return self.frame[key]

    def __setitem__(self, key, value):
        raise ClearMapNotImplementedError(f'{type(self).__name__} does not support item assignment: tables are '
                                          f'written whole. Read the frame, modify it, and write() it back.',
                                          operation='setitem', backend=type(self).__name__)

    # ## Geometry
    @property
    def columns(self):
        return tuple(self.frame.columns)

    @property
    def dtypes(self):
        """Mapping of column name to dtype."""
        return dict(self.frame.dtypes.items())

    @property
    def n_rows(self):
        return len(self.frame.index)

    @property
    def n_columns(self):
        return len(self.frame.columns)

    @property
    def shape(self):
        """``(n_rows, n_columns)``. Provided for parity, not for layout."""
        return self.n_rows, self.n_columns

    @property
    def ndim(self):
        return 2

    def __len__(self):
        return self.n_rows

    @property
    def _loaded_shape(self):
        return self.shape if 'frame' in self.__dict__ else None

    @property
    def _loaded_columns(self):
        return self.columns if 'frame' in self.__dict__ else None

    # ## Conversions
    def as_memory(self):
        """A copy of the table as a DataFrame, owned by the caller."""
        return self.frame.copy()

    def as_buffer(self):
        raise ClearMapNotImplementedError(f'{type(self).__name__} has no contiguous buffer; use .frame or .read().',
                                          operation='as_buffer', backend=type(self).__name__)

    def as_real(self):
        return self

    def as_virtual(self):
        """A table source is its own lightweight handle (location and mode), so it is its own virtual source."""
        return self

    def __getstate__(self):
        """Pickle as a handle:
        drop the cached frame so sending a source to workers never ships the table."""
        state = self.__dict__.copy()
        for key in self._CACHED_PROPERTIES:
            state.pop(key, None)
        return state

    # ## Backend protocol implementations
    @classmethod
    def open_ro(cls, source_, **kwargs):
        """Open *source_* read-only. Existing sources of this class are returned as is."""
        if isinstance(source_, cls):
            return source_
        if isinstance(source_, Source):
            raise ClearMapValueError(f'Cannot open {source_!r} as a {cls.__name__}.',
                                     value=type(source_).__name__, expected=cls.__name__)
        kwargs.setdefault('mode', 'r')
        return cls(source_, **kwargs)

    @classmethod
    def read_table(cls, source_, slicing=None, **load_kwargs):
        """Read a table as a DataFrame (or a Series, depending on *slicing*)."""
        as_source = load_kwargs.pop('as_source', None)
        source = cls.open_ro(source_)
        if as_source:
            warnings.warn(f'read(..., as_source=True) is deprecated for {cls.__name__}; use open_ro() instead.',
                          DeprecationWarning, stacklevel=3)
            if slicing is not None:
                raise ClearMapValueError('as_source=True cannot be combined with slicing for tables; '
                                         'read the frame and select from it.')
            return source
        return source.read(slicing=slicing, **load_kwargs)

    @classmethod
    def write_table(cls, sink, data=None, slicing=None, overwrite=True, columns=None, **dump_kwargs):
        """Write a complete table to *sink* and return *sink* unchanged.

        *data* is a DataFrame or a TableSource, or an array together with *columns*
        naming its columns (as in ``pd.DataFrame(array, columns=...)``).
        """
        if slicing is not None:
            raise ClearMapNotImplementedError(f'{cls.__name__} does not support sliced writes; write the complete '
                                              f'table instead.', operation='write', backend=cls.__name__)
        if data is None:
            raise ClearMapValueError(f'{cls.__name__} write requires data.')
        if isinstance(sink, Source) and not isinstance(sink, cls):
            raise ClearMapValueError(f'Cannot write a {cls.__name__} table into {sink!r}.',
                                     value=type(sink).__name__, expected=cls.__name__)

        location = sink.location if isinstance(sink, cls) else fu.normalize_location_spec(sink)
        if not overwrite and fu.is_file(location):
            raise FileExistsError(f'Table file already exists: {location}')

        frame = _as_frame(data, columns=columns)  # load before dumping: data may be read from this very file
        cls._dump(_with_default_index(frame), location, **dump_kwargs)

        if isinstance(sink, cls):
            sink._invalidate_cache()
        return sink

    @classmethod
    def create_table(cls, location=None, shape=None, dtype=None, order=None,
                     mode=None, array=None, as_source=True, columns=None, **dump_kwargs):
        """Create a table file from a DataFrame passed as *array* (or an array plus *columns*).

        Blank creation from ``shape`` / ``dtype`` is unsupported: a table has named
        columns, each with its own dtype.
        """
        if location is None:
            raise ClearMapValueError(f'{cls.__name__} create requires a location.')
        if array is None:
            raise ClearMapNotImplementedError(f'A blank {cls.__name__} table cannot be created from shape/dtype; '
                                              f'pass a DataFrame as array= or use write().',
                                              operation='create', backend=cls.__name__)
        if shape is not None or dtype is not None or order is not None:
            raise ClearMapValueError('shape, dtype and order are not meaningful creation arguments for a table.')
        if mode not in (None, 'w+'):
            raise ClearMapValueError(f'{cls.__name__} create only supports mode="w+".',
                                     value=mode, expected="'w+' or None")

        cls.write_table(location, array, overwrite=True, columns=columns, **dump_kwargs)
        source = cls(location)
        return source if as_source else source.read()

    @classmethod
    def edit(cls, source_, **kwargs):
        raise ClearMapNotImplementedError(f'{cls.__name__} has no edit mode: tables are read and written whole. '
                                          f'Use frame = io.read(location), modify it, then io.write(location, frame).',
                                          operation='edit', backend=cls.__name__)


def _as_frame(data, columns=None):
    """Coerce table-like *data* to a DataFrame, refusing anything that is not a table.

    An array becomes a table only when *columns* names its columns. *columns* is
    refused for data that already has column names, rather than guessing whether
    it means renaming or selecting.
    """
    import pandas as pd

    if isinstance(data, (TableSource, pd.DataFrame)):
        if columns is not None:
            raise ClearMapValueError(f'columns= only names the columns of an array; {type(data).__name__} already '
                                     f'has named columns. Rename or select them on the DataFrame instead.',
                                     value=columns, expected=None)
        frame = data.frame if isinstance(data, TableSource) else data
    elif isinstance(data, np.ndarray):
        if columns is None:
            raise ClearMapValueError('Arrays are not tables: pass columns=[...] to name the columns, '
                                     'or build a DataFrame yourself.', value=data.shape, expected='columns=[...]')
        if data.dtype.names is not None:
            raise ClearMapValueError('Structured arrays are not supported as tables; build a DataFrame.',
                                     value=data.dtype, expected='a plain 1-d or 2-d array')
        try:
            frame = pd.DataFrame(data, columns=list(columns))
        except ValueError as err:
            raise ClearMapValueError(f'Cannot name the columns of an array of shape {data.shape} with '
                                     f'{list(columns)!r}: {err}', value=list(columns), expected=None) from err
    else:
        raise ClearMapValueError(f'Expected a DataFrame, a TableSource, or an array with columns=, '
                                 f'got {type(data).__name__}.',
                                 value=type(data).__name__, expected='DataFrame, TableSource or ndarray')

    unnamed = [column for column in frame.columns if not isinstance(column, str)]
    if unnamed:
        raise ClearMapValueError(f'Table columns must be named with strings; got {unnamed!r}. '
                                 f'A DataFrame built from a bare array has integer labels: name the columns.',
                                 value=unnamed, expected='str column names')
    return frame


def _with_default_index(frame):
    """Reduce *frame* to a default RangeIndex, since table files store columns only.

    A named index carries data and becomes a column. An unnamed non-default index is
    positional residue (e.g. from filtering rows) and is dropped.
    """
    import pandas as pd

    if frame.index.equals(pd.RangeIndex(len(frame))):
        return frame
    has_names = any(name is not None for name in frame.index.names)
    return frame.reset_index(drop=not has_names)


GRAPH = ReprField('graph', convert=str)


class GraphSource(Source):
    """Source whose data is a graph."""

    data_model: ClassVar[str] = 'graph'
    _REPR_FIELDS = (ReprFields.NAME, GRAPH, ReprFields.LOCATION)  # FIXME: no Fraph in ReprFields. Move ??

    @property
    def graph(self):
        raise ClearMapNotImplementedError(
            f'{type(self).__name__} does not expose a graph.',
            operation='graph', backend=type(self).__name__)

    @property
    def shape(self):
        return self.graph.shape

    @shape.setter
    def shape(self, value):
        self.graph.shape = value

    @property
    def n_vertices(self):
        return self.graph.n_vertices

    @property
    def n_edges(self):
        return self.graph.n_edges

    def info(self):
        return self.graph.info()

    def as_buffer(self):
        raise ClearMapNotImplementedError(
            f'{type(self).__name__} has no buffer representation.',
            operation='as_buffer', backend=type(self).__name__)


###############################################################################
### Tests
###############################################################################

def _test():
    import ClearMap.IO.source.Source as src
    # reload(src)

    s = src.VirtualSource(shape=(50,50), dtype=float, location='/tmp/test.npy', order='F')
    print(s)

    print(s.size, s.ndim)
