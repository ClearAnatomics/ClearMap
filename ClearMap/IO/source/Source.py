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
from typing import ClassVar

import numpy as np

import ClearMap.IO.FileUtils as fu
from ClearMap.IO.source import geometry_utils
from ClearMap.IO.source.source_modes import WRITABLE_MODES, PERSISTABLE_MODES

from ClearMap.Utils.Formatting import ensure
from ClearMap.Utils.exceptions import (ClearMapValueError, ClearMapRuntimeError, ClearMapPermissionError,
                                       ClearMapNotImplementedError)


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
    backend: ClassVar['BackendName | None'] = None

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
        self._location = ensure(location, str)

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

    @property
    def location(self):
        """The location of the source's data.

        Returns
        -------
        location : str or None
            Returns the location of the data source or None if there is none.
        """
        return self._location

    @location.setter
    def location(self, value):
        self._location = ensure(value, str)

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
        raise ClearMapNotImplementedError(
            f'{type(self).__name__} does not expose an array.',
            operation='array', backend=type(self).__name__)

    # ## Geometry
    @property
    def shape(self):
        return self.array.shape

    @shape.setter
    def shape(self, value):
        raise ClearMapNotImplementedError(
            f'Cannot set shape on {type(self).__name__}.',
            operation='shape', backend=type(self).__name__)

    @property
    def dtype(self):
        return self.array.dtype

    @dtype.setter
    def dtype(self, value):
        raise ClearMapNotImplementedError(
            f'Cannot set dtype on {type(self).__name__}.',
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


class TableSource(Source):
    """Source whose data is a table of named columns.

    ``order`` is deliberately absent: memory layout is not a meaningful
    property of a table, and ``dtype`` is per-column rather than global.
    """

    data_model: ClassVar[str] = 'table'
    _REPR_FIELDS = (ReprFields.NAME, ReprFields.SHAPE, ReprFields.COLUMNS, ReprFields.LOCATION)

    @property
    def frame(self):
        """The table as a pandas DataFrame."""
        raise ClearMapNotImplementedError(
            f'{type(self).__name__} does not expose a frame.',
            operation='frame', backend=type(self).__name__)

    @property
    def columns(self):
        return tuple(self.frame.columns)

    @property
    def dtypes(self):
        """Mapping of column name to dtype."""
        return {name: dtype for name, dtype in self.frame.dtypes.items()}

    @property
    def n_rows(self):
        return len(self.frame.index)

    @property
    def n_columns(self):
        return len(self.columns)

    @property
    def shape(self):
        """``(n_rows, n_columns)``. Provided for parity, not for layout."""
        return self.n_rows, self.n_columns

    @property
    def ndim(self):
        return 2

    def __len__(self):
        return self.n_rows

    def _getitem(self, slicing):
        return self.frame[slicing]

    def as_memory(self):
        return self.frame.to_numpy()

    def as_buffer(self):
        raise ClearMapNotImplementedError(
            f'{type(self).__name__} has no contiguous buffer; use .frame '
            f'or .as_memory().', operation='as_buffer',
            backend=type(self).__name__)


class GraphSource(Source):
    """Source whose data is a graph."""

    data_model: ClassVar[str] = 'graph'
    _REPR_FIELDS = (ReprFields.NAME, ReprFields.GRAPH, ReprFields.LOCATION)  # FIXME: no Fraph in ReprFields. Move ??

    @property
    def graph(self):
        raise ClearMapNotImplementedError(
            f'{type(self).__name__} does not expose a graph.',
            operation='graph', backend=type(self).__name__)

    @property
    def shape(self):
        return self.graph.shape

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
