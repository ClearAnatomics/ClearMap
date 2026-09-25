# -*- coding: utf-8 -*-
"""
NPY
===

IO interface to numpy arrays.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'https://idisco.info'
__download__  = 'https://www.github.com/ChristophKirst/ClearMap2'

import warnings

import numpy as np

import ClearMap.IO.source.Source as source_mod
from ClearMap.IO.source import geometry_utils
from ClearMap.IO.source.protocol import Backend

from ClearMap.Utils.exceptions import ClearMapValueError


###############################################################################
### NumpySource class
###############################################################################

class NumpySource(source_mod.ArraySource):
    """Numpy array source."""
    backend = Backend.NPY

    def __init__(self, array=None, shape=None, dtype=None,
                 order=None, name=None, mode=None):
        """Numpy source class constructor.

        Arguments
        ---------
        array : array
            The underlying data array of this source.
        """
        super().__init__(name=name, mode=mode)
        self._array = _array(shape=shape, dtype=dtype, order=order, array=array)

    def __getattr__(self, name):
        # numpy attributes
        if name != '_array' and hasattr(self, '_array') and hasattr(self._array, name):
            return getattr(self._array, name)
        else:
            raise AttributeError(f'Not such attribute {name!r}!')

    @property
    def array(self):
        """The underlying data array.

        Returns
        -------
        array : array
            The underlying data array of this source.
        """
        return self._array

    @array.setter
    def array(self, value):
        self._array = _array(array=value)

    @property
    def shape(self):
        """The shape of the source.

        Returns
        -------
        shape : tuple
            The shape of the source.
        """
        return self._array.shape

    @shape.setter
    def shape(self, value):
        self._array.shape = value

    @property
    def dtype(self):
        """The data type of the source.

        Returns
        -------
        dtype : dtype
            The data type of the source.
        """
        return self._array.dtype

    @dtype.setter
    def dtype(self, value):
        self._array = np.asarray(self._array, dtype=value)

    @property
    def order(self):
        """The order of how the data is stored in the source.

        Returns
        -------
        order : str
            Returns 'C' for C contigous and 'F' for fortran contigous, None otherwise.
        """
        return geometry_utils.order(self.array)

    @order.setter
    def order(self, value):
        self._array = np.asarray(self._array, order = value)

    ### Parallel processing
    def as_virtual(self):
        # TODO: convert to shared memory array ? -> needs to be implemented to make block processing work for in memory  numpy arrays !
        return self

    # ## Backend protocol implementations (open_ro: ArraySource default)
    @classmethod
    def read_array(cls, source_, slicing=None, as_source=None, as_array=None, **kwargs):
        """Read with the legacy NPY rules: what goes in decides what comes out.

        * A NumpySource reads as itself, or as its array if *as_array*. With *slicing*
          it reads as the sliced ndarray.
        * An ndarray, list or tuple (cast with ``np.asarray``) reads as an ndarray, or
          as a new NumpySource if *as_source*.

        Other keyword arguments (e.g. ``processes``) are accepted and ignored.
        """
        if isinstance(source_, NumpySource):
            if slicing is not None:
                return source_[slicing]  # an ndarray (the old code then crashed on as_array=True)
            return source_.array if as_array else source_
        data = super().read_array(source_, slicing=slicing)  # casts lists and tuples via the constructor
        return cls(array=data) if as_source else data

    @classmethod
    def write_array(cls, sink, data, slicing=None, **kwargs):
        """Write *data* into an array or array Source; with no sink, return the (sliced) data."""
        if sink is None:  # legacy: io_ops.write(None, data) dispatches here
            return data[() if slicing is None else slicing]
        return super().write_array(sink, data, slicing=slicing, **kwargs)

    @classmethod
    def create_array(cls, location=None, shape=None, dtype=None, order=None,
                     mode=None, array=None, as_source=True, **kwargs):
        """Create an in-memory array, blank from *shape* or conformed from *array*.

        *mode* and other keyword arguments are accepted and ignored, as before (FIXME:
        callers such as ``source_initialization.initialize`` forward options meant for
        file backends). The source is created without a mode: in memory, writable.
        """
        if location is not None:
            raise ClearMapValueError('In-memory numpy arrays have no location.', value=location, expected=None)
        array = _array(shape=shape, dtype=dtype, order=order, array=array)
        return cls(array=array) if as_source else array

    @classmethod
    def edit(cls, source_, **kwargs):
        """An in-memory array is always editable: wrap it (no copy), or return the source as is."""
        if isinstance(source_, cls):
            return source_
        return cls(source_, **kwargs)


SOURCE_CLASS = NumpySource

###############################################################################
### Functionality
###############################################################################

def order(array):
    warnings.warn('NPY.order is deprecated; use Source.order instead.',
                  DeprecationWarning, stacklevel=2)
    return geometry_utils.order(array)


###############################################################################
### IO Interface
###############################################################################

def is_numpy(source):
    if isinstance(source, (NumpySource, np.ndarray, list, tuple)):
        return True
    # elif isinstance(source, str): # and fu.file_extension(source) == 'npy':
    #     return True
    else:
        return False


def open_ro(source_, **kwargs):
    return NumpySource.open_ro(source_, **kwargs)


def read(source_, slicing=None, **kwargs):
    return NumpySource.read_array(source_, slicing=slicing, **kwargs)


# TODO: add processes keyword for parallel writing
def write(sink, data, slicing=None, **kwargs):
    return NumpySource.write_array(sink, data, slicing=slicing, **kwargs)


def create(shape=None, dtype=None, order=None, array=None, as_source=True, **kwargs):
    """Create a numpy array, or a NumpySource holding it.

    Arguments
    ---------
    shape : tuple or None
        The shape of the memory map to create.
    dtype : dtype
        The data type of the memory map.
    order : 'C', 'F', or None
        The contiguous order of the memmap.
    array : array, Source or None
        Optional source with data to fill the numpy array with.
    as_source : bool
        If True, return as Source class.

    Other keyword arguments, including *location* and *mode*, are accepted and ignored,
    as before (FIXME: callers such as ``source_initialization.initialize`` forward
    options meant for file backends).
    """
    return NumpySource.create_array(shape=shape, dtype=dtype, order=order, array=array, as_source=as_source)


def edit(source_, **kwargs):
    return NumpySource.edit(source_, **kwargs)


###############################################################################
### Helpers
###############################################################################


def _array(shape=None, dtype=None, order=None, array=None):
    """Create a numpy array, or conform an existing one to the requested geometry.

    Arguments
    ---------
    shape : tuple or None
        The shape of the array. Inferred from *array* if None.
    dtype : dtype or None
        The data type of the array. Inferred from *array* if None.
    order : 'C', 'F', or None
        The contiguous order of the array. Inferred from *array* if None.
    array : array, list, tuple or None
        Optional data. Returned as is when it already matches the requested
        geometry; converted (copied) otherwise.

    Returns
    -------
    array : np.ndarray
        The array.
    """
    if array is None:
        if shape is None:
            raise ClearMapValueError('Cannot create an array without a shape.')
        shape, dtype, order = geometry_utils.resolve_geometry(shape, dtype, order)
        return np.zeros(shape, dtype=dtype, order=order)

    if isinstance(array, (list, tuple)):
        array = np.asarray(array, order=order, dtype=dtype)
    if not isinstance(array, np.ndarray):
        raise ClearMapValueError(f'Cannot create a numpy array from {type(array).__name__}.',
                                 value=type(array).__name__, expected='ndarray, list or tuple')

    shape, dtype, order = geometry_utils.resolve_geometry(shape, dtype, order, array=array)
    if shape != array.shape:
        raise ClearMapValueError(f'Requested shape {shape} does not match array shape {array.shape}.',
                                 value=shape, expected=array.shape)

    if _matches(array, dtype, order):
        return array
    return np.asarray(array, dtype=dtype, order=order)


def _matches(array, dtype, order):
    """True if *array* already has *dtype* and *order* (None means "don't care")."""
    if order is not None and geometry_utils.order(array) is None:
        return False  # non-contiguous: any requested order needs a copy
    return geometry_utils.properties_match(array, dtype=dtype, order=order)


###############################################################################
### Tests
###############################################################################

def _test():
    import numpy as np
    from ClearMap.IO.source.backends.npy_backend import NumpySource
    # reload(npy)

    s = NumpySource(array=np.zeros((5, 7)))
    print(s)

    import ClearMap.IO.source.Slice as slc
    t = slc.Slice(source= s, slicing= (1,))
    print(t)

    v = t.as_virtual()
    print(v)

    x = np.ones(250*1000*1000)
    xs = NumpySource(array=x)

    print(xs)

    del x
    del xs
