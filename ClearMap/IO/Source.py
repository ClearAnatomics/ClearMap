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

from ClearMap.Utils.Formatting import ensure
from ClearMap.Utils.exceptions import (ClearMapValueError, ClearMapRuntimeError, ClearMapPermissionError,
                                       ClearMapNotImplementedError)

VALID_MODES         = ('r', 'c', 'r+', 'w+')
EXISTING_FILE_MODES = ('r', 'c', 'r+')
CREATING_MODES      = ('w+',)            # truncate-or-create; one-shot, never stored
READ_ONLY_MODES     = ('r',)
WRITABLE_MODES      = ('c', 'r+', 'w+')  # 'c' accepts writes but does not save
PERSISTABLE_MODES   = ('r+', 'w+')       # writes actually reach disk.  Not 'C' because 'C' is copy-on-write in memory
DEFAULT_READ_MODE   = 'r'
DEFAULT_EDIT_MODE   = 'r+'


def _normalise_order(value):
    value = ensure(value, str).upper()
    if value not in ('C', 'F'):
        raise ClearMapValueError('Invalid order.', value=value, expected=('C', 'F'))
    return value


_NORMALISERS = {'shape': tuple, 'dtype': np.dtype, 'order': _normalise_order}


def order(array):
    """Returns the contiguous order of an array.

    Arguments
    ---------
    array : ndarray or Source

    Returns
    -------
    order : 'C', 'F', None
        None if the array is not contiguous, or if *array* is not something whose order
        can be determined. Note that for shapes with at most one axis longer than 1 both
        orders hold and 'C' is returned arbitrarily; use ``order_is_ambiguous`` before
        treating a mismatch as meaningful.
    """
    if isinstance(array, Source):
        value = array.order
        return _normalise_order(value) if value is not None else None
    elif isinstance(array, np.ndarray):
        if array.flags['C_CONTIGUOUS']:
            return 'C'
        elif array.flags['F_CONTIGUOUS']:
            return 'F'
        else:
            return None
    else:
        return None


def order_is_ambiguous(shape):
    """Whether C and F order are indistinguishable for this shape."""
    return sum(n > 1 for n in tuple(shape)) <= 1


def validate_mode(mode, *, allow_none=False, context=''):
    """Normalise and check a mode string; raises rather than letting np.memmap decide."""
    if mode is None:
        if allow_none:
            return None
        raise ClearMapValueError(f'{context or "mode"}: a mode is required.', value=mode, expected=VALID_MODES)
    mode = ensure(mode, str)
    if mode not in VALID_MODES:
        raise ClearMapValueError(f'{context or "mode"}: invalid mode {mode!r}.', value=mode, expected=VALID_MODES)
    return mode


def mode_after_create(mode):
    """The mode a source must carry *after* a create call has consumed its creation intent.

    ``'w+'`` is a one-shot instruction: retaining it means every later reopen,
    ``as_real()`` or relocation re-truncates the file. Everything else passes through
    unchanged, including ``None``.
    """
    return DEFAULT_EDIT_MODE if mode in CREATING_MODES else mode


def properties_match(source, **properties):
    """True if *source* matches every recognised property given.

    Unrecognised keys and ``None`` values are ignored, so a kwargs bag can be passed
    straight in. Values are normalised before comparison, so ``[10] == (10,)`` and
    ``'f4' == float32``. ``order`` is skipped where the shape makes both orders equally
    true, so a length-1 or 1-d source is never reported as mismatched on order alone.
    """
    for key, normalise in _NORMALISERS.items():
        requested = properties.get(key)
        if requested is None:  # not asked about
            continue
        if key == 'order':
            if order_is_ambiguous(source.shape):
                continue
            actual = order(source)
            if actual is None:  # non-contiguous: no order to compare against
                raise ClearMapValueError(
                    f'Cannot compare order of non-contiguous {source!r}.',
                    value=source, expected='a contiguous source')
        else:
            actual = normalise(getattr(source, key))
        if normalise(requested) != actual:
            return False
    return True


def resolve_geometry(shape=None, dtype=None, order_=None, *,
                     array=None, like=None, default_order=None):
    """Resolve normalised shape, dtype and order from explicit values and templates."""
    if array is not None:
        if shape is None:
            shape = getattr(array, 'shape', None)
        if dtype is None:
            dtype = getattr(array, 'dtype', None)
        if order_ is None:
            order_ = order(array)

    if like is not None:
        if shape is None:
            shape = getattr(like, 'shape', None)
        if dtype is None:
            dtype = getattr(like, 'dtype', None)
        if order_ is None:
            order_ = order(like)

    if order_ is None:
        order_ = default_order

    if shape is not None:
        shape = tuple(shape)
    if dtype is not None:
        dtype = np.dtype(dtype)
    if order_ is not None:
        order_ = _normalise_order(order_)

    return shape, dtype, order_

###############################################################################
### Source base class
###############################################################################

class Source:
    """Base abstract source class."""
    _name: ClassVar[str | None] = None  # override in subclasses as class variable
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
    def shape(self):
        """The shape of the source.

        Returns
        -------
        shape : tuple
            The shape of the source.
        """
        return None

    @shape.setter
    def shape(self, value):
        raise ValueError('Cannot set shape for this source.')

    @property
    def dtype(self):
        """The data type of the source.

        Returns
        -------
        dtype : dtype
            The data type of the source.
        """
        return None

    @dtype.setter
    def dtype(self, value):
        raise ClearMapValueError('Cannot set dtype for this source.')  # FIXME: Move to NotImplementedError

    @property
    def order(self):
        """The contiguous order of the underlying data array.

        Returns
        -------
        order : str
            Returns 'C' for C contiguous and 'F' for Fortran contiguous, None otherwise.
        """
        return None

    @order.setter
    def order(self, value):
        raise ValueError('Cannot set order for this source.')

    @property
    def location(self):
        """The location where the data of the source is stored.

        Returns
        -------
        location : str or None
            Returns the location of the data source or None if this source lives in memory only.
        """
        return None

    @location.setter
    def location(self, value):
        raise ValueError('Cannot set location for this source.')

    ### Derived properties
    @property
    def ndim(self):
        """The number of dimensions of the source.

        Returns
        -------
        ndim : int
            The number of dimension of the source.
        """
        return len(self.shape)

    @property
    def size(self):
        """The size of the source.

        Returns
        -------
        size : int
            The number of data items in the source.
        """
        return np.prod(self.shape)

    ### Functionality
    def exists(self):
        if self.location is not None:
            return fu.is_file(self.location)
        else:
            return False

    ### Source conversions
    def as_virtual(self):
        """Return virtual source without array data to pass in parallel processing.

        Returns
        -------
        source : Source class
            The source class without array data.
        """
        # return VirtualSource(source = self.source)
        raise NotImplementedError('virtual source not implemented for this source!')

    def as_real(self):
        return self

    def as_buffer(self):
        raise NotImplementedError('buffer not implemented for this source!')

    def as_memory(self):
        return np.array(self.as_buffer())

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
            return self.location is None  # FIXME: check if self.mode or self._mode
        return False

    def __getitem__(self, *args):
        raise KeyError('No getitem routine for this source!')

    def __setitem__(self, *args):
        if not self.is_writable:
            raise ClearMapPermissionError(f'Source {self} was opened read-only (mode="r"). '
                                          f'Use io.edit() to open for writing.')
        if self.location is not None and not self.is_persistable:
            raise ClearMapPermissionError(f'Source {self} was opened with mode={self._mode!r}; '
                                          f'writes would not be persisted to {self.location}. '
                                          f'Use io.edit() to open for writing.')
        raise KeyError('No setitem routine for this source!')

    def read(self, *args, **kwargs):
        raise KeyError('No read routine for this source!')

    def write(self, *args, **kwargs):
        raise KeyError('No write routine for this source!')

    ### Formatting
    def __str__(self):
        try:
            name = self.name
            name = f'{name}' if name is not None else ''
        except:
            name =''

        try:
            shape = self.shape
            shape ='%r' % ((shape,)) if shape is not None else ''
        except:
            shape = ''

        try:
            dtype = self.dtype
            dtype = f'[{dtype}]' if dtype is not None else ''
        except:
            dtype = ''

        try:
            order = self.order
            order = f'|{order}|' if order is not None else ''
        except:
            order = ''

        try:
            location = self.location
            location = '%s' % location if location is not None else ''
            if len(location) > 100:
                location = location[:50] + '...' + location[-50:]
            if len(location) > 0:
                location = '{%s}' % location
        except:
            # print('location')
            location = ''

        # try:
        #     array = self.array.__str__()
        #     if len(array) > 100:
        #         e = array[100:].find('\n')
        #         if e != -1:
        #             array = array[:100 + e] + '...'
        #     if len(array) > 0:
        #         array = '\n' + array
        # except:
        #     array = ''

        return name + shape + dtype + order + location  # + array

    def __repr__(self):
        return self.__str__()


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
            raise ValueError("Order %r not in [None, 'C' or 'F']!" % value)
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


# Module level API

def create(location=None, shape=None, dtype=None, order=None, mode=None,
           array=None, as_source=True, **kwargs):
    """Create a source.

    This generic implementation exists as the default operation for backend
    modules that do not support source creation.
    """
    raise ClearMapNotImplementedError('Creating sources is not implemented by this backend.')


def open_ro(source_, **kwargs):
    """Open a source read-only.

    Backends that cannot provide read-only access should inherit this default
    implementation.
    """
    raise ClearMapNotImplementedError('Opening sources read-only is not implemented by this backend.')


def read(source_, **kwargs):
    """Read data from a source.

    Backends that do not support reading should inherit this default
    implementation.
    """
    raise ClearMapNotImplementedError('Reading sources is not implemented by this backend.')


def write(sink, data=None, slicing=None, overwrite=False, **kwargs):
    """Write data to a source.

    Backends that do not support writing should inherit this default
    implementation.
    """
    raise ClearMapNotImplementedError('Writing sources is not implemented by this backend.')



###############################################################################
### Tests
###############################################################################

def _test():
    import ClearMap.IO.Source as src
    # reload(src)

    s = src.VirtualSource(shape=(50,50), dtype=float, location='/tmp/test.npy', order='F')
    print(s)

    print(s.size, s.ndim)
