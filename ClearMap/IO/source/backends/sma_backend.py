# -*- coding: utf-8 -*-
"""
SMA
===

Backend for shared memory arrays, used to hand arrays to worker processes
without copying them.

A :class:`SMASource` wraps a numpy array living in shared memory. Its virtual
counterpart, :class:`SMAVirtualSource`, carries only a handle into the shared
memory manager, so it pickles cheaply and every worker maps the same buffer.

Note
----
For data that already lives on disk, memory-mapped sources
(:mod:`~ClearMap.IO.source.backends.mmp_backend`) often allow faster
implementations.

The low-level shared memory helpers live in
:mod:`ClearMap.ParallelProcessing.SharedMemoryArray` and
:mod:`ClearMap.ParallelProcessing.SharedMemoryManager`: import them from there.
They are imported privately here and are not part of this module's interface.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'https://idisco.info'
__download__  = 'https://www.github.com/ChristophKirst/ClearMap2'

import numpy as np

import ClearMap.ParallelProcessing.SharedMemoryArray as _shared_array
from ClearMap.ParallelProcessing.SharedMemoryManager import SharedMemmoryManager

import ClearMap.IO.source.Source as source_mod
# noinspection PyUnusedImports
from ClearMap.IO.source.backend_defaults import read, write, open_ro  # protocol functions
from ClearMap.IO.source.backends.npy_backend import NumpySource
from ClearMap.IO.source.geometry_utils import resolve_geometry, properties_match
from ClearMap.IO.source.protocol import Backend

from ClearMap.Utils.exceptions import ClearMapValueError


__all__ = ['SMASource', 'SMAVirtualSource', 'SOURCE_CLASS',
           'is_shared', 'memory', 'as_shared',
           'create', 'read', 'write', 'open_ro']


###############################################################################
### SMASource class
###############################################################################

MEMORY = source_mod.ReprField('memory', '<>')


class SMASource(NumpySource):
    """Source backed by a numpy array in shared memory."""

    backend = Backend.SMA
    _REPR_FIELDS = NumpySource._REPR_FIELDS + (MEMORY,)

    def __init__(self, array=None, shape=None, dtype=None, order=None,
                 handle=None, name=None, mode=None):
        """Shared memory source constructor.

        Arguments
        ---------
        array : array, Source or None
            Data to share. Used as is if already in shared memory with the requested
            geometry, copied into shared memory otherwise.
        shape, dtype, order :
            Geometry of the array to create when *array* is None.
        handle : int or None
            Handle of an array already registered in the shared memory manager.
        """
        shared = _shared(shape=shape, dtype=dtype, order=order, array=array, handle=handle)
        super().__init__(array=shared, name=name, mode=mode)

        self._handle = handle

    @property
    def base(self):
        """The raw multiprocessing buffer underlying the array."""
        return _shared_array.base(self.array)

    @property
    def handle(self):
        """Handle of this array in the shared memory manager, registered on first access."""
        if self._handle is None:
            self._handle = SharedMemmoryManager.insert(self.array)
        return self._handle

    @property
    def memory(self):
        return 'shared'

    def free(self):
        """Release this array's handle in the shared memory manager."""
        if self._handle is not None:
            SharedMemmoryManager.free(self._handle)
            self._handle = None

    def as_virtual(self):
        # Must override NumpySource.as_virtual, which returns self: workers need the handle,
        # not a pickled copy of the array (writes to a copy would be silently lost).
        return SMAVirtualSource(source=self)


class SMAVirtualSource(source_mod.VirtualSource):
    """Picklable handle to an :class:`SMASource`: carries the manager handle, not the data."""

    _real_class = SMASource

    def __init__(self, source=None, shape=None, dtype=None, order=None,
                 handle=None, name=None, mode=None):
        super().__init__(source=source, shape=shape, dtype=dtype, order=order, name=name, mode=mode)
        if handle is None and source is not None:
            handle = source.handle
        self._handle = handle

    @property
    def handle(self):
        return self._handle

    def as_real(self):
        return self._real_class(handle=self.handle, mode=self._mode)


SOURCE_CLASS = SMASource


###############################################################################
### IO Interface
###############################################################################

def is_shared(source):
    """Returns True if *source* is a shared memory array or source.

    Arguments
    ---------
    source : array or Source
        The source to check.

    Returns
    -------
    is_shared : bool
        True if the data lives in shared memory.
    """
    if isinstance(source, (SMASource, SMAVirtualSource)):
        return True
    return _shared_array.is_shared(source)


def memory(source):  # FIXME: rename. Not descriptive (also update the ClearMap.IO.IO shim entry)
    """Returns the memory type of a source: 'shared' for shared-memory arrays, else None.

    Arguments
    ---------
    source : array or Source
        The source to check.

    Returns
    -------
    memory : 'shared' or None
        The memory type of the source.
    """
    return 'shared' if is_shared(source) else None


def as_shared(source):
    """Returns *source* as a shared memory source, copying it into shared memory if needed.

    Arguments
    ---------
    source : array, list, tuple or shared memory Source
        The data to share.

    Returns
    -------
    source : SMASource or SMAVirtualSource
        *source* itself if it is already a shared memory source, else a new SMASource.
    """
    if isinstance(source, (SMASource, SMAVirtualSource)):
        return source
    if _shared_array.is_shared(source):
        return SMASource(array=source)
    if isinstance(source, (list, tuple, np.ndarray)):
        return SMASource(array=_shared_array.as_shared(np.asarray(source)))
    raise ClearMapValueError(f'Source {source!r} cannot be transformed to a shared array!',
                             value=type(source).__name__, expected='array, list, tuple or SMASource')


def create(location=None, shape=None, dtype=None, order=None, mode=None,
           array=None, handle=None, as_source=True, **kwargs):
    """Create a shared memory array.

    Arguments
    ---------
    location : None
        Accepted for protocol compatibility only; shared memory has no location.
    shape : tuple or None
        The shape of the array to create.
    dtype : dtype or None
        The data type of the array.
    order : 'C', 'F', or None
        The contiguous order of the array.
    mode : str or None
        The mode of the returned source.
    array : array, Source or None
        Optional data to fill the array with.
    handle : int or None
        Optional handle of an array already registered in the shared memory manager.
    as_source : bool
        If True, return an SMASource, else the bare shared array.

    Returns
    -------
    shared : SMASource or array
        The shared memory array.
    """
    if location is not None:
        raise ClearMapValueError('SMA sources cannot have a filesystem location.', value=location, expected=None)
    shared = _shared(shape=shape, dtype=dtype, order=order, array=array, handle=handle)
    if as_source:
        return SMASource(array=shared, handle=handle, mode=mode)
    else:
        if mode is not None:
            raise ClearMapValueError('`mode` has no meaning when as_source=False', value=mode, expected=None)
        return shared


###############################################################################
### Helpers
###############################################################################

def _shared(shape=None, dtype=None, order=None, array=None, handle=None):
    """Return a shared array with the requested geometry, reusing *array* when it already fits."""
    if handle is not None:
        array = SharedMemmoryManager.get(handle)
        if array is None:
            raise ClearMapValueError(f'No shared array registered under handle {handle} in this process '
                                     f'(freed, or the worker was not started by fork).',
                                     value=handle, expected='a live handle')

    if array is None:  # No data: create an uninitialised shared array
        return _shared_array.array(shape=shape, dtype=dtype, order=order)

    if isinstance(array, SMAVirtualSource):
        array = array.as_real().array
    elif isinstance(array, source_mod.ArraySource):
        array = array.array
    elif isinstance(array, (list, tuple)):
        array = np.asarray(array)

    if not isinstance(array, np.ndarray):
        raise ClearMapValueError(f'Cannot create shared array from {array!r}!',
                                 value=type(array).__name__, expected='array, list, tuple or array Source')

    shape, dtype, order = resolve_geometry(shape=shape, dtype=dtype, order_=order, array=array)

    if shape != array.shape:
        raise ClearMapValueError(f'Shapes do not match: requested {shape}, source has {array.shape}.',
                                 value=shape, expected=array.shape)

    if _shared_array.is_shared(array) and properties_match(array, shape=shape, dtype=dtype, order=order):
        return array

    shared = _shared_array.array(shape=shape, dtype=dtype, order=order)
    shared[:] = array
    return shared


###############################################################################
### Tests
###############################################################################

def _test():
    source = SMASource(array=_shared_array.zeros(10))
    print(source)

    virtual = source.as_virtual()
    print(virtual)

    assert isinstance(virtual, SMAVirtualSource)
    real = virtual.as_real()
    assert np.shares_memory(real.array, source.array)
    source.free()
