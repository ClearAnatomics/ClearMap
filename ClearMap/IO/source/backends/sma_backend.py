# -*- coding: utf-8 -*-
"""
SMA
===

Shared memory arrays for parallel processing.

Note
----
Usage of this array can help for parallel processing of shared memory
arrays. However, using memmap sources (:mod:`~ClearMap.IO.MMP`) often enable 
faster implementations.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'https://idisco.info'
__download__  = 'https://www.github.com/ChristophKirst/ClearMap2'

import numpy as np

import ClearMap.ParallelProcessing.SharedMemoryArray as sma
import ClearMap.ParallelProcessing.SharedMemoryManager as smm

import ClearMap.IO.source.Source as source_mod
# noinspection PyUnusedImports
from ClearMap.IO.source.backend_defaults import read, write, open_ro
# TODO: read directly into shared memory !
from ClearMap.IO.source.backends.npy_backend import NumpySource


__all__ = sma.__all__

from ClearMap.IO.source.geometry_utils import resolve_geometry, properties_match

from ClearMap.IO.source.protocol import Backend
from ClearMap.Utils.exceptions import ClearMapValueError

###############################################################################
### SMASource class
###############################################################################
MEMORY = source_mod.ReprField('memory', '<>')

class SMASource(NumpySource):
    """Shared memory source."""
    backend = Backend.SMA
    _virtual_class = None

    def __init__(self, array=None, shape=None, dtype=None, order=None,
               handle=None, name=None, mode=None):
        """Shared memory source constructor."""
        shared = _shared(shape=shape, dtype=dtype, order=order, array=array, handle=handle)
        super().__init__(array=shared, name=name, mode=mode)

        self._handle = handle

    @property
    def base(self):
        return sma.base(self.array)

    @property
    def handle(self):
        if self._handle is None:
            self._handle = smm.insert(self.array)
        return self._handle

    @property
    def memory(self):
        return 'shared'

    def free(self):
        if self._handle is not None:
            smm.free(self._handle)
            self._handle = None

    def as_virtual(self):
        return SMAVirtualSource(source=self)


class SMAVirtualSource(source_mod.VirtualSource):
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
    """Returns True if array is a shared memory array

    Arguments
    ---------
    source : array
        The source array to use as template.

    Returns
    -------
    is_shared : bool
        True if the array is a shared memory array.
    """
    if isinstance(source, (SMASource, SMAVirtualSource)):
        return True
    else:
        return sma.is_shared(source)


def as_shared(source):
    """Convert array to a shared memory array

    Arguments
    ---------
    source : array
        The source array to use as template.

    Returns
    -------
    array : array
        A shared memory array wrapped as ndarray based on the source array.
    """
    if isinstance(source, (SMASource, SMAVirtualSource)):
        return source
    elif sma.is_shared(source):
        return SMASource(array=source)
    elif isinstance(source, (list, tuple, np.ndarray)):
        return SMASource(array=sma.as_shared(source))
    else:
        raise ValueError(f'Source {source!r} cannot be transformed to a shared array!')


def create(location=None, shape=None, dtype=None, order=None, mode=None,
           array=None, handle=None, as_source=True, **kwargs):
    """Create a shared memory array.

    Arguments
    ---------
    location: str | None
        This is here only to respect the general protocol.
        Do not use
    shape : tuple or None
        The shape of the memory map to create.
    dtype : dtype
        The data type of the memory map.
    order : 'C', 'F', or None
        The contiguous order of the memmap.
    array : array, Source or None
        Optional source with data to fill the memory map with.
    handle : int or None
        Optional handle to an array from which to create this source.
    as_source : bool
        If True, wrap shaed array in Source class.

    Returns
    -------
    shared : array
        The shared memory array.
    """
    if location is not None:
        raise ClearMapValueError('SMA sources cannot have a filesystem location.')
    array = _shared(shape=shape, dtype=dtype, order=order, array=array, handle=handle)
    if as_source:
        return SMASource(array=array, handle=handle, mode=mode)
    else:
        if mode is not None:
            raise ClearMapValueError('`mode` has no meaning when as_source=False')
        return array


###############################################################################
### Helpers
###############################################################################

def _shared(shape=None, dtype=None, order=None, array=None, handle=None):
    if handle is not None:
        array = smm.get(handle)

    # No source data: create an uninitialised shared array.
    if array is None:
        return sma.array(shape=shape, dtype=dtype, order=order)

    if isinstance(array, SMAVirtualSource):
        array = array.as_real().array
    elif isinstance(array, source_mod.ArraySource):
        array = array.array
    elif isinstance(array, (list, tuple)):
        array = np.asarray(array)

    if not isinstance(array, np.ndarray):
        raise ValueError(f'Cannot create shared array from {array!r}!')

    shape, dtype, order = resolve_geometry(shape=shape, dtype=dtype, order_=order, array=array)

    if shape != array.shape:
        raise ValueError(f'Shapes do not match: requested {shape}, source has {array.shape}.')

    if sma.is_shared(array) and properties_match(array, shape=shape, dtype=dtype, order=order):
        return array

    shared = sma.array(shape=shape, dtype=dtype, order=order)
    shared[:] = array
    return shared


###############################################################################
### Tests
###############################################################################

def _test():
    n = 10
    array = sma.zeros(n)

    s = SMASource(array = array)
    print(s)

    v = s.as_virtual()
    print(v)
    s2 = v.open_ro()
