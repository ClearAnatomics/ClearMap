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
from ClearMap.IO.source.backends.NPY import NumpySource

from ClearMap.ParallelProcessing.SharedMemoryArray import base  #analysis:ignore

__all__ = sma.__all__


###############################################################################
### SMASource class
###############################################################################
MEMORY = source_mod.ReprField('memory', '<>')

class SMASource(NumpySource):
    """Shared memory source."""

    def __init__(self, array=None, shape=None, dtype=None, order=None,
               handle=None, name=None, mode=None):
        """Shared memory source constructor."""
        shared = _shared(shape=shape, dtype=dtype, order=order, array=array, handle=handle)
        super().__init__(array=shared, name=name, mode=mode)

        self._handle = handle

    @property
    def base(self):
        return base(self.array)

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

    def as_real(self):
        return self

    def as_buffer(self):
        return self.array


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


# TODO: read directly into shared memory !
# read = npy_source_mod.read
# write = npy_source_mod.write

def read(*args, **kwargs):
    raise NotImplementedError('read not implemented for SharedMemoryArray!')

def write(*args, **kwargs):
    raise NotImplementedError('write not implemented for SharedMemoryArray!')


def create(shape=None, dtype=None, order=None,
           array=None, handle=None, as_source=True, **kwargs):
    """Create a shared memory array.

    Arguments
    ---------
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
    array = _shared(shape=shape, dtype=dtype, order=order, array=array, handle=handle)
    if as_source:
        return SMASource(array=array)
    else:
        return array


###############################################################################
### Helpers
###############################################################################

def _shared(shape = None, dtype = None, order = None, array=None, handle = None):
    if handle is not None:
        array = smm.get(handle)

    # No source data: create an uninitialised shared array.
    if array is None:
        return sma.array(shape=shape, dtype=dtype, order=order)
    elif is_shared(array):
        if shape is None and dtype is None and order is None:
            return array

        shape = shape if shape is not None else array.shape
        dtype = dtype if dtype is not None else array.dtype
        order = order if order is not None else source_mod.order(array)

        if shape != array.shape:
            raise ValueError('Shapes do not match!')

        if np.dtype(dtype) == array.dtype and order == source_mod.order(array):
            return array
        else:
            new = sma.array(shape=shape,dtype=dtype,order=order)
            new[:] = array
            return new
    elif isinstance(array, (np.ndarray, list, tuple)):
        array = np.asarray(array)

        shape = shape if shape is not None else array.shape
        dtype = dtype if dtype is not None else array.dtype
        order = order if order is not None else source_mod.order(array)

        if shape != array.shape:
            raise ValueError('Shapes do not match!')

        new = sma.array(shape=shape,dtype=dtype,order=order)
        new[:] = array
        return new
    else:
        raise ValueError(f'Cannot create shared array from array {array!r}!')


###############################################################################
### Tests
###############################################################################

def _test():
    n = 10
    array = SMA.zeros(n)

    s = SMA.SMASource(array = array)
    print(s)

    v = s.as_virtual()
    print(v)
    s2 = v.open_ro()
