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
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'

import numpy as np

import ClearMap.ParallelProcessing.SharedMemoryArray as sma
import ClearMap.ParallelProcessing.SharedMemoryManager as smm

import ClearMap.IO.Source as src
import ClearMap.IO.NPY as npy

from ClearMap.ParallelProcessing.SharedMemoryArray import base, ctype, empty      #analysis:ignore 
from ClearMap.ParallelProcessing.SharedMemoryArray import zeros, zeros_like, ones #analysis:ignore

__all__ = sma.__all__;

###############################################################################
### Source class
###############################################################################

class Source(npy.Source):
  """Shared memory source."""

  def __init__(self, array = None, shape = None, dtype = None, order = None, handle = None, name = None):
    """Shared memory source constructor."""
    shared = _shared(shape=shape, dtype=dtype, order=order, array=array, handle=handle);
    super(Source,self).__init__(array=shared, name=name);
    
    self._handle = handle;
  
  @property
  def name(self):
    return "Shared-Source";
  
  @property
  def base(self):
    return base(self.array);
    
  @property
  def handle(self):
    if self._handle is None:
      self._handle = smm.insert(self.array);
    return self._handle;
    
  @property
  def memory(self):
    return 'shared'

  def free(self):
    if self._handle is not None:
      smm.free(self._handle);
      self._handle = None;
  
  def as_virtual(self):
    return VirtualSource(source = self);
  
  def as_real(self):
    return self;
  
  def as_buffer(self):
    return self.array;
    

class VirtualSource(src.VirtualSource):
  def __init__(self, source = None, shape = None, dtype = None, order = None, handle = None, name = None):
    super(VirtualSource, self).__init__(source=source, shape=shape, dtype=dtype, order=order, name=name);
    if handle is None and source is not None:
      handle = source.handle;
    self._handle = handle;
      
  @property 
  def name(self):
    return 'Virtual-Shared-Source';
  
  @property
  def handle(self):
    return self._handle;
    
  def as_virtual(self):
    return self;
  
  def as_real(self):
    return Source(handle=self.handle);
  
  def as_buffer(self):
    return self.as_real().as_buffer();
  

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
  if isinstance(source, (Source, VirtualSource)):
    return True;
  else:
    return sma.is_shared(source);


def as_shared(source):
  """Convert array to a shared memory array
  
  Arguments
  ---------
  source : array
    The source array to use as template.
  copy : bool
    If True, the data in source is copied.
  order : C', 'F', or None
    The order to use for an array if copied or not a shared array. If None, the order of the source is used.

  Returns
  -------
  array : array
    A shared memory array wrapped as ndarray based on the source array.
  """
  if isinstance(source, (Source, VirtualSource)):
    return source;
  elif sma.is_shared(source):
    return Source(array=source);
  elif isinstance(source, (list, tuple, np.ndarray)):
    return Source(array=sma.as_shared(source));
  else:
    raise ValueError('Source %r cannot be transforemd to a shared array!' % source);

#TODO: read directly into shared memory !
#read = npy.read;
#write = npy.write;

def read(*args, **kwargs):
  raise NotImplementedError('read not implemented for SharedMemoryArray!')

def write(*args, **kwargs):
  raise NotImplementedError('write not implemented for SharedMemoryArray!')


def create(shape = None, dtype = None, order = None, array = None, handle = None, as_source = True, **kwargs):
  """Create a shared memory array.
  
  Arguments
  ---------
  location : str
    The filename of the memory mapped array.
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
  array = _shared(shape=shape, dtype=dtype, order=order, array=array, handle=handle);
  if as_source:
    return Source(array=array);
  else:
    return array;


###############################################################################
### Helpers
###############################################################################

def _shared(shape = None, dtype = None, order = None, array=None, handle = None):
  if handle is not None:
    array = smm.get(handle);
  
  if array is None:
    return sma.array(shape=shape, dtype=dtype, order=order);
  
  elif is_shared(array):
    if shape is None and dtype is None and order is None:
      return array;
    
    shape = shape if shape is not None else array.shape;
    dtype = dtype if dtype is not None else array.dtype;
    order = order if order is not None else npy.order(array);
    
    if shape != array.shape:
      raise ValueError('Shapes do not match!');
    
    if np.dtype(dtype) == array.dtype and order == npy.order(array):
      return array;
    else:
      new = sma.array(shape=shape,dtype=dtype,order=order);
      new[:] = array;
      return new;
  
  elif isinstance(array, (np.ndarray, list, tuple)):
    array = np.asarray(array);
    
    shape = shape if shape is not None else array.shape;
    dtype = dtype if dtype is not None else array.dtype;
    order = order if order is not None else npy.order(array);
    
    if shape != array.shape:
      raise ValueError('Shapes do not match!');
    
    new = sma.array(shape=shape,dtype=dtype,order=order);
    new[:] = array;
    return new;
  
  else:
    raise ValueError('Cannot create shared array from array %r!' % array);



###############################################################################
### Tests
###############################################################################

def _test():
  #from importlib import reload
  #import numpy as np  #analysis:ignore
  import ClearMap.IO.SMA as sma

  n = 10;
  array = sma.zeros(n)
  
  s = sma.Source(array = array)
  print(s)
  
  v = s.as_virtual();
  print(v)
  s2 = v.as_source()
  