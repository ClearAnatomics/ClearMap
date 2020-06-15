# -*- coding: utf-8 -*-
"""
SharedMemoryArray
=================

Shared ctype memory arrays.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'

import numpy as np
import multiprocessing as mp

__all__ = ['ctype', 'base', 'empty', 'zeros', 'zeros_like', 'ones']

###############################################################################
### Functionality
###############################################################################

def ctype(dtype):
  """Determine ctype from array or dtype for ctype array construction
  
  Arguments
  ---------
  dtype : array or dtype
    The array or data type to determine the c type from.
  
  Returns
  -------
  ctype : str
    The c-type correspinding to the array or dtype.
  """
  #get dtype in case argument is a ndarray
  if isinstance(dtype, np.ndarray):
    dtype = dtype.dtype;
  #convert to typestr
  a = np.empty(1, dtype = dtype);
  if a.dtype == bool:
    a = a.astype('uint8');
  #typestr = np.ctypeslib.as_ctypes(a[0]).__array_interface__['typestr'];
  typestr = a.__array_interface__['typestr'];
  #map to ctype  
  
  try:
    #numpy 1.14
    ct = np.ctypeslib._typecodes[typestr];  
  except:
    #numpy 1.16
    ct = np.ctypeslib._ctype_from_dtype_scalar(np.dtype(typestr));
   
  return ct;


def base(array):
  """Return the underlying multiprocessing shared raw array from a shared numpy array
  
  Arguments
  ---------
  array : array
    Shared array.

  Returns
  -------
  array : array
    The raw shared memory base array.
  """
  try:
    return array.base.base;
  except:
    raise RuntimeError('Array has no shared base');


def array(shape, dtype = None, order = None):
  """Create a shared array wrapped in numpy array."""
  if dtype is None:
    dtype = float;
    
  if order is None:
    order = 'A';
  
  #create shared memory
  shared = mp.RawArray(ctype(dtype), int(np.prod(shape)));
  
  #wrap with numpy array and reshape
  array = np.frombuffer(shared, dtype=dtype);
  array = array.reshape(shape, order=order);
  
  return array;


def empty(shape, dtype = None, order = None):
  """Creates a empty shared memory array with numpy wrapper
  
  Arguments
  ---------
  shape : tuple of ints
    The shape of the shared memory array to create.
  dtype : array or dtype
    The array or data type to determine the c type from, if None float is used.
  order : C', 'F', or None
    The order of the array.
  
  Returns
  -------
  array : array
    A shared memory array wrapped as ndarray.
  """
  return array(shape=shape, dtype=dtype, order=order);


def zeros(shape, dtype = None, order = None):
  """Creates a shared memory array of zeros with numpy wrapper
  
  Arguments
  ---------
  shape : tuple of ints
    The shape of the shared memory array to create.
  dtype : array or dtype
    The array or data type to determine the c type from, if None float is used.
  order : 'A', 'C', 'F', or None
    The order of the array. If None, 'A' is used.
  
  Returns
  -------
  array : array
    A shared memory array wrapped as ndarray.
  """
  return array(shape, dtype=dtype, order=order);


def zeros_like(source, shape = None,  dtype = None, order = None):
  """Creates a shared memory array with numpy wrapper using shape, dtype and order from source
  
  Arguments
  ---------
  source : array
    The source array to use as template.
  shape : tuple of ints
    The shape of the shared memory array to create.
  dtype : array or dtype
    The array or data type to determine the c type from, if None float is used.
  order : 'A', 'C', 'F', or None
    The order of the array. If None, 'A' is used.
  
  Returns
  -------
  array : array
    A shared memory array wrapped as ndarray basedon the source array.
  """
  
  if dtype is None:
    dtype = source.dtype;
  
  if shape is None:
    shape = source.shape;
  
  if order is None:
    if np.isfortran(source):
      order = 'F';
    else:
      order = 'C';
  
  return array(shape, dtype=dtype, order=order);


def ones(shape, dtype = None, order = None):
  """Creates a shared memory array of ones with numpy wrapper
  
  Arguments
  ---------
  shape : tuple of ints
    The shape of the shared memory array to create.
  dtype : array or dtype
    The array or data type to determine the c type from, if None float is used.
  order : 'A', 'C', 'F', or None
    The order of the array. If None, 'A' is used.
  
  Returns
  -------
  array : array
    A shared memory array wrapped as ndarray.
  """
  a = array(shape, dtype=dtype, order=order);
  a[:] = 1;
  return a;



def is_shared(array):
  """Returns True if array is a shared memory array
  
  Arguments
  ---------
  array : array
    The array to check if it is shared.
   
  Returns
  -------
  is_shared : bool
    True if the array is a shared memory array.
  """
  if not isinstance(array, np.ndarray):
    return False;
  try:
    base = array.base
    if base is None:
      return False
    elif type(base).__module__.startswith('multiprocessing.sharedctypes'):
      return True
    else:
      return is_shared(base)
  except:
      return False


def as_shared(source, copy=False, order=None):
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
  # already a shared array ?
  if not copy and is_shared(source):
    return source
    
  if order is None:
    order = 'A';
  
  a = array(shape=source.shape, dtype=source.dtype, order=order)
  a[:] = source
  
  return a;


###############################################################################
### Tests
###############################################################################

def _test():
  #from importlib import reload
  import numpy as np
  import ClearMap.ParallelProcessing.SharedMemoryArray as sma
  #reload(sma)

  n = 10;
  array = sma.zeros(n)
  non_shared = np.zeros(n)   

  def propagate(arg):
    i, a = arg
    for j in range(1000):
      array[i] = i
      
  def propagate_non_shared(arg):
    i, a = arg
    for j in range(1000):
      non_shared[i] = i
  
  sma.is_shared(non_shared)
  sma.is_shared(array)
  
  pool = sma.mp.Pool(processes=4)
  pp = pool.map(propagate, zip(range(n), [None] * n)); 
  print(array)

  pool = sma.mp.Pool(processes=4)
  pp = pool.map(propagate_non_shared, zip(range(n), [None] * n)); #analysis:ignore
  print(non_shared)
