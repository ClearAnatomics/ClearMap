#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Shared memory numpy arrays for parallel processing
"""
__author__    = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__   = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__copyright__ = 'Copyright (c) 2017 by Christoph Kirst, The Rockefeller University, New York City'


import multiprocessing
import numpy as np

#cTypeClasses = [c for c in dir(ctypes) if len(c) > 2 and c[:2] =='c_'];
#cTypeClasses = [getattr(ctypes, c) for c in cTypeClasses];
#cTypeClasses = [c for c in cTypeClasses if hasattr(c, '_type_')];
#cTypeNames = [getattr(c, '_type_') for c in cTypeClasses];
#cTypeDict = {c:n for c,n in zip(cTypeClasses, cTypeNames)}

def cType(dtype):
  """Determine ctype from array or dtype for ctype array construction"""
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
  ctype = np.ctypeslib._typecodes[typestr];  
  
  return ctype;


def create(shape, dtype = None, order = 'A'):
  """Creates a shared memory array with numpy wrapper"""
  #default dtype is float
  if dtype is None:
    dtype = float;
  #create shared memory
  shared = multiprocessing.RawArray(cType(dtype), np.prod(shape));
  #wrap with numpy array and reshape
  #shared_arr = np.ctypeslib.as_array(shared_obj)
  shared_arr = np.frombuffer(shared, dtype = dtype);
  shared_arr = shared_arr.reshape(shape, order = order);
  
  return shared_arr;


def zeros(shape, dtype = None, order = 'A'):
  return create(shape, dtype = dtype, order = order);


def ones(shape, dtype = None, order = 'A'):
  arr = create(shape, dtype = dtype, order = order);
  arr[:] = 1;
  return arr;


def create_like(source, dtype = None, shape = None, order = None):
  """Creates a shared memory array with numpy wrapper using shape, dtype and order from source"""
  
  if dtype is None:
    dtype = source.dtype;
  
  if shape is None:
    shape = source.shape;
  
  if order is None:
    if np.isfortran(source):
      order = 'F';
    else:
      order = 'C';
  
  return create(shape, dtype = dtype, order = order);


def isShared(array):
  """Returns True if array is a shared memory array"""
  try:
    base = array.base
    if base is None:
      return False
    elif type(base).__module__.startswith('multiprocessing.sharedctypes'):
      return True
    else:
      return isShared(base)
  except:
      return False


def asShared(array, copy = False, order = 'A'):
  """Convert array to a shared memory array"""
  # first check to see if it already a shared array
  if not copy and isShared(array):
      return array
  # get ctype from numpy array
  ctype = np.ctypeslib._typecodes[array.__array_interface__['typestr']]
  # create shared ctypes object
  shared_obj = multiprocessing.RawArray(ctype, array.size)
  # create numpy array from shared object
  #shared_arr = np.ctypeslib.as_array(shared_obj)
  shared_arr = np.frombuffer(shared_obj, dtype=array.dtype)
  shared_arr = np.reshape(shared_arr, array.shape, order = order)
  # copy data to shared array
  shared_arr[:] = array[:]
  return shared_arr


def base(array):
  """Return the underlying multiprocessing shared raw array from a shared numpy array""" 
  if isShared(array):
    return array.base.base;
  else:
    raise RuntimeError('Array not shared and no shared base');


if __name__ == '__main__':
  #from importlib import reload
  import numpy as np
  import ClearMap.ParallelProcessing.SharedMemory as shm
  reload(shm)

  n = 10;
  array = shm.create(n, dtype='float')
  non_shared = np.zeros(n)   

  def propagate(arg):
    i, a = arg
    for j in range(1000):
      array[i] = i
      
  def propagate_non_shared(arg):
    i, a = arg
    for j in range(1000):
      non_shared[i] = i
  
  shm.isShared(non_shared);  
  shm.isShared(array);
  
  pool = shm.multiprocessing.Pool(processes=4)
  pp = pool.map(propagate, zip(range(n), [None] * n)); 
  print(array)

  pool = shm.multiprocessing.Pool(processes=4)
  pp = pool.map(propagate_non_shared, zip(range(n), [None] * n)); 
  print(non_shared)

