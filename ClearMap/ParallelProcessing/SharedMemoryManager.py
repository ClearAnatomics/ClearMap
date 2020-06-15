# -*- coding: utf-8 -*-
"""
SharedMemoryManager
===================

Shared memory array manager for parallel processing using ctype shared
arrays in :mod:`~ClearMap.ParallelProcessing.SharedMemoryArray`.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


import multiprocessing as mp

import ClearMap.ParallelProcessing.SharedMemoryArray as sma

__all__ = ['get', 'insert', 'free', 'clean', 'zeros']; 

###############################################################################
### Manager
###############################################################################

class SharedMemmoryManager(object):    
  """SharedMemmoryManager provides handles to shared arrays for parallel processing."""
  
  _instance = None
  """Pointer to global instance"""

  __slots__ = ['arrays', 'current', 'count', 'lock'];

  def __new__(cls, *args, **kwargs):
    if not cls._instance:
      cls._instance = super(SharedMemmoryManager, cls).__new__(cls, *args, **kwargs)
    return cls._instance
  
  def __init__(self):
    self.arrays = [None] * 32
    self.current = 0
    self.count  = 0
    self.lock = mp.Lock()
  
  def handle(self):
    # double size if necessary
    if (self.count >= len(self.arrays)):
      self.arrays = self.arrays + [None] * len(self.arrays);
    # find free handle
    while self.arrays[self.current] is not None:
      self.current = (self.current + 1) % len(self.arrays)
    return self.current;
  
  @staticmethod
  def instance():
    if not SharedMemmoryManager._instance:
      SharedMemmoryManager._instance = SharedMemmoryManager()
    return SharedMemmoryManager._instance  
  
  @staticmethod
  def zeros(shape, dtype = None, order = None):
    self = SharedMemmoryManager.instance()
    self.lock.acquire()
    # next handle
    self.handle()        
    # create array in shared memory segment and wrap with numpy     
    self.arrays[self.current] = sma.zeros(shape, dtype, order);
    # update cnt
    self.count += 1
    self.lock.release()
    return self.current;
  
  @staticmethod
  def insert(array):
    self = SharedMemmoryManager.instance()
    # next handle
    self.handle() 
    # convert to shared array and insert into handle
    self.arrays[self.current] = sma.as_shared(array);
    # update cnt
    self.count += 1
    return self.current
  
  @staticmethod
  def free(hdl):
    self = SharedMemmoryManager.instance()
    self.lock.acquire()
    # set reference to None
    if self.arrays[hdl] is not None: # consider multiple calls to free
      self.arrays[hdl] = None
      self.count -= 1
    self.lock.release()
  
  @staticmethod
  def clean():
    self = SharedMemmoryManager.instance()
    self.lock.acquire()
    for i in range(len(self.arrays)):
      self.arrays[i] = None;
    self.current = 0
    self.count = 0
    self.lock.release()
  
  @staticmethod
  def get(i):
    self = SharedMemmoryManager.instance()
    return self.arrays[i]
  

###############################################################################
### Functionality
###############################################################################

def zeros(shape, dtype = None, order = None):
  """Creates a shared zero array and inserts it into the shared memory manager.
  
  Arguments
  ---------
  shape : tuple
    Shape of the array.
  dtype : dtype or None
    The type of the array.
  order : 'C', 'F', or None
    The contiguous order of the array.
  
  Returns
  -------
  handle : int
    The handle to this array.
  """
  return SharedMemmoryManager.zeros(shape=shape, dtype=dtype, order=order)


def get(handle):
  """Returns the array in the shared memory manager with given handle.
  
  Arguments
  ---------
  handle : int
    Shared memory handle of the array.    
    
  Returns
  -------
  array : array
    The shared array with the specified handle.
  """
  return SharedMemmoryManager.get(handle)


def insert(array):
  """Inserts the array in the shared memory manager.
  
  Arguments
  ---------
  array : array
    The array to insert into the shared memory manager.
    
  Returns
  -------
  handle : int
    The shared array handle.
  """ 
  return SharedMemmoryManager.insert(array)


def free(handle):
  """Removes the array with given handle from the shared memory manager.
  
  Arguments
  ---------
  handle : int
    Shared memory handle of the array.    
  """
  SharedMemmoryManager.free(handle)


def clean():
  """Removes all references to the shared arrays."""
  SharedMemmoryManager.clean()


###############################################################################
### Tests
###############################################################################

def _test():
  #from importlib import reload
  import ClearMap.ParallelProcessing.SharedMemoryManager as smm 
  reload(smm)

  def propagate(t):
    i, hdl = t
    a = smm.get(hdl)
    #if i % 100000 == 0:
    #  print('i=%d' % i)
    for j in range(1):
      a[i] = i

  n = 5000000;
  hdl = smm.zeros(n, dtype=float)            
  print(hdl)
  pool = smm.mp.Pool(processes=2)
  
  smm.mp.Process()

  pp = pool.map_async(propagate, zip(range(n), [hdl] * n)); #analysis:ignore
  pool.close()
  pool.join();

  result = smm.get(hdl)
  print(result)
  
  smm.sma.is_shared(result)
  smm.free(hdl)
  
  smm.clean()
