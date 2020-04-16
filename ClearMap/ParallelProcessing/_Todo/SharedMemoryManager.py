#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Shared memory numpy array manager for parallel processing

Based on code by Martin Preinfalk
"""
__author__    = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__   = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__copyright__ = 'Copyright (c) 2017 by Christoph Kirst, The Rockefeller University, New York City'


import multiprocessing as mp

import SharedMemory as shm
from SharedMemory import isShared, asShared, cType, base


class SharedMemmoryManager:    
  """SharedMemmoryManager provides handles to shared numpy arrays fur use with map in multiprocessing"""
  
  _initSize = 32
  """Initial number of shared arrays"""
  
  _instance = None
  """Pointer to global instance"""

  def __new__(cls, *args, **kwargs):
    if not cls._instance:
      cls._instance = super(SharedMemmoryManager, cls).__new__(cls, *args, **kwargs)
    return cls._instance
          

  def __init__(self):
    self.lock = mp.Lock()
    self.cur = 0
    self.cnt = 0
    self.arrays = [None] * SharedMemmoryManager._initSize
  
  
  def freeHandle(self):
    # double size if necessary
    if (self.cnt >= len(self.arrays)):
      self.arrays = self.arrays + [None] * len(self.arrays);
    # find free handle
    while self.arrays[self.cur] is not None:
      self.cur = (self.cur + 1) % len(self.arrays)
    return self.cur;
  
  
  def _create(self, shape, dtype = None, order = 'A'):
    self.lock.acquire()
    # next handle
    self.freeHandle()        
    # create array in shared memory segment and wrap with numpy     
    self.arrays[self.cur] = shm.create(shape, dtype, order);
    # update cnt
    self.cnt += 1
    self.lock.release()
    return self.cur
  
  
  def _insert(self, array):
    # next handle
    self.freeHandle() 
    # convert to shared array and insert into handle
    self.arrays[self.cur] = shm.asShared(array);
    # update cnt
    self.cnt += 1
    return self.cur
  
  
  def _free(self, hdl):
    self.lock.acquire()
    # set reference to None
    if self.arrays[hdl] is not None: # consider multiple calls to free
      self.arrays[hdl] = None
      self.cnt -= 1
    self.lock.release()
  
  def _clean(self):
    self.lock.acquire()
    for i in range(len(self.arrays)):
      self.arrays[i] = None;
    self.cur = 0
    self.cnt = 0
    self.lock.release()
  
  def _get(self, i):
    return self.arrays[i]
  
  
  @staticmethod
  def instance():
    if not SharedMemmoryManager._instance:
      SharedMemmoryManager._instance = SharedMemmoryManager()
    return SharedMemmoryManager._instance
  
  
  @staticmethod
  def create(*args, **kwargs):
    return SharedMemmoryManager.instance()._create(*args, **kwargs)
  
  
  @staticmethod
  def insert(*args, **kwargs):
    return SharedMemmoryManager.instance()._insert(*args, **kwargs)
  
  
  @staticmethod
  def get(*args, **kwargs):
    return SharedMemmoryManager.instance()._get(*args, **kwargs)
  
  
  @staticmethod    
  def free(*args, **kwargs):
    return SharedMemmoryManager.instance()._free(*args, **kwargs)
    
  @staticmethod    
  def clean(*args, **kwargs):
    return SharedMemmoryManager.instance()._clean(*args, **kwargs)


# Initialize the main instance and global routines
manager = SharedMemmoryManager()

  
def create(*args, **kwargs):
  return manager.create(*args, **kwargs)
 
 
def get(*args, **kwargs):
  return manager.get(*args, **kwargs)
 
 
def insert(*args, **kwargs):
  return manager.insert(*args, **kwargs)

 
def free(*args, **kwargs):
  return manager.free(*args, **kwargs)

  
def clean(*args, **kwargs):
  return manager.clean(*args, **kwargs)
  

if __name__ == '__main__':
  from importlib import reload
  import ClearMap.ParallelProcessing.SharedMemoryManager as smm 
  reload(smm)

  def propagate(t):
    i, hdl = t
    a = smm.get(hdl)
    for j in range(1000):
      a[i] = i

  n = 100;
  hdl = smm.create(n, dtype='float')            
  pool = smm.mp.Pool(processes=4)

  pp = pool.map(propagate, zip(range(n), [hdl] * n));

  result = smm.get(hdl)
  print(result)
  
  smm.isShared(result);

  smm.free(hdl)
