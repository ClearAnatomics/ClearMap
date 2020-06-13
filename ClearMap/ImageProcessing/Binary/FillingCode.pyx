#cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False, cdivision=True
"""
FillingCode
===========

Cython Code for the Filling module.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


cimport cython
from cython.parallel import prange, parallel


cimport numpy as np

from libcpp.queue cimport queue

ctypedef np.uint8_t source_t;

ctypedef np.uint8_t sink_t;

ctypedef np.int8_t temp_t;
  
ctypedef Py_ssize_t index_t;
  

cdef extern from "stdio.h":
  int printf(char *format, ...) nogil


cdef enum Status:
  NOT_CHECKED = -1
  EMPTY = 0
  FILLED = 1


cpdef void prepare_temp(source_t[:] source, temp_t[:] temp, int processes) nogil:
    
  cdef index_t n = source.shape[0]; 
  cdef index_t i
  
  with nogil, parallel(num_threads = processes):
    for i in prange(n, schedule = 'guided'):
      if source[i]:
        temp[i] = FILLED;
      else:
        temp[i] = NOT_CHECKED;

  return;


cpdef void label_temp(temp_t[:] temp, index_t[:] strides, index_t [:] seeds, int processes) nogil:
   
  cdef index_t size = temp.shape[0];
  cdef index_t ndim = strides.shape[0];
  cdef index_t n_seeds = seeds.shape[0];

  cdef temp_t* temp_pointer = &temp[0];
  cdef index_t* strides_pointer = &strides[0];
  cdef index_t* seeds_pointer = &seeds[0];
  
  cdef index_t i

  with nogil, parallel(num_threads = processes):
    for i in prange(n_seeds, schedule = 'guided'):
      label_from_seed(temp_pointer, size, seeds_pointer[i], strides_pointer, ndim);

  return;


cdef void label_from_seed(temp_t* temp, index_t size, index_t seed, index_t* strides, index_t ndim) nogil:
  cdef queue[index_t] q
  cdef index_t i,j,d
  #flood fill from seed    
  q.push(seed);
  while not q.empty():
    i = q.front();
    q.pop();
    if temp[i] == NOT_CHECKED:
      temp[i] = FILLED;
      #add neighbours to the queue
      for d in range(ndim):
        j = i - strides[d];
        if j >= 0 and temp[j] == NOT_CHECKED:
          q.push(j);
        j = i + strides[d];
        if j < size  and temp[j] == NOT_CHECKED:
          q.push(j);


cpdef void fill(source_t[:] source, temp_t[:] temp, sink_t[:] sink,  int processes) nogil:
    
  cdef index_t n = source.shape[0];
  cdef index_t i
  
  with nogil, parallel(num_threads = processes):
    #fill holes
    for i in prange(n, schedule = 'guided'):
      if source[i] or temp[i] == NOT_CHECKED:
        sink[i] = 1;
      else:
        sink[i] = 0;
    
  return;
  
  