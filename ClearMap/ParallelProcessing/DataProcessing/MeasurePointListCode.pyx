#cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False, cdivision=True
"""
MeasurePointListCode
====================

Cython code for fast measurents of measurements in very large arrays.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'

import numpy as np
cimport numpy as np

cimport cython
from cython.parallel import prange, parallel

ctypedef fused source_t:
  np.int32_t
  np.int64_t
  np.uint8_t
  np.uint16_t
  np.uint32_t
  np.uint64_t
  np.float32_t
  np.float64_t

ctypedef fused sink_t:
  np.int32_t
  np.int64_t
  np.uint8_t
  np.uint16_t
  np.uint32_t
  np.uint64_t
  np.float32_t 
  np.float64_t
  
ctypedef fused point_t:
  np.int32_t
  np.int64_t
  np.uint8_t
  np.uint16_t
  np.uint32_t
  np.uint64_t
  np.float32_t 
  np.float64_t
  
ctypedef np.float_t value_t

ctypedef Py_ssize_t index_t

cdef extern from "stdio.h":
  int printf(char *format, ...) nogil


###############################################################################
### Measure extremes
###############################################################################
       
#@cython.boundscheck(False)
#@cython.wraparound(False)
cpdef void measure_max(source_t[:] source, index_t[:] shape, index_t[:] strides, point_t[:,:] points, index_t[:,:] search, index_t[:] max_search, sink_t[:] sink, int processes) nogil:
    
  cdef index_t i, j, k, d, n, v
  cdef index_t n_points = points.shape[0];
  cdef index_t n_dim    = strides.shape[0];   
  cdef source_t s;                            
    
  with nogil, parallel(num_threads = processes):    
    for n in prange(n_points, schedule='guided'):
      
      #center point value      
      j = 0;
      for d in range(n_dim):  
        j = j + <index_t>(points[n,d]) * strides[d];
      s = source[j];
      
      #scan over search indices
      for i in range(max_search[n]):
        j = 0;
        v = 1;
        for d in range(n_dim):
          k = <index_t>(points[n,d]) + search[i,d];
          if not (0 <= k and k < shape[d]):
            v = 0;
            break;
          else:
            j = j + k * strides[d];
        if v == 1:
          if source[j] > s:
            s = source[j];

      sink[n] = <sink_t>s;
  return;


#@cython.boundscheck(False)
#@cython.wraparound(False)
cpdef void measure_min(source_t[:] source, index_t[:] shape, index_t[:] strides, point_t[:,:] points, index_t[:,:] search, index_t[:] max_search, sink_t[:] sink, int processes) nogil:
    
  cdef index_t i, j, k, d, n, v
  cdef index_t n_points = points.shape[0];
  cdef index_t n_dim    = strides.shape[0];   
  cdef source_t s;                            
    
  with nogil, parallel(num_threads = processes):    
    for n in prange(n_points, schedule='guided'):
      
      #center point value      
      j = 0;
      for d in range(n_dim):  
        j = j + <index_t>(points[n,d]) * strides[d];
      s = source[j];
      
      #scan over search indices
      for i in range(max_search[n]):
        j = 0;
        v = 1;
        for d in range(n_dim):
          k = <index_t>(points[n,d]) + search[i,d];
          if not (0 <= k and k < shape[d]):
            v = 0;
            break;
          else:
            j = j + k * strides[d];
        if v == 1:
          if source[j] < s:
            s = source[j];

      sink[n] = <sink_t>s;
  return;


#@cython.boundscheck(False)
#@cython.wraparound(False)
cpdef void measure_mean(source_t[:] source, index_t[:] shape, index_t[:] strides, point_t[:,:] points, index_t[:,:] search, index_t[:] max_search, value_t[:] sink, int processes) nogil:
    
  cdef index_t i, j, k, d, n, v, l
  cdef index_t n_points = points.shape[0];
  cdef index_t n_dim    = strides.shape[0];   
  #cdef source_t s;                            
    
  with nogil, parallel(num_threads = processes):      
    for n in prange(n_points, schedule='guided'):
      sink[n] = 0;
      
      #center point value      
      j = 0;
      for d in range(n_dim):  
        j = j + <index_t>(points[n,d]) * strides[d];
      sink[n] += <value_t>source[j];
      l = 1;
      
      #scan over search indices
      for i in range(max_search[n]):
        j = 0;
        v = 1;
        for d in range(n_dim):
          k = <index_t>(points[n,d]) + search[i,d];
          if not (0 <= k and k < shape[d]):
            v = 0;
            break;
          else:
            j = j + k * strides[d];
        if v == 1:
          sink[n] += <value_t>source[j];
          l = l + 1;
    
      sink[n] /= <value_t>(l);
  return;


#@cython.boundscheck(False)
#@cython.wraparound(False)
cpdef void measure_sum(source_t[:] source, index_t[:] shape, index_t[:] strides, point_t[:,:] points, index_t[:,:] search, index_t[:] max_search, value_t[:] sink, int processes) nogil:
    
  cdef index_t i, j, k, d, n, v, l
  cdef index_t n_points = points.shape[0];
  cdef index_t n_dim    = strides.shape[0];   
  #cdef source_t s;                            
    
  with nogil, parallel(num_threads = processes):      
    for n in prange(n_points, schedule='guided'):
      sink[n] = 0;
      
      #center point value      
      j = 0;
      for d in range(n_dim):  
        j = j + <index_t>(points[n,d]) * strides[d];
      sink[n] += <value_t>source[j];
      l = 1;
      
      #scan over search indices
      for i in range(max_search[n]):
        j = 0;
        v = 1;
        for d in range(n_dim):
          k = <index_t>(points[n,d]) + search[i,d];
          if not (0 <= k and k < shape[d]):
            v = 0;
            break;
          else:
            j = j + k * strides[d];
        if v == 1:
          sink[n] += <value_t>source[j];
          l = l + 1;
    
      #sink[n] /= <value_t>(l);
  return;


###############################################################################
### Find distances to voxles with smaller values
###############################################################################
       
#@cython.boundscheck(False)
#@cython.wraparound(False)
cpdef void find_smaller_than_value(source_t[:] source, index_t[:] shape, index_t[:] strides, point_t[:,:] points, index_t[:,:] search, value_t value, index_t[:] sink, int processes) nogil:
  
  cdef index_t i, j, k, d, n, v
  cdef index_t n_points = points.shape[0];
  cdef index_t n_dim    = strides.shape[0];                            
  cdef index_t n_search = search.shape[0]; 
      
  with nogil, parallel(num_threads = processes):    
    for n in prange(n_points, schedule='guided'):
      sink[n] = n_search;      
      
      #scan over search indices
      for i in range(n_search):
        j = 0;
        v = 1;
        for d in range(n_dim):
          k = <index_t>(points[n,d]) + search[i,d];
          if not (0 <= k and k < shape[d]):
            v = 0;
            break;
          else:
            j = j + k * strides[d];
        if v == 1:
          if <value_t>(source[j]) <= value:
            sink[n] = i;
            break;
  return;


#@cython.boundscheck(False)
#@cython.wraparound(False)
cpdef void find_smaller_than_fraction(source_t[:] source, index_t[:] shape, index_t[:] strides, point_t[:,:] points, index_t[:,:] search, value_t fraction, index_t[:] sink, int processes) nogil:
  
  cdef index_t i, j, k, d, n, v
  cdef index_t n_points = points.shape[0];
  cdef index_t n_dim    = strides.shape[0];                            
  cdef index_t n_search = search.shape[0]; 
  cdef value_t value;
  
  with nogil, parallel(num_threads = processes):    
    for n in prange(n_points, schedule='guided'):
      sink[n] = n_search;      
      
      #reference value
      j = 0;
      for d in range(n_dim):  
        j = j + <index_t>(points[n,d]) * strides[d];
      value = <value_t>(source[j]) * fraction;
      
      #scan over search indices
      for i in range(n_search):
        j = 0;
        v = 1;
        for d in range(n_dim):
          k = <index_t>(points[n,d]) + search[i,d];
          if not (0 <= k and k < shape[d]):
            v = 0;
            break;
          else:
            j = j + k * strides[d];
        if v == 1:
          if <value_t>(source[j]) <= value:
            sink[n] = i;
            break;
  return;    
    

#@cython.boundscheck(False)
#@cython.wraparound(False)
cpdef void find_smaller_than_values(source_t[:] source, index_t[:] shape, index_t[:] strides, point_t[:,:] points, index_t[:,:] search, value_t[:] value, index_t[:] sink, int processes) nogil:

  cdef index_t i, j, k, d, n, v
  cdef index_t n_points = points.shape[0];
  cdef index_t n_dim    = strides.shape[0];                            
  cdef index_t n_search = search.shape[0]; 
  cdef value_t ref;
  
  with nogil, parallel(num_threads = processes):    
    for n in prange(n_points, schedule='guided'):
      sink[n] = n_search; 
      
      #reference value
      ref = value[n];
      
      #scan over search indices
      for i in range(n_search):
        j = 0;
        v = 1;
        for d in range(n_dim):
          k = <index_t>(points[n,d]) + search[i,d];
          if not (0 <= k and k < shape[d]):
            v = 0;
            break;
          else:
            j = j + k * strides[d];
        if v == 1:
          if <value_t>(source[j]) <= ref:
            sink[n] = i;
            break;
  return; 
