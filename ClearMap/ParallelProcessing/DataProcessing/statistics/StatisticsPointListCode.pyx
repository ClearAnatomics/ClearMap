#cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False, cdivision=True
"""
StatisticsPointListCode
=======================

Cython code for calculating statistics around a set of points in a large array.
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

ctypedef fused point_t:
#  np.int32_t
  np.int64_t
#  np.uint8_t
# np.uint16_t
# np.uint32_t
#  np.uint64_t
#  np.float32_t 
  np.float64_t

ctypedef fused sink_t:
#  np.int32_t
  np.int64_t
#  np.uint8_t
#  np.uint16_t
#  np.uint32_t
#  np.uint64_t
#  np.float32_t 
  np.float64_t

#ctypedef fused weight_t:
#  np.int32_t
#  np.int64_t
#  np.uint8_t
#  np.uint16_t
#  np.uint32_t
#  np.uint64_t
#  np.float32_t 
#  np.float64_t
ctypedef np.float64_t weight_t

ctypedef np.float64_t kernel_t

ctypedef Py_ssize_t index_t

cdef extern from "stdio.h":
  int printf(char *format, ...) nogil


###############################################################################
### Average 
###############################################################################

#@cython.boundscheck(False)
#@cython.wraparound(False)
cpdef void average(point_t[:,:] points, weight_t[:] weights, index_t[:,:] indices, sink_t[:] sink, index_t[:] shape, index_t[:] strides, index_t[:] counter, int processes) nogil:
  """Converts a list of points into an volumetric image array."""
  
  cdef index_t i, j, k, v, d, n
  cdef index_t n_points  = points.shape[0];
  cdef index_t n_sink    = sink.shape[0];
  cdef index_t n_indices = indices.shape[0];                                   
  cdef index_t n_dim     = strides.shape[0];
  
  with nogil, parallel(num_threads = processes):    
    for n in prange(n_points, schedule='guided'):
      for i in range(n_indices):
        j = 0;
        v = 1;
        for d in range(n_dim):
          k = <index_t>points[n,d] + indices[i,d];
          if not (0 <= k and k < shape[d]):
            v = 0;
            break;
          else:
            j = j + k * strides[d];
        if v == 1:
          sink[j] += <sink_t>weights[n];
          counter[j] = counter[j] + 1;
  return;
