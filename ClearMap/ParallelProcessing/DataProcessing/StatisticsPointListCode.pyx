#cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False, cdivision=True
"""
DevolvePointListCode
====================

Cython code for devolving or smearing out points in a large array.

Note
----
This process is useful for integrating or visualizing points as densities.
"""
__author__    = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__   = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__copyright__ = 'Copyright 2019 by Christoph Kirst, The Rockefeller University, New York City'

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
