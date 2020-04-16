"""
MeasurePointListCode
====================

Cython code for fast measurents of measurements in very large arrays.
"""
__author__    = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__   = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__copyright__ = 'Copyright 2019 by Christoph Kirst, The Rockefeller University, New York City'

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
### Find smaller
###############################################################################
       
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void measure_max(source_t[:] source, point_t[:] points, index_t[:] search, index_t[:] radii, sink_t[:] sink, int processes) nogil:
    
  cdef index_t i, j, d, n
  cdef index_t n_source = source.shape[0];
  cdef index_t n_points = points.shape[0];   
  cdef source_t v;                            
  #printf('n_points=%d', n_points);
  #with gil:
  #  print(n_points);
  #  print(radii);
  
  with nogil, parallel(num_threads = processes):    
    for n in prange(n_points, schedule='guided'):
      d = <index_t> points[n];
      v = source[d];
      #with gil:
      #  print('n=%d, d=%d, v=%f' % (n,d,v))
      for i in range(radii[n]):
         j = d + search[i];
         if 0 <= j and j < n_source:
           if v < source[j]:
             v = source[j];
         #with gil:
         #  print('j=%d, i=%d, search[i]=%d, v=%f' % (j,i,search[i],v))
      #with gil:
      #  print('n=%d, r=%d, v=%f' % (n,radii[n],v));
        
      sink[n] = <sink_t>v;
  return;
