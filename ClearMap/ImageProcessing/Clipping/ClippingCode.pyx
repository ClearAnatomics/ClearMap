#cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False, cdivision=True

"""
ClippingCode
============

Cython code for the Clipping module.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


cimport cython
from cython.parallel import prange, parallel

cimport numpy as np

ctypedef fused source_t:
  np.int32_t
  np.int64_t
  np.uint8_t
  np.uint16_t
  np.uint32_t
  np.uint64_t
  np.float32_t
  np.double_t
  
ctypedef fused sink_t:
  np.int32_t
  np.int64_t
  np.uint8_t
  np.uint16_t
  np.uint32_t
  np.uint64_t
  np.float32_t
  np.double_t
  
ctypedef Py_ssize_t index_t

#cdef extern from "stdio.h":
#    int printf(char *format, ...) nogil


#@cython.boundscheck(False)
#@cython.wraparound(False)
#@cython.nonecheck(False)
cpdef void clip(source_t[:,:,:] source, sink_t[:, :, :] sink,
                double clip_min, double clip_max, double clip_norm, int processes) nogil:
    """Clip image and normalize"""

    # array sizes
    cdef index_t nx = source.shape[0]
    cdef index_t ny = source.shape[1]
    cdef index_t nz = source.shape[2]   
  
    # local variable types
    cdef index_t x,y,z
    cdef double temp
    cdef double delta = clip_max-clip_min;
    
    with nogil, parallel(num_threads = processes):   
      for x in prange(nx, schedule='guided'):
        for y in range(ny):
          for z in range(nz):
            temp = <double>source[x,y,z];
            if temp < clip_min:
              temp = clip_min;
            if temp > clip_max:
              temp = clip_max;
            sink[x, y, z] = <sink_t>(clip_norm * (temp - clip_min)/(delta));
