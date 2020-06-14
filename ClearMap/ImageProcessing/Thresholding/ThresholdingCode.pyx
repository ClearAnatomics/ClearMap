#cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False, cdivision=True
"""
ThresholdingCode
================

Cython code for the thresholding module.
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

ctypedef fused source_t: 
  np.int32_t
  np.int64_t
  np.uint8_t
  np.uint16_t
  np.uint32_t
  np.float32_t 
  np.float64_t
  
ctypedef fused sink_t:
  np.int32_t
  np.int64_t
  np.uint8_t
  np.uint16_t
  np.uint32_t
  np.float32_t 
  np.float64_t
 
ctypedef fused index_t:
  Py_ssize_t

ctypedef np.uint8_t bool_t


cdef enum Status:
  NOT_CHECKED = -1
  BACKGROUND = 0
  FOREGROUND = 1
  

cdef extern from "stdio.h":
    int printf(char *format, ...) nogil


###############################################################################
### Basic region growing algorithms
###############################################################################

#@cython.boundscheck(False)
#@cython.wraparound(False)
cpdef void threshold(source_t[:] source, sink_t[:] sink, index_t[:] strides, index_t[:] seeds, index_t[:] parameter_index, double[:] parameter_double):
    
  cdef double threshold = parameter_double[0];
  
  cdef index_t n = len(source);
  cdef index_t ndim = len(strides);
  
  cdef index_t i,j,d
  cdef queue[index_t] q;                       
  for i in seeds:
    q.push(i);

  for i in range(n):
    sink[i] = <sink_t>NOT_CHECKED;
  #for i in seeds:
  #  sink[i] = <index_t>FOREGROUND;
  
  while not q.empty():
    i = q.front();
    q.pop();
    if sink[i] == <sink_t>NOT_CHECKED:           
      if source[i] < threshold:
        sink[i] = <sink_t>BACKGROUND;      
      else:   
        sink[i] = <sink_t>FOREGROUND;
        
        #add neighbours to the queue
        for d in range(ndim):
          j = i - strides[d];
          if j > 0 and sink[j] == <sink_t>NOT_CHECKED:
            q.push(j);
          j = i + strides[d];
          if j < n and sink[j] == <sink_t>NOT_CHECKED:
            q.push(j);


#@cython.boundscheck(False)
#@cython.wraparound(False)
cpdef void threshold_to_background(source_t[:] source, sink_t[:] sink, bool_t[:] background, index_t[:] strides, index_t[:] seeds, index_t[:] parameter_index, double[:] parameter_double):
    
  cdef double threshold = parameter_double[0];
  
  cdef index_t n = len(source);
  cdef index_t ndim = len(strides);
  
  cdef index_t i,j,d
  cdef queue[index_t] q;                       
  for i in seeds:
    q.push(i);

  for i in range(n):
    sink[i] = <sink_t>NOT_CHECKED;
  #for i in seeds:
  #  sink[i] = <index_t>FOREGROUND;
  
  while not q.empty():
    i = q.front();
    q.pop();
    if sink[i] == <sink_t>NOT_CHECKED:  
      if source[i] < threshold or background[i]:
        sink[i] = <sink_t>BACKGROUND;      
      else:   
        sink[i] = <sink_t>FOREGROUND;
        
        #add neighbours to the queue
        for d in range(ndim):
          j = i - strides[d];
          if j >= 0 and sink[j] == <sink_t>NOT_CHECKED:
            q.push(j);
          j = i + strides[d];
          if j < n and sink[j] == <sink_t>NOT_CHECKED:
            q.push(j);

        