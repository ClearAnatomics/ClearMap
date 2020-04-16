#cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False, cdivision=True

"""
HysteresisThresholdingCode
==========================

Cython code for the hystersis thresholding module.
"""

__author__    = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__   = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__copyright__ = 'Copyright (c) 2020 by Christoph Kirst'

cimport cython
from cython.parallel import prange, parallel

cimport numpy as np

from libcpp.vector cimport  queue

ctypedef fused source_t:
#  np.int32_t
#  np.int64_t
  np.uint8_t
  np.uint16_t
#  np.uint32_t
#  np.uint64_t
#  np.float32_t
  np.double_t
  
ctypedef fused sink_t:
#  np.int32_t
#  np.int64_t
  np.uint8_t
#  np.uint16_t
#  np.uint32_t
#  np.uint64_t
#  np.float32_t
#  np.double_t
 
ctypedef fused index_t:
  Py_ssize_t


cdef extern from "stdio.h":
    int printf(char *format, ...) nogil

cdef inline source_t max_(source_t a, source_t b) nogil:
    return a if a >= b else b

cdef inline source_t min_(source_t a, source_t b) nogil:
    return a if a <= b else b
  


cpdef void threshold_at(source_t[:,:,:] source, sink_t[:,:,:] sink,
                        index_t x, index_t y, index_t z,
                        index_t nx, index_t ny, index_t nz,
                        double threshold) nogil:
  cdef index_t i
  cdef index_t xx, yy, zz
  
  if source[x,y,z] >= threshold:
    sink[x,y,z] = 1;
    
    xx = x - 1;
    if xx >= 0 and not sink[xx,y,z]:
      #printf('sub x: x=%d, y=%d, z=%d, p=%f s=%f\n', xx, y, z, threshold, <double>source[xx,y,z]);      
      threshold_at(source, sink, xx, y, z, nx, ny, nz, threshold)     

    xx = x + 1;
    if xx < nx and not sink[xx,y,z]:
      #printf('sub x: x=%d, y=%d, z=%d, p=%f s=%f\n', xx, y, z, threshold, <double>source[xx,y,z]);      
      threshold_at(source, sink, xx, y, z, nx, ny, nz, threshold)              
    
    yy = y - 1;
    if yy >= 0 and not sink[x,yy,z]:
      #printf('sub y: x=%d, y=%d, z=%d, p=%f s=%f\n', x, yy, z, threshold, <double>source[x,yy,z]);      
      threshold_at(source, sink, x, yy, z, nx, ny, nz, threshold)          
   
    yy = y + 1;
    if yy < ny and not sink[x,yy,z]:
      #printf('sub y: x=%d, y=%d, z=%d, p=%f s=%f\n', x, yy, z, threshold, <double>source[x,yy,z]);      
      threshold_at(source, sink, x, yy, z, nx, ny, nz, threshold)  


cpdef void threshold_from_seeds(source_t[:,:,:] source, sink_t[:, :, :] sink,
                                index_t[:] seeds_x, index_t[:] seeds_y, index_t[:] seeds_z, 
                                double percentage, double absolute, int processes = 1) nogil:
    """Threshold from seed"""

    # array sizes
    cdef index_t nx = source.shape[0]
    cdef index_t ny = source.shape[1]
    cdef index_t nz = source.shape[2]  
    cdef index_t ni = seeds_x.shape[0]

    # local variable types
    cdef index_t x,y,z,i
    cdef double relative
    
    #with nogil, parallel(num_threads = processes):   
    for i in range(ni): #, schedule='guided'):
      x = seeds_x[i];
      y = seeds_y[i];
      z = seeds_z[i];
      if not sink[x,y,z]:
        relative = <double>(source[x,y,z]) * percentage;
        if relative < absolute:
          relative = absolute; 
        printf('\n----------------------------------------------------\n')                              
        printf('x=%d, y=%d, z=%d, p=%f s=%f\n', x, y, z, relative, <double>source[x,y,z]);                                       
        threshold_at(source, sink, x, y, z, nx, ny, nz, relative);            
    
    #zz = z - 1;
    #if zz >= 0 and not sink[x,y,zz]:
    #  printf('sub z: x=%d, y=%d, z=%d, p=%f s=%f\n', x, y, zz, threshold, <double>source[x,y,zz]);      
    #  threshold_at(source, sink, x, y, zz, nx, ny, nz, threshold)            
    # 
    #zz = z + 1;
    #if zz < nz and not sink[x,y,zz]:
    #  printf('sub z: x=%d, y=%d, z=%d, p=%f s=%f\n', x, y, zz, threshold, <double>source[x,y,zz]);      
    #  threshold_at(source, sink, x, y, zz, nx, ny, nz, threshold)       
          
      
      
        
        
#
#@cython.boundscheck(False)
#@cython.wraparound(False)
#cpdef void hysteresis_threshold(source_t[:] source, sink_t[:] sink, index_t[:] strides, double threshold, double hysteresis_threshold) nogil:
#  
#  cdef index_t ndim = len(strides);
#   
#  cdef index_t xx, yy, zz
#  
#  
#  cdef 
#  
#  
#  if source[x,y,z] >= threshold:
#    sink[x,y,z] = 1;
#    
#    xx = x - 1;
#    if xx >= 0 and not sink[xx,y,z]:
#      #printf('sub x: x=%d, y=%d, z=%d, p=%f s=%f\n', xx, y, z, threshold, <double>source[xx,y,z]);      
#      threshold_at(source, sink, xx, y, z, nx, ny, nz, threshold)     
#
#    xx = x + 1;
#    if xx < nx and not sink[xx,y,z]:
#      #printf('sub x: x=%d, y=%d, z=%d, p=%f s=%f\n', xx, y, z, threshold, <double>source[xx,y,z]);      
#      threshold_at(source, sink, xx, y, z, nx, ny, nz, threshold)              
#    
#    yy = y - 1;
#    if yy >= 0 and not sink[x,yy,z]:
#      #printf('sub y: x=%d, y=%d, z=%d, p=%f s=%f\n', x, yy, z, threshold, <double>source[x,yy,z]);      
#      threshold_at(source, sink, x, yy, z, nx, ny, nz, threshold)          
#   
#    yy = y + 1;
#    if yy < ny and not sink[x,yy,z]:
#      #printf('sub y: x=%d, y=%d, z=%d, p=%f s=%f\n', x, yy, z, threshold, <double>source[x,yy,z]);      
#      threshold_at(source, sink, x, yy, z, nx, ny, nz, threshold)  
#    
#    #zz = z - 1;
#    #if zz >= 0 and not sink[x,y,zz]:
#    #  printf('sub z: x=%d, y=%d, z=%d, p=%f s=%f\n', x, y, zz, threshold, <double>source[x,y,zz]);      
#    #  threshold_at(source, sink, x, y, zz, nx, ny, nz, threshold)            
#    # 
#    #zz = z + 1;
#    #if zz < nz and not sink[x,y,zz]:
#    #  printf('sub z: x=%d, y=%d, z=%d, p=%f s=%f\n', x, y, zz, threshold, <double>source[x,y,zz]);      
#    #  threshold_at(source, sink, x, y, zz, nx, ny, nz, threshold)       
    

                          
#@cython.boundscheck(False)
#@cython.wraparound(False)
#cpdef void threshold_from_seeds(source_t[:,:,:] source, sink_t[:, :, :] sink,
#                                mask_t[:,:,:] seeds, double percentage, double absolute, int processes = 1) nogil:
#    """Threshold from seed"""
#
#    # array sizes
#    cdef index_t nx = source.shape[0]
#    cdef index_t ny = source.shape[1]
#    cdef index_t nz = source.shape[2]   
#
#    # local variable types
#    cdef index_t x,y,z
#    cdef double relative
#    
#    #with nogil, parallel(num_threads = processes):   
#    for x in range(nx): #, schedule='guided'):
#        for y in range(ny):
#          for z in range(nz):
#            if seeds[x,y,z]:
#              relative = <double>source[x,y,z] * percentage;
#              printf('x=%d, y=%d, z=%d, p=%f s=%f\n', x, y, z, relative, <double>source[x,y,z]);                                       
#              threshold_at(source, sink, x, y, z, nx, ny, nz, relative, absolute);

