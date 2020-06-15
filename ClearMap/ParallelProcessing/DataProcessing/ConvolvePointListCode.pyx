#cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False, cdivision=True

"""
ConvolvePointListCode
=====================

Cython code for fast convolution on a subset of points in very large arrays
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


from multiprocessing import cpu_count;
ncpus = cpu_count();

cimport cython
from cython.parallel import prange, parallel
import numpy as np
cimport numpy as np


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

ctypedef fused kernel_t:
  np.int32_t
  np.int64_t
  np.uint8_t
  np.uint16_t
  np.uint32_t
  np.float32_t 
  np.float64_t
  
ctypedef np.float_t max_t

ctypedef Py_ssize_t index_t

ctypedef np.uint8_t bool_t

###############################################################################
### Convolve x,y,z version
###############################################################################

#@cython.boundscheck(False)
#@cython.wraparound(False)
cpdef void convolve_3d_xyz(source_t[:, :, :] source, kernel_t[:, :, :] kernel, index_t[:] x, index_t[:] y, index_t[:] z, sink_t[:] sink, int processes) nogil:
    """Convolves binary data with binary kernel at specific x,y,z coordinates only."""
    
    cdef index_t i, j, k, xx, yy, zz, n
    cdef index_t ki = kernel.shape[0], kj = kernel.shape[1], kk = kernel.shape[2];
    cdef index_t di = source.shape[0], dj = source.shape[1], dk = source.shape[2];
    cdef index_t ki2 = ki/2, kj2 = kj/2, kk2 = kk/2;
    cdef index_t npoints = x.shape[0];
    cdef index_t xi, yj, zk
    
    with nogil, parallel(num_threads = processes):    
      for n in prange(npoints, schedule='guided'):
        xx = x[n]; 
        yy = y[n]; 
        zz = z[n];
        #out[n] = 0;
        for i in range(ki):
          xi = xx + i - ki2;
          if xi >= 0 and xi < di:
            for j in range(kj):
              yj = yy + j - kj2;
              if yj >= 0 and yj < dj:
                for k in range(kk):
                  zk = zz + k - kk2;
                  if zk >= 0 and zk < dk:
                    sink[n] += <sink_t>(source[xi, yj, zk]) * <sink_t>(kernel[i, j, k])
    
    return;


#@cython.boundscheck(False)
#@cython.wraparound(False)
cpdef void convolve_3d_xyz_no_check(source_t[:, :, :] source, kernel_t[:, :, :] kernel, index_t[:] x, index_t[:] y, index_t[:] z, sink_t[:] sink, int processes) nogil:
    """Convolves binary data with binary kernel at specific x,y,z coordinates only."""
    
    cdef index_t i, j, k, xx, yy, zz, n
    cdef index_t ki = kernel.shape[0], kj = kernel.shape[1], kk = kernel.shape[2];
    cdef index_t ki2 = ki/2, kj2 = kj/2, kk2 = kk/2;
    cdef index_t npoints = x.shape[0];
    
    with nogil, parallel(num_threads = processes):    
      for n in prange(npoints, schedule='guided'):
        xx = x[n]; 
        yy = y[n]; 
        zz = z[n];
        for i in range(ki):
          for j in range(kj):
            for k in range(kk):
              sink[n] += <sink_t>(source[xx + i - ki2, yy + j - kj2, zz + k - kk2]) * <sink_t>(kernel[i, j, k])
    
    return;



###############################################################################
### Convolve point list versions
###############################################################################

#@cython.boundscheck(False)
#@cython.wraparound(False)
cpdef void convolve_3d_points(source_t[:, :, :] source, kernel_t[:, :, :] kernel, index_t[:, :] points, sink_t[:] sink, int processes) nogil:
    """Convolves binary data with a specified kernel at specific points only."""
    
    cdef index_t i, j, k, y, z, x, n;
    cdef index_t ki = kernel.shape[0], kj = kernel.shape[1], kk = kernel.shape[2];
    cdef index_t di = source.shape[0], dj = source.shape[1], dk = source.shape[2];
    cdef index_t ki2 = ki/2, kj2 = kj/2, kk2 = kk/2;
    cdef index_t npoints = points.shape[0]
    cdef index_t xi, yj, zk
    
    #cdef np.ndarray[data_t, ndim=1] out = np.empty(npoints, dtype = data_t);
    with nogil, parallel(num_threads = processes):    
      for n in prange(npoints, schedule='guided'):
        x = points[n, 0]; 
        y = points[n, 1]; 
        z = points[n, 2];
        
        for i in range(ki):
          xi = x + i - ki2;
          if xi >= 0 and xi < di:
            for j in range(kj):
              yj = y + j - kj2;
              if yj >= 0 and yj < dj:
                for k in range(kk):
                  zk = z + k - kk2;
                  if zk >= 0 and zk < dk:
                    sink[n] += <sink_t>(source[xi, yj, zk]) * <sink_t>(kernel[i, j, k])
    
    return;


#@cython.boundscheck(False)
#@cython.wraparound(False)
cpdef void convolve_3d_points_no_check(source_t[:, :, :] source, kernel_t[:, :, :] kernel, index_t[:, :] points, sink_t[:] sink, int processes) nogil:
    """Convolves binary data with a specified kernel at specific points only."""
    
    cdef index_t i, j, k, y, z, x, n;
    cdef index_t ki = kernel.shape[0], kj = kernel.shape[1], kk = kernel.shape[2];
    cdef index_t ki2 = ki/2, kj2 = kj/2, kk2 = kk/2;
    cdef index_t npoints = points.shape[0]
    #cdef np.ndarray[data_t, ndim=1] out = np.empty(npoints, dtype = data_t);
    with nogil, parallel(num_threads = processes):    
      for n in prange(npoints, schedule='guided'):
        x = points[n, 0]; 
        y = points[n, 1]; 
        z = points[n, 2];
        for i in range(kk):
          for j in range(ki):
            for k in range(kj):
              sink[n] += <sink_t>(source[x + i - ki2, y + j - kj2, z + k - kk2]) * <sink_t>(kernel[i, j, k])
    
    return;



###############################################################################
### Convolve index list version
###############################################################################

#@cython.boundscheck(False)
#@cython.wraparound(False)
cpdef void convolve_3d_indices(source_t[:] source, index_t[:] strides, kernel_t[:, :, :] kernel, index_t[:] points, sink_t[:] sink, int processes) nogil:
    """Convolves binary data with a specified kernel at specific points given as indices of a flat array."""
    
    cdef index_t i, j, k, d, n
    cdef index_t ki = kernel.shape[0], kj = kernel.shape[1], kk = kernel.shape[2];
    cdef index_t ki2 = ki/2, kj2 = kj/2, kk2 = kk/2;
    cdef index_t npoints = points.shape[0]
    cdef index_t si = strides[0], sj = strides[1], sk = strides[2];
    cdef index_t ds = source.shape[0];
    cdef index_t p
    
    #cdef np.ndarray[data_t, ndim=1] out = np.empty(npoints, dtype = data_t);
    with nogil, parallel(num_threads = processes):    
      for n in prange(npoints, schedule='guided'):
        d = points[n];
        for i in range(ki):
          for j in range(kj):
            for k in range(kk):
              p = d + (i - ki2) * si + (j - kj2) * sj + (k - kk2) * sk;
              if p >= 0 and p < ds:
                sink[n] += <sink_t>(source[p]) * <sink_t>(kernel[i, j, k])
    
    return;


#@cython.boundscheck(False)
#@cython.wraparound(False)
cpdef void convolve_3d_indices_no_check(source_t[:] source, index_t[:] strides, kernel_t[:, :, :] kernel, index_t[:] points, sink_t[:] sink, int processes) nogil:
    """Convolves binary data with a specified kernel at specific points given as indices of a flat array."""
    
    cdef index_t i, j, k, d, n
    cdef index_t ki = kernel.shape[0], kj = kernel.shape[1], kk = kernel.shape[2];
    cdef index_t ki2 = ki/2, kj2 = kj/2, kk2 = kk/2;
    cdef index_t npoints = points.shape[0]
    cdef index_t si = strides[0], sj = strides[1], sk = strides[2];
    #cdef np.ndarray[data_t, ndim=1] out = np.empty(npoints, dtype = data_t);
    with nogil, parallel(num_threads = processes):    
      for n in prange(npoints, schedule='guided'):
        d = points[n];
        for i in range(ki):
          for j in range(kj):
            for k in range(kk):
              sink[n] += <sink_t>(source[d + (i - ki2) * si + (j - kj2) * sj + (k - kk2) * sk]) * <sink_t>(kernel[i, j, k])
    
    return;


###############################################################################
### Convolve  index list versions with max condition
###############################################################################

#@cython.boundscheck(False)
#@cython.wraparound(False)
cpdef void convolve_3d_indices_if_smaller_than(source_t[:] source, index_t[:] strides, kernel_t[:, :, :] kernel, index_t[:] points, max_t max_value, bool_t[:] sink, int processes) nogil:
    """Convolves binary data with a specified kernel at specific points given as indices and check if the result is smaller than a maximal value."""
    
    cdef index_t i, j, k, d, n
    cdef index_t ki = kernel.shape[0], kj = kernel.shape[1], kk = kernel.shape[2];
    cdef index_t ki2 = ki/2, kj2 = kj/2, kk2 = kk/2;
    cdef index_t npoints = points.shape[0]
    cdef index_t si = strides[0], sj = strides[1], sk = strides[2];
    cdef max_t res;
    cdef index_t ds = source.shape[0];
    cdef index_t p
    
    #cdef np.ndarray[data_t, ndim=1] out = np.empty(npoints, dtype = data_t);
    with nogil, parallel(num_threads = processes):    
      for n in prange(npoints, schedule='guided'):
        d = points[n];
        res = 0;
        for i in range(ki):
          for j in range(kj):
            for k in range(kk):
              p = d + (i - ki2) * si + (j - kj2) * sj + (k -kk2) * sk;
              if p >= 0 and p < ds:
                res = res + <max_t>(source[p]) * <max_t>(kernel[i, j, k])
        sink[n] = (res < max_value);

    return;


#@cython.boundscheck(False)
#@cython.wraparound(False)
cpdef void convolve_3d_indices_if_smaller_than_no_check(source_t[:] source, index_t[:] strides, kernel_t[:, :, :] kernel, index_t[:] points, max_t max_value, bool_t[:] sink, int processes) nogil:
    """Convolves binary data with a specified kernel at specific points given as indices and check if the result is smaller than a maximal value."""
    
    cdef index_t i, j, k, d, n
    cdef index_t ki = kernel.shape[0], kj = kernel.shape[1], kk = kernel.shape[2];
    cdef index_t ki2 = ki/2, kj2 = kj/2, kk2 = kk/2;
    cdef index_t npoints = points.shape[0]
    cdef index_t si = strides[0], sj = strides[1], sk = strides[2];
    cdef max_t res;
    
    #cdef np.ndarray[data_t, ndim=1] out = np.empty(npoints, dtype = data_t);
    with nogil, parallel(num_threads = processes):    
      for n in prange(npoints, schedule='guided'):
        d = points[n];
        res = 0;
        for i in range(ki):
          for j in range(kj):
            for k in range(kk):
              res = res + <max_t>(source[d + (i - ki2) * si + (j - kj2) * sj + (k - kk2) * sk]) * <max_t>(kernel[i, j, k])
        sink[n] = (res < max_value);

    return;
