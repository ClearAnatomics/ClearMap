#cython: --compile-args=-fopenmp --link-args=-fopenmp --force -a
# -*- coding: utf-8 -*-
"""
Cython Code for the ArrayProcessing module
"""
__author__    = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__   = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__copyright__ = 'Copyright (c) 2017 by Christoph Kirst, The Rockefeller University, New York City'

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
  np.float32_t
  np.float64_t
   
ctypedef fused source_int_t:
  np.int32_t
  np.int64_t
  np.uint8_t
  np.uint16_t
  np.uint32_t

ctypedef fused sink_t:
  np.int32_t
  np.int64_t
  np.uint8_t
  np.uint16_t
  np.uint32_t
  np.float32_t
  np.float64_t

ctypedef Py_ssize_t index_t;  


cdef extern from "stdio.h":
  int printf(char *format, ...) nogil


###############################################################################
### Lookup table
###############################################################################
    
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void apply_lut(source_int_t[:] source, sink_t[:] sink, sink_t[:] lut, int processes) nogil:
  cdef index_t n = source.shape[0];
  cdef index_t i
  with nogil, parallel(num_threads = processes):                 
    for i in prange(n, schedule = 'guided'):
      sink[i] = lut[source[i]];


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void apply_lut_to_index_3d(source_t[:,:,:] source, index_t[:,:,:] kernel, sink_t[:] lut, sink_t[:,:,:] sink, int processes) nogil:
  
  cdef index_t nx = source.shape[0], ny = source.shape[1], nz = source.shape[2];
  cdef index_t kx = kernel.shape[0], ky = kernel.shape[1], kz = kernel.shape[2];
  
  cdef index_t kx1 = kx / 2, ky1 = ky / 2, kz1 = kz / 2;
  cdef index_t kx2 = kx - kx1, ky2 = ky - ky1, kz2 = kz - kz1;
  cdef index_t nx_m_kx2 = nx - kx2, ny_m_ky2 = ny - ky2, nz_m_kz2 = nz - kz2;
  

  cdef index_t x,y,z, xk,yk,zk, sx,ex, sy,ey, sz,ez
  cdef index_t temp;
  
  #with gil:
  #  print('nx,ny,nz=%d,%d,%d, kx,y,z1=%d,%d,%d, kx,y,z2=%d,%d,%d' % (nx,ny,nz,kx1,ky1,kz1,kx2,ky2,kz2));
  
  with nogil, parallel(num_threads = processes):                 
    
    for x in prange(nx, schedule = 'guided'):
      sx = -kx1 if x >= kx1 else -x;
      ex =  kx2 if x <= nx_m_kx2 else nx - x;
      for y in range(ny):
        sy = -ky1 if y >= ky1 else -y;
        ey =  ky2 if y <= ny_m_ky2 else ny - y;
        for z in range(nz):
          sz = -kz1 if z >= kz1 else -z;
          ez =  kz2 if z <= nz_m_kz2 else nz - z;
  
          #with gil:
          #  print('x,y,z=%d,%d,%d, sx,y,z=%d,%d,%d, ex,y,z=%d,%d,%d' % (x,y,z,sx,sy,sz,ex,ey,ez))
  
          temp = 0;
          for xk in range(sx,ex):
            for yk in range(sy,ey):
              for zk in range(sz,ez):
                #with gil:
                #  print('xk,yk,zk=%d,%d,%d, xyz+xyzk=%d,%d,%d, xyzk+kxyz1=%d,%d,%d' % (xk,yk,zk, x+xk,y+yk,z+zk, xk+kx1,yk+ky1,zk+kz1));
                temp = temp + <index_t>(source[x + xk, y + yk, z + zk] * kernel[xk + kx1, yk + ky1, zk + kz1]);
  
          #with gil:
          #  print('temp=%r' % temp)
          
          sink[x,y,z] = lut[<index_t>(temp)];


###############################################################################
### Correlation
###############################################################################

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void correlate1d(source_t[:] source, index_t[:] line_starts, index_t line_stride, index_t line_shape,
                       sink_t[:] sink, sink_t[:] kernel, int processes) nogil:
  
  cdef index_t n = line_starts.shape[0];
  cdef index_t k = kernel.shape[0];
  cdef index_t k1 = k / 2;
  cdef index_t k2 = k - k1;
  cdef index_t line_shape_minus_k2 = line_shape - k2;
  
  cdef sink_t* kernel_pointer = &kernel[k1];

  cdef index_t i,j,ls,p,m,s,e
  cdef double temp;

  with nogil, parallel(num_threads = processes):                 
    for i in prange(n, schedule = 'guided'):
      ls = line_starts[i];
      for j in range(line_shape):
        p = ls + j * line_stride;
        temp = 0;
        s = -k1 if j >= k1 else -j;
        e = k2 if j <= line_shape_minus_k2 else line_shape - j;
        for m in range(s, e):
          temp = temp + (source[p + m * line_stride] * kernel_pointer[m]);
        sink[p] = <sink_t>(temp);
  
  return;


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void correlate_separable(source_t[:] source, index_t[:] source_shape, index_t[:] source_strides,
                       sink_t[:] sink, index_t[:] sink_strides, sink_t[:] kernel, index_t[:] kernel_shapes, int processes):
  
  cdef index_t n_dim = source_shape.shape[0];
  cdef index_t line_shape
  cdef index_t source_line_stride, sink_line_stride
  cdef index_t k, k1, k2, line_shape_minus_k2
  cdef index_t kernel_offset = 0
  cdef index_t n,d,e,i,j,p,q,m,s
  
  cdef index_t[:] source_starts; 
  cdef index_t[:] sink_starts;
  cdef index_t[:] position;
  
  cdef sink_t* kernel_pointer
  cdef double temp;
  cdef index_t source_s, sink_s
  
  for axis in range(n_dim):
    line_shape = source_shape[axis];
  
    source_line_stride = source_strides[axis];
    sink_line_stride   = sink_strides[axis];
  
    k = kernel_shapes[axis];
    k1 = k / 2;
    k2 = k - k1;
    line_shape_minus_k2 = line_shape - k2;
    
    n = 1;
    for d in range(n_dim):
      if d != axis:
        n = n * source_shape[d]; 
    
    source_starts = np.zeros(n, dtype=int); 
    sink_starts   = np.zeros(n, dtype=int);
    position = np.zeros(n_dim, dtype=int);
    
    if axis == 0:
      d = 1;
    else:
      d = 0;
  
    for i in range(n):
      p = 0; q= 0;
      for e in range(n_dim):
        if e != axis:
          p = p + source_strides[e] * position[e];
          q = q + sink_strides[e] * position[e];
      source_starts[i] = p;
      sink_starts[i] = q;
      
      #print('i=%d' % (i));
      #for m in range(n_dim):
      #  print('position[%d] = %d' % (m, position[m]))
      #print('source,sink_start = %d,%d' % (p,q));
      
      e = d;
      while e < n_dim:
        position[e] += 1;
        if position[e] >= source_shape[e]:
          position[e] = 0;
          e = e + 1;
          if e == axis:
            e = e + 1;
        else:
          break;
    
    kernel_pointer = &kernel[kernel_offset + k1];
    kernel_offset += kernel_shapes[axis];
    
    with nogil, parallel(num_threads = processes):                 
      for i in prange(n, schedule = 'guided'):
        source_s = source_starts[i];
        sink_s   = sink_starts[i];
        for j in range(line_shape):
          p = source_s + j * source_line_stride;
          q = sink_s   + j * sink_line_stride;
          #with gil:
          #  print('i=%d, p,q=%d,%d' % (i,p,q));
          temp = 0;
          s = -k1 if j >= k1 else -j;
          e = k2 if j <= line_shape_minus_k2 else line_shape - j;
          for m in range(s, e):
            temp = temp + (source[p + m * source_line_stride] * kernel_pointer[m]);
          sink[q] = <sink_t>(temp);
    
    return;