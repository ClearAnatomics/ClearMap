#cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False, cdivision=True
"""
ArrayProcessingCode
===================

Cython code for the ArrayProcessing module.
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
    
cpdef void apply_lut(source_int_t[:] source, sink_t[:] sink, sink_t[:] lut, int blocks, int processes):
  cdef index_t size = source.shape[0];
  cdef index_t nblocks = min(size, blocks);
  cdef index_t[:] ranges = np.array(np.linspace(0, size, nblocks + 1), dtype = int);
  cdef index_t p,i
  with nogil, parallel(num_threads = processes):                 
    for p in prange(nblocks, schedule = 'guided'):
      for i in range(ranges[p], ranges[p+1]):
        sink[i] = lut[source[i]];


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

cdef enum:
  MAX_DIMS = 10

cdef struct Iterator:
  index_t n_dim
  index_t shape[MAX_DIMS]
  index_t coordinates[MAX_DIMS]
  index_t strides[MAX_DIMS]
  index_t backstrides[MAX_DIMS]

cdef void init_iterator(Iterator* iterator, index_t n_dim, index_t[:] shape, index_t[:] strides) nogil:
  cdef index_t d
  iterator[0].n_dim = n_dim;
  for d in range(n_dim):
    iterator[0].shape[d]       = shape[d];
    iterator[0].coordinates[d] = 0;
    iterator[0].strides[d]     = strides[d];
    iterator[0].backstrides[d] = strides[d] * shape[d];

cdef void init_subspace_iterator(Iterator* iterator, index_t axis, index_t iter_axis) nogil:
  cdef index_t d, d_last = 0;
  for d in range(iterator[0].n_dim):
    if d != axis and d != iter_axis:
      if d != d_last:
        iterator[0].shape[d_last]       = iterator[0].shape[d]
        iterator[0].strides[d_last]     = iterator[0].strides[d];
        iterator[0].backstrides[d_last] = iterator[0].backstrides[d];
      d_last += 1;
  iterator.n_dim = d_last;


cdef inline void line_correlate(source_t* source_line, sink_t* sink_line, double* kernel_pointer,
                                index_t k1, index_t k2, index_t line_shape,
                                index_t source_stride_axis, index_t sink_stride_axis,
                                index_t n_sub_iterations,
                                Iterator source_line_iterator, Iterator sink_line_iterator) nogil:
  
  cdef index_t i,j,m,d
  cdef double temp
  cdef source_t* source_point;
  cdef sink_t* sink_point
  
  for i in range(n_sub_iterations):
    
    source_point = source_line;
    sink_point   = sink_line;
    
    #calculate 1d correlation
    #left border
    for j in range(k1):
      temp = 0;
      for m in range(-j,k2):
        temp = temp + <double>source_point[m * source_stride_axis] * kernel_pointer[m];
      sink_point[0] = <sink_t>temp;
      sink_point += sink_stride_axis;
      source_point += source_stride_axis;
    #center
    for j in range(k1,line_shape-k2):
      temp = 0;
      for m in range(-k1,k2):
        temp = temp + <double>source_point[m * source_stride_axis] * kernel_pointer[m];
      sink_point[0] = <sink_t>temp;
      sink_point += sink_stride_axis;
      source_point += source_stride_axis;
    #right border
    for j in range(line_shape-k2,line_shape):
      temp = 0;
      for m in range(-k1,line_shape-j):
        temp = temp + <double>source_point[m * source_stride_axis] * kernel_pointer[m];
      sink_point[0] = <sink_t>temp;
      sink_point += sink_stride_axis;
      source_point += source_stride_axis;
      
    #set line pointers to next line
    for d in range(source_line_iterator.n_dim):
      if source_line_iterator.coordinates[d] < source_line_iterator.shape[d]:
        source_line_iterator.coordinates[d]+=1;
        source_line += source_line_iterator.strides[d];
        sink_line_iterator.coordinates[d]+=1;
        sink_line += sink_line_iterator.strides[d];
        break;
      else:
        source_line_iterator.coordinates[d] = 0;
        source_line -= source_line_iterator.backstrides[d];
        sink_line_iterator.coordinates[d] = 0;
        sink_line -= sink_line_iterator.backstrides[d];#correlate 1d


#cdef print_iterator(Iterator iterator):
#  print('dim=%d' % iterator.n_dim);
#  for d in range(iterator.n_dim):
#    print('d=%d, shape=%d, coords=%d, strides=%d, backstrides=%d' % (d, iterator.shape[d], iterator.coordinates[d], iterator.strides[d], iterator.backstrides[d]));


cpdef void correlate_1d(source_t[:] source, index_t[:] source_shape, index_t[:] source_strides,
                        sink_t[:]   sink,   index_t[:] sink_shape,   index_t[:] sink_strides, 
                        double[:] kernel, int axis, int processes):
  
  cdef index_t n_dim = source_shape.shape[0];
  cdef index_t line_shape = source_shape[axis];
  
  cdef index_t source_stride_axis = source_strides[axis];
  cdef index_t sink_stride_axis   = sink_strides[axis];
  
  cdef index_t k = kernel.shape[0];
  cdef index_t k1 = k // 2;
  cdef index_t k2 = k - k1;
  assert(line_shape > k1 + k2)
  
  cdef source_t* source_pointer = &source[0];
  cdef sink_t*   sink_pointer   = &sink[0];
  cdef double*   kernel_pointer = &kernel[k1];
    
  #number of lines to iterate over
  cdef index_t iteration_axis;
  cdef index_t n_iterations = 0;
  cdef index_t n_subiterations = 1;
  cdef max_shape = 0;
  for d in range(n_dim):
    if d != axis:
      if max_shape < source_shape[d]:
        iteration_axis = d;
        n_iterations = source_shape[d];
  for d in range(n_dim):
    if d != axis and d != iteration_axis:
      n_subiterations *= source_shape[d];
  
  #initilaize iterators
  cdef Iterator source_iterator;
  init_iterator(&source_iterator, n_dim, source_shape, source_strides);
  
  cdef Iterator sink_iterator;
  init_iterator(&sink_iterator, n_dim, sink_shape, sink_strides);
  
  #initialize sub-space iterators excluding axis and the iteration_axis
  cdef Iterator source_line_iterator
  init_iterator(&source_line_iterator, n_dim, source_shape, source_strides);
  init_subspace_iterator(&source_line_iterator, axis, iteration_axis);
  
  cdef Iterator sink_line_iterator
  init_iterator(&sink_line_iterator, n_dim, sink_shape, sink_strides);
  init_subspace_iterator(&sink_line_iterator, axis, iteration_axis);
  
  #debug
  #print('k1,k2,ls=%d,%d,%d' % (k1,k2,line_shape));
  #print('source_iterator:');
  #print_iterator(source_iterator);
  #print('source_line');
  #print_iterator(source_line_iterator);
  #print('niter=%d, nsub=%d' % (n_iterations, n_subiterations))
  #print('axis=%d, iaxis=%d' % (axis, iteration_axis));
  
  cdef index_t iteration
  cdef source_t* source_line
  cdef sink_t*   sink_line
  
  with nogil, parallel(num_threads = processes):                 
    for iteration in prange(n_iterations, schedule = 'static'):
      #init pointers to first line
      source_line = source_pointer + source_iterator.strides[iteration_axis] * iteration;
      sink_line   = sink_pointer   +   sink_iterator.strides[iteration_axis] * iteration;
      
      line_correlate(source_line, sink_line, kernel_pointer, k1, k2, line_shape,
                     source_stride_axis, sink_stride_axis, n_subiterations,
                     source_line_iterator, sink_line_iterator);
  
  return;
        


###############################################################################
### Utils
###############################################################################

cpdef index_t[:] block_sums_1d(source_t[:] source, int blocks, int processes):
  cdef index_t i, p
  cdef index_t size = source.shape[0], 
  
  # split array in sub arrays along last dim
  cdef index_t nblocks = min(size, blocks);
  cdef index_t[:] ranges = np.array(np.linspace(0, size, nblocks + 1), dtype = int);
  
  #calculate block sum
  cdef index_t[:] blocksums = np.zeros(nblocks, dtype = int);
  with nogil, parallel(num_threads = processes):     
    for p in prange(nblocks, schedule = 'guided'):  
      for i in range(ranges[p], ranges[p+1]):
        blocksums[p] += (source[i] > 0);
  
  return blocksums;


cpdef index_t[:] block_sums_2d(source_t[:,:] source, int blocks, int processes):
  cdef index_t i, j, p
  cdef index_t shape_0 = source.shape[0], 
  cdef index_t shape_1 = source.shape[1];
  
  # split array in ncpus sub arrays along last dim
  cdef index_t nblocks = min(shape_1, blocks);
  cdef index_t[:] ranges = np.array(np.linspace(0, shape_1, nblocks + 1), dtype = int);
  
  #cdef int s = np.sum(arr);
  cdef index_t[:] blocksums = np.zeros(nblocks, dtype = int);
  with nogil, parallel(num_threads = processes):     
    for p in prange(nblocks, schedule = 'guided'):  
      for i in range(shape_0):
          for j in range(ranges[p], ranges[p+1]):
            blocksums[p] += (source[i, j] > 0);
  
  return blocksums;


cpdef index_t[:] block_sums_3d(source_t[:,:,:] source, int blocks, int processes):
  cdef index_t i, j, k, p
  cdef index_t shape_0 = source.shape[0], 
  cdef index_t shape_1 = source.shape[1];
  cdef index_t shape_2 = source.shape[2];
  
  # split array in ncpus sub arrays along last dim
  cdef index_t nblocks = min(shape_2, blocks);
  cdef index_t[:] ranges = np.array(np.linspace(0, shape_2, nblocks + 1), dtype = int);
  
  #cdef int s = np.sum(arr);
  cdef index_t[:] blocksums = np.zeros(nblocks, dtype = int);
  with nogil, parallel(num_threads = processes):     
    for p in prange(nblocks, schedule = 'guided'):  
      for i in range(shape_0):
        for j in range(shape_1):
          for k in range(ranges[p], ranges[p+1]):
            blocksums[p] += (source[i, j, k] > 0);
  
  return blocksums;


cpdef np.ndarray[Py_ssize_t, ndim=1] neighbours(index_t[:] indices, int offset, int processes):
  cdef index_t n = indices.shape[0];
  cdef index_t p, i, plo, phi, target
  
  cdef index_t[:] exists = np.empty(n, dtype = int);
  
  cdef index_t lo = min(0, offset);
  cdef index_t hi = max(0, offset);
  
  with nogil, parallel(num_threads = processes): 
    for p in prange(n, schedule = 'guided'):
      plo = max(-1, p+lo-1);
      phi = min(n, p+hi+1);
      exists[p] = -1;
      target = indices[p] + offset;
      for i in range(p, plo, -1):
        if indices[i] == target:
          exists[p] = i;
          break;
        if indices[i] < target:
          break;
      for i in range(p, phi):
        if indices[i] == target:
          exists[p] = i;
          break;
        if indices[i] > target:
          break;
      
  cdef int s = 0;
  with nogil, parallel(num_threads = processes):                 
    for i in prange(n, schedule = 'guided'):
      s += (exists[i] >= 0);
  
  cdef np.ndarray[Py_ssize_t, ndim=2] out = np.empty((s,2), dtype = int);
  
  s = 0;            
  for i in range(n):
    if (exists[i] >= 0):
      out[s,0] = i;
      out[s,1] = exists[i];
      s += 1;
      
  return out;


###############################################################################
### Where
###############################################################################

cpdef void where_1d(source_t[:] source, index_t[:] where, index_t[:] sums, int blocks, int processes):
  cdef index_t i, p
  cdef index_t size = source.shape[0];
  
  # split array in ncpus sub arrays along last dim
  cdef index_t nblocks = min(size, blocks);
  cdef index_t[:] ranges = np.array(np.linspace(0, size, nblocks + 1), dtype = int);
  
  if sums is None:
    sums = block_sums_1d(source, nblocks, processes);
  
  cdef index_t[:] l = np.append([0], np.cumsum(sums));
  
  if where is None:
    where = np.zeros(np.sum(sums), dtype = int);
  
  with nogil, parallel(num_threads = processes): 
    for p in prange(nblocks, schedule = 'guided'):
      for i in range(ranges[p], ranges[p+1]):
        if source[i] > 0:
          #printf("%d, %d, %d\n", p, l[p], i);
          where[l[p]] = i; 
          l[p]+=1;
  
  return;


cpdef void where_2d(source_t[:,:] source, index_t[:,:] where, index_t[:] sums, int blocks, int processes):
  cdef index_t i, j, k, p
  cdef index_t shape_0 = source.shape[0];
  cdef index_t shape_1 = source.shape[1];
  
  # split array in ncpus sub arrays along last dim
  cdef index_t nblocks = min(shape_1, blocks);
  cdef index_t[:] ranges = np.array(np.linspace(0, shape_1, nblocks + 1), dtype = int);
  
  if sums is None:
    sums = block_sums_2d(source, nblocks, processes);
  
  cdef index_t[:] l = np.append([0], np.cumsum(sums));
  
  if where is None:
    where = np.zeros((np.sum(sums),2), dtype = int);
  
  with nogil, parallel(num_threads = processes): 
    for p in prange(nblocks, schedule = 'guided'):
      for i in range(shape_0):
        for j in range(ranges[p], ranges[p+1]):
          if source[i,j] > 0:
            where[l[p],0] = i; where[l[p],1] = j;
            l[p]+=1;
  
  return;


cpdef void where_3d(source_t[:,:,:] source, index_t[:,:] where, index_t[:] sums, int blocks, int processes):
  cdef index_t i, j, k, p
  cdef index_t shape_0 = source.shape[0];
  cdef index_t shape_1 = source.shape[1];
  cdef index_t shape_2 = source.shape[2];
  
  # split array in ncpus sub arrays along last dim
  cdef index_t nblocks = min(shape_2, blocks);
  cdef index_t[:] ranges = np.array(np.linspace(0, shape_2, nblocks + 1), dtype = int);
  
  if sums is None:
    sums = block_sums_3d(source, nblocks, processes);
  
  cdef index_t[:] l = np.append([0], np.cumsum(sums));
  
  if where is None:
    where = np.zeros((np.sum(sums),3), dtype = int);
  
  with nogil, parallel(num_threads = processes): 
    for p in prange(nblocks, schedule = 'guided'):
      for i in range(shape_0):
        for j in range(shape_1):
          for k in range(ranges[p], ranges[p+1]):
            if source[i,j,k] > 0:
              #printf("%d, %d, %d, %d, %d\n", p, l[p], i, j, k);
              where[l[p],0] = i; where[l[p],1] = j; where[l[p],2] = k;
              l[p]+=1;
  
  return;


###############################################################################
### IO
###############################################################################

from libc.stdio cimport FILE, fopen, fread, fwrite, fclose, fseek, SEEK_SET 


cpdef read(source_t[:] source, char* filename, index_t offset, int blocks, int processes):
  cdef index_t p
  cdef index_t nbuf = source.nbytes;
  cdef index_t nblocks = min(nbuf, blocks);
  #printf("loading nbuf = %d nblocks = %d processes = %d\n", nbuf, nblocks, processes);

  cdef index_t[:] ranges = np.array(np.linspace(0, nbuf, nblocks + 1), dtype = int);
  cdef index_t[:] sizes  = np.zeros(nblocks, dtype = int);
  for p in range(nblocks):
    sizes[p] = ranges[p+1] - ranges[p];
    
  cdef FILE* fid
  cdef char* source_ptr = <char*> &source[0];
    
  with nogil, parallel(num_threads = processes):     
    for p in prange(nblocks, schedule = 'guided'):
        #printf("loading %d / %d\n", p, nblocks);
        fid = fopen(filename, "rb");
        fseek(fid, ranges[p] + offset, SEEK_SET);
        fread(source_ptr + ranges[p], 1, sizes[p], fid);
        fclose(fid);
  
  return;


cpdef write(source_t[:] source, char* filename, index_t offset, int blocks, int processes):
  cdef index_t p
  cdef index_t nbuf = source.nbytes;
  cdef index_t nblocks = min(nbuf, blocks);
  #printf("saving nbuf = %d nblocks = %d processes = %d\n", nbuf, nblocks, processes);

  cdef index_t[:] ranges = np.array(np.linspace(0, nbuf, nblocks + 1), dtype = int);
  cdef index_t[:] sizes  = np.zeros(nblocks, dtype = int);
  for p in range(nblocks):
    sizes[p] = ranges[p+1] - ranges[p];
    
  cdef FILE* fid
  cdef char* source_ptr = <char*> &source[0];
    
  with nogil, parallel(num_threads = processes):     
    for p in prange(nblocks, schedule = 'guided'):
        #printf("loading %d / %d\n", p, nblocks);
        fid = fopen(filename, "rb+");
        fseek(fid, ranges[p] + offset, SEEK_SET);
        fwrite(source_ptr + ranges[p], 1, sizes[p], fid);
        fclose(fid);
        
  return;
  
  