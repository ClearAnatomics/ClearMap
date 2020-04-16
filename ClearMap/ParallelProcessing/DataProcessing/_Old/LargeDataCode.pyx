#cython: --compile-args=-fopenmp --link-args=-fopenmp --force -a
# -*- coding: utf-8 -*-
"""
Cython Code for the LargeData module
"""
__author__    = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__   = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__copyright__ = 'Copyright (c) 2017 by Christoph Kirst, The Rockefeller University, New York City'

import numpy as np
cimport numpy as np

from multiprocessing import cpu_count;
ncpus = cpu_count();


cimport cython
from cython.parallel import prange, parallel


ctypedef fused data_t:
  np.int_t
  np.int32_t
  np.int64_t
  np.uint8_t
  np.uint16_t
  np.uint32_t
  np.float32_t
  np.float64_t

ctypedef fused index_t:
  Py_ssize_t
index_T = np.int


cdef extern from "stdio.h":
    int printf(char *format, ...) nogil
    


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double sum(data_t[:] data, int processes) nogil:
    cdef Py_ssize_t i
    cdef Py_ssize_t I = data.shape[0];
    
    #cdef int s = np.sum(arr);
    cdef double s = 0;
    with nogil, parallel(num_threads = processes):                 
      for i in prange(I, schedule = 'guided'):
        s += data[i];
    
    return s;


### Summing

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[Py_ssize_t, ndim=1] blockSums1d(data_t[:] data, int blocks, int processes):
    cdef Py_ssize_t i, p
    cdef Py_ssize_t I = data.shape[0], 
    
    # split array in ncpus sub arrays along last dim
    cdef Py_ssize_t nblocks = min(I, blocks);
    cdef Py_ssize_t[:] ranges = np.array(np.linspace(0, I, nblocks + 1), dtype = index_T);
    
    #cdef int s = np.sum(arr);
    cdef np.ndarray[Py_ssize_t, ndim=1] blocksums = np.zeros(nblocks, dtype = index_T);
    with nogil, parallel(num_threads = processes):     
      for p in prange(nblocks, schedule = 'guided'):  
        for i in range(ranges[p], ranges[p+1]):
          blocksums[p] += (data[i] > 0);
    
    return blocksums;


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[Py_ssize_t, ndim=1] blockSums3d(data_t[:,:,:] data, int blocks, int processes):
    cdef Py_ssize_t i, j, k, p
    cdef Py_ssize_t I = data.shape[0], 
    cdef Py_ssize_t J = data.shape[1];
    cdef Py_ssize_t K = data.shape[2];
    
    # split array in ncpus sub arrays along last dim
    cdef Py_ssize_t nblocks = min(K, blocks);
    cdef Py_ssize_t[:] ranges = np.array(np.linspace(0, K, nblocks + 1), dtype = index_T);
    
    #cdef int s = np.sum(arr);
    cdef np.ndarray[Py_ssize_t, ndim=1] blocksums = np.zeros(nblocks, dtype = index_T);
    with nogil, parallel(num_threads = processes):     
      for p in prange(nblocks, schedule = 'guided'):  
        for i in range(I):
          for j in range(J):
            for k in range(ranges[p], ranges[p+1]):
              blocksums[p] += (data[i, j, k] > 0);
    
    return blocksums;


### Where

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void where1d(data_t[:] data, index_t[:] out, index_t[:] sums, int blocks, int processes):
    cdef index_t i, p
    cdef index_t I = data.shape[0];
    
    # split array in ncpus sub arrays along last dim
    cdef index_t nblocks = min(I, blocks);
    cdef index_t[:] ranges = np.array(np.linspace(0, I, nblocks + 1), dtype = index_T);
    
    if sums is None:
      sums = blockSums1d(data, nblocks, processes);
    cdef index_t s = 0;
    for i in range(nblocks):
      s+= np.sum(sums);
    
    cdef index_t[:] l = np.append([0], np.cumsum(sums));
    
    if out is None:
      out = np.zeros(s, dtype = index_T);
    
    with nogil, parallel(num_threads = processes): 
      for p in prange(nblocks, schedule = 'guided'):
        for i in range(ranges[p], ranges[p+1]):
          if data[i] > 0:
            #printf("%d, %d, %d\n", p, l[p], i);
            out[l[p]] = i; 
            l[p]+=1;
    
    return;


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void where3d(data_t[:,:,:] data, index_t[:,:] out, index_t[:] sums, int blocks, int processes):
    cdef index_t i, j, k, p
    cdef index_t I = data.shape[0];
    cdef index_t J = data.shape[1];
    cdef index_t K = data.shape[2];
    #print("%d, %d, %d" % (I,J,K));
    
    # split array in ncpus sub arrays along last dim
    cdef index_t nblocks = min(K, blocks);
    cdef index_t[:] ranges = np.array(np.linspace(0, K, nblocks + 1), dtype = index_T);
    
    if sums is None:
      sums = blockSums3d(data, nblocks, processes);
    cdef index_t s = 0;
    for i in range(nblocks):
      s+= np.sum(sums);
    cdef index_t[:] l = np.append([0], np.cumsum(sums));
    
    if out is None:
      out = np.zeros((s,3), dtype = index_T);
    
    with nogil, parallel(num_threads = processes): 
      for p in prange(nblocks, schedule = 'guided'):
        for i in range(I):
          for j in range(J):
            for k in range(ranges[p], ranges[p+1]):
              if data[i,j,k] > 0:
                #printf("%d, %d, %d, %d, %d\n", p, l[p], i, j, k);
                out[l[p],0] = i; out[l[p],1] = j; out[l[p],2] = k;
                l[p]+=1;
    
    return;





### Array manipulation

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void set1d(data_t[:] data, index_t[:] indices, data_t value, int processes) nogil:
  cdef index_t n = indices.shape[0];
  cdef index_t p
  with nogil, parallel(num_threads = processes): 
    for p in prange(n, schedule = 'guided'):
          data[indices[p]] = value;
  return;


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void take1d(data_t[:] data, index_t[:] indices, data_t[:] out, int processes):
  cdef index_t n = indices.shape[0];
  cdef index_t p
  
  if out is None:
    out = np.empty(n, dtype = data.dtype);  
  
  with nogil, parallel(num_threads = processes): 
    for p in prange(n, schedule = 'guided'):
      out[p] = data[indices[p]];
  return;

 
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void set1darray(data_t[:] data, index_t[:] indices, data_t[:] values, int processes) nogil:
  cdef index_t n = indices.shape[0];
  cdef index_t p
  with nogil, parallel(num_threads = processes): 
    for p in prange(n, schedule = 'guided'):
          data[indices[p]] = values[p];
  return;

  
### Matching

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void match1d(index_t[:] match, index_t[:] indices,  index_t[:] out):
  cdef index_t n = indices.shape[0];
  cdef index_t k = match.shape[0];
  cdef index_t i, j
  if out is None:
    out = np.empty(k, dtype = match.dtype);  
  
  j = 0;
  i = 0;
  while j < k and i < n:
    if match[j] == indices[i]:
      out[j] = i;
      j+=1;
      i+=1;
    elif match[j] > indices[i]:
      i+=1;
    else: # match[j] < indices[i]
      out[j] = -1;
      j+=1;
  
  for i in range(j,k):
    out[i] = -1;
  
  return;



### Neighbour detection

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[Py_ssize_t, ndim=1] neighbours(index_t[:] indices, int offset, int processes):
  cdef index_t n = indices.shape[0];
  cdef index_t p, i, plo, phi, target
  
  cdef index_t[:] exists = np.empty(n, dtype = index_T);
  
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
  
  cdef np.ndarray[Py_ssize_t, ndim=2] out = np.empty((s,2), dtype = index_T);
  
  s = 0;            
  for i in range(n):
    if (exists[i] >= 0):
      out[s,0] = i;
      out[s,1] = exists[i];
      s += 1;
      
  return out;


from libcpp.vector cimport vector

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[Py_ssize_t, ndim=1] neighbourlistMask(index_t[:] indices, Py_ssize_t center_id,
                                                   Py_ssize_t shape_x, Py_ssize_t shape_y, Py_ssize_t shape_z,  
                                                   Py_ssize_t stride_x, Py_ssize_t stride_y, Py_ssize_t stride_z,                        
                                                   data_t[:,:,:] mask):
                                                  
  cdef Py_ssize_t n = indices.shape[0];
  
  cdef Py_ssize_t mask_x = mask.shape[0];
  cdef Py_ssize_t mask_y = mask.shape[1];
  cdef Py_ssize_t mask_z = mask.shape[2];
  cdef Py_ssize_t left_x = mask_x / 2;
  cdef Py_ssize_t left_y = mask_y / 2;
  cdef Py_ssize_t left_z = mask_z / 2;
  cdef Py_ssize_t right_x = mask_x - left_x - 1;
  cdef Py_ssize_t right_y = mask_y - left_y - 1;
  cdef Py_ssize_t right_z = mask_z - left_z - 1;

  cdef int order = 1;
  if stride_x == 1:
    order = 1;
  elif stride_z == 1:
    order = -1;

  cdef Py_ssize_t center = indices[center_id];
  cdef Py_ssize_t center_x, center_y, center_z;
  cdef Py_ssize_t rest;
  if order > 0:
    center_z, rest     = divmod(center, stride_z);
    center_y, center_x = divmod(rest,   stride_y);
  else:
    center_x, rest     = divmod(center, stride_x);
    center_y, center_z = divmod(rest,   stride_y);
  
  cdef Py_ssize_t lo_x = max(0, center_x - left_x);
  cdef Py_ssize_t lo_y = max(0, center_y - left_y);
  cdef Py_ssize_t lo_z = max(0, center_z - left_z); 
  cdef Py_ssize_t hi_x = min(shape_x-1, center_x + right_x);
  cdef Py_ssize_t hi_y = min(shape_y-1, center_y + right_y);
  cdef Py_ssize_t hi_z = min(shape_z-1, center_z + right_z);
  
  cdef vector[Py_ssize_t] nbs;
  cdef left_max  = stride_x * left_x  + stride_y * left_y  + stride_z * left_z;
  cdef right_max = stride_x * right_x + stride_y * right_y + stride_z * right_z;
  cdef Py_ssize_t plo = max(-1,    center_id - left_max - 1);
  cdef Py_ssize_t phi = min(n + 1, center_id + right_max + 1);
  cdef Py_ssize_t ilo = max(0, center - left_max);
  cdef Py_ssize_t ihi =        center + right_max;
  
  
  cdef Py_ssize_t p, x = 0, y = 0, z = 0, r = 0, m = 0;
  
#  print order
#  print stride_x, stride_y, stride_z;
#  print center
#  print center_x, center_y, center_z
#  print 'll'
#  print left_x, left_y, left_z;
#  print right_x, right_y, right_z
#  print lo_x, hi_x
#  print lo_y, hi_y
#  print lo_z, hi_z
#  print plo, phi
  
  #with nogil, parallel(num_threads = processes): 
  for p in range(center_id, plo, -1):  # does not exclude center -> do in mask !?
    i = indices[p];
    if i < ilo:  #no more neighbours to test
      break;  
    
    if order > 0:
      z, r = divmod(i, stride_z);
      y, x = divmod(r, stride_y);
    else:
      x, r = divmod(i, stride_x);
      y, z = divmod(r, stride_y);
    
    if lo_x <= x <= hi_x:
      if lo_y <= y <= hi_y:
        if lo_z <= z <= hi_z:
          if mask[x-center_x+left_x, y-center_y+left_y, z-center_z+left_z] > 0:
            nbs.push_back(i);
            
  for p in range(center_id + 1, phi, 1):
    i = indices[p];
    if i > ihi:  #no more neighbours to test
      break;  
    
    if order > 0:
      z, r = divmod(i, stride_z);
      y, x = divmod(r, stride_y);
    else:
      x, r = divmod(i, stride_x);
      y, z = divmod(r, stride_y);
    
    if lo_x <= x <= hi_x:
      if lo_y <= y <= hi_y:
        if lo_z <= z <= hi_z:
          if mask[x-center_x+left_x, y-center_y+left_y, z-center_z+left_z] > 0:
            nbs.push_back(i);
  
  cdef np.ndarray[Py_ssize_t, ndim=1] out = np.empty((nbs.size(),), dtype = index_T);
  
  s = 0;            
  for i in range(nbs.size()):
    out[i] = nbs[i];
  
  return out;





@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[Py_ssize_t, ndim=1] neighbourlistRadius(index_t[:] indices, Py_ssize_t center_id,
                                                   Py_ssize_t shape_x, Py_ssize_t shape_y, Py_ssize_t shape_z,  
                                                   Py_ssize_t stride_x, Py_ssize_t stride_y, Py_ssize_t stride_z,                        
                                                   Py_ssize_t radius_x, Py_ssize_t radius_y, Py_ssize_t radius_z):
                                                  
  cdef Py_ssize_t n = indices.shape[0];
  
  cdef Py_ssize_t left_x = radius_x;
  cdef Py_ssize_t left_y = radius_y;
  cdef Py_ssize_t left_z = radius_z;
  cdef Py_ssize_t right_x = radius_x;
  cdef Py_ssize_t right_y = radius_y;
  cdef Py_ssize_t right_z = radius_z;

  cdef int order = 1;
  if stride_x == 1:
    order = 1;
  elif stride_z == 1:
    order = -1;

  cdef Py_ssize_t center = indices[center_id];
  cdef Py_ssize_t center_x, center_y, center_z;
  cdef Py_ssize_t rest;
  if order > 0:
    center_z, rest     = divmod(center, stride_z);
    center_y, center_x = divmod(rest,   stride_y);
  else:
    center_x, rest     = divmod(center, stride_x);
    center_y, center_z = divmod(rest,   stride_y);
  
  cdef Py_ssize_t lo_x = max(0, center_x - left_x);
  cdef Py_ssize_t lo_y = max(0, center_y - left_y);
  cdef Py_ssize_t lo_z = max(0, center_z - left_z); 
  cdef Py_ssize_t hi_x = min(shape_x-1, center_x + right_x);
  cdef Py_ssize_t hi_y = min(shape_y-1, center_y + right_y);
  cdef Py_ssize_t hi_z = min(shape_z-1, center_z + right_z);
  
  cdef vector[Py_ssize_t] nbs;
  cdef left_max  = stride_x * left_x  + stride_y * left_y  + stride_z * left_z;
  cdef right_max = stride_x * right_x + stride_y * right_y + stride_z * right_z;
  cdef Py_ssize_t plo = max(-1,    center_id - left_max - 1);
  cdef Py_ssize_t phi = min(n + 1, center_id + right_max + 1);
  cdef Py_ssize_t ilo = max(0, center - left_max);
  cdef Py_ssize_t ihi =        center + right_max;
  
  
  cdef Py_ssize_t p, x = 0, y = 0, z = 0, r = 0, m = 0;
  
#  print order
#  print stride_x, stride_y, stride_z;
#  print center
#  print center_x, center_y, center_z
#  print 'll'
#  print left_x, left_y, left_z;
#  print right_x, right_y, right_z
#  print lo_x, hi_x
#  print lo_y, hi_y
#  print lo_z, hi_z
#  print plo, phi
  
  #with nogil, parallel(num_threads = processes): 
  for p in range(center_id - 1, plo, -1): # explicitly exclude center !
    i = indices[p];
    if i < ilo:  #no more neighbours to test
      break;  
    
    if order > 0:
      z, r = divmod(i, stride_z);
      y, x = divmod(r, stride_y);
    else:
      x, r = divmod(i, stride_x);
      y, z = divmod(r, stride_y);
    
    if lo_x <= x <= hi_x:
      if lo_y <= y <= hi_y:
        if lo_z <= z <= hi_z:
          #if mask[x-center_x+left_x, y-center_y+left_y, z-center_z+left_z] > 0:
          nbs.push_back(i);
            
  for p in range(center_id + 1, phi, 1):
    i = indices[p];
    if i > ihi:  #no more neighbours to test
      break;  
    
    if order > 0:
      z, r = divmod(i, stride_z);
      y, x = divmod(r, stride_y);
    else:
      x, r = divmod(i, stride_x);
      y, z = divmod(r, stride_y);
    
    if lo_x <= x <= hi_x:
      if lo_y <= y <= hi_y:
        if lo_z <= z <= hi_z:
          #if mask[x-center_x+left_x, y-center_y+left_y, z-center_z+left_z] > 0:
          nbs.push_back(i);
  
  cdef np.ndarray[Py_ssize_t, ndim=1] out = np.empty((nbs.size(),), dtype = index_T);
  
  s = 0;            
  for i in range(nbs.size()):
    out[i] = nbs[i];
  
  return out;





### IO

from libc.stdio cimport FILE, fopen, fread, fwrite, fclose, fseek, SEEK_SET  #,  fwrite, fscanf, SEEK_END, ftell, stdout, stderr
#from libc.stdlib cimport malloc, free

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef load(data_t[:] data, char* filename, Py_ssize_t offset, int blocks, int processes):
  cdef Py_ssize_t p
  cdef Py_ssize_t nbuf = data.nbytes;
  cdef Py_ssize_t nblocks = min(nbuf, blocks);
  #printf("loading nbuf = %d nblocks = %d processes = %d\n", nbuf, nblocks, processes);

  cdef Py_ssize_t[:] ranges = np.array(np.linspace(0, nbuf, nblocks + 1), dtype = index_T);
  cdef Py_ssize_t[:] sizes  = np.zeros(nblocks, dtype = index_T);
  for p in range(nblocks):
    sizes[p] = ranges[p+1] - ranges[p];
    
  cdef FILE* fid
  cdef char* data_ptr = <char*> &data[0];
    
  with nogil, parallel(num_threads = processes):     
    for p in prange(nblocks, schedule = 'guided'):
        #printf("loading %d / %d\n", p, nblocks);
        fid = fopen(filename, "rb");
        fseek(fid, ranges[p] + offset, SEEK_SET);
        fread(data_ptr + ranges[p], 1, sizes[p], fid);
        fclose(fid);
  
  return;


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef save(data_t[:] data, char* filename, Py_ssize_t offset, int blocks, int processes):
  cdef Py_ssize_t p
  cdef Py_ssize_t nbuf = data.nbytes;
  cdef Py_ssize_t nblocks = min(nbuf, blocks);
  #printf("saving nbuf = %d nblocks = %d processes = %d\n", nbuf, nblocks, processes);

  cdef Py_ssize_t[:] ranges = np.array(np.linspace(0, nbuf, nblocks + 1), dtype = index_T);
  cdef Py_ssize_t[:] sizes  = np.zeros(nblocks, dtype = index_T);
  for p in range(nblocks):
    sizes[p] = ranges[p+1] - ranges[p];
    
  cdef FILE* fid
  cdef char* data_ptr = <char*> &data[0];
    
  with nogil, parallel(num_threads = processes):     
    for p in prange(nblocks, schedule = 'guided'):
        #printf("loading %d / %d\n", p, nblocks);
        fid = fopen(filename, "rb+");
        fseek(fid, ranges[p] + offset, SEEK_SET);
        fwrite(data_ptr + ranges[p], 1, sizes[p], fid);
        fclose(fid);
        
  return;