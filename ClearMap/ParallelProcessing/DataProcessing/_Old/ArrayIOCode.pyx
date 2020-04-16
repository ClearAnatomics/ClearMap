"""
Cython Code for the ParallelReadWrite module.
"""
__author__    = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__   = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__copyright__ = 'Copyright (c) 2018 by Christoph Kirst, The Rockefeller University, New York City'


cimport cython
from cython.parallel import prange, parallel

import numpy as np
cimport numpy as np

from libc.stdio cimport FILE, fopen, fread, fwrite, fclose, fseek, SEEK_SET 


ctypedef fused source_t:
  np.int_t
  np.int32_t
  np.int64_t
  np.uint8_t
  np.uint16_t
  np.uint32_t
  np.float32_t
  np.float64_t

ctypedef Py_ssize_t index_t


#cdef extern from "stdio.h":
#    int printf(char *format, ...) nogil


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef read(source_t[:] data, char* filename, index_t offset, int blocks, int processes):
  cdef index_t p
  cdef index_t nbuf = data.nbytes;
  cdef index_t nblocks = min(nbuf, blocks);
  #printf("reading nbuf = %d nblocks = %d processes = %d\n", nbuf, nblocks, processes);

  cdef index_t[:] ranges = np.array(np.linspace(0, nbuf, nblocks + 1), dtype = np.int);
  cdef index_t[:] sizes  = np.zeros(nblocks, dtype = np.int);
  for p in range(nblocks):
    sizes[p] = ranges[p+1] - ranges[p];
    
  cdef FILE* fid
  cdef char* data_ptr = <char*> &data[0];
    
  with nogil, parallel(num_threads = processes):     
    for p in prange(nblocks, schedule = 'guided'):
        #printf("reading %d / %d\n", p, nblocks);
        fid = fopen(filename, "rb");
        fseek(fid, ranges[p] + offset, SEEK_SET);
        fread(data_ptr + ranges[p], 1, sizes[p], fid);
        fclose(fid);
  
  return;


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef write(source_t[:] data, char* filename, Py_ssize_t offset, int blocks, int processes):
  cdef Py_ssize_t p
  cdef Py_ssize_t nbuf = data.nbytes;
  cdef Py_ssize_t nblocks = min(nbuf, blocks);
  #printf("writing nbuf = %d nblocks = %d processes = %d\n", nbuf, nblocks, processes);

  cdef Py_ssize_t[:] ranges = np.array(np.linspace(0, nbuf, nblocks + 1), dtype = np.int);
  cdef Py_ssize_t[:] sizes  = np.zeros(nblocks, dtype = np.int);
  for p in range(nblocks):
    sizes[p] = ranges[p+1] - ranges[p];
    
  cdef FILE* fid
  cdef char* data_ptr = <char*> &data[0];
    
  with nogil, parallel(num_threads = processes):     
    for p in prange(nblocks, schedule = 'guided'):
        #printf("writing %d / %d\n", p, nblocks);
        fid = fopen(filename, "rb+");
        fseek(fid, ranges[p] + offset, SEEK_SET);
        fwrite(data_ptr + ranges[p], 1, sizes[p], fid);
        fclose(fid);
        
  return;
  
