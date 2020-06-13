#cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False, cdivision=True

"""
BilateralCode
=============

Cython code for the bilateral rank filters.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


cimport numpy as cnp
from libc.math cimport log, exp, sqrt

from ClearMap.ImageProcessing.Filter.Rank.RankCoreCode cimport index_t, sink_t, source_t, rank_core, rank_core_masked

cdef extern from "stdio.h":
    int printf(char *format, ...) nogil


###############################################################################
### Mean
###############################################################################

cdef inline void kernel_mean(sink_t* sink, index_t* histo, index_t pop, source_t g, 
                             index_t max_bin, index_t* p, double* q) nogil:
    cdef index_t i
    cdef double bilat_pop = 0
    cdef double mean = 0

    if pop:
        for i in range(max_bin):
            if (g > (i - p[0])) and (g < (i + p[1])):
                bilat_pop += histo[i]
                mean += histo[i] * i
        if bilat_pop:
            sink[0] = <sink_t>(mean / bilat_pop)
        else:
            sink[0] = <sink_t>0
    else:
        sink[0] = <sink_t>0


def mean(source_t[:, :, :] source, char[:, :, :] selem,
          sink_t[:, :, :, :] sink, 
          index_t max_bin, index_t[:] p, double[:] q):

  rank_core(kernel_mean[sink_t, index_t, source_t], source, selem, 
            sink, max_bin, p, q)

def mean_masked(source_t[:, :, :] source, char[:, :, :] selem,
                 char[:, :, :] mask, sink_t[:, :, :, :] sink,
                 index_t max_bin, index_t[:] p, double[:] q):

  rank_core_masked(kernel_mean[sink_t, index_t, source_t], source, selem, 
                   mask, sink, max_bin, p, q)


###############################################################################
### Pop
############################################################################### 

cdef inline void kernel_pop(sink_t* sink, index_t* histo, index_t pop, source_t g, 
                             index_t max_bin, index_t* p, double* q) nogil:
    cdef index_t i
    cdef double bilat_pop = 0

    if pop:
        for i in range(max_bin):
            if (g > (i - p[0])) and (g < (i + p[1])):
                bilat_pop += histo[i]
        sink[0] = <sink_t>bilat_pop
    else:
        sink[0] = <sink_t>0


def pop(source_t[:, :, :] source, char[:, :, :] selem,
         sink_t[:, :, :, :] sink, 
         index_t max_bin, index_t[:] p, double[:] q):

  rank_core(kernel_pop[sink_t, index_t, source_t], source, selem, 
            sink, max_bin, p, q)

def pop_masked(source_t[:, :, :] source, char[:, :, :] selem,
                 char[:, :, :] mask, sink_t[:, :, :, :] sink,
                 index_t max_bin, index_t[:] p, double[:] q):

  rank_core_masked(kernel_pop[sink_t, index_t, source_t], source, selem, 
                   mask, sink, max_bin, p, q)


###############################################################################
### Sum
###############################################################################

cdef inline void kernel_sum(sink_t* sink, index_t* histo, index_t pop, source_t g, 
                            index_t max_bin, index_t* p, double* q) nogil:
    cdef index_t i, s, e
    cdef double bilat_pop = 0
    cdef double sum = 0

    if pop:
      s = g - p[0];
      s = 0 if s < 0 else s;
      e = g + p[1] + 1;
      e = max_bin if e > max_bin else e;
      for i in range(s,e):
        sum += <double>(histo[i]) * i;
      sink[0] = <sink_t>sum
    else:
      sink[0] = <sink_t>0


def sum(source_t[:, :, :] source, char[:, :, :] selem,
         sink_t[:, :, :, :] sink, 
         index_t max_bin, index_t[:] p, double[:] q):

  rank_core(kernel_sum[sink_t, index_t, source_t], source, selem, 
            sink, max_bin, p, q)

def sum_masked(source_t[:, :, :] source, char[:, :, :] selem,
               char[:, :, :] mask, sink_t[:, :, :, :] sink,
               index_t max_bin, index_t[:] p, double[:] q):

  rank_core_masked(kernel_sum[sink_t, index_t, source_t], source, selem, 
                   mask, sink, max_bin, p, q)


###############################################################################
### Sum relative
###############################################################################

cdef inline void kernel_sum_relative(sink_t* sink, index_t* histo, index_t pop, source_t g, 
                            index_t max_bin, index_t* p, double* q) nogil:
    cdef index_t i, s, e
    cdef double bilat_pop = 0
    cdef double sum = 0

    if pop:
      s = <int>(g * q[0]);
      s = 0 if s < 0 else s;
      e = <int>(g * q[1] + 1);
      e = max_bin if e > max_bin else e;
      for i in range(s,e):
        sum += <double>(histo[i]) * i;
        bilat_pop += 1;
      if bilat_pop > 0:
        sink[0] = <sink_t>(sum/bilat_pop);
      else:
        sink[0] = <sink_t>0
    else:
      sink[0] = <sink_t>0


def sum_realtive(source_t[:, :, :] source, char[:, :, :] selem,
                 sink_t[:, :, :, :] sink, 
                 index_t max_bin, index_t[:] p, double[:] q):

  rank_core(kernel_sum_relative[sink_t, index_t, source_t], source, selem, 
            sink, max_bin, p, q)

def sum_relaive_masked(source_t[:, :, :] source, char[:, :, :] selem,
                       char[:, :, :] mask, sink_t[:, :, :, :] sink,
                       index_t max_bin, index_t[:] p, double[:] q):

  rank_core_masked(kernel_sum_relative[sink_t, index_t, source_t], source, selem, 
                   mask, sink, max_bin, p, q)


###############################################################################
### Mean scale
###############################################################################

cdef inline void kernel_mean_scale(sink_t* sink, index_t* histo, index_t pop, source_t g, 
                                   index_t max_bin, index_t* p, double* q) nogil:

    cdef index_t i, s = 0, e = 0, imax
    cdef double bilat_pop = 0
    cdef double mean = 0

    if pop:
      s = <index_t>(g - (p[0] + g * q[0]));
      s = s if s <= max_bin else max_bin;
      s = s if s >= 0 else 0;
      
      e = <index_t>(g + (p[1] + g * q[1]) + 1);
      e = e if e <= max_bin else max_bin;
      e = e if e >= 0 else 0;             
      for i in range(e,s,-1):
        if histo[i]:
          imax = i;
        #bilat_pop += histo[i]
        #mean += histo[i] * i
      if bilat_pop:
        sink[0] = <sink_t>imax #(mean / bilat_pop)
      else:
        sink[0] = <sink_t>imax
    else:
        sink[0] = <sink_t>0

    #if g > 0:
    #  printf('o=%d, m=%f, p=%f, s=%d, e=%d, g=%d, p[0]=%d, p[1]=%d, p0=%f, p1=%f\n', sink[0], mean, bilat_pop, s, e, g, p[0], p[1], p0, p1);


def mean_scale(source_t[:, :, :] source, char[:, :, :] selem,
                sink_t[:, :, :, :] sink, 
                index_t max_bin, index_t[:] p, double[:] q):

  rank_core(kernel_mean_scale[sink_t, index_t, source_t], source, selem, 
            sink, max_bin, p, q)

def mean_scale_masked(source_t[:, :, :] source, char[:, :, :] selem,
                 char[:, :, :] mask, sink_t[:, :, :, :] sink,
                 index_t max_bin, index_t[:] p, double[:] q):

  rank_core_masked(kernel_mean_scale[sink_t, index_t, source_t], source, selem, 
                   mask, sink, max_bin, p, q)
