#cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False, cdivision=True

"""
PercentileCode
==============

Cython code for the percentile rank filters.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'

cimport numpy as cnp

from libc.math cimport sqrt

from ClearMap.ImageProcessing.Filter.Rank.RankCoreCode cimport index_t, sink_t, source_t, rank_core, rank_core_masked


cdef inline source_t _max(source_t a, source_t b) nogil:
  return a if a >= b else b

cdef inline source_t _min(source_t a, source_t b) nogil:
  return a if a <= b else b


###############################################################################
### Autolevel
###############################################################################

cdef inline void kernel_autolevel(sink_t* sink, index_t* histo, index_t pop, source_t g, 
                                  index_t max_bin, index_t* p, double* q) nogil:

    cdef index_t i, imin, imax, sum, 
    cdef double delta

    if pop:
        sum = 0
        q[1] = 1.0 - q[1]
        for i in range(max_bin):
            sum += histo[i]
            if sum > q[0] * pop:
                imin = i
                break
        sum = 0
        for i in range(max_bin - 1, -1, -1):
            sum += histo[i]
            if sum > q[1] * pop:
                imax = i
                break

        delta = imax - imin
        if delta > 0:
            sink[0] = <sink_t>((max_bin - 1) * (_min(_max(imin, g), imax)
                                           - imin) / delta)
        else:
            sink[0] = <sink_t>g
    else:
        sink[0] = <sink_t>0


def autolevel(source_t[:, :, :] source, char[:, :, :] selem,
               sink_t[:, :, :, :] sink, 
               index_t max_bin, index_t[:] p, double[:] q):

  rank_core(kernel_autolevel[sink_t, index_t, source_t], source, selem, 
            sink, max_bin, p, q)

def autolevel_masked(source_t[:, :, :] source, char[:, :, :] selem,
                      char[:, :, :] mask, sink_t[:, :, :, :] sink,
                      index_t max_bin, index_t[:] p, double[:] q):

  rank_core_masked(kernel_autolevel[sink_t, index_t, source_t], source, selem, 
                   mask, sink, max_bin, p, q)


###############################################################################
### Gradient
###############################################################################

cdef inline void kernel_gradient(sink_t* sink, index_t* histo, index_t pop, source_t g, 
                                 index_t max_bin, index_t* p, double* q) nogil:

    cdef index_t i, imin, imax, sum, delta

    if pop:
        sum = 0
        q[1] = 1.0 - q[1]
        for i in range(max_bin):
            sum += histo[i]
            if sum >= q[0] * pop:
                imin = i
                break
        sum = 0
        for i in range(max_bin - 1, -1, -1):
            sum += histo[i]
            if sum >= q[1] * pop:
                imax = i
                break

        sink[0] = <sink_t>(imax - imin)
    else:
        sink[0] = <sink_t>0


def gradient(source_t[:, :, :] source, char[:, :, :] selem,
               sink_t[:, :, :, :] sink, 
               index_t max_bin, index_t[:] p, double[:] q):

  rank_core(kernel_gradient[sink_t, index_t, source_t], source, selem,
            sink, max_bin, p, q)

def gradient_masked(source_t[:, :, :] source, char[:, :, :] selem,
                     char[:, :, :] mask, sink_t[:, :, :, :] sink,
                     index_t max_bin, index_t[:] p, double[:] q):

  rank_core_masked(kernel_gradient[sink_t, index_t, source_t], source, selem, 
                   mask, sink, max_bin, p, q)


###############################################################################
### Mean
###############################################################################

cdef inline void kernel_mean(sink_t* sink, index_t* histo, index_t pop, source_t g, 
                             index_t max_bin, index_t* p, double* q) nogil:

    cdef index_t i, sum, mean, n

    if pop:
        sum = 0
        mean = 0
        n = 0
        for i in range(max_bin):
            sum += histo[i]
            if (sum >= q[0] * pop) and (sum <= q[1] * pop):
                n += histo[i]
                mean += histo[i] * i

        if n > 0:
            sink[0] = <sink_t>(mean / n)
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
### Sum
###############################################################################

cdef inline void kernel_sum(sink_t* sink, index_t* histo, index_t pop, source_t g, 
                            index_t max_bin, index_t* p, double* q) nogil:
    cdef index_t i, n
    cdef double sum, sum_g

    if pop:
        sum = 0
        sum_g = 0
        n = 0
        for i in range(max_bin):
            sum += histo[i]
            if (sum >= q[0] * pop) and (sum <= q[1] * pop):
                n += histo[i]
                sum_g += histo[i] * i

        if n > 0:
            sink[0] = <sink_t>sum_g
        else:
            sink[0] = <sink_t>0
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
### Sum bilateral
###############################################################################

cdef inline void kernel_sum_above(sink_t* sink, index_t* histo, index_t pop, source_t g, 
                                 index_t max_bin, index_t* p, double* q) nogil:
    cdef index_t n, i, i_high
    cdef double sum, sum_high
    #cdef double p0 = q[0] * pop;
    cdef double p_high = q[0] * pop;
    
    if pop:
        sum = 0
        sum_high = 0
        for i in range(max_bin):
            sum += histo[i]
            if (sum >= p_high):
               sum_high += <double>(histo[i]) * i;
            elif g <= i:
               sink[0] = <sink_t>g
               return;
        sink[0] = <sink_t>sum_high
    else:
        sink[0] = <sink_t>0


def sum_above(source_t[:, :, :] source, char[:, :, :] selem,
              sink_t[:, :, :, :] sink, 
              index_t max_bin, index_t[:] p, double[:] q):

  rank_core(kernel_sum_above[sink_t, index_t, source_t], source, selem,
            sink, max_bin, p, q)

def sum_above_masked(source_t[:, :, :] source, char[:, :, :] selem,
                     char[:, :, :] mask, sink_t[:, :, :, :] sink,
                     index_t max_bin, index_t[:] p, double[:] q):

  rank_core_masked(kernel_sum_above[sink_t, index_t, source_t], source, selem, 
                   mask, sink, max_bin, p, q)



###############################################################################
### Subtract mean
###############################################################################

cdef inline void kernel_subtract_mean(sink_t* sink, index_t* histo, index_t pop, source_t g, 
                                      index_t max_bin, index_t* p, double* q) nogil:
  cdef index_t i, sum, n
  cdef double mean

  if pop:
    sum = 0
    mean = 0
    n = 0
    for i in range(max_bin):
        sum += histo[i]
        if (sum >= q[0] * pop) and (sum <= q[1] * pop):
            n += histo[i]
            mean += histo[i] * i
    if n > 0:
        mean /= n;
        sink[0] = <sink_t>((g - mean + max_bin) * 0.5) #to avoid under/overflows ?
    else:
        sink[0] = <sink_t>0
  else:
    sink[0] = <sink_t>0


def subtract_mean(source_t[:, :, :] source, char[:, :, :] selem,
               sink_t[:, :, :, :] sink, 
               index_t max_bin, index_t[:] p, double[:] q):

  rank_core(kernel_subtract_mean[sink_t, index_t, source_t], source, selem,
            sink, max_bin, p, q)

def subtract_mean_masked(source_t[:, :, :] source, char[:, :, :] selem,
                          char[:, :, :] mask, sink_t[:, :, :, :] sink,
                          index_t max_bin, index_t[:] p, double[:] q):

  rank_core_masked(kernel_subtract_mean[sink_t, index_t, source_t], source, selem, 
                   mask, sink, max_bin, p, q)


###############################################################################
### Enhance contrast
###############################################################################

cdef inline void kernel_enhance_contrast(sink_t* sink, index_t* histo, index_t pop, source_t g, 
                                         index_t max_bin, index_t* p, double* q) nogil:

    cdef index_t i, imin, imax, sum, delta

    if pop:
        sum = 0
        q[1] = 1.0 - q[1]
        for i in range(max_bin):
            sum += histo[i]
            if sum > q[0] * pop:
                imin = i
                break
        sum = 0
        for i in range(max_bin - 1, -1, -1):
            sum += histo[i]
            if sum > q[1] * pop:
                imax = i
                break
        if g > imax:
            sink[0] = <sink_t>imax
        if g < imin:
            sink[0] = <sink_t>imin
        if imax - g < g - imin:
            sink[0] = <sink_t>imax
        else:
            sink[0] = <sink_t>imin
    else:
        sink[0] = <sink_t>0


def enhance_contrast(source_t[:, :, :] source, char[:, :, :] selem,
                      sink_t[:, :, :, :] sink, 
                      index_t max_bin, index_t[:] p, double[:] q):

  rank_core(kernel_enhance_contrast[sink_t, index_t, source_t], source, selem,
            sink, max_bin, p, q)

def enhance_contrast_masked(source_t[:, :, :] source, char[:, :, :] selem,
                             char[:, :, :] mask, sink_t[:, :, :, :] sink,
                             index_t max_bin, index_t[:] p, double[:] q):

  rank_core_masked(kernel_enhance_contrast[sink_t, index_t, source_t], source, selem, 
                   mask, sink, max_bin, p, q)


###############################################################################
### Percentile
###############################################################################

cdef inline void kernel_percentile(sink_t* sink, index_t* histo, index_t pop, source_t g, 
                                   index_t max_bin, index_t* p, double* q) nogil:

    cdef index_t i
    cdef index_t sum = 0
    cdef double percentile = q[0] * pop;
    
    if pop:
        if q[0] == 1:  # make sure q[0] = 1 returns the maximum filter
            for i in range(max_bin - 1, -1, -1):
                if histo[i]:
                    break
        else:
            for i in range(max_bin):
                sum += histo[i]
                if sum >= percentile:
                    break
        sink[0] = <sink_t>i
    else:
        sink[0] = <sink_t>0



def percentile(source_t[:, :, :] source, char[:, :, :] selem,
                      sink_t[:, :, :, :] sink, 
                      index_t max_bin, index_t[:] p, double[:] q):

  rank_core(kernel_percentile[sink_t, index_t, source_t], source, selem,
            sink, max_bin, p, q)

def percentile_masked(source_t[:, :, :] source, char[:, :, :] selem,
                             char[:, :, :] mask, sink_t[:, :, :, :] sink,
                             index_t max_bin, index_t[:] p, double[:] q):

  rank_core_masked(kernel_percentile[sink_t, index_t, source_t], source, selem, 
                   mask, sink, max_bin, p, q)

###############################################################################
### Autolevel
###############################################################################

cdef inline void kernel_pop(sink_t* sink, index_t* histo, index_t pop, source_t g, 
                            index_t max_bin, index_t* p, double* q) nogil:

    cdef index_t i, sum, n

    if pop:
        sum = 0
        n = 0
        for i in range(max_bin):
            sum += histo[i]
            if (sum >= q[0] * pop) and (sum <= q[1] * pop):
                n += histo[i]
        sink[0] = <sink_t>n
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
### Autolevel
###############################################################################

cdef inline void kernel_threshold(sink_t* sink, index_t* histo, index_t pop, source_t g, 
                                  index_t max_bin, index_t* p, double* q) nogil:

    cdef int i
    cdef index_t sum = 0

    if pop:
        for i in range(max_bin):
            sum += histo[i]
            if sum >= q[0] * pop:
                break

        sink[0] = <sink_t>((max_bin - 1) * (g >= i))
    else:
        sink[0] = <sink_t>0


def threshold(source_t[:, :, :] source, char[:, :, :] selem,
               sink_t[:, :, :, :] sink, 
               index_t max_bin, index_t[:] p, double[:] q):

  rank_core(kernel_threshold[sink_t, index_t, source_t], source, selem,
            sink, max_bin, p, q)

def threshold_masked(source_t[:, :, :] source, char[:, :, :] selem,
                     char[:, :, :] mask, sink_t[:, :, :, :] sink,
                     index_t max_bin, index_t[:] p, double[:] q):

  rank_core_masked(kernel_threshold[sink_t, index_t, source_t], source, selem, 
                   mask, sink, max_bin, p, q)
















