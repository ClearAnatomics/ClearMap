#cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False, cdivision=True

"""
RankCode
=========

Cython code for the basic rank filters.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'

cimport numpy as np
from libc.math cimport log, exp, sqrt

from ClearMap.ImageProcessing.Filter.Rank.RankCoreCode cimport index_t, sink_t, source_t, rank_core, rank_core_masked

cdef inline int round(double r) nogil:
  return <int>((r + 0.5) if (r > 0.0) else (r - 0.5))


###############################################################################
### Autolevel
###############################################################################

cdef inline void kernel_autolevel(sink_t* sink, index_t* histo, index_t pop, source_t g, 
                                  index_t max_bin, index_t* p, double* q) nogil:
  cdef index_t i, imin, imax
  cdef double delta

  if pop:
    for i in range(max_bin - 1, -1, -1):
      if histo[i]:
        imax = i
        break
    for i in range(max_bin):
      if histo[i]:
        imin = i
        break
    delta = <double>(imax - imin)
    if delta > 0:
      sink[0] = <sink_t>(((max_bin - 1) * (g - imin)) / delta)
    else:
      sink[0] = <sink_t>0
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
### Bottomhat
###############################################################################

cdef inline void kernel_bottomhat(sink_t* sink, index_t* histo, index_t pop, source_t g, 
                                  index_t max_bin, index_t* p, double* q) nogil:   
  cdef index_t i

  if pop:
    for i in range(max_bin):
      if histo[i]:
        break
    sink[0] = <sink_t>(g - i)
  else:
    sink[0] = <sink_t>0


def bottomhat(source_t[:, :, :] source, char[:, :, :] selem,
              sink_t[:, :, :, :] sink, 
              index_t max_bin, index_t[:] p, double[:] q):

  rank_core(kernel_bottomhat[sink_t, index_t, source_t], source, selem, 
            sink, max_bin, p, q)

def bottomhat_masked(source_t[:, :, :] source, char[:, :, :] selem,
                     char[:, :, :] mask, sink_t[:, :, :, :] sink,
                     index_t max_bin, index_t[:] p, double[:] q):

  rank_core_masked(kernel_bottomhat[sink_t, index_t, source_t], source, selem, 
                   mask, sink, max_bin, p, q)


###############################################################################
### Equalize
###############################################################################

cdef inline void kernel_equalize(sink_t* sink, index_t* histo, index_t pop, source_t g, 
                                 index_t max_bin, index_t* p, double* q) nogil:
  cdef index_t i
  cdef double total = 0

  if pop:
    for i in range(max_bin):
      total += histo[i]
      if i >= g:
        break
    sink[0] = <sink_t>((max_bin - 1) * (total / pop))
  else:
    sink[0] = <sink_t>0


def equalize(source_t[:, :, :] source, char[:, :, :] selem,
             sink_t[:, :, :, :] sink, 
             index_t max_bin, index_t[:] p, double[:] q):

  rank_core(kernel_equalize[sink_t, index_t, source_t], source, selem,
            sink, max_bin, p, q)

def equalize_masked(source_t[:, :, :] source, char[:, :, :] selem,
                    char[:, :, :] mask, sink_t[:, :, :, :] sink,
                    index_t max_bin, index_t[:] p, double[:] q):

  rank_core_masked(kernel_equalize[sink_t, index_t, source_t], source, selem, 
                   mask, sink, max_bin, p, q)


###############################################################################
### Gradient
###############################################################################

cdef inline void kernel_gradient(sink_t* sink, index_t* histo, index_t pop, source_t g, 
                                 index_t max_bin, index_t* p, double* q) nogil:

  cdef index_t i, imin, imax

  if pop:
    for i in range(max_bin - 1, -1, -1):
      if histo[i]:
        imax = i
        break
    for i in range(max_bin):
      if histo[i]:
        imin = i
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

  cdef index_t i
  cdef double mean = 0

  if pop:
    for i in range(max_bin):
      mean += histo[i] * i
    sink[0] = <sink_t>(mean / pop)
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
### Geometric mean
###############################################################################

cdef inline void kernel_geometric_mean(sink_t* sink, index_t* histo, index_t pop, source_t g, 
                                       index_t max_bin, index_t* p, double* q) nogil:

  cdef index_t i
  cdef double mean = 0.

  if pop:
    for i in range(max_bin):
      if histo[i]:
        mean += (histo[i] * log(i+1))
    sink[0] = <sink_t>round(exp(mean / pop)-1)
  else:
    sink[0] = <sink_t>0


def geometric_mean(source_t[:, :, :] source, char[:, :, :] selem,
                   sink_t[:, :, :, :] sink, 
                   index_t max_bin, index_t[:] p, double[:] q):

  rank_core(kernel_geometric_mean[sink_t, index_t, source_t], source, selem,
            sink, max_bin, p, q)

def geometric_mean_masked(source_t[:, :, :] source, char[:, :, :] selem,
                          char[:, :, :] mask, sink_t[:, :, :, :] sink,
                          index_t max_bin, index_t[:] p, double[:] q):

  rank_core_masked(kernel_geometric_mean[sink_t, index_t, source_t], source, selem, 
                   mask, sink, max_bin, p, q)


###############################################################################
### Subtract mean
###############################################################################

cdef inline void kernel_subtract_mean(sink_t* sink, index_t* histo, index_t pop, source_t g, 
                                      index_t max_bin, index_t* p, double* q) nogil:

  cdef index_t i
  cdef double mean = 0

  if pop:
    for i in range(max_bin):
      mean += histo[i] * i
    sink[0] = <sink_t>((g - mean / pop) / 2. + 127)
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
### Median
###############################################################################

cdef inline void kernel_median(sink_t* sink, index_t* histo, index_t pop, 
                               source_t g, index_t max_bin, index_t* p, double* q) nogil:

  cdef index_t i
  cdef double total = pop / 2.0

  if pop:
    for i in range(max_bin):
      if histo[i]:
        total -= histo[i]
        if total < 0:
          sink[0] = <sink_t>i
          return
  else:
    sink[0] = <sink_t>0


def median(source_t[:, :, :] source, char[:, :, :] selem,
           sink_t[:, :, :, :] sink, 
           index_t max_bin, index_t[:] p, double[:] q):

  rank_core(kernel_median[sink_t, index_t, source_t], source, selem,
            sink, max_bin, p, q)

def median_masked(source_t[:, :, :] source, char[:, :, :] selem,
                  char[:, :, :] mask, sink_t[:, :, :, :] sink,
                  index_t max_bin, index_t[:] p, double[:] q):

  rank_core_masked(kernel_median[sink_t, index_t, source_t], source, selem, 
                   mask, sink, max_bin, p, q)


###############################################################################
### Maximum
###############################################################################

cdef inline void kernel_maximum(sink_t* sink, index_t* histo, index_t pop, source_t g, 
                                index_t max_bin, index_t* p, double* q) nogil:

  cdef index_t i

  if pop:
    for i in range(max_bin - 1, -1, -1):
      if histo[i]:
        sink[0] = <sink_t>i
        return
  else:
    sink[0] = <sink_t>0


def maximum(source_t[:, :, :] source, char[:, :, :] selem,
            sink_t[:, :, :, :] sink, 
            index_t max_bin, index_t[:] p, double[:] q):

  rank_core(kernel_maximum[sink_t, index_t, source_t], source, selem,
            sink, max_bin, p, q)

def maximum_masked(source_t[:, :, :] source, char[:, :, :] selem,
                   char[:, :, :] mask, sink_t[:, :, :, :] sink,
                   index_t max_bin, index_t[:] p, double[:] q):

  rank_core_masked(kernel_maximum[sink_t, index_t, source_t], source, selem, 
                   mask, sink, max_bin, p, q)


###############################################################################
### Minimum
###############################################################################

cdef inline void kernel_minimum(sink_t* sink, index_t* histo, index_t pop, 
                                source_t g, index_t max_bin, index_t* p, double* q) nogil:

  cdef index_t i

  if pop:
    for i in range(max_bin):
      if histo[i]:
        sink[0] = <sink_t>i
        return
  else:
    sink[0] = <sink_t>0


def minimum(source_t[:, :, :] source, char[:, :, :] selem,
            sink_t[:, :, :, :] sink, 
            index_t max_bin, index_t[:] p, double[:] q):

  rank_core(kernel_minimum[sink_t, index_t, source_t], source, selem,
            sink, max_bin, p, q)

def minimum_masked(source_t[:, :, :] source, char[:, :, :] selem,
                   char[:, :, :] mask, sink_t[:, :, :, :] sink,
                   index_t max_bin, index_t[:] p, double[:] q):

  rank_core_masked(kernel_minimum[sink_t, index_t, source_t], source, selem, 
                   mask, sink, max_bin, p, q)


###############################################################################
### MinMax
###############################################################################

cdef inline void kernel_minmax(sink_t* sink, index_t* histo, index_t pop, 
                               source_t g, index_t max_bin, index_t* p, double* q) nogil:

  cdef index_t i

  if pop:
    for i in range(max_bin):
      if histo[i]:
        sink[0] = <sink_t>i
        break
    for i in range(max_bin - 1, -1, -1):
      if histo[i]:
        sink[1] = <sink_t>i
        return
  else:
    sink[0] = <sink_t>0
    sink[1] = <sink_t>0


def minmax(source_t[:, :, :] source, char[:, :, :] selem,
           sink_t[:, :, :, :] sink, 
           index_t max_bin, index_t[:] p, double[:] q):

  rank_core(kernel_minmax[sink_t, index_t, source_t], source, selem,
            sink, max_bin, p, q)

def minmax_masked(source_t[:, :, :] source, char[:, :, :] selem,
                  char[:, :, :] mask, sink_t[:, :, :, :] sink,
                  index_t max_bin, index_t[:] p, double[:] q):

  rank_core_masked(kernel_minmax[sink_t, index_t, source_t], source, selem, 
                   mask, sink, max_bin, p, q)


###############################################################################
### Modal
###############################################################################

cdef inline void kernel_modal(sink_t* sink, index_t* histo, index_t pop, 
                              source_t g, index_t max_bin, index_t* p, double* q) nogil:

  cdef index_t hmax = 0, imax = 0

  if pop:
    for i in range(max_bin):
      if histo[i] > hmax:
        hmax = histo[i]
        imax = i
    sink[0] = <sink_t>imax
  else:
    sink[0] = <sink_t>0


def modal(source_t[:, :, :] source, char[:, :, :] selem,
          sink_t[:, :, :, :] sink, 
          index_t max_bin, index_t[:] p, double[:] q):

  rank_core(kernel_modal[sink_t, index_t, source_t], source, selem,
            sink, max_bin, p, q)

def modal_masked(source_t[:, :, :] source, char[:, :, :] selem,
                 char[:, :, :] mask, sink_t[:, :, :, :] sink,
                 index_t max_bin, index_t[:] p, double[:] q):

  rank_core_masked(kernel_modal[sink_t, index_t, source_t], source, selem, 
                   mask, sink, max_bin, p, q)


###############################################################################
### Enhance contrast
###############################################################################

cdef inline void kernel_enhance_contrast(sink_t* sink, index_t* histo, index_t pop, 
                                         source_t g, index_t max_bin, index_t* p, double* q) nogil:

  cdef index_t i, imin, imax

  if pop:
    for i in range(max_bin - 1, -1, -1):
      if histo[i]:
        imax = i
        break
    for i in range(max_bin):
      if histo[i]:
        imin = i
        break
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
### Population
###############################################################################

cdef inline void kernel_pop(sink_t* sink, index_t* histo, index_t pop, 
                            source_t g, index_t max_bin, index_t* p, double* q) nogil:

  sink[0] = <sink_t>pop


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

cdef inline void kernel_sum(sink_t* sink, index_t* histo, index_t pop, 
                            source_t g, index_t max_bin, index_t* p, double* q) nogil:

  cdef index_t i
  cdef index_t total = 0

  if pop:
    for i in range(max_bin):
      total += histo[i] * i
    sink[0] = <sink_t>total
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
### Threshold
###############################################################################

cdef inline void kernel_threshold(sink_t* sink, index_t* histo, index_t pop, 
                                  source_t g, index_t max_bin, index_t* p, double* q) nogil:

  cdef index_t i
  cdef double mean = 0

  if pop:
    for i in range(max_bin):
      mean += histo[i] * i
    sink[0] = <sink_t>(g > (mean / pop))
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


###############################################################################
### Tophat
###############################################################################

cdef inline void kernel_tophat(sink_t* sink, index_t* histo, index_t pop, 
                               source_t g, index_t max_bin, index_t* p, double* q) nogil:

  cdef index_t i

  if pop:
    for i in range(max_bin - 1, -1, -1):
      if histo[i]:
        break
    sink[0] = <sink_t>(i - g)
  else:
    sink[0] = <sink_t>0


def tophat(source_t[:, :, :] source, char[:, :, :] selem,
           sink_t[:, :, :, :] sink, 
           index_t max_bin, index_t[:] p, double[:] q):

  rank_core(kernel_tophat[sink_t, index_t, source_t], source, selem,
            sink, max_bin, p, q)

def tophat_masked(source_t[:, :, :] source, char[:, :, :] selem,
                  char[:, :, :] mask, sink_t[:, :, :, :] sink,
                  index_t max_bin, index_t[:] p, double[:] q):

  rank_core_masked(kernel_tophat[sink_t, index_t, source_t], source, selem, 
                   mask, sink, max_bin, p, q)


###############################################################################
### Noise filter
###############################################################################

cdef inline void kernel_noise_filter(sink_t* sink, index_t* histo, index_t pop, 
                                     source_t g, index_t max_bin, index_t* p, double* q) nogil:

  cdef index_t i
  cdef index_t min_i

  # early stop if at least one pixel of the neighborhood has the same g
  if histo[g] > 0:
    sink[0] = <sink_t>0

  for i in range(g, -1, -1):
    if histo[i]:
      break
  min_i = g - i
  for i in range(g, max_bin):
    if histo[i]:
      break
  if i - g < min_i:
    sink[0] = <sink_t>(i - g)
  else:
    sink[0] = <sink_t>min_i


def noise_filter(source_t[:, :, :] source, char[:, :, :] selem,
                 sink_t[:, :, :, :] sink, 
                 index_t max_bin, index_t[:] p, double[:] q):

  rank_core(kernel_noise_filter[sink_t, index_t, source_t], source, selem,
            sink, max_bin, p, q)

def noise_filter_masked(source_t[:, :, :] source, char[:, :, :] selem,
                        char[:, :, :] mask, sink_t[:, :, :, :] sink,
                        index_t max_bin, index_t[:] p, double[:] q):

  rank_core_masked(kernel_noise_filter[sink_t, index_t, source_t], source, selem, 
                   mask, sink, max_bin, p, q)


###############################################################################
### Entropy
###############################################################################

cdef inline void kernel_entropy(sink_t* sink, index_t* histo, index_t pop, 
                                source_t g, index_t max_bin, index_t* p, double* q) nogil:
  cdef index_t i
  cdef double e, pp

  if pop:
    e = 0.
    for i in range(max_bin):
      pp = histo[i] / pop
      if pp > 0:
        e -= pp * log(pp) / 0.6931471805599453
    sink[0] = <sink_t>e
  else:
    sink[0] = <sink_t>0

def entropy(source_t[:, :, :] source, char[:, :, :] selem,
            sink_t[:, :, :, :] sink, 
            index_t max_bin, index_t[:] p, double[:] q):

  rank_core(kernel_entropy[sink_t, index_t, source_t], source, selem,
            sink, max_bin, p, q)

def entropy_masked(source_t[:, :, :] source, char[:, :, :] selem,
                   char[:, :, :] mask, sink_t[:, :, :, :] sink,
                   index_t max_bin, index_t[:] p, double[:] q):

  rank_core_masked(kernel_entropy[sink_t, index_t, source_t], source, selem, 
                   mask, sink, max_bin, p, q)


###############################################################################
### Otsu
###############################################################################

cdef inline void kernel_otsu(sink_t* sink, index_t* histo, index_t pop, 
                             source_t g, index_t max_bin, index_t* p, double* q) nogil:
  cdef index_t i
  cdef index_t max_i
  cdef double P, mu1, mu2, q1, new_q1, sigma_b, max_sigma_b
  cdef double mu = 0.0

  # compute local mean
  if pop:
    for i in range(max_bin):
      mu += histo[i] * i
    mu = mu / pop
  else:
    sink[0] = <sink_t>0

  # maximizing the between class variance
  max_i = 0
  q1 = histo[0] / pop
  mu1 = 0.
  max_sigma_b = 0.

  for i in range(1, max_bin):
    P = histo[i] / pop
    new_q1 = q1 + P
    if new_q1 > 0:
      mu1 = (q1 * mu1 + i * P) / new_q1
      mu2 = (mu - new_q1 * mu1) / (1. - new_q1)
      sigma_b = new_q1 * (1. - new_q1) * (mu1 - mu2) ** 2
      if sigma_b > max_sigma_b:
        max_sigma_b = sigma_b
        max_i = i
      q1 = new_q1
  
  sink[0] = <sink_t>max_i


def otsu(source_t[:, :, :] source, char[:, :, :] selem,
         sink_t[:, :, :, :] sink, 
         index_t max_bin, index_t[:] p, double[:] q):

  rank_core(kernel_otsu[sink_t, index_t, source_t], source, selem,
            sink, max_bin, p, q)

def otsu_masked(source_t[:, :, :] source, char[:, :, :] selem,
                char[:, :, :] mask, sink_t[:, :, :, :] sink,
                index_t max_bin, index_t[:] p, double[:] q):

  rank_core_masked(kernel_otsu[sink_t, index_t, source_t], source, selem, 
                   mask, sink, max_bin, p, q)


###############################################################################
### Std
###############################################################################

cdef inline void kernel_std(sink_t* sink, index_t* histo, index_t pop, 
                            source_t g, index_t max_bin, index_t* p, double* q) nogil:
  cdef index_t i
  cdef index_t max_i
  cdef double sigma = 0.0
  cdef double mu = 0.0

  # compute local standard deviation
  if pop:
    for i in range(max_bin):
       mu += histo[i] * i
    mu = mu / pop
  else:
    sink[0] = <sink_t>0

  for i in range(max_bin):
    sigma += histo[i] * (i - mu)**2;
  sigma /= pop;                     

  sink[0] = <sink_t>sqrt(sigma)


def std(source_t[:, :, :] source, char[:, :, :] selem,
        sink_t[:, :, :, :] sink, 
        index_t max_bin, index_t[:] p, double[:] q):

  rank_core(kernel_std[sink_t, index_t, source_t], source, selem,
            sink, max_bin, p, q)

def std_masked(source_t[:, :, :] source, char[:, :, :] selem,
               char[:, :, :] mask, sink_t[:, :, :, :] sink,
               index_t max_bin, index_t[:] p, double[:] q):

  rank_core_masked(kernel_std[sink_t, index_t, source_t], source, selem, 
                   mask, sink, max_bin, p, q)


###############################################################################
### Histogram
###############################################################################

cdef inline void kernel_histogram(sink_t* sink, index_t* histo, index_t pop, 
                                  source_t g, index_t max_bin, index_t* p, double* q) nogil:
  cdef index_t odepth = p[0];    
  cdef index_t i
  cdef index_t max_i
  cdef double scale
  if pop:
    scale = 1.0 / pop
    for i in range(odepth):
      sink[i] = <sink_t>(histo[i] * scale)
  else:
    for i in range(odepth):
      sink[i] = <sink_t>0


def histogram(source_t[:, :, :] source, char[:, :, :] selem,
              sink_t[:, :, :, :] sink, 
              index_t max_bin, index_t[:] p, double[:] q):

  rank_core(kernel_histogram[sink_t, index_t, source_t], source, selem,
            sink, max_bin, p, q)

def histogram_masked(source_t[:, :, :] source, char[:, :, :] selem,
                     char[:, :, :] mask, sink_t[:, :, :, :] sink,
                     index_t max_bin, index_t[:] p, double[:] q):

  rank_core_masked(kernel_histogram[sink_t, index_t, source_t], source, selem, 
                   mask, sink, max_bin, p, q)

