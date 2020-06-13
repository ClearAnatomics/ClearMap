#cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False, cdivision=True
"""
ParametricCode
==============

Cython code for the parametric rank filters.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


cimport numpy as np
from libc.math cimport sqrt

from ClearMap.ImageProcessing.Filter.Rank.RankCoreCode cimport index_t, sink_t, source_t, rank_core, rank_core_masked

cdef extern from "stdio.h":
  int printf(char *format, ...) nogil


###############################################################################
### Nilblack
###############################################################################

cdef inline void kernel_nilblack(sink_t* sink, index_t* histo, index_t pop, source_t g, 
                                 index_t max_bin, index_t* p, double* q) nogil:
    cdef index_t i, max_i
    cdef double sigma = 0.0
    cdef double mu = 0.0

    # compute local mean and standard deviation
    if pop:
        for i in range(max_bin):
            mu += histo[i] * i
        mu = mu / pop
    else:
        sink[0] = <sink_t>0

    for i in range(max_bin):
        sigma += histo[i] * (i - mu)**2;
    sigma /= pop;                     

    sink[0] = <sink_t>(mu + q[0] * sqrt(sigma))


def nilblack(source_t[:, :, :] source, char[:, :, :] selem,
               sink_t[:, :, :, :] sink, 
               index_t max_bin, index_t[:] p, double[:] q):

  rank_core(kernel_nilblack[sink_t, index_t, source_t], source, selem, 
            sink, max_bin, p, q)

def nilblack_masked(source_t[:, :, :] source, char[:, :, :] selem,
                      char[:, :, :] mask, sink_t[:, :, :, :] sink,
                      index_t max_bin, index_t[:] p, double[:] q):

  rank_core_masked(kernel_nilblack[sink_t, index_t, source_t], source, selem, 
                   mask, sink, max_bin, p, q)


###############################################################################
### Sauvola
###############################################################################

cdef inline void kernel_sauvola(sink_t* sink, index_t* histo, index_t pop, source_t g, 
                                 index_t max_bin, index_t* p, double* q) nogil:
    cdef index_t i, max_i
    cdef double sigma = 0.0
    cdef double mu = 0.0

    # compute local mean and standard deviation
    if pop:
        for i in range(max_bin):
          mu += histo[i] * i
        mu = mu / pop
    else:
        sink[0] = <sink_t>0

    for i in range(max_bin):
      sigma += histo[i] * (i - mu)**2;
    sigma /= pop;                     

    sink[0] = <sink_t>(mu * (1.0 + q[0] * (sqrt(sigma)/q[1] - 1.0)));


def sauvola(source_t[:, :, :] source, char[:, :, :] selem,
             sink_t[:, :, :, :] sink, 
             index_t max_bin, index_t[:] p, double[:] q):

  rank_core(kernel_sauvola[sink_t, index_t, source_t], source, selem, 
            sink, max_bin, p, q)

def sauvola_masked(source_t[:, :, :] source, char[:, :, :] selem,
                    char[:, :, :] mask, sink_t[:, :, :, :] sink,
                    index_t max_bin, index_t[:] p, double[:] q):
  
  rank_core_masked(kernel_sauvola[sink_t, index_t, source_t], source, selem, 
                   mask, sink, max_bin, p, q)




###############################################################################
### Contrast limited percentile
###############################################################################

cdef inline void kernel_clp_index(sink_t* sink, index_t* histo, index_t pop, source_t g, 
                            index_t max_bin, index_t* p, double* q) nogil:

    cdef index_t clip_limit = p[0];                   
    cdef index_t* clip_histo = &p[1];
    cdef double percentile = q[0];                              
    cdef index_t i, clipped, redist, residual, residual_step, residual_max 
    cdef index_t sum = 0

    
    if pop:
      # create clipped historgram
      clipped = 0;
      for i in range(max_bin):
        if histo[i] > clip_limit:
          clipped += histo[i] - clip_limit;
          clip_histo[i] = clip_limit;
        else:
          clip_histo[i] = histo[i];
          
      # redistribute clipped pixels
      redist = clipped / max_bin;
      residual = clipped - redist * max_bin;
          
      for i in range(max_bin):
        clip_histo[i] += redist;

      if residual > 0:
        residual_step = max_bin / residual;
        residual_step = residual_step if residual_step > 1 else 1;
        residual_max = residual * residual_step;
        residual_max = residual_max if residual_max < max_bin else max_bin;
        i = 0;
        while i < residual_max:
          clip_histo[i] += 1;
          i += residual_step;              
      
      percentile *= pop; 
      #printf('%d,%d,%f\n', clip_limit, clipped, percentile);
      #sum = 0;            
      #for i in range(max_bin):
      #  sum += histo[i];
      #  printf('(%d,%d)', histo[i], clip_histo[i]);
      #printf('\n%d\n', sum);
      #sum = 0;            
      
      for i in range(max_bin):
        sum += clip_histo[i];
        if sum >= percentile:
            break
        sink[0] = <sink_t>i
    else:
        sink[0] = <sink_t>0





cdef inline void kernel_clp(sink_t* sink, index_t* histo, index_t pop, source_t g, 
                            index_t max_bin, index_t* p, double* q) nogil:

    cdef double clip_limit = q[1];                   
    cdef double* clip_histo = &q[2];
    cdef double percentile = q[0];                              
    cdef index_t i, 
    cdef double clipped, redist
    cdef double sum = 0

    
    if pop:
      # create normalized and clipped historgram
      clipped = 0;
      for i in range(max_bin):
         clip_histo[i] = histo[i] / pop * max_bin;
         if clip_histo[i] > clip_limit:
          clipped += clip_histo[i] - clip_limit;
          clip_histo[i] = clip_limit;
          
      # redistribute clipped pixels
      redist = clipped / max_bin;   
      for i in range(max_bin):
        clip_histo[i] += redist;      
      
      #printf('%d,%d,%f\n', clip_limit, clipped, percentile);
      #sum = 0;            
      #for i in range(max_bin):
      #  sum += histo[i];
      #  printf('(%d,%d)', histo[i], clip_histo[i]);
      #printf('\n%d\n', sum);
      #sum = 0;            
      
      percentile *= max_bin;
      for i in range(max_bin):
        sum += clip_histo[i];
        if sum >= percentile:
            break
        sink[0] = <sink_t>i
    else:
        sink[0] = <sink_t>0


def clp(source_t[:, :, :] source, char[:, :, :] selem,
                      sink_t[:, :, :, :] sink, 
                      index_t max_bin, index_t[:] p, double[:] q):

  rank_core(kernel_clp[sink_t, index_t, source_t], source, selem,
            sink, max_bin, p, q)

def clp_masked(source_t[:, :, :] source, char[:, :, :] selem,
               char[:, :, :] mask, sink_t[:, :, :, :] sink,
               index_t max_bin, index_t[:] p, double[:] q):

  rank_core_masked(kernel_clp[sink_t, index_t, source_t], source, selem, 
                   mask, sink, max_bin, p, q)




###############################################################################
### Light sheet artifact correction
###############################################################################

cdef inline void kernel_lsac(sink_t* sink, index_t* histo, index_t pop, source_t g, 
                             index_t max_bin, index_t* p, double* q) nogil:

    cdef index_t clip_limit = p[0];                   
    cdef index_t* clip_histo = &p[1];
    cdef double percentile_low = q[0];                              
    cdef double percentile_high = 1.0 - q[1];                                  
    cdef index_t i, imin, imax, clipped, redist, residual, residual_step, residual_max 
    cdef index_t sum = 0, sum_low = 0, sum_high = 0, sum_valid, index, value
    cdef double total
    
    if pop:
      # create clipped historgram
      clipped = 0;
      for i in range(max_bin):
        if histo[i] > clip_limit:
          clipped += histo[i] - clip_limit;
          clip_histo[i] = clip_limit;
        else:
          clip_histo[i] = histo[i];
          
      # redistribute clipped pixels
      redist = clipped / max_bin;
      residual = clipped - redist * max_bin;
          
      for i in range(max_bin):
        clip_histo[i] += redist;

      if residual > 0:
        residual_step = max_bin / residual;
        residual_step = residual_step if residual_step > 1 else 1;
        residual_max = residual * residual_step;
        residual_max = residual_max if residual_max < max_bin else max_bin;
        i = 0;
        while i < residual_max:
          clip_histo[i] += 1;
          i += residual_step;              
      
      printf('pop=%d, limit=%d, clipped=%d, p0=%f, p1=%f\n', pop, clip_limit, clipped, percentile_low, percentile_high);               
      #sum = 0;
      #sum_high = 0;       
      #for i in range(max_bin):
      #  sum += histo[i];
      #  sum_high += clip_histo[i];
      #  printf('(%d,%d)', histo[i], clip_histo[i]);                      
      #printf('\nhisto sums=%d,%d\n', sum , sum_high)
      
      
      #calculate percentiles based on clipped histogram
      sum_low = 0;
      percentile_low *= pop;
      for i in range(max_bin):
        sum_low += clip_histo[i];
        if sum_low >= percentile_low:
          imin = i;
          sum_low -= clip_histo[i];
          break;
      
      percentile_high *= pop;
      sum_high = 0;
      for i in range(max_bin-1,-1,-1):
        sum_high += clip_histo[i];
        if sum_high >= percentile_high:
          imax = i;
          sum_high -= clip_histo[i];
          break;  
      sum_valid = pop - sum_low - sum_high;
      printf('pl,ph=%f,%f  sl,sh=%d,%d\n', percentile_low, percentile_high, sum_low, sum_high);
      
      
      value = g;
      if value > imax:
        value = imax;
      value = value - imin;
      if value < 0:
        sink[0] = <sink_t>0;
      else: 
        total = 0;
        for i in range(value):
          total += clip_histo[i + imin];
        sink[0] = <sink_t>((max_bin-1)*(total / sum_valid))
        
        #if imax > imin:
        #  sink[0] = <sink_t>(<double>(max_bin-1)*(value / (imax - imin)))
        #else:
        #  sink[0] = <sink_t>0
       
      printf('imin,imax = %d,%d,  sum_valid=%d ', imin, imax, sum_valid)      
      if value > 0:
        total = (max_bin-1)*(total / sum_valid)
      else:
        total = 0
      printf('value,sink=%f,%f\n', total, sink[0]);
      #sum = 0;            
      #for i in range(max_bin):
      #  sum += histo[i];
      #  printf('(%d,%d)', histo[i], clip_histo[i]);
      #printf('\n%d\n', sum);
      #sum = 0;     
    else:
      sink[0] = <sink_t>0
          
          

def lsac(source_t[:, :, :] source, char[:, :, :] selem,
         sink_t[:, :, :, :] sink, 
         index_t max_bin, index_t[:] p, double[:] q):

  rank_core(kernel_lsac[sink_t, index_t, source_t], source, selem,
            sink, max_bin, p, q)

def lsac_masked(source_t[:, :, :] source, char[:, :, :] selem,
                char[:, :, :] mask, sink_t[:, :, :, :] sink,
                index_t max_bin, index_t[:] p, double[:] q):

  rank_core_masked(kernel_lsac[sink_t, index_t, source_t], source, selem, 
                   mask, sink, max_bin, p, q)




################################################################################
#### Highest slope of histogram
################################################################################
#
#cdef inline void kernel_max_slope(sink_t* sink, index_t* histo, index_t pop, source_t g, 
#                             index_t max_bin, index_t* p, double* q) nogil:
#
#    cdef index_t clip_limit = p[0];                   
#    cdef index_t* clip_histo = &p[1];
#    cdef double percentile_low = q[0];                              
#    cdef double percentile_high = 1.0 - q[1];                                  
#    cdef index_t i, imin, imax, clipped, redist, residual, residual_step, residual_max 
#    cdef index_t sum = 0, sum_low = 0, sum_high = 0, sum_valid, index, value
#    cdef double total
#    
#    if pop:
#      # create clipped historgram
#      clipped = 0;
#      for i in range(max_bin):
#        if histo[i] > clip_limit:
#          clipped += histo[i] - clip_limit;
#          clip_histo[i] = clip_limit;
#        else:
#          clip_histo[i] = histo[i];
#          
#      # redistribute clipped pixels
#      redist = clipped / max_bin;
#      residual = clipped - redist * max_bin;
#          
#      for i in range(max_bin):
#        clip_histo[i] += redist;
#
#      if residual > 0:
#        residual_step = max_bin / residual;
#        residual_step = residual_step if residual_step > 1 else 1;
#        residual_max = residual * residual_step;
#        residual_max = residual_max if residual_max < max_bin else max_bin;
#        i = 0;
#        while i < residual_max:
#          clip_histo[i] += 1;
#          i += residual_step;              
#      
#      printf('pop=%d, limit=%d, clipped=%d, p0=%f, p1=%f\n', pop, clip_limit, clipped, percentile_low, percentile_high);               
#      #sum = 0;
#      #sum_high = 0;       
#      #for i in range(max_bin):
#      #  sum += histo[i];
#      #  sum_high += clip_histo[i];
#      #  printf('(%d,%d)', histo[i], clip_histo[i]);                      
#      #printf('\nhisto sums=%d,%d\n', sum , sum_high)
#      
#      
#      #calculate percentiles based on clipped histogram
#      sum_low = 0;
#      percentile_low *= pop;
#      for i in range(max_bin):
#        sum_low += clip_histo[i];
#        if sum_low >= percentile_low:
#          imin = i;
#          sum_low -= clip_histo[i];
#          break;
#      
#      percentile_high *= pop;
#      sum_high = 0;
#      for i in range(max_bin-1,-1,-1):
#        sum_high += clip_histo[i];
#        if sum_high >= percentile_high:
#          imax = i;
#          sum_high -= clip_histo[i];
#          break;  
#      sum_valid = pop - sum_low - sum_high;
#      printf('pl,ph=%f,%f  sl,sh=%d,%d\n', percentile_low, percentile_high, sum_low, sum_high);
#      
#      
#      value = g;
#      if value > imax:
#        value = imax;
#      value = value - imin;
#      if value < 0:
#        sink[0] = <sink_t>0;
#      else: 
#        total = 0;
#        for i in range(value):
#          total += clip_histo[i + imin];
#        sink[0] = <sink_t>((max_bin-1)*(total / sum_valid))
#        
#        #if imax > imin:
#        #  sink[0] = <sink_t>(<double>(max_bin-1)*(value / (imax - imin)))
#        #else:
#        #  sink[0] = <sink_t>0
#       
#      printf('imin,imax = %d,%d,  sum_valid=%d ', imin, imax, sum_valid)      
#      if value > 0:
#        total = (max_bin-1)*(total / sum_valid)
#      else:
#        total = 0
#      printf('value,sink=%f,%f\n', total, sink[0]);
#      #sum = 0;            
#      #for i in range(max_bin):
#      #  sum += histo[i];
#      #  printf('(%d,%d)', histo[i], clip_histo[i]);
#      #printf('\n%d\n', sum);
#      #sum = 0;     
#    else:
#      sink[0] = <sink_t>0
#          
#          
#
#def lsac(source_t[:, :, :] source, char[:, :, :] selem,
#         sink_t[:, :, :, :] sink, 
#         index_t max_bin, index_t[:] p, double[:] q):
#
#  rank_core(kernel_lsac[sink_t, index_t, source_t], source, selem,
#            sink, max_bin, p, q)
#
#def lsac_masked(source_t[:, :, :] source, char[:, :, :] selem,
#                char[:, :, :] mask, sink_t[:, :, :, :] sink,
#                index_t max_bin, index_t[:] p, double[:] q):
#
#  rank_core_masked(kernel_lsac[sink_t, index_t, source_t], source, selem, 
#                   mask, sink, max_bin, p, q)

