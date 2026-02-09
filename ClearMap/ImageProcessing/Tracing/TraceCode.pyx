#distutils: language = c++
#cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False, cdivision=True

"""
TraceCode
==========

Cython code for the tracing module.
"""

__author__    = 'Christoph Kirst <ckirst@rockefeller.edu>, Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__   = 'MIT License <https://www.opensource.org/licenses/mit-license.php>'
__copyright__ = 'Copyright (c) 2020 by Christoph Kirst'


import numpy as np
from numpy cimport uint8_t, uint16_t, float32_t, double_t

cimport numpy as cnp

ctypedef fused source_t:
    # uint8_t
    # uint16_t
    # float32_t
    double_t

# ctypedef fused dtype_t_tubeness:
#    # uint8_t
#    # uint16_t
#    # float32_t
#    double_t

ctypedef fused mask_t:
     uint8_t
     uint16_t
     float32_t
     double_t

ctypedef Py_ssize_t index_t


cdef extern from "TraceCode.hpp":
    cdef cppclass Tracer[source_t, index_t]:
        Tracer()
        
        int run(source_t* source_, index_t shape_x_, index_t shape_y_, index_t shape_z_,
                                   index_t stride_x_, index_t stride_y_, index_t sstride_z_,
                source_t* reward_,
                index_t start_x_, index_t start_y_, index_t start_z_,
                index_t goal_x_,  index_t goal_y_,  index_t goal_z_)
          
        int getPathSize()
        
        void getPath(index_t* path_array)
        
        double getPathQuality()
        
        double cost_per_distance        
        double minimum_cost_per_distance
        
        double reward_multiplier
        
        double minimal_reward
        
        long max_step

        bint verbose


cdef extern from "TraceCode.hpp":
    cdef cppclass TracerToMask[source_t, index_t, mask_t]:
        TracerToMask()
        
        int run(source_t* source_, index_t shape_x_, index_t shape_y_, index_t shape_z_,
                index_t stride_x_, index_t stride_y_, index_t sstride_z_,
                source_t* reward_,
                index_t start_x_, index_t start_y_, index_t start_z_,
                mask_t* mask_)
          
        int getPathSize()
        
        void getPath(index_t* path_array)
        
        double getPathQuality()
        
        double cost_per_distance        
        double minimum_cost_per_distance
        
        double reward_multiplier
        double minimal_reward
        
        long max_step

        bint verbose



def trace(source_t[:,:,:] source, 
          source_t[:,:,:] reward, 
          index_t[:] start, index_t[:] goal,
          double_t cost_per_distance, double_t minimum_cost_per_distance,
          double_t reward_multiplier, double_t minimal_reward, 
          bint return_quality,
          long max_step,
          bint verbose):
    
    # source  = np.ascontiguousarray(source)
    # tubness = np.ascontiguousarray(reward)
    
    cdef Tracer[source_t, index_t] tracer = Tracer[source_t, index_t]();
    tracer.cost_per_distance = cost_per_distance
    tracer.minimum_cost_per_distance = minimum_cost_per_distance
    tracer.reward_multiplier = reward_multiplier
    tracer.minimal_reward = minimal_reward
    tracer.max_step = max_step
    tracer.verbose = verbose

    strides = np.array(source.strides) / source.itemsize
    res = tracer.run(&source[0,0,0], source.shape[0], source.shape[1], source.shape[2],
                                     strides[0], strides[1], strides[2],
                     &reward[0,0,0],
                     start[0], start[1], start[2],
                     goal[0],  goal[1],  goal[2])

    if res < 0:
      if return_quality:
        return  np.zeros((0,3), dtype = int), 0
      else:
        return np.zeros((0,3), dtype = int)

    n = tracer.getPathSize()
    cdef cnp.ndarray[index_t, ndim=2] path = np.zeros((n,3), dtype = int);
    tracer.getPath(&path[0,0])
    if return_quality:
      return path, tracer.getPathQuality()
    else:
      return path

def trace_to_mask(source_t[:,:,:] source, 
                  source_t[:,:,:] reward, 
                  index_t[:] start, mask_t[:,:,:] mask,
                  double_t cost_per_distance, double_t minimum_cost_per_distance,
                  double_t reward_multiplier, double_t minimal_reward,
                  bint return_quality,
                  long max_step,
                  bint verbose):
    
    # source = np.ascontiguousarray(source)
    # tubness = np.ascontiguousarray(reward)
    # mask = np.ascontiguousarray(mask)
    
    cdef TracerToMask[source_t, index_t, mask_t] tracer = TracerToMask[source_t, index_t, mask_t]()
    tracer.cost_per_distance = cost_per_distance
    tracer.minimum_cost_per_distance = minimum_cost_per_distance
    tracer.reward_multiplier = reward_multiplier
    tracer.minimal_reward = minimal_reward
    tracer.max_step = max_step
    tracer.verbose = verbose

    strides = np.array(source.strides) / source.itemsize
    # print('strides: ', strides)
    res = tracer.run(&source[0,0,0], source.shape[0], source.shape[1], source.shape[2],
                     strides[0], strides[1], strides[2],
                     &reward[0,0,0],
                     start[0], start[1], start[2],
                     &mask[0,0,0])

    if res < 0:
      if return_quality:
        return  np.zeros((0,3), dtype = int), 0
      else:
        return np.zeros((0,3), dtype = int)

    n = tracer.getPathSize()
    cdef cnp.ndarray[index_t, ndim=2] path = np.zeros((n, 3), dtype = int);
    tracer.getPath(&path[0,0])
    if return_quality:
      return path, tracer.getPathQuality()
    else:
      return path
