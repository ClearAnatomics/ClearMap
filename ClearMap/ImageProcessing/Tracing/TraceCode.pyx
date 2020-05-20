#distutils: language = c++
#distutils: sources = trace.cpp
#cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False, cdivision=True

"""
TraceCode
==========

Cython code for the tracing module.
"""

__author__    = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__   = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__copyright__ = 'Copyright (c) 2020 by Christoph Kirst'


cimport numpy as np
import numpy as np

from numpy cimport uint8_t, uint16_t, float32_t, double_t

from libcpp cimport bool as bool_t, long as long_t, int as int_t

ctypedef fused source_t:
    #uint8_t
    #uint16_t
    #float32_t
    double_t

#ctypedef fused dtype_t_tubness:
#    #uint8_t
#    #uint16_t
#    #float32_t
#    double_t

ctypedef fused mask_t:
     uint8_t
     uint16_t
     float32_t
     double_t

ctypedef Py_ssize_t index_t


cdef extern from "TraceCode.hpp":
    cdef cppclass Tracer[SOURCE_T, INDEX_T]:
        Tracer()
        
        int run(SOURCE_T* source_, INDEX_T shape_x_, INDEX_T shape_y_, INDEX_T shape_z_,
                                   INDEX_T stride_x_,INDEX_T stride_y_,INDEX_T sstride_z_,
                SOURCE_T* reward_,
                INDEX_T start_x_, INDEX_T start_y_, INDEX_T start_z_,
                INDEX_T goal_x_,  INDEX_T goal_y_,  INDEX_T goal_z_)
          
        int getPathSize()
        
        void getPath(INDEX_T* path_array)
        
        double getPathQuality()
        
        double cost_per_distance        
        double minimum_cost_per_distance
        
        double reward_multiplier
        
        double minimal_reward
        
        long max_step

        bool_t verbose


cdef extern from "TraceCode.hpp":
    cdef cppclass TracerToMask[D, I, M]:
        TracerToMask()
        
        int run(D* source_, I shape_x_, I shape_y_, I shape_z_,
                            I stride_x_,I stride_y_,I sstride_z_,
                D* reward_,
                I start_x_, I start_y_, I start_z_,
                M* mask_)
          
        int getPathSize()
        
        void getPath(I* path_array)
        
        double getPathQuality()
        
        double cost_per_distance        
        double minimum_cost_per_distance
        
        double reward_multiplier
        double minimal_reward
        
        long max_step

        bool_t verbose



def trace(source_t[:,:,:] source, 
          source_t[:,:,:] reward, 
          index_t[:] start, index_t[:] goal,
          double_t cost_per_distance, double_t minimum_cost_per_distance,
          double_t reward_multiplier, double_t minimal_reward, 
          bool_t quality,
          long_t max_step,
          bool_t verbose):
    
    #source  = np.ascontiguousarray(source);
    #tubness = np.ascontiguousarray(reward);
    
    cdef Tracer[source_t, index_t] tracer = Tracer[source_t, index_t]();
    tracer.cost_per_distance = cost_per_distance;
    tracer.minimum_cost_per_distance = minimum_cost_per_distance;    
    tracer.reward_multiplier = reward_multiplier;
    tracer.minimal_reward = minimal_reward;
    tracer.max_step = max_step;
    tracer.verbose = verbose;
    
    strides = np.array(source.strides) / source.itemsize;
    res = tracer.run(&source[0,0,0], source.shape[0], source.shape[1], source.shape[2],
                                     strides[0], strides[1], strides[2],
                     &reward[0,0,0],
                     start[0], start[1], start[2],
                     goal[0],  goal[1],  goal[2]);
   

                  
    if res < 0:
      if quality:
        return  np.zeros((0,3), dtype = int), 0;
      else:
        return np.zeros((0,3), dtype = int);
  
    n = tracer.getPathSize();
    cdef np.ndarray[index_t, ndim=2] path = np.zeros((n,3), dtype = int);
    tracer.getPath(&path[0,0]); 
    if quality:
      return path, tracer.getPathQuality();
    else:
      return path;
    
    
def trace_to_mask(source_t[:,:,:] source, 
                  source_t[:,:,:] reward, 
                  index_t[:] start, mask_t[:,:,:] mask,
                  double_t cost_per_distance, double_t minimum_cost_per_distance,
                  double_t reward_multiplier, double_t minimal_reward,
                  bool_t quality,
                  long_t max_step,
                  bool_t verbose):
    
    #source  = np.ascontiguousarray(source);
    #tubness = np.ascontiguousarray(reward);
    #mask = np.ascontiguousarray(mask);
    
    cdef TracerToMask[source_t, index_t, mask_t] tracer = TracerToMask[source_t, index_t, mask_t]();
    tracer.cost_per_distance = cost_per_distance;
    tracer.minimum_cost_per_distance = minimum_cost_per_distance;    
    tracer.reward_multiplier = reward_multiplier;
    tracer.minimal_reward = minimal_reward;
    tracer.max_step = max_step;
    tracer.verbose = verbose;
    
    strides = np.array(source.strides) / source.itemsize;
    #print('strides: ', strides)
    res = tracer.run(&source[0,0,0], source.shape[0], source.shape[1], source.shape[2],
                                     strides[0], strides[1], strides[2],
                     &reward[0,0,0],
                     start[0], start[1], start[2],
                     &mask[0,0,0]);
                 
    if res < 0:
      if quality:
        return  np.zeros((0,3), dtype = int), 0;
      else:
        return np.zeros((0,3), dtype = int);
  
    n = tracer.getPathSize();
    cdef np.ndarray[index_t, ndim=2] path = np.zeros((n,3), dtype = int);
    tracer.getPath(&path[0,0]);    
    if quality:
      return path, tracer.getPathQuality();
    else:
      return path;