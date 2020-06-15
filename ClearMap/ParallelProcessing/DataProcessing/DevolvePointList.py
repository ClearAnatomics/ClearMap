# -*- coding: utf-8 -*-
"""
DevolvePointList
================

Converts point data into a devolved or smeared out image for 
visulalization and analysis purposes. 

Note
----
This is effecitively blurring with a specified kernel.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'

import math
import numpy as np

import pyximport;
pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)

import ClearMap.IO.IO as io

import ClearMap.ParallelProcessing.DataProcessing.ArrayProcessing as ap

import ClearMap.ParallelProcessing.DataProcessing.DevolvePointListCode as code

###############################################################################
### Voxelization
###############################################################################

def devolve(source, sink = None, shape = None, dtype = None, 
            weights = None, indices = None, kernel = None, 
            processes = None, verbose = False):
  """Converts a list of points into an volumetric image array.
  
  Arguments
  ---------
  source : str, array or Source
    Source of point of nxd coordinates.
  sink : str, array or None
    The sink for the devolved image, if None return array.
  shape : tuple, str or None
    Shape of the final devolved data. If None, determine from points.
    If str, determine shape from the source at the specified location.
  dtype : dtype or None
    Optional data type of the sink.
  weights : array or None
    Weight array of length n for each point. If None, use uniform weights.  
  method : str
    Method for voxelization: 'sphere', 'rectangle' or 'pixel'.
  indices : array 
    The relative indices to the center to devolve over as nxd array.
  kernel : array
    Optional kernel weights for each index in indices.
  processes : int or None
    Number of processes to use.
  verbose : bool
    If True, print progress info.                        
 
  Returns
  -------
  sink : str, array
    Volumetric data of devolved point data.
  """
  processes, timer = ap.initialize_processing(processes=processes, verbose=verbose, function='devolve');
  
  #points, points_buffer = ap.initialize_source(points);
  points_buffer = io.as_source(source).as_buffer();
  if points_buffer.ndim == 1:
    points_buffer = points_buffer[:,None];
  
  if sink is None and shape is None:
    if points_buffer.ndim > 1:
      shape = tuple(int(math.ceil(points_buffer[:,d].max())) for d in range(points_buffer.shape[1]));
    else:
      shape = (int(math.ceil(points_buffer[:].max())),)
  elif isinstance(shape, str):
    shape= io.shape(shape);
  
  if sink is None and dtype is None:
    if weights is not None:
      dtype = io.dtype(weights);
    elif kernel is not None:
      kernel = np.asarray(kernel);
      dtype = kernel.dtype;
    else:
      dtype = int; 
  
  sink, sink_buffer, sink_shape, sink_strides, = ap.initialize_sink(sink=sink, shape=shape, dtype=dtype, return_shape=True, return_strides=True, as_1d=True);
  
  if indices is None:
    return sink;
  indices = np.asarray(indices, dtype=int);
  if indices.ndim == 1:
    indices = indices[:,None];
  
  if kernel is not None:
    kernel = np.asarray(kernel, dtype=float);
  
  if weights is None:
    if kernel is None:
      code.devolve_uniform(points_buffer, indices, sink_buffer, sink_shape, sink_strides, processes);
    else:
      code.devolve_uniform_kernel(points_buffer, indices, kernel, sink_buffer, sink_shape, sink_strides, processes);
  else:
    if kernel is None:
      code.devolve_weights(points_buffer, weights, indices, sink_buffer, sink_shape, sink_strides, processes);
    else:
      code.devolve_weights_kernel(points_buffer, weights, indices, kernel, sink_buffer, sink_shape, sink_strides, processes);

  ap.finalize_processing(verbose=verbose, function='devolve', timer=timer);

  return sink;


###############################################################################
### Tests
###############################################################################

def _test():
  """Tests"""
  import numpy as np
  import ClearMap.Visualization.Plot3d as p3d
  import ClearMap.ParallelProcessing.DataProcessing.DevolvePointList as dpl
  
  from importlib import reload
  reload(dpl)
  
  points = np.array([1,9,10], dtype=float)
  indices = [-2,-1,0,1];
  v = dpl.devolve(points, shape=(11,), indices=indices);
  print(v.array)
  
  points = np.array([[10,10,18]])
  indices = [[-1,-1,-1],[0,0,0],[1,1,1]];
  v = dpl.devolve(points, shape=(20,20,20), indices=indices, weights=None);
  p3d.plot(v)
  
  points = np.array([[10,10,10]]);
  weights = np.random.rand(len(points));
  v = dpl.devolve(points, shape=(20,20,20), indices=indices, weights=weights);
  p3d.plot(v)
  
  kernel = np.random.rand(len(indices));
  v = dpl.devolve(points, shape=(20,20,20), indices=indices, weights=None, kernel=kernel);
  p3d.plot(v)
  