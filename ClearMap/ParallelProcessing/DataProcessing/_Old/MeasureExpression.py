# -*- coding: utf-8 -*-
"""
MeasureExpression
=================

Measurements on a subset of points in large arrays.

Paralllel measurements at specified points of the data only.
Useful to speed up processing in large arrays on a smaller number of 
measurement points.
"""
__author__    = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__   = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__copyright__ = 'Copyright 2017 by Christoph Kirst, The Rockefeller University, New York City'


import numpy as np;

import pyximport; 
pyximport.install(setup_args={"include_dirs": [np.get_include()]}, reload_support=True)
 
import ClearMap.ParallelProcessing.DataProcessing.ArrayProcessing as ap

import ClearMap.ParallelProcessing.DataProcessing.MeasureExpressionCode as code


###############################################################################
### Find in local neighbourhood
###############################################################################

def measure_max(source, points, search, radii, sink = None, processes = None, verbose = False):
  """Find index in local search indices with a voxel with value smaller than a specified value for a list of points. 
    
  Arguments
  ---------
  source : array
    Data source.
  points : array
    List of linear indices of center points.
  search : array
    List of linear indices to add to the center index defining the local search area.
  radii : array
    The maximal index in the search array for each point to use.
  sink : array or None
    Optional sink for result indices.
  processes : int or None
    Number of processes to use.
  verbose : bool
    If True, print progress info.
  
  Returns
  -------
  sink : array
    Linear array with length of points containing the first search index with voxel below value.
  """
  
  processes, timer =  ap._initialize_processing(processes=processes, verbose=verbose, function='measure_max');
  source1d = ap._initialize_source(source, as_1d=True);
  
  sink = ap._initialize_sink(sink=sink, shape=points.shape, dtype=source.dtype);
  
  if sink.shape != points.shape:
     raise RuntimeError('Sink has invalid size %r not %r' % (points.shape, points.shape));
  
  #print(source1d.dtype, points.dtype, search.dtype, type(value), sink.dtype);
  #print(source1d.shape, points.shape, sink.shape)
  print source1d, points, search, radii, sink
  code.measure_max(source1d, points, search, radii, sink, processes);
  
  ap._finalize_processing(verbose=verbose, function='measure_max', timer=timer);
  
  return sink;


###############################################################################
### Tests
###############################################################################

def test():
  pass
#  import numpy as np
#  import ClearMap.ParallelProcessing.DataProcessing.MeasureExpression as mex
#  import ClearMap.Analysis.MeasureRadius as mr;
#  
#  reload(mex);
#  reload(mex.code);
#  
#  
#  source = np.random.rand(10,10,10);
#  
#      
#  d = np.ones((100,100,100));
#  d[goal] = 0.3;
#   
#
#  
#  strides = np.array(d.strides) / np.array(d.itemsize);
#  search, dist = mr.indices_from_center(radius = 15, strides = strides);
#  indices = np.array([np.ravel_multi_index((50,50,50), d.shape)]);
#                                           
#  result = mpl.find_smaller_than_value(d, indices, search, max_value = 0.5, out = None, processes = None);
#                                 
#  coords = np.unravel_index(indices + search[result], d.shape)
#  coords = np.array(coords).reshape(-1);
#  
#  print coords
#  if np.all(coords == goal):
#    print('works')
#  else:                           
#    print('error')
#   
#   
#  result = mpl.find_smaller_than_fraction(d, indices, search, fraction = 0.5, out = None, processes = None);
#                                         
#  coords = np.unravel_index(indices + search[result], d.shape)
#  coords = np.array(coords).reshape(-1);
#  
#  print coords
#  if np.all(coords == goal):
#    print('works')
#  else:                           
#    print('error')
#    
#    
#  
#  #import os
#  #os.remove('data.npy');
#  #os.remove('points.npy')
