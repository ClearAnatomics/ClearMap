# -*- coding: utf-8 -*-
"""
MeasurePointList
================

Measurements on a subset of points in large arrays.

Paralllel measuremnts at specified points of the data only.
Useful to speed up processing in large arrays and only a smaller number 
of measurement points.

See also
--------
:mod:`ClearMap.Analysis.Measurements`.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


import numpy as np;

import pyximport; 
pyximport.install(setup_args={"include_dirs": [np.get_include()]}, reload_support=True)
 
import ClearMap.ParallelProcessing.DataProcessing.ArrayProcessing as ap

import ClearMap.ParallelProcessing.DataProcessing.MeasurePointListCode as code


###############################################################################
### Measure extrema
###############################################################################

def measure_max(source, points, search, max_search_indices, sink = None, processes = None, verbose = False):
  """Find local maximum in a large array for a list of center points.
    
  Arguments
  ---------
  source : array
    Data source.
  points : array
    List of linear indices of center points.
  search : array
    List of linear indices to add to the center index defining the local search area.
  max_search_indices : array
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
    Linear array with length of points containing the local maxima.
  """ 
  processes, timer = ap.initialize_processing(processes=processes, verbose=verbose, function='measure_max');
  
  source, source_buffer, source_shape, source_strides = ap.initialize_source(source, as_1d=True, return_shape=True, return_strides=True);
  
  sink, sink_buffer = ap.initialize_sink(sink=sink, shape=points.shape[0], dtype=source.dtype);
  
  if sink_buffer.shape[0] != points.shape[0]:
     raise RuntimeError('Sink has invalid size %r not %r' % (sink.shape, points.shape));
  
  points, points_buffer = ap.initialize_source(points);
  if points_buffer.ndim == 1:
    points_buffer = points_buffer[:,None];
  
  #print(source1d.shape, shape, strides, points.shape, search.shape, max_search_indices.shape, sink.shape)
  code.measure_max(source_buffer, source_shape, source_strides, points_buffer, search, max_search_indices, sink_buffer, processes);
  
  ap.finalize_processing(verbose=verbose, function='measure_max', timer=timer);
  
  return sink;


def measure_min(source, points, search, max_search_indices, sink = None, processes = None, verbose = False):
  """Find local minimum in a large array for a list of center points.
    
  Arguments
  ---------
  source : array
    Data source.
  points : array
    List of linear indices of center points.
  search : array
    List of linear indices to add to the center index defining the local search area.
  max_search_indices : array
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
    Linear array with length of points containing the local minima.
  """ 
  processes, timer =  ap.initialize_processing(processes=processes, verbose=verbose, function='measure_max');
  
  source, source_buffer, source_shape, source_strides = ap.initialize_source(source, as_1d=True, return_shape=True, return_strides=True);
  
  sink, sink_buffer = ap.initialize_sink(sink=sink, shape=points.shape[0], dtype=source.dtype);
  
  if sink.shape[0] != points.shape[0]:
     raise RuntimeError('Sink has invalid size %r not %r' % (sink.shape, points.shape));
  
  
  points, points_buffer = ap.initialize_source(points);
  if points_buffer.ndim == 1:
    points_buffer = points_buffer[:,None];  
  
  code.measure_min(source_buffer, source_shape, source_strides, points_buffer, search, max_search_indices, sink_buffer, processes);
  
  ap.finalize_processing(verbose=verbose, function='measure_max', timer=timer);
  
  return sink;


def measure_mean(source, points, search, max_search_indices, sink = None, processes = None, verbose = False):
  """Find local mean in a large array for a list of center points.
    
  Arguments
  ---------
  source : array
    Data source.
  points : array
    List of linear indices of center points.
  search : array
    List of linear indices to add to the center index defining the local search area.
  max_search_indices : array
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
    Linear array with length of points containing the local mean.
  """ 
  processes, timer =  ap.initialize_processing(processes=processes, verbose=verbose, function='measure_mean');
  
  source, source_buffer, source_shape, source_strides = ap.initialize_source(source, as_1d=True, return_shape=True, return_strides=True);
  
  sink, sink_buffer = ap.initialize_sink(sink=sink, shape=points.shape[0], dtype=float);
  
  points_buffer = ap.initialize_source(points);
  if points_buffer.ndim == 1:
    points_buffer = points_buffer[:,None];  
  
  if sink.shape[0] != points_buffer.shape[0]:
     raise RuntimeError('Sink has invalid size %r not %r' % (sink.shape, points_buffer.shape));
  
  code.measure_mean(source_buffer, source_shape, source_strides, points_buffer, search, max_search_indices, sink_buffer, processes);
  
  ap.finalize_processing(verbose=verbose, function='measure_mean', timer=timer);
  
  return sink;


def measure_sum(source, points, search, max_search_indices, sink = None, processes = None, verbose = False):
  """Find local mean in a large array for a list of center points.
    
  Arguments
  ---------
  source : array
    Data source.
  points : array
    List of linear indices of center points.
  search : array
    List of linear indices to add to the center index defining the local search area.
  max_search_indices : array
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
    Linear array with length of points containing the local mean.
  """ 
  processes, timer =  ap.initialize_processing(processes=processes, verbose=verbose, function='measure_mean');
  
  source, source_buffer, source_shape, source_strides = ap.initialize_source(source, as_1d=True, return_shape=True, return_strides=True);
  
  sink, sink_buffer = ap.initialize_sink(sink=sink, shape=points.shape[0], dtype=float);
  
  points_buffer = ap.initialize_source(points);
  if points_buffer.ndim == 1:
    points_buffer = points_buffer[:,None];  
  
  if sink.shape[0] != points_buffer.shape[0]:
     raise RuntimeError('Sink has invalid size %r not %r' % (sink.shape, points_buffer.shape));
  
  code.measure_sum(source_buffer, source_shape, source_strides, points_buffer, search, max_search_indices, sink_buffer, processes);
  
  ap.finalize_processing(verbose=verbose, function='measure_mean', timer=timer);
  
  return sink;




###############################################################################
### Find in local neighbourhood
###############################################################################

def find_smaller_than_value(source, points, search, value, sink = None, processes = None, verbose = False):
  """Find index in local search indices with a voxel with value smaller than a specified value for a list of points. 
    
  Arguments
  ---------
  source : array
    Data source.
  points : array
    List of linear indices of center points.
  search : array
    List of linear indices to add to the center index defining the local search area.
  value : float
    Search for first voxel in local area with value smaller than this value.
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
  processes, timer =  ap.initialize_processing(processes=processes, verbose=verbose, function='find_smaller_than_value');
  source, source_buffer, source_shape, source_strides = ap.initialize_source(source, as_1d=True, return_shape=True, return_strides=True);
  
  sink, sink_buffer = ap.initialize_sink(sink=sink, shape=points.shape[0], dtype=int);
  
  if sink.shape[0] != points.shape[0]:
     raise RuntimeError('Sink has invalid size %r not %r' % (sink.shape, points.shape));
  
  points, points_buffer = ap.initialize_source(points);
  if points_buffer.ndim == 1:
    points_buffer = points_buffer[:,None];   
  
  code.find_smaller_than_value(source_buffer, source_shape, source_strides, points_buffer, search, value, sink_buffer, processes);
  
  ap.finalize_processing(verbose=verbose, function='find_smaller_than_value', timer=timer);
  
  return sink;


def find_smaller_than_fraction(source, points, search, fraction, sink = None, processes = None, verbose = False):
  """Find index in local search indices with a voxel with value smaller than a fraction of the value of the center voxel for a list of points. 
    
  Arguments
  ---------
  source : array
    Data source.
  points : array
    List of linear indices of center points.
  search : array
    List of linear indices to add to the center index defining the local search area.
  fraction : float
    Search for first voxel in local area with value smaller than this fraction of the center value.
  sink : array or None
    Optional sink for result indices.
  processes : int or None
    Number of processes to use.
  verbose : bool
    If True, print progress info.
  
  Returns
  -------
  sink : array
    Linear array with length of points containing the first search index with voxel below the fraction of the center value.
  """
  processes, timer =  ap.initialize_processing(processes=processes, verbose=verbose, function='find_smaller_than_fraction');
  source, source_buffer, source_shape, source_strides = ap.initialize_source(source, as_1d=True, return_shape=True, return_strides=True);
  
  sink, sink_buffer = ap.initialize_sink(sink=sink, shape=points.shape[0], dtype=int);
  
  if sink.shape[0] != points.shape[0]:
     raise RuntimeError('Sink has invalid size %r not %r' % (sink.shape, points.shape));
  
  points, points_buffer = ap.initialize_source(points);
  if points.ndim == 1:
    points = points[:,None];
  
  code.find_smaller_than_fraction(source_buffer, source_shape, source_strides, points_buffer, search, fraction, sink_buffer, processes);

  ap.finalize_processing(verbose=verbose, function='find_smaller_than_fraction', timer=timer);
  
  return sink;


def find_smaller_than_values(source, points, search, values, sink = None, processes = None, verbose = False):
  """Find index in local search indices with a voxel with value smaller than a fraction of the value of the center voxel for a list of points. 
    
  Arguments
  ---------
  source : array
    Data source.
  points : array
    List of linear indices of center points.
  search : array
    List of linear indices to add to the center index defining the local search area.
  fraction : float
    Search for first voxel in local area with value smaller than this fraction of the center value.
  sink : array or None
    Optional sink for result indices.
  processes : int or None
    Number of processes to use.
  verbose : bool
    If True, print progress info.
  
  Returns
  -------
  sink : array
    Linear array with length of points containing the first search index with voxel below the fraction of the center value.
  """
  processes, timer =  ap.initialize_processing(processes=processes, verbose=verbose, function='find_smaller_than_values');
  source, source_buffer, source_shape, source_strides = ap.initialize_source(source, as_1d=True, return_shape=True, return_strides=True);
  
  sink, sink_buffer = ap.initialize_sink(sink=sink, shape=points.shape, dtype=int);
  
  if sink.shape[0] != points.shape[0]:
     raise RuntimeError('Sink has invalid size %r not %r' % (sink.shape, points.shape));
  
  points, points_buffer = ap.initialize_source(points);
  if points_buffer.ndim == 1:
    points_buffer = points_buffer[:,None];  
   
  code.find_smaller_than_fraction(source_buffer, source_shape, source_strides, points_buffer, search, values, sink_buffer, processes);

  ap.finalize_processing(verbose=verbose, function='find_smaller_than_values', timer=timer);
  
  return sink;


###############################################################################
### Tests
###############################################################################

def test():
  import numpy as np
  import ClearMap.Analysis.Measurements.MeasureRadius as mr;
  import ClearMap.ParallelProcessing.DataProcessing.MeasurePointList as mpl
  
  from importlib import reload
  reload(mpl);
  reload(mpl.code);

  # find_smaller_than_value 
  goal = (50,55,42);      
  d = np.ones((100,100,100));
  d[goal] = 0.3;
  
  strides = np.array(d.strides) / np.array(d.itemsize);
  search, dist = mr.indices_from_center(radius = 15, strides = strides);
  indices = np.array([np.ravel_multi_index((50,50,50), d.shape)]);
                                           
  result = mpl.find_smaller_than_value(d, indices, search, max_value = 0.5, out = None, processes = None);
                                 
  coords = np.unravel_index(indices + search[result], d.shape)
  coords = np.array(coords).reshape(-1);
  assert np.all(coords == goal)
  
  # find_smaller_than_fraction
  result = mpl.find_smaller_than_fraction(d, indices, search, fraction = 0.5, out = None, processes = None);
                                         
  coords = np.unravel_index(indices + search[result], d.shape)
  coords = np.array(coords).reshape(-1);
  assert np.all(coords == goal) 
  