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
__license__   = 'GPLv3 - GNU General Public License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


import numpy as np;

import pyximport; 
pyximport.install(setup_args={"include_dirs": [np.get_include()]}, reload_support=True)
 
import ClearMap.ParallelProcessing.DataProcessing.ArrayProcessing as ap

import ClearMap.ParallelProcessing.DataProcessing.MeasurePointListCode as code
import ClearMap.Utils.array_checks as ac


# dtypes of source_t, sink_t and point_t in MeasurePointListCode.pyx
_DTYPES = (np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float32, np.float64)


def _prepare(function, source, points, search, sink, sink_dtype, max_search_indices=None, processes=None, verbose=False):
  """Validate and coerce the arguments shared by all the measurements.

  The Cython code does not check any bound, so this checks everything that would make it read or
  write out of bounds: the dimension of the points and search offsets, the center points inside
  the source and the number of search indices per point.

  Returns
  -------
  processes, timer, source_buffer, source_shape, source_strides, points, search, max_search, sink, sink_buffer
  """
  processes, timer = ap.initialize_processing(processes=processes, verbose=verbose, function=function)
  source, source_buffer, source_shape, source_strides = ap.initialize_source(source, as_1d=True, return_shape=True,
                                                                             return_strides=True)
  ac.check_dtype(source_buffer, _DTYPES, name='source')
  ndim = len(source_shape)

  _, points = ap.initialize_source(points)
  if points.ndim == 1:
    points = points[:, None]
  ac.check_dtype(points, _DTYPES, name='points')
  if points.ndim != 2 or points.shape[1] != ndim:
    raise ValueError(f'The points must be coordinates of shape (n, {ndim:d}), found {points.shape!r}!')
  if points.size > 0 and (np.any(points < 0) or np.any(points >= source_shape)):  # the center values are read unchecked
    raise ValueError('Some points are outside of the source!')
  n_points = points.shape[0]

  search = ac.as_index_array(search, name='search')
  if search.ndim == 1:
    search = search[:, None]
  if search.ndim != 2 or search.shape[1] != ndim:
    raise ValueError(f'The search offsets must have shape (m, {ndim:d}), found {search.shape!r}!')

  max_search = None
  if max_search_indices is not None:
    max_search = ac.as_index_array(max_search_indices, name='max_search_indices', ndim=1)
    if max_search.shape[0] != n_points:
      raise ValueError(f'max_search_indices has {max_search.shape[0]:d} entries, expected one per point ({n_points:d})!')
    if n_points > 0 and (max_search.min() < 0 or max_search.max() > search.shape[0]):
      raise ValueError(f'max_search_indices must be in [0, {search.shape[0]:d}]!')

  if sink_dtype is None:
    sink_dtype = source_buffer.dtype
  sink, sink_buffer = ap.initialize_sink(sink=sink, shape=(n_points,), dtype=sink_dtype)
  if sink_buffer.shape != (n_points,):
    raise ValueError(f'The sink has shape {sink_buffer.shape!r}, expected ({n_points:d},)!')

  return processes, timer, source_buffer, source_shape, source_strides, points, search, max_search, sink, sink_buffer



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
  processes, timer, source_buffer, source_shape, source_strides, points_buffer, search, max_search, sink, sink_buffer = \
    _prepare('measure_max', source, points, search, sink, None, max_search_indices=max_search_indices,
             processes=processes, verbose=verbose)
  ac.check_dtype(sink_buffer, _DTYPES, name='sink')

  code.measure_max(source_buffer, source_shape, source_strides, points_buffer, search, max_search, sink_buffer, processes)

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
  processes, timer, source_buffer, source_shape, source_strides, points_buffer, search, max_search, sink, sink_buffer = \
    _prepare('measure_min', source, points, search, sink, None, max_search_indices=max_search_indices,
             processes=processes, verbose=verbose)
  ac.check_dtype(sink_buffer, _DTYPES, name='sink')

  code.measure_min(source_buffer, source_shape, source_strides, points_buffer, search, max_search, sink_buffer, processes)

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
  processes, timer, source_buffer, source_shape, source_strides, points_buffer, search, max_search, sink, sink_buffer = \
    _prepare('measure_mean', source, points, search, sink, np.float64, max_search_indices=max_search_indices,
             processes=processes, verbose=verbose)
  ac.check_dtype(sink_buffer, (np.float64,), name='sink')

  code.measure_mean(source_buffer, source_shape, source_strides, points_buffer, search, max_search, sink_buffer, processes)

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
  processes, timer, source_buffer, source_shape, source_strides, points_buffer, search, max_search, sink, sink_buffer = \
    _prepare('measure_sum', source, points, search, sink, np.float64, max_search_indices=max_search_indices,
             processes=processes, verbose=verbose)
  ac.check_dtype(sink_buffer, (np.float64,), name='sink')

  code.measure_sum(source_buffer, source_shape, source_strides, points_buffer, search, max_search, sink_buffer, processes)

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
  processes, timer, source_buffer, source_shape, source_strides, points_buffer, search, _, sink, sink_buffer = \
    _prepare('find_smaller_than_value', source, points, search, sink, np.intp, processes=processes, verbose=verbose)
  ac.check_dtype(sink_buffer, (np.intp,), name='sink')

  code.find_smaller_than_value(source_buffer, source_shape, source_strides, points_buffer, search, float(value), sink_buffer, processes)

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
  processes, timer, source_buffer, source_shape, source_strides, points_buffer, search, _, sink, sink_buffer = \
    _prepare('find_smaller_than_fraction', source, points, search, sink, np.intp, processes=processes, verbose=verbose)
  ac.check_dtype(sink_buffer, (np.intp,), name='sink')

  code.find_smaller_than_fraction(source_buffer, source_shape, source_strides, points_buffer, search, float(fraction), sink_buffer, processes)

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
  processes, timer, source_buffer, source_shape, source_strides, points_buffer, search, _, sink, sink_buffer = \
    _prepare('find_smaller_than_values', source, points, search, sink, np.intp, processes=processes, verbose=verbose)
  ac.check_dtype(sink_buffer, (np.intp,), name='sink')
  values = ac.as_dtype(values, np.float64, name='values')
  if values.shape != (points_buffer.shape[0],):
    raise ValueError(f'values must have one entry per point ({points_buffer.shape[0]:d}), found shape {values.shape!r}!')

  code.find_smaller_than_values(source_buffer, source_shape, source_strides, points_buffer, search, values, sink_buffer, processes)

  ap.finalize_processing(verbose=verbose, function='find_smaller_than_values', timer=timer);
  
  return sink;


###############################################################################
### Tests
###############################################################################

def test():
  import numpy as np
  import ClearMap.Analysis.Measurements.radius_measurements as mr;
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
