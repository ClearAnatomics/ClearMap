# -*- coding: utf-8 -*-
"""
MeasureRadius
=============

Measures intensity decays at spcified points in the data.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


import numpy as np;

import ClearMap.IO.IO as io

import ClearMap.ParallelProcessing.DataProcessing.MeasurePointList as mpl

import ClearMap.Utils.Timer as tmr


###############################################################################
### Measure Radius
###############################################################################

def measure_expression(source, points, search_radius, method = 'max',
                       sink = None, processes = None, verbose = False):
  """Measures the expression around a list of points in a source.
  
  Arguments
  ---------
  source : array 
    Source for measurement.
  points : array
    List of indices to measure radis for.
  search_radius : int or array
    List of search radii to use around each point. If int  use
    this radius for all points. Array should be of length of points.
  method : 'max' or 'min', 'mean'
    Measurement type.
  processes : int or None
    Number of processes to use.
  verbose : bool
    If True, print progress info.  
    
    
  """
  source = io.as_source(source);
  ndim = source.ndim;

  if verbose:
    timer = tmr.Timer();
    print('Measuring expression of %d points in array of shape %r.' % (points.shape[0], source.shape));
  
  if not hasattr(search_radius, '__len__'):
    search_radius = search_radius * np.ones(points.shape[0]);
  if len(search_radius) != len(points):
    raise ValueError('The search_radius is not valid!');
  
  indices, radii_indices = search_indices(search_radius, ndim);
  
  if method == 'max':
    expression = mpl.measure_max(source, points, indices, radii_indices, sink=sink, processes=processes, verbose=verbose);
  elif method == 'min':
    expression = mpl.measure_min(source, points, indices, radii_indices, sink=sink, processes=processes, verbose=verbose);
  elif method == 'mean':
    expression = mpl.measure_mean(source, points, indices, radii_indices, sink=sink, processes=processes, verbose=verbose);
  elif method == 'sum':
    expression = mpl.measure_sum(source, points, indices, radii_indices, sink=sink, processes=processes, verbose=verbose);  
  else:
    raise ValueError("Method %r not in 'max', 'min', 'mean'" % method);
  
  if verbose:
    timer.print_elapsed_time('Measuring expression done');
  
  return expression;


###############################################################################
### Search indices
###############################################################################

def search_indices(radii, ndim):
  """Creates all relative indices within a sphere of specified radius in an array with specified strides.
  
  Arguments
  ---------
  radius : tuple or float
    Radius of the sphere of the search index list.
  strides : tuple of ints
    Srides of the array
  scale : float, tuple or None
    Spatial scale in each array dimension.
    
  Returns
  -------
  indices : array
     Array of ints of relative indices for the search area voxels.
  """
  radius = int(np.ceil(np.max(radii)));
  
  #create coordiante grid          
  grid = [np.arange(-radius,radius+1)] * ndim;                    
  grid = np.array(np.meshgrid(*grid, indexing = 'ij'));
  
  #sort indices by radius  
  dist = np.sum(grid*grid, axis = 0);
  dist_shape = dist.shape;
  dist = dist.reshape(-1);            
  dist_index = np.argsort(dist);
  dist = np.sqrt(dist[dist_index]);
  dist = np.hstack([dist, np.inf]);
  #print(dist)
  
  radii_indices = np.searchsorted(dist, radii, side='right');
  
  # convert coordinates to full indices via strides                   
  indices = np.array(np.unravel_index(dist_index, dist_shape)).T;
  indices -= radius;
  
  # remove center point 
  indices = indices[1:];
  radii_indices[radii_indices > 0] -= 1;
  
  return indices, radii_indices;


###############################################################################
### Tests
###############################################################################

def test():
  import numpy as np
  import ClearMap.Analysis.Measurements.MeasureExpression as mex;
  reload(mex);
  
  data = 10-np.abs(10-np.arange(0,21));
  radii = [2,4]
  search, indices = mex.search_indices(radii=radii, ndim=1)      
       
  points = np.array([10,10]);   
  d = mex.measure_expression(data, points, search_radius=radii, method='min', verbose = True, processes = None);             
  print(d)            
  
  d = mex.measure_expression(data, points, search_radius=radii, method='mean', verbose = True, processes = None);             
  print(d) 
  
  #3d data
  data = np.zeros((30,40,50));
  data[10:20,10:20,20:30] = 1;
  data[13:16, 13:16, 23:26] = 2;
  data[14,14,24] = 3;
  
  points = np.array([[14,14,24], [3,3,3]]);
  d = mex.measure_expression(data, points, search_radius=1, method='min', verbose = True, processes = None);             
  print(d) 

  
  