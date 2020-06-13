# -*- coding: utf-8 -*-
"""
MeasureRadius
=============

Measures intensity decays at spcified points in the data.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
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

def measure_radius(source, points, fraction = None, value = None, 
                   max_radius = 100, method = 'sphere', default = np.inf, scale = None,
                   return_radii = True, 
                   return_radii_as_scalar = True,
                   return_indices = False, 
                   processes = None, verbose = False):
  """Measures a radius via decay of intensity values for a list of points.
  
  Arguments
  ---------
  source : array 
    Source for measurement.
  points : array
    List of indices to measure radis for.
  fraction : float or None
    Fraction of center intensity that needs to be reached to detemrine the 
    radius.
    If None, value needs to be given.
  value : array or float or None:
    The value below which the inensity has to fall to measure the radius
    from the center pixel. If array, it has to be the same size as the 
    points. If None, fraction has to be given.
  max_radius : int or tuple of ints
    The maximal pixel radius to consider in each dimension. The larger the 
    slower the measurement.
  default : number or None
    Default value to use if no radius was detected.
  scale: tuple or float
    An optional scale in each direction to determine the distance. 
  return_radii : bool
    If True, return the radii measured.
  return_radii_as_scalar : bool
    If True, returnt the radii as single floats, otherwise a radius for 
    each dimension.
  return_indices : bool
    If True, return the indices of the search which allows to idenitfy the 
    pixel at which the radius condition was met.
  processes : int or None
    Number of processes to use.
  verbose : bool
    If True, print progress info.  
    
  Returns
  -------
  radii : array
    Array of measured radii if return_radii is True.
  indices : array
    Array of measured indices at which the radius detrection condition 
    is met.
  """
  source = io.as_source(source).array;
  
  if verbose:
    timer = tmr.Timer();
    print('Measuring radii of %d points in array of shape %r.' % (points.shape[0], source.shape));
  
  ndim = source.ndim;
  if not hasattr(max_radius, '__len__'):
    max_radius = [max_radius] * ndim;
  if len(max_radius) != ndim:
    raise ValueError(' The maximal search radius %r has wronf dimension!' % max_radius);
  
  if method == 'sphere':
    search = search_indices_sphere(max_radius);
  elif method == 'rectangle':
    search = search_indices_rectangle(max_radius);
  else:
    raise ValueError("The method is not 'sphere' or 'rectangle' but %r!" % method);
  
  if value is not None:
    if hasattr(value, '__len__'):
      measured = mpl.find_smaller_than_values(source, points, search, value, sink=None, processes=processes, verbose=verbose);
    else:  
      measured = mpl.find_smaller_than_value(source, points, search, value, sink=None, processes=processes, verbose=verbose);                                          
  elif fraction is not None:    
    measured = mpl.find_smaller_than_fraction(source, points, search, fraction, sink=None, processes=processes, verbose=verbose);
  else:
    raise ValueError('fraction or value cannot both be None!');                                   
  
  if verbose:
    timer.print_elapsed_time('Measuring radii done');
  
  result = ();
  if return_radii:
    if scale is None:
      scale = 1;
    if not hasattr(scale, '__len__'):
      scale = [scale] * ndim;
      scale = np.asarray(scale);
    radii = np.abs(search) * scale;  
    if return_radii_as_scalar:
      radii = np.sqrt(np.sum(radii*radii, axis=1));
      radii = np.hstack([radii, default]);
    else:
      radii = np.vstack([radii, [default] * ndim]);
    radii = radii[measured];
    result += (radii,);
  if return_indices:
    search = np.vstack([search, [np.max(search, axis=0) + 1]]);
    indices = search[measured]; 
    result += (indices,);
  
  if len(result) == 1:
    result = result[0];
  return result;


###############################################################################
### Search indices
###############################################################################

def search_indices_sphere(radius):
  """Creates all relative indices within a sphere of specified radius in an array with specified strides.
  
  Arguments
  ---------
  radius : tuple of int
    Radius of the sphere of the search index list.
    
  Returns
  -------
  indices : array
     Array of ints of relative indices for the search area voxels.
  """
  grid = [np.arange(-r,r+1, dtype=float)/np.maximum(1,r) for r in radius];                    
  grid = np.array(np.meshgrid(*grid, indexing = 'ij'));

  #sort indices by radius  
  dist = np.sum(grid*grid, axis = 0);
  dist_shape = dist.shape;
  dist = dist.reshape(-1);            
  dist_index = np.argsort(dist);
  dist_sorted = dist[dist_index];
  keep = dist_sorted <= 1;
  dist_index = dist_index[keep];
  
  # convert to relative coordinates                  
  indices = np.array(np.unravel_index(dist_index, dist_shape)).T;
  indices -= radius;                    
  
  return indices;


def search_indices_rectangle(radius):
  """Creates all relative indices within a rectangle.
  
  Arguments
  ---------
  radius : tuple or float
    Radius of the sphere of the search index list.
  
  Returns
  -------
  indices : array
     Array of ints of relative indices for the search area voxels.
  """
  #create coordiante grid
  grid = [np.arange(-r,r+1, dtype=int) for r in radius];                    
  grid = np.array(np.meshgrid(*grid, indexing = 'ij'));
  
  #sort indices by radius  
  dist = np.sum(grid*grid, axis = 0);
  dist_shape = dist.shape;
  dist = dist.reshape(-1);            
  dist_index = np.argsort(dist);
  
  # convert to relative coordinates                  
  indices = np.array(np.unravel_index(dist_index, dist_shape)).T;
  indices -= radius;                    
  
  return indices;


###############################################################################
### Tests
###############################################################################

def test():
  import numpy as np
  import ClearMap.IO.IO as io
  import ClearMap.Analysis.Measurements.MeasureRadius as mr;
  
  data = 10-np.abs(10-np.arange(0,21));
  search = mr.search_indices_sphere(radius=[10,10,10])      
  print(search)
       
  points = np.array([10]);   
  d,i = mr.measure_radius(data, points, fraction = 0.75, max_radius = 10, scale = 2, verbose = True, processes = 4, return_indices=True);             
  
  data = np.random.rand(*(30,40,50));
  io.write('data.npy', data)
  
  points = np.array([np.random.randint(0,s, size=10) for s in data.shape]).T
  d,i = mr.measure_radius(data, points, value = 0.5, max_radius = 10, scale = 2, verbose = True, processes = 4, return_indices=True);      

  data = np.zeros((30,40,50), dtype=int);
  data[10:20, 15:25,10:20] = 1;
  data[15,20,15] = 2;
  
  import ClearMap.Visualization.Plot3d as p3d
  p3d.plot(data)
  
  points = np.array([[15, 20, 15],[4,4,4]])
  d,i = mr.measure_radius(data, points, value = 0.0, max_radius = 10, scale = None, verbose = True, processes = None, return_indices=True);      
    
