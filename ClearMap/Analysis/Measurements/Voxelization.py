# -*- coding: utf-8 -*-
"""
Voxelization
============

Converts point data into a voxelized image for visulalization and analysis.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'

import numpy as np

import ClearMap.IO.IO as io

import ClearMap.ParallelProcessing.DataProcessing.DevolvePointList as dpl


###############################################################################
### Voxelization
###############################################################################

def voxelize(source, sink = None, shape = None, dtype = None, weights = None,
             method = 'sphere', radius = (1,1,1), kernel = None, 
             processes = None, verbose = False):
  """Converts a list of points into an volumetric image array
  
  Arguments
  ---------
  source : str, array or Source
    Source of point of nxd coordinates.
  sink : str, array or None
    The sink for the voxelized image, if None return array.
  shape : tuple or None
    Shape of the final voxelized data. If None, deterimine from points.
  dtype : dtype or None
    Optional data type of the sink.
  weights : array or None
    Weight array of length n for each point. If None, use uniform weights.  
  method : str
    Method for voxelization: 'sphere', 'rectangle' or 'pixel'.
  radius : tuple 
    Radius of the voxel region to integrate over.
  kernel : function
    Optional function of distance to set weights in the voxelization.
  processes : int or None
    Number of processes to use.
  verbose : bool
    If True, print progress info.                        
 
  Returns
  -------
  sink : str, array
    Volumetric data of voxelied point data.
  """
  points = io.read(source);
  
  points_shape = points.shape;
  if len(points_shape) > 1:
    ndim = points_shape[1];
  else:
    ndim = 1;
  
  if not hasattr(radius, '__len__'):
    radius = [radius] * ndim;
  if len(radius) != ndim:
    raise ValueError('Radius %r and points with shape %r do not match in dimension!' % (radius, points_shape));
  
  if method == 'sphere':
    indices, kernel = search_indices_sphere(radius, kernel)
  elif method == 'rectangle':
    indices, kernel = search_indices_rectangle(radius, kernel)
  elif method == 'pixel': 
    indices = np.array(0, dtype=int);
    if kernel is not None:
      kernel = np.array([kernel(0)])
  else:
    raise ValueError("method not 'sphere', 'rectangle', or 'pixel', but %r!" % method)
  
  return dpl.devolve(points, sink=sink, shape=shape, dtype=dtype,
                     weights=weights, indices=indices, kernel=kernel, processes=processes, verbose=verbose);

###############################################################################
### Search indices
###############################################################################

def search_indices_sphere(radius, kernel = None):
  """Creates all relative indices within a sphere of specified radius.
  
  Arguments
  ---------
  radius : tuple or int
    Radius of the sphere of the search index list.
  
  Returns
  -------
  indices : array
     Array of ints of relative indices for the search area voxels.
  """
  #create coordiante grid          
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
  
  if kernel is not None:
    dist_sorted = np.sqrt(dist_sorted[keep]);
    kernel = np.array([kernel(d) for d in dist_sorted], dtype=float);
  else:
    kernel = None;
  
  # convert to relative coordinates
  indices = np.array(np.unravel_index(dist_index, dist_shape)).T;
  indices -= radius;                    
  
  return indices, kernel;


def search_indices_rectangle(radius, kernel = None):
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
  
  if kernel is not None:
    dist = np.sqrt(np.sum(grid*grid, axis =0));
    kernel = np.array([kernel(d) for d in dist], dtype=float);
  else:
    kernel = None;
    
  indices = grid.reshape((len(radius),-1)).T;
  
  return indices, kernel;


###############################################################################
### Tests
###############################################################################

def _test():
  """Tests"""
  import numpy as np
  import ClearMap.Visualization.Plot3d as p3d
  import ClearMap.Analysis.Measurements.Voxelization as vox
  
  
  #points = np.random.rand(20,3) * 15;
  #points = np.asarray(points, dtype=int);
  points = np.array([[10,1,1]])  
  v = vox.voxelize(points, shape = (20,20,20), radius = (0,0,0));
  w = vox.voxelize(points, shape = (20,20,20), radius = (2,2,2));
  p3d.plot([[v,w]])
  
  indices, kernel = vox.search_indices_sphere((2,2,2), kernel=None)
  print(indices)
  
  points = np.array([[10,10,18]])
  v = vox.voxelize(points, shape = (20,20,20),  weights=None, radius=(5,7,10));
  p3d.plot(v)
  
  points = np.array([[10,10,10]]);
  weights = np.random.rand(len(points));
  v = vox.voxelize(points, shape = (20,20,20),  weights=weights, radius=(2,2,2), method = 'rectangle');
  p3d.plot(v)
  
  def kernel(d):
    return np.exp(-d);
  v = vox.voxelize(points, shape = (20,20,20),  weights=None, radius=(5,5,8), kernel=kernel, method = 'sphere');
  p3d.plot(v)
  