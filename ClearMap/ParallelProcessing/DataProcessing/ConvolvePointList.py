# -*- coding: utf-8 -*-
"""
ConvolvePointList
=================

Convolution on a subset of points only. 

Paralllel convolution of a kernel at specified points of the source only.
Useful to speed up processing in large arrays and only a smaller number of 
convolution points.

Note
----
  This module is heavily used in the 
  :mod"`~ClearMap.ImageProcessing.Skeletonization` algorithm.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


import numpy as np;
from multiprocessing import cpu_count;

import ClearMap.IO.IO as io

import pyximport; 
pyximport.install(setup_args={"include_dirs": [np.get_include()]}, reload_support=True)

import ClearMap.ParallelProcessing.DataProcessing.ConvolvePointListCode as code

###############################################################################
### Convolve point lists
###############################################################################

#TODO: use ArrayProcessing initialization tools 
def convolve_3d(source, kernel, points = None, indices = None, x = None, y = None, z = None, sink = None, sink_dtype = None, strides = None, check_border = True, processes = cpu_count()):
  """Convolves source with a specified kernel at specific points only.
    
  Arguments
  ---------
  source : array
    3d binary array to convolve.
  kernel : array
    Convolution kernel.
  points : array
    List of points to convolve.
  indices : array
    Indices to convolve.
  x,y,z : array
    Arrays of x,y,z coordinates of points to convolve on
  sink : array
    Optional sink to write result to.
  sink_dtype : dtype)
    Optional type of the sink. If None, the kernel type is used as default.
  strides : array
    The strides of the source in case its given as a 1d list.
  check_border : bool
    If True, check if each kernel element is inside the source array shape.
  processes : int or None
    Number of processes to use.
  
  Returns
  -------
  convolved : array
    List of results of convolution at specified points
    
  Note
  ----
  Either points x,y,z or an index array needs to be given. This function wraps
  more specialized functions
        
  See also
  --------
  convolve_3d_points, convolve_3d_xyz, convolve_3d_indices
  """
    
  if points is not None and points.ndim == 1:
      indices = points;
      points = None;
    
  if points is not None:
    return convolve_3d_points(source, kernel, points, sink = sink, sink_dtype = sink_dtype, check_border = check_border, processes = processes);
  elif indices is not None:
    return convolve_3d_indices(source, kernel, indices, sink = sink, sink_dtype = sink_dtype, check_border = check_border, processes = processes);
  elif x is not None and y is not None and z is not None:
    return convolve_3d_xyz(source, kernel, x, y, z, sink = sink, sink_dtype = sink_dtype, check_border = check_border, processes = processes);
  else:
    raise RuntimeError('Positions expected to be given as points, index or x,y,z arrays');
  


def convolve_3d_points(source, kernel, points, sink = None, sink_dtype = None, check_border = True, processes = cpu_count()):
  """Convolves source with a specified kernel at specific points only
  
  Arguments
  ---------
  source : array
    3d binary array to convolve.
  kernel : array
    Convolution kernel.
  points : array
    List of points to convolve.
  sink : array
    Optional sink to write result to.
  sink_dtype : dtype)
    Optional type of the sink. If None, the kernel type is used as default.
  check_border : bool
    If True, check if each kernel element is inside the source array shape.
  processes : int or None
    Number of processes to use.
  
  Returns
  -------
  convolved : array
    List of results of convolution at specified points
  """
  
  if source.dtype == bool:
    d = source.view('uint8');
  else:
    d = source;
  
  npts = points.shape[0];  
    
  if sink is None:
    if sink_dtype is None:
      sink_dtype = kernel.dtype;
    sink = np.zeros(npts, dtype = sink_dtype);
  
  if sink.shape[0] != npts:
     raise RuntimeError('The sinkput has not the expected size of %d but %d' % (npts, sink.shape[0]));
  
  if sink.dtype == bool:
    o = sink.view('uint8');
  else:
    o = sink;

  if kernel.dtype == bool:
    k = np.array(kernel, 'uint8');
  else:
    k = kernel;
  
  if processes is None:
    processes = cpu_count();
  
  if check_border:
    code.convolve_3d_points(d, k, points, o, processes);
  else:
    code.convolve_3d_points_no_check(d, k, points, o, processes);
  
  return sink;


def convolve_3d_xyz(source, kernel, x, y, z, sink = None, sink_dtype = None, check_border = True, processes = cpu_count()):
  """Convolves source with a specified kernel at specific points only
    
  Arguments
  ---------
  source : array
    3d binary array to convolve.
  kernel : array
    Convolution kernel.
  x,y,z : array
    Arrays of x,y,z coordinates of points to convolve on
  sink : array
    Optional sink to write result to.
  sink_dtype : dtype)
    Optional type of the sink. If None, the kernel type is used as default.
  check_border : bool
    If True, check if each kernel element is inside the source array shape.
  processes : int or None
    Number of processes to use.
  
  Returns
  -------
  convolved : array
    List of results of convolution.
  """
  
  if source.dtype == bool:
    d = source.view('uint8');
  else:
    d = source;
    
  npts = len(x);
    
  if sink is None:
    if sink_dtype is None:
      sink_dtype = kernel.dtype;
    sink = np.zeros(npts, dtype = sink_dtype);
  
  if sink.shape[0] != npts or len(y) != npts or len(z) != npts:
     raise RuntimeError('The sinkput has size %d and does not match the x,y,z coordinates of sizes: %d = %d = %d' % (sink.shape[0], len(x), len(y), len(z)));
  
  if sink.dtype == bool:
    o = sink.view('uint8');
  else:
    o = sink;

  if kernel.dtype == bool:
    k = np.array(kernel, 'uint8');
  else:
    k = kernel;
    
  if processes is None:
    processes = cpu_count();
  
  if check_border:
    code.convolve_3d_xyz(d, k, x, y, z, o, processes);
  else:
    code.convolve_3_xyz_no_check(d, k, x, y, z, o, processes);
  
  return sink;


def convolve_3d_indices(source, kernel, indices, sink = None, sink_dtype = None, strides = None, check_border = True, processes = cpu_count()):
  """Convolves source with a specified kernel at specific points given by a flat array index.
    
  Arguments
  ---------
  source : array
    3d binary array to convolve.
  kernel : array
    Convolution kernel.
  indices : array
    Indices to convolve.
  sink : array
    Optional sink to write result to.
  sink_dtype : dtype)
    Optional type of the sink. If None, the kernel type is used as default.
  strides : array
    The strides of the source in case its given as a 1d list.
  check_border : bool
    If True, check if each kernel element is inside the source array shape.
  processes : int or None
    Number of processes to use.
  
  Returns
  -------
  convolved : array
    List of results of convolution.
  """
  d = source.reshape(-1, order = 'A');
  if source.dtype == bool:
    d = d.view('uint8');
    
  npts = indices.shape[0];
    
  if sink is None:
    if sink_dtype is None:
      sink_dtype = kernel.dtype;
    sink = np.zeros(npts, dtype = sink_dtype);
  
  if sink.shape[0] != npts:
     raise RuntimeError('The sinkput has not the expected size of %d but %d' % (npts, sink.shape[0]));
  
  if sink.dtype == bool:
    o = sink.view('uint8');
  else:
    o = sink;

  if kernel.dtype == bool:
    k = np.array(kernel, 'uint8');
  else:
    k = kernel; 

  if processes is None:
    processes = cpu_count();
  
  if strides is None:
    strides = np.array(io.element_strides(source));
  
  #print d.dtype, strides.dtype, kernel.dtype, o.dtype
  if check_border:
    code.convolve_3d_indices(d, strides, k, indices, o, processes);
  else:
    code.convolve_3d_indices_no_check(d, strides, k, indices, o, processes);
  
  return sink;


def convolve_3d_indices_if_smaller_than(source, kernel, indices, max_value, sink = None, strides = None, check_border = True, processes = cpu_count()):
  """Convolves source with a specified kernel at specific points given by a flat array indx under conditon the value is smaller than a number
    
  Arguments
  ---------
  source : array
    3d binary array to convolve.
  kernel : array
    Convolution kernel.
  indices : array
    Indices to convolve.
  max_value : float
    Checks if the convolution result is smaller than this value.
  sink : array
    Optional sink to write result to.
  strides : array
    The strides of the source in case its given as a 1d list.
  check_border : bool
    If True, check if each kernel element is inside the source array shape.
  processes : int or None
    Number of processes to use.
  
  Returns
  -------
  convolved : array
    List of results of convolution at specified points
  """
  d = source.reshape(-1, order = 'A');
  if source.dtype == bool:
    d = d.view('uint8');
    
  npts = indices.shape[0];
  
  if sink is None:
    sink = np.zeros(npts, dtype = bool);
  
  if sink.shape[0] != npts:
     raise RuntimeError('The sinkput has not the expected size of %d but %d' % (npts, sink.shape[0]));
  
  if sink.dtype == bool:
    o = sink.view('uint8');
  else:
    o = sink;

  if kernel.dtype == bool:
    k = np.array(kernel, dtype = 'uint8');
  else:
    k = kernel;
  
  if processes is None:
    processes = cpu_count();
  
  if strides is None:
    strides = np.array(io.element_strides(source), dtype=int);
  
  #print d.dtype, strides.dtype, kernel.dtype, o.dtype
  if check_border:
    print(d.dtype, strides.dtype, k.dtype, indices.dtype, np.array(max_value).dtype, o.dtype)
    code.convolve_3d_indices_if_smaller_than(d, strides, k, indices, max_value, o, processes);
  else:
    code.convolve_3d_indices_if_smaller_than_no_check(d, strides, k, indices, max_value, o, processes);
  
  return sink;



def convolve_3d_find_smaller_than(source, search, indices, max_value, sink = None, processes = cpu_count()):
  """Convolves source with a specified kernel at specific points given by a flat array indx under conditon the value is smaller than a number
    
  Arguments:
        source (array): 3d binary array to convolve
        kernel (array): binary orinteger kernel to convolve
        points (array): list of points to convolve given by the flat array coordinates
        max_value (int): if result of convolution is smaller then this value return True otherwise False in the sink array
        processes (int): number of processors
    
  Returns:
        array: list of results of convolution at specified points
    
  Note:
        cython does not support bools -> use view on uint8 as numpy does
  """
  d = source.reshape(-1, order = 'A');
  if source.dtype == bool:
    d = d.view('uint8');
    
  npts = indices.shape[0];
  
  if sink is None:
    sink = np.zeros(npts, dtype = int);
  
  if sink.shape[0] != npts:
     raise RuntimeError('The sinkput has not the expected size of %d but %d' % (npts, sink.shape[0]));
  
  if sink.dtype == bool:
    o = sink.view('uint8');
  else:
    o = sink;
  
  if processes is None:
    processes = cpu_count();
  
  code.convolve_3d_find_smaller_than(d, search, indices, max_value, o, processes);
  
  return sink;


###############################################################################
### Tests
###############################################################################

def test():
  import numpy as np
  import ClearMap.sourceProcessing.ConvolvePointList as cpl
  
  from importlib import reload
  reload(cpl);
  reload(cpl.code);
  
  #source = np.random.rand(2000,1000,1000) > 0.5;
  source = np.random.rand(100,100,500) > 0.5;
  source[[0,-1],:,:] = 0;
  source[:,[0,-1],:] = 0;
  source[:,:,[0,-1]] = 0;
  pts = np.where(source);
  pts = np.array(pts, dtype = int).T;
  
  #np.save('source.npy', source);
  #np.save('points.npy', pts)
  
  from ClearMap.ImageProcessing.Topology.Topology3d import n6
  n6i = np.array(n6, dtype = int)
  
  import time;
  t0 = time.time();
  result = cpl.convolve_3d(source, n6i, pts, processes=24, check_border = True)
  t1 = time.time();
  print('%f secs' % ((t1-t0)));
  
  import scipy.ndimage
  t0 = time.time();
  good = scipy.ndimage.filters.convolve(np.array(source, dtype = float), np.array(n6, dtype = float), mode = 'constant', cval = 0)
  t1 = time.time();
  print('%f secs' % ((t1-t0)));
  
  x,y,z = pts.T
  if np.all(good[x,y,z].astype(result.dtype) == result):
    print('works')
  else:
    print('error!')
  
  
  ptsi = np.where(source.reshape(-1, order = 'A'))[0];
  t0 = time.time();
  result = cpl.convolve_3d_index(source, n6i, ptsi, processes=24)
  t1 = time.time();
  print('%f secs' % ((t1-t0)));
  
  x,y,z = pts.T
  if np.all(good[x,y,z].astype(result.dtype) == result):
    print('works')
  else:
    print('error!')
  
  ptsi = np.where(source.reshape(-1, order = 'A'))[0];
  t0 = time.time();
  resultc = cpl.convolve_3d_index_if_smaller_than(source, n6i, ptsi, 3, processes=24)
  t1 = time.time();
  print('%f secs' % ((t1-t0)));
  
  np.all(resultc == (result < 3))
  
  
  x,y,z = pts.T
  if np.all(good[x,y,z].astype(result.dtype) == result):
    print('works');
  else:
    print('error!');
  
  #import os
  #os.remove('source.npy');
  #os.remove('points.npy')
