# -*- coding: utf-8 -*-
"""
LocalStatistics
===============

Module provides functions to calculate local data and statistics of an image 
and apply a function to those. It is useful for local and adaptive image 
processing of large 3d images.

Note
----
The module provides ways to speed up the local statistics by only sampling
on a sub-grid of the image and resample the result to the full image shape.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'

import numpy as np

import scipy.ndimage as ndi

###############################################################################
### Local image processing 
###############################################################################
    
def apply_local_function(source, function, selem = (50,50), spacing = None, step = None, interpolate = 2, mask = None, fshape = None, dtype = None, return_centers = False):
  """Calculate local histograms on a sub-grid, apply a scalar valued function and resmaple to original image shape.
  
  Arguments
  ---------
  source : array
    The source to process.
  function : function
    Function to apply to the linear array of the local source data.
    If the function does not return a scalar, fshape has to be given.
  selem : tuple or array or None
    The structural element to use to extract the local image data.
    If tuple, use a rectangular region of this shape. If array, the array
    is assumed to be bool and acts as a local mask around the center point.
  spacing : tuple or None
    The spacing between sample points. If None, use shape of selem.
  step : tuple of int or None
    If tuple, subsample the local region by these step. Note that the
    selem is applied after this subsampling.
  interpolate : int or None
    If int, resample the result back to the original source shape using this
    order of interpolation. If None, return the results on the sub-grid.
  mask : array or None
    Optional mask to use.
  fshape : tuple or None
    If tuple, this is the shape of the function output. 
    If None assumed to be (1,).
  dtype : dtype or None
    Optional data type for the result.
  return_centers : bool
    If True, additionaly return the centers of the sampling.
    
  Returns
  -------
  local : array
    The reuslt of applying the function to the local samples.
  cetners : array
    Optional cttners of the sampling.
  """
  
  if spacing is None:
    spacing = selem;
  shape = source.shape;
  ndim = len(shape);
  
  if step is None:
    step = (None,) * ndim
    
  if len(spacing) != ndim or len(step) != ndim:
    raise ValueError('Dimension mismatch in the parameters!')  
    
  #histogram centers
  n_centers = tuple(s//h for s,h in zip(shape, spacing))
  left = tuple((s - (n-1) * h)//2  for s,n,h in zip(shape, n_centers, spacing));
  
  #center points
  centers = np.array(np.meshgrid(*[range(l, s, h) for l,s,h in zip(left, shape, spacing)], indexing = 'ij'));
  #centers = np.reshape(np.moveaxis(centers, 0, -1),(-1,len(shape)));               
  centers = np.moveaxis(centers, 0, -1)                                          
  
  #create result
  rshape = (1,) if fshape is None else fshape;
  rdtype = source.dtype if dtype is None else dtype;
  results = np.zeros(n_centers + rshape, dtype = rdtype);
  
  #calculate function
  centers_flat = np.reshape(centers, (-1,ndim));
  results_flat = np.reshape(results, (-1,) + rshape);
  
  #structuring element
  if isinstance(selem, np.ndarray):
    selem_shape = selem.shape
  else:
    selem_shape = selem;
    selem = None
  
  hshape_left = tuple(h//2 for h in selem_shape);           
  hshape_right = tuple(h - l for h,l in zip(selem_shape, hshape_left));
  
  for result, center in zip(results_flat,centers_flat):
    sl = tuple(slice(max(0,c-l), min(c+r,s), d) for c,l,r,s,d in zip(center, hshape_left, hshape_right, shape, step));
    if selem is None:
      if mask is not None:
        data = source[sl][mask[sl]];
      else:
        data = source[sl].flatten();
    else:
      slm = tuple(slice(None if c-l >= 0 else min(l-c,m), None if c+r <= s else min(m - (c + r - s), m), d) for c,l,r,s,d,m in zip(center, hshape_left, hshape_right, shape, step, selem_shape));
      data = source[sl];
      if mask is not None:
        data = data[np.logical_and(mask[sl], selem[slm])]
      else:
        data = data[selem[slm]];
      
    #print result.shape, data.shape, function(data)
    result[:] = function(data);
  
  #resample
  if interpolate:
    res_shape = results.shape[:len(shape)];
    zoom = tuple(float(s) / float(r) for s,r in zip(shape, res_shape));
    results_flat = np.reshape(results, res_shape + (-1,));  
    results_flat = np.moveaxis(results_flat, -1, 0);
    full = np.zeros(shape + rshape, dtype = results.dtype);
    full_flat = np.reshape(full, shape + (-1,));     
    full_flat = np.moveaxis(full_flat, -1, 0); 
    #print results_flat.shape, full_flat.shape
    for r,f in zip(results_flat, full_flat):
      f[:] = ndi.zoom(r, zoom=zoom, order = interpolate);   
    results = full;
    
  if fshape is None: 
    results.shape = results.shape[:-1];
  
  if return_centers:
    return results, centers
  else:
    return results


def local_histogram(source, max_bin = 2**12, selem = (50,50), spacing = None, step = None, interpolate = None, mask = None, dtype = None, return_centers = False):
  """Calculate local histograms on a sub-grid.
  
  Arguments
  ---------
  source : array
    The source to process.
  selem : tuple or array or None
    The structural element to use to extract the local image data.
    If tuple, use a rectangular region of this shape. If array, the array
    is assumed to be bool and acts as a local mask around the center point.
  spacing : tuple or None
    The spacing between sample points. If None, use shape of selem.
  step : tuple of int or None
    If tuple, subsample the local region by these step. Note that the
    selem is applied after this subsampling.
  interpolate : int or None
    If int, resample the result back to the original source shape using this
    order of interpolation. If None, return the results on the sub-grid.
  mask : array or None
    Optional mask to use.
  max_bin : int
    Maximal bin value to account for.
  return_centers : bool
    If True, additionaly return the centers of the sampling.
    
  Returns
  -------
  histograms : array
    The local histograms.
  cetners : array
    Optional centers of the sampling.
    
  Note
  ----
  For speed, this function works only for uint sources as the histogram is 
  calculated directly via the source values. The source values should be 
  smaller than max_bin.
  """
  
  def _hist(data):
    data, counts = np.unique(data, return_counts=True);
    histogram = np.zeros(max_bin, dtype=int);
    histogram[data] = counts;
    return histogram;
  
  return apply_local_function(source, selem=selem, spacing=spacing, step=step, interpolate=interpolate, mask=mask, dtype=dtype, return_centers=return_centers,
                             function=_hist, fshape = (max_bin,));
    


def local_percentile(source, percentile, selem = (50,50), spacing = None, step = None, interpolate = 1, mask = None, dtype = None, return_centers = False):
  """Calculate local percentile.
  
  Arguments
  ---------
  source : array
    The source to process.
  percentile : float or array
    The percentile(s) to estimate locally.
  selem : tuple or array or None
    The structural element to use to extract the local image data.
    If tuple, use a rectangular region of this shape. If array, the array
    is assumed to be bool and acts as a local mask around the center point.
  spacing : tuple or None
    The spacing between sample points. If None, use shape of selem.
  step : tuple of int or None
    If tuple, subsample the local region by these step. Note that the
    selem is applied after this subsampling.
  interpolate : int or None
    If int, resample the result back to the original source shape using this
    order of interpolation. If None, return the results on the sub-grid.
  mask : array or None
    Optional mask to use.
  return_centers : bool
    If True, additionaly return the centers of the sampling.
    
  Returns
  -------
  percentiles : array
    The local percentiles.
  cetners : array
    Optional centers of the sampling.
  """
  if isinstance(percentile, (tuple, list)):
    percentile = np.array([100*p for p in percentile]);
    fshape = (len(percentile),)
    def _percentile(data):
      if len(data) == 0:
        return np.array((0,) * len(percentile));
      return np.percentile(data, percentile, axis = None);
  
  else:
    percentile = 100 * percentile;
    fshape = None;
    def _percentile(data):
      if len(data) == 0:
        return 0;
      return np.percentile(data, percentile, axis = None);
  
  return apply_local_function(source, selem=selem, spacing=spacing, step=step, interpolate=interpolate, mask=mask, dtype=dtype, return_centers=return_centers,
                              function=_percentile, fshape=fshape);


###############################################################################
### Tests
###############################################################################                         
                              
def _test():
  """Tests."""
  import numpy as np
  import ClearMap.Visualization.Plot3d as p3d

  import ClearMap.ImageProcessing.LocalStatistics as ls     
  from importlib import reload
  reload(ls)                
                      
  source = np.random.rand(100,200,150) + np.arange(100)[:,None,None];
  p = ls.local_percentile(source, percentile=0.5, selem=(30,30,30), interpolate=1);
  p3d.plot([source, p])
  
                           
                              
                              
                              
                              
#def apply_histogram_function(histograms, function, shape = None, interpolation = 2):
#
#  hist_shape = histograms.shape[:-1];
#  max_bin = histograms.shape[-1];
#  result = np.zeros(np.prod(hist_shape), dtype = float);
#  histograms_flat = np.reshape(histograms, (-1,max_bin));
#  for i,h in enumerate(histograms_flat):
#    result[i] = function(h);
#  result.shape = hist_shape;
#  
#  if shape is not None:
#    zoom = tuple(float(s) / float(h) for s,h in zip(shape, hist_shape))
#    result = ndi.zoom(result, zoom=zoom, order = interpolation);
#  
#  return result;




