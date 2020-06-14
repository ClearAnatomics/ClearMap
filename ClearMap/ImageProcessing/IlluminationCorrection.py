# -*- coding: utf-8 -*-
"""
IlluminationCorrection
======================

The module provides a function to correct systematic illumination variations
and vignetting in intensity.

The intensity image :math:`I(x)` given a flat field :math:`F(x)` and 
a background :math:`B(x)` the image is corrected to :math:`C(x)` as:
     
.. math:
   C(x) = \\frac{I(x) - B(x)}{F(x) - B(x)}

The module also has functionality to create flat field corections from measured 
intensity changes in a single direction, useful e.g. for lightsheet images,
see e.g. :func:`flatfield_from_regression`.

References
----------
..[LSM] Fundamentals of Light Microscopy and Electronic Imaging, p. 421
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


import numpy as np
import os

from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

import ClearMap.IO.IO as io
import ClearMap.Utils.Timer as tmr
import ClearMap.Utils.HierarchicalDict as hdict

import ClearMap.Settings as settings

###############################################################################
### Default parameter
###############################################################################

default_flat_field_line_file_name = os.path.join(settings.resources_path, "Microscope/lightsheet_flatfield_correction.csv");
"""Default file of points along the illumination changing line for the flat field correction

See Also:
    :func:`correct_illumination`
"""

###############################################################################
### Illuminaton correction
###############################################################################

def correct_illumination(source, flatfield = None, background = None, scaling = None, dtype = None, verbose = False):
  """Correct illumination and background.
 
  Arguments
  ---------
  source : array, str, or Source
    The image to correct for illumination.
  sink : array, str, Source, or None
    The sink to write results to.
  flatfield : str, array, Source or None
    The flatfield estimate. If None, no flat field correction is done.
  background : str, array, Source or None
    The background estimate. If None, backgorund is assumed to be zero.
  scaling : float, 'max', 'mean' or None
    Scale the corrected result by this factor. If 'max' or 'mean' scale the 
    result to match the 'max' or 'mean'. If None, dont scale the result.
  processes : int or None
    Number of processes to use. If None use macimal available.
  verbose : bool
    If true, print progrss infomration.
    
  Returns
  ------- 
  corrected : array
    Illumination corrected image.
        
  Note
  ----
  The intensity image :math:`I(x)` given a flat field :math:`F(x)` and 
  a background :math:`B(x)` image is corrected to :math:`C(x)` as:
  
  .. math:
      C(x) = \\frac{I(x) - B(x)}{F(x) - B(x)}
      
  If the background is not given :math:`B(x) = 0`. 
  
  The correction is done slice by slice assuming the data was collected with 
  a light sheet microscope.
  
  The image is finally optionally scaled.
       
  References
  ----------
  [1] Fundamentals of Light Microscopy and Electronic Imaging, p 421        
        
  See Also
  --------
  :const:`default_flatfield_line_file_name`
  """   
  
  if background is not None:
    background = io.as_source(background);
   
  if flatfield is None:
    return source; 
  if flatfield is True:
    # default flatfield correction
    flatfield = default_flat_field_line_file_name;
  if isinstance(flatfield, str):
    flatfield = io.as_source(flatfield);
  if flatfield.ndim == 1:
    flatfield = flatfield_from_line(flatfield, source.shape[1]);
  if flatfield.shape[:2] != source.shape[:2]:
      raise ValueError("The flatfield shape %r does not match the source shape %r!" % (flatfield.shape[:2],  source.shape[:2]));
  flatfield = io.as_source(flatfield);
  
  if verbose:    
    timer = tmr.Timer();
    hdict.pprint(head = 'Illumination correction:', flatfield=flatfield, background=background, scaling=scaling);  
  
  #initilaize source
  source = io.as_source(source);
  if dtype is None:
    dtype = source.dtype;
  
  # rescale factor
  flatfield = flatfield.array.astype(dtype);
  if scaling is True:
    scaling = "mean";
  if isinstance(scaling, str):
    if scaling.lower() == "mean":
      # scale back by average flat field correction:
      scaling = flatfield.mean();
    elif scaling.lower() == "max":
      scaling = flatfield.max();
    else:
      raise RuntimeError('Scaling not "max" or "mean" but %r!' % (scaling,));
  
  # illumination correction in each slice
  corrected = np.array(source.array, dtype=dtype)
  if background is None:
    for z in range(source.shape[2]):
      corrected[:,:,z] = source[:,:,z] / flatfield;
  else:
    if background.shape != flatfield.shape:
      raise RuntimeError("Illumination correction: background does not match source shape: %r vs %r!" % (background.shape,  source[:,:,0].shape));        
    background = background.array.astype('float32');
    flatfield = (flatfield - background);
    for z in range(source.shape[2]):
      corrected[:,:,z] = (source[:,:,z] - background) / flatfield;
  
  if not scaling is None:
    corrected= corrected * scaling;
  
  if verbose:
    timer.print_elapsed_time('Illumination correction');    
  
  return corrected 
    


def flatfield_from_line(line, shape, axis = 0, dtype = float):
  """Creates a 2d flat field image from a 1d line of estimated intensities.
  
  Arguments
  ---------
  line : array
    Array of intensities along the specified axis.
  shape : tuple
    Shape of the resulting image.
  axis : int
    Axis of the flat field line estimate.
  
  Returns
  -------
  flatfield : array 
    Full 2d flat field.
  """
  line = io.as_source(line);
  
  if isinstance(shape, int):
    shape = (line.shape[0], shape) if axis == 0 else (shape, line.shape[0]);
  if shape[axis] != line.shape[0]:
    raise ValueError('Line shape %d does not match image shape %d!' % (line.shape[0], shape[axis]));
  
  shape = shape[axis];
  flatfield = np.array([line.array] * shape, dtype=dtype)
  if axis == 1:
    flatfield = flatfield.T;
  
  return flatfield;


def flatfield_line_from_regression(source, sink = None, positions = None, method = 'polynomial', reverse = None, return_function = False, verbose = False):
  """Create flat field line fit from a list of positions and intensities.
      
  Arguments
  ---------
  source : str, array or Source
    Intensities as (n,)-vector or (n,m)-array of m intensity measurements
    at n points along an axis.
  sink : str, array, Source or None
    Sink for the result.
  positions : array, 'source' or None
    The positions of the soource points. If None, a linear increasing
    positions with equal spaccing is assumed. If 'source' take positions from
    first line of the source array.
  method : 'Gaussian' or 'Polynomial'
    function type for the fit.
  reverse : bool
    Reverse the line fit after fitting.
  return_function : bool
    If True, also return the fitted function.
  verbose :bool
    Print and plot information for the fit.
      
  Returns
  -------
  fit : array
    Fitted intensities on points.
  fit_function : function
    Fitted function.
      
  Note
  ----
  The fit is either to be assumed to be a 'Gaussian':
  
  .. math:
      I(x) = a \\exp^{- (x- x_0)^2 / (2 \\sigma)) + b"
      
  or follows a order 6 radial 'Polynomial'
      
  .. math:
      I(x) = a + b (x- x_0)^2 + c (x- x_0)^4 + d (x- x_0)^6
  """
  source = io.as_source(source);
  
  # split source
  if source.ndim == 1:
    y = np.atleast_2d(source.array);
  elif source.ndim == 2:
    if positions == 'source':
      positions = source[:,0];
      y = source[:,1:-1];
    else:
      y = source.array;
  else:
    raise RuntimeError('flatfield_line_from_regression: input data not a line or array of x,i data');
  
  if positions is None:
    positions = np.arange(source.shape[0])
  
  #calculate mean of the intensity measurements
  x = positions;
  ym = np.mean(y, axis = 1);

  if verbose > 1:
    plt.figure()
    for i in range(1,source.shape[1]):
      plt.plot(x, source[:,i]);
    plt.plot(x, ym, 'k');
  
  if method.lower() == 'polynomial':
    ## fit r^6
    mean = sum(ym * x)/sum(ym)

    def f(x,m,a,b,c,d):
      return a + b * (x-m)**2 + c * (x-m)**4 + d * (x-m)**6;
    
    popt, pcov = curve_fit(f, x, ym, p0 = (mean, 1, 1, 1, .1));
    m = popt[0]; a = popt[1]; b = popt[2];
    c = popt[3]; d = popt[4];     
    
    def fopt(x):
      return f(x, m = m, a = a, b = b, c = c, d = d);        
    
    if verbose:
      print("polynomial fit: %f + %f (x- %f)^2 + %f (x- %f)^4 + %f (x- %f)^6" % (a, b, m, c, m, d, m));
  
  else: 
    ## Gaussian fit       
    mean  = sum(ym * x)/sum(ym)
    sigma = sum(ym * (x-mean)**2)/(sum(ym))
    
    def f(x, a, m, s, b):
      return a * np.exp(- (x - m)**2 / 2 / s) + b;
      
    popt, pcov = curve_fit(f, x, ym, p0 = (1000, mean, sigma, 400));
    a = popt[0]; m = popt[1]; s = popt[2]; b = popt[3];
    
    def fopt(x):
      return f(x, a=a, m=m, s=s, b=b);
    
    if verbose:
      print("Gaussian fit: %f exp(- (x- %f)^2 / (2 %f)) + %f" % (a, m, s, b));
   
  fit = fopt(x);
  if reverse:
    fit.reverse();
  
  if verbose > 1:
    plt.plot(x, fit);
    plt.title('flatfield_line_from_regression')
  
  result = io.write(sink, fit);
  if return_function:
    result = (result, fopt)
  return result;



###############################################################################
### Tests
###############################################################################

def _test():
  """Tests"""
  import ClearMap.Visualization.Plot3d as p3d
  import ClearMap.ImageProcessing.IlluminationCorrection as ic
  
  ff = ic.flatfield_from_line(ic.default_flat_field_line_file_name, 100);
  p3d.plot(ff)