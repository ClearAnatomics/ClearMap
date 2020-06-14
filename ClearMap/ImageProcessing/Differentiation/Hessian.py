"""
Hessian
=======

Module to compute curvature measures based on Hessian Matrix

Usefull for filtering vasculature data
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'

import numpy as np
import scipy.ndimage as ndi    


import ClearMap.IO.IO as io


import pyximport;
pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)

from . import HessianCode as code

#import ClearMap.IO.IO as io

__all__ = ['hessian', 'eigenvalues', 'lambda123', 'tubeness'];


###############################################################################
### Curvature measures
###############################################################################
          
def hessian(source, sink = None, sigma = None):
  """Returns the hessian matrix at each location calculatd via finite differences.
  
  Arguments
  ---------
  source : array
    Input array.
  sink : array
    Output, if None, a new array is allocated.  

  Returns
  -------
  hessian : array:
      5d array with the hessian matrix in the first two dimensions.
  """    
  return _apply_code(code.hessian, source, sink, sink_shape_per_pixel = (2,2), parameter = None, sigma = sigma);
  

def eigenvalues(source, sink = None, sigma = None):
  """Hessiean eigenvalues of source data

  Arguments
  ---------
  source : array
    Input array.
  sink : array
    Output, if None, a new array is allocated.
  sigma : float or None
    If not None, a Gaussian filter with std sigma is applied initialliy.
  
  Returns
  -------
  sink : array
    The three eigenvalues along the first axis for each source.
  """
  return _apply_code(code.eigenvalues, source, sink, sink_shape_per_pixel = (3,), parameter = None, sigma = sigma);
          

def tubeness(source, sink = None, threshold = None, sigma = None):
  """Tubeness mesure of source data

  Arguments
  ---------
  srouce : array
    Input array.
  sink : array
    Output, if None, a new array is allocated.
  threshold : float or None
    If float, the tubeness is thresholded at this level.
  sigma : float or None
    If not None, a Gaussian filter with std sigma is applied initialliy.

  Returns
  -------
  sink : 3-D array
      Tubness output.

  Note
  ----
  The tubness is the geometric mean of the two smallest eigenvalues.      
  """
  if threshold is None:
    return _apply_code(code.tubeness, source, sink, sigma = sigma, sink_dtype = float)
  else: 
    return _apply_code(code.tubeness_threshold, source, sink, parameter = threshold, sigma = sigma, sink_dtype = bool)


def lambda123(source, sink = None, gamma12 = 1.0, gamma23 = 1.0, alpha = 0.25, sigma = None, threshold = None):
  """Generalized tubness measure of source data.

  Arguments
  ---------
  source : array
    Input array.
  sink : array
    Output, if None, a new array is allocated.
  gamma12, gamma23, alpha : float
    Parameters for the tubness measure.
  sigma : float or None
    If not None, a Gaussian filter with std sigma is applied initialliy.
  
  Returns
  -------
  sink : array
    The tubness measure.
  
  Note
  ----
  Reference: Sato et al. Three-dimensional multi-scale line filter for segmentation and visualization of curvilinear structures in medical images, Medical Image Analysis 1998, pp 143--168.
  """  
  if threshold is None:
    parameter = np.array([gamma12, gamma23, alpha], dtype = float);
    return _apply_code(code.lambda123, source, sink, parameter=parameter, sigma=sigma, sink_dtype=float);          
  else:
    parameter = np.array([gamma12, gamma23, alpha, threshold], dtype = float);
    return _apply_code(code.lambda123_threshold, source, sink, parameter=parameter, sigma=sigma, sink_dtype=bool);          


###############################################################################
### Helpers
###############################################################################

def _apply_code(function, source, sink, sink_dtype = None, sink_shape_per_pixel = None, parameter = None, sigma = None):
  """Helper to apply the core functions"""
  if source.ndim != 3:
    raise ValueError('The tubness measure is implemented for 3d data, found %dd!' % source.ndim);
  
  if source is sink:
    raise ValueError("Cannot perform operation in place!")

  if sink_shape_per_pixel is None:
    shape_per_pixel = (1,);
  else:
    shape_per_pixel = sink_shape_per_pixel;
  
  if sink is None:
    if sink_dtype is None:
      sink_dtype = float
    sink = np.zeros(source.shape + shape_per_pixel, dtype = sink_dtype, order = 'F')
  else:
    if shape_per_pixel != (1,):
      if sink.shape != source.shape + shape_per_pixel:
        raise ValueError('The sink of shape %r does not have expected shape %r!' % (sink.shape, source.shape + shape_per_pixel));
    if sink.shape != source.shape + shape_per_pixel:
      sink = sink.reshape(source.shape + shape_per_pixel)
  
  if sink.dtype == bool:
    s = sink.view('uint8')
  else:
    s = sink;

  sink_stride = io.element_strides(sink)[-1];
    
  if parameter is None:
    parameter = np.zeros(0);
  parameter = np.asarray([parameter], dtype = float).flatten();
                     
  if sigma is not None:
    data = ndi.gaussian_filter(np.asarray(source, dtype=float), sigma=sigma);
  else:
    data = np.asarray(source, dtype=float);
  
  function(source=data, sink=s, sink_stride=sink_stride,  parameter=parameter)
  
  if sink_shape_per_pixel is None:
    sink = sink.reshape(sink.shape[:-1]);
  
  return sink


###############################################################################
### Tests
###############################################################################

def _test():
  import numpy as np
  import ClearMap.Test.Files as tst
  import ClearMap.Visualization.Plot3d as p3d
  
  import ClearMap.ImageProcessing.Differentiation as dif
  
  from importlib import reload
  reload(dif);
  
  source = tst.init('v')[:20, :20, :20]; 
  
  sink = dif.eigenvalues(source, sigma = 1.0);
  p3d.plot([source] + list(sink.transpose([3,0,1,2])));
    
  sink = np.zeros(source.shape + (3,), order = 'C');
  dif.eigenvalues(source, sink, sigma = 1.0);
  p3d.plot([source] + list(sink.transpose([3,0,1,2])));

  sink = np.zeros(source.shape + (3,), order = 'F');
  dif.eigenvalues(source, sink, sigma = 1.0);
  p3d.plot([source] + list(sink.transpose([3,0,1,2])));
  
  sink = np.zeros(source.shape);
  dif.lambda123(source, sink, sigma = 1.0);
  p3d.plot([source, sink])
  
  sink = dif.lambda123(source, sigma = 1.0, threshold = 0.2);
  p3d.plot([source, sink]);
  
  #import ClearMap.ImageProcessing.Differentiation.Gradient as grd
  #res2 = grd.eigenvalues(data, sigma = 1.0);

  #import numpy as np
  #np.max(np.abs(res[[2,1,0]] - res2))
  
  #print res[:,10,10,10], res2[:,10,10,10]



# Python implementations of HessianCode for testing
#from scipy import ndimage as ndi  
#
#def hessian(source):
#  """Returns the hessian matrix at each location calculatd via finite differences.
#  
#  Arguments
#  ---------
#  source : array
#    Input array.
#  sink : array
#    Output, if None, a new array is allocated.  
#
#  Returns
#  -------
#    ndarray:
#      5d array with the hessian matrix in the first two dimensions
#  """    
#  c = 2*source;
#  h[0,0] = mm[0:-2,1:-1,1:-1] - c + mm[2:,1:-1,1:-1];
#  h[1,1] = mm[1:-1,0:-2,1:-1] - c + mm[1:-1,2:,1:-1];
#  h[2,2] = mm[1:-1,1:-1,0:-2] - c + mm[1:-1,1:-1,2:];
#  
#  h[0,1] = h[1,0] = (mm[2:,2:,1:-1] - mm[:-2,2:,1:-1] - mm[2:,:-2,1:-1] + mm[:-2,:-2,1:-1]) / 4;
#  h[0,2] = h[2,0] = (mm[2:,1:-1,2:] - mm[:-2,1:-1,2:] - mm[2:,1:-1,:-2] + mm[:-2,1:-1,:-2]) / 4;
#  h[1,2] = h[2,1] = (mm[1:-1,2:,2:] - mm[1:-1,:-2,2:] - mm[1:-1,2:,:-2] + mm[1:-1,:-2,:-2]) / 4;
#
#  return h;
#
#def eigenvalues3D(m):
#    """Find the coefficients of the characteristic polynomial:
#       http://en.wikipedia.org/wiki/Eigenvalue_algorithm
#
#		// The matrix looks like:
#		/*
#			A  B  C
#			B  D  E
#			C  E  F
#		*/
#    """
#    
#    A = m[0,0];
#    B = m[0,1];
#    C = m[0,2];
#    D = m[1,1];
#    E = m[1,2];
#    F = m[2,2];
#
#    a = -1;
#    b = A + D + F;
#    c = B * B + C * C + E * E - A * D - A * F - D * F;
#    d = A * D * F - A * E * E	- B * B * F + 2 * B * C * E - C * C * D;
#
#    third = 0.333333333333333333333333333333333333;
#
#    #Now use the root-finding formula described here:
#    #http://en.wikipedia.org/wiki/Cubic_equation#oot-finding_formula
#
#    q = (3*a*c - b*b) / (9*a*a);
#    r = (9*a*b*c - 27*a*a*d - 2*b*b*b) / (54*a*a*a);
#
#    discriminant = q*q*q + r*r;
#    #print discriminant.shape
#    
#    eigenValues = np.zeros((3,) + m.shape[2:]);
#    #print eigenValues.shape
#
#    #ids = discriminant > 0;
#    #if np.any(ids):
#        #should never happen for symmetric matrix
#    #    return None;
#
#    ids = discriminant < 0;
#    
#    rr = r[ids];
#      
#    rootThree = 1.7320508075688772935;
#    innerSize = np.sqrt( rr*rr - discriminant[ids] );
#      
#    ids2 = rr > 0;
#    innerAngle = np.zeros(rr.shape);
#    
#    innerAngle[ids2] = np.arctan(np.sqrt(-discriminant[ids][ids2]) / rr[ids2] );
#    
#    ids2 = np.logical_not(ids2);
#    innerAngle[ids2] = ( np.pi - np.arctan( np.sqrt(-discriminant[ids][ids2]) / -rr[ids2] ) );
#       
#    # So now s is the cube root of innerSize * e ^ (   innerAngle * i )
#    # and t is the cube root of innerSize * e ^ ( - innerAngle * i )       
#       
#    stSize = np.power(innerSize,third);
#      
#    sAngle = innerAngle / 3;
#    #tAngle = - innerAngle / 3;
#     
#    sPlusT = 2 * stSize * np.cos(sAngle);
#    
#    eigenValues[0][ids] = ( sPlusT - (b[ids] / (3*a)) );
#    firstPart = - (sPlusT / 2) - (b[ids] / 3*a);
#    lastPart = - rootThree * stSize * np.sin(sAngle);
#      
#    eigenValues[1][ids] = ( firstPart + lastPart );
#    eigenValues[2][ids] = ( firstPart - lastPart );
#
#    
#    #The discriminant is zero (or very small),
#    #so the second two solutions are the same:
#    ids = discriminant == 0;
#    rr = r[ids];    
#    
#    ids2 = rr >= 0;
#        
#    
#    sPlusT = np.zeros(rr.shape);
#    sPlusT[ids2] = 2 * np.power(rr[ids2],third);
#    ids2 = np.logical_not(ids2);
#    sPlusT[ids2] = -2 * np.power(-rr[ids2],third);
#        
#    bOver3A = b[ids] / (3 * a);
#      
#    eigenValues[0][ids] = ( sPlusT   - bOver3A );
#    eigenValues[1][ids] = (-sPlusT/2 - bOver3A );
#    eigenValues[2][ids] = eigenValues[1][ids];
#      
#    return eigenValues;
##
##
#def eigenvalues(m, sigma = 1.0, sort = True):
#  if sigma is not None:
#    smoothed = ndi.gaussian_filter(np.asarray(m, dtype = float), sigma=sigma);
#  else:
#    smoothed = m;
#
#  h3 = hessian(smoothed);
#  e3 = eigenvalues3D(h3);
#  #e3a = np.abs(e3);
#  e3a = e3;
#  
#  if sort:
#    index = list(np.ix_(*[np.arange(i) for i in e3a.shape]))
#    index[0] = e3a.argsort(axis = 0)
#    return e3[index];
#  else:
#    return e3;
#
#
#def tubeness(m, sigma = 1.0):   
#  if sigma is not None:
#    smoothed = ndi.gaussian_filter(np.asarray(m, dtype = float), sigma=sigma);
#  else:
#    smoothed = m;
#
#  h3 = hessian(smoothed);
#  e3 = eigenvalues3D(h3);
#  e3a = np.abs(e3);
#  
#  index = list(np.ix_(*[np.arange(i) for i in e3a.shape]))
#  index[0] = e3a.argsort(axis = 0)
#  e3s = e3[index];
#  
#  tb = np.zeros(m.shape);
#  ids = np.logical_and(e3s[1] < 0, e3s[2] < 0);
#  tb[ids] = np.sqrt(e3s[1][ids] * e3s[2][ids]);
#
#  return tb;
#
#
#def frangi(m, alpha = 0.5, beta  = 0.5, gamma  = 100, sigma = 1.0):
#  if sigma is not None:
#    smoothed = ndi.gaussian_filter(np.asarray(m, dtype = float), sigma=sigma);
#  else:
#    smoothed = m;
#
#  h3 = hessian(smoothed);
#  e3 = eigenvalues3D(h3);
#  e3a = np.abs(e3);
#  
#  index = list(np.ix_(*[np.arange(i) for i in e3a.shape]))
#  index[0] = e3a.argsort(axis = 0)
#  e3s = e3[index];
#  
#  ra = np.abs(e3s[1]) / np.abs(e3s[2]);
#  rb = np.abs(e3s[0]) / np.sqrt(np.abs(e3s[1] * e3s[2]));
#  s  = np.sqrt(np.square(e3s[0]) + np.square(e3s[1]) + np.square(e3s[2]))
#
#  plate = 1 - np.exp(- np.square(ra) / (2 * np.square(alpha)));
#  blob  = np.exp(-np.square(rb) / (2 * np.square(beta)));
#  background = 1 - np.exp(-np.square(s) / (2 * np.square(gamma)));
#
#  f = plate * blob * background;
#  
#  ids = np.logical_and(e3s[1] >= 0, e3s[2] >= 0);
#  f[ids] = 0;
#  
#  return f;
#