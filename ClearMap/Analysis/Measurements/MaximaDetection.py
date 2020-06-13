# -*- coding: utf-8 -*-
"""
MaximaDetection
===============

Collection of routines to detect maxima in scalar images.

Used for finding cells or intensity peaks.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'

import numpy as np

import scipy.ndimage.measurements as ndm
import scipy.ndimage.filters as ndf

import ClearMap.ImageProcessing.GreyReconstruction as gr

import ClearMap.Utils.Timer as tmr
import ClearMap.Utils.HierarchicalDict as hdict


##############################################################################
# Basic Transforms
##############################################################################
   
def h_max_transform(source, h_max):
  """H-maximum transform of an array.
  
  Arguments
  ---------
  source : array
    Input image.
  h_max : float or None
    H parameter of h-max transform, if None return source.
      
  Returns
  -------
  transformed : array
    H-max transformed image if h is not None.
  """
  if h_max is None:
    return source;
  else:
    return gr.reconstruct(source - h_max, source);


def local_max(source, shape = 5):
  """Calculates local maxima of an image
      
  Arguments
  ---------
  source : array
    Input image.
  shape : int, tuple or None
    Shape of the volume to search for maxima.
      
  Returns
  -------
  local_max : array
    Mask that is True at local maxima.
  """
  
  if shape is None:
    return source;
  
  if not isinstance(shape, tuple):
    shape = (shape,) * source.ndim;
  
  return ndf.maximum_filter(source, size=shape) == source
    

def extended_max(source, h_max = 0, shape = 5):
  """Calculates extended h-maxima of an image
  
  Arguments
  ---------
  source : array
    Input image.
  h_max : float or None)
    H parameter of h-max transform. If None, calculate local maxima only.
  shape : int, tuple or None
    Shape of the volume to search for maxima.
      
  Returns
  -------
  maxima : array
    Extended maxima of the image.
      
  Note
  ----
  Extended maxima are the local maxima of the h-max transform.
  """
  #h max transformimport scipy
  if not(h_max is None) and h_max > 0:
    source = h_max_transform(source, h_max);
      
  #max
  return local_max(source);


##############################################################################
# Maxima Detection 
##############################################################################


def find_maxima(source, h_max = None, shape = 5, threshold = None, verbose = None):
  """Find local and extended maxima in an image.
  
  Arguments
  ---------
  source : array
    The source data.
  h_max : float or None
    H parameter for the initial h-Max transform.
    If None, do not perform a h-max transform.
  shape : int or tuple
    Shape for the structure element for the local maxima filter.
  threshold : float or None
    If float, include only maxima larger than this threshold.
  verbose : bool
    Print progress info.
  
  Returns
  -------
  maxima : array
    Binary image with True pixel at extended maxima.
  
  Notes
  ----- 
  This routine performs a h-max transfrom, followed by a local maxima search 
  and thresholding of the maxima.
    
  See also
  --------
  :func:`h_max_transform`, :func:`local_max`
  """
  if verbose:
    timer = tmr.Timer();
    hdict.pprint(head='Find Maxima:', h_max=h_max, shape=shape, threshold=threshold);
  
  # extended maxima    
  maxima = h_max_transform(source, h_max=h_max);
      
  #local maxima
  maxima = local_max(maxima, shape=shape);
  
  #thresholding    
  if not threshold is None:
    maxima = np.logical_and(maxima, source >= threshold);
  
  if verbose:
    timer.print_elapsed_time(head='Find Maxima');
  
  return maxima


def find_center_of_maxima(source, maxima = None, label = None, verbose = False):
  """Find center of detected maxima weighted by intensity
  
  Arguments
  ---------
  source : array
    Intensity image data.
  maxima : array or None
    Binary array indicating the maxima. I label is not None this can be None.
  label : array or None
    Labeled image of the shapes of the maxima.  
    If None, determined from maxima.
  verbose : bool
    Print progress info.
  
  Returns
  -------
  coordinates : array
    Coordinates of the n centers of maxima as (n,d)-array.
  """
  if verbose:
    timer = tmr.Timer();
    print('Center of Maxima initialized!');
  
  #center of maxima
  if label is None:
    label, n_label = ndm.label(maxima);  
  else:
    n_label = label.max();
  
  if n_label > 0:
    centers = np.array(ndm.center_of_mass(source, label, index=np.arange(1, n_label)));    
  else:
    centers = np.zeros((0,source.ndim));
  
  if verbose:
    timer.print_elapsed_time('Center of Maxima: %d maxima detected' % centers.shape[0]);
    
  return centers;

