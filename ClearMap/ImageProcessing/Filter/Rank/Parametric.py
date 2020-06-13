# -*- coding: utf-8 -*-
"""
Parameteric rank filters 

This module provides rank filters that depend on several parameters.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


import numpy as np

from . import Rank as rnk

import pyximport;
pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)

from . import ParametricCode as code


__all__ = ['nilblack', 'sauvola', 'contrast_limited_percentile', 'light_sheet_artifact_correction']


def nilblack(source, selem=None, sink=None, mask=None, k=0.0, **kwargs):
  """Local nilblack threshold of an image.

  The resulting binary mask is True if the greyvalue of the center pixel is
  greater than the local mean.

  Arguments
  ---------
  source : array
    Input array.
  selem : array
    Structure element. If None, use a cube of size 3.
  sink : array
    Output array. If None, a new array is allocated.
  mask : array
    Optional mask, if None, the complete source is used.
    Pixels on the mask are zero in the output.
  max_bin : int or None
    Maximal number of bins.
  k : float
    The factor between mean and standard deviation: threshold = mu + k * std

  Returns
  -------
  sink : array
    The filtered array.
  """
  return rnk._apply_code(code.nilblack, code.nilblack_masked, sink_dtype = bool,
                         source=source, selem=selem, sink=sink, mask=mask, parameter_float=[k], **kwargs);
    
    
def sauvola(source, selem=None, sink=None, mask=None, k=0.0, R=1.0):
  """Local Sauvola threshold of an image.

  The resulting binary mask is True if the greyvalue of the center pixel is
  greater than the local mean.

  Arguments
  ---------
  source : array
    Input array.
  selem : array
    Structure element. If None, use a cube of size 3.
  sink : array
    Output array. If None, a new array is allocated.
  mask : array
    Optional mask, if None, the complete source is used.
    Pixels on the mask are zero in the output.
  max_bin : int or None
    Maximal number of bins.
  k, R : float
    The factors in the formula: threshold = mu + k * (std/R - 1) * (mu - max_bin)

  Returns
  -------
  sink : array
    The filtered array.
  """
  return rnk._apply_code(code.sauvola, code.sauvola_masked, sink_dtype = None,
                         source=source, selem=selem, sink=sink, mask=mask, parameter_float=[k,R]);
 


    
def contrast_limited_percentile(source, selem=None, sink=None, mask=None, percentile=0.0, limit = None, contrast_limit = 0.1, **kwargs):
  """Local Sauvola threshold of an image.

  The resulting binary mask is True if the greyvalue of the center pixel is
  greater than the local mean.

  Arguments
  ---------
  source : array
    Input array.
  selem : array
    Structure element. If None, use a cube of size 3.
  sink : array
    Output array. If None, a new array is allocated.
  mask : array
    Optional mask, if None, the complete source is used.
    Pixels on the mask are zero in the output.
  max_bin : int or None
    Maximal number of bins.
  k, R : float
    The factors in the formula: threshold = mu + k * (std/R - 1) * (mu - max_bin)

  Returns
  -------
  sink : array
    The filtered array.
  """
  if limit is None:
    selem = rnk._initialize_selem(selem, source.ndim);
    n_selem = np.sum(selem);
    limit = int(np.round(contrast_limit * n_selem));
  
  parameter_float = [percentile];
  parameter_index = np.hstack([[limit],  np.zeros(max_bin, dtype=int)]);                         
  return rnk._apply_code(code.clp, code.clp_masked, sink_dtype = float,
                         source=source, selem=selem, sink=sink, mask=mask, 
                         parameter_index=parameter_index, parameter_float=parameter_float, **kwargs);
   
                         
                         
def light_sheet_artifact_correction(source, selem=None, sink=None, mask=None, percentiles = (0.0, 1.0), limit = None, contrast_limit = 0.5, **kwargs):
  """Local Sauvola threshold of an image.

  The resulting binary mask is True if the greyvalue of the center pixel is
  greater than the local mean.

  Arguments
  ---------
  source : array
    Input array.
  selem : array
    Structure element. If None, use a cube of size 3.
  sink : array
    Output array. If None, a new array is allocated.
  mask : array
    Optional mask, if None, the complete source is used.
    Pixels on the mask are zero in the output.
  max_bin : int or None
    Maximal number of bins.
  k, R : float
    The factors in the formula: threshold = mu + k * (std/R - 1) * (mu - max_bin)

  Returns
  -------
  sink : array
    The filtered array.
  """
  if limit is None:
    selem = rnk._initialize_selem(selem, source.ndim);
    n_selem = np.sum(selem);
    limit = int(np.round(contrast_limit * n_selem));
  
  parameter_float = np.array(percentiles, dtype = float);
  parameter_index = np.hstack([[limit],  np.zeros(max_bin, dtype=int)]);
                             
  return rnk._apply_code(code.lsac, code.lsac_masked, sink_dtype = float,
                         source=source, selem=selem, sink=sink, mask=mask, 
                         parameter_index=parameter_index, parameter_float=parameter_float, **kwargs);                         
                         