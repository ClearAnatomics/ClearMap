"""Inferior and superior ranks, provided by the user, are passed to the kernel
function to provide a softer version of the rank filters. E.g.
``autolevel_percentile`` will stretch image levels between percentile [p0, p1]
instead of using [min, max]. It means that isolated bright or dark pixels will
not produce halos.

The local histogram is computed using a sliding window.

Input image can be 8-bit or 16-bit, for 16-bit input images, the number of
histogram bins is determined from the maximum value present in the image.

Result image is 8-/16-bit or double with respect to the input image and the
rank filter operation.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'
__note__      = "Code adpated from skimage.filters.rank"


import numpy as np

from . import Rank as rnk

import pyximport;
pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)

from . import PercentileCode as code


__all__ = ['autolevel_percentile', 'gradient_percentile',
           'mean_percentile', 'subtract_mean_percentile',
           'enhance_contrast_percentile', 'percentile', 'pop_percentile',
           'sum_percentile', 'sum_above_percentile', 'threshold_percentile']


def autolevel_percentile(source, selem=None, sink=None, mask=None, percentiles=(0,1), **kwargs):
  """Return greyscale local autolevel of an image.
  
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
  percentiles : tuple of floats
    The lower and upper percentiles in [0,1] to use for 
    calculating the histogram.

  Returns
  -------
  sink : array
    The filtered array.
  """
  return rnk._apply_code(code.autolevel, code.autolevel_masked,
                         source=source, selem=selem, sink=sink, mask=mask, parameter_float=percentiles, **kwargs);
    


def gradient_percentile(source, selem=None, sink=None, mask=None, percentiles=(0,1), **kwargs):
  """Return local gradient of an image (i.e. local maximum - local minimum).

  Only greyvalues between percentiles [p0, p1] are considered in the filter.
  
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
  percentiles : tuple of floats
    The lower and upper percentiles in [0,1] to use for 
    calculating the histogram.

  Returns
  -------
  sink : array
    The filtered array.
  """
  return rnk._apply_code(code.gradient, code.gradient_masked,
                         source=source, selem=selem, sink=sink, mask=mask, parameter_float=percentiles, **kwargs);



def mean_percentile(source, selem=None, sink=None, mask=None, percentiles=(0,1), **kwargs):
  """Return local mean of an image.

  Only greyvalues between percentiles [p0, p1] are considered in the filter.
  
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
  percentiles : tuple of floats
    The lower and upper percentiles in [0,1] to use for 
    calculating the histogram.

  Returns
  -------
  sink : array
    The filtered array.
  """
  return rnk._apply_code(code.mean, code.mean_masked, sink_dtype = float,
                         source=source, selem=selem, sink=sink, mask=mask, parameter_float=percentiles **kwargs);



def subtract_mean_percentile(source, selem=None, sink=None, mask=None, percentiles=(0,1), **kwargs):
  """Return image subtracted from its local mean.

  Only greyvalues between percentiles [p0, p1] are considered in the filter.
  
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
  percentiles : float
    The lower and upper percentiles in [0,1] to use for 
    calculating the histogram.

  Returns
  -------
  sink : array
    The filtered array.
  """
  return rnk._apply_code(code.subtract_mean, code.subtract_mean_masked,
                         source=source, selem=selem, sink=sink, mask=mask, parameter_float=percentiles **kwargs);



def enhance_contrast_percentile(source, selem=None, sink=None, mask=None, percentiles=(0,1), **kwargs):
  """Enhance contrast of an image.

  This replaces each pixel by the local maximum if the pixel greyvalue is
  closer to the local maximum than the local minimum. Otherwise it is
  replaced by the local minimum.

  Only greyvalues between percentiles [p0, p1] are considered in the filter.
  
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
  percentiles : tuple of floats
    The lower and upper percentiles in [0,1] to use for 
    calculating the histogram.

  Returns
  -------
  sink : array
    The filtered array.
  """
  return rnk._apply_code(code.enhance_contrast, code.enhance_contrast_masked,
                         source=source, selem=selem, sink=sink, mask=mask, parameter_float=percentiles, **kwargs);



def percentile(source, selem=None, sink=None, mask=None, percentile=0, **kwargs):
  """Return local percentile of an image.

  Returns the value of the p0 lower percentile of the local greyvalue
  distribution.

  Only greyvalues between percentiles [p0, p1] are considered in the filter.
  
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
  percentile : float
    The percentile in [0,1] to use for 
    calculating the histogram.

  Returns
  -------
  sink : array
    The filtered array.
  """
  return rnk._apply_code(code.percentile, code.percentile_masked,
                         source=source, selem=selem, sink=sink, mask=mask, parameter_float=percentile, **kwargs);



def pop_percentile(source, selem=None, sink=None, mask=None, percentiles=(0,1), **kwargs):
  """Return the local number (population) of pixels.

  The number of pixels is defined as the number of pixels which are included
  in the structuring element and the mask.

  Only greyvalues between percentiles [p0, p1] are considered in the filter.

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
  percentiles : tuple of floats
    The lower and upper percentiles in [0,1] to use for 
    calculating the histogram.

  Returns
  -------
  sink : array
    The filtered array.
  """
  return rnk._apply_code(code.pop, code.pop_masked,
                         source=source, selem=selem, sink=sink, mask=mask, parameter_float=percentiles, **kwargs);



def sum_percentile(source, selem=None, sink=None, mask=None, percentiles=(0,1), **kwargs):
  """Return the local sum of pixels.

  Only greyvalues between percentiles [p0, p1] are considered in the filter.

  Note that the sum may overflow depending on the data type of the input
  array.

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
  percentiles : tuple of floats
    The lower and upper percentiles in [0,1] to use for 
    calculating the histogram.

  Returns
  -------
  sink : array
    The filtered array.
  """
  return rnk._apply_code(code.sum, code.sum_masked,
                        source=source, selem=selem, sink=sink, mask=mask, parameter_float=percentiles, **kwargs);



def sum_above_percentile(source, selem=None, sink=None, mask=None, percentile = 0.5, **kwargs):
  """Return the local sum of pixels.

  Only greyvalues between percentiles [p0, p1] are considered in the filter.

  Note that the sum may overflow depending on the data type of the input
  array.

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
  percentile : float
    The percentiles in [0,1] to use for 
    calculating the histogram.

  Returns
  -------
  sink : array
    The filtered array.
  """
  return rnk._apply_code(code.sum_above, code.sum_above_masked,
                        source=source, selem=selem, sink=sink, mask=mask, parameter_float=[percentile], **kwargs);




def threshold_percentile(source, selem=None, sink=None, mask=None, percentile=0, **kwargs):
  """Local threshold of an image.

  The resulting binary mask is True if the greyvalue of the center pixel is
  greater than the local mean.

  Only greyvalues between percentiles [p0, p1] are considered in the filter.

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
  percentile : float
    The  percentile in [0,1] to use for 
    calculating the histogram.

  Returns
  -------
  sink : array
    The filtered array.
  """
  return rnk._apply_code(code.threshold, code.threshold_masked,
                        source=source, selem=selem, sink=sink, mask=mask, parameter_float=[percentile], **kwargs);


    
