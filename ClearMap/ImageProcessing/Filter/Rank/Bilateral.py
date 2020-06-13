"""Approximate bilateral rank filter for local (custom kernel) mean.

The local histogram is computed using a sliding window similar to the method
described in [Huang]_.

The pixel neighborhood is defined by:

* the given structuring element
* an interval [g-s0, g+s1] in greylevel around g the processed pixel greylevel

The kernel is flat (i.e. each pixel belonging to the neighborhood contributes
equally).

Result image is 8-/16-bit or double with respect to the input image and the
rank filter operation.

References
----------

.. [Huang] Huang, T. ,Yang, G. ;  Tang, G.. "A fast two-dimensional
       median filtering algorithm", IEEE Transactions on Acoustics, Speech and
       Signal Processing, Feb 1979. Volume: 27 , Issue: 1, Page(s): 13 - 18.

"""
__note__ = "Code adpated to 3D images from skimage.filters.rank by Christoph Kirst"


import numpy as np

from . import Rank as rnk

import pyximport;
pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)

from . import BilateralCode as code


__all__ = ['mean_bilateral', 'pop_bilateral', 'sum_bilateral', 'sum_relative_bilateral', 'mean_scale_bilateral']


 
###############################################################################
### Bilateral filter
###############################################################################

def mean_bilateral(source, selem=None, sink=None, mask=None, s0=10, s1=10, **kwargs):
  """Apply a flat kernel bilateral filter.

  This is an edge-preserving and noise reducing denoising filter. It averages
  pixels based on their spatial closeness and radiometric similarity.

  Spatial closeness is measured by considering only the local pixel
  neighborhood given by a structuring element.

  Radiometric similarity is defined by the greylevel interval [g-s0, g+s1]
  where g is the current pixel greylevel.

  Only pixels belonging to the structuring element and having a graylevel
  inside this interval are averaged.

  Arguments
  ---------
  source : array
    Input array.
  selem : array
    Structuring element, if None use a cube of size 3.
  sink : array
    Output array, if None, a new array is allocated.
  mask : array
    Optional mask, if None, the complete source is used.
    Pixels in the mask are zero in the output.
  s0,s1 : int
    The lower and upper widths around the center value
    for the bilateral window. 

  Returns
  -------
  sink : array
      The filtered array.
  """
  return rnk._apply_code(code.mean, code.mean_masked, sink_dtype = float,
                         source=source, selem=selem, sink=sink, mask=mask, parameter_index=[s0,s1], **kwargs);


def pop_bilateral(source, selem=None, sink=None, mask=None, s0=10, s1=10, **kwargs):
  """Return the local number (population) of pixels.
  
  The number of pixels is defined as the number of pixels which are included
  in the structuring element and the mask. Additionally pixels must have a
  graylevel inside the interval [g-s0, g+s1] where g is the gray value of the
  center pixel.

  Arguments
  ---------
  source : array
    Input array.
  selem : array
    Structuring element, if None use a cube of size 3.
  sink : array
    Output array, if None, a new array is allocated.
  mask : array
    Optional mask, if None, the complete source is used.
    Pixels in the mask are zero in the output.
  s0,s1 : int
    The lower and upper widths around the center value
    for the bilateral window. 

  Returns
  -------
  sink : array
      The filtered array.
  """
  return rnk._apply_code(code.pop, code.pop_masked, sink_dtype = float,
                         source=source, selem=selem, sink=sink, mask=mask, parameter_index=[s0,s1], **kwargs);

  

def sum_bilateral(source, selem=None, sink=None, mask=None, s0=10, s1=10, **kwargs):
  """Apply a flat kernel bilateral filter.

  This is an edge-preserving and noise reducing denoising filter. It sums
  pixels based on their spatial closeness and radiometric similarity.

  Spatial closeness is measured by considering only the local pixel
  neighborhood given by a structuring element (selem).

  Radiometric similarity is defined by the greylevel interval [g-s0, g+s1]
  where g is the current pixel greylevel.

  Only pixels belonging to the structuring element AND having a greylevel
  inside this interval are summed.

  Note that the sum may overflow depending on the data type of the input
  array.

  Arguments
  ---------
  source : array
    Input array.
  selem : array
    Structuring element, if None use a cube of size 3.
  sink : array
    Output array, if None, a new array is allocated.
  mask : array
    Optional mask, if None, the complete source is used.
    Pixels in the mask are zero in the output.
  s0,s1 : int
    The lower and upper widths around the center value
    for the bilateral window. 

  Returns
  -------
  sink : array
      The filtered array.
  """
  return rnk._apply_code(code.sum, code.sum_masked, sink_dtype = float,
                         source=source, selem=selem, sink=sink, mask=mask, parameter_index=[s0,s1], **kwargs);



def sum_relative_bilateral(source, selem=None, sink=None, mask=None, s0=0.5, s1=2.0, **kwargs):
  """Apply a flat kernel bilateral filter.

  This is an edge-preserving and noise reducing denoising filter. It sums
  pixels based on their spatial closeness and radiometric similarity.

  Spatial closeness is measured by considering only the local pixel
  neighborhood given by a structuring element (selem).

  Radiometric similarity is defined by the greylevel interval [g-s0, g+s1]
  where g is the current pixel greylevel.

  Only pixels belonging to the structuring element AND having a greylevel
  inside this interval are summed.

  Note that the sum may overflow depending on the data type of the input
  array.

  Arguments
  ---------
  source : array
    Input array.
  selem : array
    Structuring element, if None use a cube of size 3.
  sink : array
    Output array, if None, a new array is allocated.
  mask : array
    Optional mask, if None, the complete source is used.
    Pixels in the mask are zero in the output.
  s0,s1 : int
    The lower and upper widths around the center value
    for the bilateral window. 

  Returns
  -------
  sink : array
    The filtered array.
  """
  return rnk._apply_code(code.sum, code.sum_masked, sink_dtype = float,
                         source=source, selem=selem, sink=sink, mask=mask, parameter_float=[s0,s1], **kwargs);


def mean_scale_bilateral(source, selem=None, sink=None, mask=None, s0=10, s1=10, p0=0, p1=0, **kwargs):
  """Mean pixel value using a bilateral window size that scales with local pixel intensity.
  
  The filter linearly scales the biateral window size as [g-s0-p0*g, g+s1+p1*g].

  Arguments
  ---------
  source : array
    Input array.
  selem : array
    Structuring element, if None use a cube of size 3.
  sink : array
    Output array, if None, a new array is allocated.
  mask : array
    Optional mask, if None, the complete source is used.
    Pixels in the mask are zero in the output.
  s0,s1 : int
    The lower and upper widths around the center value
    for the bilateral window. 
  p0,p1 : float
    The lower and upper scale values for the bilateral window.

  Returns
  -------
  sink : array
    The filtered array.
  """
  return rnk._apply_code(code.mean_scale, code.mean_scale_masked, sink_dtype = float,
                         source=source, selem=selem, sink=sink, mask=mask, 
                         parameter_index=[s0,s1], parameter_float=[p0,p1], **kwargs);


