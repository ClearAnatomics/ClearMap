"""
Rank
====

Main 3d rank filter module
--------------------------

The package is based on the 2d skimage.filters.rank filter module.

These filters compute the local histogram at each pixel, using a sliding window
similar to the method described in [1]_. A histogram is built using a moving
window in order to limit redundant computation. The moving window follows a
snake-like path:

...------------------------\
/--------------------------/
\--------------------------...

The local histogram is updated at each pixel as the structuring element window
moves by, i.e. only those pixels entering and leaving the structuring element
update the local histogram. The histogram size is 8-bit (256 bins) for 8-bit
images and 2- to 16-bit for 16-bit images depending on the maximum value of the
image.

The filter is applied up to the image border, the neighborhood used is
adjusted accordingly. The user may provide a mask image (same size as input
image) where non zero values are the part of the image participating in the
histogram computation. By default the entire image is filtered.

This implementation outperforms grey.dilation for large structuring elements.

Input image can be 8-bit or 16-bit, for 16-bit input images, the number of
histogram bins is determined from the maximum value present in the image.

Result image is 8-/16-bit or double with respect to the input image and the
rank filter operation.


References
----------

.. [1] Huang, T. ,Yang, G. ;  Tang, G.. "A fast two-dimensional
       median filtering algorithm", IEEE Transactions on Acoustics, Speech and
       Signal Processing, Feb 1979. Volume: 27 , Issue: 1, Page(s): 13 - 18.

"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'
__note__      = "Code adpated to 3D images from skimage.filters.rank"


import warnings
import numpy as np

import ClearMap.IO.IO as io

import pyximport;
pyximport.install(setup_args={"include_dirs":np.get_include()}, 
                  reload_support=True,
                  language_level=3)

from . import RankCode as code

__all__ = ['autolevel', 'bottomhat', 'equalize', 'gradient', 'mean',
           'geometric_mean', 'subtract_mean', 'median',  'maximum', 'minimum', 'minmax', 'modal',
           'enhance_contrast', 'pop', 'threshold', 'tophat', 'noise_filter',
           'entropy', 'otsu', 'std', 'histogram']


###############################################################################
### Rank filter
###############################################################################

def autolevel(source, selem=None, sink=None, mask=None, **kwargs):
  """Auto-level image using local histogram.

  This filter locally stretches the histogram of greyvalues to cover the
  entire range of values from "white" to "black".

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

  Returns
  -------
  sink : array
      The filtered array.
  """

  return _apply_code(code.autolevel, code.autolevel_masked, 
                     source=source, selem=selem, sink=sink, mask=mask, **kwargs);
                                   

def bottomhat(source, selem=None, sink=None, mask=None, **kwargs):
  """Local bottom-hat.

  This filter computes the morphological closing of the image and then
  subtracts the result from the original image.
 
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

  Returns
  -------
  sink : array
      The filtered array.
  """
  return _apply_code(code.bottomhat, code.bottomhat_masked, 
                     source=source, selem=selem, sink=sink, mask=mask, **kwargs);


def equalize(source, selem=None, sink=None, mask=None, **kwargs):
  """Equalize image using local histogram.

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

  Returns
  -------
  sink : array
      The filtered array.
  """
  return _apply_code(code.equalize, code.equalize_masked, 
                     source=source, selem=selem, sink=sink, mask=mask, **kwargs);


def gradient(source, selem=None, sink=None, mask=None, **kwargs):
  """Return local gradient of an image (i.e. local maximum - local minimum).

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

  Returns
  -------
  sink : array
      The filtered array.
  """
  return _apply_code(code.gradient, code.gradient_masked, 
                     source=source, selem=selem, sink=sink, mask=mask, **kwargs);


def mean(source, selem=None, sink=None, mask=None, **kwargs):
  """Return local mean.

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

  Returns
  -------
  sink : array
      The filtered array.
  """
  return _apply_code(code.mean, code.mean_masked, 
                     source=source, selem=selem, sink=sink, mask=mask, **kwargs);


def geometric_mean(source, selem=None, sink=None, mask=None, **kwargs):
  """Return local geometric mean.
  
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

  Returns
  -------
  sink : array
      The filtered array.
  """
  return _apply_code(code.geometric_mean, code.geometric_mean_masked, 
                     source=source, selem=selem, sink=sink, mask=mask, **kwargs);


def subtract_mean(source, selem=None, sink=None, mask=None, **kwargs):
  """Return image subtracted from its local mean.
  
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

  Returns
  -------
  sink : array
      The filtered array.
  """
  return _apply_code(code.subtract_mean, code.subtract_mean_masked, 
                     source=source, selem=selem, sink=sink, mask=mask, **kwargs);


def median(source, selem=None, sink=None, mask=None, **kwargs):
  """Return local median.

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

  Returns
  -------
  sink : array
      The filtered array.
  """
  return _apply_code(code.median, code.median_masked, 
                     source=source, selem=selem, sink=sink, mask=mask, **kwargs);


def maximum(source, selem=None, sink=None, mask=None, **kwargs):
  """Return local maximum.
  
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

  Returns
  -------
  sink : array
      The filtered array.
  """
  return _apply_code(code.maximum, code.maximum_masked, 
                     source=source, selem=selem, sink=sink, mask=mask, **kwargs);


def minimum(source, selem=None, sink=None, mask=None, **kwargs):
  """Return local minimum.

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

  Returns
  -------
  sink : array
      The filtered array.
  """
  return _apply_code(code.minimum, code.minimum_masked,
                     source=source, selem=selem, sink=sink, mask=mask, **kwargs);


def minmax(source, selem=None, sink=None, mask=None, **kwargs):
  """Return local maximum and minimum.

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

  Returns
  -------
  sink : array
      The filtered array.
  """
  return _apply_code(code.minmax, code.minmax_masked, sink_shape_per_pixel = (2,),
                     source=source, selem=selem, sink=sink, mask=mask, **kwargs);



def modal(source, selem=None, sink=None, mask=None, **kwargs):
  """Return local mode.

  The mode is the value that appears most often in the local histogram.

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

  Returns
  -------
  sink : array
      The filtered array.
  """
  return _apply_code(code.modal, code.modal_masked,
                     source=source, selem=selem, sink=sink, mask=mask, **kwargs);


def enhance_contrast(source, selem=None, sink=None, mask=None, **kwargs):
  """Enhance contrast.

  This replaces each pixel by the local maximum if the pixel gray value is
  closer to the local maximum than the local minimum. Otherwise it is
  replaced by the local minimum.
  
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

  Returns
  -------
  sink : array
      The filtered array.
  """
  return _apply_code(code.enhance_contrast, code.enhance_contrast_masked, 
                     source=source, selem=selem, sink=sink, mask=mask, **kwargs);


def pop(source, selem=None, sink=None, mask=None, **kwargs):
  """Return the local number (population) of pixels.

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

  Returns
  -------
  sink : array
      The filtered array.
  """
  return _apply_code(code.pop, code.pop_masked, 
                     source=source, selem=selem, sink=sink, mask=mask, **kwargs);


def sum(source, selem=None, sink=None, mask=None, **kwargs):
  """Return the local sum of pixels.

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

  Returns
  -------
  sink : array
      The filtered array.
  """
  return _apply_code(code.sum, code.sum_masked, 
                     source=source, selem=selem, sink=sink, mask=mask, **kwargs);


def threshold(source, selem=None, sink=None, mask=None, **kwargs):
  """Local threshold.

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

  Returns
  -------
  sink : array
      The filtered array.
  """
  return _apply_code(code.threshold, code.threshold_masked, 
                     source=source, selem=selem, sink=sink, mask=mask, **kwargs);


def tophat(source, selem=None, sink=None, mask=None, **kwargs):
  """Local top-hat.

  This filter computes the morphological opening of the image and then
  subtracts the result from the original image.

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

  Returns
  -------
  sink : array
      The filtered array.
  """
  return _apply_code(code.tophat, code.tophat_masked, 
                     source=source, selem=selem, sink=sink, mask=mask, **kwargs);


def noise_filter(source, selem=None, sink=None, mask=None, **kwargs):
  """Noise feature.
  
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

  Returns
  -------
  sink : array
      The filtered array.
  """
  # ensure that the central pixel in the structuring element is empty
  if selem is None:
    selem_cpy = np.ones((3,3,3), dtype = bool);
  else:
    selem_cpy = selem.copy();
    
  center = tuple(int(s / 2)  for s in selem.shape)
  selem_cpy[center] = 0;
  
  return _apply_code(code.noise_filter, code.noise_filter_masked, 
                     source=source, selem=selem_cpy, sink=sink, mask=mask, **kwargs);


def entropy(source, selem=None, sink=None, mask=None, **kwargs):
  """Local entropy.

  The entropy is computed using base 2 logarithm i.e. the filter returns the
  minimum number of bits needed to encode the local greylevel
  distribution.

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

  Returns
  -------
  sink : array
      The filtered array.
  """
  return _apply_code(code.entropy, code.entropy_masked, sink_dtype = float,
                     source=source, selem=selem, sink=sink, mask=mask, **kwargs);


def otsu(source, selem=None, sink=None, mask=None, **kwargs):
  """Local Otsu's threshold value for each pixel.

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

  Returns
  -------
  sink : array
      The filtered array.
  """
  return _apply_code(code.otsu, code.otsu_masked, 
                     source=source, selem=selem, sink=sink, mask=mask, **kwargs);



def std(source, selem=None, sink=None, mask=None, **kwargs):
  """Local standard deviation.

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

  Returns
  -------
  sink : array
      The filtered array.
  """
  return _apply_code(code.std, code.std_masked, sink_dtype=float,
                     source=source, selem=selem, sink=sink, mask=mask, **kwargs);


def histogram(source, selem=None, sink=None, mask=None, max_bin=None):
    """Normalized sliding window histogram

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

    Returns
    -------
    sink : array
      Array of the source shape pluse on extra dimension for the histogram
      at each pixel.
    """

    if max_bin is None:
      max_bin = io.max_value(source.dtype);
    if max_bin >= 2**16:
      raise ValueError('The histograms are to large for this code to be efficient!');
    parameter_index = [max_bin];

    return _apply_code(code.histogram, code.histogram_masked, max_bin=max_bin,
                       sink_shape_per_pixel=(max_bin,), sink_dtype = float,
                       source=source, selem=selem, sink=sink, mask=mask, parameter_index=parameter_index);
                    

###############################################################################
### Helper
###############################################################################

import ClearMap.ImageProcessing.Filter.StructureElement as se

def _initialize_selem(selem, ndim):
  if selem is None:
    selem = 3;
  if isinstance(selem, int):
    selem = (selem,) * ndim;
  if isinstance(selem, tuple):
    if len(selem) == 2 and isinstance(selem[0], str):
      form, shape = selem;
      selem = se.structure_element(form=form, shape=shape);
    else:
      selem = np.ones(selem, dtype = bool);          
  selem = np.asarray(selem > 0, dtype = 'uint8');
  return selem;                    

def _apply_code(function, function_mask, source, selem = None, 
                sink = None, sink_dtype = None, sink_shape_per_pixel = None, 
                mask = None, max_bin = None,  
                parameter_index = None, parameter_float = None):
  """Helper to apply code."""
  selem = _initialize_selem(selem, source.ndim)
  
  if source.ndim > 3:
    raise ValueError('Source dimension %d not supported!' % source.ndim);
  if source.ndim != 3:
    source = np.asarray(source);
    shape_remove = [d for d in range(source.ndim, 3)];
    source = source.view();
    source.shape = source.shape + (1,) * (3 - source.ndim);
    if mask is not None:
      mask = np.asarray(mask);
      mask = mask.view();
      mask.shape = mask.shape +  (1,) * (3 - mask.ndim);
    selem = selem.view();
    selem.shape = selem.shape + (1,) * (3 - selem.ndim);
  else:
    shape_remove = [];
  
  if source.dtype not in (np.uint8, np.uint16, np.int, np.int16, np.int32, np.int64, np.uint32, np.uint64, np.bool):
    raise ValueError('The rank filter requires a source of integer type, found %r!' % source.dtype);
  
  if mask is not None:
    if mask.dtype == bool:
      mask = mask.view('uint8');
    if mask.shape != source.shape:
      raise ValueError('Source shape %r and mask shape %r do not match!' % (source.shape, mask.shape))

  if source is sink:
    raise ValueError("Cannot perform rank filter in place!")

  if sink_shape_per_pixel is None:
    shape_per_pixel = (1,);
    shape_remove += [3];
  else:
    shape_per_pixel = sink_shape_per_pixel;
  
  if sink is None:
    if sink_dtype is None:
      sink_dtype = source.dtype
    sink = np.zeros(source.shape + shape_per_pixel, dtype=sink_dtype)
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
  
  if max_bin is None:
    max_bin = io.max_value(source.dtype);
  if max_bin >= 2**16:
    raise ValueError('The histograms are to large for this code to be efficient!');

  bitdepth = int(np.log2(max_bin))
  if bitdepth > 12:
    warnings.warn("Bitdepth of %d may result in bad rank filter performance." % bitdepth);

  if parameter_index is None:
    parameter_index = np.zeros(0, dtype = int);
  parameter_index = np.asarray([parameter_index], dtype = int).flatten();

  if parameter_float is None:
    parameter_float = np.zeros(0, dtype = float);
  parameter_float = np.asarray([parameter_float], dtype = float).flatten();           
  
  #print(source.__class__, source.dtype, s.__class__, s.dtype, selem.dtype, mask.dtype, max_bin)                           
                              
  if mask is None:
    function(source=source, selem=selem, sink=s, max_bin=max_bin, p=parameter_index, q=parameter_float);
  else:
    function_mask(source=source, selem=selem, mask=mask, sink=s, max_bin=max_bin, p=parameter_index, q=parameter_float);
  
  if len(shape_remove) > 0:
    shape = tuple(s for d,s in enumerate(sink.shape) if d not in shape_remove);
    sink = sink.reshape(shape);
  
  return sink


###############################################################################
### Tests
###############################################################################


def _test():
  import numpy as np
  import  ClearMap.ImageProcessing.Filter.Rank.Rank as rnk
  reload(rnk)

  import ClearMap.Tests.Files as tfs;
  data = np.asarray(tfs.source('vr')[:100,:100,50], dtype = float);
  data = np.asarray(255 * data / data.max(), dtype = 'uint8');
  
  funcs = rnk.__all__[:-1]; 
  n = len(funcs);
  m = int(np.ceil(np.sqrt(n)));
  p = int(np.ceil(float(n)/m));
  
  import matplotlib.pyplot as plt;
  plt.figure(1); plt.clf();
  ax = plt.subplot(m,p,1);
  plt.imshow(data);
  plt.title('original')
  
  for i,f in enumerate(funcs):
    func = eval('rnk.' + f);
    res = func(data, selem=np.ones((5,5,5), dtype = bool));
    
    plt.subplot(m,p,i+2, sharex=ax, sharey=ax);
    plt.imshow(res);
    plt.title(f);
    
  plt.tight_layout()  


  #masked version
  mask = np.zeros(data.shape, dtype = bool);
  mask[30:60, 30:60] = True;

  import matplotlib.pyplot as plt;
  plt.figure(2); plt.clf();
  ax = plt.subplot(m,p,1);
  plt.imshow(data);
  plt.title('original')
  
  for i,f in enumerate(funcs):
    func = eval('rnk.' + f);
    res = func(data, selem=np.ones((5,5), dtype = bool), mask=mask);
    
    plt.subplot(m,p,i+2, sharex=ax, sharey=ax);
    plt.imshow(res);
    plt.title(f);
    
  plt.tight_layout()  