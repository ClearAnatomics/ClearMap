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
__license__   = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'
__note__      = "Code adpated to 3D images from skimage.filters.rank"


import numbers
import warnings
import numpy as np

import pyximport

from ClearMap.IO import dtypes
import ClearMap.ImageProcessing.Filter.StructureElement as se
import ClearMap.Utils.array_checks as ac

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
                       source=source, selem=selem, sink=sink, mask=mask, **kwargs)


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
                       source=source, selem=selem, sink=sink, mask=mask, **kwargs)


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
                       source=source, selem=selem, sink=sink, mask=mask, **kwargs)


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
                       source=source, selem=selem, sink=sink, mask=mask, **kwargs)


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
                       source=source, selem=selem, sink=sink, mask=mask, **kwargs)


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
                       source=source, selem=selem, sink=sink, mask=mask, **kwargs)


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
                       source=source, selem=selem, sink=sink, mask=mask, **kwargs)


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
                       source=source, selem=selem, sink=sink, mask=mask, **kwargs)


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
                       source=source, selem=selem, sink=sink, mask=mask, **kwargs)


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
                       source=source, selem=selem, sink=sink, mask=mask, **kwargs)


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
                       source=source, selem=selem, sink=sink, mask=mask, **kwargs)


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
                       source=source, selem=selem, sink=sink, mask=mask, **kwargs)


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
                       source=source, selem=selem, sink=sink, mask=mask, **kwargs)


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
                       source=source, selem=selem, sink=sink, mask=mask, **kwargs)


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
                       source=source, selem=selem, sink=sink, mask=mask, **kwargs)


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
                       source=source, selem=selem, sink=sink, mask=mask, **kwargs)


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
                       source=source, selem=selem, sink=sink, mask=mask, **kwargs)


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
    selem = _initialize_selem(selem, np.ndim(source))  # always a new array, safe to modify
    center = tuple(s // 2 for s in selem.shape)
    selem[center] = 0

    return _apply_code(code.noise_filter, code.noise_filter_masked,
                       source=source, selem=selem, sink=sink, mask=mask, **kwargs)


def entropy(source, selem=None, sink=None, mask=None, **kwargs):
    """Local entropy.

    The entropy is computed using base 2 logarithm i.e. the filter returns the
    minimum number of bits needed to encode the local grey_level
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
                       source=source, selem=selem, sink=sink, mask=mask, **kwargs)


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
                       source=source, selem=selem, sink=sink, mask=mask, **kwargs)


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
                       source=source, selem=selem, sink=sink, mask=mask, **kwargs)


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

    max_bin = _resolve_max_bin(source.dtype, max_bin)
    parameter_index = [max_bin]

    return _apply_code(code.histogram, code.histogram_masked, max_bin=max_bin,
                       sink_shape_per_pixel=(max_bin,), sink_dtype = float,
                       source=source, selem=selem, sink=sink, mask=mask, parameter_index=parameter_index)


###############################################################################
### Helper
###############################################################################

# The Cython code (RankCoreCode.pxd) is only compiled for these dtypes.
# A bool array is accepted for sinks and masks and viewed as uint8.
_SOURCE_DTYPES = (np.uint8, np.uint16, np.int64)
_SINK_DTYPES = (np.uint8, np.uint16, np.int64, np.float64)


def _resolve_max_bin(source_dtype, max_bin=None):
    """Number of histogram bins used by the Cython code (``max_bin``).

    Arguments
    ---------
    source_dtype : dtype
        dtype of the source.
    max_bin : int or None
        Explicit number of bins, if None the maximal value of the source dtype.

    Returns
    -------
    max_bin : int
        The number of bins.
    """
    if max_bin is None:
        max_bin = dtypes.max_value(source_dtype)
    max_bin = int(max_bin)
    if max_bin >= 2**16:
        raise ValueError('The histograms are to large for this code to be efficient!')
    return max_bin


def _initialize_selem(selem, ndim):
    """Convert a structuring element specification to the array expected by the Cython code.

    Arguments
    ---------
    selem : None, int, sequence of int, (str, shape) or array-like
        Structuring element specification:

        * None: cube of size 3 along each of the ``ndim`` axes.
        * int: cube of that size along each of the ``ndim`` axes.
        * tuple or list of ints: cube (rectangle) of that shape.
        * (form, shape) with ``form`` a str: element built by
          :func:`ClearMap.ImageProcessing.Filter.StructureElement.structure_element`,
          e.g. ``('disk', 5)`` or ``('sphere', (5, 5, 3))``. A scalar shape is repeated ``ndim`` times.
        * array-like (anything else, e.g. a numpy array): used as the element itself,
          every non-zero entry belongs to the element.
    ndim : int
        Number of dimensions of the image the element will be applied to,
        used for the specifications that do not define the number of axes themselves.

    Returns
    -------
    selem : array of uint8
        New array (safe to modify) with 1 where the element is set and 0 elsewhere.

    Note
    ----
    The rank filters only use the element as a set (which neighbours take part in the local
    histogram), weights, such as those of a 'sphere' element, are discarded by the ``> 0`` test.

    The dtype is dictated by the Cython code, which declares the element as
    ``const uint8_t[:, :, :]``. This is deliberately not ``char``: numpy ``uint8`` arrays are
    rejected by ``const char`` memoryviews.
    """
    if selem is None:
        selem = 3

    if isinstance(selem, numbers.Integral):
        selem = np.ones((int(selem),) * ndim, dtype=bool)
    elif isinstance(selem, (tuple, list)) and len(selem) == 2 and isinstance(selem[0], str):
        form, shape = selem
        # scalar shapes need the dimension, sequences define their own
        selem = se.structure_element(shape=shape, form=form, ndim=ndim if np.ndim(shape) == 0 else None)
    elif isinstance(selem, (tuple, list)) and len(selem) > 0 and all(isinstance(s, numbers.Integral) for s in selem):
        selem = np.ones(tuple(int(s) for s in selem), dtype=bool)
    else:  # array-like used as the element itself
        selem = np.asarray(selem)
    # WARNING: uint8 cast here is essential for cython code
    return np.array(np.asarray(selem) > 0, dtype=np.uint8)  # always a fresh writable array


def _prepare_mask(mask, source_shape, ndim):
    """Validate the mask against the (3D padded) source shape and return it as a 3D uint8 array.

    Arguments
    ---------
    mask : array-like
        The mask. Non zero values take part in the computation.
    source_shape : tuple
        Shape of the source after padding to 3D.
    ndim : int
        Number of dimensions of the source before padding.

    Note
    ----
    Trailing singleton axes of the mask are tolerated (e.g. a ``(x, y, 1)`` mask for a ``(x, y)``
    source) with a warning.
    """
    mask = np.asarray(mask)
    mask_ndim = mask.ndim
    mask = ac.pad_to_ndim(mask)
    if mask.shape != source_shape:
        raise ValueError(f'Source shape {source_shape!r} and mask shape {mask.shape!r} do not match!')
    if mask_ndim != ndim:
        warnings.warn(f'The mask has {mask_ndim:d} dimensions but the source has {ndim:d}, '
                      f'the extra singleton axes are ignored.', stacklevel=4)
    return ac.as_uint8_flags(mask, name='mask')


def _prepare_sink(sink, sink_dtype, source, shape_per_pixel):
    """Allocate the sink or validate and reshape the one provided (without ever copying it).

    Arguments
    ---------
    sink : array or None
        The sink provided by the caller.
    sink_dtype : dtype or None
        dtype of the sink to allocate, defaults to that of the source.
    source : array
        The 3D source.
    shape_per_pixel : tuple
        Shape of the output at each pixel.

    Returns
    -------
    sink : array
        Array of shape ``source.shape + shape_per_pixel``.
    """
    expected = source.shape + shape_per_pixel

    if sink is None:
        return np.zeros(expected, dtype=source.dtype if sink_dtype is None else sink_dtype)

    ac.check_dtype(sink, _SINK_DTYPES, name='sink', allow_bool=True)
    if sink.shape == expected:
        return sink
    if shape_per_pixel != (1,):
        raise ValueError(f'The sink of shape {sink.shape!r} does not have expected shape {expected!r}!')

    reshaped = sink.reshape(expected)
    if not np.may_share_memory(sink, reshaped):  # would silently write the result in a copy
        raise ValueError(f'The sink of shape {sink.shape!r} cannot be reshaped to {expected!r} without copying!')
    return reshaped


def _apply_code(function, function_mask, source, selem=None,
                sink=None, sink_dtype=None, sink_shape_per_pixel=None,
                mask=None, max_bin=None,
                parameter_index=None, parameter_float=None):
    """Prepare the arguments and call the compiled rank filter.

    This is the common back end of all the rank filters. It normalises the inputs to what the Cython
    code expects (3D arrays, uint8 structuring element and mask), calls the masked or unmasked
    version and restores the dimensions of the result.

    Arguments
    ---------
    function : callable
        Compiled filter (from RankCode or PercentileCode) used when there is no mask.
    function_mask : callable
        Compiled filter used when a mask is given.
    source : array
        Input array with 1 to 3 dimensions and dtype uint8, uint16 or int64 (the types the Cython code is
        compiled for). Lower dimensional sources are processed as 3D arrays with trailing
        singleton axes.
    selem : None, int, sequence, (str, shape) or array
        Structuring element, see :func:`_initialize_selem`. Its dimension may be smaller than 3
        (it is padded as the source) but not larger.
    sink : array or None
        Output array of shape ``source.shape + sink_shape_per_pixel`` (or just ``source.shape`` when
        there is a single value per pixel). Cannot be the source. Its dtype must be one of
        uint8, uint16, int64, float64 or bool. If None, a new array is allocated.
    sink_dtype : dtype or None
        dtype of the sink when it is allocated, the dtype of the source if None.
    sink_shape_per_pixel : tuple or None
        Shape of the output for each pixel, e.g. ``(2,)`` for min and max or ``(max_bin,)`` for a
        histogram. None for a single value per pixel, that axis is then absent from the result.
    mask : array or None
        Optional mask with the shape of the source. Non zero values take part in the computation.
    max_bin : int or None
        Number of histogram bins, the maximal value of the source dtype if None.
    parameter_index : int, sequence of int or None
        Integer parameters passed to the filter (``p`` in the Cython code).
    parameter_float : float, sequence of float or None
        Floating point parameters passed to the filter (``q`` in the Cython code).

    Returns
    -------
    sink : array
        The filtered array, of the same dimension as the source (plus the per pixel axes if any).

    Note
    ----
    ``parameter_index`` and ``parameter_float`` are always passed as fresh writable copies because
    some kernels use them as scratch space.
    """
    if source is sink:
        raise ValueError('Cannot perform rank filter in place!')

    source = np.asarray(source)
    ndim = source.ndim
    if ndim > 3:
        raise ValueError(f'Source dimension {ndim:d} not supported!')
    ac.check_dtype(source, _SOURCE_DTYPES, name='source')

    selem = _initialize_selem(selem, ndim)
    if selem.ndim > 3:
        raise ValueError(f'Structuring element dimension {selem.ndim:d} not supported!')

    source = ac.pad_to_ndim(source)
    selem = ac.pad_to_ndim(selem)
    if mask is not None:
        mask = _prepare_mask(mask, source.shape, ndim)

    # axes of the result to drop again: the padding of the source and the per pixel axis
    axes_to_remove = list(range(ndim, 3))
    if sink_shape_per_pixel is None:
        shape_per_pixel = (1,)
        axes_to_remove.append(3)
    else:
        shape_per_pixel = tuple(sink_shape_per_pixel)

    sink = _prepare_sink(sink, sink_dtype, source, shape_per_pixel)
    sink_view = sink.view('uint8') if sink.dtype == bool else sink

    max_bin = _resolve_max_bin(source.dtype, max_bin)

    bit_depth = int(np.log2(max_bin))
    if bit_depth > 12:
        warnings.warn(f"Bit depth of {bit_depth:d} may result in bad rank filter performance.")

    parameter_index = ac.scratch_parameters(parameter_index, np.intp)
    parameter_float = ac.scratch_parameters(parameter_float, float)

    if mask is None:
        function(source=source, selem=selem, sink=sink_view, max_bin=max_bin, p=parameter_index, q=parameter_float)
    else:
        function_mask(source=source, selem=selem, mask=mask, sink=sink_view, max_bin=max_bin, p=parameter_index, q=parameter_float)

    return sink.squeeze(axis=tuple(axes_to_remove)) if axes_to_remove else sink


###############################################################################
### Tests
###############################################################################


def _test():
    import numpy as np
    import  ClearMap.ImageProcessing.Filter.Rank.Rank as rnk
    from importlib import reload
    reload(rnk)

    import ClearMap.Tests.Files as tfs
    data = np.asarray(tfs.source('vr')[:100,:100,50], dtype = float)
    data = np.asarray(255 * data / data.max(), dtype = 'uint8')

    funcs = rnk.__all__[:-1]
    n = len(funcs)
    m = int(np.ceil(np.sqrt(n)))
    p = int(np.ceil(float(n)/m))

    import matplotlib.pyplot as plt
    plt.figure(1); plt.clf()
    ax = plt.subplot(m,p,1)
    plt.imshow(data)
    plt.title('original')

    for i, f in enumerate(funcs):
        func = eval('rnk.' + f)
        res = func(data, selem=np.ones((5,5,5), dtype = bool))

        plt.subplot(m,p,i+2, sharex=ax, sharey=ax)
        plt.imshow(res)
        plt.title(f)

    plt.tight_layout()


    #masked version
    mask = np.zeros(data.shape, dtype = bool)
    mask[30:60, 30:60] = True

    import matplotlib.pyplot as plt
    plt.figure(2); plt.clf()
    ax = plt.subplot(m,p,1)
    plt.imshow(data)
    plt.title('original')

    for i, f in enumerate(funcs):
        func = eval('rnk.' + f)
        res = func(data, selem=np.ones((5,5), dtype = bool), mask=mask)

        plt.subplot(m,p,i+2, sharex=ax, sharey=ax)
        plt.imshow(res)
        plt.title(f)

    plt.tight_layout()
