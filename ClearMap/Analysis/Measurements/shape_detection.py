# -*- coding: utf-8 -*-
"""
ShapeDetection
==============

Module with routines for shape and size detection of objects such as cells.

Note
----
The shape detection is based on a seeded and masked watershed. The module is 
based on the ndimage library. For faster implementation of intensity 
and radial measurements see the modules listed below.

See also
--------
:mod:`ClearMap.Analysis.Measurements.MeasureExpression` and 
:mod:`ClearMap.Analysis.Measurements.MeasureRadius` 
"""
__author__ = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__ = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__ = 'https://idisco.info'
__download__ = 'https://github.com/ClearAnatomics/ClearMap'


import numpy as np

import skimage.morphology
import scipy.ndimage.measurements

import ClearMap.IO.IO as io

import ClearMap.Analysis.Measurements.Voxelization as vox

import ClearMap.Utils.Timer as tmr
import ClearMap.Utils.HierarchicalDict as hdict
from ClearMap.Utils.exceptions import ClearMapValueError

##############################################################################
# Cell shape detection
##############################################################################

# version for python >=3.11
# def labeled_pixels_from_centers(centers,weights,shape):
#     """label the pixels specify by the centers with the weights.

#     Parameters
#     ----------
#     centers : np.ndarray
#         (n_points,dim) array with points coords, its dtype is the one of weights
#     weights : np.ndarray
#         (n_points,) shaped array
#     shape : tuple
#         the shape of the output array.
#     """

#     if centers.shape[0]!=weights.shape[0]:
#         raise ValueError("The number of weights must equal the number of points.")
    
#     if len(shape)!=centers.shape[1]:
#         raise ValueError("Received shape, points with points.shape[1] != len(shape) ")
    
#     B = np.zeros(shape,dtype = weights.dtype)
#     transposed = centers.transpose()
#     B[*(transposed)]=weights
#     return B


def labeled_pixels_from_centers(centers,weights,shape):
    """label the pixels specify by the centers with the weights.

    This is restricted to 3d shapes for now, a code that will work with python >=3.11
    is commented out for now

    Parameters
    ----------
    centers : np.ndarray
        (n_points,dim) array with points coords, its dtype is the one of weights
    weights : np.ndarray
        (n_points,) shaped array
    shape : tuple
        the shape of the output array.
    """

    if centers.shape[0]!=weights.shape[0]:
        raise ValueError("The number of weights must equal the number of points.")
    
    if len(shape)!=centers.shape[1]:
        raise ValueError("Received shape, points with points.shape[1] != len(shape) ")
    
    B = np.zeros(shape,dtype = weights.dtype)
    xs,ys,zs = centers.transpose()
    B[xs,ys,zs]=weights
    return B

def detect_shape(source, seeds, threshold=None, verbose=False, processes=None, as_binary_mask=False, return_sizes=False, seeds_as_labels = False, watershed_line=True):
    """Detect object shapes by generating a labeled image from seeds.

    Optionally, the output is replaced by to a mere binary mask and the
    distinct shapes sizes are also returned.

    Parameters
    ----------
    source : array, str or Source
        Source image.
    seeds : array, str or Source
        Cell centers as point coordinates if seeds_as_labels is False. See below otherwise.
    threshold : float or None
        Threshold to determine mask for watershed, pixel below this are
        treated as background. If None, the seeds are expanded indefinitely.
    verbose :bool
        If True, print progress info.
    as_binary_mask : bool, optional
        If the first output is to be the mask of all shapes, by default False.
    return_sizes : bool, optional
        If the sizes of the various shapes are to be returned too, by default False.
    seeds_as_labels: bool, optional
        Defaults to False. If True the input seeds is considered to be a labeled image w/
        the initial basins.
    watershed_line: bool, optional
        Defaults to True. If True the watershedding is made with a line of background pixels
        inbetween the labeled regions. The value True is forced if as_binary_mask is True.
        
    Returns
    -------
    shapes : array
        Labeled image, where each label indicates an object. Optionally replaced
        by shapes>0, see above.

    sizes : (optional) the sizes of the shapes, in the same order are the seeds.
    """
    # if we as_binary_mask=True, there is only one reasonable choice for watershed_line
    watershed_line = as_binary_mask or watershed_line
    if verbose:
        timer = tmr.Timer()
        hdict.pprint(head='Shape detection', threshold=threshold)
  
    source = io.as_source(source).array
    seeds = io.as_source(seeds)
    mask = None if threshold is None else source > threshold
    if seeds_as_labels:
        peaks = seeds
    else:
        peaks = labeled_pixels_from_centers(seeds,np.arange(1, seeds.shape[0]+1), source.shape)

    # We check that source has no 0 value otherwise the map source -> -source is not necessarily decreasing, eg for source.dtype=uint16.
    if np.any(source == 0) and np.issubdtype(source.dtype,np.unsignedinteger):
        print('For sanity, we shift the source intensity by 1 prior to taking its opposite.')
        if source.max() < np.ma.minimum_fill_value(source):
            source+=1

        else:
            raise ClearMapValueError('An uint array with 0 values will lead to inconsistent results, consider a histogram transform or dtype conversion.')

    try:
        shapes = skimage.morphology.watershed(-source, peaks, mask=mask, watershed_line=watershed_line)
    except AttributeError:
        shapes = skimage.segmentation.watershed(-source, peaks, mask=mask, watershed_line=watershed_line)

    if np.unique(shapes).size != np.unique((peaks if mask is None else peaks*mask)).size:
        raise RuntimeError(f'watersheding yields unexpected results: the seed number was {np.unique(peaks*mask).size-1}'
                           + f'and the number of labeled region in output was {np.unique(shapes).size} counting the zero labeled region'
                           + 'However,' + ( 'there was no zero labeled pixel' if np.count_nonzero(shapes==0)==0  else 'there was some zero labeled pixel') )

    if verbose:
        timer.print_elapsed_time('Shape detection')

    if return_sizes:
        max_label = shapes.max()
        sizes = find_size(shapes, max_label=max_label)
    
        if as_binary_mask:
            return (shapes>0),sizes
        else:
            return shapes,sizes
    else:
        if as_binary_mask:
            return shapes>0
        else:
            return shapes

def find_size(label, max_label=None, verbose=False):
    """
    Find size given object shapes as a labled image

    Arguments
    ---------
    label : array, str or Source
        Labeled image in which each object has its own label.
    max_label : int or None
        Maximal label to include, if None use all label.
    verbose : bool
        Print progress info.

    Returns
    -------
    sizes : array
        Measured intensities
    """

    if verbose:
        timer = tmr.Timer()
        hdict.pprint(head='Size detection:', max_label=max_label)

    label = io.as_source(label)

    if max_label is None:
        max_label = int(label.max())

    sizes = scipy.ndimage.measurements.sum(np.ones(label.shape, dtype=bool),
                                           labels=label, index=np.arange(1, max_label + 1))

    if verbose:
        timer.print_elapsed_time(head='Size detection')

    return sizes


def find_intensity(source, label, max_label=None, method='sum', verbose=False):
    """
    Find integrated intensity given object shapes as labeled image.

    Arguments
    ---------
    source : array, str, or Source
      Source to measure intensities from.
    label : array, str, or Source
      Labeled image with a separate label for each object.
    max_label : int or None
      Maximal label to include. If None, use all.
    method : {'sum', 'mean', 'max', 'min'}
      Method to use to measure the intensities in each object's area.
    verbose : bool
      If True, print progress information.

    Returns
    -------
    intensities : array
      Measured intensities.
    """
    method = method.lower()
    if verbose:
        timer = tmr.Timer()
        hdict.pprint(head='Intensity detection:', max_label=max_label, method=method)

    source = io.as_source(source).array
    label = io.as_source(label)

    if max_label is None:
        max_label = label.max()

    measure_functions = {
      'sum': scipy.ndimage.measurements.sum,
      'mean': scipy.ndimage.measurements.mean,
      'max': scipy.ndimage.measurements.maximum,
      'min': scipy.ndimage.measurements.minimum
    }

    try:
        intensities = measure_functions[method](source, labels=label, index=np.arange(1, max_label + 1))
    except KeyError:
        raise RuntimeError(f'Unknown method {method}, expected one of {measure_functions.keys()}')

    if verbose:
        timer.print_elapsed_time(head='Intensity detection')

    return intensities
  
