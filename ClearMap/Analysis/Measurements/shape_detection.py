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
__download__ = 'https://www.github.com/ChristophKirst/ClearMap2'


import numpy as np

import skimage.morphology
import scipy.ndimage.measurements

import ClearMap.IO.IO as io

import ClearMap.Analysis.Measurements.Voxelization as vox

import ClearMap.Utils.Timer as tmr
import ClearMap.Utils.HierarchicalDict as hdict


##############################################################################
# Cell shape detection
##############################################################################


def detect_shape(source, seeds, threshold=None, verbose=False, processes=None):
    """
    Detect object shapes by generating a labeled image from seeds.
  
    Arguments
    ---------
    source : array, str or Source
        Source image.
    seeds : array, str or Source
        Cell centers as point coordinates.
    threshold : float or None
        Threshold to determine mask for watershed, pixel below this are
        treated as background. If None, the seeds are expanded indefinitely.
    verbose :bool
        If True, print progress info.
  
    Returns
    -------
    shapes : array
        Labeled image, where each label indicates an object.
    """

    if verbose:
        timer = tmr.Timer()
        hdict.pprint(head='Shape detection', threshold=threshold)
  
    source = io.as_source(source).array
    seeds = io.as_source(seeds)

    mask = None if threshold is None else source > threshold

    peaks = vox.voxelize(seeds, shape=source.shape, weights=np.arange(1, seeds.shape[0]+1), processes=processes).array
    try:
        shapes = skimage.morphology.watershed(-source, peaks, mask=mask)
    except AttributeError:
        shapes = skimage.segmentation.watershed(-source, peaks, mask=mask)
    # shapes = watershed_ift(-source.astype('uint16'), peaks)
    # shapes[numpy.logical_not(mask)] = 0

    if verbose:
        timer.print_elapsed_time('Shape detection')

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
  