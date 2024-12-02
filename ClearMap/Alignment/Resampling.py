# -*- coding: utf-8 -*-
"""
Resampling
==========

This module provides methods to resample and reorient data. 

Resampling the data is usually necessary as the first step to match the 
resolution and orientation of the reference object.
"""
__author__ = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2023 by Christoph Kirst'
__webpage__ = 'https://idisco.info'
__download__ = 'https://github.com/ClearAnatomics/ClearMap'

import os
import tempfile
import itertools
import functools as ft
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import cv2

import ClearMap.IO.IO as io
import ClearMap.IO.FileList as fl

import ClearMap.ParallelProcessing.ProcessWriter as pw
import ClearMap.ParallelProcessing.ParallelTraceback as ptb

import ClearMap.Utils.Timer as tmr
from ClearMap.Utils.tag_expression import Expression

from .Transformations.Transformation import TransformationBase
from ClearMap.Alignment.orientation import (format_orientation, orientation_to_transposition, orient_resolution,
                                            orient_shape, orient, orient_points)
from ClearMap.Utils.utilities import handle_deprecated_args


def resample_shape_from_resolution(original_shape, original_resolution, resampled_resolution,
                                   orientation=None, discretize=True):
    """Calculate the resampled shape given resolution information.

    Arguments
    ---------
    original_shape : tuple
      Shape of the data array to be resampled.
    original_resolution : tuple
       Resolution of the data array to be resampled.
    resampled_resolution : tuple
       Resolution of the resampled data array.
    orientation : tuple
       Orientation specifications.
    discretize : bool
       If True, return next largest integer values.

    Returns
    -------
    resampled_shape : tuple or None
      The shape of the resampled data array.
    """

    # orient onto resampled array
    original_shape_oriented = orient_shape(original_shape, orientation)
    original_resolution_oriented = orient_resolution(original_resolution, orientation)

    # shape
    resampled_shape = tuple(float(s) * float(r) / float(rr) for s, r, rr
                            in zip(original_shape_oriented, original_resolution_oriented, resampled_resolution))

    if discretize:
        resampled_shape = tuple(int(np.ceil(s)) for s in resampled_shape)

    return resampled_shape


def original_shape_from_resolution(resampled_shape, original_resolution, resampled_resolution,
                                   orientation=None, discretize=True):
    """Calculate the original shape given resolution information.

    Arguments
    ---------
    resampled_shape : tuple
      Shape of the resampled data array.
    original_resolution : tuple
       Resolution of the data array to be resampled.
    resampled_resolution : tuple
       Resolution of the resampled data array.
    orientation : tuple
       Orientation specifications.
    discretize : bool
       If True, return next largest integer values.

    Returns
    -------
    resampled_shape : tuple or None
      The shape of the resampled data array.
    """

    # orient onto original array
    resampled_shape_oriented = orient_shape(resampled_shape, orientation, inverse=True)
    resampled_resolution_oriented = orient_resolution(resampled_resolution, orientation, inverse=True)

    # shape
    original_shape = tuple(float(rs) * float(rr) / float(r) for r, rs, rr
                           in zip(original_resolution, resampled_shape_oriented, resampled_resolution_oriented))

    if discretize:
        original_shape = tuple(int(np.ceil(s)) for s in original_shape)

    return original_shape


def resample_resolution_from_shape(original_shape, resampled_shape, original_resolution, orientation=None):
    """Calculate the resampled resolution given shape information.

    Arguments
    ---------
    original_shape : tuple
      Shape of the data array to be resampled.
    resampled_shape : tuple
       Resolution of the resampled data array.
    original_resolution : tuple
       Resolution of the data array to be resampled.
    orientation : tuple
       Orientation specifications.

    Returns
    -------
    resampled_resolution : tuple or None
      The resolution of the resampled data array.
    """

    # orient onto resampled array
    original_shape_oriented = orient_shape(original_shape, orientation)
    original_resolution_oriented = orient_resolution(original_resolution, orientation)

    resampled_resolution = tuple(float(s) * float(r) / float(rs) for s, r, rs
                                 in zip(original_shape_oriented, original_resolution_oriented, resampled_shape))

    return resampled_resolution


def original_resolution_from_shape(original_shape, resampled_shape, resampled_resolution, orientation=None):
    """Calculate the original resolution given shape information.

    Arguments
    ---------
    original_shape : tuple
      Shape of the data array to be resampled.
    resampled_shape : tuple
       Resolution of the resampled data array.
    resampled_resolution : tuple
       Resolution of the resampled data array.
    orientation : tuple
       Orientation specifications.

    Returns
    -------
    resampled_resolution : tuple or None
      The resolution of the resampled data array.
    """

    # orient onto resampled array
    resampled_shape_oriented = orient_shape(resampled_shape, orientation, inverse=True)
    resampled_resolution_oriented = orient_resolution(resampled_resolution, orientation, inverse=True)

    original_resolution = tuple(float(rs) * float(rr) / float(s) for s, rs, rr
                                in zip(original_shape, resampled_shape_oriented, resampled_resolution_oriented))

    return original_resolution


def resample_information(original_shape=None, resampled_shape=None,
                         original_resolution=None, resampled_resolution=None,
                         original=None, resampled=None,
                         orientation=None, discretize=True, consistent=True):
    """Convert resampling information to standard form.

    This function takes in various optional parameters related to the original and resampled data,
    and returns a tuple containing the original shape, resampled shape, original resolution,
    resampled resolution, and orientation in a standard format.

    Arguments
    ---------
    original_shape : tuple or None
      Optional value of the shape of the data array to be resampled.
    resampled_shape : tuple or None
      Optional value of the shape of the resampled data array.
    original_resolution : tuple or None
      Optional value of the resolution of the data array to be resampled.
    resampled_resolution : tuple or None
       Optional value of the resolution of the resampled data array.
    original : str, array or None
      Optional source to be resampled used to determine missing resampling information.
    resampled: str, array or None
       Optional source of the resampled data used to determine missing resampling information.
    orientation : tuple
      Orientation as specified.
    discretize : bool
      If True and sufficient shape information is given, discretize the shapes.
    consistent:
      If True, recalculate resolutions to match the discrete shapes.


    Returns
    -------
    original_shape : tuple or None
      Value of the shape of the data array to be resampled
    resampled_shape : tuple or None
      Value of the shape of the resampled data array.
    original_resolution : tuple or None
      Value of the resolution of the data array to be resampled.
    resampled_resolution : tuple or None
       Value of the resolution of the resampled data array.
    orientation : tuple
      Orientation in standard formatting.

    See also
    --------
    resample : Function that performs the actual resampling based on the information provided.
    """
    orientation = format_orientation(orientation)

    # shapes form sources
    if original_shape is None and original is not None:
        original_shape = io.shape(original)

    if resampled_shape is None:
        try:
            resampled_shape = io.shape(resampled)
        except FileNotFoundError:
            pass

    # ndim
    for var in [original_shape, resampled_shape, original_resolution, resampled_resolution]:
        if var is not None:
            ndim = len(var)
            break
    else:
        raise ValueError('The resampling information is not sufficient!')

    # resolutions
    if original_shape is None and original_resolution is None:
        original_resolution = (1,) * ndim

    if resampled_shape is None and resampled_resolution is None:
        resampled_resolution = (1,) * ndim

    if original_resolution is None and resampled_resolution is None:
        original_resolution = (1,) * ndim

    # auto complete (if one piece of information is missing)
    if all_not_none([original_shape, original_resolution, resampled_shape]) and resampled_resolution is None:
        resampled_resolution = resample_resolution_from_shape(original_shape, resampled_shape,
                                                              original_resolution, orientation)
    if all_not_none([original_shape, original_resolution, resampled_shape, resampled_resolution]):
        resampled_shape = resample_shape_from_resolution(original_shape,
                                                         original_resolution, resampled_resolution,
                                                         orientation, discretize)
        if discretize and consistent:
            resampled_resolution = resample_resolution_from_shape(original_shape, resampled_shape,
                                                                  original_resolution, orientation)
    if original_shape is not None and original_resolution is None \
            and resampled_shape is not None and resampled_resolution is not None:
        original_resolution = original_resolution_from_shape(original_shape, resampled_shape,
                                                             resampled_resolution, orientation)
    if original_shape is None and all_not_none([original_resolution, resampled_shape, resampled_resolution]):
        original_shape = original_shape_from_resolution(resampled_shape,
                                                        original_resolution, resampled_resolution,
                                                        orientation, discretize)
        if discretize and consistent:
            original_resolution = original_resolution_from_shape(original_shape, resampled_shape,
                                                                 resampled_resolution, orientation)

    return original_shape, resampled_shape, original_resolution, resampled_resolution, orientation


def all_not_none(*args):
    return all([a is not None for a in args])


def resample_shape(original_shape=None, resampled_shape=None,
                   original_resolution=None, resampled_resolution=None,
                   original=None, resampled=None,
                   orientation=None, consistent=True):
    """Calculate the resampled shape.

    Arguments
    ---------
    original_shape : tuple or None
      Optional value of the shape of the data array to be resampled.
    resampled_shape : tuple or None
      Optional value of the shape of the resampled data array.
    original_resolution : tuple or None
      Optional value of the resolution of the data array to be resampled.
    resampled_resolution : tuple or None
       Optional value of the resolution of the resampled data array.
    original : str, array or None
      Optional source to be resampled used to determine missing resampling information.
    resampled: str, array or None
       Optional source of the resampled data used to determine missing resampling information.
    orientation : tuple
      Orientation as specified.
    consistent:
      If True, recalculate resolutions to match the discrete shapes.

    Returns
    -------
    original_shape : tuple or None
      Value of the shape of the data array to be resampled
    resampled_shape : tuple or None
      Value of the shape of the resampled data array.
    original_resolution : tuple or None
      Value of the resolution of the data array to be resampled.
    resampled_resolution : tuple or None
       Value of the resolution of the resampled data array.

    See also
    --------
    resample: For more details.

    :mod:`ClearMap.Alignment.orientation`
    """
    if original_shape is None and resampled_shape is None:
        raise RuntimeError('Either the original or resampled shape must be defined to determine all shapes!')

    original_shape, resampled_shape, original_resolution, resampled_resolution, orientation = \
        resample_information(original_shape, resampled_shape,
                             original_resolution, resampled_resolution,
                             original, resampled,
                             orientation, discretize=True, consistent=consistent)

    return original_shape, resampled_shape, original_resolution, resampled_resolution


def resample_resolution(original_shape=None, resampled_shape=None,
                        original_resolution=None, resampled_resolution=None,
                        original=None, resampled=None,
                        orientation=None, discretize=True, consistent=True):
    """Calculate resolutions for original and resampled data.

    Arguments
    ---------
    original_shape : tuple or None
      Optional value of the shape of the data array to be resampled.
    resampled_shape : tuple or None
      Optional value of the shape of the resampled data array.
    original_resolution : tuple or None
      Optional value of the resolution of the data array to be resampled.
    resampled_resolution : tuple or None
       Optional value of the resolution of the resampled data array.
    original : str, array or None
      Optional source to be resampled used to determine missing resampling information.
    resampled: str, array or None
       Optional source of the resampled data used to determine missing resampling information.
    orientation : tuple
      Orientation as specified.
    discretize : bool
      If True and sufficient shape information is given, discretize the shapes.
    consistent:
      If True, recalculate resolutions to match the discrete shapes.

    Returns
    -------
    original_resolution : tuple or None
      Value of the resolution of the data array to be resampled.
    resampled_resolution : tuple or None
       Value of the resolution of the resampled data array.

    See also
    --------
    resample : For more details.
    :mod:`ClearMap.Alignment.orientation`
    """
    original_shape, resampled_shape, original_resolution, resampled_resolution, orientation = \
        resample_information(original_shape, resampled_shape,
                             original_resolution, resampled_resolution,
                             original, resampled,
                             orientation, discretize=discretize, consistent=consistent)

    if original_resolution is None or resampled_resolution is None:
        raise ValueError('Cant determine original or resmapled resolutions from the resampling specifications.')

    return original_resolution, resampled_resolution


# Usage
@handle_deprecated_args({
    'source': 'original',
    'sink': 'resampled',
    'source_shape': 'original_shape',
    'sink_shape': 'resampled_shape',
    'source_resolution': 'original_resolution',
    'sink_resolution': 'resampled_resolution'
})
def resample_factor(original_shape=None, resampled_shape=None,
                    original_resolution=None, resampled_resolution=None,
                    original=None, resampled=None,
                    orientation=None, discretize=True, consistent=True):
    """Calculate scaling factors for the resampling.

    Arguments
    ---------
    original_shape : tuple or None
      Optional value of the shape of the data array to be resampled.
    resampled_shape : tuple or None
      Optional value of the shape of the resampled data array.
    original_resolution : tuple or None
      Optional value of the resolution of the data array to be resampled.
    resampled_resolution : tuple or None
       Optional value of the resolution of the resampled data array.
    original : str, array or None
      Optional source to be resampled used to determine missing resampling information.
    resampled: str, array or None
       Optional source of the resampled data used to determine missing resampling information.
    orientation : tuple
      Orientation as specified.
    discretize : bool
      If True and sufficient shape information is given, discretize the shapes.
    consistent:
      If True, recalculate resolutions to match the discrete shapes.

    .. deprecated:: 2.1.0
        The following arguments are deprecated and will be removed in version 3.0.0:

        - source (now original)
        - sink (now resampled)
        - source_shape (now original_shape)
        - sink_shape (now resampled_shape)
        - source_resolution (now original_resolution)
        - sink_resolution (now resampled_resolution)

    Returns
    -------
    original_resolution : tuple or None
      Value of the resolution of the data array to be resampled.
    resampled_resolution : tuple or None
       Value of the resolution of the resampled data array.

    See also
    --------
    resample : For more details.
    :mod:`ClearMap.Alignment.orientation`
    """

    original_resolution, resampled_resolution = \
        resample_resolution(original_shape, resampled_shape,
                            original_resolution, resampled_resolution,
                            original, resampled,
                            orientation, discretize=discretize, consistent=consistent)

    resampled_resolution_in_source_orientation = orient_resolution(resampled_resolution, orientation, inverse=True)

    factor = tuple(float(r) / float(rr) for r, rr
                   in zip(original_resolution, resampled_resolution_in_source_orientation))

    return factor


########################################################################################
# Resample
########################################################################################

@handle_deprecated_args({
    'source': 'original',
    'sink': 'resampled',
    'source_shape': 'original_shape',
    'sink_shape': 'resampled_shape',
    'source_resolution': 'original_resolution',
    'sink_resolution': 'resampled_resolution'
})
def resample(original, resampled=None,
             original_shape=None, resampled_shape=None,
             original_resolution=None, resampled_resolution=None, orientation=None,
             interpolation='linear', axes_order=None, method='shared', processes=None, workspace=None, verbose=True):
    """Resample data of source in new shape/resolution and orientation.

    Arguments
    ---------
    original : str, array or None
      Data array source to be resampled.
    resampled: str, array or None
       Optional sink for the resampled data.
    original_shape : tuple or None
      Optional value for the shape of the data array to be resampled.
      Determined by the shape of the original source by default.
    resampled_shape : tuple or None
      Optional value of the shape of the resampled data array.
    original_resolution : tuple or None
      Optional value of the resolution of the data array to be resampled.
    resampled_resolution : tuple or None
       Optional value of the resolution of the resampled data array.
    orientation : tuple or None
      Orientation specification.
    interpolation : str
      Interpolation method to use. Available methods are 'linear', 'nearest', 'area'.
    axes_order : list of tuples of int or None
      The axes pairs along which to resample the data at each step.
      If None, this is determined automatically. For a FileList source,
      setting the first tuple should point to axis not indicating files.
      If 'size' the axis order is determined automatically to maximally reduce
      the size of the array in each resampling step.
      If 'order' the axis order is chosen automatically to optimize io speed.
    method : 'shared' or 'memmap'
      Method to handle intermediate resampling results. If 'shared' use shared
      memory, otherwise use a memory map on disk.
    processes : int, None or 'serial'
      Number of processes to use for parallel resampling, if None use maximal
      processes available, if 'serial' process in serial.
    verbose : bool
      If True, display progress information.

    .. deprecated:: 2.1.0
        The following arguments are deprecated and will be removed in version 3.0.0:

        - source (now original)
        - sink (now resampled)
        - source_shape (now original_shape)
        - sink_shape (now resampled_shape)
        - source_resolution (now original_resolution)
        - sink_resolution (now resampled_resolution)


    Returns
    -------
    resampled : array or str
      The data or filename of resampled sink.

    Notes
    -----
    * Resolutions are assumed to be given for the axes of the intrinsic
      orientation of the data and reference (as when viewed by ImageJ).
    * Orientation: permutation of 1,2,3 with potential sign, indicating which
      axes map onto the reference axes, a negative sign indicates reversal
      of that particular axes.
    * Only a minimal set of information to determine the resampling parameter
      has to be given, e.g. original_shape or the original source and resampled_shape.
    * The resampling is done by iterating two-dimensional resampling steps.
    """
    # TODO: write full nd resampling routine extending cv2 lib.
    if verbose:
        timer = tmr.Timer()

    if os.path.splitext(original)[1] == '.tif':  # Resampling is much faster from .npy files
        exp = Expression(original)
        for tag in exp.tags:
            exp = Expression(exp.pattern[0].replace(str(tag), ''))
        new_path = exp.pattern[0] + '.npy'
        io.convert(original, new_path)
        original = new_path
    else:
        new_path = None

    original = io.as_source(original)
    dtype = original.dtype
    order = original.order

    if original_shape is not None and original_shape != original.shape:
        original_resolution, resampled_resolution = resample_resolution(original_shape, resampled_shape,
                                                                        original_resolution, resampled_resolution,
                                                                        original, resampled,
                                                                        orientation, consistent=True)
        original_shape = io.shape(original)
        resampled_shape = None
    else:
        original_shape = original.shape

    if original.ndim == 4 and 3 not in original_shape:  # 4D but not color
        raise ValueError(f'Unsupported shape for original: "{original_shape}"')

    original_shape, resampled_shape, original_resolution, resampled_resolution = \
        resample_shape(original_shape, resampled_shape,
                       original_resolution, resampled_resolution,
                       original, resampled,
                       orientation, consistent=True)

    resampled_shape_in_original_orientation = orient_shape(resampled_shape, orientation, inverse=True)

    interpolation = _interpolation_to_cv2(interpolation, dtype=dtype)

    if not isinstance(processes, int) and processes != 'serial':
        processes = io.mp.cpu_count()

    # determine order of resampling
    axes_order, shape_order = _axes_order(axes_order, original_shape, resampled_shape_in_original_orientation,
                                          order=order, source=original, minimize_size=True)
    if verbose:
        print('Resampling: %r -> %r using axes:%r and intermediate shapes:%r' %
              (original_shape, resampled_shape, axes_order, shape_order))

    if len(axes_order) == 0:
        if verbose:
            print('Resampling: no resampling necessary, source has same size as sink!')
        if original != resampled:  # TODO: this should be handled by Source functionality !
            return io.write(resampled, original)
        else:
            return original

    # resample
    n_steps = len(axes_order)
    resampled_data = last = original
    delete_files = []
    if new_path is not None:
        delete_files.append(new_path)
    for step, axes, shape in zip(range(n_steps), axes_order, shape_order):
        if step == n_steps - 1 and orientation is None:  # Create final resampled file for last step
            resampled_data = io.initialize(resampled, shape=resampled_shape, dtype=dtype, as_source=True)
        else:
            if method == 'shared':
                resampled_data = io.sma.create(shape, dtype=dtype, order=order, as_source=True)
            else:
                location = tempfile.mktemp() + '.npy'
                resampled_data = io.mmp.create(location, shape=shape, dtype=dtype, order=order, as_source=True)
                delete_files.append(location)

        # indices for non-resampled axes
        indices = tuple(range(s) for d, s in enumerate(shape) if d not in axes)
        indices = tuple(i for i in itertools.product(*indices))
        n_indices = len(indices)

        # resample step
        last_virtual = last.as_virtual()
        resampled_data_virtual = resampled_data.as_virtual()
        _resample = ft.partial(_resample_2d, source=last_virtual, sink=resampled_data_virtual, axes=axes, shape=shape,
                               interpolation=interpolation, n_indices=n_indices, verbose=verbose)

        if processes == 'serial':
            for index in indices:
                _resample(index=index)
        else:
            # with CancelableProcessPoolExecutor(processes) as executor:
            # ThreadPool because of documented cv2 instability w/ multiprocessing. Is this still true ?
            with ThreadPoolExecutor(processes) as executor:
                chunk_size = len(indices) // (processes * 3)  # REFACTOR: explain calculation
                result = executor.map(_resample, indices, chunksize=chunk_size)  # default chunk_size is 1 (too small)
                if workspace is not None:
                    workspace.executor = executor
                result = list(result)

        last = resampled_data

    if orientation is not None:
        resampled_data = orient(resampled_data, orientation)
        resampled = io.write(resampled, resampled_data)
    else:
        resampled = resampled_data

    for f in delete_files:
        io.delete_file(f)

    if verbose:
        timer.print_elapsed_time('Resampling')

    return resampled


def resample_inverse(resampled, original=None,
                     original_shape=None, resampled_shape=None,
                     original_resolution=None, resampled_resolution=None, orientation=None,
                     axes_order=None, method='memmap', interpolation='linear',
                     processes=None, verbose=True, workspace=None):
    """Resample data inversely to :func:`resample` routine.

    Arguments
    ---------
    resampled: str, array or None
       Data array to be inversely resampled.
    original : str, array or None
      Optional sink for the inversely resampled array.
    original_shape : tuple or None
      Optional value for the shape of the original data array.
    resampled_shape : tuple or None
      Optional value of the shape of the resampled data array.
      Determined by the shape of the resampled source by default.
    original_resolution : tuple or None
      Optional value of the resolution of the data array to be resampled.
    resampled_resolution : tuple or None
       Optional value of the resolution of the resampled data array.
    orientation : tuple or None
      Orientation specification.
    interpolation : str
      Interpolation method to use. Available methods are 'linear', 'nearest', 'area'.
    axes_order : list of tuples of int or None
      The axes pairs along which to resample the data at each step.
      If None, this is determined automatically. For a FileList source,
      setting the first tuple should point to axis not indicating files.
      If 'size' the axis order is determined automatically to maximally reduce
      the size of the array in each resampling step.
      If 'order' the axis order is chosen automatically to optimize io speed.
    method : 'shared' or 'memmap'
      Method to handle intermediate resampling results. If 'shared' use shared
      memory, otherwise use a memory map on disk.
    processes : int, None or 'serial'
      Number of processes to use for parallel resampling, if None use maximal
      processes available, if 'serial' process in serial.
    verbose : bool
      If True, display progress information.

    Returns
    -------
    resampled : array or str
       Data or file name of inversely resampled image.

    Notes
    -----
    * All arguments, except source and sink should be passed as :func:`resample`
      to invert the resampling.
    """
    resampled = io.as_source(resampled)

    # invert orientation
    resampled = orient(resampled, orientation, inverse=True)

    # resampled data in original orientation
    resampled_shape = orient_shape(resampled_shape, orientation, inverse=True)
    resampled_resolution = orient_resolution(resampled_resolution, orientation, inverse=True)
    if axes_order is not None:
        transposition = orientation_to_transposition(orientation, inverse=True)
        axes_map = {i: t for i, t in enumerate(transposition)}
        axes_order = [(axes_map[axes[0]], axes_map[axes[1]]) for axes in axes_order]

    # inversely resample
    return resample(resampled, original,
                    resampled_shape, original_shape,
                    resampled_resolution, original_resolution, orientation=None,
                    interpolation=interpolation, axes_order=axes_order,
                    method=method, processes=processes, workspace=workspace, verbose=verbose)


########################################################################################
# Resample Points
########################################################################################
@handle_deprecated_args({
    'source': 'original_points',
    'sink': 'resampled_points',
    'source_shape': 'original_shape',
    'sink_shape': 'resampled_shape',
    'source_resolution': 'original_resolution',
    'sink_resolution': 'resampled_resolution'
})
def resample_points(original_points, resampled_points=None,
                    original_shape=None, resampled_shape=None,
                    original_resolution=None, resampled_resolution=None,
                    original=None, resampled=None,
                    orientation=None):
    """Transform point coordinates according to resampling specifications.

    Arguments
    ---------
    original_points : str, array or None
      Data array source to be resampled.
    resampled_points: str, array or None
       Optional sink for the resampled points
    original_shape : tuple or None
      Optional value for the shape of the data array to be resampled.
      Determined by the shape of the original source by default.
    resampled_shape : tuple or None
      Optional value of the shape of the resampled data array.
    original_resolution : tuple or None
      Optional value of the resolution of the data array to be resampled.
    resampled_resolution : tuple or None
       Optional value of the resolution of the resampled data array.
    original : array or str
       Optional original source data used to infer resampling specifications.
    resampled : array or str
       Optional resampled source data used to infer resampling specifications.
    orientation : tuple or None
      Orientation specification.

    .. deprecated:: 2.1.0
        The following arguments are deprecated and will be removed in version 3.0.0:

        - source (now original_points)
        - sink (now resampled_points)
        - source_shape (now original_shape)
        - sink_shape (now resampled_shape)
        - source_resolution (now original_resolution)
        - sink_resolution (now resampled_resolution)

    Returns
    -------
    resampled : array or str
      The array or filename of resampled points.

    Notes
    -----
    * The resampling of points here corresponds to he resampling of a data array in :func:`resample`.
    * The arguments should be passed exactly as in :func:`resample` except for the additional original_points
      and resampled_points.
    """
    factor = resample_factor(original_shape, resampled_shape,
                             original_resolution, resampled_resolution,
                             original, resampled,
                             orientation, discretize=True, consistent=True)

    original_shape, resampled_shape, original_resolution, resampled_resolution = \
        resample_shape(original_shape, resampled_shape,
                       original_resolution, resampled_resolution,
                       original, resampled,
                       orientation, consistent=True)

    resampled = io.as_source(original_points)
    resampled = resampled[:] * factor
    resampled = orient_points(resampled, orientation, shape=orient_shape(resampled_shape, orientation, inverse=True))
    return io.write(resampled_points, resampled)


def resample_points_inverse(resampled_points, original_points=None,
                            original_shape=None, resampled_shape=None,
                            original_resolution=None, resampled_resolution=None,
                            original=None, resampled=None,
                            orientation=None):
    """Transform point coordinates inversely according to resampling specifications.

    Arguments
    ---------
    resampled_points: str, array or None
       Optional source of the resampled points to be resampled inversely.
    original_points : str, array or None
      Data array sink for inversely resampled points.
    original_shape : tuple or None
      Optional value for the shape of the data array to be resampled.
      Determined by the shape of the original source by default.
    resampled_shape : tuple or None
      Optional value of the shape of the resampled data array.
    original_resolution : tuple or None
      Optional value of the resolution of the data array to be resampled.
    resampled_resolution : tuple or None
       Optional value of the resolution of the resampled data array.
    original : array or str
       Optional original source data used to infer resampling specifications.
    resampled : array or str
       Optional resampled source data used to infer resampling specifications.
    orientation : tuple or None
      Orientation specification.

    Returns
    -------
    resampled : array or str
      The array or filename of resampled points.

    Notes
    -----
    * The resampling of points here corresponds to he resampling of a data array in :func:`resample`.
    * The arguments should be passed exactly as in :func:`resample` except for the additional original_points
      and resampled_points.
    """
    factor = resample_factor(original_shape, resampled_shape,
                             original_resolution, resampled_resolution,
                             original, resampled,
                             orientation, discretize=True, consistent=True)

    original_shape, resampled_shape, original_resolution, resampled_resolution = \
        resample_shape(original_shape, resampled_shape,
                       original_resolution, resampled_resolution,
                       original, resampled,
                       orientation, consistent=True)

    resampled_points = io.as_source(resampled_points)
    original = orient_points(resampled_points, orientation, shape=resampled_shape, inverse=True)
    original = original[:] / factor
    return io.write(original_points, original)


########################################################################################
# Transformation interface
########################################################################################

class ResamplingTransformation(TransformationBase):
    def __init__(self, ttype='Resampling', inverse=False,
                 original_resolution=None, resampled_resolution=None,
                 original_shape=None, resampled_shape=None,
                 original=None, resampled=None,
                 orientation=None):

        super().__init__(ttype=ttype, inverse=inverse)

        original_shape, resampled_shape, original_resolution, resampled_resolution, orientation = \
            resample_information(original_shape, resampled_shape,
                                 original_resolution, resampled_resolution,
                                 original, resampled,
                                 orientation, discretize=True, consistent=True)

        self.original_shape = original_shape
        self.resampled_shape = resampled_shape
        self.original_resolution = original_resolution
        self.resampled_resolution = resampled_resolution
        self.orientation = orientation

    def resample_kwargs(self):
        return dict(original_shape=self.original_shape, resampled_shape=self.resampled_shape,
                    resampled_resolution=self.resampled_resolution, original_resolution=self.original_resolution,
                    orientation=self.orientation)

    def resample_factor(self):
        return resample_factor(**self.resample_kwargs())

    def transform_data(self, source, sink=None, inverse=False, **kwargs):
        inverse = self.get_inverse(inverse)
        resample_kwargs = self.resample_kwargs()
        resample_kwargs.update(kwargs)
        if not inverse:
            return resample(source, sink, **resample_kwargs)
        else:
            return resample_inverse(source, sink, **resample_kwargs)

    def transform_points(self, source, sink=None, inverse=False, **kwargs):
        inverse = self.get_inverse(inverse)
        resample_kwargs = self.resample_kwargs()
        resample_kwargs.update(kwargs)
        if not inverse:
            return resample_points(source, sink, **resample_kwargs)
        else:
            return resample_points_inverse(source, sink, **resample_kwargs)

    def transform_shape(self, shape, inverse=False, **kwargs):
        kwargs = self.resample_kwargs()
        kwargs.update(**kwargs)
        kwargs.update(original_shape=shape)
        return resample_shape(shape, **kwargs)

    def to_dict(self) -> dict:
        dictionary = super().to_dict()
        dictionary.update(self.resample_kwargs())
        return dictionary

    def __repr__(self):
        orientation = '' if self.orientation is None else ('%r ' % (self.orientation,))
        return '%s[%s%r->%r]' % (super().__repr__(), orientation, self.original_resolution, self.resampled_resolution)


class OrientationTransformation(TransformationBase):

    ttype = 'Orientation'

    def __init__(self, orientation=None, shape=None,inverse=None):
        super().__init__(inverse=inverse)
        self.orientation = format_orientation(orientation)
        self.shape = shape

    def transform_data(self, source, sink=None, inverse=False):
        inverse = self.get_inverse(inverse)
        return io.write(sink, orient(source, orientation=self.orientation, inverse=inverse))

    def transform_points(self, source, sink=None, inverse=False, **kwargs):
        inverse = self.get_inverse(inverse)
        return io.write(sink, orient_points(source, orientation=self.orientation, shape=self.shape, inverse=inverse))

    def transform_shape(self, shape, inverse=False, **kwargs):
        return orient_shape(self.orientation, shape, inverse=inverse)

    def to_dict(self) -> dict:
        dictionary = super().to_dict()
        dictionary.update(orientation=self.orientation,
                          shape=self.shape)
        return dictionary

    def __repr__(self):
        orientation = '' if self.orientation is None else ('%r' % (self.orientation,))
        shape = '' if self.shape is None else (' %r ' % (self.shape,))
        return '%s[%s%s]' % (super().__repr__(), orientation, shape)


########################################################################################
# Helpers
########################################################################################

@ptb.parallel_traceback
def _resample_2d(index, source, sink, axes, shape, interpolation, n_indices, verbose):
    """Resampling helper function to use for parallel resampling of image slices"""
    if verbose:
        pw.ProcessWriter(index).write("Resampling: resampling axes %r, slice %r / %d" % (axes, index, n_indices))

    # slicing
    slicing_ = ()
    i = 0
    for d in range(len(shape)):
        if d in axes:
            slicing_ += (slice(None),)
        else:
            slicing_ += (index[i],)
            i += 1

    # resample
    sink = sink.as_real()
    source = source.as_real()
    new_shape = (shape[axes[1]], shape[axes[0]])
    sink[slicing_] = cv2.resize(source[slicing_], new_shape, interpolation=interpolation)
    # note cv2 takes reverse shape order !


def _order_axes(original_shape, resampled_shape, resample_axes, resample_factors,
                sort_factors=None, minimize_size=True):
    """Helper to order axes resampling according to minimize or maximize the change in data size at each step.

    Note
    ----
    The change in data size at each step is given by the change in size between two intermediate 2d resampling steps.
    Note that the resampled_shape is assumed to be given in the original orientation.

    # (1000, 50, 10) -> (500, 10, 5) / factors (1/2, 1/5, 1/2)
    # 0,1 (1000, 50, 10) -> (500,10,10) -> size:50000
    # 0,2 (1000, 50, 10) -> (500,50,5)  -> size:125000
    # 1,2 (1000, 50, 10) -> (1000,10,5) -> size:50000

    # (1000, 100, 10) -> (500, 50, 5) / factors (1/2, 1/2, 1/2)
    # 0,1 (1000, 100, 10) -> (500,50,10) -> size:250000
    # 0,2 (1000, 100, 10) -> (500,100,5) -> size:250000
    # 1,2 (1000, 100, 10) -> (1000,50,5) -> size:250000
    """
    resample_axes = np.array(resample_axes)
    if sort_factors is None:
        sort_factors = resample_factors

    axes_order = []
    shape_order = []
    current_shape = original_shape

    while len(resample_axes) > 0:
        if len(resample_axes) >= 2:
            # take the best two resampling factors
            best = np.argsort(sort_factors)[:2] if minimize_size else np.argsort(sort_factors)[-2:]
            current_axes = tuple(np.sort(resample_axes[best]))
            resample_axes = np.array([s for a, s in enumerate(resample_axes) if a not in best])
            resample_factors = np.array([s for a, s in enumerate(resample_factors) if a not in best])
            sort_factors = np.array([s for a, s in enumerate(sort_factors) if a not in best])
        else:
            # take remaining axis and best resampled axis
            axis = resample_axes[0]
            best_axis = np.argsort(current_shape) if minimize_size else np.argsort(current_shape)[::-1]
            best_axis = [a for a in best_axis if a != axis][0]
            current_axes = (axis, best_axis) if axis < best_axis else (best_axis, axis)
            resample_axes = []

        current_shape = tuple(s if d not in current_axes else t
                              for d, (s, t) in enumerate(zip(current_shape, resampled_shape)))
        axes_order.append(current_axes)
        shape_order.append(current_shape)

    return axes_order, shape_order


def _axes_order(axes_order, original_shape, resampled_shape, order=None, source=None, minimize_size=True):
    """Helper to find axes order for subsequent 2d resampling steps."""

    # specified axes_order
    if axes_order is not None and isinstance(axes_order, list):
        axes_order = [(a[0], a[1]) if a[0] < a[1] else (a[1], a[0]) for a in axes_order]
        shape_order = []
        last_shape = original_shape
        for axes in axes_order:
            if not isinstance(axes, tuple) and len(axes) != 2:
                raise ValueError('resampling; expected a tuple of len 2 for axes_order entry, got %r!' % axes)
            last_shape = tuple(s if d not in axes else t for d, (s, t) in enumerate(zip(last_shape, resampled_shape)))
            shape_order.append(last_shape)
        return axes_order, shape_order

    # determine axes order automatically
    if axes_order is None:
        axes_order = 'order'
    if axes_order == 'order' and order is None and not isinstance(source, fl.Source):
        axes_order = 'size'

    # only select axes that need resampling
    resample_axes = [d for d, (s, rs) in enumerate(zip(original_shape, resampled_shape)) if s != rs]
    resample_factors = [float(rs) / float(s) for s, rs in zip(original_shape, resampled_shape) if s != rs]

    # axes and shape order results
    if axes_order == 'size':  # order to reduce size as much as possible in each sub-resampling step
        return _order_axes(original_shape, resampled_shape, resample_axes, resample_factors, None, minimize_size)
    elif axes_order == 'order':  # order axes according to file or array order for faster io
        # determine order according to file structure (i.e. resample individual files first)
        if isinstance(source, fl.Source):
            axes_list = source.axes_list  # axes for individual files
            shift = -(np.max(resample_factors) + 1)  # make file factors the smallest
            if not minimize_size:
                shift = -shift  # make file factors the biggest
            sort_factors = [f + shift if a in axes_list else f for a, f in zip(resample_axes, resample_factors)]
            return _order_axes(original_shape, resampled_shape, resample_axes,
                               resample_factors, sort_factors, minimize_size)
        else:  # follow 'C' or 'F' array order
            axes_order = []
            shape_order = []
            current_shape = original_shape
            while len(resample_axes) > 0:
                if len(resample_axes) >= 2:
                    slicing = slice(-2, None) if order == 'C' else slice(None, 2)
                    current_axes = tuple(resample_axes[slicing])
                else:
                    current_axes = (resample_axes[0], axes_order[-1][0]) if order == 'C' \
                        else (axes_order[-1][1], resample_axes[0])
                current_shape = tuple(s if d not in current_axes else t
                                      for d, (s, t) in enumerate(zip(current_shape, resampled_shape)))

                axes_order.append(current_axes)
                shape_order.append(current_shape)

                resample_axes = np.array([a for a in resample_axes if a not in current_axes])

            return axes_order, shape_order
    else:
        raise ValueError("axes_order not 'size','order' or list but %r!" % axes_order)


_interpolation_to_cv2_map = {
    cv2.INTER_NEAREST: cv2.INTER_NEAREST,
    'nearest': cv2.INTER_NEAREST,
    None: cv2.INTER_NEAREST,

    cv2.INTER_AREA: cv2.INTER_AREA,
    'area': cv2.INTER_AREA,

    cv2.INTER_LINEAR: cv2.INTER_LINEAR,
    'linear': cv2.INTER_LINEAR,

    cv2.INTER_CUBIC: cv2.INTER_CUBIC,
    'cubic': cv2.INTER_CUBIC,

    cv2.INTER_LANCZOS4: cv2.INTER_LANCZOS4,
    'lanczos': cv2.INTER_LANCZOS4,

    cv2.INTER_LINEAR_EXACT: cv2.INTER_LINEAR_EXACT,
    'linear_exact': cv2.INTER_LINEAR_EXACT,

    cv2.INTER_NEAREST_EXACT: cv2.INTER_NEAREST_EXACT,
    'nearest_exact': cv2.INTER_NEAREST_EXACT
}

_interpolation_method_for_int_map = {
    cv2.INTER_LINEAR: cv2.INTER_LINEAR_EXACT,
    cv2.INTER_CUBIC: cv2.INTER_LINEAR_EXACT,
    cv2.INTER_LANCZOS4: cv2.INTER_LINEAR_EXACT,
    cv2.INTER_AREA: cv2.INTER_NEAREST,
}


def _interpolation_to_cv2(interpolation, default=cv2.INTER_LINEAR, dtype=None):
    """Helper to convert interpolation specification to CV2 format."""
    interpolation = _interpolation_to_cv2_map.get(interpolation, default)

    # check if consistent with data type
    if dtype is not None and np.dtype(dtype) not in [float, np.dtype('float32')]:
        correct_interpolation = _interpolation_method_for_int_map.get(interpolation, interpolation)
        if correct_interpolation != interpolation:
            print('Resampling: Warning: the interpolation method requires a float data type, '
                  'using an exact method instead!')
        interpolation = correct_interpolation

    return interpolation


########################################################################################
# Test
########################################################################################

def _test():
    """Tests"""
    import numpy as np
    import ClearMap.Settings as settings
    import ClearMap.IO.IO as io

    import ClearMap.Alignment.Resampling as res
    from ClearMap.Alignment.orientation import orient, orient_points
    from importlib import reload
    reload(res)

    # orientation
    data = np.zeros((15, 16, 17))
    data[5, 6, 7] = 1
    orientation = (-3, 1, 2)

    data_oriented = orient(data, orientation)
    data_inverse = orient(data_oriented, orientation, inverse=True)
    np.all(data == data_inverse)

    points = np.array(np.where(data)).T
    points_oriented = orient_points(points, orientation, shape=data.shape)
    np.all(points_oriented == np.array(np.where(data_oriented)).T)

    r = res.resample_information(original_shape=(100, 200, 300), resampled_shape=(50, 50, 30))
    print('original_shape=%r, resampled_shape=%r, original_resolution=%r resampled_resolution=%r orientation=%r' % r)

    r = res.resample_shape(original_shape=(100, 200, 300), resampled_shape=(50, 50, 30))
    print('original_shape=%r, resampled_shape=%r, original_resolution=%r resampled_resolution=%r' % r)

    r = res.resample_shape(original_shape=(100, 200, 300), original_resolution=(2, 2, 2),
                           resampled_resolution=(10, 2, 1))
    print('original_shape=%r, resampled_shape=%r, original_resolution=%r resampled_resolution=%r' % r)

    # random sources
    from importlib import reload
    reload(res)

    shape = (40, 30, 10)

    original = np.random.rand(40, 30, 10)
    x, y, z = np.meshgrid(*tuple(np.arange(s) for s in shape))
    # original = x + y + z
    original = np.array(x + y + z, dtype=float)

    resampled = res.resample(original, original_resolution=(1, 1, 1), resampled_shape=(10, 10, 10), processes='serial')
    print(resampled.shape)
    upsampled = res.resample_inverse(resampled, original_resolution=(1, 1, 1), original_shape=original.shape,
                                     processes='serial')
    print(upsampled.shape)
    import ClearMap.Visualization.Plot3d as p3d
    p3d.plot([resampled])
    p3d.plot([original, upsampled])

    source = io.join(settings.test_data_path, 'Resampling/test.tif')
    sink = io.join(settings.test_data_path, "Resampling/resampled.npy")

    source = io.join(settings.test_data_path, 'Tif/sequence/sequence<Z,4>.tif')
    sink = io.join(settings.test_data_path, "Resampling/resampled_sequence.tif")

    source_shape, sink_shape, source_res, sink_res = res.resample_shape(source_shape=io.shape(source),
                                                                        source_resolution=(1., 1., 1.),
                                                                        sink_resolution=(1.6, 1.6, 2))
    axes_order = res._axes_order(None, source_shape, sink_shape)
    print(axes_order)
    resampled = res.resample(source, sink, source_resolution=(1., 1., 1.), sink_resolution=(1.6, 1.6, 2),
                             orientation=None, processes=None)
    p3d.plot(resampled)
    p3d.plot(source)

    inverse = res.resample_inverse(resampled, sink=None, resample_source=source, source_resolution=(1, 1, 1),
                                   sink_resolution=(10, 10, 2), orientation=None, processes='serial')
    p3d.plot([source, inverse])

    resampled = res.resample(source, sink, source_resolution=(1, 1, 1), sink_resolution=(10, 10, 2),
                             orientation=(2, -1, 3), processes=None)
    p3d.plot(resampled)

    inverse = res.resample_inverse(resampled, sink=None, resample_source=source, source_resolution=(1, 1, 1),
                                   sink_resolution=(10, 10, 2), orientation=(2, -1, 3), processes=None)
    p3d.plot([source, inverse])

    resampled = res.resample(source, sink=None, source_resolution=(1.6, 1.6, 2), sink_shape=(10, 20, 30),
                             orientation=None, processes=None)
    p3d.plot(resampled)

    # ponints
    points = res.np.array([[0, 0, 0], [1, 1, 1], [1, 2, 3]], dtype=float)
    resampled_points = res.resample_points(points, resample_source=source, resample_sink=sink, orientation=None)
    print(resampled_points)

    inverse_points = res.resample_points_inverse(resampled_points, resample_source=source, resample_sink=sink,
                                                 orientation=None)
    print(inverse_points)
    print(res.np.allclose(points, inverse_points))

    # random sources
    from importlib import reload
    reload(res)

    source = np.random.rand(20, 30, 40)
    resampled = res.resample(source, source_resolution=(1, 1, 1), sink_shape=(10, 11, 12), processes='serial')
    print(resampled.shape)
    upsampled = res.resample_inverse(resampled, source_resolution=(1, 1, 1), resample_source=source, processes='serial')
    print(upsampled.shape)
    import ClearMap.Visualization.Plot3d as p3d
    p3d.plot([resampled])
    p3d.plot([source, upsampled])

# def _axes_order(axes_order, source, sink_shape_in_source_orientation, order=None):
#     """Helper to find axes order for subsequent 2d resampling steps."""
#
#     source_shape = source.shape
#     ndim = source.ndim
#
#     if axes_order is not None and isinstance(axes_order, list):
#         axes_order = [(a[0], a[1]) if a[0] < a[1] else (a[1], a[0]) for a in axes_order]
#         shape_order = []
#         last_shape = source_shape
#         for axes in axes_order:
#             if not isinstance(axes, tuple) and len(axes) != 2:
#                 raise ValueError('resampling; expected a tuple of len 2 for axes_order entry, got %r!' % axes)
#             last_shape = tuple([s if d not in axes else t for d, s, t in
#                                 zip(range(ndim), last_shape, sink_shape_in_source_orientation)])
#             shape_order.append(last_shape)
#         return axes_order, shape_order
#
#     else:  # determine automatically
#         if axes_order is None:
#             axes_order = 'order'
#         if axes_order == 'order' and order is None and not isinstance(source, fl.Source):
#             axes_order = 'size'
#
#         if axes_order == 'size':  # order to reduce size as much as possible in each sub-resampling step
#             resample_axes = np.array(
#                 [d for d, s, t in zip(range(ndim), sink_shape_in_source_orientation, source_shape) if s != t])
#             resample_factors = np.array(
#                 [float(t) / float(s) for s, t in zip(sink_shape_in_source_orientation, source_shape) if s != t])
#
#             axes_order = []
#             shape_order = []
#             last_shape = source_shape
#
#             while len(resample_axes) > 0:
#                 if len(resample_axes) >= 2:
#                     # take the largest two resampling factors
#                     ids = np.argsort(resample_factors)[-2:]
#                     axes = tuple(np.sort(resample_axes[ids]))
#                     last_shape = tuple([s if d not in axes else t for d, s, t in
#                                         zip(range(ndim), last_shape, sink_shape_in_source_orientation)])
#
#                     axes_order.append(axes)
#                     shape_order.append(last_shape)
#
#                     resample_axes = np.array([s for a, s in enumerate(resample_axes) if a not in ids])
#                     resample_factors = np.array([s for a, s in enumerate(resample_factors) if a not in ids])
#
#                 else:
#                     axis = resample_axes[0]
#                     small_axis = np.argsort(last_shape)
#                     small_axis = [a for a in small_axis if a != axis][0]
#                     if axis < small_axis:
#                         axes = (axis, small_axis)
#                     else:
#                         axes = (small_axis, axis)
#                     last_shape = tuple([s if d not in axes else t for d, s, t in
#                                         zip(range(ndim), last_shape, sink_shape_in_source_orientation)])
#
#                     axes_order.append(axes)
#                     shape_order.append(last_shape)
#
#                     resample_axes = []
#
#             return axes_order, shape_order
#
#         elif axes_order == 'order':  # order axes according to array order for faster io
#
#             if isinstance(source, fl.Source):
#                 # FileList determine order according to file structure
#                 axes_list = source.axes_list
#                 # axes_file = source.axes_file;
#
#                 resample_axes = np.array(
#                     [d for d, s, t in zip(range(ndim), sink_shape_in_source_orientation, source_shape) if s != t])
#                 resample_factors = np.array(
#                     [float(t) / float(s) for s, t in zip(sink_shape_in_source_orientation, source_shape) if s != t])
#
#                 axes_order = []
#                 shape_order = []
#                 last_shape = source_shape
#
#                 # modify factors to account for file structure
#                 resample_factors_list = [f for a, f in zip(resample_axes, resample_factors) if a in axes_list]
#                 if len(resample_factors_list) > 0:
#                     max_resample_factor_list = np.max(resample_factors_list)
#                 else:
#                     max_resample_factor_list = 0
#                 resample_factors_sort = np.array([f if a in axes_list else f + max_resample_factor_list for a, f in
#                                                   zip(resample_axes, resample_factors)])
#                 # print(resample_factors_sort, resample_factors)
#
#                 while len(resample_axes) > 0:
#                     if len(resample_axes) >= 2:
#                         ids = np.argsort(resample_factors_sort)[-2:]
#
#                         axes = tuple(np.sort(resample_axes[ids]))
#                         last_shape = tuple([s if d not in axes else t for d, s, t in
#                                             zip(range(ndim), last_shape, sink_shape_in_source_orientation)])
#
#                         axes_order.append(axes)
#                         shape_order.append(last_shape)
#
#                         resample_axes = np.array([s for a, s in enumerate(resample_axes) if a not in ids])
#                         resample_factors = np.array([s for a, s in enumerate(resample_factors) if a not in ids])
#                         resample_factors_sort = np.array(
#                             [s for a, s in enumerate(resample_factors_sort) if a not in ids])
#                     else:
#                         axis = resample_axes[0]
#                         small_axis = np.argsort(last_shape)
#                         small_axis = [a for a in small_axis if a != axis][0]
#                         if axis < small_axis:
#                             axes = (axis, small_axis)
#                         else:
#                             axes = (small_axis, axis)
#                         last_shape = tuple([s if d not in axes else t for d, s, t in
#                                             zip(range(ndim), last_shape, sink_shape_in_source_orientation)])
#
#                         axes_order.append(axes)
#                         shape_order.append(last_shape)
#
#                         resample_axes = []
#
#                 return axes_order, shape_order
#             else:
#                 # not a FileList
#                 resample_axes = np.array(
#                     [d for d, s, t in zip(range(ndim), sink_shape_in_source_orientation, source_shape) if s != t])
#
#                 axes_order = []
#                 shape_order = []
#                 last_shape = source_shape
#                 while len(resample_axes) > 0:
#                     if len(resample_axes) >= 2:
#                         if order == 'C':
#                             slicing = slice(-2, None)
#                         else:
#                             slicing = slice(None, 2)
#                         axes = tuple(resample_axes[slicing])
#                     else:
#                         if order == 'C':
#                             axes = (resample_axes[0], axes_order[-1][0])
#                         else:
#                             axes = (axes_order[-1][1], resample_axes[0])
#
#                     axes_order.append(axes)
#                     last_shape = tuple([s if d not in axes else t for d, s, t in
#                                         zip(range(ndim), last_shape, sink_shape_in_source_orientation)])
#                     shape_order.append(last_shape)
#                     resample_axes = np.array([a for a in resample_axes if a not in axes])
#
#                 return axes_order, shape_order
#
#         else:
#             raise ValueError("axes_order not 'size','order' or list but %r!" % axes_order)
