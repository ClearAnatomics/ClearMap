# -*- coding: utf-8 -*-
"""
Cells
=====

Expert cell image processing pipeline.

This module provides the basic routines for processing immediate early
gene data. 

The routines are used in the :mod:`ClearMap.Scripts.CellMap` pipeline.
"""
__author__ = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__ = 'https://idisco.info'
__download__ = 'https://github.com/ClearAnatomics/ClearMap'

import gc
import multiprocessing
import warnings

import numpy as np

import cv2
import scipy.ndimage.filters as ndf
import scipy.ndimage as ndi

import ClearMap.IO.IO as clearmap_io

import ClearMap.ParallelProcessing.BlockProcessing as bp
import ClearMap.ParallelProcessing.DataProcessing.ArrayProcessing as ap

import ClearMap.ImageProcessing.IlluminationCorrection as ic
import ClearMap.ImageProcessing.Filter.StructureElement as se
import ClearMap.ImageProcessing.Filter.FilterKernel as fk

import ClearMap.Analysis.Measurements.maxima_detection as md
import ClearMap.Analysis.Measurements.shape_detection as sd
import ClearMap.Analysis.Measurements.MeasureExpression as me

import ClearMap.Utils.Timer as tmr
from ClearMap.ImageProcessing.Experts.utils import initialize_sinks, run_step, print_params
from ClearMap.ImageProcessing.LocalStatistics import local_percentile

from ClearMap.Utils.exceptions import ClearMapValueError
###############################################################################
# ## Default parameter
###############################################################################

default_cell_detection_parameter = dict(
    # flatfield
    illumination_correction=dict(flatfield=None,
                                 scaling='mean'),

    # background removal
    background_correction=dict(shape=(10, 10),
                               form='Disk',
                               save=False),

    # equalization
    equalization=None,

    # difference of gaussians filter
    dog_filter=dict(shape=None,
                    sigma=None,
                    sigma2=None),

    # extended maxima detection
    maxima_detection=dict(h_max=None,
                          shape=5,
                          threshold=0,
                          valid=True,
                          save=False),

    # cell shape detection
    shape_detection=dict(threshold=700,
                         save=False),

    # cell intensity detection
    intensity_detection=dict(method='max',
                             shape=3,
                             measure=['source', 'background_correction']),
)
"""Parameter for the cell detection pipeline. 
See :func:`detect_cells` for details."""


default_cell_detection_processing_parameter = dict(
    size_max=100,
    size_min=50,
    overlap=32,
    axes=[2],
    optimization=True,
    optimization_fix='all',
    verbose=None,
    processes=None
)
"""Parallel processing parameter for the cell detection pipeline. 
See :func:`ClearMap.ParallelProcessing.BlockProcessing.process` for details."""       


###############################################################################
# ## Cell detection
###############################################################################
                   
def detect_cells(source, sink=None, cell_detection_parameter=default_cell_detection_parameter,
                 processing_parameter=default_cell_detection_processing_parameter, workspace=None):
    """Cell detection pipeline.

    Arguments
    ---------
    source : source specification
        The source of the stitched raw data.
    sink : sink specification or None
        The sink to write the result to. If None, an array is returned.
    cell_detection_parameter : dict
        Parameter for the binarization. See below for details.
    processing_parameter : dict
        Parameter for the parallel processing.
        See :func:`ClearMap.ParallelProcessing.BlockProcessing.process` for
        description of all the parameter.
    workspace: Workspace
        The optional workspace object to have a handle to cancel the multiprocess

    Returns
    -------
    sink : Source
        The result of the cell detection.

    Notes
    -----
    Effectively this function performs the following steps:
        * illumination correction via :func:`~ClearMap.ImageProcessing.IlluminationCorrection.correct_illumination`
        * background removal
        * difference of Gaussians (DoG) filter
        * maxima detection via :func:`~ClearMap.Analysis.Measurements.MaximaDetection.find_extended_maxima`
        * cell shape detection via :func:`~ClearMap.Analysis.Measurements.ShapeDetection.detect_shape`
        * cell intensity and size measurements via:
            :func:`~ClearMap.ImageProcessing.Measurements.ShapeDetection.find_intensity`,
            :func:`~ClearMap.ImageProcessing.Measurements.ShapeDetection.find_size`.


    The parameters for each step are passed as sub-dictionaries to the
    cell_detection_parameter dictionary.

    * If None is passed for one of the steps this step is skipped.
    * Each step also has an additional parameter 'save' that enables saving
        of the result of that step to a file to inspect the pipeline.


    Illumination correction
    -----------------------
    illumination_correction : dict or None
        Illumination correction step parameter.

        flatfield : array or str
            The flat field estimate for the image planes.

        background : array or None
            A background level to assume for the flatfield correction.

        scaling : float, 'max', 'mean' or None
            Optional scaling after the flat field correction.

        save : str or None
            Save the result of this step to the specified file if not None.

    See also :func:`ClearMap.ImageProcessing.IlluminationCorrection.correct_illumination`


    Background removal
    ------------------
    background_correction : dict or None
        Background removal step parameter.

        shape : tuple
            The shape of the structure element to estimate the background.
            This should be larger than the typical cell size.

        form : str
            The form of the structure element (e.g. 'Disk')

        save : str or None
            Save the result of this step to the specified file if not None.

    Equalization
    ------------
    equalization : dict or None
        Equalization step parameter.
        See also :func:`ClearMap.ImageProcessing.LocalStatistics.local_percentile`

        percentile : tuple
            The lower and upper percentiles used to estimate the equalization.
            The lower percentile is used for normalization, the upper to limit the
            maximal boost to a maximal intensity above this percentile.

        max_value : float
            The maximal intensity value in the equalized image.

        selem : tuple
            The structural element size to estimate the percentiles.
            Should be larger than the larger vessels.

        spacing : tuple
            The spacing used to move the structural elements.
            Larger spacings speed up processing but become locally less precise.

        interpolate : int
            The order of the interpolation used in constructing the full
            background estimate in case a non-trivial spacing is used.

        save : str or None
          Save the result of this step to the specified file if not None.


    DoG Filter
    ----------
    dog_filter : dict or None
        Difference of Gaussian filter step parameter.

        shape : tuple
            The shape of the filter.
            This should be near the typical cell size.

        sigma : tuple or None
             The std of the inner Gaussian.
             If None, determined automatically from shape.

        sigma2 : tuple or None
             The std of the outer Gaussian.
             If None, determined automatically from shape.

        save : str or None
            Save the result of this step to the specified file if not None.


    Maxima detection
    ----------------
    maxima_detection : dict or None
        Extended maxima detection step parameter.

        h_max : float or None
            The 'height' for the extended maxima.
            If None, simple local maxima detection is used.

        shape : tuple
            The shape of the structural element for extended maxima detection.
            This should be near the typical cell size.

        threshold : float or None
            Only maxima above this threshold are detected. If None, all maxima
            are detected.

        valid : bool
            If True, only detect cell centers in the valid range of the blocks with
            overlap.

        save : str or None
          Save the result of this step to the specified file if not None.


    Shape detection
    ---------------
    shape_detection : dict or None
        Shape detection step parameter.

        threshold : float
            Cell shape is expanded from maxima if pixels are above this threshold
            and not closer to another maxima.

        save : str or None
          Save the result of this step to the specified file if not None.


    Intensity detection
    -------------------
    intensity_detection : dict or None
        Intensity detection step parameter.

        method : {'max'|'min','mean'|'sum'}
            The method to use to measure the intensity of a cell.

        shape : tuple or None
            If no cell shapes are detected a disk of this shape is used to measure
            the cell intensity.

        save : str or None
            Save the result of this step to the specified file if not None.

    References
    ----------
    [1] Renier, Adams, Kirst, Wu et al., "Mapping of Brain Activity by Automated Volume Analysis of Immediate Early Genes.", Cell 165, 1789 (2016)
    [1] Kirst et al., "Mapping the Fine-Scale Organization and Plasticity of the Brain Vasculature", Cell 180, 780 (2020)
    """

    # initialize sink
    shape = clearmap_io.shape(source)
    order = clearmap_io.order(source)

    initialize_sinks(cell_detection_parameter, shape, order)

    cell_detection_parameter.update(verbose=processing_parameter.get('verbose', False))

    n_processes = multiprocessing.cpu_count() if processing_parameter.get('processes') is None else processing_parameter.get('processes')
    n_threads = int(multiprocessing.cpu_count() / n_processes)  # Number of threads so that * n_processes, fills CPUs

    results, blocks = bp.process(detect_cells_block, source, sink=None, function_type='block', return_result=True,
                                 return_blocks=True, parameter=cell_detection_parameter, workspace=workspace,
                                 **{**processing_parameter, **{'n_threads': n_threads}})

    # merge results
    results = np.vstack([np.hstack(r) for r in results])

    # create column headers  # FIXME: use pd.DataFrame instead
    header = ['x', 'y', 'z']
    dtypes = [int, int, int]
    if cell_detection_parameter['shape_detection'] is not None:
        header += ['size']
        dtypes += [int]
    measures = cell_detection_parameter['intensity_detection']['measure']
    header += measures
    dtypes += [float] * len(measures)

    dt = {'names': header, 'formats': dtypes}
    cells = np.zeros(len(results), dtype=dt)
    for i, h in enumerate(header):
        cells[h] = results[:, i]

    # save results
    return clearmap_io.write(sink, cells)


def detect_cells_block(source, parameter=default_cell_detection_parameter, n_threads=None):
    """Detect cells in a Block."""

    # initialize parameter and slicing
    if parameter.get('verbose'):
        prefix = 'Block %s: ' % (source.info(),)
        total_time = tmr.Timer(prefix)
    else:
        prefix = ''

    base_slicing = source.valid.base_slicing
    valid_slicing = source.valid.slicing
    valid_lower = source.valid.lower
    valid_upper = source.valid.upper
    lower = source.lower

    steps_to_measure = {}  # FIXME: rename
    parameter_intensity = parameter.get('intensity_detection')
    if parameter_intensity:
        parameter_intensity = parameter_intensity.copy()
        measure = parameter_intensity.pop('measure', [])
        measure = measure if measure else []
        valid_measurement_keys = list(default_cell_detection_parameter.keys()) + ['source']
        for m in measure:
            if m not in valid_measurement_keys:
                raise KeyError(f'Unknown measurement: {m}')
            steps_to_measure[m] = None

    if 'source' in steps_to_measure:
        steps_to_measure['source'] = source

    default_step_params = {'parameter': parameter, 'steps_to_measure': steps_to_measure, 'prefix': prefix,
                           'base_slicing': base_slicing, 'valid_slicing': valid_slicing}

    # WARNING: if param_illumination: previous_step = source, not np.array(source.array)
    corrected = run_step('illumination_correction', np.array(source.array),
                         ic.correct_illumination, **default_step_params)

    background = run_step('background_correction', corrected, remove_background,
                          remove_previous_result=True, **default_step_params)

    equalized = run_step('equalization', background, equalize, remove_previous_result=True,
                         extra_kwargs={'mask': None}, **default_step_params)

    dog = run_step('dog_filter', equalized, dog_filter,  # TODO: DoG filter != .title()
                   remove_previous_result=True, **default_step_params)

    # Maxima detection
    parameter_maxima = parameter.get('maxima_detection')
    parameter_shape = parameter.get('shape_detection')

    if parameter_shape or parameter_intensity:
        if not parameter_maxima:
            print(f'{prefix}Warning: maxima detection needed for shape and intensity detection!')
            parameter_maxima = {}

    if parameter_maxima:
        valid = parameter_maxima.pop('valid', None)
        maxima = run_step('maxima_detection', dog, md.find_maxima, extra_kwargs={'verbose': parameter.get('verbose')},
                          remove_previous_result=False, **default_step_params)
        # center of maxima
        if parameter_maxima['h_max']:  # FIXME: check if source or dog
            centers = md.find_center_of_maxima(source, maxima=maxima, verbose=parameter.get('verbose'))
        else:
            if parameter_shape:
                threshold = parameter_shape.get('threshold',0)
                print(f"masking maxima centers by (dog > {threshold}) for shape detection")
                mask = dog > threshold
                maxima = maxima * mask

            # We treat the eventuality of connected components of size>1 in the mask (maxima>0);
            # the choice of the structure matrix for connectivity could need discussion.
            # with no further adaptation the maxima_labeling consumes too much memory
            maxima_labels, _ = ndi.label(maxima, structure=np.ones((3,)*3,dtype='bool'))
            centers = np.vstack(md.label_representatives(maxima_labels)).transpose()
            # we could come back to the ancient version
            # centers = ap.where(maxima, processes=n_threads).array 
        del maxima

        # correct for valid region
        if valid:
            print('Filtering centers for correct block processing.')
            ids = np.ones(len(centers), dtype=bool)
            for c, l, u in zip(centers.T, valid_lower, valid_upper):
                ids = np.logical_and(ids, np.logical_and(l <= c, c < u))
            centers = centers[ids]
            del ids
        results = (centers,)
    else:
        results = ()

    # cell shape detection  # FIXME: may use centers without assignment
    if parameter_shape:
        try:
            parser = (lambda t: t[0]>0)
            shape, sizes = run_step('shape_detection', dog, sd.detect_shape, remove_previous_result=True, **default_step_params,
            args = [centers], presave_parser=parser, extra_kwargs={'verbose': parameter.get('verbose'), 'processes': n_threads, 'return_sizes': True})
        except ClearMapValueError as err:
            if str(err) == 'An uint array with 0 values will lead to inconsistent results, consider a histogram transform or dtype conversion.':
                warnings.warn('This block is likely to contain corrupted data, an empty output will be provided for this block.')
                results = (centers[:0],)
                sizes = np.array([])
                shape = None
            else:
                raise err
                          

            
        valid = sizes > 0
        results += (sizes,)
    else:
        valid = None
        shape = None

    # cell intensity detection
    if parameter_intensity:
        parameter_intensity, timer = print_params(parameter_intensity, 'intensity_detection', prefix,
                                                  parameter.get('verbose'))

        if shape is not None:
            r = parameter_intensity.pop('shape', 3)
            if isinstance(r, tuple):
                r = r[0]

        for m in measure:
            if shape is not None:
                max_label = centers.shape[0]
                intensity = sd.find_intensity(steps_to_measure[m], label=shape,
                                              max_label=max_label, **parameter_intensity)
            else:  # WARNING: prange but me.measure_expression not parallel since processes=1
                # FIXME : How can r be defined in this branch ???
                intensity = me.measure_expression(steps_to_measure[m], centers, search_radius=r,
            
                                                    **parameter_intensity, processes=1, verbose=False)

            results += (intensity,)

        if parameter.get('verbose'):
            timer.print_elapsed_time('Shape detection')

    if valid is not None:
        results = tuple(r[valid] for r in results)
    # correct coordinate offsets of blocks
    results = (results[0] + lower,) + results[1:]
    # correct shapes for merging
    results = tuple(r[:, None] if r.ndim == 1 else r for r in results)

    if parameter.get('verbose'):
        total_time.print_elapsed_time('Cell detection')
  
    gc.collect()

    return results


###############################################################################
# ## Cell detection processing steps
###############################################################################

def remove_background(source, shape, form='Disk'):
    selem = se.structure_element(shape, form=form, ndim=2)  # FIXME: use skimage kernel
    selem = np.array(selem).astype('uint8')
    removed = np.empty(source.shape, dtype=source.dtype)
    for z in range(source.shape[2]):
        # img[:,:,z] = img[:,:,z] - grey_opening(img[:,:,z], structure=structureElement('Disk', (30,30)))
        # img[:,:,z] = img[:,:,z] - morph.grey_opening(img[:,:,z], structure=structureElement('Disk', (150,150)))
        removed[:, :, z] = source[:, :, z] - np.minimum(source[:, :, z],
                                                        cv2.morphologyEx(source[:, :, z], cv2.MORPH_OPEN, selem))
    return removed


def dog_filter(source, shape, sigma=None, sigma2=None):
    if shape is not None:
        fdog = fk.filter_kernel(ftype='dog', shape=shape, sigma=sigma, sigma2=sigma2)
        fdog = fdog.astype('float32')
        filtered = ndf.correlate(source, fdog)
        filtered[filtered < 0] = 0
        return filtered
    else:
        return source


def detect_maxima(source, h_max=None, shape=5, threshold=None, verbose=False):  # FIXME: use to refactor
    # extended maxima
    maxima = md.find_maxima(source, h_max=h_max, shape=shape, threshold=threshold, verbose=verbose)

    # center of maxima
    if h_max:
        centers = md.find_center_of_maxima(source, maxima=maxima, verbose=verbose)
    else:
        centers = ap.where(maxima).array  # FIXME: prange

    return centers


def equalize(source, percentile=(0.5, 0.95), max_value=1.5, selem=(200, 200, 5), spacing=(50, 50, 5),
             interpolate=1, mask=None):
    equalized = local_percentile(source, percentile=percentile, mask=mask, dtype=float,
                                 selem=selem, spacing=spacing, interpolate=interpolate)
    normalize = 1/np.maximum(equalized[..., 0], 1)
    maxima = equalized[..., 1]
    ids = maxima * normalize > max_value
    normalize[ids] = max_value / maxima[ids]
    equalized = np.array(source, dtype=float) * normalize
    return equalized


###############################################################################
# ## Cell filtering
###############################################################################


def filter_cells(source, sink, thresholds):
    """
    Filter an array of detected cells according to the thresholds.

    Arguments
    ---------
    source : str, array or Source
        The source for the cell data.
    sink : str, array or Source
        The sink for the results.
    thresholds : dict
        Dictionary of the form {name : threshold} where name refers to the
        column in the cell data and threshold can be None, a float
        indicating a minimal threshold or a tuple (min, max) where min, max can be
        None or a minimal and maximal threshold value.

    Returns
    -------
    sink : str, array or Source
        The thresholded cell data.
    """
    source = clearmap_io.as_source(source)

    ids = np.ones(source.shape[0], dtype=bool)
    for filter_name, thrsh in thresholds.items():
        if thrsh:
            if not isinstance(thrsh, (tuple, list)) and isinstance(thrsh, (int, float)):
                thrsh = (thrsh, None)
            if thrsh[0] is not None:  # high pass
                ids = np.logical_and(ids, thrsh[0] <= source[filter_name])
            if thrsh[1] is not None:  # low pass
                ids = np.logical_and(ids, thrsh[1] > source[filter_name])
    cells_filtered = source[ids]

    return clearmap_io.write(sink, cells_filtered)


###############################################################################
# ## Tests
###############################################################################

def _test():
    """Tests."""
