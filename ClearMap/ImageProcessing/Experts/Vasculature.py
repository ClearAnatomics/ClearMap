# -*- coding: utf-8 -*-
"""
Vasculature
===========

Expert vasculature image processing pipeline.

This module provides the basic routines for processing vasculature data.
The routines are used in the :mod:`ClearMap.Scripts.TubeMap` pipeline.
"""
__author__ = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__ = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__ = 'https://idisco.info'
__download__ = 'https://www.github.com/ChristophKirst/ClearMap2'
  
import gc
import tempfile as tmpf

import numpy as np
import scipy.ndimage as ndi
import skimage.filters as skif

import ClearMap.IO.IO as io
from ClearMap.Utils.exceptions import MissingRequirementException

import ClearMap.ParallelProcessing.BlockProcessing as bp
import ClearMap.ParallelProcessing.DataProcessing.ArrayProcessing as ap

import ClearMap.ImageProcessing.Filter.Rank as rnk
import ClearMap.ImageProcessing.LocalStatistics as ls
import ClearMap.ImageProcessing.LightsheetCorrection as lc
import ClearMap.ImageProcessing.Differentiation.Hessian as hes
import ClearMap.ImageProcessing.Binary.Filling as bf
import ClearMap.ImageProcessing.Binary.Smoothing as bs

import ClearMap.Utils.Timer as tmr
from ClearMap.ImageProcessing.Experts.utils import initialize_sinks, run_step, print_params
from ClearMap.Utils.utilities import check_enough_temp_space, get_free_temp_space

###############################################################################
# ## Generic parameter
###############################################################################

MAX_BIN = 2**12
"""Number of intensity levels to use for the data after preprocessing.

Note
----
      * Higher values will increase the intensity resolution but slow down
        processing. 
      * 2**12 is a good choice for the vasculature data.
"""

DTYPE = 'uint16'
"""Data type for the data after preprocessing.

Note
----
      * The data type should fit numbers as big as the :const:`MAX_BIN` parameter. 
      * 'uint16' is a good choice for the vasculature data.
"""

BINARY_NAMES = ['High', 'Equalized', 'Deconvolved', 'Adaptive', 'Tube', 'Fill', 'Tracing']
"""Names for the multi-path binarization steps."""

BINARY_STATUS = {n: 2**k for k, n in enumerate(BINARY_NAMES)}
"""Binary representation for the multi-path binarization steps."""
     
###############################################################################
# ## Default parameter
###############################################################################

default_binarization_parameter = dict(
    # initial clipping and mask generation
    clip=dict(clip_range=(400, 60000),
              save=False),

    # lightsheet correction
    lightsheet=dict(percentile=0.25,
                    lightsheet=dict(selem=(150, 1, 1)),
                    background=dict(selem=(200, 200, 1),
                                    spacing=(25, 25, 1),
                                    step=(2, 2, 1),
                                    interpolate=1),
                    lightsheet_vs_background=2,
                    save=False),

    # median
    median=dict(selem=((3,)*3),
                save=False),

    # deconvolution
    deconvolve=dict(sigma=10,
                    threshold=750,
                    save=False),

    # equalization
    equalize=dict(percentile=(0.4, 0.975),
                  selem=(200, 200, 5),
                  spacing=(50, 50, 5),
                  interpolate=1,
                  threshold=1.1,
                  save=False),

    # adaptive threshold
    adaptive=dict(selem=(250, 250, 3),
                  spacing=(50, 50, 3),
                  interpolate=1,
                  save=False),

    # tubeness
    vesselize=dict(background=dict(selem=('disk', (30, 30, 1)),
                                   percentile=0.5),
                   tubeness=dict(sigma=1.0,
                                 gamma12=0.0),
                   threshold=120,
                   save=False),

    # fill
    fill=None,

    # smooth
    smooth=None,

    # controls
    binary_status=None,
    max_bin=MAX_BIN
)
"""Parameter for the vasculature binarization pipeline. 
See :func:`binarize` for details."""


default_binarization_processing_parameter = dict(
    size_max=40,
    size_min=5,
    overlap=0,
    axes=[2],
    optimization=True,
    optimization_fix='all',
    verbose=None,
    processes=None
)
"""Parallel processing parameter for the vasculature binarization pipeline. 
See :func:`ClearMap.ParallelProcessing.BlockProcessing.process`. for details."""       

                   
default_postprocessing_parameter = dict(
    # binary smoothing
    smooth=dict(iterations=6),

    # binary filling
    fill=True,

    # temporary file
    temporary_filename=None
)                   
"""Parameter for the postprocessing step of the binarized data.
See :func:`postprocess` for details."""


default_postprocessing_processing_parameter = dict(
    overlap=None,
    size_min=None,
    optimization=True,
    optimization_fix='all',
    as_memory=True
)
"""Parallel processing parameter for the vasculature postprocessing pipeline. 
See :func:`ClearMap.ParallelProcessing.BlockProcessing.process`. for details."""     


###############################################################################
# ## Binarization
###############################################################################
                   
def binarize(source, sink=None, binarization_parameter=default_binarization_parameter,
             processing_parameter=default_binarization_processing_parameter):
    """
    Multi-path binarization of iDISCO+ cleared vasculature data.

    Arguments
    ---------
    source : source specification
        The source of the stitched raw data.
    sink : sink specification or None
        The sink to write the result to. If None, an array is returned.
    binarization_parameter : dict
        Parameter for the binarization. See below for details.
    processing_parameter : dict
        Parameter for the parallel processing.
        See :func:`ClearMap.ParallelProcessing.BlockProcesing.process` for
        description of all the parameter.

    Returns
    -------
    sink : Source
        The result of the binarization.

    Notes
    -----
    * The binarization pipeline is composed of several steps. The parameters for
        each step are passed as sub-dictionaries to the binarization_parameter
        dictionary.

    * If None is passed for one of the steps this step is skipped.

    * Each step also has an additional parameter 'save' that enables saving of
        the result of that step to a file to inspect the pipeline.

    General parameter
    -----------------
    binary_status : str or None
        File name to save the information about which part of the multi-path
        binarization contributed to the final result.

    max_bin : int
        Number of intensity levels to use for the data after preprocessing.
        Higher values will increase the intensity resolution but slow down
        processing.

        For the vasculature a typical value is 2**12.

    Clipping
    --------
    clip : dict or None
        Clipping and mask generation step parameter.

        clip_range : tuple
            The range to clip the raw data as (lowest, highest)
            Voxels above lowest define the foreground mask used
            in the following steps.

            For the vasculature a typical value is (400,60000).

        save : str or None
            Save the result of this step to the specified file if not None.

    See also :mod:`ClearMap.ImageProcessing.Clipping.Clipping`

    Lightsheet correction
    ---------------------
    lightsheet : dict or None
        Lightsheet correction step parameter.

        percentile : float
            Percentile in [0,1] used to estimate the lightsheet artifact.

            For the vasculature a typical value is 0.25.

        lightsheet : dict
            Parameter for the ligthsheet artifact percentile estimation.
            See :func:`ClearMap.ImageProcessing.LightsheetCorrection.correct_lightsheet`
            for list of all parameters. The crucial parameter is

            selem : tuple
                The structural element shape used to estimate the stripe artifact.
                It should match the typical length, width, and depth of the artifact
                in the data.

                For the vasculature a typical value is (150,1,1).

        background : dict
            Parameter for the background estimation in the light sheet correction.
            See :func:`ClearMap.ImageProcessing.LightsheetCorrection.correct_lightsheet`
            for list of all parameters. The crucial parameters are

            selem : tuple
                The structural element shape used to estimate the background.
                It should be bigger than the largest vessels,

                For the vasculature a typical value is (200,200,1).

            spacing : tuple
                The spacing to use to estimate the background. Larger spacings speed up
                processing but become less local estimates.

                For the vasculature a typical value is (25,25,1)

            step : tuple
                This parameter enables to subsample from the entire array defined by
                the structural element using larger than single voxel steps.

                For the vasculature a typical value is (2,2,1).

            interpolate : int
                The order of the interpolation used in constructing the full
                background estimate in case a non-trivial spacing is used.

                For the vasculature a typical value is 1.

        lightsheet_vs_background : float
            The background is multiplied by this weight before comparing to the
            lightsheet artifact estimate.

            For the vasculature a typical value is 2.

        save : str or None
            Save the result of this step to the specified file if not None.

    Median filter
    -------------
    median : dict or None
        Median correction step parameter.
        See :func:`ClearMap.ImageProcessing.Filter.Rank.median` for all parameter.
        The important parameters are

        selem : tuple
            The structural element size for the median filter.

            For the vasculature a typical value is (3,3,3).

        save : str or None
            Save the result of this step to the specified file if not None.

    Pseudo Deconvolution
    --------------------
    deconvolve : dict
        The deconvolution step parameter.

        sigma : float
            The std of a Gaussian filter applied to the high intensity pixel image.
            The number should reflect the scale of the halo effect seen around high
            intensity structures.

            For the vasculature a typical value is 10.

        save : str or None
            Save the result of this step to the specified file if not None.

        threshold : float
            Voxels above this threshold will be added to the binarization result
            in the multi-path binarization.

            For the vasculature a typical value is 750.

    Adaptive Thresholding
    ---------------------
    adaptive : dict or None
        Adaptive thresholding step parameter.
        A local ISODATA threshold is estimated.
        See also :mod:`ClearMap.ImageProcessing.LocalStatistics`.

        selem : tuple
            The structural element size to estimate the percentiles.
            Should be larger than the larger vessels.

            For the vasculature a typical value is (200,200,5).

        spacing : tuple
            The spacing used to move the structural elements.
            Larger spacings speed up processing but become locally less precise.

            For the vasculature a typical value is (50,50,5)

        interpolate : int
            The order of the interpolation used in constructing the full
            background estimate in case a non-trivial spacing is used.

            For the vasculature a typical value is 1.

        save : str or None
            Save the result of this step to the specified file if not None.


    Equalization
    ------------
    equalize : dict or None
        Equalization step parameter.
        See also :func:`ClearMap.ImageProcessing.LocalStatistics.local_percentile`

        precentile : tuple
            The lower and upper percentiles used to estimate the equalization.
            The lower percentile is used for normalization, the upper to limit the
            maximal boost to a maximal intensity above this percentile.

            For the vasculature a typical value is (0.4, 0.975).

        max_value : float
            The maximal intensity value in the equalized image.

            For the vasculature a typical value is 1.5.

        selem : tuple
            The structural element size to estimate the percentiles.
            Should be larger than the larger vessels.

            For the vasculature a typical value is (200,200,5).

        spacing : tuple
            The spacing used to move the structural elements.
            Larger spacings speed up processing but become locally less precise.

            For the vasculature a typical value is (50,50,5)

        interpolate : int
            The order of the interpolation used in constructing the full
            background estimate in case a non-trivial spacing is used.

            For the vasculature a typical value is 1.

        save : str or None
            Save the result of this step to the specified file if not None.

        threshold : float
            Voxels above this threshold will be added to the binarization result
            in the multi-path binarization.

            For the vasculature a typical value is 1.1.

    Tube filter
    -----------
    vesselize : dict
        The tube filter step parameter.

        background : dict or None
            Parameters to correct for local background. See
            :func:`ClearMap.ImageProcessing.Filter.Rank.percentile`.
            If None, no background correction is done before the tube filter.

            selem : tuple
                The structural element specification to estimate the percentiles.
                Should be larger than the largest vessels intended to be
                boosted by the tube filter.

                For the vasculature a typical value is ('disk', (30,30,1)).

            percentile : float
                Percentile in [0,1] used to estimate the background.

                For the vasculature a typical value is 0.5.

        tubness : dict
            Parameters used for the tube filter. See
            :func:`ClearMap.ImageProcessing.Differentiation.Hessian.lambda123`.

            sigma : float
                The scale of the vessels to boos in the filter.

                For the vasculature a typical value is 1.0.

        save : str or None
            Save the result of this step to the specified file if not None.

        threshold : float
            Voxels above this threshold will be added to the binarization result
            in the multi-path binarization.

            For the vasculature a typical value is 120.

    Binary filling
    --------------
    fill : dict or None
        If not None, apply a binary filling the binarized result.

    For the vasculature this step is set to None and done globally
    in the postprocessing step.

    Binary smoothing
    ----------------
    smooth : dict or None
        The smoothing parameter passed to
        :func:`ClearMap.ImageProcessing.Binary.Smoothing.smooth_by_configuration`.

    For the vasculature this step is set to None and done globally
    in the postprocessing step.

    References
    ----------
    [1] C. Kirst et al., "Mapping the Fine-Scale Organization and Plasticity of the Brain Vasculature", Cell 180, 780 (2020)
    """

    # initialize sink
    shape = io.shape(source)
    order = io.order(source)
    sink, sink_buffer = ap.initialize_sink(sink=sink, shape=shape, order=order, dtype=bool)  # , memory='shared')

    # initialize addition output sinks
    binary_status = binarization_parameter.get('binary_status')
    if binary_status:
        ap.initialize_sink(binary_status, source=sink, shape=shape, order=order, dtype='uint16')

    initialize_sinks(binarization_parameter, shape, order)

    binarization_parameter.update(verbose=processing_parameter.get('verbose', False))

    bp.process(binarize_block, source, sink, function_type='block',
               parameter=binarization_parameter, **processing_parameter)

    return sink


def binarize_block(source, sink, parameter=default_binarization_parameter):
    """Binarize a Block."""

    # initialize parameter and slicings
    verbose = parameter.get('verbose', False)
    if verbose:
        prefix = 'Block %s: ' % (source.info(),)
        total_time = tmr.Timer(prefix)
    else:
        prefix = ''

    max_bin = parameter.get('max_bin', MAX_BIN)

    base_slicing = sink.valid.base_slicing
    valid_slicing = source.valid.slicing

    # initialize binary status for inspection
    binary_status = parameter.get('binary_status')
    if binary_status:
        binary_status = io.as_source(binary_status)
        binary_status = binary_status[base_slicing]

    default_step_params = {'parameter': parameter, 'steps_to_measure': {}, 'prefix': prefix,
                           'base_slicing': base_slicing, 'valid_slicing': valid_slicing}

    # clipping
    parameter_clip = parameter.get('clip')
    if parameter_clip:
        parameter_clip, timer = print_params(parameter_clip, 'clip', prefix, verbose)

        parameter_clip.update(norm=max_bin, dtype=DTYPE)
        save = parameter_clip.pop('save', None)

        clipped, mask, high, low = clip(source, **parameter_clip)
        not_low = np.logical_not(low)

        if save:
            save = io.as_source(save)
            save[base_slicing] = clipped[valid_slicing]

        if binary_status is not None:
            binary_status[high[valid_slicing]] += BINARY_STATUS['High']
        else:
            sink[valid_slicing] = high[valid_slicing]

        del high, low

        if verbose:
            timer.print_elapsed_time('Clipping')
    else:
        clipped = source
        mask = not_low = np.ones(source.shape, dtype=bool)
    # active arrays: clipped, mask, not_low

    # lightsheet correction
    corrected = run_step('lightsheet', clipped, lc.correct_lightsheet, remove_previous_result=True,
                         extra_kwargs={'mask': mask, 'max_bin': max_bin}, **default_step_params)
    # active arrays: corrected, mask, not_low

    # median filter
    median = run_step('median', corrected, rnk.median, remove_previous_result=True,
                      extra_kwargs={'max_bin': max_bin, 'mask': not_low}, **default_step_params)
    del not_low
    # active arrays: median, mask

    # pseudo deconvolution
    parameter_deconvolution = parameter.get('deconvolve')
    if parameter_deconvolution:
        parameter_deconvolution, timer = print_params(parameter_deconvolution, 'deconvolve', prefix, verbose)

        save = parameter_deconvolution.pop('save', None)
        threshold = parameter_deconvolution.pop('threshold', None)

        if binary_status is not None:
            binarized = binary_status > 0
        else:
            binarized = sink[:]
        deconvolved = deconvolve(median, binarized[:], **parameter_deconvolution)
        del binarized

        if save:
            save = io.as_source(save)
            save[base_slicing] = deconvolved[valid_slicing]

        if verbose:
            timer.print_elapsed_time('Deconvolution')

        if threshold:
            binary_deconvolved = deconvolved > threshold

            if binary_status is not None:
                binary_status[binary_deconvolved[valid_slicing]] += BINARY_STATUS['Deconvolved']
            else:
                sink[valid_slicing] += binary_deconvolved[valid_slicing]

            del binary_deconvolved

            if verbose:
                timer.print_elapsed_time('Deconvolution: binarization')
    else:
        deconvolved = median

    # active arrays: median, mask, deconvolved

    # adaptive
    parameter_adaptive = parameter.get('adaptive')
    adaptive = run_step('adaptive', deconvolved, threshold_adaptive, remove_previous_result=False,
                        **default_step_params)
    if parameter_adaptive:
        binary_adaptive = deconvolved > adaptive

        if binary_status is not None:
            binary_status[binary_adaptive[valid_slicing]] += BINARY_STATUS['Adaptive']
        else:
            sink[valid_slicing] += binary_adaptive[valid_slicing]

        del binary_adaptive, adaptive

        # if verbose:
        #     timer.print_elapsed_time('Adaptive')

    del deconvolved
    # active arrays: median, mask

    # equalize
    parameter_equalize = parameter.get('equalize')
    if parameter_equalize:
        parameter_equalize, timer = print_params(parameter_equalize, 'equalize', prefix, verbose)

        save = parameter_equalize.pop('save', None)
        threshold = parameter_equalize.pop('threshold', None)

        equalized = equalize(median, mask=mask, **parameter_equalize)

        if save:
            save = io.as_source(save)
            save[base_slicing] = equalized[valid_slicing]

        if verbose:
            timer.print_elapsed_time('Equalization')

        if threshold:
            binary_equalized = equalized > threshold

            if binary_status is not None:
                binary_status[binary_equalized[valid_slicing]] += BINARY_STATUS['Equalized']
            else:
                sink[valid_slicing] += binary_equalized[valid_slicing]

            # prepare equalized for use in vesselization
            parameter_vesselization = parameter.get('vesselize')
            if parameter_vesselization and parameter_vesselization.get('background'):
                equalized[binary_equalized] = threshold
                equalized = float(max_bin-1) / threshold * equalized

            del binary_equalized

            if verbose:
                timer.print_elapsed_time('Equalization: binarization')
    else:
        equalized = median

    del median
    # active arrays: mask, equalized

    # smaller vessels /capillaries
    parameter_vesselization = parameter.get('vesselize')
    if parameter_vesselization:
        parameter_vesselization, timer = print_params(parameter_vesselization, 'vesselize', prefix, verbose)

        parameter_background = parameter_vesselization.get('background')
        parameter_background = parameter_background.copy()
        if parameter_background:
            save = parameter_background.pop('save', None)

            equalized = np.array(equalized, dtype='uint16')
            background = rnk.percentile(equalized, max_bin=max_bin, mask=mask, **parameter_background)
            tubeness = equalized - np.minimum(equalized, background)

            del background

            if save:
                save = io.as_source(save)
                save[base_slicing] = tubeness[valid_slicing]
        else:
            tubeness = equalized

        parameter_tubeness = parameter_vesselization.get('tubeness', {})
        tubeness = tubify(tubeness, **parameter_tubeness)

        save = parameter_vesselization.get('save')
        if save:
            save = io.as_source(save)
            save[base_slicing] = tubeness[valid_slicing]

        if verbose:
            timer.print_elapsed_time('Vesselization')

        threshold = parameter_vesselization.get('threshold')
        if threshold:
            binary_vesselized = tubeness > threshold

            if binary_status is not None:
                binary_status[binary_vesselized[valid_slicing]] += BINARY_STATUS['Tube']
            else:
                sink[valid_slicing] += binary_vesselized[valid_slicing]

            del binary_vesselized

            if verbose:
                timer.print_elapsed_time('Vesselization: binarization')

        del tubeness

    del equalized, mask
    # active arrays: None

    # fill holes
    parameter_fill = parameter.get('fill')
    if parameter_fill:
        step_param, timer = print_params(parameter_fill, 'fill', prefix, verbose)

        if binary_status is not None:
            foreground = binary_status > 0
            filled = ndi.morphology.binary_fill_holes(foreground)
            binary_status[np.logical_and(filled, np.logical_not(foreground))] += BINARY_STATUS['Fill']
            del foreground, filled
        else:
            filled = ndi.morphology.binary_fill_holes(sink[:])
            sink[valid_slicing] += filled[valid_slicing]
            del filled

        if verbose:
            timer.print_elapsed_time('Filling')

    if binary_status is not None:
        sink[valid_slicing] = binary_status[valid_slicing] > 0

    # smooth binary
    if parameter.get('smooth'):  # WARNING: otherwise removes sink if no smoothing
        parameter['smooth']['save'] = False
        smoothed = run_step('smooth', sink, bs.smooth_by_configuration, remove_previous_result=False,
                            extra_kwargs={'sink': None, 'processes': 1}, **default_step_params)
        sink[valid_slicing] = smoothed[valid_slicing]
        del smoothed

    if verbose:
        total_time.print_elapsed_time('Binarization')

    gc.collect()


###############################################################################
# ## Postprocessing
###############################################################################

def postprocess(source, sink=None, postprocessing_parameter=default_postprocessing_parameter,
                processing_parameter=default_postprocessing_processing_parameter, processes=None, verbose=True):
    """
    Postprocess a binarized image.

    Arguments
    ---------
    source : source specification
        The binary  source.
    sink : sink specification or None
        The sink to write the postprocessed result to.
        If None, an array is returned.
    postprocessing_parameter : dict
        Parameter for the postprocessing.
    processing_parameter : dict
        Parameter for the parallel processing.
    processes: int or None
        Number of parallel processes to run. Defaults to max if None
    verbose : bool
        If True, print progress output.

    Returns
    -------
    sink : Source
        The result of the binarization.

    Notes
    -----
    * The postprocessing pipeline is composed of several steps. The parameters
        for each step are passed as sub-dictionaries to the
        postprocessing_parameter dictionary.

    * If None is passed for one of the steps the step is skipped.

    Smoothing
    ---------
    smooth : dict or None
        Smoothing step parameter. See
        :func:`ClearMap.ImageProcessing.Binary.Smoothing.smooth_by_configuration`

        iterations : int
            Number of smoothing iterations.
            For the vasculature a typical value is 6.

    Filling
    -------
    fill : bool or None
        If True, fill holes in the binary data.
    """

    source = io.as_source(source)
    sink = ap.initialize_sink(sink, shape=source.shape, dtype=source.dtype, order=source.order, return_buffer=False)

    if verbose:
        timer = tmr.Timer()
        print('Binary post processing: initialized.')

    postprocessing_parameter = postprocessing_parameter.copy()
    run_binary_filling = postprocessing_parameter.pop('fill', False)
    parameter_smooth = postprocessing_parameter.get('smooth')

    # smoothing
    if parameter_smooth:
        fill_source, tmp_f_path, save = apply_smoothing(source, sink, processing_parameter, postprocessing_parameter,
                                                        processes, verbose)
    else:
        fill_source = source
        save = False

    if run_binary_filling:
        bf.fill(fill_source, sink=sink, processes=processes, verbose=verbose)
        if parameter_smooth and not save:
            io.delete_file(tmp_f_path)

    if verbose:
        timer.print_elapsed_time('Binary post processing')

    gc.collect()


def apply_smoothing(source, sink, processing_parameter, postprocessing_parameter, processes=None, verbose=True):
    parameter_smooth = postprocessing_parameter.pop('smooth', None)
    if postprocessing_parameter.get('fill'):
        save = parameter_smooth.pop('save', False)
        # initialize temporary files if needed
        if not check_enough_temp_space():
            raise MissingRequirementException(f'Free space in temporary directory is insufficient, required 200 GB, '
                                              f'got {get_free_temp_space() // (2**30)} GB'
                                              f'Please free some space or use a different temporary directory.'
                                              f'You can set the "TMP" environment variable to a directory of your choice')
        tmp_f_path = save if save else postprocessing_parameter['temporary_filename']
        tmp_f_path = tmp_f_path if tmp_f_path else tmpf.mktemp(prefix='TubeMap_Vasculature_postprocessing',
                                                               suffix='.npy')
        sink_smooth = ap.initialize_sink(tmp_f_path, shape=source.shape, dtype=source.dtype,
                                         order=source.order, return_buffer=False)
    else:
        sink_smooth = sink
        save = False
        tmp_f_path = ''
    # run smoothing
    fill_source = bs.smooth_by_configuration(source, sink=sink_smooth, processing_parameter=processing_parameter,
                                             processes=processes, verbose=verbose, **parameter_smooth)
    return fill_source, tmp_f_path, save


###############################################################################
# ## Binarization processing steps
###############################################################################

def clip(source, clip_range=(300, 60000), norm=MAX_BIN, dtype=DTYPE):
    clip_low, clip_high = clip_range

    clipped = np.array(source[:], dtype=float)

    low = clipped < clip_low
    clipped[low] = clip_low

    high = clipped >= clip_high
    clipped[high] = clip_high

    mask = np.logical_not(np.logical_or(low, high))
    clipped -= clip_low
    clipped *= float(norm-1) / (clip_high - clip_low)
    clipped = np.asarray(clipped, dtype=dtype)
    return clipped, mask, high, low


def deconvolve(source, binarized, sigma=10):
    convolved = np.zeros(source.shape, dtype=float)
    convolved[binarized] = source[binarized]

    for z in range(convolved.shape[2]):
        convolved[:, :, z] = ndi.gaussian_filter(convolved[:, :, z], sigma=sigma)

    deconvolved = source - np.minimum(source, convolved)
    deconvolved[binarized] = source[binarized]
    return deconvolved


def threshold_isodata(source):
    try:
        thresholds = skif.threshold_isodata(source, return_all=True)
        if len(thresholds) > 0:
            return thresholds[-1]
        else:
            return 1
    except:  # FIXME: too broad
        return 1


def threshold_adaptive(source, function=threshold_isodata, selem=(100, 100, 3), spacing=(25, 25, 3),
                       interpolate=1, mask=None, step=None):
    source = io.as_source(source)[:]
    threshold = ls.apply_local_function(source, function=function, mask=mask, dtype=float,
                                        selem=selem, spacing=spacing, interpolate=interpolate, step=step)
    return threshold


def equalize(source, percentile=(0.5, 0.95), max_value=1.5, selem=(200, 200, 5), spacing=(50, 50, 5),
             interpolate=1, mask=None):
    equalized = ls.local_percentile(source, percentile=percentile, mask=mask, dtype=float,
                                    selem=selem, spacing=spacing, interpolate=interpolate)
    normalize = 1/np.maximum(equalized[..., 0], 1)
    maxima = equalized[..., 1]
    ids = maxima * normalize > max_value
    normalize[ids] = max_value / maxima[ids]
    equalized = np.array(source, dtype=float) * normalize
    return equalized


def tubify(source, sigma=1.0, gamma12=1.0, gamma23=1.0, alpha=0.25):
    return hes.lambda123(source=source, sink=None, sigma=sigma, gamma12=gamma12, gamma23=gamma23, alpha=alpha)


###############################################################################
# ## Helper
###############################################################################

def status_to_description(status):
    """
    Converts a status int to its description.

    Arguments
    ---------
    status : int
        The status.

    Returns
    -------
    description : str
        The description corresponding to the status.
    """
    description = ''
    for k in range(len(BINARY_NAMES)-1, -1, -1):
        if status / 2**k == 1:
            description = BINARY_NAMES[k] + ',' + description
            status -= 2**k
    if len(description) == 0:
        description = 'Background'
    else:
        description = description[:-1]
    return description


def binary_statistics(source):
    """
    Counts the binarization types.

    Arguments
    ---------
    source : array
        The status array of the binarization process.

    Returns
    -------
    statistics : dict
       A dict with entires {description : count}.
    """
    status, counts = np.unique(io.as_source(source)[:], return_counts=True)
    return {status_to_description(s): c for s, c in zip(status, counts)}


###############################################################################
# ## Tests
###############################################################################

def _test():
    """Tests."""
    import numpy as np
    import ClearMap.Visualization.Plot3d as p3d
    import ClearMap.Tests.Files as tsf
    import ClearMap.ImageProcessing.Experts.Vasculature as vasc

    source = np.array(tsf.source('vls')[:300, :300, 80:120])
    source[:, :, [0, -1]] = 0
    source[:, [0, -1], :] = 0
    source[[0, -1], :, :] = 0

    bpar = vasc.default_binarization_parameter.copy()
    bpar['clip']['clip_range'] = (150, 7000)
    bpar['as_memory'] = True
    # bpar['binary_status'] = 'binary_status.npy'

    ppar = vasc.default_processing_parameter.copy()
    ppar['processes'] = 10
    ppar['size_max'] = 10

    sink = 'binary.npy'
    # sink=None;

    binary = vasc.binarize(source, sink=sink, binarization_parameter=bpar, processing_parameter=ppar)
    p3d.plot([source, binary])

    import ClearMap.IO.IO as io
    io.delete_file(sink)

    pppar = vasc.default_postprocessing_parameter.copy()
    pppar['smooth']['iterations'] = 3
    smoothed = vasc.postprocess(binary, postprocessing_parameter=pppar)
    p3d.plot([binary, smoothed])
 
