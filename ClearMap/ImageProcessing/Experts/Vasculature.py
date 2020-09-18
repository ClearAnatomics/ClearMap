# -*- coding: utf-8 -*-
"""
Vasculature
===========

Expert vasculature image processing pipeline.

This module provides the basic routines for processing vasculature data.
The routines are used in the :mod:`ClearMap.Scripts.TubeMap` pipeline.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'
  
              
import numpy as np
import tempfile as tmpf 
import gc

import scipy.ndimage as ndi
import skimage.filters as skif

import ClearMap.IO.IO as io

import ClearMap.ParallelProcessing.BlockProcessing as bp
import ClearMap.ParallelProcessing.DataProcessing.ArrayProcessing as ap

import ClearMap.ImageProcessing.Filter.Rank as rnk
import ClearMap.ImageProcessing.LocalStatistics as ls
import ClearMap.ImageProcessing.LightsheetCorrection as lc
import ClearMap.ImageProcessing.Differentiation.Hessian as hes
import ClearMap.ImageProcessing.Binary.Filling as bf
import ClearMap.ImageProcessing.Binary.Smoothing as bs

import ClearMap.Utils.Timer as tmr
import ClearMap.Utils.HierarchicalDict as hdict

###############################################################################
### Generic parameter
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

BINARY_STATUS = {n : 2**k for k,n in enumerate(BINARY_NAMES)}
"""Binary representation for the multi-path binarization steps."""
     
###############################################################################
### Default parameter
###############################################################################

default_binarization_parameter = dict(
  #initial clipping and mask generation
  clip = dict(clip_range = (400,60000), 
              save = False),

  #lightsheet correction    
  lightsheet = dict(percentile = 0.25, 
                    lightsheet = dict(selem = (150,1,1)),
                    background = dict(selem = (200,200,1),
                                      spacing = (25,25,1), 
                                      step = (2,2,1),
                                      interpolate = 1),
                    lightsheet_vs_background = 2, 
                    save = False),

  #median
  median = dict(selem = ((3,)*3),
                save = False),
 
  #deconvolution
  deconvolve = dict(sigma = 10,
                    threshold = 750,
                    save = False),
                        
  #equalization
  equalize = dict(percentile = (0.4, 0.975), 
                  selem = (200,200,5), 
                  spacing = (50,50,5), 
                  interpolate = 1,
                  threshold = 1.1,
                  save = False),
  
                  
  #adaptive threshold
  adaptive = dict(selem = (250,250,3),
                  spacing = (50,50,3), 
                  interpolate=1,
                  save = False),


  #tubeness
  vesselize = dict(background = dict(selem = ('disk', (30,30,1)), 
                                     percentile = 0.5),
                   tubeness = dict(sigma = 1.0, 
                                   gamma12 = 0.0),
                   threshold = 120,
                   save = False),
  
  #fill
  fill = None,
  
  #smooth
  smooth = None,

  #controls
  binary_status = None,
  max_bin = MAX_BIN
)
"""Parameter for the vasculature binarization pipeline. 
See :func:`binarize` for details."""


default_binarization_processing_parameter = dict(
  size_max = 40,
  size_min = 5,
  overlap = 0,
  axes = [2],
  optimization = True,
  optimization_fix = 'all',
  verbose = None,
  processes = None
)
"""Parallel processing parameter for the vasculature binarization pipeline. 
See :func:`ClearMap.ParallelProcessing.BlockProcessing.process`. for details."""       

                   
default_postprocessing_parameter = dict(
  #binary smoothing
  smooth = dict(iterations=6),
  
  #binary filling
  fill = True,
  
  #temporary file
  temporary_filename = None
)                   
"""Parameter for the postprocessing step of the binarized data.
See :func:`postprocess` for details."""


default_postprocessing_processing_parameter = dict(
  overlap = None,
  size_min = None,
  optimization = True,
  optimization_fix = 'all',
  as_memory = True,
)
"""Parallel processing parameter for the vasculature postprocessing pipeline. 
See :func:`ClearMap.ParallelProcessing.BlockProcessing.process`. for details."""     


###############################################################################
### Binarization
###############################################################################
                   
def binarize(source, sink = None, binarization_parameter = default_binarization_parameter, processing_parameter = default_binarization_processing_parameter):
  """Multi-path binarization of iDISCO+ cleared vasculature data.
  
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
  verbose : bool
    If True, print progress output.
  
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
      Voxels above lowest define the foregournd mask used 
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
      Percentile in [0,1] used to estimate the lightshieet artifact.
      
      For the vasculature a typical value is 0.25.
      
    lightsheet : dict
      Parameter for the ligthsheet artifact percentile estimation. 
      See :func:`ClearMap.ImageProcessing.LightsheetCorrection.correct_lightsheet`
      for list of all parameters. The crucial parameter is
      
      selem : tuple
        The structural element shape used to estimate the stripe artifact.
        It should match the typical lenght, width, and depth of the artifact 
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
        The order of the interpoltation used in constructing the full 
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
      
      For the vascualture a typical value is (3,3,3).
    
    save : str or None
      Save the result of this step to the specified file if not None.  
  
  Pseudo Deconvolution
  --------------------
  deconvolve : dict
    The deconvolution step parameter.
    
    sigma : float
      The std of a Gaussina filter applied to the high intensity pixel image.
      The number should reflect the scale of the halo effect seen around high
      intensity structures.
      
      For the vasculature a typical value is 10.
    
    save : str or None
      Save the result of this step to the specified file if not None.   
      
    threshold : float 
      Voxels above this threshold will be added to the binarization result
      in the multi-path biniarization.
      
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
      The order of the interpoltation used in constructing the full 
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
      The order of the interpoltation used in constructing the full 
      background estimate in case a non-trivial spacing is used.
      
      For the vasculature a typical value is 1.
      
    save : str or None
      Save the result of this step to the specified file if not None.   
      
    threshold : float 
      Voxels above this threshold will be added to the binarization result
      in the multi-path biniarization.
      
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
        
        For the vasculature a typical value is ('disk', (30,30,1)) .

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
      in the multi-path biniarization.
      
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
    
  #initialize sink
  shape = io.shape(source);
  order = io.order(source);
  sink, sink_buffer = ap.initialize_sink(sink=sink, shape=shape, order=order, dtype=bool); #, memory='shared');
  
  #initialize addition output sinks  
  binary_status = binarization_parameter.get('binary_status', None);
  if binary_status:
    ap.initialize_sink(binary_status, source=sink, shape=shape, order=order, dtype='uint16');

  for key in binarization_parameter.keys():
    par = binarization_parameter[key];
    if isinstance(par, dict):
      filename = par.get('save', None);
      if filename:
        ap.initialize_sink(filename, shape=shape, order=order, dtype='float');
        
  binarization_parameter.update(verbose=processing_parameter.get('verbose', False));
  
  bp.process(binarize_block, source, sink, function_type='block', parameter=binarization_parameter, **processing_parameter)                   
  
  return sink;                


def binarize_block(source, sink, parameter = default_binarization_parameter):
  """Binarize a Block."""
  
  #initialize parameter and slicings
  verbose = parameter.get('verbose', False);
  if verbose:
    prefix = 'Block %s: ' % (source.info(),);
    total_time = tmr.Timer(prefix);
  
  max_bin = parameter.get('max_bin', MAX_BIN);
    
  base_slicing = sink.valid.base_slicing;
  valid_slicing = source.valid.slicing;
  
  #initialize binary status for inspection
  binary_status = parameter.get('binary_status', None);
  if binary_status:
    binary_status = io.as_source(binary_status);
    binary_status = binary_status[base_slicing];
  
  
  #clipping
  parameter_clip = parameter.get('clip', None);        
  if parameter_clip:
    parameter_clip = parameter_clip.copy();
    if verbose:
      timer = tmr.Timer(prefix);
      hdict.pprint(parameter_clip, head = prefix + 'Clipping:')
    parameter_clip.update(norm=max_bin, dtype=DTYPE);
    save = parameter_clip.pop('save', None);   
                          
    clipped, mask, high, low = clip(source, **parameter_clip);  
    not_low = np.logical_not(low);
    
    if save:
      save = io.as_source(save);
      save[base_slicing] = clipped[valid_slicing];
     
    if binary_status is not None:
      binary_status[high[valid_slicing]] += BINARY_STATUS['High']
    else:
      sink[valid_slicing] = high[valid_slicing]; 

    del high, low
    
    if verbose:
      timer.print_elapsed_time('Clipping');                          
  
  else:
    clipped = source
    mask = not_low = np.ones(source.shape, dtype=bool);
    #high = low = np.zeros(source.shape, dtype=bool);
    #low = np.zeros(source.shape, dtype=bool);
    #not_low = np.logical_not(low); 
  
  #active arrays: clipped, mask, not_low

                       
  #lightsheet correction
  parameter_lightsheet = parameter.get('lightsheet', None);                                
  if parameter_lightsheet:
    parameter_lightsheet = parameter_lightsheet.copy();
    if verbose:
      timer = tmr.Timer(prefix);
      hdict.pprint(parameter_lightsheet, head = prefix + 'Lightsheet:')                        
                                    
    #parameter_lightsheet.update(max_bin=max_bin);
    save = parameter_lightsheet.pop('save', None);                      
    
    corrected = lc.correct_lightsheet(clipped, mask=mask, max_bin=max_bin, **parameter_lightsheet);
           
    if save:
      save = io.as_source(save);
      save[base_slicing] = corrected[valid_slicing];
    
    if verbose:
      timer.print_elapsed_time('Lightsheet');                           
  else:
    corrected = clipped;
 
  del clipped
  #active arrays: corrected, mask, not_low
  
  #median filter
  parameter_median = parameter.get('median', None);
  if parameter_median:
    parameter_median = parameter_median.copy();
    if verbose:
      timer = tmr.Timer(prefix);
      hdict.pprint(parameter_median, head = prefix + 'Median:')    
    
    save = parameter_median.pop('save', None);      
    median = rnk.median(corrected, max_bin=max_bin, mask=not_low, **parameter_median);                              
    
    if save:
      save = io.as_source(save);
      save[base_slicing] = median[valid_slicing];
    
    if verbose:
      timer.print_elapsed_time('Median');   
    
  else:
    median = corrected;
  
  del corrected, not_low;
  #active arrays: median, mask
  
  #pseudo deconvolution
  parameter_deconvolution = parameter.get('deconvolve', None);
  if parameter_deconvolution:
    parameter_deconvolution = parameter_deconvolution.copy();
    if verbose:
      timer = tmr.Timer(prefix);
      hdict.pprint(parameter_deconvolution, head = prefix + 'Deconvolution:')
    
    save = parameter_deconvolution.pop('save', None);
    threshold = parameter_deconvolution.pop('threshold', None);  
   
    if binary_status is not None:
      binarized = binary_status > 0;
    else:
      binarized = sink[:];
    deconvolved = deconvolve(median, binarized[:], **parameter_deconvolution)
    del binarized
    
    if save:
      save = io.as_source(save);
      save[base_slicing] = deconvolved[valid_slicing];
    
    if verbose:
      timer.print_elapsed_time('Deconvolution');   
  
    if threshold:
      binary_deconvolved = deconvolved > threshold;
      
      if binary_status is not None:
        binary_status[binary_deconvolved[valid_slicing]] += BINARY_STATUS['Deconvolved'];
      else:
        sink[valid_slicing] += binary_deconvolved[valid_slicing];
    
      del binary_deconvolved
      
      if verbose:
        timer.print_elapsed_time('Deconvolution: binarization');   
  
  else:  
    deconvolved = median;
  
  #active arrays: median, mask, deconvolved
  
  #adaptive
  parameter_adaptive = parameter.get('adaptive', None);
  if parameter_adaptive:
    parameter_adaptive = parameter_adaptive.copy();
    if verbose:
      timer = tmr.Timer(prefix);
      hdict.pprint(parameter_adaptive, head = prefix + 'Adaptive:')    
    
    save = parameter_adaptive.pop('save', None);
    
    adaptive = threshold_adaptive(deconvolved, **parameter_adaptive)
    
    if save:
      save = io.as_source(save);
      save[base_slicing] = adaptive[valid_slicing];
    
    binary_adaptive = deconvolved > adaptive;
    
    if binary_status is not None:
      binary_status[binary_adaptive[valid_slicing]] += BINARY_STATUS['Adaptive'];
    else:
      sink[valid_slicing] += binary_adaptive[valid_slicing];
    
    del binary_adaptive, adaptive;
      
    if verbose:
      timer.print_elapsed_time('Adaptive');   
  
  del deconvolved
  #active arrays: median, mask
    
  # equalize 
  parameter_equalize = parameter.get('equalize', None);
  if parameter_equalize: 
    parameter_equalize = parameter_equalize.copy();
    if verbose:
      timer = tmr.Timer(prefix);
      hdict.pprint(parameter_equalize, head = prefix + 'Equalization:')    
    
    save = parameter_equalize.pop('save', None);
    threshold = parameter_equalize.pop('threshold', None);  

    equalized = equalize(median, mask=mask, **parameter_equalize);
        
    if save:
      save = io.as_source(save);
      save[base_slicing] = equalized[valid_slicing];
    
    if verbose:
      timer.print_elapsed_time('Equalization');
        
    if threshold:
      binary_equalized = equalized > threshold
      
      if binary_status is not None:
        binary_status[binary_equalized[valid_slicing]] += BINARY_STATUS['Equalized'];
      else:
        sink[valid_slicing] += binary_equalized[valid_slicing];
      
      #prepare equalized for use in vesselization
      parameter_vesselization = parameter.get('vesselize', None);
      if parameter_vesselization and parameter_vesselization.get('background', None):  
        equalized[binary_equalized] = threshold;
        equalized = float(max_bin-1) / threshold * equalized;
      
      del binary_equalized
    
      if verbose:
        timer.print_elapsed_time('Equalization: binarization');
  else:
    equalized = median;
  
  del median
  #active arrays: mask, equalized
    
  # smaller vessels /capilarries
  parameter_vesselization = parameter.get('vesselize', None);
  if parameter_vesselization:
    parameter_vesselization = parameter_vesselization.copy();
    if verbose:
      timer = tmr.Timer(prefix);
      hdict.pprint(parameter_vesselization, head = prefix + 'Vesselization:')    
    
    parameter_background = parameter_vesselization.get('background', None)
    parameter_background = parameter_background.copy();
    if parameter_background:
      save = parameter_background.pop('save', None);
      
      equalized = np.array(equalized, dtype = 'uint16');
      background = rnk.percentile(equalized, max_bin=max_bin, mask=mask, **parameter_background);
      tubeness = equalized - np.minimum(equalized, background);  
      
      del background                  
      
      if save:
        save = io.as_source(save);
        save[base_slicing] = tubeness[valid_slicing];
    
    else:
      tubeness = equalized;
    
    parameter_tubeness = parameter_vesselization.get('tubeness', {})
    tubeness = tubify(tubeness, **parameter_tubeness);
    
    save = parameter_vesselization.get('save', None);
    if save:
      save = io.as_source(save);
      save[base_slicing] = tubeness[valid_slicing];
       
    if verbose:
      timer.print_elapsed_time('Vesselization');  
      
    threshold = parameter_vesselization.get('threshold', None);
    if threshold:      
      binary_vesselized = tubeness > threshold;
      
      if binary_status is not None:
        binary_status[binary_vesselized[valid_slicing]] += BINARY_STATUS['Tube'];
      else:
        sink[valid_slicing] += binary_vesselized[valid_slicing];
      
      del binary_vesselized 
      
      if verbose:
        timer.print_elapsed_time('Vesselization: binarization');   
    
    del tubeness
  
  del equalized, mask
  #active arrays: None
  
  #fill holes  
  parameter_fill = parameter.get('fill', None);
  if parameter_fill:
    parameter_fill = parameter_fill.copy();
    if verbose:
      timer = tmr.Timer(prefix);
      #hdict.pprint(parameter_fill, head = 'Filling:')    
      
    if binary_status is not None:
      foreground = binary_status > 0;
      filled = ndi.morphology.binary_fill_holes(foreground);
      binary_status[np.logical_and(filled, np.logical_not(foreground))] += BINARY_STATUS['Fill'];
      del foreground, filled
    else:
      filled = ndi.morphology.binary_fill_holes(sink[:]);
      sink[valid_slicing] += filled[valid_slicing]; 
      del filled
    
    if verbose:
      timer.print_elapsed_time('Filling');   

  if binary_status is not None:
    sink[valid_slicing] = binary_status[valid_slicing] > 0;

  #smooth binary  
  parameter_smooth = parameter.get('smooth', None);
  if parameter_smooth:
    parameter_smooth = parameter_smooth.copy();
    if verbose:
      timer = tmr.Timer(prefix);
      hdict.pprint(parameter_smooth, head = prefix + 'Smoothing:')    
    
    smoothed = bs.smooth_by_configuration(sink, sink=None, processes=1, **parameter_smooth);
    sink[valid_slicing] = smoothed[valid_slicing]; 
    del smoothed;
      
    if verbose:
      timer.print_elapsed_time('Smoothing');  
   
  if verbose:
    total_time.print_elapsed_time('Binarization')
  
  gc.collect()
  
  return None;

###############################################################################
### Postprocessing
###############################################################################   

def postprocess(source, sink = None, postprocessing_parameter = default_postprocessing_parameter, processing_parameter = default_postprocessing_processing_parameter, processes = None, verbose = True):
  """Postprocess a binarized image.
  
  Arguments
  ---------
  source : source specification
    The binary  source.
  sink : sink specification or None
    The sink to write the postprocesses result to. 
    If None, an array is returned.
  postprocessing_parameter : dict
    Parameter for the postprocessing.
  processing_parameter : dict
    Parameter for the parallel processing.
  verbose : bool
    If True, print progress output.
  
  Returns
  -------
  sink : Source
    The result of the binarization.
    
  Notes
  -----
  * The postporcessing pipeline is composed of several steps. The parameters
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
  
  source = io.as_source(source);  
  sink   = ap.initialize_sink(sink, shape=source.shape, dtype=source.dtype, order=source.order, return_buffer=False);
  
  if verbose:
    timer = tmr.Timer();
    print('Binary post processing: initialized.');
  
  postprocessing_parameter = postprocessing_parameter.copy();
  parameter_smooth = postprocessing_parameter.pop('smooth', None);
  parameter_fill   = postprocessing_parameter.pop('fill', None);
  #print(parameter_smooth, parameter_fill)
  
  #smoothing
  save = None;
  if parameter_smooth:
    #intialize temporary files if needed
    if parameter_fill:
      save = parameter_smooth.pop('save', None);
      temporary_filename = save; 
      if temporary_filename is None:
        temporary_filename = postprocessing_parameter['temporary_filename'];
      if temporary_filename is None:
        temporary_filename = tmpf.mktemp(prefix='TubeMap_Vasculature_postprocessing', suffix='.npy');
      sink_smooth   = ap.initialize_sink(temporary_filename, shape=source.shape, dtype=source.dtype, order=source.order, return_buffer=False);
    else:
      sink_smooth = sink;
    
    #run smoothing
    source_fill = bs.smooth_by_configuration(source, sink=sink_smooth, processing_parameter=processing_parameter, processes=processes, verbose=verbose, **parameter_smooth);
  
  else:
    source_fill = source;
  
  if parameter_fill:
    sink = bf.fill(source_fill, sink=sink, processes=processes, verbose=verbose);
    
    if parameter_smooth and save is None:
      io.delete_file(temporary_filename);
  else:
    sink = source_fill;
  
  if verbose:
    timer.print_elapsed_time('Binary post processing');
  
  gc.collect()
  return None;
  #return sink;
  

###############################################################################
### Binarization processing steps
###############################################################################

def clip(source, clip_range = (300, 60000), norm = MAX_BIN, dtype = DTYPE):
  clip_low, clip_high = clip_range;
  
  clipped = np.array(source[:], dtype = float);
  
  low = clipped < clip_low
  clipped[low] = clip_low;
  
  high = clipped >= clip_high; 
  clipped[high] = clip_high;
  
  mask = np.logical_not(np.logical_or(low, high));
  clipped -= clip_low;
  clipped *= float(norm-1) / (clip_high - clip_low);
  clipped = np.asarray(clipped, dtype=dtype);            
  return clipped, mask, high, low


def deconvolve(source, binarized, sigma = 10):
  convolved = np.zeros(source.shape, dtype = float);
  convolved[binarized] = source[binarized];
  
  for z in range(convolved.shape[2]):
    convolved[:,:,z] = ndi.gaussian_filter(convolved[:,:,z], sigma=sigma);
  
  deconvolved = source - np.minimum(source, convolved);
  deconvolved[binarized] = source[binarized]; 
  return deconvolved;


def threshold_isodata(source):
  try:
    thresholds = skif.threshold_isodata(source, return_all=True);
    if len(thresholds) > 0:
      return thresholds[-1];
    else:
      return 1;
  except:
    return 1;


def threshold_adaptive(source, function = threshold_isodata, selem = (100,100,3), spacing = (25,25,3), interpolate = 1, mask = None, step = None):
  source = io.as_source(source)[:];
  threshold = ls.apply_local_function(source, function=function, mask=mask, dtype=float, selem=selem, spacing=spacing, interpolate=interpolate, step = step);
  return threshold


def equalize(source, percentile = (0.5, 0.95), max_value = 1.5, selem = (200,200,5), spacing = (50,50,5), interpolate = 1, mask = None):
  equalized = ls.local_percentile(source, percentile=percentile, mask=mask, dtype=float, selem=selem, spacing=spacing, interpolate=interpolate);
  normalize = 1/np.maximum(equalized[...,0], 1);
  maxima = equalized[...,1];
  ids = maxima * normalize > max_value;
  normalize[ids] = max_value / maxima[ids];
  equalized = np.array(source, dtype = float) * normalize;                          
  return equalized;


def tubify(source, sigma = 1.0, gamma12 = 1.0, gamma23 = 1.0, alpha = 0.25):
  return hes.lambda123(source=source, sink=None, sigma=sigma, gamma12=gamma12, gamma23=gamma23, alpha=alpha);


###############################################################################
### Helper
###############################################################################

def status_to_description(status):
  """Converts a status int to its description.
  
  Arguments
  ---------
  status : int
    The status.
  
  Returns
  -------
  description : str
    The description corresponding to the status.
  """
  description = '';
  for k in range(len(BINARY_NAMES)-1,-1,-1):
    if status / 2**k == 1:
      description = BINARY_NAMES[k] + ',' + description;
      status -= 2**k;
  if len(description) == 0:
    description = 'Background';
  else:
    description = description[:-1];
  return description;     


def binary_statistics(source):
  """Counts the binarization types.
  
  Arguments
  ---------
  source : array
    The status array of the binarization process.
  
  Returns
  -------
  statistics : dict
    A dict with entires {description : count}.
  """
  status, counts = np.unique(io.as_source(source)[:], return_counts = True);
  return {status_to_description(s) : c for s,c in zip(status, counts)}


###############################################################################
### Tests
###############################################################################

def _test():
  """Tests."""
  import numpy as np
  import ClearMap.Visualization.Plot3d as p3d
  import ClearMap.Tests.Files as tsf
  import ClearMap.ImageProcessing.Experts.Vasculature as vasc
  
  source = np.array(tsf.source('vls')[:300,:300,80:120]);
  source[:,:,[0,-1]] = 0;
  source[:,[0,-1],:] = 0;
  source[[0,-1],:,:] = 0;
    
  bpar = vasc.default_binarization_parameter.copy();
  bpar['clip']['clip_range'] = (150, 7000)
  bpar['as_memory'] = True
  #bpar['binary_status'] = 'binary_status.npy'
  
  ppar = vasc.default_processing_parameter.copy();
  ppar['processes'] = 10;
  ppar['size_max'] = 10;
  
  sink='binary.npy'
  #sink=None;
  
  binary = vasc.binarize(source, sink=sink, binarization_parameter=bpar, processing_parameter = ppar) 
  p3d.plot([source, binary])

  import ClearMap.IO.IO as io
  io.delete_file(sink)
  
  pppar = vasc.default_postprocessing_parameter.copy();
  pppar['smooth']['iterations'] = 3;
  smoothed = vasc.postprocess(binary, postprocessing_parameter=pppar)
  p3d.plot([binary, smoothed])
 