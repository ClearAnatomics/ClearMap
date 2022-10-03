# -*- coding: utf-8 -*-
"""
Cells
=====

Expert cell image processing pipeline.

This module provides the basic routines for processing immmediate eaerely
gene data. 

The routines are used in the :mod:`ClearMap.Scripts.CellMap` pipeline.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'
      

import numpy as np
import tempfile as tmpf 
import gc            

import cv2
import scipy.ndimage as ndi
import scipy.ndimage.filters as ndf
import skimage.filters as skif

import ClearMap.IO.IO as io

import ClearMap.ParallelProcessing.BlockProcessing as bp
import ClearMap.ParallelProcessing.DataProcessing.ArrayProcessing as ap

import ClearMap.ImageProcessing.IlluminationCorrection as ic
import ClearMap.ImageProcessing.Filter.StructureElement as se
import ClearMap.ImageProcessing.Filter.FilterKernel as fk
import ClearMap.ImageProcessing.LocalStatistics as ls

import ClearMap.Analysis.Measurements.MaximaDetection as md
import ClearMap.Analysis.Measurements.ShapeDetection as sd
import ClearMap.Analysis.Measurements.MeasureExpression as me

#import ClearMap.ImageProcessing.Filter.Rank as rnk

#import ClearMap.ImageProcessing.LightsheetCorrection as lc
#import ClearMap.ImageProcessing.Differentiation.Hessian as hes
#import ClearMap.ImageProcessing.Binary.Filling as bf
#import ClearMap.ImageProcessing.Binary.Smoothing as bs

import ClearMap.Utils.Timer as tmr

import ClearMap.Utils.HierarchicalDict as hdict


###############################################################################
### Default parameter
###############################################################################

default_cell_detection_parameter = dict( 
  #flatfield
  iullumination_correction = dict(flatfield = None,
                                  scaling = 'mean'),
                       
  #background removal
  background_correction = dict(shape = (10,10),
                               form = 'Disk',
                               save = False),
  
  #equalization
  equalization = None,
  
  #difference of gaussians filter
  dog_filter = dict(shape = None,
                    sigma = None,
                    sigma2 = None),
  
  #extended maxima detection
  maxima_detection = dict(h_max = None,
                          shape = 5,
                          threshold = 0,
                          valid = True,
                          save = False),

  #cell shape detection                                  
  shape_detection = dict(threshold = 700,
                         save = False),
  
  #cell intenisty detection                   
  intensity_detection = dict(method = 'max',
                             shape = 3,
                             measure = ['source', 'background']), 
)
"""Parameter for the cell detectrion pipeline. 
See :func:`detect_cells` for details."""


default_cell_detection_processing_parameter = dict(
  size_max = 100,
  size_min = 50,
  overlap = 32,
  axes = [2],
  optimization = True,
  optimization_fix = 'all',
  verbose = None,
  processes = None
)
"""Parallel processing parameter for the cell detection pipeline. 
See :func:`ClearMap.ParallelProcessing.BlockProcessing.process` for details."""       


###############################################################################
### Cell detection
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
    See :func:`ClearMap.ParallelProcessing.BlockProcesing.process` for 
    description of all the parameter.
  verbose : bool
    If True, print progress output.
  
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
    * cell intensity and size measurements via: :func:`~ClearMap.ImageProcessing.Measurements.ShapeDetection.find_intensity`,
      :func:`~ClearMap.ImageProcessing.Measurements.ShapeDetection.find_size`. 

  
  The parameters for each step are passed as sub-dictionaries to the 
    cell_detection_parameter dictionary.
  
  * If None is passed for one of the steps this step is skipped.
  
  * Each step also has an additional parameter 'save' that enables saving of 
    the result of that step to a file to inspect the pipeline.
  
  
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
    
    precentile : tuple
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
      The order of the interpoltation used in constructing the full 
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
       If None, detemined automatically from shape.
    
    sigma2 : tuple or None
       The std of the outer Gaussian.
       If None, detemined automatically from shape.
    
    save : str or None
      Save the result of this step to the specified file if not None.
  
  
  Maxima detection
  ----------------
  maxima_detection : dict or None
    Extended maxima detection step parameter.

    h_max : float or None
      The 'height'for the extended maxima.
      If None, simple local maxima detection isused.

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
      Cell shape is expanded from maxima if pixles are above this threshold
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
    
  #initialize sink
  shape = io.shape(source);
  order = io.order(source);
  
  for key in cell_detection_parameter.keys():
    par = cell_detection_parameter[key];
    if isinstance(par, dict):
      filename = par.get('save', None);
      if filename:
        ap.initialize_sink(filename, shape=shape, order=order, dtype='float');
        
  cell_detection_parameter.update(verbose=processing_parameter.get('verbose', False));

  results, blocks = bp.process(detect_cells_block, source, sink=None, function_type='block', return_result=True,
                               return_blocks=True, parameter=cell_detection_parameter, workspace=workspace,
                               **processing_parameter)
  
  #merge results
  results = np.vstack([np.hstack(r) for r in results])
  
  #create column headers
  header = ['x','y','z'];
  dtypes = [int, int, int];
  if cell_detection_parameter['shape_detection'] is not None:
    header += ['size'];
    dtypes += [int];
  measures = cell_detection_parameter['intensity_detection']['measure'];
  header +=  measures
  dtypes += [float] * len(measures)

  dt = {'names' : header, 'formats' : dtypes};
  cells = np.zeros(len(results), dtype=dt);
  for i,h in enumerate(header):
    cells[h] = results[:,i];
  
  #save results  
  return io.write(sink, cells);


def detect_cells_block(source, parameter = default_cell_detection_parameter):
  """Detect cells in a Block."""
  
  #initialize parameter and slicings
  verbose = parameter.get('verbose', False);
  if verbose:
    prefix = 'Block %s: ' % (source.info(),);
    total_time = tmr.Timer(prefix);
  
  base_slicing = source.valid.base_slicing;
  valid_slicing = source.valid.slicing;
  valid_lower = source.valid.lower;
  valid_upper = source.valid.upper;
  lower = source.lower;
  
  parameter_intensity  = parameter.get('intensity_detection', None);
  measure_to_array = dict();
  if parameter_intensity: 
    parameter_intensity = parameter_intensity.copy();
    measure = parameter_intensity.pop('measure', []);
    if measure is None:
      measure = [];
    for m in measure:
      measure_to_array[m] = None;

  if 'source' in measure_to_array:
    measure_to_array['source'] = source;
  
  
  # correct illumination
  parameter_illumination = parameter.get('illumination_correction', None);        
  if parameter_illumination:
    parameter_illumination = parameter_illumination.copy();
    if verbose:
      timer = tmr.Timer(prefix);
      hdict.pprint(parameter_illumination, head=prefix + 'Illumination correction')
    save = parameter_illumination.pop('save', None);   
           
    corrected = ic.correct_illumination(source, **parameter_illumination)               
    
    if save:
      save = io.as_source(save);
      save[base_slicing] = corrected[valid_slicing];
    
    if verbose:
      timer.print_elapsed_time('Illumination correction');   
  else:
   corrected = np.array(source.array);           
  
  if 'illumination' in measure_to_array:
    measure_to_array['illumination'] = corrected; 
  
  #background subtraction
  parameter_background = parameter.get('background_correction', None);        
  if parameter_background:
    parameter_background = parameter_background.copy();
    if verbose:
      timer = tmr.Timer(prefix);
      hdict.pprint(parameter_background, head = prefix + 'Background removal')
    save = parameter_background.pop('save', None);   
           
    background = remove_background(corrected, **parameter_background)               
    
    if save:
      save = io.as_source(save);
      save[base_slicing] = background[valid_slicing];
    
    if verbose:
      timer.print_elapsed_time('Illumination correction');                          
  else:
    background = corrected;
    
  del corrected;

  if 'background' in measure_to_array:
    measure_to_array['background'] = background; 
  
  # equalize 
  parameter_equalize = parameter.get('equalization', None);
  if parameter_equalize: 
    parameter_equalize = parameter_equalize.copy();
    if verbose:
      timer = tmr.Timer(prefix);
      hdict.pprint(parameter_equalize, head = prefix + 'Equalization:')    
    
    save = parameter_equalize.pop('save', None);
    
    equalized = equalize(background, mask=None, **parameter_equalize);
        
    if save:
      save = io.as_source(save);
      save[base_slicing] = equalized[valid_slicing];
    
    if verbose:
      timer.print_elapsed_time('Equalization');
  
  else:
    equalized = background;

  del background;

  if 'equalized' in measure_to_array:
    measure_to_array['equalized'] = equalized;
  
    
  #DoG filter
  parameter_dog_filter = parameter.get('dog_filter', None);
  if parameter_dog_filter: 
    parameter_dog_filter = parameter_dog_filter.copy();
    if verbose:
      timer = tmr.Timer(prefix);
      hdict.pprint(parameter_dog_filter, head = prefix + 'DoG filter:')    
    
    save = parameter_dog_filter.pop('save', None);
    
    dog = dog_filter(equalized, **parameter_dog_filter);
        
    if save:
      save = io.as_source(save);
      save[base_slicing] = dog[valid_slicing];
    
    if verbose:
      timer.print_elapsed_time('DoG filter');
  
  else:
    dog = equalized;

  del equalized;
  
  if 'dog' in measure_to_array:
    measure_to_array['dog'] = dog;
  
  
  #Maxima detection
  parameter_maxima     = parameter.get('maxima_detection', None);
  parameter_shape      = parameter.get('shape_detection', None);
  
  if parameter_shape or parameter_intensity:
    if not parameter_maxima:
      print(prefix + 'Warning: maxima detection needed for shape and intensity detection!');
      parameter_maxima = dict();
  
  if parameter_maxima: 
    parameter_maxima = parameter_maxima.copy();
    if verbose:
      timer = tmr.Timer(prefix);
      hdict.pprint(parameter_maxima, head = prefix + 'Maxima detection:')    
    
    save = parameter_maxima.pop('save', None);
    valid = parameter_maxima.pop('valid', None);
    
    # extended maxima
    maxima = md.find_maxima(dog, **parameter_maxima, verbose=verbose);
  
    if save:
      save = io.as_source(save);
      save[base_slicing] = maxima[valid_slicing];
    
    #center of maxima
    if parameter_maxima['h_max']:
      centers = md.find_center_of_maxima(source, maxima=maxima, verbose=verbose);
    else:
      centers = ap.where(maxima).array;

    if verbose:
      timer.print_elapsed_time('Maxima detection');
  
    #correct for valid region
    if valid:
      ids = np.ones(len(centers), dtype=bool);
      for c,l,u in zip(centers.T, valid_lower, valid_upper):
        ids = np.logical_and(ids, np.logical_and(l <= c, c < u));
      centers = centers[ids];
      del ids;
  
  del maxima;
  
  results = (centers,);
  
  #cell shape detection
  if parameter_shape: 
    parameter_shape = parameter_shape.copy();
    if verbose:
      timer = tmr.Timer(prefix);
      hdict.pprint(parameter_shape, head = prefix + 'Shape detection:')    
    
    save = parameter_shape.pop('save', None);
    
    # shape detection
    shape = sd.detect_shape(dog, centers, **parameter_shape, verbose=verbose);

    if save:
      save = io.as_source(save);
      save[base_slicing] = shape[valid_slicing];

    #size detection
    max_label = centers.shape[0];
    sizes = sd.find_size(shape, max_label=max_label);
    valid = sizes > 0;
    
    if verbose:
      timer.print_elapsed_time('Shape detection');
      
    results += (sizes,);
  
  else:
    valid = None;
    shape = None;
    
  del dog;

  #cell intensity detection
  if parameter_intensity: 
    parameter_intensity = parameter_intensity.copy();
    if verbose:
      timer = tmr.Timer(prefix);
      hdict.pprint(parameter_intensity, head = prefix + 'Intensity detection:')    
    
    if not shape is None:
      r = parameter_intensity.pop('shape', 3);
      if isinstance(r, tuple):
        r = r[0];
    
    for m in measure:
      if shape is not None:
        intensity = sd.find_intensity(measure_to_array[m], label=shape, max_label=max_label, **parameter_intensity);
      else:
        intensity = me.measure_expression(measure_to_array[m], centers, search_radius=r, **parameter_intensity, processes=1, verbose=False)
      
      results += (intensity,)
    
    if verbose:
      timer.print_elapsed_time('Shape detection');
  
  if valid is not None:
    results = tuple(r[valid] for r in results);
  
  #correct coordinate offsets of blocks
  results = (results[0] + lower,) + results[1:];
  
  #correct shapes for merging
  results = tuple(r[:,None] if r.ndim == 1 else r for r in results);
  
  if verbose:
    total_time.print_elapsed_time('Cell detection')
  
  gc.collect()
  
  return results;


###############################################################################
### Cell detection processing steps
###############################################################################

def remove_background(source, shape, form = 'Disk'):
  selem = se.structure_element(shape, form=form, ndim=2)
  selem = np.array(selem).astype('uint8')
  removed = np.empty(source.shape, dtype=source.dtype)
  for z in range(source.shape[2]):
    #img[:,:,z] = img[:,:,z] - grey_opening(img[:,:,z], structure = structureElement('Disk', (30,30)));
    #img[:,:,z] = img[:,:,z] - morph.grey_opening(img[:,:,z], structure = self.structureELement('Disk', (150,150)));
    removed[:,:,z] = source[:,:,z] - np.minimum(source[:,:,z], cv2.morphologyEx(source[:,:,z], cv2.MORPH_OPEN, selem))
  return removed; 


def equalize(source, percentile = (0.5, 0.95), max_value = 1.5, selem = (200,200,5), spacing = (50,50,5), interpolate = 1, mask = None):
  equalized = ls.local_percentile(source, percentile=percentile, mask=mask, dtype=float, selem=selem, spacing=spacing, interpolate=interpolate);
  normalize = 1/np.maximum(equalized[...,0], 1);
  maxima = equalized[...,1];
  ids = maxima * normalize > max_value;
  normalize[ids] = max_value / maxima[ids];
  equalized = np.array(source, dtype = float) * normalize;                          
  return equalized;


def dog_filter(source, shape, sigma = None, sigma2 = None):
  if not shape is None:
    fdog = fk.filter_kernel(ftype='dog', shape=shape, sigma=sigma, sigma2=sigma2);
    fdog = fdog.astype('float32');
    filtered = ndf.correlate(source, fdog);
    filtered[filtered < 0] = 0;
    return filtered
  else:
    return source;


def detect_maxima(source, h_max = None, shape = 5, threshold = None, verbose = False):
  # extended maxima
  maxima = md.find_maxima(source, h_max=h_max, shape=shape, threshold=threshold, verbose=verbose);
  
  #center of maxima
  if h_max:
    centers = md.find_center_of_maxima(source, maxima=maxima, verbose=verbose);
  else:
    centers = ap.where(maxima).array;
  
  return centers;


###############################################################################
### Cell filtering
###############################################################################


def filter_cells(source, sink, thresholds):
  """Filter a array of detected cells according to the thresholds.
  
  Arguments
  ---------
  source : str, array or Source
    The source for the cell data.
  sink : str, array or Source
    The sink for the results.
  thresholds : dict
    Dictionary of the form {name : threshold} where name refers to the 
    column in the cell data and threshold can be None, a float 
    indicating a minimal threshold or a tuple (min,max) where min,max can be
    None or a minimal and maximal threshold value.
  
  Returns
  -------
  sink : str, array or Source
    The thresholded cell data.
  """
  source = io.as_source(source);
  
  ids = np.ones(source.shape[0], dtype=bool);
  for k,t in thresholds.items():
    if t:
      if not isinstance(t, (tuple, list)):
        t = (t, None);
      if t[0] is not None:
        ids = np.logical_and(ids, t[0] <= source[k])
      if t[1] is not None:
        ids = np.logical_and(ids, t[1] > source[k]);
  cells_filtered = source[ids];

  return io.write(sink, cells_filtered)


###############################################################################
### Tests
###############################################################################

def _test():
  """Tests."""
  import numpy as np
  import ClearMap.Visualization.Plot3d as p3d
  import ClearMap.Tests.Files as tsf
  import ClearMap.ImageProcessing.Experts.Cells as cells
