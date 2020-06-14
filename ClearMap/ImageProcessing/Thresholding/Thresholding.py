"""
Thresholding
============

This module contains vairous thresholding routines, including 
hysteresis thresholding.

Module towards smart thresholding routines.
"""
__author__    = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__   = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__copyright__ = 'Copyright (c) 2017 by Christoph Kirst, The Rockefeller University, New York City'

import os
import numpy as np

import ClearMap.IO.IO as io;

import pyximport;
#pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)

old_get_distutils_extension = pyximport.pyximport.get_distutils_extension

def new_get_distutils_extension(modname, pyxfilename, language_level=None):
    extension_mod, setup_args = old_get_distutils_extension(modname, pyxfilename, language_level)
    extension_mod.language='c++'
    return extension_mod,setup_args

pyximport.pyximport.get_distutils_extension = new_get_distutils_extension

pyximport.install(setup_args = {"include_dirs" : [np.get_include(), os.path.dirname(os.path.abspath(__file__))]},
                  reload_support=True)


from . import ThresholdingCode as code


###############################################################################
### Hysteresis thresholding
###############################################################################

def threshold(source, sink = None, threshold = None, hysteresis_threshold = None, seeds = None, background = None):
    """Hysteresis thresholding.
    
    Arguments
    ---------
    source : array
      Input source.
    sink : array or None
      If None, a new array is allocated.
    threshold : float
      The threshold for the intial seeds.
    hysteresis_threshold : float or None
      The hysteresis threshold to extend the initial seeds. 
      If None, no hysteresis thresholding is performed.
    seeds : array or None
      The seeds from which to start the hysteresis thresholds 
    background : array or None
      Exclude this area from the hysteresis thresholding.
    
    Returns
    -------
    sink : array
        Thresholded output.
    """
    if threshold is None and seeds is None:
      raise ValueError('The threshold and seeds cannot both be None!');
    
    if source is sink:
      raise NotImplementedError("Cannot perform operation in place.")

    if sink is None:
      sink = np.zeros(source.shape, dtype = 'int8', order = io.order(source));
    
    source_flat = source.reshape(-1, order=io.order(source));
    sink_flat   = sink.reshape(-1, order=io.order(sink));                           
    strides     = np.array(io.element_strides(source));
     
    if seeds is None:
      seeds = np.where(source_flat >= threshold)[0];
    else:
      seeds_flat = seeds.reshape(-1, order=io.order(seeds));
      seeds = np.where(seeds_flat)[0];
    
    if hysteresis_threshold is not None:
      parameter_index = np.zeros(0, dtype = int);
      parameter_double = np.array([hysteresis_threshold], dtype = float);   

      if background is not None:
        background_flat = background.reshape(-1, order=io.order(background));                                     
        background_flat = background_flat.view(dtype='uint8')                                
        code.threshold_to_background(source_flat, sink_flat, background_flat, strides, seeds, parameter_index, parameter_double);
      else:
        code.threshold(source_flat, sink_flat, strides, seeds, parameter_index, parameter_double);
    
    else:
      sink_flat[seeds] = 1;  
    
    return sink;


###############################################################################
### Tests
###############################################################################

def _test():
  import numpy as np
  import ClearMap.Visualization.Plot3d as p3d
  import ClearMap.ImageProcessing.Thresholding.Thresholding as th    
   
  r = np.arange(50);        
  x,y,z = np.meshgrid(r,r,r);
  x,y,z = [i - 25 for i in (x,y,z)];                     
  d = np.exp(-(x*x + y*y +z*z)/10.0**2)                  
               
  t = th.threshold(d, sink = None, threshold = 0.9, hysteresis_threshold=0.5);               
  p3d.plot([[d,t]])     
        
