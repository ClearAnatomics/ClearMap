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

from ClearMap.IO import source_geometry
import ClearMap.Utils.array_checks as ac

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


# dtypes of source_t and sink_t in ThresholdingCode.pyx (bool arrays are viewed as uint8)
_DTYPES = (np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.float32, np.float64)
_SINK_DTYPES = (np.int8,) + _DTYPES


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
      The threshold for the initial seeds.
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

    # The kernels work on flat arrays with the strides of the source: every flat array (source,
    # sink, seeds, background) must be flattened in the same order, that of the source.
    order = source_geometry.order(source)  # keep the layout of contiguous sources
    if order not in ('C', 'F'):  # non contiguous: copy to ClearMap's default (Fortran) order
      order = 'F'
    source = np.asarray(source, order=order)  # copy only if not contiguous (read-only is fine)

    if sink is None:
      sink = np.zeros(source.shape, dtype='int8', order=order)
    if sink.shape != source.shape:
      raise ValueError(f'The sink shape {sink.shape!r} does not match the source shape {source.shape!r}!')

    source_flat = ac.check_dtype(ac.bool_as_uint8(source.reshape(-1, order=order)), _DTYPES, name='source')
    sink_flat = ac.check_dtype(ac.bool_as_uint8(sink.reshape(-1, order=order)), _SINK_DTYPES, name='sink')
    if not np.shares_memory(sink_flat, sink):  # the result would be written to a copy
      raise ValueError(f'The sink must be {order}-contiguous like the source!')
    strides = ac.as_index_array(source_geometry.element_strides(source), name='strides', ndim=1)
     
    if seeds is None:
      seeds = np.where(source_flat >= threshold)[0]
    else:
      seeds = np.asarray(seeds)
      if seeds.shape != source.shape:
        raise ValueError(f'The seeds shape {seeds.shape!r} does not match the source shape {source.shape!r}!')
      seeds = np.where(seeds.reshape(-1, order=order))[0]
    seeds = ac.as_index_array(seeds, name='seeds', ndim=1)
    
    if hysteresis_threshold is not None:
      parameter_index = np.zeros(0, dtype = np.intp)
      parameter_double = np.array([hysteresis_threshold], dtype=float)

      if background is not None:
        background = np.asarray(background)
        if background.shape != source.shape:
          raise ValueError(f'The background shape {background.shape!r} does not match the source shape {source.shape!r}!')
        background_flat = ac.as_uint8_flags(background.reshape(-1, order=order), name='background')
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
