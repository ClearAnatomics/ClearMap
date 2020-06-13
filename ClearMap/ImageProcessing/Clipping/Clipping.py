"""
Clipping
========

Module to compute clipped images

Usefull to sace memory in large data sets
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'

import numpy as np

import pyximport;
pyximport.install(setup_args={"include_dirs":np.get_include()}, 
                  reload_support=True)

import ClearMap.ParallelProcessing.DataProcessing.ArrayProcessing as ap

from . import ClippingCode as code

###############################################################################
### Clipping
###############################################################################

def clip(source, sink = None, clip_min = None, clip_max = None, clip_norm = None, processes = None, verbose = False):
  """Clip and normalize data.

  Arguments
  ---------
  source : array
      Input source.
  sink : array, dtype or None
      output sink or output data type, if None, a new array is allocated.
  clip_min : number
      Minimal number to clip source data to.
  clip_max : number
      Maximal number to clip source data to.
  clip_norm : number
      Normalization constant.

  Returns
  -------
  sink : array
      Clipped output.
  """
  processes, timer = ap.initialize_processing(verbose=verbose, processes=processes, function='clip');
  
  source, source_buffer = ap.initialize_source(source);

  if source.ndim != 3:
    raise ValueError('Source assumed to be 3d found %dd!' % source.ndim);
  
  if clip_min is None:
    clip_min = ap.io.min_value(source);
  
  if clip_max is None:
    clip_max = ap.io.max_value(source);
  
  if clip_norm is None:
    clip_norm = clip_max - clip_min;

  sink, sink_buffer = ap.initialize_sink(sink = sink, source = source);
                                            
  code.clip(source_buffer, sink_buffer, clip_min, clip_max, clip_norm, processes);
  
  return sink;

 
###############################################################################
### Tests
###############################################################################

def _test():
  import numpy as np
  import ClearMap.Utils.Timer as tmr
  import ClearMap.ImageProcessing.Clipping.Clipping as clp
  
  data = np.random.rand(1000,1000,2000);
  
  for p in [1, 10, None]:
    print('Clipping: processes = %r' % p);
    timer = tmr.Timer();
    clipped = clp.clip(data, clip_max = 0.5, processes = p);  
    timer.print_elapsed_time('Clipping');
    
    
