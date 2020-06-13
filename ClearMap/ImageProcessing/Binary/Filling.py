"""
Filling
=======

Parallel binary filling on arbitraily sized images.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'

import os
import numpy as np
import gc

import multiprocessing as mp

import pyximport;

old_get_distutils_extension = pyximport.pyximport.get_distutils_extension

def new_get_distutils_extension(modname, pyxfilename, language_level=None):
    extension_mod, setup_args = old_get_distutils_extension(modname, pyxfilename, language_level)
    extension_mod.language='c++'
    return extension_mod,setup_args

pyximport.pyximport.get_distutils_extension = new_get_distutils_extension

pyximport.install(setup_args = {"include_dirs" : [np.get_include(), os.path.dirname(os.path.abspath(__file__))]},
                  reload_support=True)

from . import FillingCode as code

import ClearMap.IO.IO as io;
import ClearMap.IO.SMA as sma

import ClearMap.Utils.Timer as tmr


#%%############################################################################
### Binarization
###############################################################################

def fill(source, sink = None, seeds = None, processes = None, verbose = False):
  """Fill binary holes.
  
  Arguments
  ---------
  source : array
      Input source.
  sink : array or None
      If None, a new array is allocated.
 

  Returns
  -------
  sink : array
      Binary image with filled holes.
  """
  if source is sink:
    raise NotImplementedError("Cannot perform operation in place.")
    
  if verbose:
    print('Binary filling: initialized!')
    timer = tmr.Timer()
  
  #create temporary shared array 
  order = io.order(source);
  #temp = sma.empty(source.shape, dtype='int8', order=order);
  temp = np.empty(source.shape, dtype='int8', order=order);
  
  source_flat = source.reshape(-1, order='A');
  temp_flat = temp.reshape(-1, order='A');                        
  
  if source_flat.dtype == bool:
    source_flat = source_flat.view(dtype = 'uint8')
    
  if processes is None:
    processes = mp.cpu_count();
  if not isinstance(processes, int):
    processes = 1;
  
  #prepare flood fill
  code.prepare_temp(source_flat, temp_flat, processes=processes);
    
  #flood fill in parallel using mp
  if seeds is None:
    seeds = border_indices(source);
  else:
    seeds = np.where(seeds.reshape(-1, order=order))[0];
  
  strides = np.array(io.element_strides(source));
  
  code.label_temp(temp_flat, strides, seeds, processes=processes)
  
  if sink is None:
    #sink = sma.empty(source.shape, dtype=bool, order=order);
    sink = np.empty(source.shape, dtype=bool, order=order);
  sink_flat = sink.reshape(-1, order=order);    
  
  if sink_flat.dtype == 'bool':
    sink_flat = sink_flat.view(dtype = 'uint8')
  
  code.fill(source_flat, temp_flat, sink_flat, processes=processes);
  
  if verbose:
    timer.print_elapsed_time('Binary filling')

  del temp, temp_flat    
  gc.collect();
  
  return sink;


def border_indices(source):
  """Returns the flat indices of the border pixels in source"""  
  
  ndim = source.ndim;
  shape = source.shape;
  strides = io.element_strides(source);
    
  border = [];
  for d in range(ndim):
    offsets = tuple(0 if i>d else 1 for i in range(ndim));
    for c in [0, shape[d]-1]:
      sl = tuple(slice(o, None if o==0 else -o) if i != d else c for i,o in enumerate(offsets));
      where = np.where(np.logical_not(source[sl]));
      n = len(where[0]);
      if n > 0:
        indices = np.zeros(n, dtype = int);   
        l = 0;
        for k in range(ndim):
          if k == d:
            indices += strides[k] * c;
          else:
            indices += strides[k] * (where[l] + offsets[k]);
            l += 1;
        border.append(indices);
  return np.concatenate(border);


def _test():
  """Tests."""
  import numpy as np
  import ClearMap.Visualization.Plot3d as p3d
  import ClearMap.ImageProcessing.Binary.Filling as bf
  
  from importlib import reload
  reload(bf);
  
  test = np.zeros((50,50,50), dtype = bool);
  test[20:30, 30:40, 25:35] = True
  test[25:27, 34: 38, 27: 32] = False;
  test[5:15, 5:15, 23:35] = True
  test[8:12, 8:12, 27:32] = False;
  
  filled = bf.fill(test, sink=None, processes=10)
  p3d.plot([test, filled])
