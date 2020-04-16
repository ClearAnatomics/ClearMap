#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
ArrayProcessing
===============

Tools for parallel processing of large arrays.

Note
----
This module provides an interface to deal with large numpy arrays and speed up 
numpy routines that get very slow for data arrays above 100-500GB of size.

The processing is based on the buffer interface used by cython.
"""
__author__    = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__   = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__copyright__ = 'Copyright (c) 2020 by Christoph Kirst'

import os
import numpy as np
import multiprocessing as mp

import ClearMap.IO.IO as io
import ClearMap.IO.Slice as slc

import ClearMap.Utils.Timer as tmr

import pyximport;

old_get_distutils_extension = pyximport.pyximport.get_distutils_extension

def _new_get_distutils_extension(modname, pyxfilename, language_level=None):
    extension_mod, setup_args = old_get_distutils_extension(modname, pyxfilename, language_level)
    extension_mod.language='c++'
    return extension_mod,setup_args

pyximport.pyximport.get_distutils_extension = _new_get_distutils_extension

pyximport.install(setup_args = {"include_dirs" : [np.get_include(), os.path.dirname(os.path.abspath(__file__))]},
                  reload_support=True)

import ClearMap.ParallelProcessing.DataProcessing.ArrayProcessingCode_New as code


###############################################################################
### Default Machine Settings
###############################################################################

default_processes = mp.cpu_count();
"""Default number of processes to use"""


default_blocks_per_process = 10;
"""Default number of blocks per process to split the data.

Note
----
10 blocks per process is a good choice.
"""


default_cutoff = 20000000;
"""Default size of array below which ordinary numpy is used.

Note
----
Ideally test this on your machine for different array sizes.
"""


###############################################################################
### Correlation
###############################################################################

def correlate1d(source, kernel, sink = None, axis = 0, processes = None, verbose = False):
  """Correlates the source along the given axis wih ta 1d kernel.
  
  Arguments
  ---------
  source : array 
    The source array.
  lut : array
    The lookup table.
  sink : array or None
    The result array, if none an array is created.
  processes : None or int
    Number of processes to use, if None use number of cpus
  verbose : bool
    If True, print progress information.
  
  Returns
  -------
  sink : array
    The source transformed via the lookup table.
  """ 
  processes, timer =  initialize_processing(processes=processes, verbose=verbose, function='correlate1d');

  source, source_buffer, source_shape, source_strides = initialize_source(source, as_1d=True, return_shape=True, return_strides=True);
  kernel, kernel_buffer = initialize_source(kernel);
  
  dtype = np.result_type(source_buffer.dtype, kernel_buffer.dtype) if sink is None else None;
  sink, sink_buffer, sink_shape, sink_strides = initialize_sink(sink=sink, as_1d=True, return_shape=True, shape=source_shape, dtype=dtype, source=source, return_strides=True);
  
  kernel_buffer = np.asarray(kernel_buffer, dtype=float);
  
  code.correlate_1d(source_buffer, source_shape, source_strides, axis, sink_buffer, sink_shape, sink_strides, kernel_buffer, processes=processes);

  finalize_processing(verbose=verbose, function='correlate1d', timer=timer);

  return sink;




###############################################################################
### Initialization
###############################################################################

def initialize_processing(processes = None, verbose = False, function = None, blocks = None, return_blocks = False):
  """Initialize parallel array processing.
  
  Arguments
  ---------
  processes : int, 'seial' or None
    The number of processes to use. If None use number of cpus.
  verbose : bool
    If True, print progress information.
  function : str or None
    The nae of the function.
  
  Returns
  -------
  processes : int
    The number of processes.
  timer : Timer
    A timer for the processing.
  """
  if processes is None:
    processes = default_processes;
  if processes == 'serial':
    processes = 1;
  
  if verbose:
    if function:
      print('%s: initialized!' % function);
    timer = tmr.Timer();
  else:
    timer = None;
  
  results = (processes, timer);
  
  if return_blocks:
    if blocks is None:
      blocks = processes * default_blocks_per_process;
    results += (blocks,)
  
  return results;


def finalize_processing(verbose = False, function = None, timer = None):
  """Finalize parallel array processing.
  
  Arguments
  ---------
  verbose : bool
    If True, print progress information.
  function : str or None
    The nae of the function.
  timer : Timer or None
    A processing timer.
  """
  if verbose:
    if function and timer:
      timer.print_elapsed_time(function);


def initialize_source(source, return_buffer = True, as_1d = False, 
                      return_shape = False, return_strides = False, return_order = False):
  """Initialize a source buffer for parallel array processing.
  
  Arguments
  ---------
  source : source specification
    The source to initialize.
  return_buffer : bool
    If True, return a buffer compatible with cython memory views.
  return_shape : bool
    If True, also return shape of the source.
  return_strides : bool
    If True, also return the element strides of the source.
  return_order : bool
    If True, also return order of the source.
  
  Returns
  -------
  source : Source or buffer
    The intialized source.
  source_buffer
  shape : tuple of int
    Shape of the source.
  return_Strides : tuple of int
    Element strides of the source. 
  """
  source = io.as_source(source)

  if return_shape:
    shape = np.array(source.shape, dtype=int);
  
  if return_strides:
    strides = np.array(source.element_strides, dtype=int);
    
  if return_order:
    order = source.order;
    
  if return_buffer:
    source_buffer = source.as_buffer();

    if source_buffer.dtype == bool:
      source_buffer = source_buffer.view('uint8');  
  
    if as_1d:
      source_buffer = source_buffer.reshape(-1, order = 'A');
  
  result = (source,);
  if return_buffer:
    result += (source_buffer,)
  if return_shape:
    result += (shape,);
  if return_strides:
    result += (strides,);
  if return_order:
    result += (order,);
  
  if len(result) == 1:
    return result[0];
  else:
    return result;


def initialize_sink(sink = None, shape = None, dtype = None, order = None, memory = None, location = None, mode = None, source = None, 
                    return_buffer = True, as_1d = False, return_shape = False, return_strides = False):
  """Initialze or create a sink.
  
  Arguments
  ---------
  sink : sink specification
    The source to initialize.
  shape : tuple of int
    Optional shape of the sink. If None, inferred from the source.
  dtype : dtype
    Optional dtype of the sink. If None, inferred from the source.
  order : 'C', 'F' or None
    Optonal order of the sink. If None, inferred from the source.
  memory : 'shared' or None
    If 'shared' create a shared memory sink.
  location : str
    Optional location specification of the sink.
  source : Source or None
    Optional source to infer sink specifictions from.
  return_buffer : bool
    If True, return alos a buffer compatible with cython memory views. 
  return_shape : bool
    If True, also return shape of the sink.
  return_strides : bool
    If True, also return the element strides of the sink.
  
  Returns
  -------
  sink : Source
    The intialized sink.
  buffer : array
    Buffer of the sink.
  shape : tuple of int
    Shape of the source.
  strides : tuple of int
    Element strides of the source. 
  """
  if sink is None:
    if shape is None:
      if source is None:
        raise ValueError("Cannot determine shape for sink without source or shape.");
      else:
        shape = source.shape;
    if dtype is None:
      if source is None:
        raise ValueError("Cannot determine dtype for sink without source or dtype.");
      else:
        dtype = source.dtype;  
    if order is None and source is not None:
      order = source.order;
      
  #print('shape=%r, dtype=%r, order=%r, memory=%r, location=%r' % (shape, dtype, order, memory, location));
        
  sink = io.initialize(sink, shape=shape, dtype=dtype, order=order, memory=memory, location=location, mode=mode, as_source=True);
  
  if return_buffer:
    buffer = sink.as_buffer();
  
    if buffer.dtype == bool:
      buffer = sink.view('uint8');
      
    if as_1d:
      buffer = buffer.reshape(-1, order = 'A');  
  
  result = (sink,)
  if return_buffer:
    result += (buffer,);
  if return_shape:
    result += (np.array(sink.shape,dtype=int),);
  if return_strides:
    result += (np.array(sink.element_strides, dtype=int),);
  
  if len(result) == 1:
    return result[0];
  else:
    return result;






###############################################################################
### Tests
###############################################################################

def _test():
  import numpy as np
  import ClearMap.ParallelProcessing.DataProcessing.ArrayProcessing_New as ap
  import ClearMap.ParallelProcessing.DataProcessing.ArrayProcessing as ap2
  
  #from importlib import reload
  #reload(ap)

  ## Correlation 
  
  #correlate1d
  #reload(ap)
  processes = 1;
  axis = 1;
  #kernel = np.array(range(11), dtype='uint32');  
  kernel = np.array([-1,0,1], dtype=int);
  #data = np.random.randint(0, 2**27, (100, 150,5), dtype='uint32');
  
  data = np.array(np.arange(10*5),dtype=float);
  data = data.reshape(10,-1,1);
  data = np.array(np.random.rand(30,40,50), order='F');
 
  data = np.array(np.random.rand(300,400,500), order='F');
  
  
  axis = 1;
  corr = ap.correlate1d(data, kernel, axis=axis, verbose=True, processes=processes);  
  print('done!')
  
  print(corr.array);


def temp():  
  import scipy.ndimage as ndi
  corr_ndi = ndi.correlate1d(data, kernel, axis=axis, mode='constant',cval=0);
  
  assert np.allclose(corr.array, corr_ndi)

  c = corr.array;
  c[0,:,0]
  corr_ndi[0,:,0]
  
  data = np.array(np.random.rand(1000, 1000,500), order='F');
  
  data = np.array(np.random.rand(300,400,1500), order='F');
  kernel = np.array([1,2,3,4,5]);

  
  import ClearMap.Utils.Timer as tmr
  timer = tmr.Timer();
  for axis in range(3):
    corr = ap.correlate1d(data, kernel, axis=axis, verbose=False, processes=None);  
  timer.print_elapsed_time('ap')
  
  import ClearMap.Utils.Timer as tmr
  timer = tmr.Timer();
  for axis in range(3):
    corr2 = ap2.correlate1d(data, kernel, axis=axis, verbose=False, processes=None);  
  timer.print_elapsed_time('ap')
  
  
  import scipy.ndimage as ndi
  timer = tmr.Timer();
  for axis in range(3):
    corr_ndi = ndi.correlate1d(data, kernel, axis=axis, mode='constant',cval=0);
  timer.print_elapsed_time('ndi')  
  
  assert np.allclose(corr.array, corr_ndi)
  assert np.allclose(corr2.array, corr_ndi)
  
  # IO
  import ClearMap.ParallelProcessing.DataProcessing.ArrayProcessing as ap
  import numpy as np
  reload(ap)
  
  
  data = np.random.rand(10,200,10)
  
  sink = ap.write('test.npy', data, verbose=True)  
  assert(np.all(sink.array == data))
  
  read = ap.read('test.npy', verbose=True)
  assert(np.all(read.array == data))
  
  ap.io.delete_file('test.npy')


  # where
  reload(ap)
  data = np.random.rand(30,20,40) > 0.5;
  
  where_np = np.array(np.where(data)).T
  where = ap.where(data, cutoff = 2**0)
  
  check_np = np.zeros(data.shape, dtype=bool);
  check = np.zeros(data.shape, dtype=bool);
  check_np[tuple(where_np.T)] = True;
  check[tuple(where.array.T)] = True;
  assert(np.all(check_np == check))
  

if __name__ == '__main__':
  _test();


#
#def setValue(data, indices, value, cutoff = defaultCutoff, processes = defaultProcesses):
#  """Set value at specified indices of an array
#  
#  Arguments:
#    data : array
#      array to search for nonzero indices
#    indices : array or None
#      list of indices to set
#    value : numeric or bool
#      value to set elements in data to
#    processes : None or int
#      number of processes, if None use number of cpus
#    
#  Returns:
#    array
#      array with specified entries set to new value
#  
#  Note:
#    Uses numpy if there is no match of dimension implemented!
#  """
#  if data.ndim != 1:
#    raise Warning('Using numpy where for dimension %d and type %s!' % (data.ndim, data.dtype))
#    data[indices] = value;
#    return data;
#    
#  if cutoff is None:
#    cutoff = 1;
#  cutoff = min(1, cutoff);
#  if data.size <= cutoff:
#    data[indices] = value;
#    return data;
#  
#  if processes is None:
#    processes = defaultProcesses;
#  
#  if data.dtype == bool:
#    d = data.view('uint8')
#  else:
#    d = data;
#  
#  code.set1d(d, indices, value, processes = processes);
#  
#  return data;
#
#
#def setArray(data, indices, values, cutoff = defaultCutoff, processes = defaultProcesses):
#  """Set value at specified indices of an array
#  
#  Arguments:
#    data : array
#      array to search for nonzero indices
#    indices : array or None
#      list of indices to set
#    values : array
#      values to set elements in data to
#    processes : None or int
#      number of processes, if None use number of cpus
#    
#  Returns:
#    array
#      array with specified entries set to new value
#  
#  Note:
#    Uses numpy if there is no match of dimension implemented!
#  """
#  if data.ndim != 1:
#    raise Warning('Using numpy where for dimension %d and type %s!' % (data.ndim, data.dtype))
#    data[indices] = values;
#    return data;
#    
#  if cutoff is None:
#    cutoff = 1;
#  cutoff = min(1, cutoff);
#  if data.size <= cutoff:
#    data[indices] = values;
#    return data;
#  
#  if processes is None:
#    processes = defaultProcesses;
#  
#  if data.dtype == bool:
#    d = data.view('uint8')
#  else:
#    d = data;
#  
#  code.set1darray(d, indices, values, processes = processes);
#  
#  return data;
#
#
#
#def take(data, indices, out = None, cutoff = defaultCutoff, processes = defaultProcesses):
#  """Extracts the values at specified indices
#  
#  Arguments:
#    data : array
#      array to search for nonzero indices
#    out : array or None
#      if not None results is written into this array
#    cutoff : int
#      number of elements below whih to switch to numpy.where
#    processes : None or int
#      number of processes, if None use number of cpus
#    
#  Returns:
#    array
#      positions of the nonzero entries of the input array
#  
#  Note:
#    Uses numpy data[indices] if there is no match of dimension implemented!
#  """ 
#  if data.ndim != 1:
#    raise Warning('Using numpy where for dimension %d and type %s!' % (data.ndim, data.dtype))
#    return data[indices];
#
#  if cutoff is None:
#    cutoff = 1;
#  cutoff = min(1, cutoff);
#  if data.size < cutoff:
#    return data[indices];
#
#  if processes is None:
#    processes = defaultProcesses;
#  
#  if data.dtype == bool:
#    d = data.view('uint8')
#  else:
#    d = data;
#
#  if out is None:
#    out = np.empty(len(indices), dtype = data.dtype);
#  if out.dtype == bool:
#    o = out.view('uint8');
#  else:
#    o = out;
#  
#  code.take1d(d, indices, o, processes = processes);
#  
#  return out;
#
#
#def match(match, indices, out = None):
#  """Matches a sorted list of 1d indices to another larger one 
#  
#  Arguments:
#    match : array
#      array of indices to match to indices
#    indices : array or None
#      array of indices
#  
#  Returns:
#    array
#      array with specified entries set to new value
#  
#  Note:
#    Uses numpy if there is no match of dimension implemented!
#  """
#  if match.ndim != 1:
#    raise ValueError('Match array dimension required to be 1d, found %d!' % (match.ndim))
#  if indices.ndim != 1:
#    raise ValueError('Indices array dimension required to be 1d, found %d!' % (indices.ndim))  
#  
#  if out is None:
#    out = np.empty(len(match), dtype = match.dtype);
#  
#  code.match1d(match, indices, out);
#  
#  return out;
#
#
# Find neighbours in an index list
#
#
#def neighbours(indices, offset, processes = defaultProcesses):
#  """Returns all pairs of indices that are apart a specified offset"""
#  return code.neighbours(indices, offset = offset,  processes = processes);
#
#
#def findNeighbours(indices, center, shape, strides, mask):
#  """Finds all indices within a specified kernel region centered at a point"""
#  
#  if len(strides) != 3 or len(shape) != 3 or (strides[0] != 1 and strides[2] != 1):
#    raise RuntimeError('only 3d C or F contiguous arrays suported');
#
#  if isinstance(mask, int):
#    mask = (mask,);
#  if isinstance(mask, tuple):
#    mask = mask * 3;
#    return code.neighbourlistRadius(indices, center, shape[0], shape[1], shape[2], 
#                                                     strides[0], strides[1], strides[2], 
#                                                     mask[0], mask[1], mask[2]);
#  else:
#    if mask.dtype == bool:
#      mask = mask.view(dtype = 'uint8');
#                                                
#    return code.neighbourlistMask(indices, center, shape[0], shape[1], shape[2], strides[0], strides[1], strides[2], mask);
