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

The implementation builds on the buffer interface used by cython.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'

import os
import numpy as np
import multiprocessing as mp

import ClearMap.IO.IO as io
import ClearMap.IO.Slice as slc

import ClearMap.Utils.Timer as tmr

import pyximport;

_old_get_distutils_extension = pyximport.pyximport.get_distutils_extension

def _new_get_distutils_extension(modname, pyxfilename, language_level=None):
    extension_mod, setup_args = _old_get_distutils_extension(modname, pyxfilename, language_level)
    extension_mod.language='c++'
    return extension_mod,setup_args

pyximport.pyximport.get_distutils_extension = _new_get_distutils_extension

pyximport.install(setup_args = {"include_dirs" : [np.get_include(), os.path.dirname(os.path.abspath(__file__))]},
                  reload_support=True)

import ClearMap.ParallelProcessing.DataProcessing.ArrayProcessingCode as code


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
### Lookup table transforms
###############################################################################

def apply_lut(source, lut, sink = None, blocks = None, processes = None, verbose = False):
  """Transforms the source via a lookup table.
  
  Arguments
  ---------
  source : array 
    The source array.
  lut : array
    The lookup table.
  sink : array or None
    The result array, if none an array is created.
  processes : None or int
    Number of processes to use, if None use number of cpus.
  verbose : bool
    If True, print progress information.
  
  Returns
  -------
  sink : array
    The source transformed via the lookup table.
  """
  processes, timer, blocks = initialize_processing(processes=processes, function='apply_lut', verbose=verbose, blocks=blocks, return_blocks=True);

  source, source_buffer = initialize_source(source, as_1d=True);
  lut, lut_buffer       = initialize_source(lut);

  sink, sink_buffer = initialize_sink(sink=sink, source=source, as_1d=True, dtype=lut.dtype);
  
  code.apply_lut(source_buffer, sink_buffer, lut_buffer, blocks=blocks, processes=processes)

  finalize_processing(verbose=verbose, function='apply_lut', timer=timer);

  return sink;


def apply_lut_to_index(source, kernel, lut, sink = None, processes = None, verbose = False):
  """Correlates the source with an index kernel and returns the value of the the look-up table.
  
  Arguments
  ---------
  source : array 
    The source array.
  kernel : array
    The correlation kernel.
  lut : array
    The lookup table.
  sink : array or None
    The result array, if none an array is created.
  processes : None or int
    Number of processes to use, if None use number of cpus
  
  Returns
  -------
  sink : array
    The source transformed via the lookup table.
  """
  processes, timer =  initialize_processing(processes=processes, verbose=verbose, function='apply_lut_to_index');

  source, source_buffer, source_shape   = initialize_source(source, return_shape=True);
  kernel, kernel_buffer, kernel_shape   = initialize_source(kernel, return_shape=True);
  sink, sink_buffer, sink_shape = initialize_sink(sink=sink, dtype=lut.dtype, source=source, return_shape=True);
  lut, lut_buffer = initialize_source(lut);
  
  if len(source_shape) != 3 or len(kernel_shape) != 3 or len(sink_shape) != 3:
    raise NotImplementedError('apply_lut_index not implemented for non 3d sources, found %d dimensions!'% len(source_shape));
    
  code.apply_lut_to_index_3d(source_buffer, kernel_buffer, lut_buffer, sink_buffer, processes=processes)

  finalize_processing(verbose=verbose, function='apply_lut_to_index', timer=timer);

  return sink;


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
  processes, timer =  initialize_processing(processes=processes, 
                                            verbose=verbose, 
                                            function='correlate1d');

  source, source_buffer, source_shape, source_strides = initialize_source(source, as_1d=True, return_shape=True, return_strides=True);
  kernel, kernel_buffer = initialize_source(kernel);
  
  dtype = np.result_type(source_buffer.dtype, kernel_buffer.dtype) if sink is None else None;
  sink, sink_buffer, sink_shape, sink_strides = initialize_sink(sink=sink, as_1d=True, shape=tuple(source_shape), dtype=dtype, source=source, return_shape=True, return_strides=True);
  
  kernel_buffer = np.asarray(kernel_buffer, dtype=float);
  
  code.correlate_1d(source_buffer, source_shape, source_strides, 
                    sink_buffer, sink_shape, sink_strides, 
                    kernel_buffer, axis, processes=processes);

  finalize_processing(verbose=verbose, function='correlate1d', timer=timer);

  return sink;


###############################################################################
### Where
###############################################################################

def where(source, sink = None, blocks = None, cutoff = None, processes = None, verbose = False):
  """Returns the indices of the non-zero entries of the array.
  
  Arguments
  ---------
  source : array
    Array to search for nonzero indices.
  sink : array or None
    If not None, results is written into this array
  blocks : int or None
    Number of blocks to split array into for parallel processing
  cutoff : int
    Number of elements below whih to switch to numpy.where
  processes : None or int
    Number of processes, if None use number of cpus.
    
  Returns
  -------
  where : array
      Positions of the nonzero entries of the input array
  
  Note
  ----
    Uses numpy.where if there is no match of dimension implemented!
  """  
  source, source_buffer = initialize_source(source);
  
  ndim = source_buffer.ndim;
  if not ndim in [1,2,3]:
    raise Warning('Using numpy.where for dimension %d!' % (ndim,))
    return io.as_source(np.vstack(np.where(source_buffer)).T);

  processes, timer, blocks = initialize_processing(processes=processes, function='where', verbose=verbose, blocks=blocks, return_blocks=True)
    
  if cutoff is None:
    cutoff = 1;
  cutoff = min(1, cutoff);
  
  if source_buffer.size <= cutoff:
    result = np.vstack(np.where(source_buffer)).T;
    if sink is None:
      sink = io.as_source(result);
    else:
      sink, sink_buffer = initialize_sink(sink=sink, shape=result.shape)
      sink[:] = result;  
  else:
    if ndim == 1:
      sums = code.block_sums_1d(source_buffer, blocks=blocks, processes=processes);
    elif ndim == 2:
      sums = code.block_sums21d(source_buffer, blocks=blocks, processes=processes);
    else:
      sums = code.block_sums_3d(source_buffer, blocks=blocks, processes=processes);
    
    if ndim == 1:
      sink_shape = (np.sum(sums),)
    else:
      sink_shape = (np.sum(sums), ndim);
    sink, sink_buffer = initialize_sink(sink=sink, shape=sink_shape, dtype=int);
     
    if ndim == 1:
      code.where_1d(source_buffer, where=sink_buffer, sums=sums, blocks=blocks, processes=processes);
    elif ndim == 2:
      code.where_2d(source_buffer, where=sink_buffer, sums=sums, blocks=blocks, processes=processes);
    else:
      code.where_3d(source_buffer, where=sink_buffer, sums=sums, blocks=blocks, processes=processes);
  
  finalize_processing(verbose=verbose, function='where', timer=timer);
    
  return sink;


def neighbours(indices, offset, processes = None, verbose = False):
  """Returns all pairs in a list of indices that are apart a specified offset.
  
  Arguments
  ---------
  indices : array
    List of indices.
  offset : int
    The offset to search for.
  processes : None or int
    Number of processes, if None use number of cpus.
  verbose : bool
    If True, print progress.
    
  Returns
  -------
  neighbours : array 
    List of pairs of neighbours.
    
  Note
  ----
  This function can be used to create graphs from binary images.
  """
  processes, timer = initialize_processing(processes=processes, verbose=verbose, function='neighbours');
  
  neighbours =  code.neighbours(indices, offset=offset,  processes=processes);
  
  finalize_processing(verbose=verbose, timer=timer, function='neighbours')
  
  return neighbours;

###############################################################################
### IO
###############################################################################

def read(source, sink = None, slicing = None, memory = None, blocks = None, processes = None, verbose = False, **kwargs):
  """Read a large array into memory in parallel.
  
  Arguments
  ---------
  source : str or Source
    The source on diks to load.
  slicing : slice, tuple, or None
    Optional sublice to read.
  memory : 'shared; or None
    If 'shared', read into shared memory.
  blocks : int or None
    number of blocks to split array into for parallel processing
  processes : None or int
    number of processes, if None use number of cpus
  verbose : bool
    print info about the file to be loaded
    
  Returns
  -------
  sink : Source class
    The read source in memory.
  """
  processes, timer, blocks = initialize_processing(processes=processes, verbose=verbose, function='read', blocks=blocks, return_blocks=True);
  
  #source info
  source = io.as_source(source);
  if slicing is not None:
    source = slc.Slice(source=source, slicing=slicing);
  
  shape, location, dtype, order, offset = source.shape, source.location, source.dtype, source.order, source.offset;

  if location is None:
    raise ValueError('The source has not valid location to read from!');  
  if order not in ['C', 'F']:
    raise NotImplementedError('Cannot read in parallel from non-contigous source!');
    #TODO: implement parallel reader with strides !
  
  sink, sink_buffer = initialize_sink(sink=sink, shape=shape, dtype=dtype, order=order, memory=memory, as_1d=True);
  
  code.read(sink_buffer, location.encode(), offset=offset, blocks=blocks, processes=processes);
  
  finalize_processing(verbose=verbose, function='read', timer=timer);
           
  return sink;


def write(sink, source, slicing = None, overwrite = True, blocks = None, processes = None, verbose = False):
  """Write a large array to disk in parallel.
  
  Arguments
  ---------
  sink : str or Source
    The sink on disk to write to.
  source : array or Source
    The data to write to disk.
  slicing : slicing or None
    Optional slicing for the sink to write to.
  overwrite : bool
    If True, create new file if the source specifications do not match.
  blocks : int or None
    Number of blocks to split array into for parallel processing.
  processes : None or int
    Number of processes, if None use number of cpus.
  verbose : bool
    Print info about the file to be loaded.
    
  Returns
  -------
  sink : Source class
      The sink to which the source was written.
  """
  processes, timer, blocks = initialize_processing(processes=processes, verbose=verbose, function='write', blocks=blocks, return_blocks=True);
  
  source, source_buffer, source_order = initialize_source(source, as_1d=True, return_order = True);  
  
  try:
    sink = io.as_source(sink);
    location = sink.location;
  except:
    if isinstance(sink, str):
      location = sink;
      sink = None;
    else:
      raise ValueError('Sink is not a valid writable sink specification!')
  if location is None:
    raise ValueError('Sink is not a valid writable sink specification!');

  if slicing is not None:
    if not io.is_file(location):
      raise ValueError('Cannot write a slice to a non-existent sink %s!' % location);
    sink = slc.Slice(source=sink, slicing=slicing);
  else:
    if io.is_file(location):
      mode = None;
      if (sink.shape != source.shape or sink.dtype != source.dtype or sink.order != source_order):
        if overwrite:
          mode = 'w+';
        else:
          raise ValueError('Sink file %s exists but does not match source!');
      sink_shape = source.shape;
      sink_dtype = source.dtype;
      sink_order = source.order;
      sink = None;
    else:
      sink_shape = None;
      sink_dtype = None;
      sink_order = None;
      mode = None;
    
    sink = initialize_sink(sink=sink, location=location, shape=sink_shape, dtype=sink_dtype, order=sink_order, mode=mode, source=source, return_buffer=False);
  sink_order, sink_offset = sink.order, sink.offset;
  
  if sink_order not in ['C', 'F']:
    raise NotImplementedError('Cannot read in parallel from non-contigous source!');  
  if (source_order != sink_order):
    raise RuntimeError('Order of source %r and sink %r do not match!' % (source_order, sink_order));    
  
  #print(source_buffer.shape, location, sink_offset, blocks, processes)
  code.write(source_buffer, location.encode(), offset=sink_offset, blocks=blocks, processes=processes);
  
  finalize_processing(verbose=verbose, function='write', timer=timer);
           
  return sink;


###############################################################################
### Utility
###############################################################################

def block_ranges(source, blocks = None, processes = None):
  """Ranges of evenly spaced blocks in array.
  
  Arguments
  ---------
  source : array
    Source to divide in blocks.
  blocks : int or None
    Number of blocks to split array into.
  processes : None or int
    Number of processes, if None use number of cpus.
    
  Returns
  -------
  block_ranges : array
    List of the range boundaries
  """
  processes, _ = initialize_processing(processes=processes, verbose=False)
  if blocks is None:
    blocks = processes * default_blocks_per_process;
  
  size = io.size(source);
  blocks = min(blocks, size);
  return np.array(np.linspace(0, size, blocks + 1), dtype=int);


def block_sums(source, blocks = None, processes = None):
  """Sums of evenly spaced blocks in array.
  
  Arguments
  ---------
  data : array
    Array to perform the block sums on.
  blocks : int or None
    Number of blocks to split array into.
  processes : None or int
    Number of processes, if None use number of cpus.
  
  Returns
  -------
  block_sums : array
    Sums of the values in the different blocks.
  """
  processes, _ = initialize_processing(processes=processes, verbose=False)
  if blocks is None:
    blocks = processes * default_blocks_per_process;
  
  source, source_buffer = initialize_source(source, as_1d=True);
  
  return code.block_sums_1d(source_buffer, blocks=blocks, processes=processes);
 
  
def index_neighbours(indices, offset, processes = None):
  """Returns all pairs of indices that are a part of a specified offset.
  
  Arguments
  ---------
  indices : array
    List of indices.
  offset : int
    The offset to check for.
  processes : None or int
    Number of processes, if None use number of cpus.
  """
  processes, _ = initialize_processing(processes=processes, verbose=False)
  indices, indices_buffer = initialize_source(indices);
  return code.index_neighbours(indices_buffer, offset=offset, processes=processes);


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
  source : Source
    The intialized source.
  source_buffer
    The initialized source as buffer.
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
       
  sink = io.initialize(sink, shape=shape, dtype=dtype, order=order, memory=memory, location=location, mode=mode, like=source, as_source=True);
  
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
  import ClearMap.ParallelProcessing.DataProcessing.ArrayProcessing as ap  
  
  ## Lookup table processing
  
  #apply_lut  
  x = np.random.randint(0, 100, size=(20,30));
  lut = np.arange(100) + 1;
  y = ap.apply_lut(x, lut)
  assert np.all(y == x+1)

  #apply_lut_to_index
  import ClearMap.ImageProcessing.Topology.Topology3d as t3d
  kernel = t3d.index_kernel(dtype=int);
  
  import ClearMap.ImageProcessing.Binary.Smoothing as sm
  lut = sm.initialize_lookup_table();
    
  data = np.array(np.random.rand(150,30,40) > 0.75, order='F');
  
  result = ap.apply_lut_to_index(data, kernel, lut, sink=None, verbose=True)

  import ClearMap.Visualization.Plot3d as p3d
  p3d.plot([[data, result]])    
  
  
  ### Correlation 
  
  #correlate1d
  kernel = np.array(range(11), dtype='uint32');  
  data = np.array(np.random.randint(0, 2**27, (300, 400, 1500), dtype='uint32'), order='F');
  #data = np.array(np.random.rand(3,4,5), order='F');
  
  data = np.empty((300,400,1500), order='F');
  kernel = np.array([1,2,3,4,5], dtype='uint8');
  
  sink = 'test.npy'
  
  import ClearMap.Utils.Timer as tmr
  import scipy.ndimage as ndi
  timer = tmr.Timer();
  for axis in range(3):
    print(axis);
    corr_ndi = ndi.correlate1d(data, axis=axis, mode='constant',cval=0);
  timer.print_elapsed_time('ndi')  
  
  timer = tmr.Timer();
  for axis in range(3):
    print(axis)
    corr = ap.correlate1d(data, sink=sink, kernel=kernel, axis=axis, verbose=False, processes=None);  
  timer.print_elapsed_time('ap')
  

  assert np.allclose(corr.array, corr_ndi)
  
  
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
