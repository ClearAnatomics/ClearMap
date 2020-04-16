#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
ArrayProcessing
===============

Tools for parallel processing of large arrays.

Note
----
This module provides an interface to tools to deal with large numpy arrays
and speed up numpy routines that get very slow for data arrays above
100-500GB of size.
"""
__author__    = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__   = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__copyright__ = 'Copyright (c) 2019 by Christoph Kirst, The Rockefeller University, New York City'

import os
import numpy as np
import multiprocessing as mp

import ClearMap.IO.IO as io

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

import ClearMap.ParallelProcessing.DataProcessing.ArrayProcessingCode as code


default_processes = mp.cpu_count();
"""Default number of processes to use"""


###############################################################################
### Lookup table transforms
###############################################################################

def apply_lut(source, lut, sink = None, processes = None, verbose = False):
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
  
  processes, timer =  _initialize_processing(processes=processes, verbose=verbose, function='apply_lut');

  source1d = _initialize_source(source, as_1d=True);
  sink, sink1d = _initialize_sink(sink=sink, source=source, return_1d=True, dtype=lut.dtype);
  
  lut = _initialize_source(lut);
  
  code.apply_lut(source1d, sink1d, lut, processes=processes)

  _finalize_processing(verbose=verbose, function='apply_lut', timer=timer);

  return sink;


def apply_lut_to_index(source, kernel, lut, sink = None, processes = None, verbose = False):
  """Correlates the source with an index kernel and returns the value of the the look-up table.
  
  Arguments
  ---------
  source : array 
    The source array.
  kernel : array
    
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
  
  processes, timer =  _initialize_processing(processes=processes, verbose=verbose, function='apply_lut_to_index');

  source = _initialize_source(source);
  sink   = _initialize_sink(sink=sink, source=source, dtype=lut.dtype);
  
  lut = _initialize_source(lut);
  kernel = _initialize_source(kernel);
  
  print(source.shape, kernel.shape, lut.shape, sink.shape)
  
  code.apply_lut_to_index_3d(source, kernel, lut, sink, processes=processes)

  _finalize_processing(verbose=verbose, function='apply_lut_to_index', timer=timer);

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
  
  processes, timer =  _initialize_processing(processes=processes, verbose=verbose, function='correlate1d');

  source1d, strides = _initialize_source(source, as_1d=True, return_strides=True);
  #sink, sink1d = _initialize_sink(sink=sink, source=source, return_1d=True, dtype=kernel.dtype, memory='shared');
  sink, sink1d = _initialize_sink(sink=sink, source=source, return_1d=True, dtype=kernel.dtype);
  
  shape = source.shape;
  line_shape = shape[axis];
  line_stride = strides[axis];
  
  shape_wo_axis = [s for d,s in enumerate(shape) if d != axis];
  strides_wo_axis = [s for d,s in enumerate(strides) if d != axis];
  grid = np.meshgrid(*[range(s) for s in shape_wo_axis], indexing='ij');
  line_starts = np.sum([s * g for s,g in zip(strides_wo_axis, grid)], axis = 0).flatten();
  
  #print(line_starts.dtype, type(line_stride), type(line_shape));
  code.correlate1d(source1d, line_starts, line_stride, line_shape, sink1d, kernel, processes=processes);

  _finalize_processing(verbose=verbose, function='correlate1d', timer=timer);

  return sink;


###############################################################################
### Initialization 
###############################################################################

def _initialize_processing(processes = None, verbose = False, function = None):
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
    
  return processes, timer


def _finalize_processing(verbose = False, function = None, timer = None):
  """finalize parallel array processing.
  
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


def _initialize_source(source, as_source = False, as_1d = False, return_shape = False, return_strides = False):
  """Initialize a source for parallel array processing.
  
  Arguments
  ---------
  source : source specification
    The source to initialize.
  as_source : bool
    If True, return source as Source class. If false, a buffer compatible with
    cython memory views is returned.
  return_shape : bool
    If True, also return shape of the source.
  return_strides : bool
    If True, also return the element strides of the source.
  
  Returns
  -------
  source : Source or buffer
    The intialized source.
  shape : tuple of int
    Shape of the source.
  return_Strides : tuple of int
    Element strides of the source. 
  """
  source = io.as_source(source)

  if return_shape:
    shape = source.shape;
  
  if return_strides:
    strides = source.element_strides;
    
  if not as_source:
    #make sure the source is a buffer compatible with cython memoryviews
    source = source.as_buffer();

  if source.dtype == bool:
    source = source.view('uint8');  

  if as_1d:
    source = source.reshape(-1, order = 'A');

  
  
  result = (source,);  
  if return_shape:
    result += (shape,);
  if return_strides:
    result += (strides,);
  
  if len(result) == 1:
    return result[0];
  else:
    return result;


def _initialize_sink(sink = None, source = None, return_1d = False, return_strides = False, shape =  None, dtype = None, order = None, memory = None, location = None):
  """Helper to initialize sinks."""
  if sink is None:
    if shape is None:
      if source is None:
        raise ValueError("Cannot determine sink without source.");
      else:
        shape = source.shape;
    if dtype is None:
      if source is None:
        raise ValueError("Cannot determine sink without source.");
      else:
        dtype = source.dtype;  
    if order is None and source is not None:
      order = io.order(source);
  #print('shape=%r, dtype=%r, order=%r, memory=%r, location=%r' % (shape, dtype, order, memory, location));
        
  sink = io.prepare_sink(sink, shape=shape, dtype=dtype, order=order, memory=memory, location=location);
  
  if sink.dtype == bool:
    sink = sink.view('uint8');  

  if return_strides:
    strides = io.element_strides(sink);
  
  if return_1d:
    sink1d = sink.reshape(-1, order = 'A');
  
  if not return_strides and not return_1d:
    return sink;

  result = (sink,)
  if return_strides:
    result += (strides,);
  if return_1d:
    result += (sink1d,);
  return result;



###############################################################################
### Tests
###############################################################################

def _test():
  import numpy as np  
  import ClearMap.ParallelProcessing.DataProcessing.ArrayProcessing as ap 
  
  from importlib import reload
  reload(ap)
  
  
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
  lut = sm.initialize_lookup_table()
    
  data = np.random.randint(0, 2, (1500,300,400), dtype = bool)
  
  #reload(ap)
  result = ap.apply_lut_to_index(data, kernel, lut, sink=None, verbose=True)

  import ClearMap.Visualization.Plot3d as p3d
  p3d.plot([data, result])    
  
  
  ## Correlation 
  
  #correlate1d
  #reload(ap)
  axis = 1;
  kernel = np.array(range(11), dtype='uint32');  
  data = np.random.randint(0, 2**27, (1000, 1500,100), dtype='uint32');

  corr = ap.correlate1d(data, kernel, axis=axis, verbose=True, processes=10);
  
  import scipy.ndimage as ndi
  import ClearMap.Utils.Timer as tmr
  timer = tmr.Timer();
  corr_ndi = ndi.correlate1d(data, kernel, axis=axis, mode='constant',cval=0);
  timer.print_elapsed_time('ndi')  
  
  assert np.allclose(corr, corr_ndi)
  
  

#
#
#
#
#
#default_blocks_per_process = 10;
#"""Default number of blocks per process to split the data.
#
#Note
#----
#10 blocks per process is a good choice.
#"""
#
#default_cutoff = 20000000;
#"""Default size of array below which ordinary numpy is used.
#
#Note
#----
#Ideally test this on your machine for different array sizes.
#"""
#
#
#
#def blockRanges(data, blocks = None,  processes = defaultProcesses):
#  """Ranges of evenly spaced blocks in array
#  
#  Arguments:
#    data : array
#      array to divide in blocks
#    blocks : int or None
#      number of blocks to split array into
#    processes : None or int
#      number of processes, if None use number of cpus
#    
#  Returns:
#    array
#      list of the range boundaries
#  """
#  if processes is None:
#    processes = defaultProcesses;
#  if blocks is None:
#    blocks = processes * defaultBlocksPerProcess;
#   
#  d = data.reshape(-1, order = 'A'); 
#  blocks = min(blocks, d.shape[0]);
#  return np.array(np.linspace(0,  d.shape[0], blocks + 1), dtype = int);
#
#
#def blockSums(data, blocks = None, processes = defaultProcesses):
#  """Sums of evenly spaced blocks in array
#  
#  Arguments:
#    data : array
#      array to perform the block sums on
#    blocks : int or None
#      number of blocks to split array into
#    processes : None or int
#      number of processes, if None use number of cpus
#    
#  Returns:
#    array
#      sums of the values in the different blocks
#  """
#  if processes is None:
#    processes = defaultProcesses;
#  if blocks is None:
#    blocks = processes * defaultBlocksPerProcess;
#  
#  d = data.reshape(-1, order = 'A');
#  if data.dtype == bool:
#    d = d.view('uint8')
#  
#  return code.blockSums1d(d, blocks = blocks, processes = processes);
#  
#
#def where(data, out = None, blocks = None, cutoff = defaultCutoff, processes = defaultProcesses):
#  """Returns the indices of the non-zero entries of the array
#  
#  Arguments:
#    data : array
#      array to search for nonzero indices
#    out : array or None
#      if not None results is written into this array
#    blocks : int or None
#      number of blocks to split array into for parallel processing
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
#    Uses numpy.where if there is no match of dimension implemented!
#  """ 
#  if data.ndim != 1 and data.ndim != 3:
#    raise Warning('Using numpy where for dimension %d and type %s!' % (data.ndim, data.dtype))
#    return np.vstack(np.where(data)).T;
#
#  if cutoff is None:
#    cutoff = 1;
#  cutoff = min(1, cutoff);
#  if data.size <= cutoff:
#    return np.vstack(np.where(data)).T;
#
#  if processes is None:
#    processes = defaultProcesses;
#  if blocks is None:
#    blocks = processes * defaultBlocksPerProcess;
#  
#  if data.dtype == bool:
#    d = data.view('uint8')
#  else:
#    d = data;
#  
#  if out is None:
#    if d.ndim == 1:
#      sums = code.blockSums1d(d, blocks = blocks, processes = processes);
#    else:
#      sums = code.blockSums3d(d, blocks = blocks, processes = processes);
#    out = np.squeeze(np.zeros((np.sum(sums), data.ndim), dtype = np.int));
#  else:
#    sums = None;
#  
#  if d.ndim == 1:
#    code.where1d(d, out = out, sums = sums, blocks = blocks, processes = processes);
#  else: # d.ndim == 3:
#    code.where3d(d, out = out, sums = sums, blocks = blocks, processes = processes);
#    
#  return out;
#
#
#
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
# 
# Loading and saving
#
#def readNumpyHeader(filename):
#  """Read numpy array information including offset to data
#  
#  Arguments:
#    filename : str
#      file name of the numpy file
#      
#  Returns:
#    shape : tuple
#      shape of the array
#    dtype : dtype
#      data type of array 
#    order : str
#      'C' for c and 'F' for fortran order
#    offset : int
#      offset in bytes to data buffer in file
#  """
#  with open(filename, 'rb') as fhandle:
#    major, minor = np.lib.format.read_magic(fhandle);
#    shape, fortran, dtype = np.lib.format.read_array_header_1_0(fhandle);
#    offset = fhandle.tell()
#  
#  order = 'C';
#  if fortran:
#    order = 'F';
#    
#  return (shape, dtype, order, offset)
# 
# 
#def _offsetFromSlice(sourceSlice, order = 'F'):
#  """Checks if slice is compatible with the large data loader and returns z coordiante"""
#   
#  if order == 'C':
#    os = 1; oe = 3; oi = 0;
#  else:
#    os = 0; oe = 2; oi = 2;
#  
#  for s in sourceSlice[os:oe]:
#    if s.start is not None or s.stop is not None or s.step is not None:
#        raise RuntimeError('sub-regions other than in slowest dimension %d not supported!  slice = %r' % (oi, sourceSlice))
#  
#  s = sourceSlice[oi];
#  if s.step is not None:
#      raise RuntimeError('sub-regions with non unity steps not supported')
#  
#  if s.start is None:
#    s = 0;
#  else:
#    s = s.start;
#    
#  return s;
#
#
#def load(filename, region = None, shared = False, blocks = None, processes = cpu_count(), verbose = False):
#  """Load a large npy array into memory in parallel
#  
#  Arguments:
#    filename : str
#      filename of array to load
#    region : Region or None
#      if not None this specifies the sub-region to read
#    shared : bool
#      if True read into shared memory
#    blocks : int or None
#      number of blocks to split array into for parallel processing
#    processes : None or int
#      number of processes, if None use number of cpus
#    verbose : bool
#      print info about the file to be loaded
#    
#  Returns:
#    array 
#      the data as numpy array
#  """
#  if processes is None:
#    processes = cpu_count();
#  if blocks is None:
#    blocks = processes * defaultBlocksPerProcess;
#  
#  #get specs from header specs
#  shape, dtype, order, offset = readNumpyHeader(filename);
#  if verbose:
#    timer = tmr.Timer();
#    print('Loading array of shape = %r, dtype = %r, order = %r, offset = %r' %(shape, dtype, order, offset)); 
#  
#  if region is not None:
#    shape = region.shape();  
#    sourceSlice = region.sourceSlice();
#    off = _offsetFromSlice(sourceSlice, order = order);
#  
#  if shared:
#    data = shm.create(shape, dtype = dtype, order = order);
#  else:
#    data = np.empty(shape, dtype = dtype, order = order);
#  
#  d = data.reshape(-1, order = 'A');
#  if dtype == bool:
#    d = d.view('uint8');  
#  
#  if region is not None:
#    if order == 'F':
#      offset += data.strides[-1] * off;  
#    else:
#      offset += data.strides[1] * off;  
#  
#  code.load(data = d, filename = filename, offset = offset, blocks = blocks, processes = processes);
#  
#  if verbose:
#    timer.printElapsedTime(head = 'Loading array from %s' % filename);
#           
#  return data;
#
#
#
#
#def save(filename, data, region = None, blocks = None, processes = cpu_count(), verbose = False):
#  """Save a large npy array to disk in parallel
#  
#  Arguments:
#    filename : str
#      filename of array to load
#    data : array
#      array to save to disk
#    blocks : int or None
#      number of blocks to split array into for parallel processing
#    processes : None or int
#      number of processes, if None use number of cpus
#    verbose : bool
#      print info about the file to be loaded
#    
#  Returns:
#    str 
#      the filename of the numpy array on disk
#  """
#  if processes is None:
#    processes = cpu_count();
#  if blocks is None:
#    blocks = processes * defaultBlocksPerProcess;
#  
#  if region is None:
#    #create file on disk via memmap
#    memmap = np.lib.format.open_memmap(filename, mode = 'w+', shape = data.shape, dtype = data.dtype, fortran_order = np.isfortran(data));
#    memmap.flush();
#    del(memmap);
#  
#  #get specs from header specs
#  shape, dtype, order, offset = readNumpyHeader(filename);
#  if verbose:
#    timer = tmr.Timer();
#    print('Saving array of shape = %r, dtype = %r, order = %r, offset = %r' %(shape, dtype, order, offset)); 
#  
#  if (np.isfortran(data) and order != 'F') or (not np.isfortran(data) and order != 'C'):
#    raise RuntimeError('Order of arrays do not match isfortran=%r and order=%s' % (np.isfortran(data), order));
#  
#  d = data.reshape(-1, order = 'A');
#  if dtype == bool:
#    d = d.view('uint8');
#    
#  if region is not None:
#    sourceSlice = region.sourceSlice();
#    off = _offsetFromSlice(sourceSlice, order = order);
#    if order == 'F':
#      offset += data.strides[-1] * off;
#    else:
#      offset += data.strides[1] * off;
#  
#  #print d.dtype, filename, offset, blocks, processes
#  
#  code.save(data = d, filename = filename, offset = offset, blocks = blocks, processes = processes);
#  
#  if verbose:
#    timer.printElapsedTime(head = 'Saving array to %s' % filename);
#           
#  return filename;
#
#
#
#
#
#
#if __name__ == "__main__":
#  
#  import numpy as np
#  from ClearMap.Utils.Timer import Timer;
#  import ClearMap.DataProcessing.LargeData as ld
#  reload(ld)
#  
#  
#  #dat = np.random.rand(2000,2000,1000) > 0.5;
#  #dat = np.random.rand(1000,1000,500) > 0.5;
#  dat = np.random.rand(200,300,400) > 0.5;  
#  #datan = io.MMP.writeData('test.npy', dat);
#  
#  dat = np.load('data.npy')
#  xyz1 = np.load('points.npy')
#  
#  s = ld.sum(dat)
#  print(s == np.sum(s))
#
#
#  timer = Timer();
#  xyz = ld.where(dat)
#  timer.printElapsedTime('parallel')
#  #parallel: elapsed time: 0:00:25.807
#  
#  timer = Timer();
#  xyz1 = np.vstack(np.where(dat)).T
#  timer.printElapsedTime('numpy')
#  #numpy: elapsed time: 0:05:45.590
#  
#  
#  d0 = np.zeros(dat.shape, dtype = bool);
#  d1 = np.zeros(dat.shape, dtype = bool);
#  
#  d0[xyz[:,0], xyz[:,1], xyz[:,2]] = True;
#  d1[xyz1[:,0], xyz1[:,1], xyz1[:,2]] = True;
#  np.all(d0 == d1)
#  
#  dat2 = np.array(np.random.rand(1000, 1000, 1000) > 0, dtype = 'bool');
#  filename = 'test.npy';
#  np.save(filename, dat2)
#  
#  filename = '/disque/raid/vasculature/4X-test2/170824_IgG_2/170824_IgG_16-23-46/rank_threshold.npy'
#  
#  timer = Timer();
#  ldat = ld.load(filename, verbose = True);
#  timer.printElapsedTime('load')
#  #load: elapsed time: 0:00:04.867
#  
#  timer = Timer(); 
#  ldat2 = np.load(filename);  
#  timer.printElapsedTime('numpy')
#  #numpy: elapsed time: 0:00:27.982
#  
#  np.all(ldat == ldat2)
#  
#  timer = Timer();
#  xyz = ld.where(ldat)
#  timer.printElapsedTime('parallel')
#  #parallel: elapsed time: 0:07:25.698
#  
#  lldat = ldat.reshape(-1, order = 'A')
#  timer = Timer();
#  xyz = ld.where(lldat)
#  timer.printElapsedTime('parallel 1d')
#  #parallel 1d: elapsed time: 0:00:49.034
#  
#  timer = Timer();
#  xyz = np.where(ldat)
#  timer.printElapsedTime('numpy')
#  
#  
#  import os
#  #os.remove(filename)
#  
#  filename = './ClearMap/Test/Skeletonization/test_bin.npy';
#  timer = Timer();
#  ldat = ld.load(filename, shared = True, verbose = True);
#  timer.printElapsedTime('load')
#  
#  ld.shm.isShared(ldat);
#  
#  
#  
#  import numpy as np
#  from ClearMap.Utils.Timer import Timer;
#  import ClearMap.DataProcessing.LargeData as ld
#  reload(ld)
#  
#  filename = 'test_save.npy';
#  
#  dat = np.random.rand(100,200,100);
#  
#  ld.save(filename, dat)
#  
#  
#  dat2 = ld.load(filename)
#  
#  np.all(dat == dat2)
#  
#  os.remove(filename)
#  
#  
#    
#  import numpy as np
#  from ClearMap.Utils.Timer import Timer;
#  import ClearMap.DataProcessing.LargeData as ld
#  reload(ld)
#  
#  dat = np.zeros(100, dtype = bool);
#  dat2 = dat.copy();
#  
#  indices = np.array([5,6,7,8,13,42])  
#  
#  ld.setValue(dat, indices, True, cutoff = 0);
#  
#  dat2[indices] = True;
#  np.all(dat2 == dat)
#  
#  d = ld.take(dat, indices, cutoff = 0)
#  np.all(d)
#  
#  
#  import numpy as np
#  from ClearMap.Utils.Timer import Timer;
#  import ClearMap.DataProcessing.LargeData as ld
#  reload(ld)
#  
#  
#  pts = np.array([0,1,5,6,10,11], dtype = int);
#  
#  ld.neighbours(pts, -10)
#  
#  
#  import numpy as np
#  from ClearMap.Utils.Timer import Timer;
#  import ClearMap.DataProcessing.LargeData as ld
#  import ClearMap.ImageProcessing.Filter.StructureElement as sel;
#  reload(ld)
#  
#  dat = np.random.rand(30,40,50) > 0.5;
#  mask = sel.structureElement('Disk', (5,5,5));
#  indices = np.where(dat.reshape(-1))[0];
#  c_id = len(indices)/2;
#  c = indices[c_id];
#  xyz = np.unravel_index(c, dat.shape)
#  l = np.array(mask.shape)/2
#  r = np.array(mask.shape) - l;
#  dlo = [max(0,xx-ll) for xx,ll in zip(xyz,l)];
#  dhi = [min(xx+rr,ss) for xx,rr,ss in zip(xyz,r, dat.shape)]
#  mlo = [-min(0,xx-ll) for xx,ll in zip(xyz,l)];
#  mhi = [mm + min(0, ss-xx-rr) for xx,rr,ss,mm in zip(xyz,r, dat.shape, mask.shape)]
#  
#  nbh = dat[dlo[0]:dhi[0], dlo[1]:dhi[1], dlo[2]:dhi[2]];
#  nbhm = np.logical_and(nbh, mask[mlo[0]:mhi[0], mlo[1]:mhi[1], mlo[2]:mhi[2]] > 0);
#  nxyz = np.where(nbhm);
#  nxyz = [nn + dl for nn,dl in zip(nxyz, dlo)];
#  nbi = np.ravel_multi_index(nxyz, dat.shape);
#  
#  nbs = ld.findNeighbours(indices, c_id , dat.shape, dat.strides, mask)
#  
#  nbs.sort();
#  print np.all(nbs == nbi)
#  
#  
#  dat = np.random.rand(30,40,50) > 0.5;
#  indices = np.where(dat.reshape(-1))[0];
#  c_id = len(indices)/2;
#  c = indices[c_id];
#  xyz = np.unravel_index(c, dat.shape)
#  l = np.array([2,2,2]);
#  r = l + 1;
#  dlo = [max(0,xx-ll) for xx,ll in zip(xyz,l)];
#  dhi = [min(xx+rr,ss) for xx,rr,ss in zip(xyz, r, dat.shape)]  
#  nbh = dat[dlo[0]:dhi[0], dlo[1]:dhi[1], dlo[2]:dhi[2]];
#  nxyz = np.where(nbh);
#  nxyz = [nn + dl for nn,dl in zip(nxyz, dlo)];
#  nbi = np.ravel_multi_index(nxyz, dat.shape);
#  
#  nbs = ld.findNeighbours(indices, c_id , dat.shape, dat.strides, tuple(l))
#  
#  nbs.sort();
#  print np.all(nbs == nbi)
#  
#  print nbs
#  print nbi
#  
#  
#  import numpy as np
#  from ClearMap.Utils.Timer import Timer;
#  import ClearMap.DataProcessing.LargeData as ld
#  reload(ld)
#  
#  data = np.random.rand(100);
#  values =np.random.rand(50);
#  indices = np.arange(50);
#  ld.setArray(data, indices, values, cutoff = 1)
#  print np.all(data[:50] == values)
#  
#  import numpy as np
#  from ClearMap.Utils.Timer import Timer;
#  import ClearMap.DataProcessing.LargeData as ld
#  reload(ld)
#  
#  m = np.array([1,3,6,7,10]);
#  i = np.array([1,2,3,4,6,7,8,9]);
#  
#  o = ld.match(m,i)
#  
#  o2 = [np.where(i==l)[0][0] for l in m]
#
#
#  