# -*- coding: utf-8 -*-
"""
BlockProcessing
===============

Module to process data in parallel for large data sets

This strategy allows memory intensive processing of larger data sets.

Example
-------

>>> import numpy as np
>>> import ClearMap.IO.IO as io
>>> import ClearMap.ParallelProcessing.BlockProcessing as bp
>>> source = io.as_source(np.asarray(np.random.rand(50,100,200), order = 'F'))
>>> blocks = bp.split_into_blocks(source, processes=10, axes=[2], size_min=30, size_max=50, overlap=20);
>>> blocks[0]
Block-Numpy-Source(50, 100, 38)[float64]|F|

>>> blocks[0].info()
'0/10<(0, 0, 0)/(1, 1, 10)> (50, 100, 38)@(50, 100, 200)[(:,:,0:38)]'

>>> b.valid
'Sliced-Block-Numpy-Source(50, 100, 28)[float64]|F|'

>>> b = blocks[0];
>>> print(b.valid.base_shape)
>>> print(b.valid.base_slicing)
>>> print(b.iteration)
(50, 100, 200)
(slice(None, None, None), slice(None, None, None), slice(None, 28, None))
0
 
>>> shape = (2,3,20);
>>> source = io.npy.Source(array = np.random.rand(*shape));
>>> sink = io.npy.Source(array = np.zeros(shape))
>>>  
>>> def process_image(source, sink=None):
>>>    if sink is None:
>>>      sink = np.zeros(source.shape);
>>>    sink[:] = 100 * source[:];
>>>    return sink;
>>> 
>>> bp.process(process_image, source, sink,
>>>            processes = 'serial', size_max = 4, size_min = 1, overlap = 0, axes = [2],
>>>            optimization = True, verbose = True);
>>>
>>> print(np.all(sink[:] == process_image(source)))
True
  
>>> bp.process(process_image, source, sink,
>>>            processes = None, size_max = 10, size_min = 6, overlap = 3, axes = all,
>>>            optimization = True, verbose = True);

"""
__author__    = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__   = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__copyright__ = 'Copyright 2020 by Christoph Kirst'


import functools as ft
import multiprocessing as mp
import concurrent.futures as cf
import numpy as np
import gc

import ClearMap.ParallelProcessing.Block as blk
import ClearMap.ParallelProcessing.ParallelTraceback as ptb

import ClearMap.IO.IO as io
import ClearMap.IO.SMA as sma

import ClearMap.Utils.Timer as tmr;


#TODO: clean up block functions: act on sources, arrays or blocks + memory or views
#TODO: integrate with Torch or tensorflow ? GPU processing ?

###############################################################################
### Default parameter
###############################################################################

default_size_max = None
"""Default maximal size of a block.

Note
----
Set this to limit the maximal block sizes automatically if size_max passed 
to :func:`ClearMap.ParallelProcessing.BlockProcessing.process` is None.

If this is None, the full source size will be used.
"""

default_size_min = None
"""Default minimal size of a block.

Note
----
Set this to limit the minimal block sizes automatically if size_min passed 
to :func:`ClearMap.ParallelProcessing.BlockProcessing.process` is None.

If this is None, the full source size will be used.
"""

default_overlap = None
"""Default overlap between blocks.

Note
----
This value is used if overlap passed 
to :func:`ClearMap.ParallelProcessing.BlockProcessing.process` is None.

If this is None, a zero overlap will be used.
"""


###############################################################################
### Processing
###############################################################################

def process(function, source, sink = None,
            axes = None, size_max = None, size_min = None, overlap = None,  
            optimization = True, optimization_fix = 'all', neighbours = False,
            function_type = None, as_memory = False, return_result = False,
            return_blocks = False,
            processes = None, verbose = False, 
            **kwargs):
  """Create blocks and process a function on them in parallel.
  
  Arguments
  ---------
  function : function
    The main data processing script.
  source : str, Source, or list
    The source or list of sources to apply a function to 
  sink : str, Source, list, or None
    The sink or list of sinks to write the result to.
    If None, return single array.
  axes : int, list of ints, or None
    Axes along which to split the source. If None, the 
    splitting is determined automaticlly from the order of the array.
  size_max : int, list of ints or None
    Maximal size of a block along the axes. 
    If None, :const:`default_size_max` is used.
  size_min : int or list of ints
    Minial size of a block along the axes. 
    If None, :const:`default_size_min` is used.
  overlap : int, list of ints or None
    Minimal overlap between blocks along the axes.
    If None, :const:`default_overlap` is used.
  optimization : bool or list of bools
    If True, optimize block sizes to best fit number of processes.
  optimization_fix : 'increase', 'decrease', 'all' or None or list
    Increase, decrease or optimally change the block size when optimization 
    is active.
  neighbours : bool
    If True, also include information about the neighbourhood in the blocks.
  function_type : 'array', 'source', 'block' or None
    The function type passed. If None, 'array' is used.
    
    * 'array'
      Reading and writing the valid slices from the blocks is automatic 
      and the function gets passed numpy arrays.
    * 'source' 
      Reading and writing the valid slices from the blocks is automatic 
      and the function gets passed Source classes as inputs. 
    * 'block' 
      The function is assumed to act on and update blocks itself.
    
  as_memory : bool
    If True, load full blocks into memory before applying the function.
    Can be useful to reduce frequent reading and writing operations of memmaps.
  return_result : bool
    If True, return the results of the proceessing functions.
  return_blocks : bool
    If True, return the block information used to distribute the processing.
  processes : int
    The number of parallel processes, if 'serial', use serial processing.
  verbose : bool
    Print information on sub-stack generation.
      
  Returns
  -------
  sink : str, Source, list or array 
    The results of the processing.
  
  Note
  ----
  This implementation only supports processing into sinks with the same shape as the source.
  """     
  #sources and sinks
  if isinstance(source, list):
    sources = source;
  else:
    sources = [source];
  sources = [io.as_source(s).as_virtual() for s in sources];
  
  #if sink is None:
  #  sink = sma.Source(shape=sources[0].shape, dtype=sources[0].dtype, order=sources[0].order);
  if isinstance(sink, list):
    sinks = sink;
  elif sink is None:
    sinks = [];
  else:
    sinks = [sink];
  
  sinks = [io.initialize(s, hint=sources[0]) for s in sinks];
  sinks = [io.as_source(s).as_virtual() for s in sinks];

  axes = block_axes(sources[0], axes=axes);

  split = ft.partial(split_into_blocks, processes=processes, axes=axes, 
                     size_max=size_max, size_min=size_min, 
                     overlap=overlap, optimization=optimization, 
                     optimization_fix=optimization_fix, neighbours=neighbours,
                     verbose=False); 

  source_blocks = [split(s) for s in sources];
  sink_blocks = [split(s) for s in sinks];
  n_blocks = len(source_blocks[0]);
  
  source_blocks = [[blocks[i] for blocks in source_blocks] for i in range(n_blocks)];  
  sink_blocks =  [[blocks[i] for blocks in sink_blocks] for i in range(n_blocks)];  
  
  if function_type is None:
    function_type = 'array';
  if function_type == 'block':
    func = ft.partial(process_block_block, function=function, as_memory=as_memory, return_result=return_result, verbose=verbose, **kwargs);
  elif function_type == 'source':
    func = ft.partial(process_block_source, function=function, as_memory=as_memory, as_array=False, verbose=verbose, **kwargs);
  elif function_type == 'array':
    func = ft.partial(process_block_source, function=function, as_memory=as_memory, as_array=True, verbose=verbose, **kwargs);
  else:
    raise ValueError("function type %r not 'array', 'source', 'block' or None!");
  
  if not isinstance(processes, int) and processes != "serial":
    processes = mp.cpu_count();
  
  if verbose:
    timer = tmr.Timer();
    print("Processing %d blocks with function %r." % (n_blocks, function.__name__))
  
  if isinstance(processes, int):
    #from bounded_pool_executor import BoundedProcessPoolExecutor
    with cf.ProcessPoolExecutor(max_workers=processes) as executor:
    #with BoundedProcessPoolExecutor(max_workers=processes) as executor:
      futures = [executor.submit(func, *args) for args in zip(source_blocks, sink_blocks)];
      result  = [f.result() for f in futures];
      #executor.map(function, source_blocks, sink_blocks)
  else:
    result = [func(*args) for args in zip(source_blocks, sink_blocks)]; #analysis:ignore
  
  if verbose:
    timer.print_elapsed_time("Processed %d blocks with function %r" % (n_blocks, function.__name__))
  
  #gc.collect();

  if return_result:
    ret = result;
  else:
    ret = sink;
  if return_blocks:
    ret = (ret, [source_blocks, sink_blocks]);
  return ret;


###############################################################################
### Helpers
###############################################################################

@ptb.parallel_traceback
def process_block_source(sources, sinks, function, as_memory = False, as_array = False, verbose = False, **kwargs):
  """Process a block with full traceback.
  
  Arguments
  ---------
  sources :  source specifications
    Sources passed to the function.
  sinks : sourcespecifications
    Sinks where data is written to.
  function  func : function
    The function to call.
  """
  if verbose:
    timer = tmr.Timer();
    print('Processing block %s' % (sources[0].info(),));
  
  #sources = [s.as_real() for s in sources];
  sources_input = sources;
  if as_memory:
    sources = [s.as_memory() for s in sources];
  if as_array:
    sources = [s.array for s in sources];
  
  results = function(*sources, **kwargs);
  if not isinstance(results, (list, tuple)):
    results = [results];
  
  if len(sources_input) != len(sinks):
    sources_input = sources_input + [sources_input[0]] * (len(sinks) - len(sources));
  
  for sink, source, result in zip(sinks, sources_input, results):
    #sink = sink.as_real();
    sink.valid[:] = result[source.valid.slicing];
    
  if verbose:
    timer.print_elapsed_time('Processing block %s' % (sources_input[0].info(),));
   
  gc.collect(); 
    
  return None;


@ptb.parallel_traceback
def process_block_block(sources, sinks, function, as_memory = False, return_result = False, verbose=False, **kwargs):
  """Process a block with full traceback.
  
  Arguments
  ---------
  sources :  source specifications
    Sources passed to the function.
  sinks : sourcespecifications
    Sinks where data is written to.
  function  func : function
    The function to call.
  """
  if verbose:
    timer = tmr.Timer();
    print('Processing block %s' % (sources[0].info(),));

  if as_memory:
    sinks = sinks;
    sinks_memory = [s.as_memory_block() for s in sinks]
    sources_and_sinks = [s.as_memory_block() for s in sources] + sinks_memory;
  else:
    sources_and_sinks = sources + sinks;
  result = function(*sources_and_sinks, **kwargs);
  if as_memory:
    for sink, sink_memory in zip(sinks, sinks_memory):
      sink.valid[:] = sink_memory.valid[:];

  if verbose:
    timer.print_elapsed_time('Processing block %s' % (sources[0].info(),));
   
  gc.collect();
  
  if return_result:
    return result;
  else:
    return None;


###############################################################################
### Source splitting into blocks
###############################################################################

def block_sizes(size, processes = None, 
                size_max = None, size_min = None, overlap = None, 
                optimization = True, optimization_fix = 'all', verbose = False):
  """Calculates the block sizes along a single axis when splitting up a source .
  
  Arguments
  ---------
  size : int 
    Size of the array dimension to be split up.
  processes : int
    Number of parallel processes to use.
  size_max : int or None.
    Maximal size of a block. If None, do not split.
  size_min : int, 'fixed', or None
    Minimal size of a block. If 'fixed' blocks will be of fixed size given by
    size_max and the overlap is increased if the last block is too small.
    If None, the minimal size is determined from the overlap.
  overlap : int or None
    Minimal overlap between blocks in a single axis.
    If None, the overlap defaults to zero. 
  optimization : bool 
    If True, optimize block sizes to best fit number of processes.
  optimization_fix : 'increase', 'decrease', 'all' or None
    Increase, decrease or optimally change the block size when optimization 
    is active.
  verbose : bool
    Print information on block generation.
      
  Returns
  -------
  n_blocks : int
   Number of blocks. 
  block_ranges : list of tuple of ints
    Ranges of the blocks of the form [(lo0,hi0),(lo1,hi1),...].
  valid_ranges : list of tuple of ints
    Valid ranges of the blocks of the form [(lo0,hi0),(lo1,hi1),...].
    
  Note
  ----
  The optimization allows block sizes to change slightly to better distribute
  the blocks over processes, assuming each block processes a similar amount of
  time.
  """
  if processes is None:
    processes = mp.cpu_count();
  if not isinstance(processes, int):
    processes = 1;
  if processes <= 0:
    processes = 1;
   
  if size_max is None or size_max > size: 
    size_max = size;
  
  if overlap is None:
    overlap = 0;  
  if overlap >= size:
    overlap = size - 1;
  
  fixed = False;  
  if size_min == 'fixed':
    size_min = size_max;
    fixed = True;
    optimization = False;
  elif size_min is None:
    size_min = min(size_max, overlap + 1);
    
  if size_min > size:
    size_min = size;
  
  #check consistency
  if size_min > size:
    raise RuntimeError('Minimal block size is larger than the data size %d > %d !' % (size_min, size)); 
  if size_min > size_max:
    raise RuntimeError('Minimal block size larger than maximal block size %d > %d !' % (size_min, size_max));
  if overlap >= size_max:
    raise ValueError('Overlap is larger than maximal block size: %d >= %d!' % (overlap, size_max));
  if overlap >= size_min:
    raise ValueError('Overlap is larger than minimal block size: %d >= %d!' % (overlap, size_min));
  
  #calcualte block size estimates
  block_size = size_max;
  n_blocks = int(np.ceil(float(size - block_size) / (block_size - overlap) + 1)); 
  if n_blocks <= 0:
    n_blocks = 1;   
  if not fixed:
    block_size = float(size + (n_blocks-1) * overlap) / n_blocks;
  
  if verbose:
    print("Estimated block size %d in %d blocks!" % (block_size, n_blocks));
  
  if n_blocks == 1:
    return 1, [(0, size)], [(0, size)]
      
  #optimize number of blocks wrt to number of processors
  if optimization:
    n_add = n_blocks % processes;
    if n_add != 0:
      if optimization_fix in [None, 'all', all]:
        if n_add < processes / 2.0:
          optimization_fix = 'increase';
        else:
          optimization_fix = 'decrease';
                
      if verbose:
        print("Optimizing block size to fit number of processes!")
            
      if optimization_fix == 'decrease':
        #try to deccrease block size / increase block number to fit distribution on processors
        n_blocks = n_blocks - n_add + processes;
        block_size = float(size + (n_blocks-1) * overlap) / n_blocks;
            
        if verbose:
          print("Optimized block size decreased to %d in %d blocks!" % (block_size, n_blocks));
                
      elif optimization_fix == 'decrease' and n_blocks > n_add:
        #try to increase chunk size and decrease chunk number to fit  processors
        n_blocks = n_blocks - n_add;
        block_size = float(size + (n_blocks-1) * overlap) / n_blocks;
                              
        if verbose:
          print("Optimized block size increased to %d in %d blocks!" % (block_size, n_blocks));
            
      else:
        if verbose:
          print("Optimized block size %d unchanged in %d blocks!" % (block_size, n_blocks));
    
    else:
      if verbose:
        print("Block size %d optimal in %d chunks!" % (block_size, n_blocks));
  
  if block_size < size_min:
    #raise Warning("Warning: Some chunks with average chunk size %f.02 may be smaller than minima chunk size %d!" % (chunksize, sizeMin)); 
    if verbose:
      print("Warning: Some blocks with average block size %.02f may be smaller than minimal block size %d due to optimization!" % (block_size, size_min)); 
  if block_size > size_max:
    #raise Warning("Warning: optimized chunk size %f.02 is larger than maximum chunk size %d!" % (chunksize, sizeMax)); 
    if verbose:
      print("Warning: Some blocks with average block size %.02f may be larger than maximum block size %d due to optimization!" % (block_size, size_max)); 
     
  
  #calculate actual block sizes
  block_size_rest = block_size;
  block_size = int(np.floor(block_size));
  block_size_rest = block_size_rest - block_size;
  
  block_ranges = [(0, block_size)]; 
  valid_ranges = [];
  valid_prev = 0;
  sr = block_size_rest;
  hi = block_size;
  n = 1;
  while n < n_blocks:
    n+=1;
    
    #range    
    hi_prev = hi;
    lo = hi - overlap;
    hi = lo + block_size;
    
    sr += block_size_rest;
    if sr >= 1:
      sr -= 1;
      hi += 1;
    
    if n == n_blocks:        
      hi = size;
      if fixed:
        lo = hi - block_size;
    
    block_ranges.append((lo, hi));
    
    #borders
    valid = int(round((hi_prev - lo) / 2. + lo));
    if valid > size:
      valid = size;
    valid_ranges.append((valid_prev, valid)); 
    valid_prev = valid;
  
  valid_ranges.append((valid_prev, size));  
  
  if verbose:
    n_prt = min(10, n_blocks);
    if n_blocks > n_prt:
      pr = '...'
    else:
      pr = '';
    print("Final blocks : %d" % n_blocks);
    print("Final blocks : " + str(block_ranges[:n_prt]) + pr);
    print("Final borders: " + str(valid_ranges[:n_prt]) + pr);
    sizes = np.unique([r[1]- r[0] for r in block_ranges]);
    print("Final sizes  : " + str(sizes));
  
  return n_blocks, block_ranges, valid_ranges;


def block_axes(source, axes = None):
  """Determine the axes for block processing from source order.
  
  Arguments
  ---------
  source : array or Source
    The source on which the block processing is used.
  axes : list or None
    The axes over which to split the block processing.
  
  Returns
  -------
   axes : list or None
    The axes over which to split the block processing.
  """
  if axes is all:
    axes = [d for d in range(source.ndim)];
  if axes is not None:
    if np.max(axes) >= source.ndim or np.min(axes) < 0:
      raise ValueError('Axes specification %r for source with dimnesion %d not valid!' % (axes, source.ndim));
    return axes;
  
  source = io.as_source(source);
  if source.order == 'F':
    axes = [source.ndim-1];
  else:
    axes = [0];
    
  return axes;


def split_into_blocks(source, processes = None, axes = None, 
                      size_max = None, size_min = None, overlap = None,  
                      optimization = True, optimization_fix = 'all', 
                      neighbours = False, verbose = False, **kwargs):
  """splits a source into a list of Block sources for parallel processing.
  
  The block information is described in :mod:`ClearMapBlock`  
  
  Arguments
  ---------
  source : Source 
    Source to divide into blocks.
  processes : int
    Number of parallel processes to use.
  axes : int or list of ints or None
    Axes along which to split the source. If None, all axes are split.
  size_max : int or list of ints
    Maximal size of a block along the axes.
  size_min : int or list of ints
    Minial size of a block along the axes..
  overlap : int or list of ints
    Minimal overlap between blocks along the axes.
  optimization : bool or list of bools
    If True, optimize block sizes to best fit number of processes.
  optimization_fix : 'increase', 'decrease', 'all' or None or list
    Increase, decrease or optimally change the block size when optimization is active.
  neighbours : bool
    If True, also include information about the neighbourhood in the blocks.
  verbose : bool
    Print information on block generation.
      
  Returns
  -------
  blocks : list of Blocks
    List of Block classes dividing the source.
  """
  shape = source.shape;
  ndim = len(shape);  
  
  axes = block_axes(source, axes=axes);
  n_axes = len(axes);
  
  size_max = _unpack(size_max, n_axes);
  size_min = _unpack(size_min, n_axes);
  overlap  = _unpack(overlap, n_axes);
  optimization = _unpack(optimization, n_axes);
  optimization_fix = _unpack(optimization_fix, n_axes);
  #print size_max, size_min, overlap, optimization, optimization_fix
  
  #calculate block shapes
  blocks_shape = tuple();
  blocks_block_ranges = [];
  blocks_offsets = [];
  a = 0;
  for d in range(ndim):
    if d in axes:
      n_blocks, block_ranges, valid_ranges = \
        block_sizes(shape[d], processes=processes, 
                    size_max=size_max[a], size_min=size_min[a], overlap=overlap[a], 
                    optimization=optimization[a], optimization_fix=optimization_fix[a], 
                    verbose=verbose);
      a += 1;
    else:
      n_blocks = 1;
      block_ranges = [(None, None)];
      valid_ranges = [(None, None)];
    #print(d, block_ranges, valid_ranges) 
    
    offsets = [(v[0]-b[0], b[1]-v[1]) if b != (None, None) else (None, None) for b,v in zip(block_ranges, valid_ranges)];     
     
    blocks_shape += (n_blocks,);
    blocks_block_ranges.append(block_ranges);
    blocks_offsets.append(offsets);
  
  #create blocks
  blocks_size = np.prod(blocks_shape);  
  blocks = [];
  index_to_block = {};
  for i in range(blocks_size):
    index = np.unravel_index(i, blocks_shape);
    slicing = tuple(slice(b[0], b[1]) for b in [blocks_block_ranges[d][index[d]] for d in range(ndim)]);
    offsets = [(o[0], o[1]) for o in [blocks_offsets[d][index[d]] for d in range(ndim)]];   
    block = blk.Block(source=source, slicing=slicing, offsets=offsets, index=index, blocks_shape=blocks_shape);
    blocks.append(block);
    
    if neighbours:
      index_to_block[index] = block;
  
  
  if neighbours:
    for b in blocks:
      index = np.array(b.index);
      nbs = {};
      for d,i in enumerate(index):
        if i > 0:
          ii = index.copy(); ii[d] -= 1; ii = tuple(ii);
          nbs[ii] = index_to_block[ii];
        if i < blocks_shape[d] - 1:
          ii = index.copy(); ii[d] += 1; ii = tuple(ii);
          nbs[ii] = index_to_block[ii];
      b._neighbours = nbs;
  
  return blocks;


def _unpack(values, ndim = None):
  """Helper to parse values into standard form (value0,value1,...)."""
  if not isinstance(values, (list, tuple)):
    values = [values] * (ndim or 1);
  
  if ndim is not None and len(values) != ndim:
    raise ValueError('Dimension %d does not match data dimensions %d' % (len(values), ndim));

  return values;
    

###############################################################################
### Tests
###############################################################################

def _test():
  import numpy as np
  import ClearMap.IO.IO as io
  import ClearMap.ParallelProcessing.BlockProcessing as bp
  
  source = io.as_source(np.asarray(np.random.rand(50,100,200), order = 'F'))
  
  blocks = bp.split_into_blocks(source, processes=10, axes=[2], size_min=30, size_max=50, overlap=20);
  print(blocks)
  
  b = blocks[0];
  print(b.valid.base_shape)
  print(b.valid.base_slicing)
  print(blocks[5].iteration)
  
  blocks = bp.split_into_blocks(source, processes=10, axes=[1,2], size_min=30, size_max=50, overlap=20, neighbours=True)
  b = blocks[0]; 
  print(b.valid.base_shape)
  print(b.valid.base_slicing)
  
  
  blocks = bp.split_into_blocks(source, processes=10, axes=[1,2], size_min='fixed', size_max=50, overlap=20, neighbours=True)
  b = blocks[0]; 
  print(b.valid.base_shape)
  print(b.valid.base_slicing)
  
  shape = (2,3,20);
  source = io.npy.Source(array = np.random.rand(*shape));
  sink = io.npy.Source(array = np.zeros(shape))
  
  def process_image(source, sink = None):
    if sink is None:
      sink = np.zeros(source.shape);
    sink[:] = 100 * source[:];
    return sink;
  
  bp.process(process_image, source, sink,
             processes = 'serial', size_max = 4, size_min = 1, overlap = 0, axes = [2],
             optimization = True, verbose = True);
                        
  print(np.all(sink[:] == process_image(source)))
  
  bp.process(process_image, source, sink,
             processes = None, size_max = 10, size_min = 6, overlap = 3, axes = all,
             optimization = True, verbose = True);
                        
  assert(np.all(sink[:] == process_image(source))) 

    
  result, blocks = bp.process(process_image, source, sink,
                              size_max = 15, size_min = 4, overlap = 3, axes = [2], optimization = True, 
                              return_blocks = True, processes = None, verbose = True);
                        

  #memmaps loading
  source = io.mmp.create(location='source.npy', shape=shape);
  source[:] = np.random.rand(*shape);
  sink = io.mmp.create(location='sink.npy', shape=shape)
  
  bp.process(process_image, source, sink,
             size_max = 10, size_min = 6, overlap = 3, axes = [2],
             optimization = True, as_memory=True, verbose = True, processes=None);

  assert(np.all(sink[:] == process_image(source))) 
  
  io.delete_file(source.location)
  io.delete_file(sink.location)

  #multiple sources and sinks
  shape = (2,50,30);
  source1 = io.sma.Source(array = np.random.rand(*shape));
  source2 = io.sma.Source(array = np.random.rand(*shape));
  sink1 = io.sma.Source(array = np.zeros(shape));
  sink2 = io.sma.Source(array = np.zeros(shape));
  
  def sum_and_difference(source1, source2, sink1 = None, sink2 = None):
    if sink1 is None:
      sink1 = np.zeros(source1.shape);
    if sink2 is None:
      sink2 = np.zeros(source2.shape);
    
    sink1[:] = source1[:] + source2[:];
    sink2[:] = source1[:] - source2[:];
    return sink1, sink2;
  
  
  bp.process(sum_and_difference, [source1, source2], [sink1, sink2],
             processes = '!serial', size_max = 10, size_min = 5, overlap = 3, axes = [1,2],
             optimization = True, verbose = True);
       
  s,d = sum_and_difference(source1, source2)                 
  assert(np.all(sink1[:] == s))
  assert(np.all(sink2[:] == d))
  
  
  #trace backs
  shape = (3,4)
  source = io.sma.Source(array = np.random.rand(*shape));
  sink = io.sma.Source(array = np.zeros(shape))
  
  def raise_error(source, sink = None):
    raise RuntimeError('test');

  
  bp.process(raise_error, source, sink,
             processes = '!serial', size_max = 10, size_min = 5, overlap = 0, axes = [2],
             optimization = True, verbose = True);
