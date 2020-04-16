# -*- coding: utf-8 -*-
"""
Module to apply functions to data in parallel.

This strategy allows memory intensive processing of larger data sets
"""
__author__    = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__   = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__copyright__ = 'Copyright 2018 by Christoph Kirst, The Rockefeller University, New York City'


import numpy as np
import inspect

from multiprocessing import Pool, cpu_count

import ClearMap.IO as io

import ClearMap.DataProcessing.LargeData as ld

#import ClearMap.ParallelProcessing.SharedMemoryManager as smm

import ClearMap.ParallelProcessing.SubStack as sub
import ClearMap.ParallelProcessing.ProcessWriter as pcw

from ClearMap.Utils.Timer import Timer;

#from ClearMap.ParallelProcessing.SharedMemory_Future import create, create_like, is_shared, as_shared, base

#TODO: cleanup, scalar and vector, any tensor returns ! 

def process(source, sink, function = None,
            chunk_size_max = 100, chunk_size_min = 30, chunk_overlap = 15,
            chunk_optimization = True, chunk_optimization_size = all, chunk_axis = 2,
            processes = None, verbose = False, 
            **parameter):
  """Create sub stacks and process a function on it in parallel.
  
  Arguments
  ---------
  source : str
    The data source.
  sink : str or None
    The sink to write the resukt to, if None return as array.
  function : function
    The main data processing script.
  processes : int
    The number of parallel processes, if 'serial', us serial processing.
  chunk_size_max : int
    Maximal size of a sub-stack.
  chunk_size_min : int
    minial size of a sub-stack
  chunk_overlap : int
    Minimal sub-stack overlap
  chunk_optimization : bool
    Optimize chunck sizes to best fit number of processes
  chunk_optimization_size : bool or all
    If True, only decrease the chunk size when optimizing.
  verbose : bool
    Print information on sub-stack generation.
  **parameter
    additional paramameter passed to the processing function
      
  Returns
  -------
    str or array: 
      The results of the processing.
  """     
    
  # calculate substacks
  subStacks = sub.calculateSubStacks(source, axis = chunk_axis,
                                     processes = processes, sizeMax = chunk_size_max, sizeMin = chunk_size_min, overlap = chunk_overlap,
                                     optimization = chunk_optimization, optimizationSize = chunk_optimization_size, verbose = verbose);
  
  # loop over st
  timer = Timer();
    
  if not isinstance(source, str) or not isinstance(sink, str):
    raise RuntimeError('LargeDataProcessor assumes source and sink are existing npy files')
 
  nSubStacks = len(subStacks);

  if verbose:
    print("Parallel processing batch in shared memory:");
    print("Source : %s" % (source));
    print("Number of substacks: %d" % nSubStacks);
  
  argdata = [];
  for i in range(nSubStacks):
      argdata.append((function, parameter, source, sink, i, nSubStacks, subStacks[i].sourceSlice(), subStacks[i].validSourceSlice(simplify = True), subStacks[i].validSlice(), verbose));    
  #print argdata
  
  if processes is None or processes is all:
    processes = cpu_count();
  
  if processes is 'serial':
     results = [_]
    
  # process in parallel
  pool = Pool(processes = processes);    
  pool.map_async(_processSubStack, argdata);
  pool.close();
  pool.join();
  
  return sink;
      
    
    # process batch
    if debug:
      sequentiallyProcess(source, sink, subStacks, function, processes = processes, verbose = verbose, **parameter);
    else:
      parallelProcess(source, sink, subStacks, function, processes = processes, verbose = verbose, **parameter);

    if verbose:
      timer.printElapsedTime("Processed large data");


#define the subroutine for the processing
def _processSubStack(args):
  """Helper to process stack in parallel"""

  function, parameter, source, sink, sid, nSubStacks, sourceSlice, validSourceSlice, validSlice, verbose = args;
  
  pw = pcw.ProcessWriter(sid);
  
  if verbose:
    pw.write("Processing substack %d / %d" % (sid, nSubStacks));
    pw.write("function  = %r" % function.__name__);
    pw.write("sub stack = %s" % str(sourceSlice)); 
  
  #img = sub.readData();
  source = ld.load(source, region = io.Region.Region(region = sourceSlice, source = source));
  
  #print source.flags;
  
  if verbose:
    pw.write("Loaded source stack of size %r" % (source.shape,)); 

  timer = Timer();
  if 'out' in inspect.getargspec(function).args:
    result = function(source, out = pw, **parameter);
  else:
    result = function(source, **parameter)
  
  
  #print result.flags;
  
  if verbose:    
    pw.write(timer.elapsedTime(head = 'Processed substack of size ' + str(source.shape)));
  
  #print result.flags
  #print validSlice
  #print result[validSlice].flags
  
  ld.save(sink, result[validSlice], region = io.Region.Region(region = validSourceSlice, source = sink))
  
  if verbose:    
    pw.write(timer.elapsedTime(head = 'Wrote processed valid substack of size ' + str(validSourceSlice)));
  
  return True;



def parallelProcess(source, sink, subStacks, function, processes = cpu_count(), verbose = False, **parameter):                                     
  """Parallel process a fucntion from a shared memory to another
        
  Arguments
    source : str
      image source
    sink : str or None
      destination for the result
    function : function
      the main image processing script
    processes : int
      number of parallel processes
    chunk_size_max : int
      maximal size of a sub-stack
    chunk_size_min : int
      minial size of a sub-stack
    chunk_overlap : int
      minimal sub-stack overlap
    chunk_optimization : bool
      optimize chunck sizes to best fit number of processes
    chunk_optimization_size : bool or all
      if True only decrease the chunk size when optimizing
    verbose : bool
      print information on sub-stack generation
    debug: bool
      switch to sequential processing for debugging purposes
    **parameter
      additional paramameter passed to the processing function
        
    Returns
        str or array: 
          results of the image processing
    
    Note
      The sub-stacks created are assumed to have the same shape as the input
  """     
  nSubStacks = len(subStacks);

  if verbose:
    print("Parallel processing batch in shared memory:");
    print("Source : %s" % (source));
    print("Number of substacks: %d" % nSubStacks);
  
  argdata = [];
  for i in range(nSubStacks):
      argdata.append((function, parameter, source, sink, i, nSubStacks, subStacks[i].sourceSlice(), subStacks[i].validSourceSlice(simplify = True), subStacks[i].validSlice(), verbose));    
  #print argdata
  
  # process in parallel
  pool = Pool(processes = processes);    
  pool.map_async(_processSubStack, argdata);
  pool.close();
  pool.join();
  
  return sink;


def sequentiallyProcess(source, sink, subStacks, function, processes = cpu_count(), verbose = False, **parameter):                                     
  """Parallel process a fucntion from a shared memory to another
        
  Arguments
    source : str
      image source
    sink : str or None
      destination for the result
    function : function
      the main image processing script
    processes : int
      number of parallel processes
    chunk_size_max : int
      maximal size of a sub-stack
    chunk_size_min : int
      minial size of a sub-stack
    chunk_overlap : int
      minimal sub-stack overlap
    chunk_optimization : bool
      optimize chunck sizes to best fit number of processes
    chunk_optimization_size : bool or all
      if True only decrease the chunk size when optimizing
    verbose : bool
      print information on sub-stack generation
    debug: bool
      switch to sequential processing for debugging purposes
    **parameter
      additional paramameter passed to the processing function
        
    Returns
        str or array: 
          results of the image processing
    
    Note
      The sub-stacks created are assumed to have the same shape as the input
  """     
  nSubStacks = len(subStacks);

  if verbose:
    print("Sequential Processing on Shared Memory:");
    print("Source: %s, Sink %s" % (source, sink));
    print("Number of SubStacks: %d" % nSubStacks);
  
  argdata = [];
  for i in range(nSubStacks):
      argdata.append((function, parameter, source, sink, i, nSubStacks, subStacks[i].sourceSlice(), subStacks[i].validSourceSlice(simplify = True), subStacks[i].validSlice(), verbose));    
  #print argdata
  
  # process in sequentially
  results = [_processSubStack(a) for a in argdata];
  
  return sink;



if __name__ == "__main__":
    import numpy as np
    import ClearMap.ParallelProcessing.LargeDataProcessing as ldp
    reload(ldp)
    
    sourceFile = 'source.npy';
    sinkFile = 'sink.npy';
    source = np.asarray(np.random.rand(20,10,1000), order = 'F');
    sink = np.asarray(np.zeros((20,10,1000)), order = 'F');
    
    np.save(sourceFile, source);
    np.save(sinkFile, sink);
    
    def processImg(img, out = None):
      return img * 5;
    
    reload(ldp)
    ldp.process(sourceFile, sinkFile, processes = 4, 
                       chunk_size_max = 40, chunk_overlap = 0, 
                       function = processImg, 
                       verbose = True, debug = False);
    
    res = np.load(sinkFile);                          
    print(np.all(res == processImg(source)))
    
    
    
    
    
    
    