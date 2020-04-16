#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Module to process data in parallel for large data sets

This strategy allows memory intensive processing of larger data sets
"""
__author__    = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__   = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__copyright__ = 'Copyright 2017 by Christoph Kirst, The Rockefeller University, New York City'

#import sys
#import math
import numpy as np
#import copy
import inspect

from multiprocessing import Pool, cpu_count

import ClearMap.IO as io

import ClearMap.DataProcessing.LargeData as ld

import ClearMap.ParallelProcessing.SharedMemoryManager as smm

import ClearMap.ParallelProcessing.SubStack as sub
import ClearMap.ParallelProcessing.ProcessWriter as pcw

from ClearMap.Utils.Timer import Timer;

from ClearMap.ParallelProcessing.SharedMemory import create, create_like, isShared, asShared, base

#TODO: if array is passed, write to temp file, option for shared mem if passed as shared mem arrays !
#TODO: dont need separate modules for shared mem and memmaps  !
#TODO: cleanup, scalar and vector, any tensor returns ! 

def process(source, sink, function = None, processes = cpu_count(), 
            chunkSizeMax = 100, chunkSizeMin = 30, chunkOverlap = 15,
            chunkOptimization = True, chunkOptimizationSize = all, chunkAxis = 2,
            verbose = False, debug = False, **parameter):
    """Create sub stacks and process a function on it in parallel
    
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
    chunkSizeMax : int
      maximal size of a sub-stack
    chunkSizeMin : int
      minial size of a sub-stack
    chunkOverlap : int
      minimal sub-stack overlap
    chunkOptimization : bool
      optimize chunck sizes to best fit number of processes
    chunkOptimizationSize : bool or all
      if True only decrease the chunk size when optimizing
    verbose : bool
      print information on sub-stack generation
    debug: bool
      switch to sequential processing for debugging purposes
    **parameter
      additional paramameter passed to the processing function
        
    Returns
    -------
      str or array: 
        results of the image processing
    
    Note
    ----
      This implementation so far only supports processing into sinks with the same shape as the source
    """     
    
    # calculate substacks
    subStacks = sub.calculateSubStacks(source, axis = chunkAxis,
                                       processes = processes, sizeMax = chunkSizeMax, sizeMin = chunkSizeMin, overlap = chunkOverlap,
                                       optimization = chunkOptimization, optimizationSize = chunkOptimizationSize, verbose = verbose);
                                       
    #nStacks = len(subStacks);
    
    # loop over st
    timer = Timer();
    
    if not isinstance(source, str) or not isinstance(sink, str):
      raise RuntimeError('LargeDataProcessor assumes source and sink are existing npy files')
    
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
    chunkSizeMax : int
      maximal size of a sub-stack
    chunkSizeMin : int
      minial size of a sub-stack
    chunkOverlap : int
      minimal sub-stack overlap
    chunkOptimization : bool
      optimize chunck sizes to best fit number of processes
    chunkOptimizationSize : bool or all
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
    print("Sink   : %s" % (sink));     
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
    chunkSizeMax : int
      maximal size of a sub-stack
    chunkSizeMin : int
      minial size of a sub-stack
    chunkOverlap : int
      minimal sub-stack overlap
    chunkOptimization : bool
      optimize chunck sizes to best fit number of processes
    chunkOptimizationSize : bool or all
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
                       chunkSizeMax = 40, chunkOverlap = 0, 
                       function = processImg, 
                       verbose = True, debug = False);
    
    res = np.load(sinkFile);                          
    print(np.all(res == processImg(source)))
    
    
    
    
    
    
    