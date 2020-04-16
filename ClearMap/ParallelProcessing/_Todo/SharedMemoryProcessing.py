#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Module to process data in parallel using shared memory
"""
__author__    = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__   = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__copyright__ = 'Copyright 2017 by Christoph Kirst, The Rockefeller University, New York City'

#import sys
#import math
import numpy as np
import inspect

from multiprocessing import Pool, cpu_count

import ClearMap.IO as io

import ClearMap.DataProcessing.LargeData as ld

import ClearMap.ParallelProcessing.SharedMemoryManager as smm
import ClearMap.ParallelProcessing.SubStack as sub
import ClearMap.ParallelProcessing.ProcessWriter as pcw

from ClearMap.Utils.Timer import Timer;

from ClearMap.ParallelProcessing.SharedMemory import create, create_like, isShared, asShared, base

def process(source, sink, function = None, processes = cpu_count(), 
                        chunkSizeMax = 100, chunkSizeMin = 30, chunkOverlap = 15,
                        chunkOptimization = True, chunkOptimizationSize = all, chunkAxis = 2,
                        verbose = False, debug = False, **parameter):
    """Create sub stacks and process a function on it in parallel
    
    Arguments
    ---------
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
    -------
      str or array: 
        results of the image processing
    
    Note
    ----
      The sub-stacks created are assumed to have the same shape as the input
    """     
    
    # create shared memory object and load data
    if isinstance(source, str):
      source = ld.load(source, shared = True);
    else:
      source = smm.asShared(source);
      
    sink = smm.asShared(sink);
    
    subStacks = sub.calculateSubStacks(source, axis = chunkAxis,
                                       processes = processes, sizeMax = chunkSizeMax, sizeMin = chunkSizeMin, overlap = chunkOverlap,
                                       optimization = chunkOptimization, optimizationSize = chunkOptimizationSize, verbose = verbose);
    
    if debug:
      return sequentiallyProcess(source, sink, subStacks, function, processes = processes, verbose = verbose, **parameter);
    else:
      return parallelProcess(source, sink, subStacks, function, processes = processes, verbose = verbose, **parameter);






#define the subroutine for the processing
def _processSubStack(args):
    """Helper to process stack in parallel"""

    function, parameter, sourcehandle, sinkhandle, sid, nSubStacks, sourceSlice, validSourceSlice, validSlice, verbose = args;
    
    pw = pcw.ProcessWriter(sid);
    
    if verbose:
        pw.write("Processing substack %d / %d" % (sid, nSubStacks));
        pw.write("function  = %r" % function);
        pw.write("sub stack = %s" % str(sourceSlice)); 
    
    #img = sub.readData();
    source = smm.get(sourcehandle)[sourceSlice];

    timer = Timer();
    if 'out' in inspect.getargspec(function).args:        
      result = function(source, out = pw, **parameter);
    else:
      result = function(source, **parameter)
    
    if verbose:    
      pw.write(timer.elapsedTime(head = 'Processed substack of size ' + str(source.shape)));
    
    
    sink = smm.get(sinkhandle);
    sink[validSourceSlice] = result[validSlice];   
    #sub.writeValidData(sink = sink, data = seg);
    
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
    print("Parallel Processing on Shared Memory:");
    print("Source size: %s, Sink size %s" % (source.shape, sink.shape));
    print("Number of SubStacks: %d" % nSubStacks);
  
  sourceHandle = smm.insert(source);
  sinkHandle   = smm.insert(sink);    
  
  argdata = [];
  for i in range(nSubStacks):
      argdata.append((function, parameter, sourceHandle, sinkHandle, i, nSubStacks, subStacks[i].sourceSlice(), subStacks[i].validSourceSlice(), subStacks[i].validSlice(), verbose));    
  #print argdata
  
  # process in parallel
  pool = Pool(processes = processes);    
  results = pool.map(_processSubStack, argdata);
  pool.close();
  pool.join();
  
  smm.free(sourceHandle);
  smm.free(sinkHandle);
  
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
    print("Source size: %s, Sink size %s" % (source.shape, sink.shape));
    print("Number of SubStacks: %d" % nSubStacks);
  
  sourceHandle = smm.insert(source);
  sinkHandle   = smm.insert(sink);    
  
  argdata = [];
  for i in range(nSubStacks):
      argdata.append((function, parameter, sourceHandle, sinkHandle, i, nSubStacks, subStacks[i].sourceSlice(), subStacks[i].validSourceSlice(), subStacks[i].validSlice(), verbose));    
  #print argdata
  
  # process in sequentially
  results = [_processSubStack(a) for a in argdata];
  
  smm.free(sourceHandle);
  smm.free(sinkHandle);

  return sink;



if __name__ == "__main__":
    import numpy as np
    import ClearMap.ParallelProcessing.SharedMemoryProcessing as smp
    reload(smp)
    
    source = np.random.rand(20,10,1000);
    sink = np.zeros((20,10,1000));
    
    def processImg(img, out = None):
      return img * 5;
    
    reload(smp)
    res = smp.process(source, sink, processes = 4, chunkSizeMax = 40, chunkOverlap = 0, verbose = True,
                             function = processImg, debug = False);
                             
    print(np.all(res == processImg(source)))
    
    
    
    
    
    
    