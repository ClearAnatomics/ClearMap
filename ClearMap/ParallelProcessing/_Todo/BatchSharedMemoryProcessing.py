#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Module to process data in a two step sequential/parallel fashion for large data sets

This strategy allows memory intensive processing of larger data sets working on batches
of the data loaded into shared memory in parallel but processing the batches sequentially.
"""
__author__    = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__   = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__copyright__ = 'Copyright 2017 by Christoph Kirst, The Rockefeller University, New York City'

#import sys
#import math
import numpy as np
import copy
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
            batchSize = 2, batchOptimization = True,
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
      batchSize : int 
        number of partial steps, each partial step processes ~nchunks/batchSize chunks 
      batchOptimization : bool
        if True distribute chunks more evenly onto batches by decreasing batch sizes
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
                                   
    # group into batches
    nStacks = len(subStacks);
    
    nBatches = float(nStacks) / batchSize;
    if batchOptimization:
      nBatches = int(np.ceil(nBatches));
      batchSizeNew = float(nStacks) / nBatches;
      batchSizeRest = batchSizeNew % 1.0;
      batchSizeNew = int(batchSizeNew);
      
      bpos = [0];
      br = bp = 0;
      for n in range(nBatches):
        br += batchSizeRest;
        if n == nBatches-1:
          bp = nStacks;
        elif br >= 1:
          bp += batchSizeNew + 1;
          br -= 1;
        else:
          bp += batchSizeNew;
        bpos.append(bp);
        
      print bpos
      
      batches = [subStacks[b0:b1] for b0,b1 in zip(bpos[:-1], bpos[1:])]; 
    
    else: # no optimization
      nBatches = np.ceil(nBatches);
      batches = [subStacks[n*batchSize:(n+1)*batchSize] for n in range(nBatches)];
    
    if verbose:
      print("Processing data in %d sub stacks using %d batches" % (nStacks, nBatches));
    
    
    # loop over batches
    timer = Timer();
    for bid, subStacks in enumerate(batches):
      
      if verbose:
        print("Processing batch %d / %d" % (bid, nBatches));
      
      # load source for this batch
      sourceSubStack = sub.joinSubStacks(subStacks);
      batchSource = ld.load(source, shared = True, region = sourceSubStack);
      
      # adjust source regions to actual batch
      batchSubStacks = [];
      for s in subStacks:
        ss = copy.deepcopy(s);
        ss.shift(shift = np.array(0) - sourceSubStack.lowerBound());
        batchSubStacks.append(ss);
      
      # shared memory for sink
      batchSink = create_like(batchSource, dtype = io.dataType(sink));
        
      # process batch
      if debug:
        sequentiallyProcess(batchSource, batchSink, batchSubStacks, function, processes = processes, verbose = verbose, **parameter);
      else:
        parallelProcess(batchSource, batchSink, batchSubStacks, function, processes = processes, verbose = verbose, **parameter);

      if verbose:
        timer.printElapsedTime("Processed data of shape %r in shared memory for batch %d / %d " % (batchSource.shape, bid, nBatches));
        timer.reset();

      # save data to sink
      ld.save(sink, batchSink, region = sourceSubStack);
     
      if verbose:
        timer.printElapsedTime("Saved data of shape %r for batch %d / %d " % (batchSink.shape, bid, nBatches));
        timer.reset();



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
    print("Parallel processing batch in shared memory:");
    print("Source size: %r" % (source.shape,));
    print("Number of substacks: %d" % nSubStacks);
  
  sourceHandle = smm.insert(source);
  sinkHandle   = smm.insert(sink);    
  
  argdata = [];
  for i in range(nSubStacks):
      argdata.append((function, parameter, sourceHandle, sinkHandle, i, nSubStacks, subStacks[i].sourceSlice(), subStacks[i].validSourceSlice(), subStacks[i].validSlice(), verbose));    
  #print argdata
  
  # process in parallel
  pool = Pool(processes = processes);    
  pool.map_async(_processSubStack, argdata);
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
    import ClearMap.ParallelProcessing.BatchSharedMemoryProcessing as bsmp
    reload(bsmp)
    
    sourceFile = 'source.npy';
    sinkFile = 'sink.npy';
    source = np.asarray(np.random.rand(20,10,1000), order = 'F');
    sink = np.asarray(np.zeros((20,10,1000)), order = 'F');
    
    np.save(sourceFile, source);
    np.save(sinkFile, sink);
    
    def processImg(img, out = None):
      return img * 5;
    
    reload(bsmp)
    bsmp.process(sourceFile, sinkFile, processes = 4, 
                       chunkSizeMax = 40, chunkOverlap = 0, 
                       batchSize = 4,
                       function = processImg, 
                       verbose = True, debug = False);
    
    res = np.load(sinkFile);                          
    print(np.all(res == processImg(source)))
    
    
    
    
    
    
    