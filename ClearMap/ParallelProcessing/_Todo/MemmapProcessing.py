#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Process Volumetric data on memmory maps in parallel
"""
__author__    = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__   = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__copyright__ = 'Copyright (c) 2017 by Christoph Kirst, The Rockefeller University, New York City'

import sys
import math
import numpy as np

import numbers

from multiprocessing import Pool, cpu_count

import ClearMap.IO as io
import ClearMap.IO.MMP

from ClearMap.IO.Region import Region

from ClearMap.ImageProcessing.StackProcessing import calculateChunkSize

from ClearMap.Utils.ParameterTools import writeParameter
from ClearMap.Utils.ProcessWriter import ProcessWriter;
from ClearMap.Utils.Timer import Timer;




#define the subroutine for the processing
def _processSubStack(dsr):
    """Helper to process stack in parallel"""

    sf  = dsr[0];
    pp  = dsr[1];
    sub = dsr[2];
    verbose = dsr[3];
    sink = dsr[4];

    pw = ProcessWriter(sub.id);
    
    if verbose:
        pw.write("Processing substack " + str(sub.id) + "/" + str(sub.nStacks));
        #pw.write("file          = " + sub["source"]);
        pw.write("function  = %s" % str(sf));
        pw.write("sub stack = %s" % str(sub)); 
    
    img = sub.readData();
    
    timer = Timer();
    seg = sf(img, out = pw, **pp);    
    
    if verbose:    
        pw.write(timer.elapsedTime(head = 'Processed substack of size ' + str(img.shape)));
    
    #print seg.shape, sink
    sub.writeValidData(sink = sink, data = seg);
    
    
    if verbose:    
      pw.write(timer.elapsedTime(head = 'Wrote processed valid substack of size ' + str(sub.validSize())));
    
    return True;



def parallelProcessMemmap(source, sink,
                          processes = 2, chunkSizeMax = 100, chunkSizeMin = 30, chunkOverlap = 15,
                          chunkOptimization = True, chunkOptimizationSize = all, chunkAxis = 2,
                          function = None, verbose = False, debug = False, **parameter):
    """Parallel process an image stack
    
    Main routine that distributes image processing on paralllel processes.
       
    Arguments
        source : str
          image source
        sink : str or None
          destination for the result
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
        function : function
          the main image processing script
        join : function
          the fuction to join the results from the image processing script
        verbose : bool
          print information on sub-stack generation
        
    Returns
        str or array: 
          results of the image processing
    """     
    
    # debug in sequential mode
    if debug:
      return sequentiallyProcessMemmap(source, sink,
                          processes = processes, chunkSizeMax = chunkSizeMax, chunkSizeMin = chunkSizeMin, chunkOverlap = chunkOverlap,
                          chunkOptimization = chunkOptimization, chunkOptimizationSize = chunkOptimizationSize, chunkAxis = chunkAxis,
                          function = function, verbose = verbose, **parameter);
    
    
    subStacks = calculateSubStacks(source, axis = chunkAxis,
                                   processes = processes, chunkSizeMax = chunkSizeMax, chunkSizeMin = chunkSizeMin, chunkOverlap = chunkOverlap,
                                   chunkOptimization = chunkOptimization, chunkOptimizationSize = chunkOptimizationSize, verbose = verbose);
                                   
    nSubStacks = len(subStacks);
    if verbose:
        print("Parallel Memmap Processing:");
        if isinstance(source, basestring):
          print("Source: %s" % source);
        if isinstance(sink, basestring):
          print("Sink  : %s" % sink);
        print("Number of SubStacks: %d" % nSubStacks);
                                       
    #for i in range(nSubStacks):
    #    self.printSubStackInfo(subStacks[i]);
    
    argdata = [];
    for i in range(nSubStacks):
        argdata.append((function, parameter, subStacks[i], verbose, sink));    
    #print argdata
    
    # process in parallel
    pool = Pool(processes = processes);    
    results = pool.map(_processSubStack, argdata);
    pool.close();
    pool.join();
    
    #print '=========== results';
    #print results;
        
    #join the results
    #results = join(results, subStacks = subStacks, **parameter);
    
    #write / or return 
    #return io.writePoints(sink, results);
    
    return results;
  


def sequentiallyProcessMemmap(source, sink,
                          processes = 2, chunkSizeMax = 100, chunkSizeMin = 30, chunkOverlap = 15,
                          chunkOptimization = True, chunkOptimizationSize = all, chunkAxis = 2,
                          function = None, verbose = False, **parameter):
    """Parallel process an image stack
    
    Main routine that distributes image processing on paralllel processes.
       
    Parameters
    ----------
        source : str
          image source
        sink : str or None
          destination for the result
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
        function : function
          the main image processing script
        join : function
          the fuction to join the results from the image processing script
        verbose : bool
          print information on sub-stack generation
        
    Returns
    -------
        str or array: results of the image processing
    """     
    
    subStacks = calculateSubStacks(source, axis = chunkAxis,
                                   processes = processes, chunkSizeMax = chunkSizeMax, chunkSizeMin = chunkSizeMin, chunkOverlap = chunkOverlap,
                                   chunkOptimization = chunkOptimization, chunkOptimizationSize = chunkOptimizationSize, verbose = verbose);
                                   
    nSubStacks = len(subStacks);
    if verbose:
        print("Number of SubStacks: %d" % nSubStacks);
                                       
    #for i in range(nSubStacks):
    #    self.printSubStackInfo(subStacks[i]);
    
    argdata = [];
    for i in range(nSubStacks):
        argdata.append((function, parameter, subStacks[i], verbose, sink));    
    #print argdata
    
    # process in sequentially
    results = [_processSubStack(a) for a in argdata];
    
    #print '=========== results';
    #print results;
        
    #join the results
    #results = join(results, subStacks = subStacks, **parameter);
    
    #write / or return 
    #return io.writePoints(sink, results);
    
    return results;  
    
if __name__ == "__main__":
    import numpy as np
    import ClearMap.ImageProcessing.MemmapProcessing as mp
    import ClearMap.IO as io
    import ClearMap.IO.MMP
    
     
    datan = 'test.npy'
    outn = 'test_out.npy'      
    
    data = io.MMP.create(datan, (100,100,100));
    data[:,:,:] = np.random.rand(100,100,100);
    
    out = io.MMP.create(outn, (100,100,100));
    #data = []; out = [];
    
    substacks = mp.calculateSubStacks(source = datan, processes = 2, chunkSizeMax = 40, chunkOverlap = 10)
    
    
    def processImg(img, out = None):
      return img * 5;
    
    reload(mp)
    res = mp.parallelProcessMemmap(datan, outn, processes = 2, chunkSizeMax = 40, chunkOverlap = 10, verbose = True,
                             function = processImg);
                             
    print(np.any(out != processImg(data)))
    
    import os
    os.remove(datan); os.remove(outn)
    
    
    
    
    
    
    