# -*- coding: utf-8 -*-
"""
Process a image stack in parallel or sequentially

In this toolbox image processing is parallized via splitting a volumetric
image stack into several sub-stacks, typically in z-direction. As most of 
the image processig steps are non-local sub-stacks are created with overlaps 
and the results rejoined accordingly to minimize boundary effects.

Parallel processing is handled via this module.

.. _SubStack:

Sub-Stacks
----------

The parallel processing module creates a dictionary with information on
the sub-stack as follows:

========================== ==================================================
Key                        Description
========================== ==================================================
``stackId``                id of the sub-stack
``nStacks``                total number of sub-stacks
``source``                 source file/folder/pattern of the stack
``x``, ``y``, ``z``        the range of the sub-stack with in the full image
``zCenters``               tuple of the centers of the overlaps
``zCenterIndices``         tuple of the original indices of the centers of 
                           the overlaps
``zSubStackCenterIndices`` tuple of the indices of the sub-stack that
                           correspond to the overlap centers
========================== ==================================================

For exmaple the :func:`writeSubStack` routine makes uses of this information
to write out only the sub-parts of the image that is will contribute to the
final total image. 
"""
#:copyright: Copyright 2015 by Christoph Kirst, The Rockefeller University, New York City
#:license: GNU, see LICENSE.txt for details.

import sys
import math
import numpy

from multiprocessing import Pool

import ClearMap.IO as io

from ClearMap.Utils.ParameterTools import writeParameter
from ClearMap.Utils.ProcessWriter import ProcessWriter;
from ClearMap.Utils.Timer import Timer;

   
def printSubStackInfo(subStack, out = sys.stdout):
    """Print information about the sub-stack
    
    Arguments:
        subStack (dict): the sub-stack info
        out (object): the object to write the information to
    """
    writeParameter(head = "Sub Stack: ", out = out, **subStack);
    out.write('\n');


#define the subroutine for the processing
def _processSubStack(dsr):
    """Helper to process stack in parallel"""

    sf  = dsr[0];
    pp  = dsr[1];
    sub = dsr[2];
    verbose = dsr[3];

    timer = Timer();
    pw = ProcessWriter(sub["stackId"]);
    
    if verbose:
        pw.write("processing substack " + str(sub["stackId"]) + "/" + str(sub["nStacks"]));
        pw.write("file          = " + sub["source"]);
        pw.write("segmentation  = " + str(sf));
        pw.write("ranges: x,y,z = " + str(sub["x"]) +  "," + str(sub["y"]) + "," + str(sub["z"])); 
    
    img = io.readData(sub["source"], x = sub["x"], y = sub["y"], z = sub["z"]);
    
    if verbose:
        pw.write(timer.elapsedTime(head = 'Reading data of size ' + str(img.shape)));
    
    timer.reset();
    seg = sf(img, subStack = sub, out = pw, **pp);    

    if verbose:    
        pw.write(timer.elapsedTime(head = 'Processing substack of size ' + str(img.shape)));
    
    return seg;


def writeSubStack(filename, img, subStack = None):
    """Write the non-redundant part of a sub-stack to disk
    
    The routine is used to write out images when porcessed in parallel.
    It assumes that the filename is a patterned file name.
    
    Arguments:
        filename (str or None): file name pattern as described in 
                        :mod:`~ClearMap.Io.FileList`, if None return as array
        img (array): image data of sub-stack
        subStack (dict or None): sub-stack information, if None write entire image
                                 see :ref:`SubStack`
    
    Returns:
       str or array: the file name pattern or image
    """
    
    if not subStack is None:
        ii = subStack["zSubStackCenterIndices"][0];
        ee = subStack["zSubStackCenterIndices"][1];
        si = subStack["zCenterIndices"][0];
    else:
        si = 0;
        ii = 0;
        ee = -1;
    
    return io.writeData(filename, img[:,:,ii:ee], startIndex = si );     



def joinPoints(results, subStacks = None, shiftPoints = True, **args):
    """Joins a list of points obtained from processing a stack in chunks
    
    Arguments:
        results (list): list of point results from the individual sub-processes
        subStacks (list or None): list of all sub-stack information, see :ref:`SubStack`
        shiftPoints (bool): if True shift points to refer to origin of the image stack considered
                            when range specification is given. If False, absolute 
                            position in entire image stack.
    
    Returns:
       tuple: joined points, joined intensities
    """
    
    nchunks = len(results);
    pointlist = [results[i][0] for i in range(nchunks)];
    intensities = [results[i][1] for i in range(nchunks)]; 
    
    results = [];
    resultsi = [];
    for i in range(nchunks):
        cts = pointlist[i];
        cti = intensities[i];

        if cts.size > 0:
            cts[:,2] += subStacks[i]["z"][0];
            iid = numpy.logical_and(subStacks[i]["zCenters"][0] <= cts[:,2] , cts[:,2] < subStacks[i]["zCenters"][1]);
            cts = cts[iid,:];
            results.append(cts);
            if not cti is None:
                cti = cti[iid];
                resultsi.append(cti);
            
    if results == []:
        if not intensities is None:
            return (numpy.zeros((0,3)), numpy.zeros((0)));
        else:
            return numpy.zeros((0,3))
    else:
        points = numpy.concatenate(results);
        
        if shiftPoints:
            points = points + io.pointShiftFromRange(io.dataSize(subStacks[0]["source"]), x = subStacks[0]["x"], y = subStacks[0]["y"], z = 0);
        else:
            points = points - io.pointShiftFromRange(io.dataSize(subStacks[0]["source"]), x = 0, y = 0, z = subStacks[0]["z"]); #absolute offset is added initially via zranges !
            
        if intensities is None:
            return points;
        else:
            return (points, numpy.concatenate(resultsi));





def calculateSubStacks(source, z = all, x = all, y = all, **args):
    """Calculates the chunksize and other info for parallel processing and returns a list of sub-stack objects
    
    The sub-stack information is described in :ref:`SubStack`  
    
    Arguments:
        source (str): image source
        x,y,z (tuple or all): range specifications
        processes (int): number of parallel processes
        chunkSizeMax (int): maximal size of a sub-stack
        chunkSizeMin (int): minial size of a sub-stack
        chunkOverlap (int): minimal sub-stack overlap
        chunkOptimization (bool): optimize chunck sizes to best fit number of processes
        chunkOptimizationSize (bool or all): if True only decrease the chunk size when optimizing
        verbose (bool): print information on sub-stack generation
        
    Returns:
        list: list of sub-stack objects
    """    
    
    #determine z ranges
    fs = io.dataSize(source);
    zs = fs[2];
    zr = io.toDataRange(zs, r = z);
    nz = zr[1] - zr[0];
    
    #calculate optimal chunk sizes
    nchunks, zranges, zcenters = calculateChunkSize(nz, **args);
    
    #adjust for the zrange
    zcenters = [c + zr[0] for c in zcenters];
    zranges = [(zc[0] + zr[0], zc[1] + zr[0]) for zc in zranges];
    
    #create substacks
    subStacks = [];
    indexlo = zr[0];
    
    for i in range(nchunks):
        
        indexhi = int(round(zcenters[i+1]));
        if indexhi > zr[1] or i == nchunks - 1:
            indexhi = zr[1];
        
        zs = zranges[i][1] - zranges[i][0];
        
        subStacks.append({"stackId" : i, "nStacks" : nchunks, 
                          "source" : source, "x" : x, "y" : y, "z" : zranges[i], 
                          "zCenters" : (zcenters[i], zcenters[i+1]),
                          "zCenterIndices" : (indexlo, indexhi),
                          "zSubStackCenterIndices" : (indexlo - zranges[i][0], zs - (zranges[i][1] - indexhi))});
        
        indexlo = indexhi; # + 1;
    
    return subStacks;


        
def noProcessing(img, **parameter):
    """Perform no image processing at all and return original image
    
    Used as the default functon in :func:`parallelProcessStack` and
    :func:`sequentiallyProcessStack`.
    
    Arguments:
        img (array): imag
        
    Returns:
        (array): the original image
    """

def parallelProcessStack(source, x = all, y = all, z = all, sink = None,
                         processes = 2, chunkSizeMax = 100, chunkSizeMin = 30, chunkOverlap = 15,
                         chunkOptimization = True, chunkOptimizationSize = all, 
                         function = noProcessing, join = joinPoints, verbose = False, **parameter):
    """Parallel process an image stack
    
    Main routine that distributes image processing on paralllel processes.
       
    Arguments:
        source (str): image source
        x,y,z (tuple or all): range specifications
        sink (str or None): destination for the result
        processes (int): number of parallel processes
        chunkSizeMax (int): maximal size of a sub-stack
        chunkSizeMin (int): minial size of a sub-stack
        chunkOverlap (int): minimal sub-stack overlap
        chunkOptimization (bool): optimize chunck sizes to best fit number of processes
        chunkOptimizationSize (bool or all): if True only decrease the chunk size when optimizing
        function (function): the main image processing script
        join (function): the fuction to join the results from the image processing script
        verbose (bool): print information on sub-stack generation
        
    Returns:
        str or array: results of the image processing
    """     
    
    subStacks = calculateSubStacks(source, x = x, y = y, z = z, 
                                   processes = processes, chunkSizeMax = chunkSizeMax, chunkSizeMin = chunkSizeMin, chunkOverlap = chunkOverlap,
                                   chunkOptimization = chunkOptimization, chunkOptimizationSize = chunkOptimizationSize, verbose = verbose);
                                   
    nSubStacks = len(subStacks);
    if verbose:
        print("Number of SubStacks: %d" % nSubStacks);
                                       
    #for i in range(nSubStacks):
    #    self.printSubStackInfo(subStacks[i]);
    
    argdata = [];
    for i in range(nSubStacks):
        argdata.append((function, parameter, subStacks[i], verbose));    
    #print argdata
    
    # process in parallel
    pool = Pool(processes = processes);    
    results = pool.map(_processSubStack, argdata);
    
    #print '=========== results';
    #print results;
        
    #join the results
    results = join(results, subStacks = subStacks, **parameter);
    
    #write / or return 
    return io.writePoints(sink, results);


def sequentiallyProcessStack(source, x = all, y = all, z = all, sink = None,
                             chunkSizeMax = 100, chunkSizeMin = 30, chunkOverlap = 15,
                             function = noProcessing, join = joinPoints, verbose = False, **parameter):
    """Sequential image processing on a stack
    
    Main routine that sequentially processes a large image on sub-stacks.
       
    Arguments:
        source (str): image source
        x,y,z (tuple or all): range specifications
        sink (str or None): destination for the result
        processes (int): number of parallel processes
        chunkSizeMax (int): maximal size of a sub-stack
        chunkSizeMin (int): minial size of a sub-stack
        chunkOverlap (int): minimal sub-stack overlap
        chunkOptimization (bool): optimize chunck sizes to best fit number of processes
        chunkOptimizationSize (bool or all): if True only decrease the chunk size when optimizing
        function (function): the main image processing script
        join (function): the fuction to join the results from the image processing script
        verbose (bool): print information on sub-stack generation
        
    Returns:
        str or array: results of the image processing
    """     
    #determine z ranges  
    
    subStacks = calculateSubStacks(source, x = x, y = y, z = z, 
                                   processes = 1, chunkSizeMax = chunkSizeMax, chunkSizeMin = chunkSizeMin, chunkOverlap = chunkOverlap,  
                                   chunkOptimization = False, verbose = verbose);
    
    nSubStacks = len(subStacks);
    #print nSubStacks;    
    
    argdata = [];
    for i in range(nSubStacks):
        argdata.append((function, parameter, subStacks[i], verbose));    
    
    #run sequentially
    results = [];
    for i in range(nSubStacks):
        results.append(_processSubStack(argdata[i]));
    
    #join the results
    results = join(results, subStacks = subStacks, **parameter);
    
    #write / or return 
    return io.writePoints(sink, results);










### Pickle does not like classes:

## sub stack information
#class SubStack(object):
#    """Class containing all info of a sub stack usefull for the image processing and result joining functions"""
#    
#    # sub stack id
#    stackId = None;
#    
#    # number of stacks
#    nStacks = None;
#    
#    # tuple of x,y,z range of this sub stack
#    z = all; 
#    x = all;
#    y = all;
#    
#    #original source
#    source = None;    
#    
#    # tuple of center point of the overlaping regions
#    zCenters = None;
#    
#    # tuple of z indices that would generate full image without overlaps
#    zCenterIndices = None;
#    
#    # tuple of z indices in the sub image as returned by readData that would generate full image without overlaps
#    zSubCenterIndices = None;
#    
#    
#    def __init__(slf, stackId = 0, nStacks = 1, source = None, x = all, y = all, z = all, zCenters = all, zCenterIndices = all):
#        slf.stackId = stackId;
#        slf.nStacks = nStacks;
#        slf.source = source; 
#        slf.x = x;
#        slf.y = y;
#        slf.z = z;
#        slf.zCenters = zCenters;
#        slf.zCenterIndices = zCenterIndices;
#        if not zCenterIndices is all and not z is all:
#            slf.zSubCenterIndices = (c - z[0] for c in zCenterIndices);
#        else:
#            slf.zSubCenterIndices = all;
#       
#    
#def printSubStackInfo(slf, out = sys.stdout):
#    out.write("Sub Stack: %d / %d\n" % (slf.stackId, slf.nStacks));
#    out.write("source:         %s\n" %       slf.source);
#    out.write("x,y,z:          %s, %s, %s\n" % (str(slf.x), str(slf.y), str(slf.z)));
#    out.write("zCenters:       %s\n" %       str(slf.zCenters));   
#    out.write("zCenterIndices: %s\n" %       str(slf.zCenterIndices));