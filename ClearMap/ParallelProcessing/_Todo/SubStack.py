# -*- coding: utf-8 -*-
"""
The SubStack module provides tools to calculate sub-stacks for parallel processing
"""
__author__    = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__   = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__copyright__ = 'Copyright 2017 by Christoph Kirst, The Rockefeller University, New York City'


import math
import numpy as np

import numbers

import ClearMap.IO as io
from ClearMap.IO.Region import Region

#TODO:join with IO !!!

###############################################################################
### SubStack Size
###############################################################################

def calculateSubStackSize(size, processes = 2, sizeMax = 100, sizeMin = 30, overlap = 15,  optimization = True, optimizationSize = 'all', verbose = True):
    """Calculates the substack sizes and other info for parallel processing along a single axis
    
    Arguments:
      size : int 
        size of the array dimension to be split up
      processes : int
        number of parallel processes
      sizeMax : int
        maximal size of a sub-stack
      sizeMin : int
        minial size of a sub-stack
      overlap : int
        minimal sub-stack overlap
      optimization : bool 
        optimize chunck sizes to best fit number of processes
      optimizationSize : ('increase', 'decrease', 'all')
        increase, decrease or optimally change the chunk size when optimizing
      verbose : bool
        print information on sub-stack generation
        
    Returns:
      tuple: 
        number of chunks, 
      tuple:
        ranges of each chunk
      tuple:
        centers in overlap regions
    """
    
    #some checks
    if overlap >= sizeMax:
        raise RuntimeError('Chunk overlap is larger than maximal chunk size: %d >= %d!' % (overlap, sizeMax));
    if overlap >= sizeMin:
        raise RuntimeError('Chunk overlap is larger than minimal chunk size: %d >= %d!' % (overlap, sizeMin));
    if sizeMin > sizeMax:
        raise RuntimeError('Minimal chunk size larger than maximal chunk size %d > %d !' % (sizeMin, sizeMax));
    
    #calcualte chunk size estimates
    chunksize = sizeMax;
    nchunks = int(math.ceil(float(size - chunksize) / (chunksize - overlap) + 1)); 
    if nchunks <= 0:
        nchunks = 1;   
    chunksize = float(size + (nchunks-1) * overlap) / nchunks;
    
    if verbose:
        print("Estimated chunk size %d in %d chunks!" % (chunksize, nchunks));
    
    if nchunks == 1:
        return 1, [(0, size)], [0, size]
        
    #optimize number of chunks wrt to number of processors
    if optimization:
        nadd = nchunks % processes;
        if nadd != 0:
            if optimizationSize == 'all' or optimizationSize is all:
                if nadd < processes / 2.0:
                    optimizationSize = 'increase';
                else:
                    optimizationSize = 'decrease';
                    
            if verbose:
                print("Optimizing chunk size to fit number of processes!")
                
            if optimizationSize == 'decrease':
                #try to deccrease chunksize / increase chunk number to fit distribution on processors
                nchunks = nchunks - nadd + processes;
                chunksize = float(size + (nchunks-1) * overlap) / nchunks;
                
                if verbose:
                    print("Optimized chunk size decreased to %d in %d chunks!" % (chunksize, nchunks));
                    
            elif nchunks > nadd:
                #try to increase chunk size and decrease chunk number to fit  processors
                nchunks = nchunks - nadd;
                chunksize = float(size + (nchunks-1) * overlap) / nchunks;
                                  
                if verbose:
                    print("Optimized chunk size increased to %d in %d chunks!" % (chunksize, nchunks));
                
            else:
                if verbose:
                    print("Optimized chunk size unchanged %d in %d chunks!" % (chunksize, nchunks));
        
        else:
            if verbose:
                print("Chunk size optimal %d in %d chunks!" % (chunksize, nchunks));
    
    
    #increase overlap if chunks to small
    if chunksize < sizeMin:
        #raise Warning("Warning: Some chunks with average chunk size %f.02 may be smaller than minima chunk size %d!" % (chunksize, sizeMin)); 
        if verbose:
            print("Warning: Some chunks with average chunk size %.02f may be smaller than minima chunk size %d!" % (chunksize, sizeMin)); 
    if chunksize > sizeMax:
        #raise Warning("Warning: optimized chunk size %f.02 is larger than maximum chunk size %d!" % (chunksize, sizeMax)); 
        if verbose:
            print("Warning: Some chunks with average chunks size %.02f may be larger than maximum chunk size %d!" % (chunksize, sizeMax)); 
        
        #chunksize = sizeMin;
        #overlap = math.ceil(chunksize - (size - chunksize) / (nchunks -1));
        #if verbose:        
        #    print("Warning: setting chunk overlap to %d!" % overlap);
           
    #calucalte actual chunk sizes
    chunksizerest = chunksize;
    chunksize = int(math.floor(chunksize));
    chunksizerest = chunksizerest - chunksize;
    
    zranges = [(0, chunksize)];
    zcenters = [0];
    n = 1;
    csr = chunksizerest;
    zhi = chunksize;
    
    while (n < nchunks):
        n += 1;
        
        zhiold = zhi;
        zlo = zhi - overlap;
        zhi = zlo + chunksize;
        
        csr += chunksizerest;
        if csr >= 1:
            csr = csr - 1;
            zhi += 1;
        
        if n == nchunks:        
            zhi = size;
        
        zranges.append((int(zlo), int(zhi)));
        zcenters.append((zhiold - zlo) / 2. + zlo); 
        
    zcenters.append(size);
    
    if verbose:
      naddr = min(10, nchunks);
      if nchunks > naddr:
        pr = '...'
      else:
        pr = '';
      print("Final chunks : %d" % nchunks);
      print("Final chunks : " + str(zranges[:naddr]) + pr);
      print("Final centers: " + str(zcenters[:(naddr+1)]) + pr);
      sizes = np.unique([z[1]- z[0] for z in zranges]);
      print("Final sizes  : " + str(sizes));
    
    return nchunks, zranges, zcenters;


###############################################################################
### SubStack 
###############################################################################

class SubStack(Region):
  """SubStack is a Region with a sub-region indicating the valid region to use when combining the sub-stacks"""
   
  def __init__(self, region = all, source = None, x = None, y = None, z = None, dim = None, offsets = None, id = None, nStacks = None):           
    """Constructor"""
    super(self.__class__, self).__init__(region = region, source = source, x = x, y = y, z = z, dim = dim);
    
    # handle offset
    if offsets is None:
      self._valid = Region(region = self);
    else:
      dim = self.dim();
      if len(offsets) != dim:
        raise RuntimeError('offset dimension %d does not fit data dimensions %d' % (len(offsets), dim));
  
      rsize = self.size();
      
      voff = [];
      for d in range(dim):
        off = offsets[d];
        if not isinstance(off, list) and not isinstance(off, tuple):
          off = list([off]);
        if len(off) == 1:
          off = (off[0], off[0]);
        
        vr = [0,0];
        o = off[0];
        if isinstance(o, numbers.Number):  
          vr[0] = min(o, rsize[d]);
        
        o = off[1];
        if isinstance(o, numbers.Number):  
          vr[1] = rsize[d] - min(o, rsize[d] - vr[0]);
        
        voff.append(vr);
        
      self._valid = Region(region = voff, source = self);
    
    self.id = id;
    self.nStacks = nStacks;
    
    #if not isinstance(self._source, nadd.memmap):
    #  raise RuntimeError('Substack assumes source to be of type numpy.memmap');
  
  def valid(self):
    """Return the valid region of the substack"""
    return self._valid;

  def validSlice(self):
    """Return region specifications as slice tuple for use with numpy"""
    return self._valid.slice();
  
  def validSourceSlice(self, simplify = False):
    """Return source region specifications as slice tuple for use with numpy"""
    return self._valid.sourceSlice(simplify = simplify);
    
  def validSize(self):
    """Return size of the valid region"""
    return self._valid.size();
    
  def validShape(self):
    """Return shape of the valid region"""
    return self._valid.shape();
  
  def readData(self, source = None, valid = False):
    if valid:
      return self.readValidData(source = source);
      
    if source is None:
      source = self.source();
    
    if isinstance(source, str):
      return io.MMP.readData(source, region = self);
    else:
      vsslice = self.sourceSlice();
      return source[vsslice];
      
  def read(self, source = None, valid = False):
    return self.readData(source = source, valid = valid);
  
  def readValidData(self, source = None):
    if source is None:
      source = self.source();
    
    if isinstance(source, str):
      return io.MMP.readData(source, region = self.valid());
    else:
      vsslice = self.validSourceSlice();
      return source[vsslice];
      
  def readValid(self, source = None):
    return self.readValidData(source = source);    
    
  
  def writeData(self, data, sink = None, valid = False):
    if valid is True:
      self.writeValidData(data, sink = sink);
    if sink is None:
      sink = self.source();
    if isinstance(sink, str):
      return io.MMP.write(sink, data, region = self);
    else:
      vsslice  = self.sourceSlice();
      sink[vsslice] = data;
      return sink;

  def write(self, data, sink = None, valid = False):
    return self.writeData(data, sink = sink, valid = valid);
  
  def writeValidData(self, data, sink = None):
    if sink is None:
      sink = self.source();
    if isinstance(sink, str):
      sink = io.MMP.write(sink, self.extractValidData(data), region = self.valid());
    else:
      vsslice  = self.validSourceSlice();
      sink[vsslice] = self.extractValidData(data);
    return sink;
    
  def writeValid(self, data, sink = None):
    return self.writeValidData(data, sink = sink);
    
  def extractValidData(self, data):
    vslice = self._valid.slice();
    return data[vslice];  

  def _strSlice(self, arg):
    if isinstance(arg, slice):
      #if arg.start is None and arg.stop is None and arg.step is None:
      #  return None;
      return (arg.start, arg.stop, arg.step);
    if isinstance(arg, list):
      return str([self._strSlice(a) for a in arg]).replace(' ', '')
    if isinstance(arg, tuple):
      return str(tuple([self._strSlice(a) for a in arg])).replace(' ', '');
    return str(arg);
  
  def __str__(self):
    return "<<SubStack %r: Source:%s Region:%s Valid:%s>>" % (self.id, str(self.sourceSize()), self._strSlice(self.sourceSlice()), self._strSlice(self.validSourceSlice()));
    
  def __repr__(self):
    return self.__str__() 
  


def calculateSubStacks(source, axis = 2, **args):
    """Calculates the chunksize and other info for parallel processing and returns a list of sub-stack objects
    
    The sub-stack information is described in :ref:`SubStack`  
    
    Arguments:
        source : str
          image source
        processes : int
          number of parallel processes
        sizeMax : int
          maximal size of a sub-stack
        sizeMin : int
          minial size of a sub-stack
        overlap : int
          minimal sub-stack overlap
        optimization : bool
          optimize chunck sizes to best fit number of processes
        optimizationSize : bool or all
          if True only decrease the chunk size when optimizing
        verbose : bool
          print information on sub-stack generation
        
    Returns:
        list: 
          list of SubStack objects
    """
    
    #determine z ranges
    ssize = io.dataSize(source);
    nz = ssize[axis];
    zr = (0, nz);
    
    #calculate optimal chunk sizes
    nchunks, zranges, zcenters = calculateSubStackSize(nz, **args);
    
    #adjust for the zrange
    zcenters = [c + zr[0] for c in zcenters];
    zranges = [(zc[0] + zr[0], zc[1] + zr[0]) for zc in zranges];
    
    #create substacks
    subStacks = [];
    #indexlo = zr[0];
    indexlo = 0;
    
    for i in range(nchunks):
        
        indexhi = int(round(zcenters[i+1]));
        if indexhi > zr[1] or i == nchunks - 1:
            indexhi = zr[1];
        
        region = [all,all,all];
        region[axis] = zranges[i];
        
        offsets = [0,0,0];
        offsets[axis] = (indexlo - zranges[i][0], zranges[i][1] - indexhi);
        
        subStacks.append(SubStack(region = region, source = source, 
                                  offsets = offsets, id = i, nStacks = nchunks));
        
        indexlo = indexhi; # + 1;
    
    return subStacks;


def joinSubStacks(substacks):
  """Returns a minimal SubStack that includes all the sub-stacks in the list
  
  Arguments
  ---------
    substacks: list of SubStack objects
       list of substacks to cover in new SubStack
  
  Returns
  -------
    SubStack 
      The substack covering all substacks
      
  Note
  ----
    The returned substack directly referes to the underlying source data removing all nested structures
  """
  
  ndim = substacks[0].dim();
  region = [[np.inf, -np.inf] for i in range(ndim)];
  valid  = [[np.inf, -np.inf] for d in range(ndim)];

  for sub in substacks:
    s = sub.sourceSlice();
    for d in range(ndim):
      if region[d][0] is not None:
        if s[d].start is None:
          region[d][0] = None;
        else:
          region[d][0] = min(region[d][0], s[d].start);
      
      if region[d][1] is not None:
        if s[d].stop is None:
          region[d][1] = None;
        else:
          region[d][1] = max(region[d][1], s[d].stop);
    
    s = sub.validSourceSlice();
    for d in range(ndim):
      if valid[d][0] is not None:
        if s[d].start is None:
          valid[d][0] = None;
        else:
          valid[d][0] = min(valid[d][0], s[d].start);
      
      if valid[d][1] is not None:
        if s[d].stop is None:
          valid[d][1] = None;
        else:
          valid[d][1] = max(valid[d][1], s[d].stop);
  
  source = substacks[0].source();
  sub = SubStack(region = [slice(r[0], r[1]) for r in region], source = source);
  
  #calculate valid region within the sub-stack
  for d in range(ndim):
    if region[d][0] is not None:
      shift = region[d][0];
    else:
      shift = 0;
    
    if valid[d][0] is not None:
      valid[d][0] = valid[d][0] - shift;
    
    if valid[d][1] is not None:
      valid[d][1] = valid[d][1] - shift;
  
  sub._valid = Region(region = [slice(r[0], r[1]) for r in valid], source = sub);
  
  return sub;



if __name__ == "__main__":
    from importlib import reload
    import numpy as np
    import ClearMap.ParallelProcessing.SubStack as sbs
    reload(sbs)
    
    data = np.random.rand(50,100,200);
    subs = sbs.calculateSubStacks(data)
    print(subs)
    
    sub = subs[0];
    
    sub.validSourceSlice()
    
    sub.source() is data
    
    np.all(sub.readData() == data[sub.slice()])
    
    sub.writeData(np.zeros(sub.size()))
    
    s2 = sbs.joinSubStacks(subs[1:3])