"""
Trace
=====

Module to trace paths of minimal resitance between two points in a 3d array.
"""
__author__    = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__   = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__copyright__ = 'Copyright (c) 2019 by Christoph Kirst'

import os
import numpy as np

import pyximport

from ClearMap.Utils.exceptions import ClearMapValueError

pyximport.install(setup_args={"include_dirs": [np.get_include(), os.path.dirname(os.path.abspath(__file__))]},
                  reload_support=True)


import ClearMap.ImageProcessing.Tracing.TraceCode as code
import ClearMap.Utils.array_checks as ac


def _prepare(source, score, start, mask=None):
  """Coerce the arguments to what TraceCode.pyx expects.

  The C++ tracer receives raw pointers to the source, the score and the mask together with the
  strides *of the source only*: the three arrays must have the same shape and the same memory
  layout, hence they are all made C-contiguous (copies only when needed).
  The source and score must be float64 (the only source_t compiled).
  """
  source = np.ascontiguousarray(ac.as_dtype(source, np.float64, name='source'))
  score = np.ascontiguousarray(ac.as_dtype(score, np.float64, name='score'))
  if source.ndim != 3:
    raise ClearMapValueError(f'The source must be 3d, found {source.ndim:d} dimensions!')
  if score.shape != source.shape:
    raise ClearMapValueError(f'The score shape {score.shape!r} does not match the source shape {source.shape!r}!')
  start = _prepare_point(start, source.shape, 'start')
  if mask is not None:
    mask = np.ascontiguousarray(ac.check_dtype(ac.bool_as_uint8(mask), (np.uint8, np.uint16, np.float32, np.float64), name='mask'))
    if mask.shape != source.shape:
      raise ClearMapValueError(f'The mask shape {mask.shape!r} does not match the source shape {source.shape!r}!')
  return source, score, start, mask


def _prepare_point(point, shape, name):
  """A 3d index inside ``shape`` (the tracer does not check the bounds)."""
  point = ac.as_index_array(point, name=name, ndim=1)
  if point.shape != (3,) or np.any(point < 0) or np.any(point >= np.asarray(shape)):
    raise ClearMapValueError(f'{name} {tuple(point)!r} is not a 3d point inside the source of shape {shape!r}!')
  return point


###############################################################################
### Tracing
###############################################################################

def trace(source, score, start, stop, 
          costPerDistance = 1.0, minimumCostPerDistance = 1/60.0,
          tubenessMultiplier = 4.0, minimalTubeness = 0.1,
          returnQuality = False,
          maxSteps = None, verbose = False):
  """Trace a path in the 3d source image from start to stop

  Arguments
  ----------
  srouce : array
    Input source.
  score : array
    A measure at each point to score a path. The higher the more likely
    is it that the path will go through and thus the score is a reward
    like measure. The cost for the path is approximately 1/score.
  start : array
    Start position.
  stop : array
    Stop position.
  costPerDistance : float
    Cost used to when estimating remaining distance.
    Can be used to weigh the estimated distance measure.
  minimalCostperDistance : float
    Minimal cost per distance used when tubeness measure is below this value.
  tubenessMultipler : float
    Multiply the tubeness measure by this value before estimating coadt via inverse.
    Can be used to weigh tubness vs. distiance measures.
  minimalTubeness : float
    Minimal tubness measure to use (note the inverse of the tubness measure is used
    to calculate the cost, this effectively limits the maximal cost).
  maxSteps : int or None
    Number of maximal iteration steps.

  Returns
  -------
  path : 2-D array
    The path a list of coordinates.
  """

#    if not source.flags.c_contiguous:
#        raise RuntimeError('Source array not c-contigous');
#    
#    if not tubeness.flags.c_contiguous:
#        raise RuntimeError('Tubeness array not c-contigous');    
  
  if maxSteps is None:
    maxSteps = -1;
  
  source, score, start, _ = _prepare(source, score, start)
  stop = _prepare_point(stop, source.shape, 'stop')
  path = code.trace(source, score, start, stop,
                    costPerDistance, minimumCostPerDistance, 
                    tubenessMultiplier, minimalTubeness,
                    returnQuality,
                    maxSteps, verbose);
  
  return path;

  
def trace_to_mask(source, tubeness, start, mask, 
                  costPerDistance = 1.0, minimumCostPerDistance = 1/60.0,
                  tubenessMultiplier = 4.0, minimalTubeness = 0.1,
                  returnQuality = False,
                  maxSteps = None, verbose = False):
  """Trace a path in the 3d source image from start to a point on the mask

  Parameters
  ----------
  source : array 
      Input source.
  tubeness : array
      Tubness measure used to score path
  start : array
      starting point for tracing
  mask : 3-D array
      distance array to mask (goal points on mask == 0).  
  costPerDistance : float
      Cost used to when estimating remaining distance.
      Can be used to weigh the estimated distance measure
  minimalCostperDistance : float
      Minimal cost per distance used when tubeness measure is below this value.
  tubenessMultipler : float
      Multiply the tubeness measure by this value before estimating coadt via inverse.
      Can be used to weigh tubness vs. distiance measures.
  maxSteps : int or None
      Number of maximal iteration steps.

  Returns
  -------
  path : 2-D array
      the path a list of coordinates
  """

#    if not source.flags.c_contiguous:
#        raise RuntimeError('Source array not c-contigous');
#    
#    if not tubeness.flags.c_contiguous:
#        raise RuntimeError('Tubeness array not c-contigous');  
#        
#    if not mask.flags.c_contiguous:
#        raise RuntimeError('Mask array not c-contigous');    
  
  if maxSteps is None:
    maxSteps = -1;
  #maxSteps = long(maxSteps);
  
  source, tubeness, start, mask = _prepare(source, tubeness, start, mask)
  path = code.trace_to_mask(source, tubeness, start, mask,
                    costPerDistance, minimumCostPerDistance,
                    tubenessMultiplier, minimalTubeness, 
                    returnQuality,
                    maxSteps, verbose);
  
  return path




def _test():
  import numpy as np;
  import scipy.ndimage as ndi;
  
  import ClearMap.Visualization.Plot3d as p3d
  import ClearMap.ImageProcessing.Filter.Curvature.Curvature as cur 
  import ClearMap.ImageProcessing.Tracing.Trace as trc;
  
  from importlib import reload
  reload(trc);
  
  #x = np.random.rand(50,50,50);
  #x = ndi.gaussian_filter(x, sigma = 2);
  
  x = np.load('/home/ckirst/Desktop/data.npy')[:50,:50,:50];
  x = np.ascontiguousarray(x, dtype = float);
  
  start, stop = [37,20,24], [27,32,38]
  
#  from collections import defaultdict
#  import gc
#  gc.collect();
  #before = defaultdict(int)
  #after = defaultdict(int)
  #for i in gc.get_objects():
  #  before[type(i)] += 1   
  
  t = cur.tubeness(x)
  
  #reload(trc);
  p = trc.trace(x, t, start, stop, verbose = False);
  
  #gc.collect();
  
  #for i in gc.get_objects():
  #   after[type(i)] += 1
  #print [(k, after[k] - before[k]) for k in after if after[k] - before[k]]
  
  #import mem_top as mt;
  #print mt.mem_top(width = 150);
  
  
  #%%
  xp = np.asarray(x, dtype = int);
  for pp in p:
    xp[pp[0], pp[1], pp[2]] = 512;
 
  try:
    dd[0].setSource(xp);
    dd[1].setSource(t);
  except:
    dd = p3d.plot([xp,t]);


  import scipy.ndimage as ndi
  
  mask = np.zeros_like(x);
  mask[:,:,:40] = 1; # zeros are the goal !!
  dist = ndi.distance_transform_edt(mask, sampling = [1,1,1]);
  
  start = [37, 20, 25];
  path = trc.traceToMask(x, t, start, dist)
  
  xp = np.asarray(x, dtype = int);
  for pp in path:
    xp[pp[0], pp[1], pp[2]] = 512;  
  
  try:
    dd[0].setSource(xp);
    dd[1].setSource(t);
  except:
    dd = p3d.plot([xp,t]);
