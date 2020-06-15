# -*- coding: utf-8 -*-
"""
Skeletonization
===============

Main routines for fast 3d skeletonization.

Note
----
Supported algorithsm are:
  
* :mod:`ClearMap.ImageProcessing.Skeletonization.PK12` - parallel 3d 12 
  sub-iteration thinning algorithm by Palagyi and Kuba [Palagy1999]_.
  
* RC6  - parallel 3d 6 sub-iteration istmus-based thinning algorithms 
  [Raynal2000]_.

References
----------
.. [Palagy1999] Palagyi & Kuba, A Parallel 3D 12-Subiteration Thinning Algorithm, Graphical Models and Image Processing 61, 199-221 (1999).

.. [Raynal2000] B. Raynal and M. Couprie, Istmus-Based 6-Directional Parallel Thinning Algorithms.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'

import numpy as np

import ClearMap.IO.IO as io

import ClearMap.ParallelProcessing.DataProcessing.ArrayProcessing as ap

import ClearMap.ImageProcessing.Skeletonization.PK12 as PK12

import ClearMap.Utils.Timer as tmr

###############################################################################
### Skeletonization
###############################################################################

def skeletonize(source, sink = None, points = None, method = 'PK12i', steps = None, in_place = False, verbose = True, **kwargs):
  """Skeletonize 3d binary arrays.
  
  Arguments
  ---------
  source : array or source 
    Binary image to skeletonize.
  sink : sink specification
    Optional sink.
  points : array or None
    Optional point list of the foreground points in the binary.
  method : str
    'PK12' or faster index version 'PK12i'.
  steps : int or None
    Number of maximal iteration steps. If None, maximal thinning.
  in_place : bool
    If True, the skeletonization is done directly on the input array.
    
  Returns
  -------
  skeleton : Source
    The skeletonized array.
  """
  if verbose:
    timer = tmr.Timer();
  
  if not in_place and io.is_file(source):
    binary_buffer = ap.read(source).as_buffer();
  else:
    binary, binary_buffer = ap.initialize_source(source);
    if not in_place:
      binary_buffer = np.array(binary_buffer);
  
  if method == 'PK12':
    result = PK12.skeletonize(binary_buffer, points=points, steps=steps, verbose=verbose, **kwargs)
  elif method == 'PK12i':
    result = PK12.skeletonize_index(binary_buffer, points=points, steps=steps, verbose=verbose, **kwargs)
  else:
    raise RuntimeError('Skeletonizaton method %r is not valid!' % method);
                      
  if verbose:
    timer.print_elapsed_time(head='Skeletonization');

  if sink is None:
    sink = ap.io.as_source(result);
  elif isinstance(sink, str):
    sink = ap.write(sink, result);
  else:
    sink = io.write(sink, result);
  return sink

###############################################################################
### Tests
###############################################################################
  
def _test():
  import numpy as np;
  import ClearMap.IO.IO as io
  import ClearMap.ImageProcessing.Skeletonization.Skeletonization as skl;
  import ClearMap.Visualization.Plot3d as p3d
  import ClearMap.Tests.Files as tsf
  
  from importlib import reload
  reload(skl)
  
  binary = tsf.skeleton_binary;
  
  #default version
  skeleton = skl.skeletonize(binary, delete_border=True, verbose=True);
  p3d.plot([[binary, skeleton]])  
  
  #fast index version
  skeleton = skl.PK12.skeletonize_index(binary.copy(), delete_border=True, verbose = True);  
  p3d.plot([[binary, skeleton]])  