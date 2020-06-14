# -*- coding: utf-8 -*-
"""
Smoothing
=========

Smooth a binary image based on the local configuration of voxels in a cube.

See also
--------
The algortihm has similarities to the skeletonization algorithm using
parallel thinning (:mod:`~ClearMap.ImageProcessing.Skeletonization`).
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'

import os
import functools

import numpy as np
import scipy.ndimage as ndi

import multiprocessing as mp

import ClearMap.IO.IO as io
import ClearMap.IO.FileUtils as fu

import ClearMap.ImageProcessing.Topology.Topology3d as t3d

import ClearMap.ParallelProcessing.BlockProcessing as bp
import ClearMap.ParallelProcessing.DataProcessing.ArrayProcessing as ap

import ClearMap.Utils.Timer as tmr

###############################################################################
### Smoothing by number of neighbours
###############################################################################

def smooth_by_counting(source, sink = None, low = 5, high = 10, shape = None):
  """Smooth binary image by counting neighbours.
  
  Arguments
  ---------
  source : array
    The binary source to smooth.
  sink : array or None.
    The sink to write the smoothed source to.
  low : int
    If a voxel has less then this number of 26-neighbours it is set to False.
  high : int
    If a voxel has more then this number of 26-neighbours it is set to True.
  shape : tuple of int or None
    The shape of the square structuring element to consider.
    
  Returns
  -------
  sink : array
    The smoothed sbinary source.
    
  Note
  ----
  The algorithm uses a sequence of 1d convoluions for speed, allowing only 
  rectangular like structuring elements.
  """
  ndim = source.ndim;
  if shape is None:
    shape = (3,) * ndim;
  
  filtered = source;
  for d in range(ndim):
    weights = np.ones(shape[d], dtype = int);
    temp = np.zeros(source.shape, dtype = 'uint8');
    ap.correlate1d(filtered, weights, sink=temp, axis=d, mode='constant', cval=0);
    filtered = temp;
  
  if sink is None:
    sink = np.array(source, dtype = bool);

  sink[filtered >= high] = True;
  sink[filtered < low] = False;
  
  return sink


###############################################################################
### Topological smoothing
###############################################################################

def rotations_faces(cube):
  U = cube.copy();
  N = t3d.rotate(cube, axis=0, steps=1);
  W = t3d.rotate(cube, axis=1, steps=1);
  UNW = [U,N,W];
  DSE = [t3d.reflect(X) for X in UNW];
  return UNW + DSE;


def rotations_edges(cube):
  UN = cube.copy();
  UE = t3d.rotate(cube, axis=2, steps=1);
  US = t3d.rotate(cube, axis=2, steps=2); 
  UW = t3d.rotate(cube, axis=2, steps=3); 
  NW = t3d.rotate(cube, axis=1, steps=3); 
  NE = t3d.rotate(cube, axis=1, steps=1);  
  R = [t3d.reflect(X) for X in [UN,UE,US,UW,NW,NE]];
  return [UN,UE,US,UW,NW,NE] + R;


def rotations_nodes(cube):
  UNW = cube.copy();
  UNE = t3d.rotate(cube, axis=2,steps=1);
  USE = t3d.rotate(cube, axis=2,steps=2);
  USW = t3d.rotate(cube, axis=2,steps=3);
  R = [t3d.reflect(X) for X in [UNW,UNE,USE,USW]];
  return [UNW,UNE, USE,USW] + R


def rotations_node_faces(cube):
  U_UNW = cube.copy();                       #U1
  U_UNE = t3d.rotate(cube, axis=2, steps=1); #U3
  U_USE = t3d.rotate(cube, axis=2, steps=2); #U5
  U_USW = t3d.rotate(cube, axis=2, steps=3); #U7
  
  Us = [U_UNW, U_UNE, U_USE, U_USW];
  Ns = [t3d.rotate(X, axis=0, steps=1) for X in Us]; # N7. N1, N3, N5
  Ws = [t3d.rotate(X, axis=1, steps=1) for X in Us]; # W3, W1, W7, W5

  UNWs = Us + Ns + Ws;
  DSEs = [t3d.reflect(X) for X in UNWs];
  
  return UNWs + DSEs;


def U0(cube):
  return (cube[1,1,1] & cube[1,1,2] & 
         (not cube[0,0,0]) & (not cube[1,0,0]) & (not cube[2,0,0]) & 
         (not cube[0,1,0]) & (not cube[1,1,0]) & (not cube[2,1,0]) & 
         (not cube[0,2,0]) & (not cube[1,2,0]) & (not cube[2,2,0]) & 
         (not cube[0,0,1]) & (not cube[1,0,1]) & (not cube[2,0,1]) &
         (not cube[0,1,1]) & (not cube[2,1,1]) &
         (not cube[0,2,1]) & (not cube[1,2,1]) & (not cube[2,2,1]));


def U1(cube):
  return (cube[1,1,1] & cube[1,1,2] & 
         (not cube[0,0,0]) & (not cube[1,0,0]) & (not cube[2,0,0]) & 
         (not cube[0,1,0]) & (not cube[1,1,0]) & (not cube[2,1,0]) & 
         (not cube[0,2,0]) & (not cube[1,2,0]) & (not cube[2,2,0]) & 
         (not cube[0,0,1]) & (not cube[1,0,1]) & (not cube[2,0,1]) &
         (not cube[0,1,1]) & (not cube[2,1,1]) &
         (cube[0,2,1]) & (not cube[1,2,1]) & (not cube[2,2,1]) &
         (cube[0,2,2]))

def U2(cube):
  return (cube[1,1,1] & cube[1,1,2] & 
         (not cube[0,0,0]) & (not cube[1,0,0]) & (not cube[2,0,0]) & 
         (not cube[0,1,0]) & (not cube[1,1,0]) & (not cube[2,1,0]) & 
         (not cube[0,2,0]) & (not cube[1,2,0]) & (not cube[2,2,0]) & 
         (not cube[0,0,1]) & (not cube[1,0,1]) & (not cube[2,0,1]) &
         (not cube[0,1,1]) & (not cube[2,1,1]) &
         (not cube[0,2,1]) & (cube[1,2,1]) & (not cube[2,2,1]) &
         (cube[1,2,2]))


def R2(cube):
  return (cube[1,1,1] & cube[1,2,2] & 
         (not cube[0,0,0]) & (not cube[1,0,0]) & (not cube[2,0,0]) & 
         (not cube[0,1,0]) & (not cube[1,1,0]) & (not cube[2,1,0]) & 
         (not cube[0,2,0]) & (not cube[1,2,0]) & (not cube[2,2,0]) & 
         (not cube[0,0,1]) & (not cube[1,0,1]) & (not cube[2,0,1]) &
         (not cube[0,1,1]) & (not cube[2,1,1]) &
         (not cube[0,0,2]) & (not cube[1,0,2]) & (not cube[2,0,2]));


#def S3_old(cube):
#  return (cube[1,1,1] & cube[0,2,2] & 
#         (not cube[0,0,0]) & (not cube[1,0,0]) & (not cube[2,0,0]) & 
#         (not cube[0,1,0]) & (not cube[1,1,0]) & (not cube[2,1,0]) & 
#         (not cube[0,2,0]) & (not cube[1,2,0]) & (not cube[2,2,0]) & 
#         (not cube[0,0,1]) & (not cube[1,0,1]) & (not cube[2,0,1]) &
#         (not cube[2,1,1]) &
#         (not cube[2,1,2]) & 
#         (not cube[2,2,0]) & (not cube[2,2,1]) & (not cube[2,2,2]));
# 

def S3(cube):
  return (cube[1,1,1] & cube[0,2,2] & 
         (not cube[0,0,0]) & (not cube[1,0,0]) & (not cube[2,0,0]) & 
         (not cube[0,1,0]) & (not cube[1,1,0]) & (not cube[2,1,0]) & 
         (not cube[0,2,0]) & (not cube[1,2,0]) & (not cube[2,2,0]) & 
         (not cube[0,0,1]) & (not cube[1,0,1]) & (not cube[2,0,1]) &
         (not cube[2,1,1]) &
         (not cube[2,2,1]) & 
         (not cube[0,0,2]) & (not cube[1,0,2]) & (not cube[2,0,2]) &
         (not cube[2,1,2]) &
         (not cube[2,2,2]));
     

def cube_to_smoothing(cube):
  """Match cube configurations to delete, add or keep a voxel."""
  ### Delete center voxel:
  if cube[1,1,1]:
    
    # isolated or end
    if np.sum(cube) <= 2:
      return False
    
    # isolated voxels or voxels 'sticking out' 
    for cr in rotations_faces(cube):
      if U0(cr):
        return False;
    
    for cr in rotations_node_faces(cube):
      if U1(cr):
        return False;
      if U2(cr):
        return False;
    
    # voxels on edges
    for r in rotations_edges(cube):
      if R2(r):
        return False;
    
    # voxels on nodes
    for r in rotations_nodes(cube):
      if S3(r):
        return False;
  
  ### Add voxels
  if (not cube[1,1,1]):
    
    # mostly surrounded
    if np.sum(cube) >= 27-1-6:
      return True;
    
    # 3 direct neighbours, 2 in line
    if ((cube[1,1,0] & cube[1,1,2]) or 
        (cube[1,0,1] & cube[1,2,1]) or
        (cube[0,1,1] & cube[2,1,1])) and (
         np.sum(cube[np.where(t3d.n6)]) >= 3):
      return True;
    
    not_cube = np.logical_not(cube);
    
    # isolated voxels or voxels 'sticking out' 
    for cr in rotations_faces(not_cube):
      if U0(cr):
        return True;
    
    for cr in rotations_node_faces(not_cube):
      if U1(cr):
        return True;
      if U2(cr):
        return True;
    
    # voxels on edges
    for r in rotations_edges(not_cube):
      if R2(r):
        return True;
    
    # voxels on nodes
    for r in rotations_nodes(not_cube):
      if S3(r):
        return True;
  
  ### No change
  if cube[1,1,1]:
    return True;
  else:
    return False;
  

def index_to_smoothing(index, verbose = True):
  """Match index of configuration to smoothing action"""
  if verbose and index % 2**14 == 0:
    print('Smoothing LUT: %d / %d' % (index, 2**27));
  cube = t3d.cube_from_index(index=index, center=None);
  return cube_to_smoothing(cube);


def generate_lookup_table(function = index_to_smoothing, verbose = True, processes = None):
  """Generates lookup table for templates""" 
  if verbose:
    print('Smoothing: Generating look-up table!')
    
  if processes is None:
    processes = mp.cpu_count();
  
  if processes == 'serial':
    lut = [function(i) for i in range(2**27)];
  else:
    #import concurrent.futures as cf
    #with cf.ProcessPoolExecutor(max_workers=processes) as executor:
    #  lut = executor.map(function, range(2**27));
    pool = mp.Pool(mp.cpu_count());
    lut = pool.map(function, range(2**27), chunksize=2**27//8//mp.cpu_count());
  
  return np.array(lut, dtype = bool);


smooth_by_configuration_filename = "Smoothing.npy";
"""Filename for the look up table mapping a cube configuration to the smoothing action for the center pixel."""


def initialize_lookup_table(function = index_to_smoothing, filename = smooth_by_configuration_filename, verbose = True, processes = None):
  """Initialize the lookup table"""
  
  filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename);
  
  #uncompress if only zip file exists.
  fu.uncompress(filename);
  
  #load lookup table
  if os.path.exists(filename):
    if verbose:
      print('Smoothing: Loading look-up table from %s!' % filename)
    return np.load(filename);
  else:
    if verbose:
      print('Smoothing: Look-up table does not exists! Pre-calculating it!')
    lut = generate_lookup_table(function = function, verbose=verbose, processes=processes);
    np.save(filename, lut);
    return lut;


def smooth_by_configuration_block(source, iterations = 1, verbose = False):
  """Smooth a binary source using the local configuration around each pixel.
  
  Arguments
  ---------
  source : array
    The binary source to smooth.
  iterations : int
    Number of smoothing iterations.
  verbose : bool
    If True, print progress information.
    
  Returns
  -------
  smoothed : array
    Thre smoothed binary array.
  """
  if isinstance(source, io.src.Source):
    smoothed = source.array;
  else:
    smoothed = source;
  smoothed = np.asarray(smoothed, dtype='uint32');
  ndim = smoothed.ndim;
  
  lut = np.asarray(initialize_lookup_table(verbose=verbose), dtype='uint32');
  
  for i in range(iterations):   
    #index 
    for axis in range(ndim):
      kernel = t3d.index_kernel(axis=axis);
      smoothed = ndi.correlate1d(smoothed, kernel, axis=axis, output='uint32', mode='constant', cval=0);
    smoothed = lut[smoothed];
   
    if verbose:
      print('Binary Smoothing: itertion %d / %d done!' % (i+1, iterations));
  
  return np.asarray(smoothed, dtype=bool);


def smooth_by_configuration(source, sink = None, iterations = 1, 
                            processing_parameter = None,
                            processes = None, verbose = False):
  """Smooth a binary source using the local configuration around each pixel.
  
  Arguments
  ---------
  source : array or Source
    The binary source to smooth.
  sink : array, Source or None
    The sink to write result of smoothing. If None, return array.
  iterations : int
    Number of smoothing iterations.
  processing_parameter : None or dict
    The parameter passed to 
    :func:`ClearMap.ParallelProcessing.BlockProcessing.process`.
  processes : int or None
    number of processes to use.
  verbose : bool
    If True, print progress information.
    
  Returns
  -------
  smoothed : array or Source
    Thre smoothed binary array.

  Note
  ----
  The algorithm is based on a topological smoothing operation defined by adding
  or removing forground pixels based on the local topology of the binary array.
  """
  if verbose:
    print('Binary smoothing: initialized!');
    timer = tmr.Timer();
  
  #smoothing function
  smooth = functools.partial(smooth_by_configuration_block, iterations=iterations, verbose=False);
  smooth.__name__ = 'smooth_by_configuration'
  
  #initialize sources and sinks
  source = io.as_source(source);
  sink   = io.initialize(sink, shape=source.shape, dtype=bool, order=source.order); 
  
  #block processing parameter
  block_processing_parameter = dict(axes = bp.block_axes(source), 
                                    as_memory=True, 
                                    overlap=None, 
                                    function_type='source',
                                    processes=processes, 
                                    verbose=verbose);
  if processing_parameter is not None:
    block_processing_parameter.update(processing_parameter);
  if not 'overlap' in block_processing_parameter or block_processing_parameter['overlap'] is None:
    block_processing_parameter['overlap'] = 2 + 2 * iterations;
  if not 'size_min' in block_processing_parameter or block_processing_parameter['size_min'] is None:
    block_processing_parameter['size_min'] = 2 + 2 * iterations + 1;
  if not 'axes' in block_processing_parameter or block_processing_parameter['axes'] is None:
    block_processing_parameter['axes'] = bp.block_axes(source);
  #print(block_processing_parameter)
  
  #block process
  bp.process(smooth, source, sink, **block_processing_parameter);
  
  if verbose:
    timer.print_elapsed_time('Binary smoothing: done');
  
  return sink;


###############################################################################
### Testing
###############################################################################

def _test():
  import numpy as np
  import ClearMap.ImageProcessing.Binary.Smoothing as sm
  #reload(sm)
  
  lut = sm.initialize_lookup_table() #analysis:ignore
  #lut = sm.generate_lookup_table(verbose=True);
  
  shape = (30,40,50);
  binary = np.zeros(shape, dtype = bool, order='F');
  grid = np.meshgrid(*[range(s) for s in shape], indexing='ij');
  center = tuple(s/2 for s in shape);
  distance = np.sum([(g-c)**2 for g,c in zip(grid, center)], axis=0);
  binary[distance <= 10**2] = True
  
  import scipy.ndimage as ndi
  border = ndi.convolve(np.asarray(binary, dtype=int), np.ones((3,3,3)))
  border = np.logical_and(border > 0, border < 27)
  
  noisy = binary.copy();
  noisy[np.logical_and(border, np.random.rand(*shape) > 0.925)] = True
  smoothed  = sm.smooth_by_configuration(noisy, iterations=3, verbose=True, processes='serial')
  
  import ClearMap.Visualization.Plot3d as p3d
  p3d.plot([noisy, smoothed])
  
  #import scipy.io as sio
  #sio.savemat('binary_noisy.mat', {'noisy' : noisy})
  #sio.savemat('binary_smoothed.mat', {'smoothed' : smoothed.array})
  #noisy = sio.loadmat('binary_noisy.mat')['noisy']
