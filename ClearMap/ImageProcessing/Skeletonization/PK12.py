# -*- coding: utf-8 -*-
"""
3d Skeletonization PK12
=======================

This module implements the 3d parallel 12-subiteration thinning algorithm 
by Palagy & Kuba via parallel convolution of the data with a base template
and lookup table matching.

Reference
---------
  Palagyi & Kuba, A Parallel 3D 12-Subiteration Thinning Algorithm, 
  Graphical Models and Image Processing 61, 199-221 (1999)
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


import os
import numpy as np
import multiprocessing as mp

import ClearMap.ImageProcessing.Topology.Topology3d as t3d
 
import ClearMap.ParallelProcessing.DataProcessing.ArrayProcessing as ap
import ClearMap.ParallelProcessing.DataProcessing.ConvolvePointList as cpl

import ClearMap.Utils.Timer as tmr

import ClearMap.IO.FileUtils as fu


###############################################################################
### Topology
###############################################################################

def match(cube):
  """Match one of the masks in the algorithm.
  
  Arguments
  ---------
  cube : 3x3x3 bool array
    The local binary image.
  
  Returns
  -------
  match : bool
    True if one of the masks matches
  
  Note
  ----
  Algorithm as in Palagyi & Kuba (1999)
  """
  #T1
  T1 = (cube[1,1,0] & cube[1,1,1] & 
        (cube[0,0,0] or cube[1,0,0] or cube[2,0,0] or
         cube[0,1,0] or cube[2,1,0] or
         cube[0,2,0] or cube[1,2,0] or cube[2,2,0] or
         cube[0,0,1] or cube[1,0,1] or cube[2,0,1] or
         cube[0,1,1] or cube[2,1,1] or 
         cube[0,2,1] or cube[1,2,1] or cube[2,2,1]) &
        (not cube[0,0,2]) & (not cube[1,0,2]) & (not cube[2,0,2]) &
        (not cube[0,1,2]) & (not cube[1,1,2]) & (not cube[2,1,2]) &
        (not cube[0,2,2]) & (not cube[1,2,2]) & (not cube[2,2,2]));
  if T1:
    return True;
  
  #T2
  T2 = (cube[1,1,1] & cube[1,2,1] & 
        (cube[0,1,0] or cube[1,1,0] or cube[2,1,0] or
         cube[0,2,0] or cube[1,2,0] or cube[2,2,0] or
         cube[0,1,1] or cube[2,1,1] or
         cube[0,2,1] or cube[2,2,1] or
         cube[0,1,2] or cube[1,1,2] or cube[2,1,2] or
         cube[0,2,2] or cube[1,2,2] or cube[2,2,2]) &
        (not cube[0,0,0]) & (not cube[1,0,0]) & (not cube[2,0,0]) &
        (not cube[0,0,1]) & (not cube[1,0,1]) & (not cube[2,0,1]) &
        (not cube[0,0,2]) & (not cube[1,0,2]) & (not cube[2,0,2]));
  if T2: 
    return True;
    
  #T3
  T3 = (cube[1,1,1] & cube[1,2,0] & 
        (cube[0,1,0] or cube[2,1,0] or
         cube[0,2,0] or cube[2,2,0] or
         cube[0,1,1] or cube[2,1,1] or
         cube[0,2,1] or cube[2,2,1]) &
        (not cube[0,0,0]) & (not cube[1,0,0]) & (not cube[2,0,0]) &
        (not cube[0,0,1]) & (not cube[1,0,1]) & (not cube[2,0,1]) &
        (not cube[0,0,2]) & (not cube[1,0,2]) & (not cube[2,0,2]) &
        (not cube[0,1,2]) & (not cube[1,1,2]) & ( not cube[2,1,2]) &
        (not cube[0,2,2]) & (not cube[1,2,2]) & (not cube[2,2,2]));
  if T3:
    return True;
  
  #T4
  T4 = (cube[1,1,0] & cube[1,1,1] & cube[1,2,1] & 
        ((not cube[0,0,1]) or (not cube[0,1,2])) &
        ((not cube[2,0,1]) or (not cube[2,1,2])) &
        (not cube[1,0,1]) & 
        (not cube[0,0,2]) & (not cube[1,0,2]) & (not cube[2,0,2]) &
        (not cube[1,1,2]));
  if T4:
    return True;    
  
  #T5
  T5 = (cube[1,1,0] & cube[1,1,1] & cube[1,2,1] & cube[2,0,2] &
        ((not cube[0,0,1]) or (not cube[0,1,2])) &
        (((not cube[2,0,1]) & cube[2,1,2]) or (cube[2,0,1] & (not cube[2,1,2]))) &
        (not cube[1,0,1]) & 
        (not cube[0,0,2]) & (not cube[1,0,2]) &
        (not cube[1,1,2]));
  if T5:
    return True;
    
  #T6
  T6 = (cube[1,1,0] & cube[1,1,1] & cube[1,2,1] & cube[0,0,2] &
        ((not cube[2,0,1]) or (not cube[2,1,2])) &
        (((not cube[0,0,1]) & cube[0,1,2]) or (cube[0,0,1] & (not cube[0,1,2]))) &
        (not cube[1,0,1]) & 
        (not cube[1,0,2]) & (not cube[2,0,2]) &
        (not cube[1,1,2]));
  if T6:
    return True;
    
  #T7
  T7 = (cube[1,1,0] & cube[1,1,1] & cube[2,1,1] &  cube[1,2,1] &
        ((not cube[0,0,1]) or (not cube[0,1,2])) &
        (not cube[1,0,1]) & 
        (not cube[0,0,2]) & (not cube[1,0,2]) &
        (not cube[1,1,2]));
  if T7:
    return True;
  
  #T8
  T8 = (cube[1,1,0] & cube[0,1,1] & cube[1,1,1] & cube[1,2,1] &
        ((not cube[2,0,1]) or (not cube[2,1,2])) &
        (not cube[1,0,1]) & 
        (not cube[1,0,2]) & (not cube[2,0,2]) &
        (not cube[1,1,2]));
  if T8:
    return True; 
    
  #T9
  T9 = (cube[1,1,0] & cube[1,1,1] & cube[2,1,1] & cube[0,0,2] & cube[1,2,1] &
        (((not cube[0,0,1]) & cube[0,1,2]) or (cube[0,0,1] & (not cube[0,1,2]))) &
        (not cube[1,0,1]) & 
        (not cube[1,0,2]) &
        (not cube[1,1,2]));
  if T9:
    return True;   
    
  #T10
  T10= (cube[1,1,0] & cube[0,1,1] & cube[1,1,1] & cube[2,0,2] & cube[1,2,1] &
        (((not cube[2,0,1]) & cube[2,1,2]) or (cube[2,0,1] & (not cube[2,1,2]))) &
        (not cube[1,0,1]) & 
        (not cube[1,0,2]) &
        (not cube[1,1,2]));
  if T10:
    return True;  
    
  #T11
  T11= (cube[2,1,0] & cube[1,1,1] & cube[1,2,0] &
        (not cube[0,0,0]) & (not cube[1,0,0]) & 
        (not cube[0,0,1]) & (not cube[1,0,1]) &
        (not cube[0,0,2]) & (not cube[1,0,2]) & (not cube[2,0,2]) &
        (not cube[0,1,2]) & (not cube[1,1,2]) & (not cube[2,1,2]) &
        (not cube[0,2,2]) & (not cube[1,2,2]) & (not cube[2,2,2]));
  if T11: 
    return True;
    
  #T12
  T12= (cube[0,1,0] & cube[1,2,0] & cube[1,1,1] &
        (not cube[1,0,0]) & (not cube[2,0,0]) & 
        (not cube[1,0,1]) & (not cube[2,0,1]) &
        (not cube[0,0,2]) & (not cube[1,0,2]) & (not cube[2,0,2]) &
        (not cube[0,1,2]) & (not cube[1,1,2]) & (not cube[2,1,2]) &
        (not cube[0,2,2]) & (not cube[1,2,2]) & (not cube[2,2,2]));
  if T12: 
    return True;
    
  #T13
  T13= (cube[1,2,0] & cube[1,1,1] & cube[2,2,1] &
        (not cube[0,0,0]) & (not cube[1,0,0]) & (not cube[2,0,0]) & 
        (not cube[0,0,1]) & (not cube[1,0,1]) & (not cube[2,0,1]) & 
        (not cube[0,0,2]) & (not cube[1,0,2]) & (not cube[2,0,2]) &
        (not cube[0,1,2]) & (not cube[1,1,2]) &
        (not cube[0,2,2]) & (not cube[1,2,2]));
  if T13: 
    return True; 
    
  #T14
  T14= (cube[1,2,0] & cube[1,1,1] & cube[0,2,1] &
        (not cube[0,0,0]) & (not cube[1,0,0]) & (not cube[2,0,0]) & 
        (not cube[0,0,1]) & (not cube[1,0,1]) & (not cube[2,0,1]) & 
        (not cube[0,0,2]) & (not cube[1,0,2]) & (not cube[2,0,2]) &
        (not cube[1,1,2]) & (not cube[2,1,2]) &
        (not cube[1,2,2]) & (not cube[2,2,2]));
  if T14: 
    return True; 
    
  return False;
 

def match_index(index, verbose = True):
  if verbose and index % 2**14 == 0:
    print('PK12 LUT: %d / %d' % (index, 2**26));
  cube = t3d.cube_from_index(index=index, center=True);
  return match(cube);


def match_non_removable(index, verbose = True):
  if verbose and index % 2**14 == 0:
    print('PK12 LUT non-removables: %d / %d' % (index, 2**26));
  cube = t3d.cube_from_index(index=index, center=False);
  n = cube.sum();
  if n < 2:
    return True;
  if n > 3:
    return False;
  x,y,z = np.where(cube); 
  if n == 2:
    if np.any(np.abs([x[1]-x[0], y[1]-y[0], z[1]-z[0]]) == 2):
      return True;
    else:
      return False;
  else:
     if np.any(np.abs([x[1]-x[0], y[1]-y[0], z[1]-z[0]]) == 2) and np.any(np.abs([x[2]-x[0], y[2]-y[0], z[2]-z[0]]) == 2) and np.any(np.abs([x[1]-x[2], y[1]-y[2], z[1]-z[2]]) == 2):
       return True;
     else:
       return False;


def generate_lookup_table(function = match_index, verbose = True):
  """Generates lookup table for templates"""
   
  pool = mp.Pool(mp.cpu_count());
  lut = pool.map(function, range(2**26),chunksize=2**26/8/mp.cpu_count());
  
  return np.array(lut, dtype = bool);


filename = "PK12.npy";
"""Filename for the look up table mapping a cube configuration to the deleatability of the center pixel"""

def initialize_lookup_table(function = match_index, filename = filename):
  """Initialize the lookup table"""
  
  filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename);
  
  #check if only compressed file exists
  fu.uncompress(filename)
  
  if os.path.exists(filename):
    return np.load(filename);
  else:
    lut = generate_lookup_table(function=function);
    np.save(filename, lut);
    return lut;


base = t3d.cube_base_2(center=False);
"""Base kernel to multiply with cube to obtain index of cube"""

delete = initialize_lookup_table();
"""Lookup table mapping cube index to its deleteability"""

keep = np.logical_not(delete);
"""Lookup table mapping cube index to its non-deleteability"""


filename_non_removable = "PK12nr.npy";
"""Filename for the lookup table mapping a cube configuration to the non-removeability of the center pixel"""

non_removable = initialize_lookup_table(filename = filename_non_removable, function = match_non_removable);
"""Lookup table mapping cube index to its non-removeability"""

consider = np.logical_not(non_removable);
"""Lookup table mapping cube index to whether it needs to be considered further"""

rotations = t3d.rotations12(base);
"""Rotations of the base cube for the sub-iterations"""


###############################################################################
### Skeletonization
###############################################################################

def skeletonize(binary, points = None, steps = None, removals = False, radii = False,
                check_border = True, delete_border = False, return_points = False, verbose = True):
  """Skeletonize a binary 3d array using PK12 algorithm.
  
  Arguments
  ---------
  binary : array
    Binary image to skeletonize.
  points : array or None.
    Optional list of points in the binary to speed up processing.
  steps : int or None
    Number of maximal iteration steps (if None maximal reduction).
  removals : bool
    If True, returns also the steps at which the pixels in the input data 
    where removed. 
  radii : bool
    If True, the estimate of the local radius is returned.
  check_border : bool
    If True, check if the boder is empty. The algorithm reuqires this.
  delete_border : bool
    If True, delete the border.
  verbose : bool
    If True print progress info.
    
  Returns
  -------
  skeleton : array
    The skeleton of the binary.
  points : array
    The point coordinates of the skeleton nx3
  
  Note
  ----
  The skeletonization is done in place on the binary. Copy the binary if
  needed for further processing.
  """
  
  if verbose:    
    print('#############################################################'); 
    print('Skeletonization PK12 [convolution]');
    timer = tmr.Timer();
  
  #TODO: make this work for any memmapable source !
  if not isinstance(binary, np.ndarray):
    raise ValueError('Numpy array required for binary in skeletonization!');
  if binary.ndim != 3:
    raise ValueError('The binary array dimension is %d, 3 is required!' % binary.ndim);    
  
  if delete_border:
    binary = t3d.delete_border(binary);
    check_border = False;
  
  if check_border:
    if not t3d.check_border(binary):
      raise ValueError('The binary array needs to have no points on the border!');  
  
  # detect points
  #points = np.array(np.nonzero(binary)).T;
  if points is None:
    points = ap.where(binary).array;
  
  if verbose:
    timer.print_elapsed_time(head='Foreground points: %d' % (points.shape[0],));

  if removals is True or radii is True:
    #birth = np.zeros(binary.shape, dtype = 'uint16');
    death = np.zeros(binary.shape, dtype = 'uint16');
    with_info = True;
  else:
    with_info = False;
  
  # iterate
  if steps is None:
    steps = -1;
  step = 1;
  removed = 0;
  while True:
    if verbose:
      print('#############################################################');
      print('Iteration %d' % step);
      timer_iter = tmr.Timer();
  
    border = cpl.convolve_3d_points(binary, t3d.n6, points) < 6;
    borderpoints = points[border];
    borderids    = np.nonzero(border)[0];
    keep         = np.ones(len(border), dtype = bool);
    if verbose:  
      timer_iter.print_elapsed_time('Border points: %d' % (len(borderpoints),));
    
    #if info is not None:
    #  b = birth[borderpoints[:,0], borderpoints[:,1], borderpoints[:,2]];
    #  bids = b == 0;
    #  birth[borderpoints[bids,0], borderpoints[bids,1], borderpoints[bids,2]] = step;
      
    # sub iterations
    remiter = 0;
    for i in range(12):
      if verbose:
        print('-------------------------------------------------------------');
        print('Sub-Iteration %d' % i);
        timer_sub_iter = tmr.Timer();
      
      remborder = delete[cpl.convolve_3d_points(binary, rotations[i], borderpoints)];
      rempoints = borderpoints[remborder];
      if verbose:
        timer_sub_iter.print_elapsed_time('Matched points: %d' % (len(rempoints),));
      
      binary[rempoints[:,0], rempoints[:,1], rempoints[:,2]] = 0;
      keep[borderids[remborder]] = False;
      rem = len(rempoints);
      remiter += rem;
      removed += rem;
      if verbose:
        print('Deleted points: %d' % (rem));
        timer_sub_iter.print_elapsed_time('Sub-Iteration %d' % (i));
        
      #death times
      if with_info is True:
        #remo = np.logical_not(keep);
        death[rempoints[:,0], rempoints[:,1], rempoints[:,2]] = 12 * step + i;

    #update foreground
    points = points[keep];
    if verbose:
      print('Foreground points: %d' % points.shape[0]);  
    
    if verbose:
      print('-------------------------------------------------------------');
      timer_iter.print_elapsed_time('Iteration %d' % (step,));
    
    step += 1;
    if steps >= 0 and step >= steps:
      break
    if remiter == 0:
      break

  if verbose:
    print('#############################################################');
    print('Total removed:   %d' % (removed));
    print('Total remaining: %d' % (len(points)));
    timer.print_elapsed_time('Skeletonization');
  
  result = [binary];
  if return_points:
    result.append(points);
  if removals is True:
    result.append(death);
  if radii is True:
    #calculate average diameter as average death of neighbourhood
    radii = cpl.convolve_3d(death, np.array(t3d.n18, dtype = 'uint16'), points);
    result.append(radii);
  
  if len(result) > 1:
    return tuple(result);
  else:
    return result[0];


def skeletonize_index(binary, points = None, steps = None, removals = False, radii = False, return_points = False, check_border = True, delete_border = False, verbose = True):
  """Skeletonize a binary 3d array using PK12 algorithm via index coordinates.
  
  Arguments
  ---------
  binary : array
    Binary image to be skeletonized. 
  steps : int or None
    Number of maximal iteration steps. If None, use maximal reduction.
  removals :bool
    If True, returns the steps in which the pixels in the input data 
    were removed.
  radii :bool
    If True, the estimate of the local radius is returned.
  verbose :bool
    If True, print progress info.
    
  Returns
  -------
  skeleton : array
    The skeleton of the binary input.
  points : nxd array
    The point coordinates of the skeleton.
  """
  
  if verbose:    
    print('#############################################################'); 
    print('Skeletonization PK12 [convolution, index]');
    timer = tmr.Timer();
  
  #TODO: make this work for any memmapable source
  if not isinstance(binary, np.ndarray):
    raise ValueError('Numpy array required for binary in skeletonization!');
  if binary.ndim != 3:
    raise ValueError('The binary array dimension is %d, 3 is required!' % binary.ndim);    
  
  if delete_border:
    binary = t3d.delete_border(binary);
    check_border = False;
  
  if check_border:
    if not t3d.check_border(binary):
      raise ValueError('The binary array needs to have not points on the border!');      
  
  binary_flat = binary.reshape(-1, order = 'A');
  
  # detect points
  if points is None:
    points = ap.where(binary_flat).array;  
  npoints = points.shape[0];
  
  if verbose:
    timer.print_elapsed_time('Foreground points: %d' % (points.shape[0],));

  if removals is True or radii is True:
    #birth = np.zeros(binary.shape, dtype = 'uint16');
    order = 'C';
    if binary.flags.f_contiguous:
      order = 'F';
    death = np.zeros(binary.shape, dtype = 'uint16', order = order);
    deathflat = death.reshape(-1, order = 'A')
    with_info = True;
  else:
    with_info = False;
  
  # iterate
  if steps is None:
    steps = -1;
  step = 1;
  nnonrem = 0;
  while True:
    if verbose:
      print('#############################################################');
      print('Iteration %d' % step);
      timer_iter = tmr.Timer();
  
  
    print(type(points), points.dtype, binary.dtype)
    border = cpl.convolve_3d_indices_if_smaller_than(binary, t3d.n6, points, 6);
    borderpoints = points[border];
    #borderids    = np.nonzero(border)[0];
    borderids    = ap.where(border).array;
    keep         = np.ones(len(border), dtype = bool);
    if verbose:  
      timer_iter.print_elapsed_time('Border points: %d' % (len(borderpoints),));
    
    #if info is not None:
    #  b = birth[borderpoints[:,0], borderpoints[:,1], borderpoints[:,2]];
    #  bids = b == 0;
    #  birth[borderpoints[bids,0], borderpoints[bids,1], borderpoints[bids,2]] = step;
      
    # sub iterations
    remiter = 0;
    for i in range(12):
      if verbose:
        print('-------------------------------------------------------------');
        print('Sub-Iteration %d' % i);
        timer_sub_iter = tmr.Timer();
      
      remborder = delete[cpl.convolve_3d_indices(binary, rotations[i], borderpoints)];
      rempoints = borderpoints[remborder];
      if verbose:
        timer_sub_iter.print_elapsed_time('Matched points  : %d' % (len(rempoints),));
      
      binary_flat[rempoints] = 0;
      keep[borderids[remborder]] = False;
      rem = len(rempoints);
      remiter += rem;

      #death times
      if with_info is True:
        #remo = np.logical_not(keep);
        deathflat[rempoints] = 12 * step + i;
        
      if verbose:
        timer_sub_iter.print_elapsed_time('Sub-Iteration %d' % (i,));

    if verbose:
      print('-------------------------------------------------------------');

    #update foregroud
    points = points[keep];
    
    if step % 3 == 0:   
      npts = len(points);
      points = points[consider[cpl.convolve_3d_indices(binary, base, points)]]; 
      nnonrem += npts - len(points)
      if verbose:
        print('Non-removable points: %d' % (npts - len(points)));
    
    if verbose:
      print('Foreground points   : %d' % points.shape[0]);  
    
    if verbose:
      print('-------------------------------------------------------------');
      timer_iter.print_elapsed_time('Iteration %d' % (step,));
    
    step += 1;
    if steps >= 0 and step >= steps:
      break
    if remiter == 0:
      break
  
  if verbose:
    print('#############################################################');
    timer.print_elapsed_time('Skeletonization done');
    print('Total removed:   %d' % (npoints - (len(points) + nnonrem)));
    print('Total remaining: %d' % (len(points) + nnonrem));
  
  if radii is True or return_points is True:
    points = ap.where(binary_flat).array
  
  if radii is True:
    #calculate average diameter as death average death of neighbourhood     
    radii = cpl.convolve_3d_indices(death, t3d.n18, points, out_dtype = 'uint16');
  else:
    radii = None;
  
  result = [binary];
  if return_points:
    result.append(points);
  if removals is True:
    result.append(death);
  if radii is not None:
    result.append(radii);
    
  if len(result) > 1:
    return tuple(result);
  else:
    return result[0];


###############################################################################
### Tests
###############################################################################

def _test():
  import numpy as np;
  import ClearMap.IO.IO as io
  import ClearMap.Visualization.Plot3d as p3d
  import ClearMap.Tests.Files as tsf
  import ClearMap.ImageProcessing.Skeletonization.PK12 as PK12;
  from importlib import reload
  reload(PK12);
  
  #Lookup tables
  #lut = PK12.generate_lookup_table();
  #np.save(PK12.filename, lut);
  #lut.sum()
  
  #lutnr = PK12.generate_lookup_table(function=PK12.match_non_removable, verbose = True);
  #np.save(PK12.filename_non_removable, lutnr);
  #lutnr.sum()
  
  #Skeletonization
  reload(PK12)
  binary = tsf.skeleton_binary;
  binary_array = np.array(io.as_source(binary));
  
  #default version
  skeleton = PK12.skeletonize(binary_array.copy(), delete_border=True, verbose=True);
  p3d.plot([[binary_array, skeleton]])  
  
  #fast index version
  skeleton = PK12.skeletonize_index(binary_array.copy(), delete_border=True, verbose = True);  
  p3d.plot([[binary_array, skeleton]])  
  
  # plotting
  import ClearMap.Visualization.Plot3d as p3d
  p3d.plot_3d(binary_array[:150,:150,:150], cmap=p3d.grays_alpha(0.05));
  p3d.plot_3d(skeleton[:150,:150,:150], cmap=p3d.single_color_colormap('red', alpha = 0.8))
  
  #save for figure
  import scipy.io as sio
  sio.savemat('binary.mat', {'binary' : binary_array[:100,:100,:100]});
  sio.savemat('binary_skeleton.mat', {'skeleton': skeleton[:100,:100,:100]});


