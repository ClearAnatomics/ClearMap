# -*- coding: utf-8 -*-
"""
Topology3d
==========

Defines basic 3d discrete topology utils. 

Note
----
The definitions are compatible with a separable convolutional kernel with 
weights along the 3 dimensions given by:
  
>>> [[(2**(3**d))**k for k in range(3)] for d in range(3)]

"""
__author__    = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__   = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__copyright__ = 'Copyright (c) 2019 by Christoph Kirst, The Rockefeller University, New York City'

import tempfile
import numpy as np

import ClearMap.IO.IO as io
import ClearMap.ParallelProcessing.DataProcessing.ArrayProcessing as ap

###############################################################################
### Neighbourhoods
###############################################################################

n6 =  np.array([[[0,0,0],[0,1,0],[0,0,0]],
                [[0,1,0],[1,0,1],[0,1,0]], 
                [[0,0,0],[0,1,0],[0,0,0]]], dtype = bool);
"""6-Neighborhood excluding center"""


n18 = np.array([[[0,1,0],[1,1,1],[0,1,0]],
                [[1,1,1],[1,0,1],[1,1,1]], 
                [[0,1,0],[1,1,1],[0,1,0]]], dtype = bool);
"""18-Neighborhood excluding center"""


n26 = np.array([[[1,1,1],[1,1,1],[1,1,1]],
                [[1,1,1],[1,0,1],[1,1,1]],
                [[1,1,1],[1,1,1],[1,1,1]]], dtype = bool);
"""26-Neighborhood excluding center"""


###############################################################################
### Label and Base 2 representations / indexing
###############################################################################

def cube_labeled(center = None):
  """Returns an array with labels on the cube"""
  cube = np.zeros((3,3,3), dtype = int);
  if center:
    i = 0;
  else:
    i = 1;
  for z in range(3):
    for y in range(3):
      for x in range(3):
        if center is not None and x == 1 and y == 1 and z == 1:
          cube[x,y,z] = center;
        else:
          cube[x,y,z] = i;
          i+=1;
  return cube;


def cube_base_2(center = None):
  """Returns an array with base 2 numbers on the cube for convolution and lut matching"""
  cube = np.zeros((3,3,3), dtype = int);
  k = 0;
  for z in range(3):
    for y in range(3):
      for x in range(3):
        if center is not None and x == 1 and y ==1 and z == 1:
          cube[x,y,z] = center;
        else:
          cube[x,y,z] = 2**k;
          k+=1;
  return cube;


def cube_from_index(index, center = None):
  """Returns a boolean cube for the corresponding index"""
  cube = np.zeros((3,3,3), dtype = bool);
  d = 0;
  for z in range(3):
    for y in range(3):
      for x in range(3):
        if center is not None and x == 1 and y == 1 and z == 1:
          cube[x,y,z] = center;
        else:
          cube[x,y,z] = (index >> d) & 0x01;
          d += 1;
  return cube;
    

def cube_to_index(cube, center = None):
  """Returns index for a boolean cube"""
  return (cube_base_2(center=center) * np.array(cube)).sum()

 
def n_cube_indices(center = None):
  """Number of different cubes"""
  return np.sum(cube_base_2(center=center)) + 1;


def xyz_to_index(x,y,z):
  return x + 3 * y + 9 * z;


def xyz_from_ndex(index):
  index9 = index % 9;
  x = index9 % 3;
  y = index9 / 3;
  z = index / 9;
  return x,y,z


def n6_indices(dtype = 'uint8'):
  """Indices as 6 Neighbourhood"""
  xyz  = np.where(n6)
  return np.sort(np.array(map(xyz_to_index, *xyz), dtype = dtype)); 


def n18_indices(dtype = 'uint8'):
  """Indices as 6 Neighbourhood"""
  xyz  = np.where(n18)
  return np.sort(np.array(map(xyz_to_index, *xyz), dtype = dtype)); 


def n26_indices(dtype = 'uint8'):
  """Indices as 6 Neighbourhood"""
  xyz  = np.where(n26)
  return np.sort(np.array(map(xyz_to_index, *xyz), dtype = dtype)); 


def index_kernel(axis = None, dtype='uint32'):
  """Separable 1d kernels to obtain configuration index."""
  kernels = [np.array([(2**(3**d))**k for k in range(3)], dtype=dtype) for d in range(3)];
  if axis is None:
    kernel = 1;
    for k in kernels:
      kernel = np.multiply.outer(kernel, k);
    return np.array(kernel, dtype=dtype)
  else:
    return kernels[axis];


def index_from_binary(source, sink = None, method = 'shared', dtype = 'uint32', processes = None, verbose = False):
  """Calculate the local 3x3x3 configuration in a binary source.
  
  Note
  ----
  The configuration kernel is separable and convolution with it 
  is calculated via a sequence of 1d convolutions.
  """
  processes, timer = ap.initialize_processing(processes=processes, verbose=verbose, function='index_from_binary');
  
  #determine configuration
  source, source_buffer, source_shape, source_strides, source_order = ap.initialize_source(source, as_1d=True, return_shape=True, return_strides=True, return_order=True);
  ndim = len(source_shape);
  
  buffer_dtype = np.result_type(source_buffer.dtype, 'uint32');
    
  delete_files = [];
  if source_order == 'C':
    axis_range = range(ndim-1,-1,-1);
    axis_last = 0;
  else:
    axis_range = range(ndim);
    axis_last = ndim-1;
  for axis in axis_range:
    if axis == axis_last:
      sink, sink_buffer, sink_shape, sink_strides = ap.initialize_sink(sink=sink, as_1d=True, source=source, dtype=dtype, return_shape=True, return_strides=True); 
    else:
      if method == 'shared':
        _, sink_buffer, sink_shape, sink_strides = ap.initialize_sink(sink=None, as_1d=True, shape=source_shape, dtype=buffer_dtype, order=source_order,  return_shape=True, return_strides=True);
      else:
        location = tempfile.mktemp() + '.npy';
        _, sink_buffer, sink_shape, sink_strides = ap.initialize_sink(sink=location, as_1d=True, shape=tuple(source_shape), dtype=buffer_dtype, order=source_order,  return_shape=True, return_strides=True);
        delete_files.append(location);
    
    kernel = index_kernel(axis=axis, dtype=float);
    
    #print(source_buffer.dtype, source_buffer.shape, source_shape, source_strides, axis, sink_buffer.shape, sink_buffer.dtype, sink_strides, kernel.dtype)
    ap.code.correlate_1d(source_buffer, source_shape, source_strides, 
                         sink_buffer, sink_shape, sink_strides,
                         kernel, axis, processes);
    source_buffer = sink_buffer;
    
  for f in delete_files:
    io.delete_file(f);
    
  ap.finalize_processing(verbose=verbose, function='index_from_binary', timer=timer);
  
  return sink;


###############################################################################
### Transformations
###############################################################################

def rotate(cube, axis = 2, steps = 0):
  """Rotate a cube around an axis in 90 degrees steps"""
  cube = cube.copy();  
  
  steps = steps % 4;
  if steps == 0:
    return cube;
  
  elif axis == 0:
    if steps == 1:
      return cube[:, ::-1, :].swapaxes(1, 2)
    elif steps == 2:  # rotate 180 degrees around x
      return cube[:, ::-1, ::-1]
    elif steps == 3:  # rotate 270 degrees around x
      return cube.swapaxes(1, 2)[:, ::-1, :]
      
  elif axis == 1:
    if steps == 1:
      return cube[:, :, ::-1].swapaxes(2, 0)
    elif steps == 2:  # rotate 180 degrees around x
      return cube[::-1, :, ::-1]
    elif steps == 3:  # rotate 270 degrees around x
      return cube.swapaxes(2, 0)[:, :, ::-1]
      
  if axis == 2: # z axis rotation
    if steps == 1:
      return cube[::-1, :, :].swapaxes(0, 1)
    elif steps == 2:  # rotate 180 degrees around z
      return cube[::-1, ::-1, :]
    elif steps == 3:  # rotate 270 degrees around z
      return cube.swapaxes(0, 1)[::-1, :, :]


def reflect(cube):
  """Generate the center point reflection."""
  reflection = cube.copy();
  for x in range(3):
    xr = 2 - x;
    for y in range(3):
      yr = 2 - y;
      for z in range(3):
        zr = 2 - z;
        reflection[xr,yr,zr] = cube[x,y,z];
  
  return reflection;
        

def rotations6(cube):
  """Generate rotations in 6 main directions"""
  
  rotU = cube.copy();
  rotD = rotate(cube, axis = 0, steps = 2); 
  rotN = rotate(cube, axis = 0, steps = 1);
  rotS = rotate(cube, axis = 0, steps = 3);
  rotE = rotate(cube, axis = 1, steps = 3);
  rotW = rotate(cube, axis = 1, steps = 1);
  
  return [rotU, rotN, rotW, rotD, rotS, rotE];


def rotations12(cube):
  """Generate rotations in 12 diagonal directions"""
  
  rotUS = cube.copy();
  rotUW = rotate(cube, axis = 2, steps = 1);  
  rotUN = rotate(cube, axis = 2, steps = 2); 
  rotUE = rotate(cube, axis = 2, steps = 3);  

  rotDS = rotate(cube,  axis = 1, steps = 2);
  rotDW = rotate(rotDS, axis = 2, steps = 1); 
  rotDN = rotate(rotDS, axis = 2, steps = 2); 
  rotDE = rotate(rotDS, axis = 2, steps = 3);

  rotSW = rotate(cube, axis = 1, steps = 1);   
  rotSE = rotate(cube, axis = 1, steps = 3); 

  rotNW = rotate(rotUN, axis = 1, steps = 1);
  rotNE = rotate(rotUN, axis = 1, steps = 3);
  
  return [rotUS, rotNE, rotDW,  rotSE, rotUW, rotDN,  rotSW, rotUN, rotDE,  rotNW, rotUE, rotDS];


def orientations():
  """Generate cubes with True voxels at each of the 13 orientations"""
  
  cube0 = np.zeros((3,3,3), dtype = bool);
  
  o1 = cube0.copy(); o1[0,1,1] = True;
  o2 = cube0.copy(); o2[0,2,1] = True;
  o3 = cube0.copy(); o3[1,2,1] = True;
  o4 = cube0.copy(); o4[2,2,1] = True;
  
  o5 = cube0.copy(); o5[0,0,0] = True;
  o6 = cube0.copy(); o6[1,0,0] = True;
  o7 = cube0.copy(); o7[2,0,0] = True;
  o8 = cube0.copy(); o8[0,1,0] = True;
  o9 = cube0.copy(); o9[1,1,0] = True;
  o10= cube0.copy(); o10[2,1,0] = True;
  o11= cube0.copy(); o11[0,2,0] = True;
  o12= cube0.copy(); o12[1,2,0] = True;
  o13= cube0.copy(); o13[2,2,0] = True;
  
  return [o1, o2, o3,  o4, o5, o6, o7, o8, o9,  o10, o11, o12, o13];


###############################################################################
### Neighbourhood lists
###############################################################################

def neighbourhood_list(img, dtype = 'int64', verbose = False):
  """Return a list of x,y,z and list indices of the 26 neighbours"""
  if verbose:
    print("Generating neighbourhood...");    
  
  x,y,z = np.where(img);
  npts = len(x);
  
  if verbose:
    print('Creating labels...');
  
  label = np.full(img.shape, -1, dtype = dtype);
  label[x,y,z] = np.arange(npts);
  
  # calculate indices (if many voxels this is only 27 loops!)
  nhood = np.full((x.shape[0],27), -1, dtype = dtype);
  rg = range(3);
  direct = 0;
  for xx in rg:
    for yy in rg:
      for zz in rg:
        if verbose:
          print('Filling direction %d / 27' % direct);
          direct+=1;
        w = xx + yy * 3 + zz * 9;
        idx = x+xx-1; idy = y+yy-1; idz = z+zz-1;
        nhood[:,w] = label[idx,idy,idz];
        
  return (x,y,z,nhood);

 
def neighbourhood_list_delete(nhl, ids, changed = True):
  """Delete points in a neighbourhood list"""
  odirs = neighbourhood_opposing_directions();
  for i,oi in odirs:
    nhs = nhl[ids,i];
    nhi = nhs[nhs >= 0];
    nhl[nhi,oi] = -1;
 
  if changed: #collect non-deleted nodes with at least one neighbour deleted
    change = np.unique(nhl[ids].flatten());
    if len(change) > 0 and change[0] == -1:
      change = change[1:];
  
  # delete remaining neighbours of deleted pixels  
  nhl[ids] = -1;
  
  if changed:
    return nhl, change;
  else:
    return nhl;
  

def neighbourhood_opposing_directions():
  """Returns a list of neighbour indices that are opposite of each other"""
  dirs = np.zeros((27, 2), dtype = int);
  opdir = np.array([2,1,0]);
  rg = range(3);
  i = 0;
  for xx in rg:
    for yy in rg:
      for zz in rg:
        #w = _xyz_to_neighbourhood[xx,yy,zz];
        w = xx + yy * 3 + zz * 9;
        w2 = opdir[xx] + opdir[yy] * 3 + opdir[zz] * 9;
        dirs[i] = [w,w2];
        i+=1;
  return dirs;


def extract_neighbourhood(img,x,y,z):
  """Return the neighbourhoods of the indicated voxels
  
  Arguments:
    img (array): the 3d image
    x,y,z (n array): coordinates of the voxels to extract neighbourhoods from
  
  Returns:
    array (nx27 array): neighbourhoods
    
  Note:
    Assumes borders of the image are zero so that 0<x,y,z<w,h,d !
  """
  nhood = np.zeros((x.shape[0],27), dtype = bool);
  
  # calculate indices (if many voxels this is only 27 loops!)
  for xx in range(3):
    for yy in range(3):
      for zz in range(3):
        #w = _xyz_to_neighbourhood[xx,yy,zz];
        w = xx + 3 * yy + 9 * zz;
        idx = x+xx-1; idy = y+yy-1; idz = z+zz-1;
        nhood[:,w]=img[idx, idy, idz];
  
  nhood.shape = (nhood.shape[0], 3, 3, 3);
  nhood[:, 1, 1, 1] = 0;
  return nhood;


###############################################################################
###  Uitility
###############################################################################
  
def delete_border(data, value = 0):
  data[[0,-1],:,:] = value;
  data[:,[0,-1],:] = value;
  data[:,:,[0,-1]] = value;
  return data;

def check_border(data, value = 0):
  if np.any(data[[0,-1],:,:] != value):
    return False;
  if np.any(data[:,[0,-1],:] != value):
    return False;
  if np.any(data[:,:,[0,-1]] != value):
    return False;
  return True;

###############################################################################
### Printing 
###############################################################################

def print_cube(cube):
  """Print the cube for debugging"""
  #for z in range(3):
  for y in range(2,-1,-1):
    print('D:{} M:{} U:{}'.format(cube[:,y,0], cube[:,y,1], cube[:,y,2]));
    #print ""
  print("---")


###############################################################################
### Testing
###############################################################################

def _test():
  import numpy as np
  import ClearMap.ImageProcessing.Topology.Topology3d as top
  
  from importlib import reload 
  reload(top)
  
  label = top.cube_labeled();
  top.print_cube(label)
  
  # Test rotations
  c = np.zeros((3,3,3), dtype= bool);
  c[1,0,0] = True;
  top.print_cube(c)
  
  cs = [top.rotate(c, axis = 2, steps = r) for r in range(4)];
  [top.print_cube(cc) for cc in cs]
  
  reload(top)
  l = top.cube_labeled();
  rts = top.rotations6(l);
  
  [top.print_cube(r) for r in rts]
  
  reload(top);
  b = top.cube_from_index(6);
  i = top.cube_to_index(b);
  print(i,6)
  
  
  us = np.zeros((3,3,3), dtype = int);
  us[1,1,2] = 1;
  us[1,0,1] = 1;
  us[1,2,0] = 2;
            
  r12 = top.rotations12(us);
  [top.print_cube(cc) for cc in r12]
  

  #check configuration utlity
  reload(top)
  index = 11607;
  source = top.cube_from_index(index);
  
  c = top.index_from_binary(source);
  c[1,1,1] == index
  
  
  x = np.random.rand(1500,500,500) > 0.6;
  c =top.index_from_binary(x)
  
  
  import numpy as np
  import ClearMap.ImageProcessing.Topology.Topology3d as top
  
  #check fortran vs c order
  x = np.random.rand(5,5,5) > 0.35;
  y = np.asanyarray(x, order='F')
  
  ix = top.index_from_binary(x)
  iy = top.index_from_binary(y)
  
  ax = ix.array;
  ay = iy.array;
  
  
  #%% profile 
  import io
  io.DEFAULT_BUFFER_SIZE = 2**32
  
  import pstats, cProfile
  
  import numpy as np
  import ClearMap.ImageProcessing.Topology.Topology3d as top
  
  x = np.ones((3000,500,1000), dtype=bool, order = 'F');
  
  import ClearMap.IO.IO as io
  import ClearMap.ParallelProcessing.DataProcessing.ArrayProcessing as ap
  ap.write('test.npy', x)
  
  y = io.as_source('test.npy')
  z = io.create('resuly.npy', shape=y.shape, order='C', dtype='uint32');
  
  cProfile.runctx("c =top.index_from_binary(y, method='!shared', sink=z, verbose=True, processes=None)", globals(), locals(), "Profile.prof")
  
  s = pstats.Stats("Profile.prof")
  s.strip_dirs().sort_stats("time").print_stats()

  import mmap
  mmap.ACCESS_COPY