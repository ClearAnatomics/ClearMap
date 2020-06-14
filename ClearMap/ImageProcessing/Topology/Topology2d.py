# -*- coding: utf-8 -*-
"""
Topology2d
==========

Defines basic 2d discrete topology utils.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'



import numpy as np

###############################################################################
### Topology
###############################################################################

# Topology numbers T8 and T4bar in 2D
#Configuration:
# 3 2 1			
# 4 X 0
# 5 6 7

t4_bar = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1,
       1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 2, 2,
       2, 2, 1, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2,
       1, 2, 1, 2, 2, 3, 2, 2, 1, 2, 1, 1, 1, 2, 1, 1, 0, 1, 0, 1, 1, 2, 1,
       1, 0, 1, 0, 1, 1, 2, 1, 2, 1, 2, 1, 2, 2, 3, 2, 2, 1, 2, 1, 1, 1, 2,
       1, 1, 0, 1, 0, 1, 1, 2, 1, 1, 0, 1, 0, 1, 1, 2, 1, 2, 1, 2, 1, 2, 2,
       3, 2, 2, 1, 2, 1, 2, 2, 3, 2, 2, 1, 2, 1, 2, 2, 3, 2, 2, 1, 2, 1, 2,
       2, 3, 2, 3, 2, 3, 2, 3, 3, 4, 3, 3, 2, 3, 2, 2, 2, 3, 2, 2, 1, 2, 1,
       2, 2, 3, 2, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 2, 3, 2, 2, 1, 2,
       1, 1, 1, 2, 1, 1, 0, 1, 0, 1, 1, 2, 1, 1, 0, 1, 0, 1, 1, 2, 1, 2, 1,
       2, 1, 2, 2, 3, 2, 2, 1, 2, 1, 1, 1, 2, 1, 1, 0, 1, 0, 1, 1, 2, 1, 1,
       0, 1, 0]);
  
t8 = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1,
       1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 2, 2,
       2, 2, 1, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2,
       1, 2, 1, 2, 2, 3, 2, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1,
       1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 2, 3, 2, 2, 1, 2, 1, 1, 1, 2,
       1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 2,
       3, 2, 2, 1, 2, 1, 2, 2, 3, 2, 2, 1, 2, 1, 2, 2, 3, 2, 2, 1, 2, 1, 2,
       2, 3, 2, 3, 2, 3, 2, 3, 3, 4, 3, 3, 2, 3, 2, 2, 2, 3, 2, 2, 1, 2, 1,
       2, 2, 3, 2, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 2, 3, 2, 2, 1, 2,
       1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1,
       2, 1, 2, 2, 3, 2, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1,
       1, 1, 1])


###############################################################################
### Labels and Indices
###############################################################################

def plane_label(center = None):
  if center is not None:
    base = np.array([[3,2,1],[4,0,0],[5,6,7]]);
    base = np.power(2, base);
    base[1,1] = center;
  else:
    base = np.array([[4,3,2],[5,0,1],[6,7,8]]);
    base = np.power(2, base);
  return base;


def plane_base_2(center = None):
  """Returns an array with base 2 numbers on the plane for convolution and lut matching"""
  plane = np.zeros((3,3), dtype = int);
  k = 0;
  for y in range(3):
    for x in range(3):
      if center is not None and x == 1 and y ==1:
        plane[x,y] = center;
      else:
        plane[x,y] = 2**k;
        k+=1;
  return plane;


def plane_from_index(index, center = None):
  """Returns a boolean plane for the corresponding index"""
  plane = np.zeros((3,3), dtype = bool);
  d = 0;
  for y in range(3):
    for x in range(3):
      if center is not None and x == 1 and y == 1:
        plane[x,y] = center;
      else:
        plane[x,y] = (index >> d) & 0x01;
        d += 1;
  return plane;

  
def plane_to_index(plane, center = None):
  """Returns index for a boolean cube"""
  return (plane_base_2(center=center) * np.array(plane)).sum()


###############################################################################
### Neighbourhoods
###############################################################################

#TODO: still needed ? -> clean up
def extract_neighbourhood(img,x,y):
  """Return the neighbourhoods of the indicated voxels
  
  Arguments:
    img (array): the 2d image
    x,y (n array): coordinates of the voxels to extract neighbourhoods from
  
  Returns:
    array (nx9 array): neighbourhoods
    
  Note:
    Assumes borders of the image are zero so that 0<x,y<w,h !
  """
  nhood = np.zeros((x.shape[0],9), dtype = bool);
  
  # calculate indices (if many voxels this is only 9 loops!)
  for xx in range(3):
    for yy in range(3):
        #w = _xyz_to_neighbourhood[xx,yy,zz];
        w = 3 * xx + yy;
        idx = x+xx-1; idy = y+yy-1;
        nhood[:,w]=img[idx, idy];
  
  nhood.shape = (nhood.shape[0], 3, 3);
  nhood[:, 1, 1] = 0;
  return nhood;


###############################################################################
### Testing
###############################################################################

def _test():
  import ClearMap.ImageProcessing.Topology.Topology2d as t2d;
  
  p = [[1,0,0],[0,0,1],[1,0,0]];
  pi = t2d.plane_to_index(p)
  print('t4b: %d, t8: %d' % (t2d.t4_bar[pi], t2d.t8[pi]))
  
  p = [[1,1,0],[0,0,1],[1,0,0]];
  pi = t2d.plane_to_index(p)
  print('t4b: %d, t8: %d' % (t2d.t4_bar[pi], t2d.t8[pi]))

  p2 = t2d.plane_from_index(pi)
  assert np.all(p == p2)


