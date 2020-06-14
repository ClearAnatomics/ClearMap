# -*- coding: utf-8 -*-
"""
StrutureElement
===============

Routines to generate structure elements for filters.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'

import numpy as np


def disk(shape = (3,3)):
  """Disk structuring element."""
  return sphere(shape = shape) > 0;


def sphere(shape = (3,3)):
  """Disk structuring element."""
  offsets = structure_element_offsets(shape);
  mesh = [range(-o[0], o[1]) for o in offsets];         
  mesh = np.array(np.meshgrid(*mesh, indexing = 'ij'), dtype = float);
                 
  add = ((np.array(shape) + 1) % 2) / 2.0;   
  nrm = np.max(offsets, axis=1);  
  for d in range(len(add)):
    mesh[d] = (mesh[d] + add[d]) / nrm[d];
  
  r = 1 - np.sum(mesh * mesh, axis = 0);
  r[r < 0] = 0;
  r /= r.sum();
  return r;


def cube(shape = (3,3)):
  """Cube structuring element."""
  return np.ones(shape, dtype = bool);


def structure_element(shape = (3,3), form = 'Disk', ndim = None):
    """Creates specific 2d and 3d structuring elements
      
    Arguments
    ---------
    shape : array or tuple
      Shape of the structure element.
    form : str
      structure element type      
    
    Returns:
        array
            structure element
    """
    if isinstance(shape, int) and ndim is not None:
      shape = (shape,) * ndim;
    
    if isinstance(shape, tuple):
      shape = np.array([shape]).flatten();   
      
      if ndim is None:
        ndim = len(shape);
      else:
        shape = np.pad(shape[:ndim], (0, max(0, ndim - len(shape))), 'wrap');
      
      if form in ['Disk', 'disk', 'd']:
        return disk(shape = shape);
      elif form in ['Sphere', 'shpere', 's']:
        return sphere(shape = shape);
      elif form in ['Cube', 'cube', 'c', 'Rectangle', 'rectangle', 'r']:
        return cube(shape = shape);
      else:
        ValueError('Form %r for structuring element not valid!' % form);
    else:
      return shape;                  


def structure_element_offsets(shape):
    """Calculates offsets to center for a structural element given its shape.
    
    Arguments
    ---------
    shape : array or tuple
      Shape of the structure element
    
    Returns
    -------
    offsets : array
      Offsets to center taking care of even/odd number of elements.
    """
    off = np.array([shape], dtype = int).flatten() // 2;
    off = np.array([off, shape - off]).T;
    return off;   


###############################################################################
### Tests
############################################################################### 

def _test():
  import ClearMap.ImageProcessing.Filter.StructureElement as se
  
  from importlib import reload
  reload(se)

  d = se.sphere((150,50,10));  
  
  import ClearMap.Visualization.Plot3d as p3d
  p3d.plot(d)             

