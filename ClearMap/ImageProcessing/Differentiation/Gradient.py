# -*- coding: utf-8 -*-
"""
Gradient
========

Module to calculate various curvature and tube measures in 3D
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'

import numpy as np


__all__ = ['gradient', 'gradient_abs', 'gradient_square'];


##############################################################################
### Gradients
##############################################################################

def gradient(source):
  """Returns the finite difference gradient vector at each point.
  
  Arguments
  ---------
  source : array
    The data source.
    
  Returns
  -------
  gradient : array
    A (ndim,) + source.shape array of the finte differences alon each axis.
  """
  ndim = source.ndim; 
     
  mm = np.pad(source, (0,1), 'edge');
  mm = mm.astype(float);    
        
  g = np.zeros((ndim,) + source.shape);
  sl0 = [slice(None, -1)] * ndim;
  for d in range(ndim):  
    sl1 = [slice(None, -1)] * ndim;
    sl1[d] = slice(1,None);             
    g[d] = mm[sl1] - mm[sl0];
  
  return d;


def gradient_abs(source):
  """Returns the absolute magnitude of the gradient vector at each point.
  
  Arguments
  ---------
  source : array
    The data source.
    
  Returns
  -------
  abs : array
    Sum of the absolute values of the gradient vector entries.
  """
  d = gradient(source);
  return np.sum(np.abs(d), axis = 0);


def gradient_square(source):
  """Returns the square sum of the gradient vector entries.
  
  Arguments
  ---------
  source : array
    The data source.
    
  Returns
  -------
  abs : array
    Sum of the absolute values of the gradient vector entries.
  """
  d = gradient(source);
  return np.sum(d*d, axis = 0);

