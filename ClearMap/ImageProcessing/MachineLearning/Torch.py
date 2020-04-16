#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Torch
=====

Utility functions for PyTorch in ClearMap.
"""
__author__    = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__   = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__copyright__ = 'Copyright (c) 2019 by Christoph Kirst'


import torch

def to(t, dtype = float):
  """Convert torch object to a specified data type.
  
  Arguments
  ---------
  t : torch object
    The object to convert to a crtian data type.
  dtype : ['float', 'double', 'float64', 'float32', float16', 'half', float]
    The data type to use for the torch object.
    
  Returns 
  -------
  t : torch object
    The torch object in the requested data type.
  """
  if dtype in ['float', 'double', 'float64', float]:
    return t.double();
  elif dtype in ['float32']:
    return t.float();
  elif dtype in ['float16', 'half']:
    return t.half();
  else:
    raise ValueError('Data type %r not supported !' % dtype)
    
    
def gpu_info():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  info = 'Device: %r\n' % device

  if device.type == 'cuda':
    info += torch.cuda.get_device_name(0) + '\n';
    info += 'Memory Usage:\n';
    info += 'Allocated: %dGB\n' % round(torch.cuda.memory_allocated(0)/1024**3,1);
    info += 'Cached:    %dGB\n' % round(torch.cuda.memory_cached(0)/1024**3,1);
    
  return info;