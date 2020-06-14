# -*- coding: utf-8 -*-
"""
Torch
=====

Utility functions for PyTorch in ClearMap.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


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