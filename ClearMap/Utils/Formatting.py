# -*- coding: utf-8 -*-
"""
Formatting
==========

Module for formatting type and functions.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


__all__ = ['as_type', 'ensure']


def as_type(value, types = [int, float]):
  """Tries to convert value to given data types.
  
  Arguments
  ---------
  value : object
    The value to be converted.
  types : list of types
    The list of types to try to convert the value to.
  
  Returns
  -------
  value : object
    The value converted to the types if possible.
  """
  for t in types:
    try:
      return t(value)
    except Exception:
      pass
  return value


def ensure(value, dtype):
  """Ensure values have a specified type but allowing for None values.
  
  Arguments
  ---------
  value : object
    The value to copy
  dtype : class
    The class type of the value to be copied.
  
  Returns
  -------
  value : object
    The value with the requested type.
  """
  if value is None:
    return None;
  else:
    if not isinstance(value, dtype):
      value = dtype(value);
    return dtype(value);
    
    
#    def __copy__(self):
#        cls = self.__class__
#        result = cls.__new__(cls)
#        result.__dict__.update(self.__dict__)
#        return result
#
#    def __deepcopy__(self, memo):
#        cls = self.__class__
#        result = cls.__new__(cls)
#        memo[id(self)] = result
#        for k, v in self.__dict__.items():
#            setattr(result, k, deepcopy(v, memo))
#        return result