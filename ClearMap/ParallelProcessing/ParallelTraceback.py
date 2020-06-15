# -*- coding: utf-8 -*-
"""
ParalllelTraceback
==================

Decorator to traceback errors in parallel processes.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


def parallel_traceback(func):
  """Wrapper to obtain a full traceback when exceuting a function in parallel.
  
  Arguments
  ---------
  func : function
   The function to call.
  
  Returns
  -------
  wrapper : function
    The function wrapped with an appropiate traceback functionality.
  """
  import traceback, functools
  @functools.wraps(func)
  def wrapper(*args, **kwargs):
    try:
      return func(*args, **kwargs)
    except Exception as e:
      msg = "{}\n\nOriginal {}".format(e, traceback.format_exc())
      raise type(e)(msg)
  return wrapper