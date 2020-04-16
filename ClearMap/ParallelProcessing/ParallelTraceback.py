# -*- coding: utf-8 -*-
"""
Traceback for parallel processes.
"""
__author__    = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__   = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__copyright__ = 'Copyright (c) 2017 by Christoph Kirst, The Rockefeller University, New York City'


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