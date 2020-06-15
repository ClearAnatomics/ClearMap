# -*- coding: utf-8 -*-
"""
Timer
=====

Module provides tools for timing information.

Example
-------

>>> import ClearMap.Utils.Timer as timer
>>> t = timer.Timer();
>>> for i in range(100000000):
>>>   x = 10 + i;
>>> t.print_elapsed_time('test')

"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'
import time

import ClearMap.Utils.Sound as snd;


class Timer(object):
  """Class to stop time and print results in formatted way
  
  Attributes
  ----------
  time : float
    The time since the timer was started.
  head : str or None
    Option prefix to the timing string.
  """
  
  def __init__(self, head = None):
    self.head = head;
    self.start();
  
  def start(self):
    """Start the timer"""
    self.time = time.time();

  def reset(self):
    """Reset the timer"""
    self.time = time.time();

  def elapsed_time(self, head = None, as_string = True):
    """Calculate elapsed time and return as formated string
    
    Arguments
    ---------
    head : str or None
      Prefix to the timing string.
    as_string : bool
      If True, return as string, else return elapsed time as float.
    
    Returns
    -------
    time : str or float
      The elapsed time information.
    """
    
    t = time.time();
    
    if as_string:
      t = self.format_time(t - self.time);
      if head is None:
        head = self.head;
      elif self.head is not None:
        head = self.head + head;
      if head is not None:
        return head + ": elapsed time: " + t;
      else:
        return "Elapsed time: " + t;
    else:
      return t - self.time;
  
  def print_elapsed_time(self, head = None, beep = False):
    """Print elapsed time.
    
    Arguments
    ---------
    head : str or None
      Prefix to the timing string.
    beep : bool
      If True, beep in addition to print the time.
    """    
    print(self.elapsed_time(head = head));
    if beep:
      snd.beep();
  
  def format_time(self, t):
    """Format time to string.
    
    Arguments
    ---------
    t :float
      Time in seconds to format.
    
    Returns
    -------
    time : str
      The time as 'hours:minutes:seconds:milliseconds'.
    """
    m, s = divmod(t, 60);
    h, m = divmod(m, 60);
    ms = 1000 * (s % 1);
    return "%d:%02d:%02d.%03d" % (h, m, s, ms);
    
  def __str__(self):
    return self.elapsed_time();
  
  def __repr__(self):
    return self.__str__();


def timeit(method):
  def timed(*args, **kw):
      ts = time.time()
      result = method(*args, **kw)
      te = time.time()
      
      m, s = divmod(te-ts, 60);
      h, m = divmod(m, 60);
      ms = 1000 * (s % 1);
      
      print("%r took %d:%02d:%02d.%03d" % (method.__name__, h, m, s, ms))
      return result

  return timed


def _test():
  import ClearMap.Utils.Timer_Future as timer
  t = timer.Timer(head = 'Testing');
  for i in range(10000):
    x = 10 + i;
  t.print_elapsed_time('test')
  
  print(t)