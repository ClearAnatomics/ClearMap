# -*- coding: utf-8 -*-
"""
ProcessWriter
=============

Provides formatting tool to print text with parallel process header.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


import sys

class ProcessWriter(object):
  """Class to handle writing from parallel processes.
  
  Attributes
  ----------
  process : int
    The process number.
  """
  
  def __init__(self, process = 0):
    self.process = process;
  
  def string(self, text):
    """Generate string with process prefix.
    
    Arguments
    ---------
    text : str
      The text to write.
        
    Returns
    -------
    text : str
      The text with process prefix.
    """
    pre = ("Process %5s: " % ('%r' % self.process));
    return pre + str(text).replace('\n', '\n' + pre);
  
  def write(self, text):
    """Write string with process prefix to sys.stdout
    
    Arguments
    ---------
    text : str
      The text to write.
    """
    print(self.string(text));
    sys.stdout.flush();

    