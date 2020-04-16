# -*- coding: utf-8 -*-
"""
Provides formatting tool to print text with parallel process header.
"""
__author__    = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__   = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__copyright__ = 'Copyright (c) 2018 by Christoph Kirst, The Rockefeller University, New York City'


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

    