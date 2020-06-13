# -*- coding: utf-8 -*-
"""
RankResmaple
============

Rank filter on resampled data.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


import sys
import numpy as np
import cv2

from . import Rank as rnk

import ClearMap.ParallelProcessing.DataProcessing.ArrayProcessing as ap

import ClearMap.Utils.Timer as tmr


###############################################################################
### rank
###############################################################################

#TODO: use resmapling module
def rank(source, sink = None, function = rnk.median, resample = None, verbose = False, out = sys.stdout, **kwargs):
  """Rank filter inbetween reshaping."""
  
  timer = tmr.Timer();
  
  sink, sink_buffer = ap.initialize_sink(sink=sink, source=source, order='F');
    
  if resample:
    interpolation = cv2.INTER_NEAREST;
    new_shape = np.round(np.array(sink.shape, dtype = float) * resample).astype(int);
    new_shape[2] = sink.shape[2];    
    data = np.zeros(tuple(new_shape), order = 'F', dtype = source.dtype);
    new_shape = tuple(new_shape[1::-1]);                   
    for z in range(source.shape[2]):              
       data[:,:,z] = cv2.resize(src = source[:,:,z], dsize = new_shape, interpolation = interpolation); 
    #print data.shape, data.dtype                     
    out.write(timer.elapsed_time(head = 'Rank filter: Resampling') + '\n');
  else:
    data = source;             
  
  #keys = inspect.getargspec(function).args;
  #kwargs = { k : v for k,v in kwargs.iteritems() if k in keys};  
  
  data = function(data, **kwargs);
  
  out.write(timer.elapsed_time(head = 'Rank filter: %s' % function.__name__) + '\n');        
  
  if resample:
    #interpolation = cv2.INTER_LINEAR;
    interpolation = cv2.INTER_AREA;
    for z in range(sink.shape[2]):              
       sink_buffer[:,:,z] = cv2.resize(src = data[:,:,z], dsize = sink.shape[1::-1], interpolation = interpolation); 
    out.write(timer.elapsed_time(head = 'Rank filter: Upsampling') + '\n');
  else:              
    sink_buffer[:] = data;
          
  return sink;