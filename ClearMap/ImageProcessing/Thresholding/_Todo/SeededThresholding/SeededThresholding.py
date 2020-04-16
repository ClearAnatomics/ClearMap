"""
Module to compute clipped images

Usefull to sace memory in large data sets
"""
__author__    = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__   = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__copyright__ = 'Copyright (c) 2017 by Christoph Kirst, The Rockefeller University, New York City'


import numpy as np

import pyximport;
pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)

import ClearMap.ImageProcessing.Filter.SeedThresholding.SeedThresholdingCode as code


def threshold(source, sink = None, seeds_x = None, seeds_y = None, seeds_z = None, percentage = 0.5, absolute = 0):
    """Threshold image from seeds
    
    Arguments
    ---------
    source : 3-D array
        Input source.
    sink : 3-D array
        If None, a new array is allocated.
   

    Returns
    -------
    sink : 3-D array
        clipped output.
    """

    if source.ndim != 3:
      raise ValueError('source assumed to be 3d found %dd' % source.ndim);
    
    if source is sink:
        raise NotImplementedError("Cannot perform operation in place.")

    if sink is None:
      sink = np.zeros(source.shape, dtype = bool, order = 'F')
    
    if sink.dtype == bool:
      s = sink.view('uint8')
    else:
      s = sink;
    
    if seeds_x is None or seeds_y is None or seeds_z is None:
      return sink;
    
#    if seeds_x.dtype == bool:
#      x = seeds_x.view('uint8')
#    else:
#      x = seeds_x;
    code.threshold_from_seeds(source, s, seeds_x, seeds_y, seeds_z, percentage, absolute);
    
    return sink;


from scipy.ndimage.filters import maximum_filter


def local_max(source, sink = None, shape = None):
  """Calculates local maxima of an image
        
    Arguments
    ---------
        img :(array): image
        shape (int, tuple or None): shape of the volume to search for maxima
        
    Returns:
        array: mask that is True at local maxima
  """
  if shape is None:
    shape = 3;
    
  if sink is None:
    sink = np.empty(source.shape, dtype = bool, order = 'F');
  
  sink[:] = (maximum_filter(source, size = shape) == source);

  return sink;


if __name__ == '__main__':
  import numpy as np
  import ClearMap.Visualization.Plot3d as p3d
  import ClearMap.ImageProcessing.Filter.SeedThresholding.SeedThresholding as sth   
  
  from importlib import reload
  reload(sth);
  
  r = np.arange(50);        
  x,y,z = np.meshgrid(r,r,r);
  x,y,z = [i - 25 for i in (x,y,z)];                     
  d = np.exp(-(x*x + y*y +z*z)/10.0**2)                  
  m = sth.local_max(d, shape = (3,3,3));
  x,y,z = np.where(m);                   
  t = sth.threshold(d, sink = None, seeds_x = x, seeds_y = y, seeds_z = z, percentage = 0.5, absolute = 0.2);               
  p3d.plot([[d,m,t]])     
        
        
  import ClearMap.Test.Files as tst        
  d = tst.init('v')[:200,:200,:100];
  import ClearMap.ImageProcessing.Filter.Rank as rnk
  d = np.array(d);
  d = rnk.median(d, np.ones((3,3,3), dtype = bool));
  
  m = sth.local_max(d, shape = (5,5,1));
  m = np.logical_and(m, d > 10);
  x,y,z = np.where(m);                      
  v = d[x,y,z];
  s = np.argsort(v)[::-1];
  x,y,z = [i[s] for i in (x,y,z)];                 
  t = sth.threshold(d, sink = None, seeds_x = x, seeds_y = y, seeds_z = z, percentage = 0.25, absolute = 5);
  p3d.plot([d, [d,m,t]])
