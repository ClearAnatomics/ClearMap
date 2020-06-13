# -*- coding: utf-8 -*-
"""
FilterKernel
=============

Implementation of various volumetric filter kernels.


.. _FilterTypes:

Filter Type
-----------

Filter types defined by the ``ftype`` key include: 

=============== =====================================
Type            Descrition
=============== =====================================
``mean``        uniform averaging filter
``gaussian``    Gaussian filter
``log``         Laplacian of Gaussian filter (LoG)
``dog``         Difference of Gaussians filter (DoG)
``sphere``      Sphere filter
``disk``        Disk filter
=============== =====================================
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'

import numpy as np
import math

import ClearMap.ImageProcessing.Filter.StructureElement as se


###############################################################################
### Filter kernel
###############################################################################

def filter_kernel(ftype = 'Gaussian', shape = (5,5), radius = None, sigma = None, sigma2 = None):
  """Creates a filter kernel of a special type
  
  Arguments
  ---------
  ftype  : str
    Filter type, see :ref:`FilterTypes`.
  shape : array or tuple
    Shape of the filter kernel.
  radius : tuple or float
    Radius of the kernel (if applicable).
  sigma : tuple or float
    Std for the first gaussian (if applicable).
  sigma2 : tuple or float
    Std of a second gaussian (if present).
  
  Returns
  -------
  krenel : array
    Filter kernel.
  """    
  
  ndim = len(shape);
  if ndim == 2:
    return filter_kernel_2d(ftype=ftype, shape=shape, sigma=sigma, sigma2=sigma2, radius=radius);
  elif ndim == 3:
    return filter_kernel_3d(ftype=ftype, shape=shape, sigma=sigma, sigma2=sigma2, radius=radius);
  else:
    raise ValueError('Kernel for %d dimensions not implemented !' % ndim);


def filter_kernel_2d(ftype = 'Gaussian', shape = (5,5), sigma = None, sigma2 = None, radius = None):
  """Creates a 2d filter kernel of a special type
  
  Arguments
  ---------
  ftype  : str
    Filter type, see :ref:`FilterTypes`.
  shape : array or tuple
    Shape of the filter kernel.
  radius : tuple or float
    Radius of the kernel (if applicable).
  sigma : tuple or float
    Std for the first gaussian (if applicable).
  sigma2 : tuple or float
    Std of a second gaussian (if present).
  
  Returns
  -------
  krenel : array
    Filter kernel.
  """    
  
  ftype = ftype.lower();
  o = se.structure_element_offsets(shape);
  mo = o.min(axis=1);
  shape = np.array(shape);
  
  if ftype == 'mean':  # unifrom
      return np.ones(shape)/ shape.prod();
  
  elif ftype == 'gaussian':        
      if sigma == None:
         sigma = shape / 2. / math.sqrt(2 * math.log(2));
      sigma = np.array(sigma);
      if len(sigma) < 3:
          sigma = np.array((sigma[0], sigma[0]));
      else:
          sigma = sigma[0:2];
      
      g = np.mgrid[-o[0,0]:o[0,1], -o[1,0]:o[1,1]];
      add = ((shape + 1) % 2) / 2.;
      x = g[0,:,:,:] + add[0];
      y = g[1,:,:,:] + add[1];
      
      ker = np.exp(-(x * x / 2. / (sigma[0] * sigma[0]) + y * y / 2. / (sigma[1] * sigma[1])));
      return ker/ker.sum();
      
  elif ftype == 'sphere':
      if radius == None:
          radius = mo;
      radius = np.array(radius);
      
      if len(radius) < 3:
          radius = np.array((radius[0], radius[0]));
      else:
          radius = radius[0:2];
      
      g = np.mgrid[-o[0,0]:o[0,1], -o[1,0]:o[1,1]];
      add = ((shape + 1) % 2) / 2.;
      x = g[0,:,:,:] + add[0];
      y = g[1,:,:,:] + add[1];
      
      ker = 1 - (x * x / 2. / (radius[0] * radius[0]) + y * y / 2. / (radius[1] * radius[1]));
      ker[ker < 0] = 0.;
      return ker / ker.sum();
      
  elif ftype == 'disk':
      if radius == None:
          radius = mo;
      radius = np.array(radius);
      
      if len(radius) < 3:
          radius = np.array((radius[0], radius[0]));
      else:
          radius = radius[0:2];
          
      g = np.mgrid[-o[0,0]:o[0,1], -o[1,0]:o[1,1]];
      add = ((shape + 1) % 2) / 2.;
      x = g[0,:,:,:] + add[0];
      y = g[1,:,:,:] + add[1];
      
      ker = 1 - (x * x / 2. / (radius[0] * radius[0]) + y * y / 2. / (radius[1] * radius[1]));
      ker[ker < 0] = 0.;
      ker[ker > 0] = 1.0;
      return ker / ker.sum();
  
  elif ftype == 'log':  # laplacian of gaussians
      if sigma == None:
          sigma = shape / 4. / math.sqrt(2 * math.log(2));
      
      sigma = np.array(sigma);
      
      if len(sigma) < 3:
          sigma = np.array((sigma[0], sigma[0]));
      else:
          sigma = sigma[0:2];
      
      g = np.mgrid[-o[0,0]:o[0,1], -o[1,0]:o[1,1]];
      add = ((shape + 1) % 2) / 2.;
      x = g[0,:,:,:] + add[0];
      y = g[1,:,:,:] + add[1];
      
      ker = np.exp(-(x * x / 2. / (radius[0] * radius[0]) + y * y / 2. / (radius[1] * radius[1])));
      ker /= ker.sum();
      arg = x * x / math.pow(sigma[0], 4) + y * y/ math.pow(sigma[1],4) - (1/(sigma[0] * sigma[0]) + 1/(sigma[1] * sigma[1]));
      ker = ker * arg;
      return ker - ker.sum()/len(ker);
      
  elif ftype == 'dog':
      if sigma2 == None:
          sigma2 = shape / 2. / math.sqrt(2 * math.log(2));
      sigma2 = np.array(sigma2);
      if len(sigma2) < 3:
          sigma2 = np.array((sigma2[0], sigma2[0]));
      else:
          sigma2 = sigma2[0:2];
      
      if sigma == None:
           sigma = sigma2 / 1.5;
      sigma = np.array(sigma);
      if len(sigma) < 3:
          sigma = np.array((sigma[0], sigma[0]));
      else:
          sigma = sigma[0:2];         
       
      g = np.mgrid[-o[0,0]:o[0,1], -o[1,0]:o[1,1]];
      add = ((shape + 1) % 2) / 2.;
      x = g[0,:,:,:] + add[0];
      y = g[1,:,:,:] + add[1];
      
      ker = np.exp(-(x * x / 2. / (sigma[0] * sigma[0]) + y * y / 2. / (sigma[1] * sigma[1])));
      ker /= ker.sum();
      sub = np.exp(-(x * x / 2. / (sigma2[0] * sigma2[0]) + y * y / 2. / (sigma2[1] * sigma2[1])));
      return ker - sub / sub.sum();
      
  else:
      raise ValueError('Filter type %r not valid!' % ftype);


def filter_kernel_3d(ftype = 'Gaussian', shape = (5,5,5), sigma = None, sigma2 = None, radius = None):
  """Creates a 3d filter kernel of a special type
   
  Arguments
  ---------
  ftype  : str
    Filter type, see :ref:`FilterTypes`.
  shape : array or tuple
    Shape of the filter kernel.
  radius : tuple or float
    Radius of the kernel (if applicable).
  sigma : tuple or float
    Std for the first gaussian (if applicable).
  sigma2 : tuple or float
    Std of a second gaussian (if present).
  
  Returns
  -------
  krenel : array
    Filter kernel.
  """   
  
  ftype = ftype.lower();
  o = se.structure_element_offsets(shape);
  mo = o.min(axis=1);
  shape = np.array(shape);
  
  if ftype == 'mean':  # differnce of gaussians
      return np.ones(shape)/ shape.prod();
      
  elif ftype == 'gaussian':        
      
      if sigma == None:
         sigma = shape / 2. / math.sqrt(2 * math.log(2));
      
      sigma = np.array(sigma);
      
      if len(sigma) < 3:
          sigma = np.array((sigma[0], sigma[0], sigma[0]));
      else:
          sigma = sigma[0:3];
      
      g = np.mgrid[-o[0,0]:o[0,1], -o[1,0]:o[1,1], -o[2,0]:o[2,1]];
      add = ((shape + 1) % 2) / 2.;
      x = g[0,:,:,:] + add[0];
      y = g[1,:,:,:] + add[1];
      z = g[2,:,:,:] + add[2];
      
      ker = np.exp(-(x * x / 2. / (sigma[0] * sigma[0]) + y * y / 2. / (sigma[1] * sigma[1]) + z * z / 2. / (sigma[2] * sigma[2])));
      return ker/ker.sum();
      
  elif ftype == 'sphere':
      
      if radius == None:
          radius = mo;
      radius = np.array(radius);
      
      if len(radius) < 3:
          radius = np.array((radius[0], radius[0], radius[0]));
      else:
          radius = radius[0:3];
      
      g = np.mgrid[-o[0,0]:o[0,1], -o[1,0]:o[1,1], -o[2,0]:o[2,1]];
      add = ((shape + 1) % 2) / 2.;
      x = g[0,:,:,:] + add[0];
      y = g[1,:,:,:] + add[1];
      z = g[2,:,:,:] + add[2];
      
      ker = 1 - (x * x / 2. / (radius[0] * radius[0]) + y * y / 2. / (radius[1] * radius[1]) + z * z / 2. / (radius[2] * radius[2]));
      ker[ker < 0] = 0.;
      return ker / ker.sum();
      
  elif ftype == 'disk':
      
      if radius == None:
          radius = mo;
      radius = np.array(radius);
      
      if len(radius) < 3:
          radius = np.array((radius[0], radius[0], radius[0]));
      else:
          radius = radius[0:3];
      
      g = np.mgrid[-o[0,0]:o[0,1], -o[1,0]:o[1,1], -o[2,0]:o[2,1]];
      add = ((shape + 1) % 2) / 2.;
      x = g[0,:,:,:] + add[0];
      y = g[1,:,:,:] + add[1];
      z = g[2,:,:,:] + add[2];
      
      ker = 1 - (x * x / 2. / (radius[0] * radius[0]) + y * y / 2. / (radius[1] * radius[1]) + z * z / 2. / (radius[2] * radius[2]));
      ker[ker < 0] = 0.;
      ker[ker > 0] = 1.0;
      
      return ker / ker.sum();
      
  elif ftype == 'log':  # laplacian of gaussians
      
      if sigma == None:
          sigma = shape / 4. / math.sqrt(2 * math.log(2));
      
      sigma = np.array(sigma);
      
      if len(sigma) < 3:
          sigma = np.array((sigma[0], sigma[0], sigma[0]));
      else:
          sigma = sigma[0:3];
      
      g = np.mgrid[-o[0,0]:o[0,1], -o[1,0]:o[1,1], -o[2,0]:o[2,1]];
      add = ((shape + 1) % 2) / 2.;
      x = g[0,:,:,:] + add[0];
      y = g[1,:,:,:] + add[1];
      z = g[2,:,:,:] + add[2];
      
      ker = np.exp(-(x * x / 2. / (radius[0] * radius[0]) + y * y / 2. / (radius[1] * radius[1]) + z * z / 2. / (radius[2] * radius[2])));
      ker /= ker.sum();
      arg = x * x / math.pow(sigma[0], 4) + y * y/ math.pow(sigma[1],4) + z * z / math.pow(sigma[2],4) - (1/(sigma[0] * sigma[0]) + 1/(sigma[1] * sigma[1]) + 1 / (sigma[2] * sigma[2]));
      ker = ker * arg;
      return ker - ker.sum()/len(ker);
      
  elif ftype == 'dog':
      
      if sigma2 == None:
          sigma2 = shape / 2. / math.sqrt(2 * math.log(2));
      sigma2 = np.array(sigma2);
      if len(sigma2) < 3:
          sigma2 = np.array((sigma2[0], sigma2[0], sigma2[0]));
      else:
          sigma2 = sigma2[0:3];
      
      if sigma == None:
           sigma = sigma2 / 1.5;
      sigma = np.array(sigma);
      if len(sigma) < 3:
          sigma = np.array((sigma[0], sigma[0], sigma[0]));
      else:
          sigma = sigma[0:3];         
       
      g = np.mgrid[-o[0,0]:o[0,1], -o[1,0]:o[1,1], -o[2,0]:o[2,1]];
      add = ((shape + 1) % 2) / 2.;
      x = g[0,:,:,:] + add[0];
      y = g[1,:,:,:] + add[1];
      z = g[2,:,:,:] + add[2];
      
      ker = np.exp(-(x * x / 2. / (sigma[0] * sigma[0]) + y * y / 2. / (sigma[1] * sigma[1]) + z * z / 2. / (sigma[2] * sigma[2])));
      ker /= ker.sum();
      sub = np.exp(-(x * x / 2. / (sigma2[0] * sigma2[0]) + y * y / 2. / (sigma2[1] * sigma2[1]) + z * z / 2. / (sigma2[2] * sigma2[2])));
      return ker - sub / sub.sum();
      
  else:
      raise ValueError('Filter type %r not valid!' % ftype);

###############################################################################
### Tests
############################################################################### 

def _test():
    """Tests"""
    import ClearMap.ImageProcessing.Filter.FilterKernel as fk
    import ClearMap.Visualization.Plot3d as p3d
    
    k = fk.filter_kernel(ftype='dog', shape=(15,15,15), sigma=None, radius=None, sigma2=None);
    p3d.plot(k)
    
    
    