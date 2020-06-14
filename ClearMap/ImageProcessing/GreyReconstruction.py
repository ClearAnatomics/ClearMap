"""
GreyReconstruction
==================

This morphological reconstruction routine was adapted from 
`CellProfiler <http://www.cellprofiler.org>`_.

Authors
-------
Original author: Lee Kamentsky, Massachusetts Institute of Technology
Modifed by Christoph Kirst for ClearMap integration.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


import numpy as np

from skimage.filters._rank_order import rank_order


import ClearMap.ImageProcessing.Filter.StructureElement as se

import ClearMap.Utils.Timer as tmr
import ClearMap.Utils.HierarchicalDict as hdict


###############################################################################
### Grey reconstruction
###############################################################################

def reconstruct(seed, mask = None, method = 'dilation', selem = None, offset = None):
  """Performs a morphological reconstruction of an image.
  
  Arguments
  ---------
  seed : array
    Seed image to be dilated or eroded.
  mask : array
    Maximum (dilation) / minimum (erosion) allowed
  method : {'dilation'|'erosion'}
    The method to use.
  selem : array
    Structuring element.
  offset : array or None
    The offset of the structuring element, None is centered.

  Returns
  -------
  reconstructed : array
    Result of morphological reconstruction.
  
  Note
  ----
  Reconstruction uses a seed image, which specifies the values
  to dilate and a mask image that gives the maximum allowed dilated value at
  each pixel.
  
  The algorithm is taken from [1]_. Applications for greyscale 
  reconstruction are discussed in [2]_ and [3]_.
  
  Effectively operates on 2d images.
  
  Reference:
  
  .. [1] Robinson, "Efficient morphological reconstruction: a downhill
         filter", Pattern Recognition Letters 25 (2004) 1759-1767.
  .. [2] Vincent, L., "Morphological Grayscale Reconstruction in Image
         Analysis: Applications and Efficient Algorithms", IEEE Transactions
         on Image Processing (1993)
  .. [3] Soille, P., "Morphological Image Analysis: Principles and
         Applications", Chapter 6, 2nd edition (2003), ISBN 3540429883.
  """
  if mask is None:
    mask = seed.copy();
  
  if seed.shape != mask.shape:
    raise ValueError('Seed shape % and mask shape %r do not match' % (seed.shape, mask.shape))
  
  if method == 'dilation' and np.any(seed > mask):
    raise ValueError("Intensity of seed image must be less than that "
                     "of the mask image for reconstruction by dilation.")
  elif method == 'erosion' and np.any(seed < mask):
    raise ValueError("Intensity of seed image must be greater than that "
                     "of the mask image for reconstruction by erosion.")
  try:
      from skimage.morphology._greyreconstruct import reconstruction_loop
  except ImportError:
      raise ImportError("_greyreconstruct extension not available.")

  if selem is None:
      selem = np.ones([3] * seed.ndim, dtype=bool)
  else:
      selem = selem.copy()

  if offset is None:
    if not all([d % 2 == 1 for d in selem.shape]):
        ValueError("Footprint dimensions must all be odd")
    offset = np.array([d // 2 for d in selem.shape])
  
  # Cross out the center of the selem
  selem[tuple(slice(d, d + 1) for d in offset)] = False

  # Make padding for edges of reconstructed image so we can ignore boundaries
  padding = (np.array(selem.shape) / 2).astype(int)
  dims = np.zeros(seed.ndim + 1, dtype=int)
  dims[1:] = np.array(seed.shape) + 2 * padding
  dims[0] = 2
  inside_slices = tuple(slice(p, -p) for p in padding)
  # Set padded region to minimum image intensity and mask along first axis so
  # we can interleave image and mask pixels when sorting.
  if method == 'dilation':
      pad_value = np.min(seed)
  elif method == 'erosion':
      pad_value = np.max(seed)
  images = np.ones(dims, dtype = seed.dtype) * pad_value
  images[(0,) + inside_slices] = seed
  images[(1,) + inside_slices] = mask

  # Create a list of strides across the array to get the neighbors within
  # a flattened array
  value_stride = np.array(images.strides[1:]) / images.dtype.itemsize
  image_stride = images.strides[0] // images.dtype.itemsize
  selem_mgrid = np.mgrid[[slice(-o, d - o)
                          for d, o in zip(selem.shape, offset)]]
  selem_offsets = selem_mgrid[:, selem].transpose()
  nb_strides = np.array([np.sum(value_stride * selem_offset)
                         for selem_offset in selem_offsets], np.int32)

  images = images.flatten()

  # Erosion goes smallest to largest; dilation goes largest to smallest.
  index_sorted = np.argsort(images).astype(np.int32)
  if method == 'dilation':
      index_sorted = index_sorted[::-1]

  # Make a linked list of pixels sorted by value. -1 is the list terminator.
  prev = -np.ones(len(images), np.int32)
  next = -np.ones(len(images), np.int32)
  prev[index_sorted[1:]] = index_sorted[:-1]
  next[index_sorted[:-1]] = index_sorted[1:]

  # Cython inner-loop compares the rank of pixel values.
  if method == 'dilation':
      value_rank, value_map = rank_order(images)
  elif method == 'erosion':
      value_rank, value_map = rank_order(-images)
      value_map = -value_map

  start = index_sorted[0]
  reconstruction_loop(value_rank, prev, next, nb_strides, start, image_stride)

  # Reshape reconstructed image to original image shape and remove padding.
  rec_img = value_map[value_rank[:image_stride]]
  rec_img.shape = np.array(seed.shape) + 2 * padding
  
  return rec_img[inside_slices]


def grey_reconstruct(source, mask = None, sink = None, method = None, shape = 3, verbose = False):
  """Calculates the grey reconstruction of the image 
  
  Arguments
  ---------
  
  source : array
    The source image data.
  method : 'dilation' or 'erosion' or None
    The mehtjod to use, if None return original image.
  shape : in or tuple
    Shape of the strucuturing element for the grey reconstruction.
  verbose : boo;
    If True, print progress info.

  Returns
  -------
  reconstructed: array
    Grey reconstructed image.
    
  Note
  ----
  The reconstruction is done slice by slice along the z-axis.
  """
  
  if verbose:
    timer = tmr.Timer();
    hdict.pprint(head='Grey reconstruction', method=method, shape=shape)
  
  if method is None:
    return source;
  
  if sink is None:
    sink = np.empty(source.shape, dtype=source.dtype);
  
  # background subtraction in each slice
  selem = se.structure_element(form='Disk', shape=shape, ndim=2).astype('uint8');
  for z in range(source.shape[2]):
    #img[:,:,z] = img[:,:,z] - grey_opening(img[:,:,z], structure = structureElement('Disk', (30,30)));
    #img[:,:,z] = img[:,:,z] - morph.grey_opening(img[:,:,z], structure = self.structureELement('Disk', (150,150)));
    sink[:,:,z] = source[:,:,z] - reconstruct(source[:,:,z], mask=mask[:,:,z], method=method, selem=selem)
  
  if verbose:
    timer.print_elapsed_time('Grey reconstruction');
  
  return sink 



###############################################################################
### Test
###############################################################################

def _test():
  import numpy as np
  import ClearMap.ImageProcessing.GreyReconstruction as gr
  import ClearMap.Visualization.Plot3d as p3d

  x = np.random.rand(*(200,200,10));
  r = gr.grey_reconstruct(x, mask=0.5 * x, method='erosion', shape=3);
  
  p3d.plot([x,r])
