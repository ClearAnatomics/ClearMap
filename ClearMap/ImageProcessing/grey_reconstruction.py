"""
GreyReconstruction
==================

This morphological reconstruction routine was adapted from 
`CellProfiler <https://www.cellprofiler.org>`_.

Authors
-------
Original author: Lee Kamentsky, Massachusetts Institute of Technology
Modifed by Christoph Kirst for ClearMap integration.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


import numpy as np

from skimage.morphology import reconstruction


import ClearMap.ImageProcessing.Filter.StructureElement as se

import ClearMap.Utils.Timer as tmr
import ClearMap.Utils.HierarchicalDict as hdict


###############################################################################
### Grey reconstruction
###############################################################################

def reconstruct(seed, mask=None, method='dilation', selem=None, offset=None):
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
        mask = seed.copy()
    # Changed from original version made for skimage 0.16 which only kept the type of the seed instead of float64,
    # to save memory: scikit-image now keeps float32 as float32 itself (same results, time and memory),
    # and computes integer images in float64. The result is given back in the type of the seed, as the copy did
    return reconstruction(seed, mask, method=method, footprint=selem, offset=offset).astype(seed.dtype, copy=False)


def grey_reconstruct(source, mask=None, sink=None, method=None, shape=3, verbose=False):
    """Calculates the grey reconstruction of the image

    Arguments
    ---------

    source : array
      The source image data.
    method : 'dilation' or 'erosion' or None
      The mehtjod to use, if None return original image.
    shape : in or tuple
      Shape of the strucuturing element for the grey reconstruction.
    verbose : bool
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
        timer = tmr.Timer()
        hdict.pprint(head='Grey reconstruction', method=method, shape=shape)

    if method is None:
        return source

    if sink is None:
        sink = np.empty(source.shape, dtype=source.dtype)

    # background subtraction in each slice
    selem = se.structure_element(form='Disk', shape=shape, ndim=2).astype('uint8')
    for z in range(source.shape[2]):
        #img[:,:,z] = img[:,:,z] - grey_opening(img[:,:,z], structure = structureElement('Disk', (30,30)))
        #img[:,:,z] = img[:,:,z] - morph.grey_opening(img[:,:,z], structure = self.structureELement('Disk', (150,150)))
        sink[:,:,z] = source[:,:,z] - reconstruct(source[:,:,z], mask=mask[:,:,z], method=method, selem=selem)

    if verbose:
        timer.print_elapsed_time('Grey reconstruction')

    return sink



###############################################################################
### Test
###############################################################################

def _test():
    import numpy as np
    import ClearMap.ImageProcessing.grey_reconstruction as gr
    import ClearMap.Visualization.Plot3d as p3d

    x = np.random.rand(*(200,200,10))
    r = gr.grey_reconstruct(x, mask=0.5 * x, method='erosion', shape=3)

    p3d.plot([x,r])
