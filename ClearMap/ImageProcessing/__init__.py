# -*- coding: utf-8 -*-
"""
ImageProcessing
===============

This sub-package provides routines for volumetric image processing in parallel

This part of the *ClearMap* toolbox is designed in a modular way to allow for 
fast and flexible extension and addition of specific image processing 
algorithms.

While most psrts contain lower level image processing methods, more specialized
routines that combine the basic routines can be found in 
:mod"`ClearMap.ImageProcessing.Experts`.


Parallel Image Processing
-------------------------

For large volumetric image data sets from e.g. light sheet microscopy 
parallel processing is essential to speed up calculations.

In this toolbox the image processing is parallelized in two different
ways depending on how the algorithms parallelzie.

  * Most routines allow to split volumetric data into several blocks, 
    typically in z-direction. Because  image processing steps are often
    non-local but still only incorporate a certan range, blocks are created 
    with overlaps and the results rejoined accordingly to minimize boundary 
    effects. 
    
    Parallel block processing is handled via the 
    :mod:`~ClearMap.ParallelProcessing` module.
    
  * Full parallelization using memory mapping and parallel implementations via
    cython. The :mod:~ClearMap.ImageProcessing.Skeletonization` algorithm is
    an example block processing is not feasable in this situation.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'