# -*- coding: utf-8 -*-
"""This sub-package provides routines for volumetric image processing in parallel

This part of the *ClearMap* toolbox is designed in a modular way to allow for 
fast and flexible extension and addition of specific image processing 
algorithms.

The toolbox part consists of two parts:
    * `Volumetric Image Processing`_
    * `Parallel Image Processing`_


Volumetric Image Processing
---------------------------

The image processing routines provided in the standard package are listed below

======================================================= ===========================================================
Module                                                  Descrition
======================================================= ===========================================================
:mod:`~ClearMap.ImageProcessing.BackgroundRemoval`      Background estimation and removal via morphological opening
:mod:`~ClearMap.ImageProcessing.IlluminationCorrection` Correction of vignetting and other illumination errors
:mod:`~ClearMap.ImageProcessing.GreyReconstruction`     Reconstruction of images
:mod:`~ClearMap.ImageProcessing.Filter`                 Filtering of images via a large set of filter kernels
:mod:`~ClearMap.ImageProcessing.MaximaDetection`        Detection of maxima and h-max transforms
:mod:`~ClearMap.ImageProcessing.SpotDetection`          Detection of local peaks / spots / nuclei
:mod:`~ClearMap.ImageProcessing.CellDetection`          Detection of cells
:mod:`~ClearMap.ImageProcessing.CellSizeDetection`      Detection of cell shapes and volumes via e.g. watershed
:mod:`~ClearMap.ImageProcessing.IlastikClassification`  Classification of voxels via interface to `Ilastik <http://ilastik.org/>`_
======================================================= ===========================================================

While some of these modules provide basic volumetric image processing 
functionality some routines combine those functions to provide predefined
higher level cell detection, cell size and intensity measurements.

The higher level routines are optimized for iDISCO+ cleared mouse brain samples
stained for c-Fos expression. Other data sets might require a redesign of these
higher level functions.


Parallel Image Processing
-------------------------

For large volumetric image data sets from e.g. light sheet microscopy 
parallel processing is essential to speed up calculations.

In this toolbox the image processing is parallelized via splitting a volumetric
image stack into several sub-stacks, typically in z-direction. Because most of 
the image processing steps are non-local sub-stacks are created with overlaps 
and the results rejoined accordingly to minimize boundary effects.

Parallel processing is handled via the 
:mod:`~ClearMap.ImageProcessing.StackProcessing` module.


External Packages
-----------------

The :mod:`~ClearMap.ImageProcessing` module makes use of external image
processing packages including:

    * `Open Cv2 <http://opencv.org/>`_
    * `Scipy <http://www.scipy.org/>`_
    * `Scikit-Image <http://scikit-image.org/docs/dev/api/skimage.html>`_
    * `Ilastik <http://ilastik.org/>`_
    
Routines form these packages were freely chosen to optimize for speed and 
memory consumption

"""
__author__    = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__   = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__copyright__ = 'Copyright (c) 2017 by Christoph Kirst, The Rockefeller University, New York City'