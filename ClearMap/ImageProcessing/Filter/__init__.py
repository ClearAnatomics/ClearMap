# -*- coding: utf-8 -*-
"""
Rank
====

This sub-package provides various volumetric filter kernels and structure elments

A set of linear filters can be applied to the data using 
:mod:`~ClearMap.ImageProcessing.Filter.LinearFilter`. 

Because its utility for cell detection the difference of Gaussians filter
is implemented directly in :mod:`~ClearMap.ImageProcessing.Filter.DoGFilter`.

The fitler kernels defined in :mod:`~ClearMap.ImageProcessing.Filter.FilterKernel` 
can be used in combination with the :mod:`~ClearMap.ImageProcessing.Convolution` 
module.

Structured elements defined in 
:mod:`~ClearMap.ImageProcessing.Filter.StructureElements` can be used in 
combination with various morphological operations, e.g. used in the
:mod:~ClearMap.ImageProcessing.BackgroundRemoval` module.

"""  
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'