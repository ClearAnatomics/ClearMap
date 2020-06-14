# -*- coding: utf-8 -*-
"""
Alignment
=========

This sub-package provides an interface to alignment tools in order to register
cleared samples to atlases or reference samples.

Supported functionality:

    * stitching of tiles in a rigid or non-rigid wobbly fashion in the
      :mod:`~ClearMap.Alignment.Stitching` module.
    * resampling and reorientation of large volumetric images in the 
      :mod:`~ClearMap.Alignment.Resampling` module.
    * registering volumetric data onto references via 
      `Elastix <http://elastix.isi.uu.nl/>`_ in the 
      :mod:`~ClearMap.Alignment.Elastix` module.

All steps are demonstrated in the the :ref:`TubeMap tutorial </TubeMap.ipynb>`.

The main routines for resampling are
:func:`~ClearMap.Alignment.Resampling.resample`.

Main routines for elastix registration are 
:func:`~ClearMap.Alignment.Elastix.align` and 
:func:`~ClearMap.Alignment.Elastix.transform`.
""" 
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'
