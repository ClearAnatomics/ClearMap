# -*- coding: utf-8 -*-
"""
``ClearMap`` is a toolbox for the analysis and registration of volumetric data.
It has been designed to analyze O(TB) 3D datasets obtained
via light sheet microscopy from iDISCO+ cleared tissue samples 
immunolabeled for proteins. 

*ClearMap* includes
* non-rigid wobbly stitching,
* image registration to a 3D annotated references
(e.g. the Allen Brain Institute Atlases),
* a toolbox for large volumetric image processing O(TB),
* object and cell detection,
* vasculature detection and graph construction,
* statistical analysis
  
``ClearMap`` has been written for mapping immediate early genes [Renier2016]_
as well as vasculature networks of whole mouse brains [Kirst2020]_.

This tools may also be useful for data obtained with other types
of microscopes, types of markers, clearing techniques, as well as other 
species, organs, or samples.

``ClearMap`` is written in `Python 3 <https://docs.python.org/3/>`_ and is
designed to take advantage of
parallel processing capabilities of modern workstations. We hope the open 
structure of the code will enable many new modules to be added to ClearMap to 
broaden the range of applications to different types of biological objects or
structures.

More information and downloads for *ClearMap* can be found in our
`repository <https://www.github.com/ChristophKirst/ClearMap2>`_.

"""
from importlib.metadata import version, PackageNotFoundError

__title__ = 'ClearMap'
__author__ = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__ = 'https://idisco.info'
__download__ = 'https://github.com/ClearAnatomics/ClearMap'
try:
    __version__ = version("ClearMap")
except PackageNotFoundError:
    __version__ = '3.0.0'
