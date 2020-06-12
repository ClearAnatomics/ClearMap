# -*- coding: utf-8 -*-
"""
*ClearMap* is a toolbox for the analysis and registration of volumetric data.

*ClearMap* has been designed to analyze O(TB) 3d datasets obtained 
via light sheet microscopy from iDISCO+ cleared tissue samples 
immunolabeled for proteins. 

*ClearMap* includes 
  * non-ridgid wobbly stitching,
  * image registration to a 3D annotated references 
    (e.g. the Allen Brain Institute Atlases), 
  * a toolbox for large volumetric image processing O(TB), 
  * object and cell detection,
  * vasculature detection and graph construction,
  * statistical analysis 
  
*ClearMap* has been written for mapping immediate early genes [Renier2016]_
as well as vasculature networks of whole mouse brains [Kirst2020]_.

*ClearMap* tools may also be useful for data obtained with other types
of microscopes, types of markers, clearing techniques, as well as other 
species, organs, or samples.

*ClearMap* is written in Python 3 and is designed to take advantage of
parallel processing capabilities of modern workstations. We hope the open 
structure of the code will enable many new modules to be added to ClearMap to 
broaden the range of applications to different types of biological objects or
structures.

More informatoin and downloads for *ClearMap* can be found in our 
`repository <http://www.github.com/ChristophKirst/ClearMap2>`_.


References
----------
.. [Renier2016] `Mapping of brain activity by automated volume analysis of immediate early genes.'Renier* N, Adams* EL, Kirst* C, Wu* Z, et al. Cell. 2016 165(7):1789-802 <https://doi.org/10.1016/j.cell.2016.05.007>`_

.. [Kirst2020] `Mapping the Fine-Scale Organization and Plasticity of the Brain Vasculature. Kirst, C., Skriabine, S., Vieites-Prado, A., Topilko, T., Bertin, P., Gerschenfeld, G., Verny, F., Topilko, P., Michalski, N., Tessier-Lavigne, M. and Renier, N., Cell, 180(4):780-795 <https://doi.org/10.1016/j.cell.2020.01.028>`_

"""
__title__     = 'ClearMap'
__version__   = '2.0.0'
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'

