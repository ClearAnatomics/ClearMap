# -*- coding: utf-8 -*-
"""
Stitching
=========

Stitching module for aligning and stitching data sets with *ClearMap*.

Stitching can be done in two ways:
  * rigidly via the :mod:`~ClearMap.Alignment.Stitching.stitching_rigid` or
  * wobbly via :mod:`~ClearMap.Alignment.Stitching.stitching_wobbly`.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Public License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'https://idisco.info'
__download__  = 'https://www.github.com/ChristophKirst/ClearMap2'

import sys
from ClearMap.Alignment.Stitching import stitching_wobbly, stitching_rigid

# Backward compat alias for layouts (pickled) objects saved under old module name
sys.modules['ClearMap.Alignment.Stitching.StitchingWobbly'] = stitching_wobbly
sys.modules['ClearMap.Alignment.Stitching.StitchingRigid'] = stitching_rigid