# -*- coding: utf-8 -*-
"""
Environment (deprecated)
========================

Initialize a ClearMap environment with all main functionality.

.. deprecated:: 3.1
    ``from ClearMap.Environment import *`` imports the whole package (slow, and it hides
    where names come from). Import the modules you use explicitly instead, e.g.::

        import ClearMap.Alignment.Resampling as res
        import ClearMap.ImageProcessing.Experts.Cells as cells

    The aliases below (``res``, ``cells``, ...) are the conventional ones; ``io`` is the
    deprecated ``ClearMap.IO.IO`` shim, use :mod:`ClearMap.IO.io_ops` and friends instead.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'

import warnings

warnings.warn('ClearMap.Environment is deprecated and will be removed in a future version. '
              'Import the ClearMap modules you use explicitly instead of '
              '"from ClearMap.Environment import *".', FutureWarning, stacklevel=2)

###############################################################################
### Python
###############################################################################

import sys   
import os    
import glob  

import numpy as np                
import matplotlib.pyplot as plt

from importlib import reload

###############################################################################
### ClearMap
###############################################################################

#generic
import ClearMap.Settings as settings

import ClearMap.IO.IO as io
import ClearMap.IO.Workspace as wsp

import ClearMap.Visualization.Plot3d as p3d
import ClearMap.Visualization.Color as col

import ClearMap.Utils.tag_expression as te
import ClearMap.Utils.Timer as tmr

import ClearMap.ParallelProcessing.BlockProcessing as bp
import ClearMap.ParallelProcessing.DataProcessing.ArrayProcessing as ap

#alignment
import ClearMap.Alignment.Annotation as ano     
import ClearMap.Alignment.Resampling as res
import ClearMap.Alignment.Elastix as elx       
import ClearMap.Alignment.Stitching.stitching_rigid as st
import ClearMap.Alignment.Stitching.stitching_wobbly as stw

#image processing
import ClearMap.ImageProcessing.Clipping.Clipping as clp
import ClearMap.ImageProcessing.Filter.Rank as rnk
import ClearMap.ImageProcessing.Filter.StructureElement as se
import ClearMap.ImageProcessing.Differentiation as dif
import ClearMap.ImageProcessing.Skeletonization.Skeletonization as skl
import ClearMap.ImageProcessing.Skeletonization.SkeletonProcessing as skp
import ClearMap.ImageProcessing.machine_learning.vessel_filling.vessel_filling as vf

#analysis
import ClearMap.Analysis.graphs.graph_gt as grp
import ClearMap.Analysis.graphs.graph_processing as gp

import ClearMap.Analysis.Measurements.MeasureExpression as me
import ClearMap.Analysis.Measurements.radius_measurements as mr
import ClearMap.Analysis.Measurements.Voxelization as vox

# experts
import ClearMap.ImageProcessing.Experts.Vasculature as vasc
import ClearMap.ImageProcessing.Experts.Cells as cells

###############################################################################
### All
###############################################################################

__all__ = ['sys', 'os', 'glob', 'np', 'plt', 'reload',
           'settings', 'io', 'wsp',
           'p3d', 'col', 'te', 'tmr',  'bp', 'ap',
           'ano', 'res', 'elx', 'st', 'stw',
           'clp', 'rnk', 'se', 'dif', 'skl', 'skp', 'vf',
           'grp', 'gp', 'me', 'mr', 'vox',
           'vasc', 'cells']
