# -*- coding: utf-8 -*-
"""
Environment
===========

Initialize a ClearMap environment with all main functionality.

Note
----
To initialize the main functions in a ClearMap script use:
>>> from ClearMap.Environment import *
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'

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

import ClearMap.Tests.Files as tfs 

import ClearMap.Visualization.Plot3d as p3d
import ClearMap.Visualization.Color as col

import ClearMap.Utils.TagExpression as te
import ClearMap.Utils.Timer as tmr

import ClearMap.ParallelProcessing.BlockProcessing as bp
import ClearMap.ParallelProcessing.DataProcessing.ArrayProcessing as ap

#alignment
import ClearMap.Alignment.Annotation as ano     
import ClearMap.Alignment.Resampling as res
import ClearMap.Alignment.Elastix as elx       
import ClearMap.Alignment.Stitching.StitchingRigid as st
import ClearMap.Alignment.Stitching.StitchingWobbly as stw

#image processing
import ClearMap.ImageProcessing.Clipping.Clipping as clp
import ClearMap.ImageProcessing.Filter.Rank as rnk
import ClearMap.ImageProcessing.Filter.StructureElement as se
import ClearMap.ImageProcessing.Differentiation as dif
import ClearMap.ImageProcessing.Skeletonization.Skeletonization as skl
import ClearMap.ImageProcessing.Skeletonization.SkeletonProcessing as skp
import ClearMap.ImageProcessing.MachineLearning.VesselFilling.VesselFilling as vf

#analysis
import ClearMap.Analysis.Graphs.GraphGt as grp
import ClearMap.Analysis.Graphs.GraphProcessing as gp

import ClearMap.Analysis.Measurements.MeasureExpression as me
import ClearMap.Analysis.Measurements.MeasureRadius as mr
import ClearMap.Analysis.Measurements.Voxelization as vox

# experts
import ClearMap.ImageProcessing.Experts.Vasculature as vasc
import ClearMap.ImageProcessing.Experts.Cells as cells

###############################################################################
### All
###############################################################################

__all__ = ['sys', 'os', 'glob', 'np', 'plt', 'reload',
           'settings', 'io', 'wsp', 'tfs', 
           'p3d', 'col', 'te', 'tmr',  'bp', 'ap',
           'ano', 'res', 'elx', 'st', 'stw',
           'clp', 'rnk', 'se', 'dif', 'skl', 'skp', 'vf',
           'grp', 'gp', 'me', 'mr', 'vox',
           'vasc', 'cells'];
