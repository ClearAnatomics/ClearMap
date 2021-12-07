# -*- coding: utf-8 -*-
"""
Compile
=======

Script to compile all ClearMap code.

Note
----
Cython or C++ coded modules are compiled in a lazy fashion in ClearMap.
To compile all modules this module can be used. 
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


###############################################################################
### ClearMap - Cython
###############################################################################

# compile ImageProcessing  (CellMap + tubemap)
print('ClearMap.ImageProcessing.Binary.Filling')
import ClearMap.ImageProcessing.Binary.Filling as fil
print('ClearMap.ImageProcessing.Clipping.Clipping')
import ClearMap.ImageProcessing.Clipping.Clipping as clp
print('ClearMap.ImageProcessing.Differentiation.Hessian')
import ClearMap.ImageProcessing.Differentiation.Hessian as hes
print('ClearMap.ImageProcessing.Filter.Rank')
import ClearMap.ImageProcessing.Filter.Rank as rnk
print('ClearMap.ImageProcessing.Thresholding.Thresholding')
import ClearMap.ImageProcessing.Thresholding.Thresholding as thr
print('ClearMap.ImageProcessing.Tracing.Trace')
import ClearMap.ImageProcessing.Tracing.Trace as trc

#%%
# compile ParallelProcessing (CellMap)
print('ClearMap.ParallelProcessing.DataProcessing.ArrayProcessing')
import ClearMap.ParallelProcessing.DataProcessing.ArrayProcessing as ap
print('ClearMap.ParallelProcessing.DataProcessing.ConvolvePointList')
import ClearMap.ParallelProcessing.DataProcessing.ConvolvePointList as cpl
print('ClearMap.ParallelProcessing.DataProcessing.DevolvePointList')
import ClearMap.ParallelProcessing.DataProcessing.DevolvePointList as dpl
print('ClearMap.ParallelProcessing.DataProcessing.MeasurePointList')
import ClearMap.ParallelProcessing.DataProcessing.MeasurePointList as mpl
print('ClearMap.ParallelProcessing.DataProcessing.StatisticsPointList')
import ClearMap.ParallelProcessing.DataProcessing.StatisticsPointList as spl


__all__ = ['fil', 'clp', 'hes', 'rnk', 'thr', 'trc',
           'ap', 'cpl', 'dpl', 'mpl', 'spl']

