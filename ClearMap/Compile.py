# -*- coding: utf-8 -*-
"""
Compile
=======

Script to compile all ClearMap code.

Note
----
Cython or C++ coded modules are compiled in a lazy fashion in ClearMap.
To compile all modules at once this module can be used. 
"""
__author__    = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__   = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__copyright__ = 'Copyright (c) 2020 by Christoph Kirst'


###############################################################################
### ClearMap - Cython
###############################################################################

#ImageProcessing
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

#ParallelProcessing
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
