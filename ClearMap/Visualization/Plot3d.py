# -*- coding: utf-8 -*-
"""
Plot3d
======

Plotting routines for 3d display of data.

Note
----
This module collects functions from various backends for use with ClearMap.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'
 

###############################################################################
### Vispy plotting
###############################################################################

from ClearMap.Visualization.Vispy.Plot3d import *           #analysis:ignore
from ClearMap.Visualization.Vispy.PlotGraph3d import *      #analysis:ignore


###############################################################################
### Qt plotting
###############################################################################

from ClearMap.Visualization.Qt.Plot3d import plot, bring_to_front, close, arrange_plots  #analysis:ignore


###############################################################################
### Matplotlib plotting
###############################################################################

from ClearMap.Visualization.Matplotlib.PlotUtils import *   #analysis:ignore
