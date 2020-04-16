# -*- coding: utf-8 -*-
"""
Plot3d Module
-------------

Plotting routines for 3d display of data.

Note
----
This module collects functions from various backends.
"""
__author__    = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__   = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__copyright__ = 'Copyright (c) 2017 by Christoph Kirst, The Rockefeller University, New York City'
 

###############################################################################
### vispy plotting
###############################################################################

from ClearMap.Visualization.Vispy.Plot3d import *           #analysis:ignore
from ClearMap.Visualization.Vispy.PlotGraph3d import *      #analysis:ignore


###############################################################################
### Qt plotting
###############################################################################

from ClearMap.Visualization.Qt.Plot3d import plot           #analysis:ignore


###############################################################################
### Matplotlib plotting
###############################################################################

from ClearMap.Visualization.Matplotlib.PlotUtils import *   #analysis:ignore