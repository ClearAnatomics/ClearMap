"""
Plot3d
======

Plotting routines based on qt.

Note
----
This module is based on the pyqtgraph package.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'

import pyqtgraph as pg
import functools as ft

import ClearMap.Visualization.Qt.DataViewer as dv
import ClearMap.Visualization.Qt.Utils as qtu

############################################################################################################
###  Plotting
############################################################################################################

#TODO: figure / windows handler to update data in existing windows

def plot(source, axis = None, scale = None, title = None, invert_y = True, min_max = None, screen = None):
  """Plot a source as 2d slices.
  
  Arguments
  ---------
  source : list of sources
    The source to plot. If a list is given several synchronized windows are
    generated. If an element in the list is a list of sources those are
    overlayed in different colors in that window.
  axis : int or None
    The axis along which to slice the data.
  scale : tuple of float
    A spatial scale for each axis used for the spatial cursor position.
  title : str or None
    The title of the window.
  invert_y : bool
    If True invert the y axis (as typically done for images).
  min_max : tuple or None
    The minal and maximal values for each source. If None, determine them from 
    the source.
  screen : int or None
    Specifiy on which screen to open the window.
  
  Returns
  -------
  plot : DataViewer
    A data viewer class.
  """
  
  if not isinstance(source, (list, tuple)):
    source = [source];
  return multi_plot(source, axis=axis, scale=scale, title=title, invert_y=invert_y, min_max=min_max, screen=screen);



def multi_plot(sources, axis = None, scale = None, title = None, invert_y = True, min_max = None, arange = True, screen = None):
  """Plot a source as 2d slices.
  
  Arguments
  ---------
  sources : list of sources
    The sources to plot.If an element in the list is a list of sources 
    those are overlayed in different colors in that window.
  axis : int or None
    The axis along which to slice the data.
  scale : tuple of float
    A spatial scale for each axis used for the spatial cursor position.
  title : str or None
    The title of the window.
  invert_y : bool
    If True invert the y axis (as typically done for images).
  min_max : tuple or None
    The minal and maximal values for each source. If None, determine them from 
    the source.
  screen : int or None
    Specifiy on which screen to open the window.
  
  Returns
  -------
  plots : list of DataViewers
    A list of viewer classes.
  """
  
  if not isinstance(title, (tuple, list)):
    title = [title] * len(sources);
  
  dvs = [dv.DataViewer(source=s, axis=axis, scale=scale, title=t,
                       invertY=invert_y, minMax=min_max) for s,t in zip(sources, title)];

  if arange:
    try:      
      geo = qtu.tiled_layout(len(dvs), percent=80, screen=screen);
    
      for d,g in zip(dvs, geo):
        #d.setFixedSize(int(0.95 * g[2]), int(0.9 * g[3]));                  
        d.setGeometry(pg.QtCore.QRect(*g));
    except:
      pass

  for d1,d2 in zip(dvs[:-1], dvs[1:]):
    synchronize(d1, d2);
  
  return dvs;
  

def synchronize(viewer1, viewer2):
  """Synchronize scrolling between two data viewers"""
  # sync dv1 -> dv2
  def sV():
    viewer2.sliceLine.setValue(viewer1.sliceLine.value());
  viewer1.sliceLine.sigPositionChanged.connect(sV);
  for d,button in enumerate(viewer1.axis_buttons):
    button.clicked.connect(ft.partial(viewer2.setSliceAxis, d));
  
  # sync dv2 -> dv1  
  def sV():
    viewer1.sliceLine.setValue(viewer2.sliceLine.value());
  viewer2.sliceLine.sigPositionChanged.connect(sV);
  for d,button in enumerate(viewer2.axis_buttons):
    button.clicked.connect(ft.partial(viewer1.setSliceAxis, d));
  
  viewer1.view.setXLink(viewer2.view);
  viewer1.view.setYLink(viewer2.view);


def set_source(viewer, source):
  """Set the source data in a viewer.

  Arguments
  ---------
  viewer : DataViewer
    The viewer to set a new source for.
  source : Source
    The source to use in the viewer.
    
  Returns
  -------
  viewer : DataViewer
    The viewer.
  """
  viewer.setSource(source);
  return viewer;

############################################################################################################
### Tests
############################################################################################################

def _test():
  import numpy as np
  import ClearMap.Visualization.Qt.Plot3d as p3d
  
  img1 = np.random.rand(*(100,80,30));
  img2 = np.random.rand(*(100,80,30)) > 0.5;
  
  p = p3d.plot([img1,img2])  #analysis:ignore
