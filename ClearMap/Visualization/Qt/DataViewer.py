"""
DataViewer
==========

Data viewer showing 3d data as 2d slices.

Usage
-----

.. image:: ../Static/DataViewer.jpg

Note
----
This viewer is based on the pyqtgraph package.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


import numpy as np
import functools as ft

#from PyQt5 import QtCore
#from PyQt5.QtWidgets import *

import pyqtgraph as pg
pg.CONFIG_OPTIONS['useOpenGL'] = False  # set to False if trouble seeing data.

if not pg.QAPP: 
  pg.mkQApp()

import ClearMap.IO.IO as io


############################################################################################################
###  Lookup tables
############################################################################################################

class LUTItem(pg.HistogramLUTItem):
  """Lookup table item for the DataViewer"""
  
  def __init__(self, *args, **kargs):
    pg.HistogramLUTItem.__init__(self, *args, **kargs);
    self.vb.setMaximumWidth(15)
    self.vb.setMinimumWidth(10)
  
  def imageChanged(self, autoLevel=False, autoRange=False):
    #mn,mx = self.quickMinMax(targetSize = 500);
    #self.plot.setData([mn, mx], [1,1]);
    if autoLevel:
      mn,mx = self.quickMinMax(targetSize = 500);
      self.region.setRegion([mn, mx])
   
  def quickMinMax(self, targetSize=1e3):
    """
    Estimate the min/max values of the image data by subsampling.
    """
    data = self.imageItem().image
    while data.size > targetSize:
      ax = np.argmax(data.shape)
      sl = [slice(None)] * data.ndim
      sl[ax] = slice(None, None, 2)
      data = data[tuple(sl)]
    return np.nanmin(data), np.nanmax(data)


class LUTWidget(pg.GraphicsView):
  """Lookup table widget for the DataViewer"""
    
  def __init__(self, parent = None,  *args, **kargs):
    background = kargs.get('background', 'default')
    pg.GraphicsView.__init__(self, parent=parent, useOpenGL=True, background=background)
    self.item = LUTItem(*args, **kargs)
    self.setCentralItem(self.item)
    #self.setSizePolicy(pg.QtGui.QSizePolicy.Minimum, pg.QtGui.QSizePolicy.Expanding)
    self.setSizePolicy(pg.QtGui.QSizePolicy.Preferred, pg.QtGui.QSizePolicy.Expanding)
    self.setMinimumWidth(120)
    
  def sizeHint(self):
    return pg.QtCore.QSize(120, 200)
    
  def __getattr__(self, attr):
    return getattr(self.item, attr)


class LUT(pg.QtGui.QWidget):
  def __init__(self, image = None, color = 'red', percentiles = [[-100,0,50,75],[50,75,100,150]], parent = None, *args):
    pg.QtGui.QWidget.__init__(self, parent, *args)
   
    self.layout = pg.QtGui.QGridLayout(self);
    self.layout.setSpacing(0);
    self.layout.setMargin(0);
    
    self.lut = LUTWidget(parent = parent, image = image);
    self.layout.addWidget(self.lut, 0, 0, 1, 1) ; 
  
    self.range_layout = pg.QtGui.QGridLayout()
    self.range_buttons = [];
    pre = ['%d', '%d'];
    for r in range(2):
      range_buttons_m = [];
      for i,p in enumerate(percentiles[r]):
        button = pg.QtGui.QPushButton(pre[r] % (p));
        button.setMaximumWidth(30);
        font = button.font();
        font.setPointSize(6);
        button.setFont(font);
        self.range_layout.addWidget(button,r,i);
        range_buttons_m.append(button);
      self.range_buttons.append(range_buttons_m);
    
    self.layout.addLayout(self.range_layout,1,0,1,1);
    
    self.precentiles = percentiles;
    self.percentile_id = [2,2];
        
    for m,ab in enumerate(self.range_buttons):
      for p, abm in enumerate(ab):
        abm.clicked.connect(ft.partial(self.updateRegionRange, m, p));  
        
    #default gradient
    if color in pg.graphicsItems.GradientEditorItem.Gradients.keys():
      self.lut.gradient.loadPreset(color);
    else:
      self.lut.gradient.getTick(0).color = pg.QtGui.QColor(0,0,0,0);
      self.lut.gradient.getTick(1).color = pg.QtGui.QColor(color);
      self.lut.gradient.updateGradient();
  
  
  def updateRegionRange(self, m,p):
    self.percentile_id[m] = p;
    pmin = self.precentiles[0][self.percentile_id[0]];
    pmax = self.precentiles[1][self.percentile_id[1]];
    self.updateRegionPercentile(pmin, pmax);  
  
  def updateRegionPercentile(self,pmin,pmax):
    iitem = self.lut.imageItem();
    if iitem is not None:
      pmax1 = max(0,min(pmax, 100))
      
      if pmin < 0:
        pmin1 = min(-pmin, 100);
      else:
        pmin1 = min(pmin, 100);
      
      if pmax1 == 0:
        pmax1 = 1; pmax = 1;
      if pmin1 == 0:
        pmin1 = 1; pmin = 1;
      
      r = [float(pmin)/pmin1, float(pmax)/pmax1] * self.quickPercentile(iitem.image, [pmin1, pmax1])
      self.lut.region.setRegion(r);
      
  def quickPercentile(self, data, percentiles, targetSize=1e3):
    while data.size > targetSize:
      ax = np.argmax(data.shape)
      sl = [slice(None)] * data.ndim
      sl[ax] = slice(None, None, 2);
      sl = tuple(sl);
      data = data[sl]
    return np.nanpercentile(data, percentiles);

  
############################################################################################################
###  DataViewer
############################################################################################################


class DataViewer(pg.QtGui.QWidget):
  def __init__(self, source, axis = None, scale = None, title = None, invertY = False, minMax = None, screen = None, parent = None, *args):
    ### Images soures
    self.initializeSources(source, axis = axis, scale = scale)
    #print('init')
    
    ### Gui Construction
    pg.QtGui.QWidget.__init__(self, parent, *args);
    #print('gui')    
                         
    if title is None:
      if isinstance(source, str):
        title = source;
      elif isinstance(source, io.src.Source):
        title = source.location;
      if title is None:
        title = 'DataViewer';                             
    self.setWindowTitle(title);
    self.resize(1600,1200)      
    #print('title')
    
    
    self.layout = pg.QtGui.QGridLayout(self);
    self.layout.setContentsMargins(0,0,0,0)     
    #print('layout')
    
    # image pane
    self.view = pg.ViewBox();
    self.view.setAspectLocked(True);
    self.view.invertY(invertY)    
    
    self.graphicsView = pg.GraphicsView()
    self.graphicsView.setObjectName("GraphicsView")
    self.graphicsView.setCentralItem(self.view)
    
    splitter = pg.QtGui.QSplitter();
    splitter.setOrientation(pg.QtCore.Qt.Horizontal)
    splitter.setSizes([self.width() - 10, 10]);
    self.layout.addWidget(splitter);
    
    image_splitter = pg.QtGui.QSplitter();
    image_splitter.setOrientation(pg.QtCore.Qt.Vertical)
    image_splitter.setSizePolicy(pg.QtGui.QSizePolicy.Expanding, pg.QtGui.QSizePolicy.Expanding)
    splitter.addWidget(image_splitter);
    #print('image')
    
    #  Image plots
    image_options = dict(clipToView = True, autoDownsample = True, autoLevels = False, useOpenGL = None);
    self.image_items = [pg.ImageItem(s[self.source_slice[:s.ndim]], **image_options) for s in self.sources];
    for i in self.image_items:
      i.setRect(pg.QtCore.QRect(0, 0, self.source_range_x, self.source_range_y))
      i.setCompositionMode(pg.QtGui.QPainter.CompositionMode_Plus);
      self.view.addItem(i);
    self.view.setXRange(0, self.source_range_x);
    self.view.setYRange(0, self.source_range_y);
    #print('plots')
    
    # Slice Selector
    self.slicePlot = pg.PlotWidget()
    sizePolicy = pg.QtGui.QSizePolicy(pg.QtGui.QSizePolicy.Preferred, pg.QtGui.QSizePolicy.Preferred)
    sizePolicy.setHorizontalStretch(0)
    sizePolicy.setVerticalStretch(0)
    sizePolicy.setHeightForWidth(self.slicePlot.sizePolicy().hasHeightForWidth())
    self.slicePlot.setSizePolicy(sizePolicy)
    self.slicePlot.setMinimumSize(pg.QtCore.QSize(0, 40))
    self.slicePlot.setObjectName("roiPlot");    
    #self.sliceCurve = self.slicePlot.plot()
    
    self.sliceLine = pg.InfiniteLine(0, movable=True)
    self.sliceLine.setPen((255, 255, 255, 200))
    self.sliceLine.setZValue(1)
    self.slicePlot.addItem(self.sliceLine)
    self.slicePlot.hideAxis('left')
    
    self.updateSlicer();
    
    self.sliceLine.sigPositionChanged.connect(self.updateSlice)
    #print('slice')
    
    # Axis Tools
    axis_tools_layout = pg.QtGui.QGridLayout()
    self.axis_buttons = [];
    axesnames = ['x', 'y', 'z'];
    for d in range(3):
      button = pg.QtGui.QRadioButton(axesnames[d]);
      button.setMaximumWidth(50);
      axis_tools_layout.addWidget(button,0,d);
      button.clicked.connect(ft.partial(self.setSliceAxis, d));
      self.axis_buttons.append(button);
    self.axis_buttons[self.source_axis].setChecked(True);
    axis_tools_widget = pg.QtGui.QWidget();
    axis_tools_widget.setLayout(axis_tools_layout);
    #print('axis')
    
    # coordinate label
    self.source_pointer = [0,0,0];
    self.source_label = pg.QtGui.QLabel("");
    axis_tools_layout.addWidget(self.source_label,0,3);
    
    self.graphicsView.scene().sigMouseMoved.connect(self.updateLabelFromMouseMove);
    #print('coords')
    
    #compose the image viewer
    image_splitter.addWidget(self.graphicsView);
    image_splitter.addWidget(self.slicePlot)
    image_splitter.addWidget(axis_tools_widget); 
    image_splitter.setSizes([self.height()-35-20, 35, 20])
    #print('viewer')
    
    # lut widgets
    if self.nsources == 1:
      cols = ['flame'];
    elif self.nsources == 2:
      cols = ['purple', 'green'];
    else:
      cols = np.array(['white', 'green','red', 'blue', 'purple'] * self.nsources)[:self.nsources];
    
    self.luts = [LUT(image = i, color = c) for i,c in zip(self.image_items, cols)];
    
    lut_layout = pg.QtGui.QGridLayout();

    lut_layout.setContentsMargins(0,0,0,0);
    for d,l in enumerate(self.luts):
      lut_layout.addWidget(l,0,d);  
    lut_widget = pg.QtGui.QWidget();
    lut_widget.setLayout(lut_layout);
    lut_widget.setContentsMargins(0,0,0,0);
    #lut_widget.setSizePolicy(pg.QtGui.QSizePolicy.Maximum, pg.QtGui.QSizePolicy.Expanding)
    lut_widget.setSizePolicy(pg.QtGui.QSizePolicy.Preferred, pg.QtGui.QSizePolicy.Expanding)
    splitter.addWidget(lut_widget);
    
    splitter.setStretchFactor(0, 1);
    splitter.setStretchFactor(1, 0);
    
    #self.source_levelMin = [];
    #self.source_levelMax = [];
    #for i,s in enumerate(self.sources):
    #  lmin, lmax = list(map(float, self.quickMinMax(s[self.source_slice])));
    #  self.levelMin.append(lmin);
    #  self.levelMax.append(lmax); 
    #print('lut')
    
    # update scale
    for l in self.luts:
      l.range_buttons[1][2].click();
    if minMax is not None:
      self.setMinMax(minMax);
     
    self.show();

  
  def initializeSources(self, source, scale = None, axis = None, update = True):
    #initialize sources and axis settings  
    if isinstance(source, tuple):
      source = list(source);
    if not isinstance(source, list):
      source = [source];
    self.nsources = len(source);
    self.sources  = [io.as_source(s) for s in source];
    
    # avoid bools
    #for i,s in enumerate(self.sources):
    #  if s.dtype == bool:
    #    self.sources[i] = s.view('uint8');
     
    # # ensure 3d images 
    # for i,s in enumerate(self.sources):
    #   if s.ndim == 2:
    #     s = s.view();
    #     s.shape = s.shape + (1,);
    #     self.sources[i] = s;
    #   if s.ndim != 3:
    #     raise RuntimeError('Sources dont have dimensions 2 or 3 but %d in source %d!' % (s.ndim, i));
    
    # source shapes
    self.source_shape = self.shape3d(self.sources[0].shape);  
    for s in self.sources:
      if s.ndim > 3:
        raise RuntimeError('Source has %d > 3 dimensions: %r!' % (s.ndim, s));
      if self.shape3d(s.shape) != self.source_shape:
        raise RuntimeError('Sources shape %r vs %r in source %r!' % (self.source_shape, s.shape, s));
        
    self.source_shape2 = np.array(np.array(self.source_shape, dtype=float)/2, dtype=int);
    # for i,s in enumerate(self.sources):
    #  
    
    # slicing
    if axis is None:
      axis = 2;
    self.source_axis = axis;
    self.source_index = self.source_shape2;
    
    # scaling
    if scale is None:
      scale = np.ones(3);
    else:
      scale = np.array(scale);
    scale = np.hstack([scale, [1]*3])[:3];
    self.source_scale = scale;
    #print(self.source_shape, self.source_scale)

    self.updateSourceRange();
    self.updateSourceSlice();
  
  
  def setSource(self, source, index = all):
    #initialize sources and axis settings

    if index is all:  
      if isinstance(source, tuple):
        source = list(source);
      if not isinstance(source, list):
        source = [source];
      if self.nsources != len(source):
        raise RuntimeError('Number of sources does not match!');
      source  = [io.as_source(s) for s in source];
      index = range(self.nsources);
    else:
      s = self.sources;
      s[index] = io.as_source(source);
      source = s;
      index = [index];
    
    for i in index:
      s = source[i];
      if s.shape != self.source_shape:
        raise RuntimeError('Shape of sources does not match!');
      
      #if s.dtype == bool:
      #  self.sources[i] = s.view('uint8');
     
      if s.ndim == 2:
        s.shape = s.shape + (1,);
      if s.ndim != 3:
        raise RuntimeError('Sources dont have dimensions 2 or 3 but %d in source %d!' % (s.ndim, i));
      
      self.image_items[i].updateImage(s[self.source_slice[:s.ndims]]);
    
    self.sources = source;
  
  def getXYAxes(self):
    return [a for a in [0,1,2] if a != self.source_axis];
  
  def updateSourceRange(self):
    x,y = self.getXYAxes();
    self.source_range_x = self.source_scale[x] * self.source_shape[x];
    self.source_range_y = self.source_scale[y] * self.source_shape[y];
  
  def updateSourceSlice(self):
    # current slice of the source
    self.source_slice = [slice(None)] * 3;
    self.source_slice[self.source_axis] = self.source_index[self.source_axis]; 
    self.source_slice = tuple(self.source_slice);
    #print(self.source_slice)

  def updateSlicer(self):
    ax = self.source_axis;
    self.slicePlot.setXRange(0, self.source_shape[ax]);
    self.sliceLine.setValue(self.source_index[ax]);
    stop = self.source_shape[ax] + 0.5;
    self.sliceLine.setBounds([0, stop])
  
  def updateLabelFromMouseMove(self, event):
    mousePoint = self.view.mapSceneToView(event)
    x, y = mousePoint.x(), mousePoint.y(); 
    x = min(max(0, x), self.source_range_x);
    y = min(max(0, y), self.source_range_y); 
    #print(x,y);    
    
    ax,ay = self.getXYAxes();
    x = min(int(x/self.source_scale[ax]), self.source_shape[ax]-1);
    y = min(int(y/self.source_scale[ay]), self.source_shape[ay]-1);
    z = self.source_index[self.source_axis];
    pos = [z] * 3;
    pos[ax] = x; pos[ay] = y;
    self.source_pointer = pos;
    self.updateLabel();   
     
  def updateLabel(self):
    x,y,z = self.source_pointer;
    xs,ys,zs = self.source_scale;
    vals = ", ".join([str(s[x,y,z]) for s in self.sources]);
    self.source_label.setText("<span style='font-size: 12pt; color: black'>(%d, %d, %d) {%.2f, %.2f, %.2f} [%s]</span>" % (x,y,z,x*xs,y*ys,z*zs,vals))
  
  def updateSlice(self):
    ax = self.source_axis;
    index = min(max(0, int(self.sliceLine.value())), self.source_shape[ax]-1);
    if index != self.source_index[ax]:
      self.source_index[ax] = index;
      self.source_slice = self.source_slice[:ax] + (index,) + self.source_slice[ax+1:];
      self.source_pointer[ax] = index;
      self.updateLabel();
      self.updateImage()
  
  def setSliceAxis(self, axis):
    self.source_axis = axis;
    self.updateSourceRange();
    self.updateSourceSlice();
    
    for i, s in zip(self.image_items, self.sources):
      i.updateImage(s[self.source_slice]);
      i.setRect(pg.QtCore.QRect(0, 0, self.source_range_x, self.source_range_y))
    self.view.setXRange(0, self.source_range_x);
    self.view.setYRange(0, self.source_range_y);
    
    self.updateSlicer();
  
  def updateImage(self):
    for i, s in zip(self.image_items, self.sources):
      #print(self.getXYAxes());
      #print(self.source_slice)
      #print(self.source_shape);
      #print(self.source_scale)
      image = s[self.source_slice[:s.ndim]];
      if image.dtype == bool:
        image = image.view('uint8');
      #print(image.shape);
      i.updateImage(image);
      
  def setMinMax(self, minMax, source = 0):
    self.luts[source].lut.region.setRegion(minMax);
  
    
  def shape3d(self, shape):
    return (shape + (1,) * 3)[:3];
  


############################################################################################################
### Tests
############################################################################################################

def _test():
  import numpy as np
  import ClearMap.Visualization.Qt.DataViewer as dv
  
  img1 = np.random.rand(*(100,80,30));

  dv.DataViewer(img1)
