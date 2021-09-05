#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Data Color Viewer
"""

#TODO: window manager: figure(x) functionality in order to replot in a dataviewer
#TODO: cleanup -> source management -> 

import numpy as np;

import pyqtgraph as pg
from functools import partial

import ClearMap.IO.IO as io;

import ClearMap.GUI.Utils as guiutil;

pg.mkQApp()


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
      data = data[sl]
    return np.nanmin(data), np.nanmax(data)


class LUTWidget(pg.GraphicsView):
  """Lookup table widget for the DataViewer"""
    
  def __init__(self, parent = None,  *args, **kargs):
    background = kargs.get('background', 'default')
    pg.GraphicsView.__init__(self, parent = parent, useOpenGL = False, background = background)
    self.item = LUTItem(*args, **kargs)
    self.setCentralItem(self.item)
    #self.setSizePolicy(pg.QtGui.QSizePolicy.Minimum, pg.QtGui.QSizePolicy.Expanding)
    self.setSizePolicy(pg.QtGui.QSizePolicy.Preferred, pg.QtGui.QSizePolicy.Expanding)
    self.setMinimumWidth(50)
    
  def sizeHint(self):
    return pg.QtCore.QSize(50, 200)
    
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
        button.setMaximumWidth(20);
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
        abm.clicked.connect(partial(self.updateRegionRange, m, p));  
        
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
      sl[ax] = slice(None, None, 2)
      data = data[sl]
    return np.nanpercentile(data, percentiles);
  


class DataViewer(pg.QtGui.QWidget):
  def __init__(self, source, axis = None, scale = None, title = None, invertY = False, minMax = None, screen = None, parent = None, *args):
    ### Images soures
    self.initializeSources(source, axis = axis, scale = scale)
    
    ### Gui Construction
    pg.QtGui.QWidget.__init__(self, parent, *args);
                             
    if title is None:
      if isinstance(source, str):
        title = source;
      elif isinstance(source, io.src.Source):
        title = source.location;
      if title is None:
        title = 'DataViewer';                             
    self.setWindowTitle(title);
    self.resize(1600,1200)      
    
    self.layout = pg.QtGui.QGridLayout(self);
    self.layout.setContentsMargins(0,0,0,0)     
    
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
    
    #  Image plots
    self.image_items = [pg.ImageItem(s[self.source_slice]) for s in self.sources];
    for i in self.image_items:
      i.setRect(pg.QtCore.QRect(0, 0, self.source_range_x, self.source_range_y))
      i.setCompositionMode(pg.QtGui.QPainter.CompositionMode_Plus);
      self.view.addItem(i);
    self.view.setXRange(0, self.source_range_x);
    self.view.setYRange(0, self.source_range_y);
    
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
    
    # Axis Tools
    axis_tools_layout = pg.QtGui.QGridLayout()
    self.axis_buttons = [];
    axesnames = ['x', 'y', 'z'];
    for d in range(3):
      button = pg.QtGui.QRadioButton(axesnames[d]);
      button.setMaximumWidth(50);
      axis_tools_layout.addWidget(button,0,d);
      button.clicked.connect(partial(self.setSliceAxis, d));
      self.axis_buttons.append(button);
    self.axis_buttons[self.source_axis].setChecked(True);
    axis_tools_widget = pg.QtGui.QWidget();
    axis_tools_widget.setLayout(axis_tools_layout);
    
    # coordinate label
    self.source_pointer = [0,0,0];
    self.source_label = pg.QtGui.QLabel("");
    axis_tools_layout.addWidget(self.source_label,0,3);
    
    self.graphicsView.scene().sigMouseMoved.connect(self.updateLabelFromMouseMove);
    
    #compose the image viewer
    image_splitter.addWidget(self.graphicsView);
    image_splitter.addWidget(self.slicePlot)
    image_splitter.addWidget(axis_tools_widget); 
    image_splitter.setSizes([self.height()-35-20, 35, 20])
    
    # lut widgets
    if self.nsources == 1:
      cols = ['flame'];
    elif self.nsources == 2:
      cols = ['purple',  'green'];
    else:
      cols = np.array(['white',  'green','red', 'blue', 'purple'] * self.nsources)[:self.nsources];
    
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
    self.sources  = [io.as_source(s)[:] for s in source];
    
    # avoid bools
    for i,s in enumerate(self.sources):
      if s.dtype == bool:
        self.sources[i] = s.view('uint8');
     
    # ensure 3d images 
    for i,s in enumerate(self.sources):
      if s.ndim == 2:
        s = s.view();
        s.shape = s.shape + (1,);
        self.sources[i] = s;
      if s.ndim != 3:
        raise RuntimeError('Sources dont have dimensions 2 or 3 but %d in source %d!' % (s.ndim, i));
    
    # source shapes
    self.source_shape = self.sources[0].shape;  
    self.source_shape2 = np.array(np.array(self.source_shape, dtype = float) / 2, dtype = int);
    for i,s in enumerate(self.sources):
      if s.shape != self.source_shape:
        raise RuntimeError('Sources dont have the same shape %r vs %r in source %d!' % (self.source_shape, s.shape, i));
    
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
      
      if s.dtype == bool:
        self.sources[i] = s.view('uint8');
     
      if s.ndim == 2:
        s.shape = s.shape + (1,);
      if s.ndim != 3:
        raise RuntimeError('Sources dont have dimensions 2 or 3 but %d in source %d!' % (s.ndim, i));
      
      self.image_items[i].updateImage(s[self.source_slice]);
    
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
      self.source_slice[ax] = index;
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
      i.updateImage(s[self.source_slice]);
      
  def setMinMax(self, minMax, source = 0):
    self.luts[source].lut.region.setRegion(minMax);
  

def plot(source, axis = None, scale = None, title = None, invertY = True, minMax = None, screen = None):
  if not isinstance(source, (list, tuple)):
    source = [source];
  return multi_plot(source, axis = axis, scale = scale, title = title, invertY = invertY, minMax = minMax, screen = screen);
  #else:          
  #  return DataViewer(source = source, axis = axis, scale = scale, title = title, invertY = invertY, minMax = minMax);


def synchronize(dv1, dv2):
  """Synchronize scrolling between two data viewers"""
  # sync dv1 -> dv2
  def sV():
    dv2.sliceLine.setValue(dv1.sliceLine.value());
  dv1.sliceLine.sigPositionChanged.connect(sV);
  for d,button in enumerate(dv1.axis_buttons):
    button.clicked.connect(partial(dv2.setSliceAxis, d));
  
  # sync dv2 -> dv1  
  def sV():
    dv1.sliceLine.setValue(dv2.sliceLine.value());
  dv2.sliceLine.sigPositionChanged.connect(sV);
  for d,button in enumerate(dv2.axis_buttons):
    button.clicked.connect(partial(dv1.setSliceAxis, d));
  
  dv1.view.setXLink(dv2.view);
  dv1.view.setYLink(dv2.view);
 

def dualPlot(source1, source2, axis = None, scale = None, title = None, invertY = True, minMax = None, arrange = True, percent = 80, screen = None):
  return multiPlot([source1, source2], axis = axis, scale = scale, title = title, invertY = invertY, minMax = minMax, screen = screen);
 

def multiPlot(sources, axis = None, scale = None, title = None, invertY = True, minMax = None, arrange = True, screen = None):

  if not isinstance(title, (tuple, list)):
    title = [title] * len(sources);
  
  dvs = [DataViewer(source = s, axis = axis, scale = scale, title = t,
                    invertY = invertY, minMax = minMax) for s,t in zip(sources, title)];

  if arrange:
    geo = guiutil.tiled_layout(len(dvs), percent = 80, screen = screen);
  
  for d,g in zip(dvs, geo):
    #d.setFixedSize(int(0.95 * g[2]), int(0.9 * g[3]));                  
    d.setGeometry(pg.QtCore.QRect(*g));

  for d1,d2 in zip(dvs[:-1], dvs[1:]):
    synchronize(d1, d2);
  
  return dvs;
  
#Future
multi_plot = multiPlot
dual_plot = dualPlot



if __name__ == '__main__':
  import pyqtgraph as pg

  import numpy as np
  import ClearMap.GUI.DataViewer as dv;
  reload(dv);
  
  #h = dv.Histogram()
  #h.show();
  
  img1 = np.random.rand(*(100,80,30));
  img2 = np.random.rand(*(100,80,30)) > 0.5;
  
  p = dv.DataViewer([img1,img2])
  
  dv.plot([img1, img2])
