#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Data Color Viewer
"""

import numpy as np;

import pyqtgraph as pg
from functools import partial

import ClearMap.IO.IO as io;

pg.mkQApp()


class HistogramLUTItemFast(pg.HistogramLUTItem):
  """A fast version of the Histogram LUT Widget"""
  
  def __init__(self, *args, **kargs):
    pg.HistogramLUTItem.__init__(self, *args, **kargs);
  
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


class HistogramLUTWidgetFast(pg.GraphicsView):
    
  def __init__(self, parent=None,  *args, **kargs):
    background = kargs.get('background', 'default')
    pg.GraphicsView.__init__(self, parent, useOpenGL=False, background=background)
    self.item = HistogramLUTItemFast(*args, **kargs)
    self.setCentralItem(self.item)
    self.setSizePolicy(pg.QtGui.QSizePolicy.Preferred, pg.QtGui.QSizePolicy.Expanding)
    self.setMinimumWidth(95)
    
  def sizeHint(self):
    return pg.QtCore.QSize(115, 200)
    
  def __getattr__(self, attr):
    return getattr(self.item, attr)


class Histogram(pg.QtGui.QWidget):
  def __init__(self, image=None, color = 'red', percentiles = [[-200,-100, 0, 25, 50],[50,75,100, 150, 200]], parent = None, *args):
    #pg.HistogramLUTItem.__init__(self, image=None, fillHistogram=True);
    pg.QtGui.QWidget.__init__(self, parent, *args)
   
    self.layout = pg.QtGui.QGridLayout(self);
    self.layout.setSpacing(0);
    self.layout.setMargin(0);
    
    self.histogram = HistogramLUTWidgetFast(parent = parent, image = image);
    self.layout.addWidget(self.histogram, 0, 0, 1, 1) ; 
  
    self.range_layout = pg.QtGui.QGridLayout()
    self.range_buttons = [];
    pre = ['%d', '%d'];
    for r in range(2):
      range_buttons_m = [];
      for i,p in enumerate(percentiles[r]):
        b = pg.QtGui.QPushButton(pre[r] % (p));
        b.setMaximumWidth(25);
        self.range_layout.addWidget(b,r,i);
        range_buttons_m.append(b);
      self.range_buttons.append(range_buttons_m);
    
    self.layout.addLayout(self.range_layout,1,0,1,1);
    
    self.precentiles = percentiles;
    self.percentile_id = [2,2];
        
    for m,ab in enumerate(self.range_buttons):
      for p, abm in enumerate(ab):
        abm.clicked.connect(partial(self.updateRegionRange, m, p));  
        
    #default gradient
    self.histogram.gradient.getTick(0).color = pg.QtGui.QColor(0,0,0,0);
    self.histogram.gradient.getTick(1).color = pg.QtGui.QColor(color);
    self.histogram.gradient.updateGradient();
  
  def updateRegionRange(self, m,p):
    self.percentile_id[m] = p;
    pmin = self.precentiles[0][self.percentile_id[0]];
    pmax = self.precentiles[1][self.percentile_id[1]];
    self.updateRegionPercentile(pmin, pmax);  
  
  def updateRegionPercentile(self,pmin,pmax):
    iitem = self.histogram.imageItem();
    if iitem is not None:
      if pmax > 100:
        pmax1 = 100;
      else:
        pmax1 = max(0, pmax);
      if pmin < 0:
        pmin1 = min(-pmin, 100);
      else:
        pmin1 = min(pmin, 100);
      
      
      if pmax1 == 0:
        pmax1 = 1; pmax = 1;
      if pmin1 == 0:
        pmin1 = 1; pmin = 1;
      r = [float(pmin)/pmin1, float(pmax)/pmax1] * np.percentile(iitem.image, [pmin1, pmax1])
      self.histogram.region.setRegion(r);



class DataViewer(pg.QtGui.QWidget):
  def __init__(self, source, axis = None, scale = None, overlay = None, title = 'DataViewer', invertY = True, parent = None, *args):
    pg.QtGui.QWidget.__init__(self, parent, *args)
    self.setWindowTitle(title);
    self.resize(1600,1200)  
        
    #initialize sources and axis settings  
    if isinstance(source, tuple):
      source = list(source);
    if not isinstance(source, list):
      source = [source];
    self.nimages = len(source);
    #self.images  = [io.readData(s) for s in source];
    self.images  = [s for s in source];                 
    
    # avoid bools to produce errors with pyqt
    for i,img in enumerate(self.images):
      if img.dtype == bool:
        self.images[i] = img.astype('uint8');
      
    self.image_size  = io.shape(self.images[0]);
    self.image_size2 = np.array(np.array(self.image_size) / 2, dtype = int);
    self.image_dim = len(self.image_size);
    if self.image_dim != 3:
      raise RuntimeError('Images expected to be 3d');
    
    self.image_order = [0,1,2]; 
    self.images = [np.transpose(i, self.image_order) for i in self.images];
    
    if scale is None:
      scale = np.ones(self.image_dim);
    else:
      scale = np.array(scale);
    self.image_scale = scale;

    if axis is None:
      axis = 2;
    self.image_axis = axis;
    
    self.image_order_current  = np.roll(self.image_order, -axis);
    self.image_size_current   = np.roll(self.image_size,  -axis);
    self.image_scale_current  = np.roll(self.image_scale, -axis);
    self.images_current = [np.transpose(i, self.image_order_current) for i in self.images];
    self.image_slice_current = self.image_size_current[0] / 2;
    
    # Gui
    self.view = pg.ViewBox();
    self.view.setAspectLocked(True);
    if invertY:
      self.view.invertY()    
    
    self.graphicsView = pg.GraphicsView()
    self.graphicsView.setObjectName("GraphicsView")
    self.graphicsView.setCentralItem(self.view)
    
    self.layout = pg.QtGui.QGridLayout(self);
    self.layout.setContentsMargins(0,0,0,0)        
  
    splitter = pg.QtGui.QSplitter();
    splitter.setOrientation(pg.QtCore.Qt.Horizontal)
    #splitter.setSizes([int(self.width()*0.95), int(self.width()*0.05)]);
    splitter.setSizes([self.width() - 10, 10]);
    self.layout.addWidget(splitter);
    
    image_splitter = pg.QtGui.QSplitter();
    image_splitter.setOrientation(pg.QtCore.Qt.Vertical)
    splitter.addWidget(image_splitter);
    
    histogram_splitter = pg.QtGui.QSplitter();
    histogram_splitter.setOrientation(pg.QtCore.Qt.Horizontal)
    splitter.addWidget(histogram_splitter);
  
    #  Image plots
    self.image_items = [pg.ImageItem(i[self.image_slice_current]) for i in self.images_current];
    for ii in self.image_items:
      ii.setRect(pg.QtCore.QRect(0, 0, self.image_size_current[1] * self.image_scale_current[1],  self.image_size_current[2] * self.image_scale_current[2]))
      ii.setCompositionMode(pg.QtGui.QPainter.CompositionMode_Plus);
      self.view.addItem(ii);
    self.view.setXRange(0, self.image_size_current[1] * self.image_scale_current[1]);
    self.view.setYRange(0, self.image_size_current[2] * self.image_scale_current[2]);
    
    # Slice Selector
    self.slicePlot = pg.PlotWidget()
    sizePolicy = pg.QtGui.QSizePolicy(pg.QtGui.QSizePolicy.Preferred, pg.QtGui.QSizePolicy.Preferred)
    sizePolicy.setHorizontalStretch(0)
    sizePolicy.setVerticalStretch(0)
    sizePolicy.setHeightForWidth(self.slicePlot.sizePolicy().hasHeightForWidth())
    self.slicePlot.setSizePolicy(sizePolicy)
    self.slicePlot.setMinimumSize(pg.QtCore.QSize(0, 40))
    self.slicePlot.setObjectName("roiPlot");
    self.sliceCurve = self.slicePlot.plot()
    
    self.sliceLine = pg.InfiniteLine(0, movable=True)
    self.sliceLine.setPen((255, 0, 0, 200))
    self.sliceLine.setZValue(1)
    self.slicePlot.addItem(self.sliceLine)
    self.slicePlot.hideAxis('left')
    
    self.slicePlot.setXRange(0, self.image_size_current[0]);
    self.sliceLine.setValue(self.image_slice_current);
    stop = self.image_size_current[0] + 0.5;
    self.sliceLine.setBounds([0, stop])
    
    
    self.ignoreSliceLine = False
    self.sliceLine.sigPositionChanged.connect(self.sliceLineChanged)
    
    # Axis Tools
    axis_tools_layout = pg.QtGui.QGridLayout()
    self.axis_buttons = [];
    for d in range(self.image_dim):
      b = pg.QtGui.QPushButton('%d' % (d));
      b.setMaximumWidth(100);
      axis_tools_layout.addWidget(b,0,d);
      self.axis_buttons.append(b);
      
      b.clicked.connect(partial(self.setSliceAxis, d));
    
    axis_tools_widget = pg.QtGui.QWidget();
    axis_tools_widget.setLayout(axis_tools_layout);
    
    # coordinate label
    self.xyz = [0,0,0];
    self.xy_label = pg.QtGui.QLabel("<span style='font-size: 10pt; color: black'>(0, 0, 0) [0]</span>");
    axis_tools_layout.addWidget(self.xy_label,0,self.image_dim);
    
    #pg.SignalProxy(self.graphicsView.scene().sigMouseMoved, rateLimit=60, slot = self.mouseMoved)
    self.graphicsView.scene().sigMouseMoved.connect(self.mouseMoved)
    
    image_splitter.addWidget(self.graphicsView);
    image_splitter.addWidget(self.slicePlot)
    image_splitter.addWidget(axis_tools_widget); 
    image_splitter.setSizes([self.height()-35-20, 35, 20])
    
    # Histogram regions
    if self.nimages == 1:
      histcols = ['white'];
    elif self.nimages == 1:
      histcols = np.pad(['purple',  'green']);
    else:
      histcols = np.pad(['white',  'green','red', 'blue', 'purple'],(0, max(0, self.nimages-5)), 'edge');
    self.histograms = [Histogram(image = i, color = c) for i,c in zip(self.image_items, histcols)];
    for h in self.histograms:
      histogram_splitter.addWidget(h);
    
    self.levelMin = [];
    self.levelMax = [];
    for i,im in enumerate(self.images_current):
      lmin, lmax = list(map(float, self.quickMinMax(im[self.image_slice_current])));
      self.levelMin.append(lmin);
      self.levelMax.append(lmax); 
    
    self.show();

  def mouseMoved(self, event):
     mousePoint = self.view.mapSceneToView(event)
     x, y = int(mousePoint.x()), int(mousePoint.y()); 
     if x < 0: 
        x = 0;
     if y < 0:
        y = 0;
     if x >= self.image_size_current[1]:
        x = self.image_size_current[1]-1;
     if y >= self.image_size_current[2]:
        y = self.image_size_current[2]-1;
     z = self.image_slice_current;
     self.xyz = [x,y,z];
     vals = " ".join([str(i[z,x,y]) for i in self.images_current]);
     self.xy_label.setText("<span style='font-size: 10pt; color: black'>(%d, %d, %d) [%s]</span>" % (x,y,z, vals))
  
  def setSliceAxis(self, axis):
    self.image_order_current  = np.roll(self.image_order, -axis);
    self.image_size_current   = np.roll(self.image_size,  -axis);
    self.image_scale_current  = np.roll(self.image_scale, -axis);
    self.images_current = [np.transpose(i, self.image_order_current) for i in self.images];
    self.image_slice_current = self.image_size_current[0] / 2;

    for ii, im in zip(self.image_items, self.images_current):
      ii.updateImage(im[self.image_slice_current]);
      ii.setRect(pg.QtCore.QRect(0, 0, self.image_size_current[1] * self.image_scale_current[1],  self.image_size_current[2] * self.image_scale_current[2]))
    self.view.setXRange(0, self.image_size_current[1] * self.image_scale_current[1]);
    self.view.setYRange(0, self.image_size_current[2] * self.image_scale_current[2]);
    
    stop = self.image_size_current[0] + 0.5;
    self.sliceLine.setBounds([0, stop])  
    self.slicePlot.setXRange(0, self.image_size_current[0]);
    #self.updateImage();
  
  def quickMinMax(self, data):
    """
    Estimate the min/max values of *data* by subsampling.
    """
    while data.size > 1e6:
        ax = np.argmax(data.shape)
        sl = [slice(None)] * data.ndim
        sl[ax] = slice(None, None, 2)
        data = data[sl]
    return np.nanmin(data), np.nanmax(data)
    
  
    
  def updateLevel(self):
    for i,im in enumerate(self.images_current):
      lmin, lmax = list(map(float, self.quickMinMax(im[self.image_slice_current])));
      self.levelMin[i] = lmin;
      self.levelMax[i] = max(1, lmax);
  
  
  def updateImage(self, autoHistogramRange=True):
    ## Redraw image on screen
    if self.images is None:
        return
    
    if autoHistogramRange:
      for i in range(self.nimages):
        self.histograms[i].histogram.setHistogramRange(self.levelMin[i], self.levelMax[i]);
    
    for ii, im in zip(self.image_items, self.images_current):
      ii.updateImage(im[self.image_slice_current]);
  
  
  def sliceIndex(self):
    ## Return the slice and frame index indicated by a slider
    if self.images is None:
      return (0,0)
    s = self.sliceLine.value()  
    ssize = self.image_size_current[0];
    if ssize <= 1:
      return (0,0)
    if s < 0:
      return (0,s);
    if s >= ssize:
      return (ssize-1,s);
    return int(s), s
  
  def sliceLineChanged(self):
    if self.ignoreSliceLine:
      return
    #self.play(0)
    (ind, sl) = self.sliceIndex();
    if ind != self.image_slice_current:
      self.image_slice_current = ind;
      self.updateImage()
      
      self.xyz[2] = ind;
      x,y,z = self.xyz;
      vals = " ".join([str(i[z,x,y]) for i in self.images_current]);
      self.xy_label.setText("<span style='font-size: 10pt; color: black'>(%d, %d, %d) [%s]</span>" % (x,y,z, vals))
      #self.sigTimeChanged.emit(ind, sl)
  
  def setSlice(self, ind):
    """Set the currently displayed slice index."""
    self.image_slice_current = np.clip(ind, 0, self.image_current_size[0]-1)
    self.updateImage()
    self.ignoreSliceLine = True
    self.sliceLine.setValue(self.image_slice_current)
    self.ignoreSliceLine = False

    
def test():
  import pyqtgraph as pg

  import numpy as np
  import ClearMap.GUI.DataViewerOld as dv;
  reload(dv);
  
  #h = dv.Histogram()
  #h.show();
  
  img1 = np.random.rand(*(100,80,30));
  img2 = np.random.rand(*(100,80,30)) > 0.5;
  
  v = dv.DataViewer([img1,img2])