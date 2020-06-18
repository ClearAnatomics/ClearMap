# -*- coding: utf-8 -*-
"""
DataViewerSimple
================

A simple data viewer (under construction).
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


import os
import time

import numpy as np;

import pyqtgraph as pg
from functools import partial


import ClearMap.IO.IO as io;


###############################################################################
### Data viewer
###############################################################################


def dataViewer(source, axis = None, scale = None):
  source = io.readData(source);
  size = io.dataSize(source);
  size2 = np.array(np.array(size) / 2, dtype = int);
  order = [0,1,2];
  #ImageView expexts z,x,y ordering
  
  #order = np.roll(order, -2)
  source = np.transpose(source, order);
  
  dim = len(size);
  if scale is None:
    scale = np.ones(dim);
  else:
    scale = np.array(scale);
    assert(len(scale) == dim);
  #scale = np.roll(scale,-2);
  
  if axis is None:
    axis = 2;
  orderR  = np.roll(order, -axis);
  sourceR = np.transpose(source, orderR);
  sizeR   = np.roll(size, -axis);
  scaleR  = np.roll(scale,-axis);
  
  print sizeR;
  print scaleR;
    
  #axes = ('t', 'x', 'y');
  axes = None;
  
  percentiles = ([0,5,10,50],[50,90, 95, 100]);
  precent_id = [1, 2];
  
  # create the gui
  pg.mkQApp()  
  
  widget = pg.QtGui.QWidget();
  widget.setWindowTitle('Data Viewer');
  widget.resize(1000,800)  
  
  layout = pg.QtGui.QVBoxLayout();
  layout.setContentsMargins(0,0,0,0)        
  
  splitter = pg.QtGui.QSplitter();
  splitter.setOrientation(pg.QtCore.Qt.Vertical)
  splitter.setSizes([int(widget.height()*0.99), int(widget.height()*0.01)]);
  layout.addWidget(splitter);
  
  #  Image plot
  img = pg.ImageView();
  img.setImage(sourceR, axes = axes);
  img.imageItem.setRect(pg.QtCore.QRect(0, 0, sizeR[1] * scaleR[1],  sizeR[2] * scaleR[2]))
  img.view.setXRange(0, sizeR[1] * scaleR[1]);
  img.view.setYRange(0, sizeR[2] * scaleR[2]);
  img.setCurrentIndex(size2[axis]);
  img.ui.histogram.region.setRegion(np.percentile(sourceR[size2[axis]], [percentiles[0][precent_id[0]], percentiles[1][precent_id[1]]]));
  splitter.addWidget(img);
  
  # Tools
  tools_layout = pg.QtGui.QGridLayout()
  axis_buttons = [];
  for d in range(dim):
    b = pg.QtGui.QPushButton('%d' % (d));
    b.setMaximumWidth(100);
    tools_layout.addWidget(b,0,d);
    axis_buttons.append(b);

  adjust_buttons = [];
  pre = ['Min %d', 'Max %d'];
  iitem = dim;
  for r in range(2):
    adjust_buttons_mm = [];
    for p in percentiles[r]:
      b = pg.QtGui.QPushButton(pre[r] % (p));
      b.setMaximumWidth(120);
      tools_layout.addWidget(b,0,iitem);
      iitem +=1;
      adjust_buttons_mm.append(b);
    adjust_buttons.append(adjust_buttons_mm);
  
  tools_widget = pg.QtGui.QWidget();
  tools_widget.setLayout(tools_layout);
  splitter.addWidget(tools_widget);
  
  widget.setLayout(layout)
  widget.show();
  
  # Callbacks for handling user interaction
  def updateAxis(a):
    axis = a;
    orderR  = np.roll(order, -axis);
    sourceR = np.transpose(source, orderR);
    sizeR   = np.roll(size, -axis);
    scaleR  = np.roll(scale,-axis);
    img.setImage(sourceR, axes = axes);
    img.imageItem.setRect(pg.QtCore.QRect(0, 0, sizeR[1] * scaleR[1],  sizeR[2] * scaleR[2]))
    img.view.setXRange(0, sizeR[1] * scaleR[1]);
    img.view.setYRange(0, sizeR[2] * scaleR[2]);
    img.setCurrentIndex(size2[axis]);
    img.ui.histogram.region.setRegion(np.percentile(sourceR[size2[axis]], [percentiles[0][precent_id[0]], percentiles[1][precent_id[1]]]));
  
  for i,ab in enumerate(axis_buttons):
    ab.clicked.connect(partial(updateAxis, i));
  
  def updateRegion(m,p):
    precent_id[m] = p;
    img.ui.histogram.region.setRegion(np.percentile(sourceR[size2[axis]], [percentiles[0][precent_id[0]], percentiles[1][precent_id[1]]]));
    
  for m,ab in enumerate(adjust_buttons):
    for p, abm in enumerate(ab):
      abm.clicked.connect(partial(updateRegion, m, p));
  
  return widget;



## Start Qt event loop unless running in interactive mode or using pyside.

if __name__ == '__main__':
  pass
 