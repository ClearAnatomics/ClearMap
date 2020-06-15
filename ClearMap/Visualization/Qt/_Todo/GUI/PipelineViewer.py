# -*- coding: utf-8 -*-
"""
PiplineViewer 
=============

This module provides a automatically constructed GUI to test out pipelines.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'

import numpy as np

from functools import partial

import pyqtgraph as pg

import ClearMap.GUI.Parameter as guipar
import ClearMap.GUI.DataViewer as guidv
import ClearMap.ImageProcessing.Pipeline as pln


class ProcessViewer(pg.QtGui.QWidget):
  def __init__(self, process, source, sink = None, title = None, scale = None, axis = None, invertY = True, **args):
    if not isinstance(process, pln.ProcessingStep):
      raise RuntimeError('process expected to be a ProcessingStep');
    
    self.process = process;
    self.source = source;   
    
    if sink is None:
      sink = np.zeros_like(source);
    self.sink = sink;
    
    ### Gui
    pg.QtGui.QWidget.__init__(self, **args)
    if title is None:
      title = process.name;
    if title is None:
      title = 'Process'
    self.setWindowTitle(title);
    self.resize(1000,800);    
    
    self.layout = pg.QtGui.QGridLayout(self);
    self.layout.setContentsMargins(0,0,0,0)        
  
    splitter = pg.QtGui.QSplitter();
    splitter.setOrientation(pg.QtCore.Qt.Horizontal)
    splitter.setSizes([50, self.width() - 50]);
    self.layout.addWidget(splitter);
    
    # parameter tree
    self.parameter = guipar.ParameterTree(name = process.name, parameter = process.parameter);
    splitter.addWidget(self.parameter);
    
    # dataviewer
    self.viewer = guidv.DataViewer(source = self.sink, scale = scale, axis = axis, invertY = invertY);    
    splitter.addWidget(self.viewer);
    
    splitter.setStretchFactor(0, 0);
    splitter.setStretchFactor(1, 1);
    
    self.parameter.paramSet.sigTreeStateChanged.connect(self.updateParameter);
    
    self.updateParameter();
    self.show();

  
  def updateParameter(self, *args, **kwargs):
    #print(self.parameter.parameterDict);
    #print(self.process.parameter);
    self.sink = self.process.execute(self.source, self.sink);
    self.updateImage();
    
  def updateImage(self):
    self.viewer.updateImage();
 

class PipelineViewer(pg.QtGui.QWidget):
  def __init__(self, pipeline, source, sink = None, title = None, scale = None, axis = None, invertY = True,  **args):
    if isinstance(pipeline, pln.ProcessingStep) and not isinstance(pipeline, pln.Pipeline):
      pipeline = pln.Pipeline(pipeline);
    if not isinstance(pipeline, pln.Pipeline):
      raise RuntimeError('process expected to be a Pipeline');
    
    self.pipeline = pipeline;
    self.source = source;   
    
    if sink is None:
      sink = np.zeros_like(source);
    self.sink = sink;

    # buffer for intermediate results
    self.sources = [self.source];      
    for s in range(pipeline.nsteps-1):
      self.sources.append(np.zeros_like(source));
    self.sources.append(sink);
    
    ### Gui
    pg.QtGui.QWidget.__init__(self, **args)
    if title is None:
      title = pipeline.name;
    if title is None:
      title = 'Pipeline'
    self.setWindowTitle(title);
    self.resize(1000,800);    
    
    self.layout = pg.QtGui.QGridLayout(self);
    self.layout.setContentsMargins(0,0,0,0)        
  
    splitter = pg.QtGui.QSplitter();
    splitter.setOrientation(pg.QtCore.Qt.Horizontal)
    splitter.setSizes([50, self.width() - 50]);
    self.layout.addWidget(splitter);
    
    # parameter trees
    self.tab = pg.QtGui.QTabWidget()
    self.tab.setTabPosition(pg.QtGui.QTabWidget.North)
    
    # source tab
    param = guipar.ParameterTree(name = 'Source', parameter = dict());
    self.tab.addTab(param, 'Source');
    
    self.parameters = [];
    for i,p in enumerate(self.pipeline.steps):
      param = guipar.ParameterTree(name = p.name, parameter = p.parameter);
      param.paramSet.sigTreeStateChanged.connect(partial(self.updateParameter, pid = i));
      self.tab.addTab(param, p.name);
      self.parameters.append(param);
    
    splitter.addWidget(self.tab);
    
    # dataviewer
    self.viewer = guidv.DataViewer(source = self.source, scale = scale, axis = axis, invertY = invertY);    
    splitter.addWidget(self.viewer);
    
    splitter.setStretchFactor(0, 0);
    splitter.setStretchFactor(1, 1);
    
    self.tab.currentChanged.connect(self.updateView);    
    
    self.show();
  
  def updateParameter(self, pid, *args, **kwargs):
    for i in range(pid, self.pipeline.nsteps):
      self.sources[i+1] = self.pipeline.steps[i].execute(self.sources[i], self.sources[i+1]);
    
    self.viewer.updateImage();
   
  def updateView(self, pid):
    if pid < 0:
      return;
    self.viewer.setSource(self.sources[pid])


class PipelineInspector(pg.QtGui.QWidget):
  """Pipeline viewer with a DataViewer for each processing step"""
  def __init__(self, pipeline, source, sink = None, title = 'ProcessViewer', scale = None, axis = None, invertY = True, **args):
    if isinstance(pipeline, pln.ProcessingStep) and not isinstance(pipeline, pln.Pipeline):
      pipeline = pln.Pipeline(pipeline);
    if not isinstance(pipeline, pln.Pipeline):
      raise RuntimeError('process expected to be a Pipeline');
    
    self.pipeline = pipeline;
    self.source = source;   
    
    if sink is None:
      sink = np.zeros_like(source);
    self.sink = sink;

    # buffer for intermediate results
    self.sources = [self.source];      
    for s in range(pipeline.nsteps-1):
      self.sources.append(np.zeros_like(source));
    self.sources.append(sink);
    
    ### Gui
    pg.QtGui.QWidget.__init__(self, **args)
    self.setWindowTitle(title);
    self.resize(1000,800);    
    
    self.layout = pg.QtGui.QGridLayout(self);
    self.layout.setContentsMargins(0,0,0,0)        
  
    self.tab = pg.QtGui.QTabWidget()
    self.tab.setTabPosition(pg.QtGui.QTabWidget.North)

    self.source_viewer = guidv.DataViewer(source = self.source, scale = scale, axis = axis, invertY = invertY);
    self.viewers = [self.source_viewer];
    self.tab.addTab(self.source_viewer, 'Source') 
    
    for i,p in enumerate(pipeline.steps):
      pv = ProcessViewer(process = p, source = self.sources[i], sink = self.sources[i+1], scale = scale, axis = axis, invertY = invertY);
      pv.parameter.paramSet.sigTreeStateChanged.disconnect(pv.updateParameter);
      pv.parameter.paramSet.sigTreeStateChanged.connect(partial(self.updateParameter, pid = i));
      self.tab.addTab(pv, p.name)
      self.viewers.append(pv);
    
    self.layout.addWidget(self.tab);
    
    self.show();
  
  def updateParameter(self, pid, *args, **kwargs):
    #print(self.parameter.parameterDict);
    #print(self.process.parameter);
    #print('called in pipeline')
    for i in range(pid, self.pipeline.nsteps):
      self.sources[i+1] = self.pipeline.steps[i].execute(self.sources[i], self.sources[i+1]);
      self.viewers[i+1].updateImage();


def plot(pipeline, source, sink = None, title = None, scale = None, axis = None, invertY = True):
  return PipelineViewer(pipeline = pipeline, source = source, sink = sink, title = title, scale = scale, axis = axis, invertY = invertY);


if __name__ == "__main__":
  import pyqtgraph as pg
  import numpy as np #analysis:ignore
  import ClearMap.ImageProcessing.Processing as imp
  import ClearMap.GUI.PipelineViewer as pv;
  reload(pv);
  
  def mul(source, sink = None, factor = 10):
    if sink is None:
      sink = np.empty_like(source)
    sink[:] = source * factor;
    return sink;
    
  def add(source, sink = None, add = 10):
    if sink is None:
      sink = np.empty_like(source)
    sink[:] = source + add;
    return sink;
    
  p1 = imp.ProcessingStep(name = 'mul', function = mul, factor = 10.0);
  p2 = imp.ProcessingStep(name = 'add', function = add, add = -5.0);  
  p = imp.Pipeline(name = 'combined', function = [p1, p2]);
  
  source = np.random.rand(*(100,80,30));
  
  pv.ProcessViewer(process = p1, source = source);
  
  pv.ProcessViewer(process = p, source = source);
  
  pv.PipelineViewer(p, source = source)