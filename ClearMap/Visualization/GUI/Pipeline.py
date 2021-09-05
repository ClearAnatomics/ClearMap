# -*- coding: utf-8 -*-
"""
Piplines are used to combine image processing step

Each pipeline is a made of single processing steps which can be pipelines again

In this way nested hierachies of steps can be build.
"""
__author__    = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__   = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__copyright__ = 'Copyright (c) 2017 by Christoph Kirst, The Rockefeller University, New York City'

import inspect

from collections import OrderedDict as odict

import ClearMap.Visualization.GUI.ParameterTools as par;


def identity(source, sink = None, **parameter):
    sink = source;
    return sink;


class ProcessingStep(object):
  """Basic Processing step"""
  def __init__(self, function = None, name = None, parameter = all, **args):
    self.name = name;
    
    if parameter is None:
      parameter = odict();
    if parameter is all and hasattr(function, '__call__'):
      argspec = inspect.getargspec(function);
      values = argspec.defaults;
      keys = argspec.args[-len(values):];
      parameter = odict([(k, v) for k,v in zip(keys, values) if k not in ['sink', 'source']]);
    self.parameter = par.joinParameter(parameter, args);
    
    if function is None:
      function = identity;
    if isinstance(function, ProcessingStep):
      self.parameter = par.joinParameter(function.parameter, parameter);
      if self.name is None:
        self.name = function.name;
      function = function.execute;
    self.function = function;
    
  def setParameter(self, **parameter):
    self.parameter = par.setParameter(self.parameter, **parameter);
  
  def execute(self, source, sink = None):
    return self.function(source, sink, **self.parameter);
    
  def __str__(self):
    s = 'Pipeline: %s\n' % str(self.name); 
    return s + par.writeParameter(self.parameter, head = ' ');
    
  def __repr__(self):
    return self.__str__();


class Pipeline(ProcessingStep):
  """A combination of processing steps"""
  def __init__(self, function = None, name = None, **args):
    self.name = name;
    
    if function is None:
      function = identity;
    if not isinstance(function, tuple) and not isinstance(function, list):
      function = [function];
    for i,f in enumerate(function):
      if not isinstance(f, Pipeline) and not isinstance(f, ProcessingStep):
        function[i] = ProcessingStep(name = "Step %d" % i, function = f);
    for i,f in enumerate(function):
      if f.name is None:
        f.name = 'p%d' % i;
      else:
        f.name = ('p%d'% i) + f.name;
    
    self.steps = function;
    self.parameter = odict([(f.name, f.parameter) for f in function]);
    self.nsteps = len(function);
    self.names = [f.name for f in self.steps];

  def execute(self, source, sink = None):
    res = source;
    for s in self.steps:
      res = s.execute(res);
    
    if sink is None:
      return res;
    else:
      sink[:] = res;
      return sink;
  

if __name__ == "__main__":
  import numpy as np  
  import ClearMap.ImageProcessing.Processing as imp
  
  def mul(source, sink = None, factor = 10.0):
    sink = source * factor;
    return sink;
    
  def add(source, sink = None, add = -0.5):
    sink = source + add;
    return sink;
    
  p1 = imp.ProcessingStep(name = 'mul', function = mul);
  p2 = imp.ProcessingStep(name = 'add', function = add, add = 0.5);  
  
  data = np.random.rand(5);
  res = p1.execute(data);
  np.all(res == mul(data, **p1.parameter))
  
  p = imp.Pipeline(name = 'combined', function = [p1, p2]);
  print (p)
  
  res = p.execute(data);
  np.all(res == add(mul(data, **p1.parameter), **p2.parameter))