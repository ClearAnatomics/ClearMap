# -*- coding: utf-8 -*-
"""
GT
==

Interface to read and write graph tool files.

Note
----
The module utilizes the gt writer/reader from graph_tool.

See also
--------
:mod`ClearMap.Analysis.Graphs`
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


import ClearMap.Analysis.Graphs.GraphGt as ggt

import ClearMap.IO.Source as src

###############################################################################
### Source classe
###############################################################################

class Source(src.Source):
  """GT graph source."""
  
  def __init__(self, location = None, graph = None, name = None):
    """GT source class construtor.
    
    Arguments
    ---------
    location : str or None
      The filename of the graph source.
    graph : Graph or None
      The graph object 
    """
    super(Source, self).__init__(name=name);
    
    if isinstance(location, ggt.Graph):
      graph = location;
      location = None;
    
    self._location = location;
    self._graph = graph;
  
    
  @property
  def name(self):
    return "Graph-Source";  
  
  @property
  def location(self):
    return self._location;
    
  @location.setter
  def location(self, value):
    if value != self.location:
      self._location = value;

  @property
  def graph(self):
    """The underlying graph.
    
    Returns
    -------
    graph : Graph
      The underlying graph of this source.
    """
    if self._graph is None:
      self._graph = _graph(self.location);
    return self._graph;
  
  @graph.setter
  def graph(self, value):
    self._graph = value; 
  
    
  @property 
  def shape(self):
    """The shape of the source.
    
    Returns
    -------
    shape : tuple
      The shape of the source.
    """
    return self.graph.shape;
  
  @shape.setter
  def shape(self, value):
    self.graph.shape = value;
  
  def as_virtual(self):
     return VirtualSource(source=self);
     
  def as_real(self):
    return self;
  
  
  ### Generic
  def info(self):
    self.graph.info();
  
  def write(self, location = None):
    if location is None:
      location = self.location;
    return _write(location, self.graph);
    
  def read(self, location = None):
    if location is None:
      location = self.location;
    self._graph = _graph(location);
  
  def copy(self):
    return Source(graph=self.graph.copy())
  
  
  ### Formatting
  def __str__(self):
    try:
      name = self.name;
      name = '%s' % name if name is not None else '';
    except:
      name ='';
    
    try:
      graph = self._graph.__str__()[5:];
    except:
      graph = '';
    
    try:
      location = self.location;
      location = '%s' % location if location is not None else '';
      if len(location) > 100:
        location = location[:50] + '...' + location[-50:]
      if len(location) > 0:
        location = '{%s}' % location;
    except:
      location = '';    

    
    return name + graph + location


class VirtualSource(src.VirtualSource):
  def __init__(self, source = None, location = None, name = None):
    if source is not None and location is None:
      location = source.location;
    super(VirtualSource, self).__init__(location=location, name=name);
  
  @property 
  def name(self):
    return 'Virtual-Graph-Source';
  
  def as_virtual(self):
    return self;
  
  def as_real(self):
    return Source(location=self.location);
  
  
  @property
  def graph(self):
    """The underlying graph.
    
    Returns
    -------
    graph : Graph
      The underlying graph of this source.
    """
    if self._graph is None:
      self._graph = _graph(self.location);
    return self._graph;
  
  @graph.setter
  def graph(self, value):
    raise NotImplementedError("Cannot set virtual graph")
  
    
  @property 
  def shape(self):
    """The shape of the source.
    
    Returns
    -------
    shape : tuple
      The shape of the source.
    """
    return self.graph.shape;
  
  @shape.setter
  def shape(self, value):
    raise NotImplementedError("Cannot set shape of virtual graph")


###############################################################################
### IO Interface
###############################################################################

def is_graph(source):
  """Checks if this source is a graph source"""
  if isinstance(source, Source):
    return True;
  if isinstance(source, str) and len(source) >= 2 and source[-2:] == 'gt':
    return True;
  return False;


def read(source, as_source = None, **kwargs):
  """Read graph from a file.
  
  Arguments
  ---------
  source : str
    The name of the graph file.
  slicing : slice, Slice or None
    An optional sub-slice to consider.
  as_source : bool
    If True, return results as a source.
  
  Returns
  -------
  graph : Graph or Source
    The graph as a Graph class or source.
  """ 
  if not isinstance(source, Source):
    source = Source(source);
  if as_source:
    return source
  else:
    return source.graph;
  

def write(sink, graph, **kwargs):
  """Write graph to a file.
  
  Arguments
  ---------
  sink : str
    The name of the CSV file.
  graph : Graph 
    The data to write into the CSV file.
  
  Returns
  -------
  sink : grpah or source
    The sink graph file.
  """ 
  if not isinstance(sink, Source):
    sink = Source(sink);
  
  return _write(sink, graph);


def create(location = None, **kwargs):
  raise NotImplementedError('Creating graph files not implemented yet!') 


###############################################################################
### Helpers
###############################################################################

def _graph(location, **kwargs):
    """Read graph from file.
    
    Arguments
    ---------
    location : str
      Location of the csv array data.
    
    Returns
    -------
    graph : Graph
      The graph as a Graph object.
    """
    graph = ggt.load(location);
    return graph;
  

def _write(filename, graph, **args):
    """Write graph  to file.
    """
    ggt.save(filename, graph)
    return filename

###############################################################################
### Tests
###############################################################################

def test():    
    """Test GT module"""
    import os
    import ClearMap.Analysis.Graphs.GraphGt as ggt
    import ClearMap.IO.GT as gt
    
    location = 'test.gt';
    
    g = ggt.Graph(n_vertices=10)
 
    s = gt.Source(graph=g, location=location);
    s.shape = (1,2,3);
    print(s)

    s.write();

    r = gt.Source(location=location) 
    print(r.shape)
    
    os.remove(location)
    
