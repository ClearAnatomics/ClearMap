#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Graph analysis module
---------------------

Module provides basic Graph classes.

"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


import numpy as np

###############################################################################
### Base graph class
###############################################################################

class Graph(object):
  """Abstract base Graph class.
  
  All graph interfaces should inherit from the class.
  """
  
  def __init__(self, name = None, n_vertices = None, edges = None, directed = None):
    
    if name is not None:
      self.name = name;
    
    if directed is not None:
      self.directed = directed;
    else:
      self.directed = False;
    
    if n_vertices is not None:
      self.add_vertex(n_vertices=n_vertices);
    
    if edges is not None:
      self.add_edge(edges);
  
  
  @property 
  def name(self):
    if hasattr(self, '_name'):
      return self._name;
    else:
      return type(self).__name__;

  @name.setter
  def name(self, value):
    self._name = str(value);
  
  @property
  def directed(self):
    return self._directed;
  
  @directed.setter
  def directed(self, value):
    self._directed = bool(value);
  
  ### Vertices
   
  @property
  def n_vertices(self):
    """Number of vertices in the graph."""
    return None;
  
  @n_vertices.setter
  def n_vertices(self, value):
    raise NotImplementedError('Not implemented in graph class %s!' % self.name)
  
  def vertex(self, index):
    raise NotImplementedError('Not implemented in graph class %s!' % self.name)
  
  @property
  def vertices(self):
    raise NotImplementedError('Not implemented in graph class %s!' % self.name)  
  
  
  def add_vertex(self, n_vertices = None, index = None, vertex = None):
    raise NotImplementedError('Not implemented in graph class %s!' % self.name)
        
  def remove_vertex(self, index = None, vertex = None):
    raise NotImplementedError('Not implemented in graph class %s!' % self.name)
  
  
  def vertex_property(self, name, index = None):
    raise NotImplementedError('Not implemented in graph class %s!' % self.name)
      
  def vertex_properties(self):
    raise NotImplementedError('Not implemented in graph class %s!' % self.name)
  
  def add_vertex_property(self, name, source, dtype = None):
    raise NotImplementedError('Not implemented in graph class %s!' % self.name)    
  
  def set_vertex_property(self, name, source):
    raise NotImplementedError('Not implemented in graph class %s!' % self.name)

  def remove_vertex_property(self, name):
    raise NotImplementedError('Not implemented in graph class %s!' % self.name)

  def vertex_degrees(self):
    raise NotImplementedError('Not implemented in graph class %s!' % self.name)    

  def vertex_degree(self, index):
    raise NotImplementedError('Not implemented in graph class %s!' % self.name)

  def vertex_out_degrees(self):
    raise NotImplementedError('Not implemented in graph class %s!' % self.name)    

  def vertex_out_degree(self, index):
    raise NotImplementedError('Not implemented in graph class %s!' % self.name)

  def vertex_in_degrees(self):
    raise NotImplementedError('Not implemented in graph class %s!' % self.name)    

  def vertex_in_degree(self, index):
    raise NotImplementedError('Not implemented in graph class %s!' % self.name)
  
  
  ### Edges 
  
  @property
  def n_edges(self):
    """Number of edges in the graph."""
    return None;
  
  @n_edges.setter
  def n_edges(self, value):
    raise NotImplementedError('Not implemented in graph class %s!' % self.name)  
  
  def edge(self, edge):
    raise NotImplementedError('Not implemented in graph class %s!' % self.name)
  
  @property
  def edges(self):
    raise NotImplementedError('Not implemented in graph class %s!' % self.name)
    
  def edge_connectivity(self):
    raise NotImplementedError('Not implemented in graph class %s!' % self.name)    
    
  def add_edge(self, edge):
    raise NotImplementedError('Not implemented in graph class %s!' % self.name)  
  
  def remove_edge(self, edge):
    raise NotImplementedError('Not implemented in graph class %s!' % self.name)    
  
    
  def edge_property(self, name):
    raise NotImplementedError('Not implemented in graph class %s!' % self.name)
      
  def edge_properties(self):
    raise NotImplementedError('Not implemented in graph class %s!' % self.name)
  
  def add_edge_property(self, name, source, dtype = None):
    raise NotImplementedError('Not implemented in graph class %s!' % self.name)    
  
  def set_edge_property(self, name, source):
    raise NotImplementedError('Not implemented in graph class %s!' % self.name)

  def remove_edge_property(self, name):
    raise NotImplementedError('Not implemented in graph class %s!' % self.name)

  ### IO
  
  def save(self, filename):
    raise NotImplementedError('Not implemented in graph class %s!' % self.name)       
    
  def load(self, filename):
    raise NotImplementedError('Not implemented in graph class %s!' % self.name)     

  def __str__(self):
    try:
      name = self.name;
      name = '%s' % name if name is not None else '';
    except:
      name ='';
      
    try:
      vertices = self.n_vertices;
      vertices = '[%d·]' % vertices if vertices is not None else '';
    except:
      vertices = '';
    
    try:
      edges = self.n_edges;
      edges = '[%d-]' % edges if edges is not None else '';
    except:
      edges = '';
      
    return name + vertices + edges;
  
  def __repr__(self):
    return self.__str__();


###############################################################################
### Graphs with spatial geometry
###############################################################################    

class GeometricGraph(Graph):
  """Base class for graphs whose vertices are embedded in an Eucledian space."""
  
  def __init__(self, name = None, n_vertices = None, edges = None, directed = False,
                     vertex_coordinates = None, vertex_radii = None,
                     edge_coordinates = None, edge_radii = None, edge_geometries = None, shape = None):
    super(GeometricGraph, self).__init__(name=name, n_vertices=n_vertices, edges=edges, directed=directed);
    
    if vertex_coordinates is not None:
      self.vertex_coordinates = vertex_coordinates;
    
    if vertex_radii is not None:
      self.vertex_radii = vertex_radii;
    
    if edge_coordinates is not None:
      self.edge_coordinates = edge_coordinates;
    
    if edge_radii is not None:
      self.edge_radii = edge_radii;
    
    if edge_geometries is not None:
      self.edge_geometries = edge_geometries;
    
    self.shape = shape;
  
  
  @property  
  def shape(self):
    """The shape of the underlying space in which the graph is embedded."""
    return self._shape;
  
  @shape.setter
  def shape(self, value):
    self._shape = value;
  
  @property
  def ndim(self):
    return len(self.shape);
  
    
  ### Vertices
  
  def vertex_coordinates(self, axis = None, vertex = None):
    raise NotImplementedError('Not implemented in graph class %s!' % self.name)           
  
  def set_vertex_coordinates(self, coordinates, axis = None, vertex = None):
    raise NotImplementedError('Not implemented in graph class %s!' % self.name)   

  def vertex_radii(self, vertex = None):
    raise NotImplementedError('Not implemented in graph class %s!' % self.name)      

  def set_vertex_radii(self, radius, vertex = None):
    raise NotImplementedError('Not implemented in graph class %s!' % self.name)  
  
  ### Edges
  
  def edge_coordinates(self, edge = None):
    raise NotImplementedError('Not implemented in graph class %s!' % self.name)           
    
  def set_edge_coordinates(self, coordinates, axis = None, edge = None):
    raise NotImplementedError('Not implemented in graph class %s!' % self.name) 
  
  
  def edge_radii(self, edge = None):
    raise NotImplementedError('Not implemented in graph class %s!' % self.name)           
  
  def set_edge_radii(self, edge = None):
    raise NotImplementedError('Not implemented in graph class %s!' % self.name)  

  
  def edge_coordinates_from_vertex_coordinates(self):    
    coords = self.vertex_coordinates();
    i,j = self.edge_connectivity().T;
    return (0.5 * (coords[i] + coords[j]));
  
  def edge_radii_from_vertex_radii(self):
    r = self.vertex_radii();
    i,j = self.edge_connectivity().T;
    return (0.5 * (r[i] + r[j]));
  
#  def set_edge_radii_from_vertex_radii(self):
#    r = self.vertex_radii();
#    i,j = self.edge_connectivity().T;
#    self.set_edge_radii(0.5 * (r[i] + r[j]));


  def edge_vectors(self, normalize = False):
    xyz = self.vertex_coordinates();
    i,j = self.edge_connectiivty().T;
    v = xyz[i] - xyz[j];
    if normalize:
      v = (v.T / np.linalg.norm(v, axis = 1)).T;
    return v;
  
  
  # Edge geometries
  
  @property
  def has_edge_geometry(self):
    return False; 
    
  @property
  def edge_geometry_type(self):
    return None;
  
  def edge_geometry(self, *args, **kwargs):
    raise NotImplementedError('Not implemented in graph class %s!' % self.name) 
  
  def set_edge_geometry(self, *args, **kwargs):
    raise NotImplementedError('Not implemented in graph class %s!' % self.name) 
    
  def reduce_edge_geometry(self):
    raise NotImplementedError('Not implemented in graph class %s!' % self.name)
 
  def expand_edge_geometry(self):
    raise NotImplementedError('Not implemented in graph class %s!' % self.name)
    
  ### Functionality
  
  #def from_skeleton(self, skeleton):
  #  raise NotImplementedError('Not implemented in graph class %s!' % self.name)
  
  #def skeleton(self, sink = None, dtype = bool):
  #  raise NotImplementedError('Not implemented in graph class %s!' % self.name);

  
  ### Functionality
  def __str__(self):
    s = super(GeometricGraph, self).__str__();
    
    try:
      shape = self.shape;
      shape = '%r' % shape if shape is not None else '';
    except:
      shape = '';
      
    try:
      edge_geometry_type = self.edge_geometry_type;
      if edge_geometry_type is None:
        edge_geometry_type = ''
      elif edge_geometry_type == 'graph':
        edge_geometry_type = '|G|' 
      else:
        edge_geometry_type = '|E|'
    except:
      edge_geometry_type = '';
      
    return s + shape + edge_geometry_type;
        

class AnnotatedGraph(GeometricGraph):
  """Base class for graphs whose vertices are embedded in an Eucledian space and have an annotation."""
  
  def __init__(self, name = None, n_vertices = None, edges = None, directed = False,
                     vertex_coordinates = None, vertex_radii = None,
                     edge_coordinates = None, edge_radii = None, edge_geometries = None, shape = None,
                     vertex_labels = None, edge_labels = None, annotation = None):
    
    super(AnnotatedGraph, self).__init__(name=name, n_vertices=n_vertices, edges=edges, directed=directed,
                                         vertex_coordinates=vertex_coordinates, vertex_radii=vertex_radii,
                                         edge_coordinates=edge_coordinates, edge_radii=edge_radii, edge_geometries=edge_geometries, shape=shape);
    if vertex_labels is not None:
      self.vertex_labels = vertex_labels;
    
    if edge_labels is not None:
      self.edge_labels = edge_labels;
    
    self.annotation = annotation;
  
  @property
  def annotation(self):
    return self._annotation;
  
  @annotation.setter
  def annotation(self, value):
    self._annotation = value;
  
  
  ### Vertices
  
  def vertex_annotation(self):
    raise NotImplementedError('Not implemented in graph class %s!' % self.name)
  

  def set_vertex_annotation(self, value):
    raise NotImplementedError('Not implemented in graph class %s!' % self.name)
 

  ### Edges
  def edge_annotation(self):
    raise NotImplementedError('Not implemented in graph class %s!' % self.name)
  
  def set_edge_annotation(self, value):
    raise NotImplementedError('Not implemented in graph class %s!' % self.name)
 
  
  def __str__(self):
    s = super(AnnotatedGraph, self).__str__();
    
    try:
      annotation = self.annotation;
      annotation = '{{%r}}' % annotation if annotation is not None else '';
    except:
      annotation = '';

    return s + annotation;
 

  
def load(filename):
  return Graph().load(filename);

def save(filename, graph):
  graph.save(filename);



###############################################################################
### Tests
###############################################################################

def _test():
  import ClearMap.Analysis.Graphs.Graph as gr
  reload(gr)
  
  g = gr.Graph();
  print(g)
  
  g = gr.GeometricGraph();
  print(g)

  g = gr.AnnotatedGraph(annotation='test.npy')
  print(g)

  
  
  
  