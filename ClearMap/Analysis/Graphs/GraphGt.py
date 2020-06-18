# -*- coding: utf-8 -*-
"""
GraphGt
=======

Module provides basic Graph interface to the
`graph_tool <https://graph-tool.skewed.de>`_ library.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


import copy
import numpy as np

import graph_tool as gt
import graph_tool.util as gtu
import graph_tool.topology as gtt
import graph_tool.generation as gtg

#fix graph tool saving / loading for very large arrays
import sys
if sys.version_info[0] < 3:
  import ClearMap.External.pickle_python_2 as pickle
else:
  #import ClearMap.External.pickle_python_3 as pickle  
  import ClearMap.External.pickle_python_3 as pickle
  

gt.gt_io.clean_picklers();

def pickler(stream, obj):
  sstream = gt.gt_io.BytesIO()
  pickle.dump(obj, sstream) #,  gt.gt_io.GT_PICKLE_PROTOCOL)
  stream.write(sstream.getvalue())

def unpickler(stream):
  data = stream.read(buflen=2**31)
  #print('unpickler loaded %d' % len(data))
  sstream = gt.gt_io.BytesIO(data)
  if sys.version_info < (3,):
      return pickle.load(sstream)
  return pickle.load(sstream, encoding="bytes")
  #return pickle.load(sstream); #, encoding="bytes")

gt.gt_io.libgraph_tool_core.set_pickler(pickler)
gt.gt_io.libgraph_tool_core.set_unpickler(unpickler)


import ClearMap.Analysis.Graphs.Graph as grp



###############################################################################
### Type Conversions
###############################################################################

def dtype_to_gtype(dtype):
  """Convert a data type to a graph_tool data type."""
  
  if isinstance(dtype, str):
    name = dtype;
  else:
    dtype = np.dtype(dtype);
    name = dtype.name;
  
  alias = { 'float64' : 'double', 
            'float32' : 'double',
            'int64'   : 'int64_t',
            'int32'   : 'int32_t',
            'uint64'  : 'int64_t',
            'uint32'  : 'int64_t'};
  if name in alias:
      name = alias[name];

  return gt._type_alias(name);


def ndim_to_gtype(ndim, gtype):
  """Convert a scalar gtype to a vector one if necessary."""
  if len(gtype) >= 6 and gtype[:6] == 'vector':
    return gtype;
  if ndim == 2:
    gtype = "vector<%s>" % gtype;
  elif ndim > 2:
    raise ValueError('Data for vertex properites can only be 1 or 2d!');
  
  return gtype;


def ndim_from_source(source):
  """Determines the dimension of a source appropiate for graph_tool."""
  if isinstance(source, (list, tuple)):
    if len(source) > 0:
      ndim = 2;
    else:
      ndim = 0;
  elif hasattr(source, 'dtype') and hasattr(source, 'ndim'):
    ndim = source.ndim;
  else:
    try:
      import ClearMap.IO.IO as io
      source = io.as_source(source);
      ndim = source.ndim;
    except:
      ndim = 0;
  return ndim;
 
 
def gtype_from_source(source, vectorize = True, graph_property = False):
    """Determines the graph_tool data type from a data source."""
    if isinstance(source, (list, tuple)):
      if len(source) > 0:
        source = source[0];
        dtype = np.asarray(source).dtype;
        gtype = dtype_to_gtype(dtype);
        ndim = 2;
      else:
        gtype = dtype_to_gtype('object');
        ndim = 0;
    elif hasattr(source, 'dtype') and hasattr(source, 'ndim'):
      dtype = source.dtype;
      gtype = dtype_to_gtype(dtype);
      ndim = source.ndim;
    else:
      try:
        import ClearMap.IO.IO as io
        source = io.as_source(source);
        dtype = source.dtype;
        gtype = dtype_to_gtype(dtype);
        ndim = source.ndim;
      except:
        gtype = dtype_to_gtype('object')
        ndim = 0;
    
    if vectorize:
      gtype = ndim_to_gtype(ndim, gtype);
    
    return gtype;


def vertex_property_map_to_python(property_map, as_array=True):
  """Convert vertex property map to a python array or list."""
  if as_array:
    array = property_map.fa;
    if array is not None:
       while isinstance(array, gt.PropertyArray):
         array = array.base
       array = array.copy();
       if property_map.value_type() == 'bool':
         array = np.asarray(array, dtype=bool);
       return array;
    else:
      try:
        ndim = len(property_map[property_map.get_graph().vertices().next()]);
      except:
        ndim = 1;
      return property_map.get_2d_array(range(ndim)).T;
  else: # return as list of vertex properties
    return [property_map[v] for v in property_map.get_graph().vertices()];


def edge_property_map_to_python(property_map, as_array=True):
  """Convert edge property map to a python array or list."""
  if as_array:
    array = property_map.fa;
    if array is not None:
       while isinstance(array, gt.PropertyArray):
         array = array.base
       array = array.copy();
       if property_map.value_type() == 'bool':
         array = np.asarray(array, dtype=bool);
       return array;
    else:
      try:
        ndim = len(property_map[property_map.get_graph().edges().next()]);
      except:
        ndim = 1;
      return property_map.get_2d_array(range(ndim)).T;
  else: # return as list of vertex properties
    return [property_map[v] for v in property_map.get_graph().edges()];


def vertex_property_map_from_python(source, graph, dtype = None):
  """Create a vertex property map from a python source."""
  if dtype is None:
    if source is None:
      raise ValueError('Cannot infer dtype for the vertex property');
    else:
      gtype = gtype_from_source(source);
  else:
    gtype = dtype_to_gtype(dtype);
    if source is not None:
      gtype = ndim_to_gtype(ndim_from_source(source), gtype);
  
  if isinstance(source, np.ndarray):  #speed up
    p = graph.base.new_vertex_property(gtype, vals=None);
    set_vertex_property_map(p, source);
  else:
    p = graph.base.new_vertex_property(gtype, vals=source); 
  
  #if shrink_to_fit and source is not None:
  #  p.shrink_to_fit();  
  
  return p;
  
  
def set_vertex_property_map(property_map, source):
  """Set values for vertex property map."""
  if isinstance(source, np.ndarray):
    if source.ndim == 2:
      property_map.set_2d_array(source.T);
    else:
      property_map.fa[:] = source;
  else:
    for e,s in zip(property_map.get_graph().vertices(), source):
      property_map[e] = s;


def edge_property_map_from_python(source, graph, dtype = None):
  """Create a edge property map from a python source."""
  if dtype is None:
    if source is None:
      raise ValueError('Cannot infer dtype for the edge property!');
    else:
      gtype = gtype_from_source(source);
  else:
    gtype = dtype_to_gtype(dtype);
    if source is not None:
      gtype = ndim_to_gtype(ndim_from_source(source), gtype);   
  
  if isinstance(source, np.ndarray): #speed up
    p = graph.base.new_edge_property(gtype, vals=None);
    set_edge_property_map(p, source);
  else:
    p = graph.base.new_edge_property(gtype, vals = source);  
  
  return p;


def set_edge_property_map(property_map, source):
  """Set values for edge property map."""
  if isinstance(source, np.ndarray):
    if source.ndim == 2:
      property_map.set_2d_array(source.T);
    else:
      property_map.fa[:] = source;
  else:
    for e,s in zip(property_map.get_graph().edges(), source):
      property_map[e] = s;  



###############################################################################
### Graph Class
###############################################################################

class Graph(grp.AnnotatedGraph):
  """Graph class to handle graph construction and analysis.
  
  Note
  ----
  This is a interface from ClearMap graphs to graph_tool.
  """
  
  def __init__(self, name = None, n_vertices = None, edges = None, directed = None,
                     vertex_coordinates = None, vertex_radii = None,
                     edge_coordinates = None, edge_radii = None, edge_geometries = None, shape = None,
                     vertex_labels = None, edge_labels = None, annotation = None,
                     base = None, edge_geometry_type = 'graph'):
    
    if base is None:
      base = gt.Graph(directed=directed);
      self.base = base;
    
      # add default graph properties
      self.add_graph_property('shape', None, dtype='object');
      self.add_graph_property('edge_geometry_type', edge_geometry_type, dtype='object');
    
      super(Graph, self).__init__(name=name, n_vertices=n_vertices, edges=edges, directed=directed,
                                  vertex_coordinates=vertex_coordinates, vertex_radii=vertex_radii,
                                  edge_coordinates=edge_coordinates, edge_radii=edge_radii, edge_geometries=edge_geometries, shape=shape,
                                  vertex_labels = None, edge_labels = None, annotation = None);
    else:
      self.base = base;
      super(Graph, self).__init__(name=name);
      
    
  @property
  def base(self):
    return self._base;
  
  @base.setter
  def base(self, value):
    if not isinstance(value, gt.Graph):
      raise ValueError('Base graph not a graph_tool Graph');
    self._base = value;
  
     
  @property
  def directed(self):
    return self._base.is_directed()
  
  @directed.setter
  def directed(self, value):
    self._base.set_directed(value);
  
   
  @property
  def is_view(self):
    return isinstance(self.base, gt.GraphView);
  
  
  ### Vertices
  
  @property
  def n_vertices(self):
    return self._base.num_vertices();
  
  def vertex(self, vertex):
    if isinstance(vertex, gt.Vertex):
      return vertex;
    else:
      return self._base.vertex(vertex);
  
  def first_vertex(self):
    return self._base.vertices().next();
  
  @property
  def vertices(self):
    return [v for v in self._base.vertices()];
  
  def vertex_iterator(self):
    return self._base.vertices();
  
  def vertex_index(self, vertex):
    return int(vertex);
  
  def vertex_indices(self):
    return np.array([int(v) for v in self.vertex_iterator()]);
  
  def add_vertex(self, n_vertices = None, vertex = None):
    if n_vertices is not None:
      self._base.add_vertex(n_vertices);
    elif isinstance(vertex, int):
      self._base.vertex(vertex, add_missing=True);
    #elif isinstance(vertex, gt.Vertex):
    #  v = self._base.add_vertex(1);
    #  v = vertex; #analysis:ignore
    else:
      raise ValueError('Cannot add vertices.');
  
  def remove_vertex(self, vertex):
    self._base.remove_vertex(vertex);
  
  def vertex_property(self, name, vertex = None, as_array = True):
    p = self._base.vertex_properties[name];
    if vertex is not None:
      return p[self.vertex(vertex)];
    else:
      return vertex_property_map_to_python(p, as_array=as_array);
#      if as_array:
#        array = p.fa;
#        if array is not None:
#           while isinstance(array, gt.PropertyArray):
#             array = array.base
#           return array;
#        else:
#          try:
#            ndim = len(p[self.first_vertex()]);
#          except:
#            ndim = 1;
#          return p.get_2d_array(range(ndim)).T;
#      else: # return as list of vertex properties
#        return [p[v] for v in self._base.vertices()];
  
  def vertex_property_map(self, name):
    return self._base.vertex_properties[name];
  
  @property  
  def vertex_properties(self):
    return self._base.vertex_properties.keys();  
  
  def add_vertex_property(self, name, source = None, dtype = None):
    p = vertex_property_map_from_python(source, self, dtype=dtype);
    self._base.vertex_properties[name] = p;
    
#    if dtype is None:
#      if source is None:
#        raise ValueError('Cannot infer dtype for the vertex property');
#      else:
#        gtype = gtype_from_source(source);
#    else:
#      gtype = dtype_to_gtype(dtype);
#      if source is not None:
#        gtype = ndim_to_gtype(ndim_from_source(source), gtype);
#    
#    if isinstance(source, np.ndarray):  #speed up
#      p = self._base.new_vertex_property(gtype, vals=None);
#      self._base.vertex_properties[name] = p;
#      self.set_vertex_property(name, source);
#    else:
#      p = self._base.new_vertex_property(gtype, vals=source); 
#      self._base.vertex_properties[name] = p;

  
  def set_vertex_property(self, name, source, vertex = None):
    if name not in self._base.vertex_properties:
      raise ValueError('Graph has no vertex property with name %s!' % name);    
    p = self._base.vertex_properties[name];
    if vertex is not None:
      p[vertex] = source;
    else:
      set_vertex_property_map(p, source)
#      if isinstance(source, np.ndarray):
#        if source.ndim == 2:
#          p.set_2d_array(source.T);
#        else:
#          p.fa[:] = source;
#      else:
#        for v,s in zip(self._base.vertices(), source):
#          p[v] = s;
    
        
  def define_vertex_property(self, name, source, vertex = None, dtype = None):
    if name in self.vertex_properties:
      self.set_vertex_property(name, source, vertex=vertex);
    else:
      if vertex is None:
        self.add_vertex_property(name, source, dtype=dtype);
      else:
        dtype = gtype_from_source(source) if dtype is None else dtype;
        self.add_vertex_property(name, dtype=dtype);
        self.set_vertex_property(name, source, vertex=vertex);

  def remove_vertex_property(self, name):
    if name not in self._base.vertex_properties:
      raise ValueError('Graph has no vertex property with name %s!' % name);
    del self._base.vertex_properties[name];
  
  
  def vertex_degrees(self):
    return self._base.get_out_degrees(self._base.get_vertices());
  
  def vertex_degree(self, index):
    return self._base.get_out_degrees([index])[0];    
  
  def vertex_out_degrees(self):
    return self._base.get_out_degrees(self._base.get_vertices());
  
  def vertex_out_degree(self, index):
    return self._base.get_out_degrees([index])[0];    
  
  def vertex_in_degrees(self):
    return self._base.get_in_degrees(self._base.get_vertices());
  
  def vertex_in_degree(self, index):
    return self._base.get_in_degrees([index])[0];    
  
  def vertex_neighbours(self, index):
    return self._base.get_out_neighbours(index);
  
  def vertex_out_neighbours(self, index):
    return self._base.get_out_neighbours(index);  
  
  def vertex_in_neighbours(self, index):
    return self._base.get_in_neighbours(index);
  

  ### Edges

  @property
  def n_edges(self):
    return self._base.num_edges();

  def edge(self, edge):
    if isinstance(edge, gt.Edge):
      return edge;
    elif isinstance(edge, tuple):
      return self._base.edge(*edge);
    elif isinstance(edge, int):
      return gtu.find_edge(self._base, self._base.edge_index, edge)[0];
    else:
      raise ValueError('Edge specification %r is not valid!' % edge)
  
  def first_edge(self):
    return self._base.edges().next();
  
  def edge_index(self, edge):
    return self._base.edge_index[self.edge(edge)];
  
  def edge_indices(self):
    p = self.base.edge_index;
    return np.array([p[e] for e in self.edge_iterator()], dtype=int);
  
  def add_edge(self, edge):
    if isinstance(edge, tuple):
      self._base.add_edge(*edge);
    else:
      self._base.add_edge_list(edge);
    
  def remove_edge(self, edge):
    edge = self.edge(edge);
    self._base.remove_edge(edge);
  
  @property
  def edges(self):
    return [e for e in self._base.edges()]
  
  def edge_iterator(self):
    return self._base.edges()
  
  def edge_connectivity(self):
    return self._base.get_edges()[:,:2];

  
  def edge_property(self, name, edge = None, as_array = True):
    p = self._base.edge_properties[name];
    if edge is not None:
      return p[self.edge(edge)];
    else:
      return edge_property_map_to_python(p, as_array=True);
#      if as_array:
#        array = p.fa;
#        if array is not None:
#          while isinstance(array, gt.PropertyArray):
#            array = array.base
#          return array;
#        else:
#          try:
#            ndim = len(p[self.first_edge()]);
#          except:
#            ndim = 1;
#          return p.get_2d_array(range(ndim)).T;
#      else:
#       return [p[e] for e in self._base.edges()];
    
  def edge_property_map(self, name):
    return self._base.edge_properties[name];
  
  @property
  def edge_properties(self):
    return self._base.edge_properties.keys();
  
  def add_edge_property(self, name, source = None, dtype = None):
    p = edge_property_map_from_python(source, self);
    self._base.edge_properties[name] = p;    
    
#    if dtype is None:
#      if source is None:
#        raise ValueError('Cannot infer dtype for the edge property!');
#      else:
#        gtype = gtype_from_source(source);
#    else:
#      gtype = dtype_to_gtype(dtype);
#      if source is not None:
#        gtype = ndim_to_gtype(ndim_from_source(source), gtype);   
#    
#    if isinstance(source, np.ndarray): #speed up
#      p = self._base.new_edge_property(gtype, vals=None);
#      self._base.edge_properties[name] = p;
#      self.set_edge_property(name, source);
#    else:
#      p = self._base.new_edge_property(gtype, vals = source);  
#      self._base.edge_properties[name] = p; 
  
  def set_edge_property(self, name, source, edge = None):
    if name not in self._base.edge_properties:
      raise ValueError('Graph has no edge property with name %s!' % name);    
    p = self._base.edge_properties[name];
    if edge is not None:
      p[self.edge(edge)] = source;
    else:
      set_edge_property_map(p, source);
#      if isinstance(source, np.ndarray):
#        if source.ndim == 2:
#          p.set_2d_array(source.T);
#        else:
#          p.fa[:] = source;
#      else:
#        for e,s in zip(self._base.edges(), source):
#          p[e] = s;
        
  def define_edge_property(self, name, source, edge = None, dtype = None):
    if name in self.edge_properties:
      self.set_edge_property(name, source, edge=edge);
    else:
      if edge is None:
        self.add_edge_property(name, source, dtype=dtype);
      else:
        dtype = gtype_from_source(source) if dtype is None else dtype
        self.add_edge_property(name, dtype=dtype);
        self.set_edge_property(name, source, edge=edge);
  
  def remove_edge_property(self, name):
    if name not in self.edge_properties:
      raise ValueError('Graph does not have edge property with name %s!' % name);
    del self._base.edge_properties[name];
    
  
  def vertex_edges(self, vertex):
    return np.array([[int(e.source()), int(e.target())] for e in self.vertex_edges_iterator(vertex)]);
  
  def vertex_out_edges(self, vertex):
    return np.array([[int(e.source()), int(e.target())] for e in self.vertex_out_edges_iterator(vertex)]);
    
  def vertex_in_edges(self, vertex):
    return np.array([[int(e.source()), int(e.target())] for e in self.vertex_in_edges_iterator(vertex)]);
  
  def vertex_edges_iterator(self, vertex):
    return self._base.vertex(vertex).out_edges();
  
  def vertex_out_edges_iterator(self, vertex):
    return self._base.vertex(vertex).out_edges();

  def vertex_in_edges_iterator(self, vertex):
    return self._base.vertex(vertex).in_edges();
  
  
  ### Graph properties
  
  def graph_property(self, name):
    return self._base.graph_properties[name];
  
  def graph_property_map(self, name):
    return self._base.graph_properties[name];
  
  @property
  def graph_properties(self):
    return self._base.graph_properties.keys(); 
  
  def add_graph_property(self, name, source, dtype = None):
    if dtype is None:
      dtype = 'object';
    gtype = dtype_to_gtype(dtype);
    p = self._base.new_graph_property(gtype);
    p.set_value(source);
    self._base.graph_properties[name] = p; 
  
  def set_graph_property(self, name, source):
    if name not in self.graph_properties:
      raise ValueError('Graph has no property named %s!' % name);
    if source is not None:
      self._base.graph_properties[name] = source;
  
  def define_graph_property(self, name, source, dtype = None):
    if name in self.graph_properties:
      self.set_graph_property(name, source);
    else:
      self.add_graph_property(name, source, dtype=dtype);
  
  def remove_graph_property(self, name):
    if name not in self.graph_properties:
      raise ValueError('Graph does not have graph property named %s!' % name);
    del self._base.graph_properties[name];
      
   
  ### Geometry
  @property
  def shape(self):
    """The shape of the space in which the graph is embedded.
    
    Returns
    -------
    shape : tuple of int
      The shape of the graph space.
    """
    return self.graph_property('shape');
  
  @shape.setter  
  def shape(self, value):
    return self.define_graph_property('shape', value);
  
  
  @property
  def ndim(self):
    if self.shape is None:
      return 3;
    else:
      return len(self.shape);
  
  def axis_indices(self, axis = None, as_list = False):
    if axis is None:
      return range(self.ndim);
    axis_to_index = {'x' : 0, 'y' : 1, 'z' : 2};
    if as_list and not isinstance(axis, (tuple, list)):
      axis = [axis];
    if isinstance(axis, (tuple, list)):
      return [axis_to_index[a] if a in axis_to_index.keys() else a for a in axis];
    else:
      return axis_to_index[axis] if axis in axis_to_index.keys() else axis;
  
  @property
  def has_vertex_coordinates(self):
    return 'coordinates' in self.vertex_properties;
  
  def vertex_coordinates(self, vertex = None, axis = None):
    p = self.vertex_property_map('coordinates');
    if vertex is not None:
      coordinates = p[vertex];
      if axis is None:
         return coordinates;
      else:
        indices = self.axis_indices(axis);
        return coordinates[indices];
    else:
      indices = self.axis_indices(axis, as_list=True);
      coordinates = p.get_2d_array(indices);
      if axis is not None and not isinstance(axis, (tuple, list)):
        return coordinates[0];
      else:
        return coordinates.T;
    
  def set_vertex_coordinates(self, coordinates, vertex = None, dtype = float):
      self.define_vertex_property('coordinates', coordinates, vertex=vertex, dtype=dtype);    
      
  def set_vertex_coordinate(self, vertex, coordinate):
      self.define_vertex_property('coordinates', coordinate, vertex=vertex);

  
  @property
  def has_vertex_radii(self):
    return 'radii' in self.vertex_properties;
  
  def vertex_radii(self, vertex = None):
    return self.vertex_property('radii', vertex=vertex);
      
  def set_vertex_radii(self, radii, vertex = None):
    self.define_vertex_property('radii', radii, vertex=vertex);

  def set_vertex_radius(self, vertex, radius):
    self.define_vertex_property('radii', radius, vertex=vertex);    

  @property
  def has_edge_coordinates(self):
    return 'coordinates' in self.edge_properties;
  
  def edge_coordinates(self, edge = None):
    return self.edge_property('coordinates', edge=edge);    
  
  def set_edge_coordaintes(self, coordinates, edge = None):
    self.define_edge_property('coordinates', coordinates, edge=edge)


  @property
  def has_edge_radii(self):
    return 'radii' in self.edge_properties;

  def edge_radii(self, edge = None):
    return self.edge_property('radii', edge=edge);
  
  def set_edge_radii(self, radii, edge = None):
    self.define_edge_property('radii', radii, edge=edge);
  
  
  ### Edge geometry
  @property
  def edge_geometry_type(self):
    """Type for storing edge properties
    
    Returns
    -------
    type : 'graph' or 'edge'
      'graph' : Stores edge coordinates in an graph property array and 
                start end indices in edges.
                
      'edge'  : Stores the edge coordinates in variable length vectors in 
                each edge. 
    """
    return self.graph_property('edge_geometry_type');
  
  @edge_geometry_type.setter
  def edge_geometry_type(self, value):
    self.set_edge_geometry_type(value);
  
  def edge_geometry_property_name(self, name = 'coordinates', prefix = 'edge_geometry'):
    return prefix + '_' + name;
  
  @property
  def edge_geometry_property_names(self):
    prefix = self.edge_geometry_property_name(name = '');
    n_prefix = len(prefix);
    if self.edge_geometry_type == 'graph':
      properties = self.graph_properties;
    else:
      properties = self.edge_properties;
    properties = [p for p in properties if len(p) >= n_prefix and p[:n_prefix] == prefix and p != 'edge_geometry_type'];
    return properties;
  
  def edge_geometry_property(self, name):
    name = self.edge_geometry_property_name(name);
    if self.edge_geometry_type == 'graph':
      return self.graph_property(name)
    else:
      return self.edge_property(name)
  
  @property
  def edge_geometry_properties(self):
    n_prefix = len(self.edge_geometry_property_name(name = ''));
    properties = [p[n_prefix:] for p in self.edge_geometry_property_names];
    return properties;
  
  def has_edge_geometry(self, name = 'coordinates'):
    return self.edge_geometry_property_name(name=name) in self.edge_geometry_property_names;
  
  
  # edge geometry stored at each edge
  def _edge_geometry_scalar_edge(self, name, edge = None):
    name = self.edge_geometry_property_name(name);
    return self.edge_property(name, edge=edge);
        
  def _edge_geometry_vector_edge(self, name, edge = None, reshape = True, ndim = None, as_list = True):
    name = self.edge_geometry_property_name(name);
    geometry = self.edge_property(name, edge=edge);
    if reshape:
      if ndim is None:
        ndim = self.ndim;
      if edge is None:
        geometry = [g.reshape((-1,ndim),order='A') for g in geometry];
        if as_list:
          return geometry;
        else:
          return np.vstack(geometry);
      else:
        return geometry.reshape(-1,ndim);
    else:
      return geometry;   
 
  def _edge_geometry_indices_edge(self):
    lengths = self.edge_geometry_lengths();
    indices = np.cumsum(lengths);
    indices = np.array([np.hstack([0,indices[:-1]]), indices]).T;
    return indices;
  
  def _edge_geometry_edge(self, name, edge = None, reshape = True, ndim = None, as_list = True, return_indices = False):
    if name in ['coordinates', 'mesh']:
      edge_geometry = self._edge_geometry_vector_edge(name, edge=edge, reshape=reshape, ndim=ndim, as_list=as_list);
    else:
      edge_geometry = self._edge_geometry_scalar_edge(name, edge=edge);
    if return_indices:
      indices = self._edge_geometry_indices();
      return edge_geometry, indices
    else:
      return edge_geometry
  
  def _set_edge_geometry_scalar_edge(self, name, scalars, edge = None, dtype = None):
    name = self.edge_geometry_property_name(name);
    self.define_edge_property(name, scalars, edge=edge, dtype=dtype);
  
  def _set_edge_geometry_vector_edge(self, name, vectors, indices = None, edge = None):    
    name = self.edge_geometry_property_name(name);
    if edge is None:
      if indices is None:
        vectors = [v.reshape(-1, order='A') for v in vectors];
      else:
        vectors = [vectors[s:e].reshape(-1, order='A') for s,e in indices];
    self.define_edge_property(name, vectors, edge=edge, dtype='vector<double>');
  
  def _set_edge_geometry_edge(self, name, values, indices = None, edge = None):
     if name in ['coordinates', 'mesh']:
        return self._set_edge_geometry_vector_edge(name, values, indices=indices, edge=edge);
     elif name in ['radii']:
        return self._set_edge_geometry_scalar_edge(name, values, edge=edge);
     else:
        return self._set_edge_geometry_scalar_edge(name, values, edge=edge, dtype=object);    
  
  def _remove_edge_geometry_edge(self, name):
    name = self.edge_geometry_property_name(name);
    self.remove_edge_property(name);
  
  
  #edge geometry data stored in a single array, start,end indices stored in edge  
  def _edge_geometry_indices_name_graph(self, name='indices'):
    return self.edge_geometry_property_name(name);
  
  def _edge_geometry_indices_graph(self, edge = None):
    return self.edge_property(self._edge_geometry_indices_name_graph(), edge=edge);
  
  def _set_edge_geometry_indices_graph(self, indices, edge = None): 
    self.set_edge_property(self._edge_geometry_indices_name_graph(), indices, edge=edge);
  
  def _edge_geometry_graph(self, name, edge = None, return_indices = False, as_list = False):
    name = self.edge_geometry_property_name(name);
    if edge is None:
      values = self.graph_property(name);
      if return_indices or as_list:
        indices = self._edge_geometry_indices_graph();
      if as_list:
        values = [values[start:end] for start,end in indices]; 
      if return_indices:
        return values, indices
      else:
        return values
    else:
      start,end = self._edge_geometry_indices_graph(edge=edge);
      values = self.graph_property(name);
      return values[start:end];
   
  def _set_edge_geometry_graph(self, name, values, indices = None, edge = None):
    if edge is not None:
      raise NotImplementedError("Setting individual edge geometries not implemented for 'graph' mode!")
    if isinstance(values, list):
      if indices is None:
        indices = np.cumsum([len(v) for v in values]);
        indices = np.array([np.hstack([[0],indices[:-1]]), indices], dtype=int).T;
      values = np.vstack(values);
    if indices is not None:
      name_indices = self._edge_geometry_indices_name_graph();
      self.define_edge_property(name_indices, indices, dtype='vector<int64_t>');
    name = self.edge_geometry_property_name(name);
    self.define_graph_property(name, values, dtype='object');
  
  def _remove_edge_geometry_graph(self, name):
    name = self.edge_geometry_property_name(name);
    if name in self.graph_properties:
      self.remove_graph_property(name);
  
  def _remove_edge_geometry_indices_graph(self):
    name = self._edge_geometry_indices_name_graph();
    if name in self.edge_properties:
      self.remove_edge_property(name);
  
  def resize_edge_geometry(self):
    if not self.has_edge_geometry() or self.edge_geometry_type != 'graph':
      return;
    
    #adjust indices
    indices = self._edge_geometry_indices_graph();
    
    indices_new = np.diff(indices, axis=1)[:,0];
    indices_new = np.cumsum(indices_new);
    indices_new = np.array([np.hstack([0, indices_new[:-1]]), indices_new]).T;
    self._set_edge_geometry_indices_graph(indices_new);
    
    #reduce arrays
    n = indices_new[-1,-1];
    for prop_name in self.edge_geometry_property_names:
      prop = self.graph_property(prop_name);
      shape_new = (n,) + prop.shape[1:];
      prop_new = np.zeros(shape_new, prop.dtype);
      for i,j in zip(indices, indices_new):
        si,ei = i;  sj,ej=j;
        prop_new[sj:ej] = prop[si:ei];
      self.set_graph_property(prop_name, prop_new)
  
  
  def edge_geometry(self, name = 'coordinates', edge = None, as_list = True, return_indices = False, reshape = True, ndim = None):
    if self.edge_geometry_type == 'graph':
      return self._edge_geometry_graph(name=name, edge=edge, return_indices=return_indices, as_list=as_list);
    else: # edge geometry type
      return self._edge_geometry_edge(name=name, edge=edge, as_list=as_list, return_indices=return_indices, reshape=reshape, ndim=ndim);
  
  
  def set_edge_geometry(self, name, values, indices = None, edge = None):
    if self.edge_geometry_type == 'graph':
      #if coordinates is not None:
      #  self._set_edge_geometry_graph('coordinates', coordinates, indices=indices, edge=edge);
      #  if indices is not None:
      #    indices = None;
      #if radii is not None:
      #  self._set_edge_geometry_graph('radii', radii, indices=indices, edge=edge);
      #  if indices is not None:
      #    indices = None;
      #if values is not None:
      self._set_edge_geometry_graph(name, values, indices=indices, edge=edge);
    else:
      #if coordinates is not None:
      #  self._set_edge_geometry_edge('coordinates', coordinates, indices=indices, edge=edge);
      #if radii is not None:
      #  self._set_edge_geometry_edge('radii', radii, indices=indices, edge=edge);
      #if values is not None:
      self._set_edge_geometry_edge(name, values, indices=indices, edge=edge);
      
  
  def remove_edge_geometry(self, name = None):
    if name is None:
      if self.edge_geometry_type == 'graph':
        self._remove_edge_geometry_indices_graph();
      name = self.edge_geometry_properties;
    if not isinstance(name, list):
      name = [name];
    for n in name:
      if self.edge_geometry_type == 'graph':
        self._remove_edge_geometry_graph(name=n);
      else:
        self._remove_edge_geometry_edge(name=n);
  
  def edge_geometry_indices(self):
    if self.edge_geometry_type == 'graph':
      return self._edge_geometry_indices_graph();
    else:
      return self._edge_geometry_indices_edge();
  

  def edge_geometry_lengths(self, name = 'coordinates'):
    if self.edge_geometry_type == 'graph':
      indices = self._edge_geometry_indices_graph();
      return np.diff(indices, axis = 1)[:,0];
    else:
      values = self.edge_geometry(name);
      return np.array([len(v) for v in values], dtype = int);

  
  def set_edge_geometry_type(self, edge_geometry_type):
    if edge_geometry_type not in ['graph', 'edge']:
      raise ValueError("Edge geometry %r not 'graph' or 'edge'!" % edge_geometry_type);
    
    if self.edge_geometry_type == edge_geometry_type:
      return;
    else:
      if self.edge_geometry_type == 'graph': # graph -> edge
        #try:
          indices = self._edge_geometry_indices_graph()
          for name in self.edge_geometry_property_names:
            values = self.edge_geometry(name, as_list=False);
            self._remove_edge_geometry_graph(name);
            self._set_edge_geometry_edge(name, values, indices=indices);
          self._remove_edge_geometry_indices_graph();
        #except:
        #  pass
        
      else: # self.edge_geometry_type == 'edge': edge -> graph
        #try:
          for name in self.edge_geometry_property_names:
            values = self.edge_geometry(name);
            self._remove_edge_geometry_edge(name);
            self._set_edge_geometry_graph(name, values);
        #except:
        #  pass
      self.set_graph_property('edge_geometry_type', edge_geometry_type);   
    
  def is_edge_geometry_consistent(self, verbose = False):
    eg, ei = self.edge_geometry(as_list=False, return_indices=True);
    vc = self.vertex_coordinates();
    ec = self.edge_connectivity();
    
    #check edge sources
    check = vc[ec[:,0]] == eg[ei[:,0]];
    if not np.all(check):
      if verbose:
        errors = np.where(check==False)[0];
        print('Found %d errors in edge sources at %r' % (len(errors), errors));
      return False
    
    #check edge targets
    check = vc[ec[:,1]] == eg[ei[:,1]-1];
    if not np.all(check):
      if verbose:
        errors = np.where(check==False)[0];
        print('Found %d errors in edge targets at %r' % (len(errors), errors));
      return False
      
    return True;
  
  
  def edge_geometry_from_edge_property(self, edge_property_name, edge_geometry_name = None): 
    edge_property = self.edge_property(edge_property_name);
    indices = self.edge_geometry_indices();
        
    shape = (len(indices),) + edge_property.shape[1:];
    edge_geometry = np.zeros(shape, dtype=edge_property.dtype);
    for i,e in zip(indices, edge_property):
      si,ei = i;
      edge_geometry[si:ei] = e;
    
    if edge_geometry_name is None:
      edge_geometry_name = edge_property_name;
    
    self.set_edge_geometry(name=edge_geometry_name, values=edge_geometry, indices=indices);
    
   
#  def edge_meshes(self, edge = None):
#    """Returns a mesh triangulation for the geometry of each edge.
#    
#    Note
#    ----
#    This functionality can be used to store geometric information of edges as
#    meshes, e.g. useful for graph rendering.
#    """
#    
#    pass
#
#
#   
#  ### Label
#  
#  def add_label(self, annotation = None, key = 'id', value = 'order'):
#
#    #lbl.AnnotationFile
#    # label points
#    aba = np.array(io.read(annotation), dtype = int);
#    
#    # get vertex coordinates
#    x,y,z = self.vertex_coordinates().T;
#  
#    ids = np.ones(len(x), dtype = bool);
#    for a,s in zip([x,y,z], aba.shape):
#      ids = np.logical_and(ids, a >= 0);
#      ids = np.logical_and(ids, a < s);
#  
#    # label points
#    g_ids = np.zeros(len(x), dtype = int);
#    g_ids[ids] = aba[x[ids],y[ids],z[ids]];
#  
#    if value is not None:
#      id_to_order = lbl.getMap(key = key, value = value)
#      g_order = id_to_order[g_ids];
#    else:
#      value = key;
#    
#    self.add_vertex_property(value, g_order);
#  
#  



  
  
  ### Functionality
  
  def sub_graph(self, vertex_filter = None, edge_filter = None, view = False):
    gv = gt.GraphView(self.base, vfilt=vertex_filter, efilt=edge_filter);
    if view:
      return Graph(base=gv);
    else:    
      g = gt.Graph(gv, prune=True);
      g = Graph(base=g);
      g.resize_edge_geometry();
      return g;
  
  def view(self, vertex_filter = None, edge_filter = None):
    return gt.GraphView(self.base, vfilt=vertex_filter, efilt=edge_filter);
  
  
  def remove_self_loops(self):
    gt.stats.remove_self_loops(self.base);
    
  def remove_isolated_vertices(self):
    non_isolated = self.vertex_degrees() > 0;
    new_graph = self.sub_graph(vertex_filter=non_isolated);
    self._base = new_graph._base;
  
  
  def label_components(self, return_vertex_counts = False):
    components, vertex_counts = gtt.label_components(self.base);
    components = np.array(components.a);
    if return_vertex_counts:
      return components, vertex_counts;
    else:
      return components;
    
  def largest_component(self, view = False):
    components, counts = self.label_components(return_vertex_counts=True);
    i = np.argmax(counts);
    vertex_filter = components == i;
    return self.sub_graph(vertex_filter=vertex_filter, view=view);
  
  def vertex_coloring(self):
    colors = gtt.sequential_vertex_coloring(self.base);
    colors = vertex_property_map_to_python(colors);
    return colors;
  
  def edge_target_label(self, vertex_label, as_array = True):
    if isinstance(vertex_label, str):
      vertex_label = self.vertex_property(vertex_label);
    if not isinstance(vertex_label, gt.PropertyMap):
      vertex_label = vertex_property_map_from_python(vertex_label, self);
    et = gt.edge_endpoint_property(self.base, vertex_label, endpoint='target');
    return edge_property_map_to_python(et, as_array=as_array);
  
  def edge_source_label(self, vertex_label, as_array = True):
    if isinstance(vertex_label, str):
      vertex_label = self.vertex_property(vertex_label);
    if not isinstance(vertex_label, gt.PropertyMap):
      vertex_label = vertex_property_map_from_python(vertex_label, self);
    et = gt.edge_endpoint_property(self.base, vertex_label, endpoint='source');
    return edge_property_map_to_python(et, as_array=as_array);
  
  def remove_isolated_edges(self):
    vertex_degree = self.vertex_degrees();
    vertex_degree = vertex_property_map_from_python(vertex_degree, self);
    es = self.edge_source_label(vertex_degree, as_array=True);
    et = self.edge_target_label(vertex_degree, as_array=True);
    edge_filter = np.logical_not(np.logical_and(es == 1, et == 1));
    new_graph = self.sub_graph(edge_filter=edge_filter);
    self._base = new_graph._base;
    self.remove_isolated_vertices()
    
  def edge_graph(self, return_edge_map = False):
    line_graph, emap = gtg.line_graph(self.base);
    line_graph = Graph(base = line_graph);
    if return_edge_map:
      emap = vertex_property_map_to_python(emap);
      return line_graph, emap
    else:
      return line_graph;
  
  
  ### Binary morpological graph operations
  
  def vertex_propagate(self, label, value, steps = 1):
    if value is not None and not hasattr(value, '__len__'):
      value = [value];
    p = vertex_property_map_from_python(label, self);
    for s in range(steps):
      gt.infect_vertex_property(self.base, p, vals=value);
    label = vertex_property_map_to_python(p);
    return label;  
  
  def vertex_dilate_binary(self, label, steps = 1):
    return self.vertex_propagate(label, value=True, steps=steps);
    
  def vertex_erode_binary(self, label, steps = 1):
    return self.vertex_propagate(label, value=False, steps=steps);  
  
  def vertex_open_binary(self, label, steps = 1):
    label = self.vertex_erode_binary(label, steps=steps);
    return self.vertex_dilate_binary(label, steps=steps);
  
  def vertex_close_binary(self, label, steps = 1):
    label = self.vertex_dilate_binary(label, steps=steps);
    return self.vertex_erode_binary(label, steps=steps);
  
  def expand_vertex_filter(self, vertex_filter, steps = 1):
    return self.vertex_dilate_binary(vertex_filter, steps=steps);  
  
  
  def edge_propagate(self, label, value, steps = 1):
    label = np.array(label);
    if steps is None:
      return label;
    for s in range(steps):
      edges = label == value;
      ec = self.edge_connectivity();
      ec = ec[edges];
      vertices = np.unique(ec);
      for v in vertices:
        for e in self.vertex_edges_iterator(v):
          i = self.edge_index(e);
          label[i] = value;
    return label; 
  
  def edge_dilate_binary(self, label, steps = 1):
    return self.edge_propagate(label, value=True, steps=steps);
    
  def edge_erode_binary(self, label, steps = 1):
    return self.edge_propagate(label, value=False, steps=steps);  
  
  def edge_open_binary(self, label, steps = 1):
    label = self.edge_erode_binary(label, steps=steps);
    return self.edge_dilate_binary(label, steps=steps);
  
  def edge_close_binary(self, label, steps = 1):
    label = self.edge_dilate_binary(label, steps=steps);
    return self.edge_erode_binary(label, steps=steps);
  
  
  def edge_to_vertex_label(self, edge_label, method='max', as_array = True):
    if isinstance(edge_label, str):
      edge_label = self.edge_property(edge_label);
    if not isinstance(edge_label, gt.PropertyMap):
      edge_label = edge_property_map_from_python(edge_label, self);
    vertex_label = gt.incident_edges_op(self.base, 'in', method, edge_label);
    return vertex_property_map_to_python(vertex_label, as_array=as_array);
  
  def edge_to_vertex_label_or(self, edge_label):
    label = np.zeros(self.n_vertices, dtype=edge_label.dtype);
    ec = self.edge_connectivity();
    #label[ec[:,0]] = edge_label;
    #label[ec[:,1]] = np.logical_or(edge_label, label[ec[:,1]]);
    ids = np.unique(ec[edge_label].flatten());
    label[ids] = True;
    return label;
  
  def vertex_to_edge_label(self, vertex_label, method = None):    
    label = np.zeros(self.n_edges, dtype=vertex_label.dtype);
    ec = self.edge_connectivity();
    
    if method is None:
      if vertex_label.dtype==bool:
        label = np.mean([vertex_label[ec[:,0]], vertex_label[ec[:,1]]], axis = 0) == 1;
      else:
        label = np.mean([vertex_label[ec[:,0]], vertex_label[ec[:,1]]], axis = 0);
    else:
      label = method(vertex_label[ec[:,0]], vertex_label[ec[:,1]]);
    
    return label


  ### Geometric manipulation
  
  def sub_slice(self, slicing, view = False, coordinates = None):
    valid = self.sub_slice_vertex_filter(slicing, coordinates=coordinates);
    return self.sub_graph(vertex_filter=valid, view=view);   
  
  def sub_slice_vertex_filter(self, slicing, coordinates = None):
    import ClearMap.IO.IO as io
    slicing = io.slc.unpack_slicing(slicing, self.ndim);
    valid = np.ones(self.n_vertices, dtype=bool);
    if coordinates is None:
      coordinates = self.vertex_coordinates();
    elif isinstance(coordinates, str):
      coordinates = self.vertex_property(coordinates); 
    for d,s in enumerate(slicing):
      if isinstance(s, slice):
        if s.start is not None:
          valid = np.logical_and(valid, s.start <= coordinates[:,d]);
        if s.stop is not None:
          valid = np.logical_and(valid, coordinates[:,d] < s.stop);
      elif isinstance(s, int):
        valid = np.logical_and(valid, coordinates[:,d] == s);
      else:
        raise ValueError('Invalid slicing %r in dimension %d for sub slicing the graph' % (s,d));
    return valid;

  def sub_slice_edge_filter(self, slicing, coordinates = None):
    import ClearMap.IO.IO as io
    slicing = io.slc.unpack_slicing(slicing, self.ndim);
    valid = np.ones(self.n_edges, dtype=bool);
    if coordinates is None:
      coordinates = self.edge_coordinates();
    elif isinstance(coordinates, str):
      coordinates = self.edge_property(coordinates); 
    for d,s in enumerate(slicing):
      if isinstance(s, slice):
        if s.start is not None:
          valid = np.logical_and(valid, s.start <= coordinates[:,d]);
        if s.stop is not None:
          valid = np.logical_and(valid, coordinates[:,d] < s.stop);
      elif isinstance(s, int):
        valid = np.logical_and(valid, coordinates[:,d] == s);
      else:
        raise ValueError('Invalid slicing %r in dimension %d for sub slicing the graph' % (s,d));
    return valid;

  
  def transform_properties(self, transformation, 
                                 vertex_properties = None, 
                                 edge_properties = None,
                                 edge_geometry_properties = None,
                                 verbose = False):
    
    if vertex_properties is None:
      vertex_properties = {};
    if isinstance(vertex_properties,list):
      vertex_properties = {n : n for n in vertex_properties}
    
    if edge_properties is None:
      edge_properties = {};
    if isinstance(edge_properties,list):
      edge_properties = {n : n for n in edge_properties}
    
    if edge_geometry_properties is None:
      edge_geometry_properties = {};
    if isinstance(edge_geometry_properties,list):
      edge_geometry_properties = {n : n for n in edge_geometry_properties}
    
    for p in vertex_properties.keys():
      if p in self.vertex_properties:
        if verbose:
          print('Transforming vertex property: %s -> %s' % (p, vertex_properties[p]));
        values = self.vertex_property(p);
        values = transformation(values);
        self.define_vertex_property(vertex_properties[p], values);
        
    for p in edge_properties.keys():
      if p in self.edge_properties:
        if verbose:
          print('Transforming edge property: %s -> %s' % (p, edge_properties[p]));
        values = self.edge_property(p);
        values = transformation(values);
        self.define_edge_property(edge_properties[p], values);
        
    as_list = self.edge_geometry_type != 'graph';
    for p in edge_geometry_properties.keys():
      if p in self.edge_geometry_properties:
        if verbose:
          print('Transforming edge geometry: %s -> %s' % (p, edge_geometry_properties[p]));
        values = self.edge_geometry(p, as_list=as_list);
        if as_list:
          values = [transformation(v) for v in values];
        else:
          values = transformation(values);
        self.set_edge_geometry(edge_geometry_properties[p], values=values);

  
  
  
  ### Annotation

  def vertex_annotation(self, vertex = None):
   return self.vertex_property('annotation', vertex=vertex);
  
  def set_vertex_annotation(self, annotation, vertex = None, dtype = 'int32'):
      self.define_vertex_property('annotation', annotation, vertex=vertex, dtype=dtype);    
  
  def edge_annotation(self, edge = None):
    return self.edge_property('annotation', edge=edge);
  
  def set_edge_annotation(self, annotation, edge = None, dtype = 'int32'):
    self.define_edge_property('annotation', annotation, edge=edge, dtype=dtype);    
  
  def annotate_properties(self, annotation,  
                                vertex_properties = None, 
                                edge_properties = None,
                                edge_geometry_properties = None):
    self.transform_properties(annotation, 
                              vertex_properties=vertex_properties,
                              edge_properties=edge_properties, 
                              edge_geometry_properties=edge_geometry_properties);
  
  ### Generic
    
  def info(self):
    print(self.__str__());
    self._base.list_properties();
    
  def save(self, filename):
    self._base.save(filename);
    
  def load(self, filename):
    self._base = gt.load_graph(filename);
  
  def copy(self):
    return Graph(name = copy.copy(self.name), base = self.base.copy())
  

def load(filename):
  g = gt.load_graph(filename);
  return Graph(base = g);

def save(filename, graph):
  graph.save(filename);


###############################################################################
### Tests
###############################################################################

def _test():
  import numpy as np
  import ClearMap.Analysis.Graphs.GraphGt as ggt
  
  from importlib import reload
  reload(ggt)
  
  g = ggt.Graph('test');
  
  g.add_vertex(10);
  
  el = [[1,3],[2,5],[6,7],[7,9]];
  g.add_edge(el)
  
  print(g)
  coords = np.random.rand(10,3);
  g.set_vertex_coordinates(coords)
  
  g.vertex_coordinates()
  
  
  #edge geometry
  elen = [3,4,5,6];
  geometry = [np.random.rand(l,3) for l in elen]
  
  g.set_edge_geometry(geometry)
  
  g.edge_geometry()
  
  
  g.add_edge_property('test', [3,4,5,6]);
  
  g2 = ggt.Graph('test2');
  g2.add_vertex(10);
  g2.add_edge([[1,3],[2,5],[6,7],[7,9]]);
  g2.edge_geometry_type = 'edge'
  
  elen = [3,4,5,6];
  geometry = [np.random.rand(l,3) for l in elen]
  g2.set_edge_geometry(geometry)
  
  g2.edge_geometry()
  
  
  # graph properties
  reload(ggt)
  g = ggt.Graph('test'); 
  g.add_vertex(10);
  g.add_edge([[1,3],[2,5],[6,7],[7,9]])
  
  #scalar vertex property
  g.add_vertex_property('test', np.arange(g.n_vertices));  
  print(g.vertex_property('test') == np.arange(g.n_vertices))
  
  #vector vertex property
  x = np.random.rand(g.n_vertices, 5);
  g.add_vertex_property('vector', x);  
  print(np.all(g.vertex_property('vector') == x)) 
  
  #vector vertex property with different lengths
  y = [np.arange(i) for i in range(g.n_vertices)]
  g.define_vertex_property('list', y);
  z = g.vertex_property('list', as_array=False)  
  print(z == y)
  
  #edge properties
  x = 10 * np.arange(g.n_edges);
  g.add_edge_property('test', x)
  g.edge_property('test') == x
  
  g.info()
  
  
  #filtering / subgraphs
  vfilter = [True] * 5 + [False] * 5;
  s = g.sub_graph(vertex_filter = vfilter);

  p = s.vertex_property_map('test')
  print(p.a)
  
  p = s.edge_property_map('test')
  print(p.a)
  
  print(s.vertex_property('list', as_array=False))
  
  #views
  vfilter = [False] * 5 + [True] * 5;
  v = g.sub_graph(vertex_filter = vfilter, view=True);
  print(v.edge_property('test'))
  print(v.vertex_property('list', as_array=False))
  
  # sub-graphs and edge geometry
  reload(ggt)
  
  g = ggt.Graph('edge_geometry');
  g.add_vertex(5);
  g.add_edge([[0,1],[1,2],[2,3],[3,4]]);
  
  geometry = [np.random.rand(l,3) for l in [3,4,5,6]]  
  g.set_edge_geometry(geometry)
  
  
  #note te difference !
  s = g.sub_graph(vertex_filter = [False]*2 + [True]*3)
  s.edge_geometry()
  s.edge_geometry(as_list=False)
  s._edge_geometry_indices_graph()
  
  v = g.sub_graph(vertex_filter = [False]*2 + [True]*3, view=True)
  v.edge_geometry()
  v.edge_geometry(as_list=False)
  v._edge_geometry_indices_graph()
  
  # vertex expansion
  reload(ggt)
  g = ggt.Graph();
  g.add_vertex(5);
  g.add_edge([[0,1],[1,2],[2,3],[3,4]]);
  vertex_filter = np.array([False, False, True, False, False], dtype = 'bool')
  expanded = g.expand_vertex_filter(vertex_filter, steps=1)  
  print(expanded)
  
  # test large arrays in graphs
  import numpy as np;
  import ClearMap.IO.IO as io
  import ClearMap.Analysis.Graphs.GraphGt as ggt
  reload(ggt)
  
  g = ggt.Graph('test');
  g.add_vertex(10);
  
  x = np.zeros(2147483648, dtype='uint8')
  
  g.define_graph_property('test', x);
  g.save('test.gt')
  #this gives an error when using unmodified graph_tool
  
  del g
  del x
  import ClearMap.Analysis.Graphs.GraphGt as ggt
  f = ggt.load('test.gt')
  f.info()
  print(f.graph_property('test').shape  )

  io.delete_file('test.gt')
  
