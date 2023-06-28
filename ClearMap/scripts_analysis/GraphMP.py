import numpy as np

import graph_tool as gt
import ClearMap.Analysis.Graphs.GraphGt as ggt

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
            source = io.as_source(source);
            ndim = source.ndim;
        except:
            ndim = 0;
    return ndim;



def edge_property_map_to_python(property_map, graph, as_array=True):
  print('edge_property_map_to_python')
  """Convert edge property map to a python array or list."""
  if as_array:
    print(as_array)
    array = property_map.fa;
    if array is not None:
        print(array)
        # while isinstance(array, gt.PropertyArray):
        array = array.base
        return array.copy();
    else:
      print('None')
      try:
        print('try')
        ndim = len(property_map[graph.edges().next()]);
      except:
        ndim = 1;
      return property_map.get_2d_array(range(ndim)).T;
  else: # return as list of vertex properties
    print('as_array')
    return [property_map[v] for v in property_map.get_graph().edges()];


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


def edge_property(graph, name, edge=None, as_array=True):
    p = graph._base.edge_properties[name];
    if edge is not None:
        return p[graph.edge(edge)];
    else:
        return edge_property_map_to_python(p, graph, as_array=True);


def edge_geometry_property_name( name = 'coordinates', prefix = 'edge_geometry'):
    print('edge_geometry_property_name')
    return prefix + '_' + name;

def _edge_geometry_indices_name_graph(graph, name='indices'):
    print('_edge_geometry_indices_name_graph')
    return edge_geometry_property_name(name);

def _edge_geometry_indices_graph(graph, edge = None):
    print('_edge_geometry_indices_graph')
    return edge_property(graph,_edge_geometry_indices_name_graph(graph), edge=edge);

def resize_edge_geometry(graph):
    if not graph.has_edge_geometry() or graph.edge_geometry_type != 'graph':
        return;

    # adjust indices
    print('adjust indices')
    indices = _edge_geometry_indices_graph(graph);
    print('adjust diff')
    indices_new = np.diff(indices, axis=1)[: ,0];
    print('adjust cumsum')
    indices_new = np.cumsum(indices_new);
    print('adjust hstack')
    indices_new = np.array([np.hstack([0, indices_new[:-1]]), indices_new]).T;
    graph._set_edge_geometry_indices_graph(indices_new);
    print('adjust reduce')
    # reduce arrays
    n = indices_new[-1 ,-1];
    for prop_name in graph.edge_geometry_properties:
        prop = graph.graph_property(prop_name);
        shape_new = (n,) + prop.shape[1:];
        prop_new = np.zeros(shape_new, prop.dtype);
        for i, j in zip(indices, indices_new):
            si, ei = i;
            sj, ej = j;
            prop_new[sj:ej] = prop[si:ei];
        graph.set_graph_property(prop_name, prop_new)



def set_graph_property(graph, name, source):
    if name not in graph.graph_properties:
        raise ValueError('Graph has no property named %s!' % name);
    if source is not None:
        graph.base.graph_properties[name] = source;

def dtype_to_gtype(dtype):
    """Convert a data type to a graph_tool data type."""

    if isinstance(dtype, str):
        name = dtype;
    else:
        dtype = np.dtype(dtype);
        name = dtype.name;

    alias = {'float64': 'double',
             'float32': 'double',
             'int64': 'int64_t',
             'int32': 'int32_t',
             'uint64': 'int64_t',
             'uint32': 'int64_t'};
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



def vertex_property_map_from_python(source, graph, dtype=bool):
    """Create a vertex property map from a python source."""

    gtype = dtype_to_gtype(dtype);
    if source is not None:
        gtype = ndim_to_gtype(ndim_from_source(source), gtype);

    if isinstance(source, np.ndarray):  # speed up
        p = graph.base.new_vertex_property(gtype, vals=None);
        set_vertex_property_map(p, source);
    else:
        p = graph.base.new_vertex_property(gtype, vals=source);

        # if shrink_to_fit and source is not None:
    #  p.shrink_to_fit();

    return p;

def graphsub(graph, vertex_filter=None, edge_filter=None):
    print('set vertex prop graph MP')
    g=graph.copy()
    g.base.set_vertex_filter(vertex_property_map_from_python(vertex_filter ,g, dtype=bool))
    # self.base.set_edge_filter(edge_property_map_from_python(edge_filter, self, dtype=bool))
    print('create graph')
    g=gt.Graph(g.base, prune=True)
    print('create graph')
    g = ggt.Graph(base=g);
    try:
        resize_edge_geometry(g);
    except(IndexError):
        print(IndexError)
    return g;
