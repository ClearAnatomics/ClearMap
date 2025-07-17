import graph_tool as gt
import numpy as np


def dtype_to_gtype(dtype):
    """Convert a data type to a graph_tool data type."""

    if isinstance(dtype, str):
        name = dtype
    else:
        dtype = np.dtype(dtype)
        name = dtype.name

    alias = {
        'float64': 'double',
        'float32': 'double',
        'int64': 'int64_t',
        'int32': 'int32_t',
        'int16': 'int16_t',
        'uint64': 'int64_t',
        'uint32': 'int64_t'
    }
    if name in alias:
        name = alias[name]

    return gt._type_alias(name)


def ndim_to_gtype(ndim, gtype):
    """Convert a scalar gtype to a vector one if necessary."""
    if len(gtype) >= 6 and gtype[:6] == 'vector':
        return gtype
    if ndim == 2:
        gtype = "vector<%s>" % gtype
    elif ndim > 2:
        raise ValueError('Data for vertex properties can only be 1 or 2d!')

    return gtype


def ndim_from_source(source):
    """Determines the dimension of a source appropriate for graph_tool."""
    if isinstance(source, (list, tuple)):
        if len(source) > 0:
            ndim = 2
        else:
            ndim = 0
    elif hasattr(source, 'dtype') and hasattr(source, 'ndim'):
        ndim = source.ndim
    else:
        try:
            import ClearMap.IO.IO as io
            source = io.as_source(source)
            ndim = source.ndim
        except:
            ndim = 0
    return ndim


def gtype_from_source(source, vectorize=True, graph_property=False):
    """Determines the graph_tool data type from a data source."""
    if isinstance(source, (list, tuple)):
        if len(source) > 0:
            source = source[0]
            dtype = np.asarray(source).dtype
            gtype = dtype_to_gtype(dtype)
            ndim = 2
        else:
            gtype = dtype_to_gtype('object')
            ndim = 0
    elif hasattr(source, 'dtype') and hasattr(source, 'ndim'):
        dtype = source.dtype
        gtype = dtype_to_gtype(dtype)
        ndim = source.ndim
    else:
        try:
            import ClearMap.IO.IO as io
            source = io.as_source(source)
            dtype = source.dtype
            gtype = dtype_to_gtype(dtype)
            ndim = source.ndim
        except:
            gtype = dtype_to_gtype('object')
            ndim = 0

    if vectorize:
        gtype = ndim_to_gtype(ndim, gtype)

    return gtype


def vertex_property_map_to_python(property_map, as_array=True):
    """Convert vertex property map to a python array or list."""
    if as_array:
        array = property_map.fa
        if array is not None:
            while isinstance(array, gt.PropertyArray):
                array = array.base
            array = array.copy()
            if property_map.value_type() == 'bool':
                array = np.asarray(array, dtype=bool)
            return array
        else:
            try:
                ndim = len(property_map[property_map.get_graph().vertices().next()])
            except:
                ndim = 1
            return property_map.get_2d_array(range(ndim)).T
    else:  # return as list of vertex properties
        return [property_map[v] for v in property_map.get_graph().vertices()]


def edge_property_map_to_python(property_map, as_array=True):
    """Convert edge property map to a python array or list."""
    if as_array:
        array = property_map.fa
        if array is not None:
            while isinstance(array, gt.PropertyArray):
                array = array.base
            array = array.copy()
            if property_map.value_type() == 'bool':
                array = np.asarray(array, dtype=bool)
            return array
        else:
            try:
                ndim = len(property_map[property_map.get_graph().edges().next()])
            except:
                ndim = 1
            return property_map.get_2d_array(range(ndim)).T
    else:  # return as list of vertex properties
        return [property_map[v] for v in property_map.get_graph().edges()]


def vertex_property_map_from_python(source, graph, dtype=None):
    """Create a vertex property map from a python source."""
    if dtype is None:
        if source is None:
            raise ValueError('Cannot infer dtype for the vertex property')
        else:
            gtype = gtype_from_source(source)
    else:
        gtype = dtype_to_gtype(dtype)
        if source is not None:
            gtype = ndim_to_gtype(ndim_from_source(source), gtype)

    if isinstance(source, np.ndarray):  # speed up
        p = graph.base.new_vertex_property(gtype, vals=None)
        set_vertex_property_map(p, source)
    else:
        p = graph.base.new_vertex_property(gtype, vals=source)

    # if shrink_to_fit and source is not None:
    #     p.shrink_to_fit()

    return p


def set_vertex_property_map(property_map, source):
    """Set values for vertex property map."""
    if isinstance(source, np.ndarray):
        if source.ndim == 2:
            property_map.set_2d_array(source.T)
        else:
            property_map.fa[:] = source
    else:
        for e, s in zip(property_map.get_graph().vertices(), source):
            property_map[e] = s


def edge_property_map_from_python(source, graph, dtype=None):  # TODO: see if we can factor with get_vertex_property_map_from_python
    """Create an edge property map from a python source."""
    if dtype is None:
        if source is None:
            raise ValueError('Cannot infer dtype for the edge property!')
        else:
            gtype = gtype_from_source(source)
    else:
        gtype = dtype_to_gtype(dtype)
        if source is not None:
            gtype = ndim_to_gtype(ndim_from_source(source), gtype)

    if isinstance(source, np.ndarray):  # speed up
        p = graph.base.new_edge_property(gtype, vals=None)
        set_edge_property_map(p, source)
    else:
        p = graph.base.new_edge_property(gtype, vals=source)

    return p


def set_edge_property_map(property_map, source):  # TODO: see if we can factor with set_vertex_property_map
    """Set values for edge property map."""
    if isinstance(source, np.ndarray):
        if source.ndim == 2:
            property_map.set_2d_array(source.T)
        else:
            property_map.fa[:] = source
    else:
        for e, s in zip(property_map.get_graph().edges(), source):
            property_map[e] = s
