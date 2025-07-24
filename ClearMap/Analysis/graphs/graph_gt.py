# -*- coding: utf-8 -*-
"""
GraphGt
=======

Module provides basic Graph interface to the
`graph_tool <https://graph-tool.skewed.de>`_ library.
"""
__author__ = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__ = 'https://idisco.info'
__download__ = 'https://www.github.com/ChristophKirst/ClearMap2'

import copy
import numbers

import numpy as np

import graph_tool as gt
import graph_tool.util as gtu
import graph_tool.topology as gtt
import graph_tool.generation as gtg

# fix graph tool saving / loading for very large arrays
import ClearMap.Analysis.graphs.graph as grp
from ClearMap.Analysis.graphs.type_conversions import dtype_to_gtype, gtype_from_source, vertex_property_map_to_python, \
  edge_property_map_to_python, vertex_property_map_from_python, set_vertex_property_map, edge_property_map_from_python, \
  set_edge_property_map
from ClearMap.Analysis.graphs.utils import pickler, unpickler, edges_to_vertices

from ClearMap.Utils.array_utils import remap_array_ranges

LARGE_GRAPH_N_EDGES_THRESHOLD = 10 ** 7

gt.gt_io.clean_picklers()
gt.gt_io.libgraph_tool_core.set_pickler(pickler)
gt.gt_io.libgraph_tool_core.set_unpickler(unpickler)


class Graph(grp.AnnotatedGraph):
    """Graph class to handle graph construction and analysis.

    Note
    ----
    This is an interface from ClearMap graphs to graph_tool.
    """
    DEFAULT_N_DIMS = 3

    def __init__(self, name=None, n_vertices=None, edges=None, directed=None,
                 vertex_coordinates=None, vertex_radii=None,
                 edge_coordinates=None, edge_radii=None, edge_geometries=None, shape=None,
                 vertex_labels=None, edge_labels=None, annotation=None,
                 base=None, edge_geometry_type='graph'):

        self.path = ''
        if base is None:
            base = gt.Graph(directed=directed)
            self.base = base

            # add default graph properties
            self.add_graph_property('shape', None, dtype='object')
            self.add_graph_property('edge_geometry_type', edge_geometry_type, dtype='object')

            super(Graph, self).__init__(name=name, n_vertices=n_vertices, edges=edges, directed=directed,
                                        vertex_coordinates=vertex_coordinates, vertex_radii=vertex_radii,
                                        edge_coordinates=edge_coordinates, edge_radii=edge_radii,
                                        edge_geometries=edge_geometries, shape=shape,
                                        vertex_labels=None, edge_labels=None, annotation=None)
        else:
            self.base = base
            super(Graph, self).__init__(name=name)

    @property
    def base(self):
        return self._base

    @base.setter
    def base(self, value):
        if not isinstance(value, gt.Graph):
            raise ValueError('Base graph not a graph_tool Graph')
        self._base = value

    @property
    def directed(self):
        return self._base.is_directed()

    @directed.setter
    def directed(self, value):
        self._base.set_directed(value)

    @property
    def is_view(self):
        return isinstance(self.base, gt.GraphView)

    # ## Vertices
    @property
    def n_vertices(self):
        return self._base.num_vertices()

    def vertex(self, vertex):
        if isinstance(vertex, gt.Vertex):
            return vertex
        else:
            return self._base.vertex(vertex)

    def first_vertex(self):
        return self._base.vertices().next()

    @property
    def vertices(self):
        return list(self.base.vertices())

    def vertex_iterator(self):
        return self._base.vertices()

    def vertex_index(self, vertex):
        return int(vertex)

    def vertex_indices(self):
        return  self._base.get_vertices()

    def add_vertex(self, n_vertices=None, vertex=None):
        if n_vertices is not None:
            self._base.add_vertex(n_vertices)
        elif isinstance(vertex, int):
            self._base.vertex(vertex, add_missing=True)
        # elif isinstance(vertex, gt.Vertex):
        #     v = self._base.add_vertex(1)
        #     v = vertex  #analysis:ignore
        else:
            raise ValueError('Cannot add vertices.')

    def remove_vertex(self, vertex):
        self._base.remove_vertex(vertex)

    def vertex_property(self, name, vertex=None, as_array=True):
        """

        .. warning::
            This risks creating a copy of the vertex property map if `as_array` is True.

        Parameters
        ----------
        name
        vertex
        as_array

        Returns
        -------

        """
        try:
            v_prop = self._base.vertex_properties[name]
        except KeyError as err:
            raise KeyError(f'Graph has no vertex property with name "{name}" '
                           f'Possible vertex properties are: {list(self.vertex_properties)};'
                           f'{err}')
        if vertex is not None:
            return v_prop[self.vertex(vertex)]
        else:
            return vertex_property_map_to_python(v_prop, as_array=as_array)

    def vertex_property_map(self, name):
        return self._base.vertex_properties[name]

    @property
    def vertex_properties(self):
        return self._base.vertex_properties.keys()

    def add_vertex_property(self, name, source=None, dtype=None):
        v_prop = vertex_property_map_from_python(source, self, dtype=dtype)
        self._base.vertex_properties[name] = v_prop

    def set_vertex_property(self, name, source, vertex=None):
        if name not in self._base.vertex_properties:
            raise ValueError(f'Graph has no vertex property with name {name}!')
        v_prop = self._base.vertex_properties[name]
        if vertex is not None:
            v_prop[vertex] = source
        else:
            set_vertex_property_map(v_prop, source)

    def define_vertex_property(self, name, source, vertex=None, dtype=None):
        if name in self.vertex_properties:
            self.set_vertex_property(name, source, vertex=vertex)
        else:
            if vertex is None:
                self.add_vertex_property(name, source, dtype=dtype)
            else:
                dtype = gtype_from_source(source) if dtype is None else dtype
                self.add_vertex_property(name, dtype=dtype)
                self.set_vertex_property(name, source, vertex=vertex)

    def remove_vertex_property(self, name):
        if name not in self._base.vertex_properties:
            raise ValueError(f'Graph has no vertex property with name {name}!')
        del self._base.vertex_properties[name]

    def vertex_degrees(self):
        return self._base.get_out_degrees(self._base.get_vertices())

    def vertex_degree(self, index):
        return self._base.get_out_degrees([index])[0]

    def vertex_out_degrees(self):
        return self._base.get_out_degrees(self._base.get_vertices())

    def vertex_out_degree(self, index):
        return self._base.get_out_degrees([index])[0]

    def vertex_in_degrees(self):
        return self._base.get_in_degrees(self._base.get_vertices())

    def vertex_in_degree(self, index):
        return self._base.get_in_degrees([index])[0]

    def vertex_neighbours(self, index):
        return self._base.get_out_neighbours(index)

    def vertex_out_neighbours(self, index):
        return self._base.get_out_neighbours(index)

    def vertex_in_neighbours(self, index):
        return self._base.get_in_neighbours(index)

    # ## Edges
    @property
    def n_edges(self):
        return self._base.num_edges()

    def edge(self, edge):
        if isinstance(edge, gt.Edge):
            return edge
        elif isinstance(edge, tuple):  # FIXME: what about list ?
            return self._base.edge(*edge)
        elif isinstance(edge, int):
            return gtu.find_edge(self._base, self._base.edge_index, edge)[0]
        elif isinstance(edge, list) and len(edge) == 2:
            return self.edge(tuple(edge))
        elif isinstance(edge, np.ndarray) and edge.shape == (2,):
            return self.edge(tuple(edge))
        else:
            raise ValueError(f'Edge specification {edge} is not valid!')

    def first_edge(self):
        return self._base.edges().next()

    def edge_index(self, edge):
        return self._base.edge_index[self.edge(edge)]

    def edge_indices(self):  # TODO: explain what this does
        table = self._base.get_edges(eprops=[self._base.edge_index])
        return table[:, 2]

    def add_edge(self, edge):
        if isinstance(edge, tuple):
            self._base.add_edge(*edge)
        else:
            self._base.add_edge_list(edge)

    def remove_edge(self, edge):
        edge = self.edge(edge)
        self._base.remove_edge(edge)

    @property
    def edges(self):
        return list(self._base.edges())

    def get_edges(self, eprops=[]):
        return self._base.get_edges(eprops=eprops)

    def edge_iterator(self):
        return self._base.edges()

    def edge_connectivity(self, order='src_vertex'):  # PERFORMANCE: see if better to cache property and invalidate when edeges added or removed
        if order == 'src_vertex':
            return self._base.get_edges()[:, :2]
        elif order == 'eid':
            table = self._base.get_edges([self._base.edge_index])
            # Sort by the eid (third column) → insertion / ID order
            return table[np.argsort(table[:, 2])][:, :2]
        else:
            raise NotImplementedError(f'Invalid edge connectivity order "{order}"! ')

    def edge_property(self, name, edge=None, as_array=True):
        e_prop = self._base.edge_properties[name]
        if edge is not None:
            return e_prop[self.edge(edge)]
        else:
            return edge_property_map_to_python(e_prop, as_array=True)

    def edge_property_map(self, name):
        return self._base.edge_properties[name]

    @property
    def edge_properties(self):
        return self._base.edge_properties.keys()

    def add_edge_property(self, name, source=None, dtype=None):
        p = edge_property_map_from_python(source, self)
        self._base.edge_properties[name] = p

    def set_edge_property(self, name, source, edge=None):
        if name not in self._base.edge_properties:
            raise ValueError(f'Graph has no edge property with name {name}!')
        p = self._base.edge_properties[name]
        if edge is not None:
            p[self.edge(edge)] = source
        else:
            set_edge_property_map(p, source)

    def define_edge_property(self, name, source, edge=None, dtype=None):
        if name in self.edge_properties:
            self.set_edge_property(name, source, edge=edge)
        else:
            if edge is None:
                self.add_edge_property(name, source, dtype=dtype)
            else:
                dtype = gtype_from_source(source) if dtype is None else dtype
                self.add_edge_property(name, dtype=dtype)
                self.set_edge_property(name, source, edge=edge)

    def remove_edge_property(self, name):
        if name not in self.edge_properties:
            raise ValueError(f'Graph does not have edge property with name {name}!')
        del self._base.edge_properties[name]

    def vertex_edges(self, vertex):
        return edges_to_vertices(self.vertex_edges_iterator(vertex))

    def vertex_out_edges(self, vertex):
        return edges_to_vertices(self.vertex_out_edges_iterator(vertex))

    def vertex_in_edges(self, vertex):
        return edges_to_vertices(self.vertex_in_edges_iterator(vertex))

    def vertex_edges_iterator(self, vertex):
        return self._base.vertex(vertex).out_edges()

    def vertex_out_edges_iterator(self, vertex):
        return self._base.vertex(vertex).out_edges()

    def vertex_in_edges_iterator(self, vertex):
        return self._base.vertex(vertex).in_edges()

    # ## Graph properties
    def graph_property(self, name):
        return self._base.graph_properties[name]

    def graph_property_map(self, name):
        return self._base.graph_properties[name]

    @property
    def graph_properties(self):
        return self._base.graph_properties.keys()

    def add_graph_property(self, name, source, dtype=None):
        if dtype is None:
            dtype = 'object'
        gtype = dtype_to_gtype(dtype)
        g_prop = self._base.new_graph_property(gtype)
        g_prop.set_value(source)
        self._base.graph_properties[name] = g_prop

    def set_graph_property(self, name, source):
        if name not in self.graph_properties:
            raise ValueError(f'Graph has no property named "{name}"')
        if source is not None:
            self._base.graph_properties[name] = source

    def define_graph_property(self, name, source, dtype=None):
        if name in self.graph_properties:
            self.set_graph_property(name, source)
        else:
            self.add_graph_property(name, source, dtype=dtype)

    def remove_graph_property(self, name):
        if name not in self.graph_properties:
            raise ValueError(f'Graph does not have graph property named {name}!')
        del self._base.graph_properties[name]

    # ## Geometry
    @property
    def shape(self):
        """The shape of the space in which the graph is embedded.

        Returns
        -------
        shape : tuple of int
          The shape of the graph space.
        """
        return self.graph_property('shape')

    @shape.setter
    def shape(self, value):
        self.define_graph_property('shape', value)

    @property
    def ndim(self):
        if self.shape is None:
            return Graph.DEFAULT_N_DIMS
        else:
            return len(self.shape)

    def axis_indices(self, axis=None, as_list=False):
        if axis is None:
            return range(self.ndim)
        axis_to_index = {k: i for i, k in enumerate('xyz')}
        if as_list and not isinstance(axis, (tuple, list)):
            axis = [axis]
        if isinstance(axis, (tuple, list)):
            return [axis_to_index[a] if a in axis_to_index.keys() else a for a in axis]
        else:
            return axis_to_index[axis] if axis in axis_to_index.keys() else axis

    @property
    def has_vertex_coordinates(self):
        return 'coordinates' in self.vertex_properties

    def vertex_coordinates(self, vertex=None, axis=None):
        p = self.vertex_property_map('coordinates')
        if vertex is not None:
            coordinates = p[vertex]
            if axis is None:
                return coordinates
            else:
                indices = self.axis_indices(axis)
                return coordinates[indices]
        else:
            indices = self.axis_indices(axis, as_list=True)
            coordinates = p.get_2d_array(indices)
            if axis is not None and not isinstance(axis, (tuple, list)):
                return coordinates[0]
            else:
                return coordinates.T

    # FIXME: not very useful
    def set_vertex_coordinates(self, coordinates, vertex=None, dtype=float):
        self.define_vertex_property('coordinates', coordinates, vertex=vertex, dtype=dtype)

    # def set_vertex_coordinate(self, vertex, coordinate):
    #     self.define_vertex_property('coordinates', coordinate, vertex=vertex)

    @property
    def has_vertex_radii(self):
        return 'radii' in self.vertex_properties

    def vertex_radii(self, vertex=None):
        return self.vertex_property('radii', vertex=vertex)

    def set_vertex_radii(self, radii, vertex=None):
        self.define_vertex_property('radii', radii, vertex=vertex)

    def set_vertex_radius(self, vertex, radius):
        self.define_vertex_property('radii', radius, vertex=vertex)

    @property
    def has_edge_coordinates(self):
        return 'coordinates' in self.edge_properties

    def edge_coordinates(self, edge=None):
        return self.edge_property('coordinates', edge=edge)

    def set_edge_coordinates(self, coordinates, edge=None):
        self.define_edge_property('coordinates', coordinates, edge=edge)

    @property
    def has_edge_radii(self):
        return 'radii' in self.edge_properties

    def edge_radii(self, edge=None):
        return self.edge_property('radii', edge=edge)

    def set_edge_radii(self, radii, edge=None):
        self.define_edge_property('radii', radii, edge=edge)

    # ## Edge geometry
    @property
    def edge_geometry_type(self):
        """Type for storing edge properties

        Returns
        -------
        type : 'graph' or 'edge'
          'graph' : Stores edge coordinates in a graph property array and
                    start end indices in edges.

          'edge'  : Stores the edge coordinates in variable length vectors in
                    each edge.
        """
        return self.graph_property('edge_geometry_type')

    @edge_geometry_type.setter
    def edge_geometry_type(self, value):
        self.set_edge_geometry_type(value)

    def edge_geometry_property_name(self, name='coordinates', prefix='edge_geometry'):
        return f'{prefix}_{name}'

    @property
    def edge_geometry_property_names(self):
        prefix = self.edge_geometry_property_name(name='')
        if self.edge_geometry_type == 'graph':
            properties = self.graph_properties
        else:
            properties = self.edge_properties
        # return the graph properties that are arrays with n_pixels elements
        return [p for p in properties if p.startswith(prefix) and p != 'edge_geometry_type']

    def edge_geometry_property(self, name):
        name = self.edge_geometry_property_name(name)
        if self.edge_geometry_type == 'graph':
            return self.graph_property(name)
        else:
            return self.edge_property(name)

    @property
    def edge_geometry_properties(self):
        prefix_len = len(self.edge_geometry_property_name(name=''))
        properties = [p[prefix_len:] for p in self.edge_geometry_property_names]
        return properties

    def has_edge_geometry(self, name='coordinates'):
        # FIXME: should probably check for indices too
        return self.edge_geometry_property_name(name=name) in self.edge_geometry_property_names

    # edge geometry stored at each edge
    def _edge_geometry_scalar_edge(self, name, edge=None):
        name = self.edge_geometry_property_name(name)
        return self.edge_property(name, edge=edge)

    def _edge_geometry_vector_edge(self, name, edge=None, reshape=True, ndim=None, as_list=True):
        name = self.edge_geometry_property_name(name)
        geometry = self.edge_property(name, edge=edge)
        if reshape:
            if ndim is None:
                ndim = self.ndim
            if edge is None:
                geometry = [g.reshape((-1, ndim), order='A') for g in geometry]
                if as_list:
                    return geometry
                else:
                    return np.vstack(geometry)
            else:
                return geometry.reshape(-1, ndim)
        else:
            return geometry

    def _edge_geometry_indices_edge(self):
        lengths = self.edge_geometry_lengths()
        indices = np.cumsum(lengths)
        indices = np.array([np.hstack([0, indices[:-1]]), indices]).T
        return indices

    def _edge_geometry_edge(self, name, edge=None, reshape=True, ndim=None, as_list=True, return_indices=False):
        if name in ['coordinates', 'mesh']:
            edge_geometry = self._edge_geometry_vector_edge(name, edge=edge, reshape=reshape, ndim=ndim, as_list=as_list)
        else:
            edge_geometry = self._edge_geometry_scalar_edge(name, edge=edge)
        if return_indices:
            indices = self._edge_geometry_indices_edge()
            return edge_geometry, indices
        else:
            return edge_geometry

    def _set_edge_geometry_scalar_edge(self, name, scalars, edge=None, dtype=None):
        name = self.edge_geometry_property_name(name)
        self.define_edge_property(name, scalars, edge=edge, dtype=dtype)

    def _set_edge_geometry_vector_edge(self, name, vectors, indices=None, edge=None):
        name = self.edge_geometry_property_name(name)
        if edge is None:
            if indices is None:
                vectors = [v.reshape(-1, order='A') for v in vectors]
            else:
                vectors = [vectors[s:e].reshape(-1, order='A') for s, e in indices]
        self.define_edge_property(name, vectors, edge=edge, dtype='vector<double>')

    def _set_edge_geometry_edge(self, name, values, indices=None, edge=None):
        if name in ['coordinates', 'mesh']:
            return self._set_edge_geometry_vector_edge(name, values, indices=indices, edge=edge)
        elif name in ['radii']:
            return self._set_edge_geometry_scalar_edge(name, values, edge=edge)
        else:
            return self._set_edge_geometry_scalar_edge(name, values, edge=edge, dtype=object)

    def _remove_edge_geometry_edge(self, name):
        name = self.edge_geometry_property_name(name)
        self.remove_edge_property(name)

    # EDGE GEOMETRY GRAPH
    # edge geometry data stored in a single array, start,end indices stored in edge
    def _edge_geometry_indices_name_graph(self, name='indices'):
        return self.edge_geometry_property_name(name)

    def _edge_geometry_indices_graph(self, edge=None):
        return self.edge_property(self._edge_geometry_indices_name_graph(), edge=edge)

    def _set_edge_geometry_indices_graph(self, indices, edge=None):
        self.set_edge_property(self._edge_geometry_indices_name_graph(), indices, edge=edge)

    def _edge_geometry_graph(self, name, edge=None, return_indices=False, as_list=False):
        name = self.edge_geometry_property_name(name)
        if edge is None:
            values = self.graph_property(name)
            if return_indices or as_list:
                indices = self._edge_geometry_indices_graph()
            if as_list:
                values = [values[start:end] for start, end in indices]
            if return_indices:
                return values, indices
            else:
                return values
        else:
            start, end = self._edge_geometry_indices_graph(edge=edge)
            values = self.graph_property(name)
            return values[start:end]

    def _set_edge_geometry_graph(self, name, values, indices=None, edge=None):
        if edge is not None:
            raise NotImplementedError("Setting individual edge geometries not implemented for 'graph' mode!")
        if isinstance(values, list):
            if indices is None:
                indices = np.cumsum([len(v) for v in values])
                indices = np.array([np.hstack([[0], indices[:-1]]), indices], dtype=int).T.astype(np.int64)

            first_val = values[0]
            # flatten the list of arrays so that it is a single array (indexed by the indices)
            if first_val.ndim == 1:  # if the first value is a 1D array, we assume all values are 1D arrays
                values = np.concatenate(values)
            else: # if the first value is a 2D array, we assume all values are 2D arrays
                values = np.vstack(values)
        # if values.ndim == 1:  # if the values are a 1D array, we assume they are scalars
        #     prop_dtype = f'vector<{gtype_from_source(values)}>'  # Store as *vector* to store as a single cpp array
        if values.ndim <= 2:  # can't store 2d as vector, so store as object (pickled np.ndarray)
            prop_dtype = 'object'  # Store as *object* to store as a single ndarray
        else:
            raise ValueError(f'Edge geometry values must be 1D or 2D arrays, got {values.ndim}D array!')
        if indices is not None:  # FIXME: see if we should update the indices in case exists but mismatched
            self.define_edge_geometry_indices_graph(indices)
        egp_name = self.edge_geometry_property_name(name)
        self.define_graph_property(egp_name, values, dtype=prop_dtype)

    def define_edge_geometry_indices_graph(self, indices):
        name_indices = self._edge_geometry_indices_name_graph()
        # if name_indices not in self.edge_properties:  # Set if missing
        self.define_edge_property(name_indices, indices, dtype='vector<int64_t>')

    def _remove_edge_geometry_graph(self, name):
        name = self.edge_geometry_property_name(name)
        if name in self.graph_properties:
            self.remove_graph_property(name)

    def _remove_edge_geometry_indices_graph(self):
        name = self._edge_geometry_indices_name_graph()
        if name in self.edge_properties:
            self.remove_edge_property(name)

    def prune_edge_geometry(self):
        """
        Remove the unused edge geometries from the graph.
        This computes the new indices and remaps the edge geometry properties to the new indices.
        """
        if not self.has_edge_geometry() or self.edge_geometry_type != 'graph':
            return

        # adjust indices
        indices = self._edge_geometry_indices_graph()

        indices_new = np.diff(indices, axis=1)[:, 0]
        indices_new = np.cumsum(indices_new)
        indices_new = np.array([np.hstack([0, indices_new[:-1]]), indices_new]).T
        self._set_edge_geometry_indices_graph(indices_new)

        self._remap_edge_geometry_properties(indices, indices_new)

    def remap_edge_geometry_properties(self, new_indices):
        """
        Remap all properties in self.edge_geometry_properties (edge_geometry_<>) to the new indices i.e.,
         copy every edge-geometry_<> array so old ranges → new ranges

        Parameters
        ----------
        new_indices : np.ndarray
            The new indices to remap the edge geometry properties to.
        """
        indices = self._edge_geometry_indices_graph()
        self._set_edge_geometry_indices_graph(new_indices)
        self._remap_edge_geometry_properties(indices, new_indices)

    def _remap_edge_geometry_properties(self, indices, indices_new):
        """
        Remap all properties in self.edge_geometry_properties (edge_geometry_<>) to
         the new indices i.e., copy every edge-geometry_<> array so old ranges → new ranges

        For example, if for the edge_geometry_coordinates, which has a shape of
        (n_voxels, 3), the indices would be (n_edges, 2) and the indices_new would be
        (n_edges_new, 2). The function would then remap the coordinates from the old
        indices to the new indices like this:
        for i in range(indices.shape[0]):
          prop_new[indices_new[i, 0]:indices_new[i, 1]] = prop[indices[i, 0]:indices[i, 1]]


        Parameters
        ----------
        indices
        indices_new

        Returns
        -------

        """
        n = indices_new[-1, -1]
        for prop_name in self.edge_geometry_property_names:
            prop = self.graph_property(prop_name)
            shape_new = (n,) + prop.shape[1:]
            prop_new = np.zeros(shape_new, dtype=prop.dtype)  # init empty, will then be filled with remapped values
            prop_new = remap_array_ranges(prop, prop_new, indices, indices_new)
            self.set_graph_property(prop_name, prop_new)

    def edge_geometry(self, name='coordinates', edge=None, as_list=True, return_indices=False, reshape=True, ndim=None):
        if self.edge_geometry_type == 'graph':
            return self._edge_geometry_graph(name=name, edge=edge, return_indices=return_indices, as_list=as_list)
        else:  # edge geometry type
            return self._edge_geometry_edge(name=name, edge=edge, return_indices=return_indices, as_list=as_list, reshape=reshape, ndim=ndim)

    def set_edge_geometry(self, name, values, indices=None, edge=None):
        """
        Set the given edge geometry property for the graph.

        .. warning::
            edge is not supported for 'graph' edge geometry type.

        Parameters
        ----------
        name: str
            The name of the original vertex or edge property to set as edge geometry
            As an edge_geometry property, the name will be prefixed (typically with 'edge_geometry_').
        values: List or np.ndarray
            The values to set as edge geometry.
        indices: np.ndarray
            How to slice the values to map to edges.
        edge: gt.Edge or int, optional
            The edge to set the geometry for. If None, the geometry is set for all edges.
            If the edge_geometry_type is 'graph', this parameter is not supported.
        """
        if self.edge_geometry_type == 'graph':
            self._set_edge_geometry_graph(name, values, indices=indices, edge=edge)
        else:
            self._set_edge_geometry_edge(name, values, indices=indices, edge=edge)

    def remove_edge_geometry(self, name=None):
        if name is None:
            if self.edge_geometry_type == 'graph':
                self._remove_edge_geometry_indices_graph()
            name = self.edge_geometry_properties
        if not isinstance(name, list):
            name = [name]
        for n in name:
            if self.edge_geometry_type == 'graph':
                self._remove_edge_geometry_graph(name=n)
            else:
                self._remove_edge_geometry_edge(name=n)

    def set_edge_geometry_vertex_properties(self, original_graph, edge_geometry_vertex_properties,
                                            branch_indices, indices):
        """
        Set the edge geometry properties from the vertex properties of the original graph.

        .. note::
            The property is processed only if it exists in the original graph and is not
            already set as an edge geometry property in the current graph.

        Parameters
        ----------
        original_graph
        edge_geometry_vertex_properties
        branch_indices
        indices

        Returns
        -------

        """
        for v_prop_name in edge_geometry_vertex_properties:
            if v_prop_name in original_graph.vertex_properties:
                # If already exists
                if self.edge_geometry_property_name(v_prop_name) in self.edge_geometry_property_names:
                    continue  # Skip if already set, it will be handled by the edge aggregation
                v_prop = original_graph.vertex_property(v_prop_name)[branch_indices]
                self.set_edge_geometry(name=v_prop_name, values=v_prop, indices=indices)

    def set_edge_geometry_edge_properties(self, original_graph, edge_geometry_edge_properties, indices, edge_to_edge_map):
        first_edge = edge_to_edge_map[0][0]
        for e_prop_name in edge_geometry_edge_properties:
            if e_prop_name in original_graph.edge_properties:
                # If already exists
                if self.edge_geometry_property_name(f'edge_{e_prop_name}') in self.edge_geometry_property_names:
                    continue  # Skip if already set, it will be handled by the edge aggregation
                values = original_graph.edge_property_map(e_prop_name)
                # there is one fewer edge than vertices in each reduced edge !
                if isinstance(first_edge, gt.Edge):
                    values = [[values[e] for e in edges + [edges[-1]]] for edges in edge_to_edge_map]
                elif isinstance(first_edge, (numbers.Integral, numbers.Real)):
                    values = values.fa
                    values = [values[np.append(edges, edges[-1])] for edges in edge_to_edge_map]
                else:
                    raise ValueError(f'Edge type "{type(first_edge)}" not supported for edge geometry!')
                # it seems that we repeat the last edge to have the same number of edges as vertices ?
                self.set_edge_geometry(name=f'edge_{e_prop_name}', values=values, indices=indices)

    def edge_geometry_indices(self):
        if self.edge_geometry_type == 'graph':
            return self._edge_geometry_indices_graph()
        else:
            return self._edge_geometry_indices_edge()

    def edge_geometry_lengths(self, name='coordinates'):
        if self.edge_geometry_type == 'graph':
            indices = self._edge_geometry_indices_graph()
            return np.diff(indices, axis=1)[:, 0]
        else:
            values = self.edge_geometry(name)
            return np.array([len(v) for v in values], dtype=int)

    def set_edge_geometry_type(self, edge_geometry_type):
        if edge_geometry_type not in ['graph', 'edge']:
            raise ValueError(f"Edge geometry must be 'graph' or 'edge', got '{edge_geometry_type}'!")

        if self.edge_geometry_type == edge_geometry_type:
            return
        else:
            if self.edge_geometry_type == 'graph':  # graph -> edge
                indices = self._edge_geometry_indices_graph()
                for name in self.edge_geometry_property_names:
                    values = self.edge_geometry(name, as_list=False)
                    self._remove_edge_geometry_graph(name)
                    self._set_edge_geometry_edge(name, values, indices=indices)
                self._remove_edge_geometry_indices_graph()
            else:  # self.edge_geometry_type == 'edge': edge -> graph
                for name in self.edge_geometry_property_names:
                    values = self.edge_geometry(name)
                    self._remove_edge_geometry_edge(name)
                    self._set_edge_geometry_graph(name, values)
            self.set_graph_property('edge_geometry_type', edge_geometry_type)

    def is_edge_geometry_consistent(self, verbose=False):
        eg, ei = self.edge_geometry(as_list=False, return_indices=True)
        vc = self.vertex_coordinates()
        ec = self.edge_connectivity()

        # check edge sources
        check = vc[ec[:, 0]] == eg[ei[:, 0]]
        if not np.all(check):
            if verbose:
                errors = np.where(check == False)[0]
                print(f'Found {len(errors)} errors in edge sources at {errors}')
            return False

        # check edge targets
        check = vc[ec[:, 1]] == eg[ei[:, 1]-1]
        if not np.all(check):
            if verbose:
                errors = np.where(check == False)[0]
                print(f'Found {len(errors)} errors in edge targets at {errors}')
            return False

        return True

    def edge_geometry_from_edge_property(self, edge_property_name, edge_geometry_name=None):
        edge_property = self.edge_property(edge_property_name)
        indices = self.edge_geometry_indices()

        shape = (len(indices),) + edge_property.shape[1:]
        edge_geometry = np.zeros(shape, dtype=edge_property.dtype)
        for i, e in zip(indices, edge_property):
            si, ei = i
            edge_geometry[si:ei] = e

        if edge_geometry_name is None:
            edge_geometry_name = edge_property_name

        self.set_edge_geometry(name=edge_geometry_name, values=edge_geometry, indices=indices)

    # def edge_meshes(self, edge=None):
    #     """Returns a mesh triangulation for the geometry of each edge.
    #
    #     Note
    #     ----
    #     This functionality can be used to store geometric information of edges as
    #     meshes, e.g. useful for graph rendering.
    #     """
    #
    #     pass

    # ## Label

    # def add_label(self, annotation=None, key='id', value='order'):
    #
    #     # lbl.AnnotationFile
    #     # label points
    #     aba = np.array(io.read(annotation), dtype=int)
    #
    #     # get vertex coordinates
    #     x,y,z = self.vertex_coordinates().T
    #
    #     ids = np.ones(len(x), dtype = bool)
    #     for a,s in zip([x,y,z], aba.shape):
    #         ids = np.logical_and(ids, a >= 0)
    #         ids = np.logical_and(ids, a < s)
    #
    #     # label points
    #     g_ids = np.zeros(len(x), dtype=int)
    #     g_ids[ids] = aba[x[ids], y[ids], z[ids]]
    #
    #     if value is not None:
    #         id_to_order = lbl.getMap(key=key, value=value)
    #         g_order = id_to_order[g_ids]
    #     else:
    #         value = key
    #
    #     self.add_vertex_property(value, g_order)

    # ## Functionality
    def sub_graph(self, vertex_filter=None, edge_filter=None, view=False):
        gv = gt.GraphView(self.base, vfilt=vertex_filter, efilt=edge_filter)
        if view:
            return Graph(base=gv)
        else:
            g = gt.Graph(gv, prune=True)
            g = Graph(base=g)
            if g.n_edges and self.has_edge_geometry():
                g.prune_edge_geometry()
            else:
                g.remove_edge_geometry()  # Drop geometries (to be readded) if we filter all edges (typically for reduc)
                # TODO: see if we keep the egeom props but with shape([n], 0)
            return g


    def view(self, vertex_filter=None, edge_filter=None):
        return gt.GraphView(self.base, vfilt=vertex_filter, efilt=edge_filter)

    def remove_self_loops(self):
        gt.stats.remove_self_loops(self.base)

    def remove_isolated_vertices(self):
        non_isolated = self.vertex_degrees() > 0
        new_graph = self.sub_graph(vertex_filter=non_isolated)
        self._base = new_graph._base

    def label_components(self, return_vertex_counts=False):
        components, vertex_counts = gtt.label_components(self.base)
        components = np.array(components.a)
        if return_vertex_counts:
            return components, vertex_counts
        else:
            return components

    def largest_component(self, view=False):
        components, counts = self.label_components(return_vertex_counts=True)
        i = np.argmax(counts)
        vertex_filter = components == i
        return self.sub_graph(vertex_filter=vertex_filter, view=view)

    def vertex_coloring(self):
        colors = gtt.sequential_vertex_coloring(self.base)
        colors = vertex_property_map_to_python(colors)
        return colors

    def edge_target_label(self, vertex_label, as_array=True):
        if isinstance(vertex_label, str):
            vertex_label = self.vertex_property(vertex_label)
        if not isinstance(vertex_label, gt.PropertyMap):
            vertex_label = vertex_property_map_from_python(vertex_label, self)
        et = gt.edge_endpoint_property(self.base, vertex_label, endpoint='target')
        return edge_property_map_to_python(et, as_array=as_array)

    def edge_source_label(self, vertex_label, as_array=True):
        if isinstance(vertex_label, str):
            vertex_label = self.vertex_property(vertex_label)
        if not isinstance(vertex_label, gt.PropertyMap):
            vertex_label = vertex_property_map_from_python(vertex_label, self)
        et = gt.edge_endpoint_property(self.base, vertex_label, endpoint='source')
        return edge_property_map_to_python(et, as_array=as_array)

    def remove_isolated_edges(self):
        vertex_degree = self.vertex_degrees()
        vertex_degree = vertex_property_map_from_python(vertex_degree, self)
        es = self.edge_source_label(vertex_degree, as_array=True)
        et = self.edge_target_label(vertex_degree, as_array=True)
        edge_filter = np.logical_not(np.logical_and(es == 1, et == 1))
        new_graph = self.sub_graph(edge_filter=edge_filter)
        self._base = new_graph._base
        self.remove_isolated_vertices()

    def edge_graph(self, return_edge_map=False):
        line_graph, emap = gtg.line_graph(self.base)
        line_graph = Graph(base=line_graph)
        if return_edge_map:
            emap = vertex_property_map_to_python(emap)
            return line_graph, emap
        else:
            return line_graph

    # ## Binary morphological graph operations

    def vertex_propagate(self, label, value, steps=1):
        if value is not None and not hasattr(value, '__len__'):
            value = [value]
        p = vertex_property_map_from_python(label, self)
        for s in range(steps):
            gt.infect_vertex_property(self.base, p, vals=value)
        label = vertex_property_map_to_python(p)
        return label

    def vertex_dilate_binary(self, label, steps=1):
        return self.vertex_propagate(label, value=True, steps=steps)

    def vertex_erode_binary(self, label, steps=1):
        return self.vertex_propagate(label, value=False, steps=steps)

    def vertex_open_binary(self, label, steps=1):
        label = self.vertex_erode_binary(label, steps=steps)
        return self.vertex_dilate_binary(label, steps=steps)

    def vertex_close_binary(self, label, steps=1):
        label = self.vertex_dilate_binary(label, steps=steps)
        return self.vertex_erode_binary(label, steps=steps)

    def expand_vertex_filter(self, vertex_filter, steps=1):
        return self.vertex_dilate_binary(vertex_filter, steps=steps)

    def edge_propagate(self, label, value, steps=1):
        label = np.array(label)
        if steps is None:
            return label
        for s in range(steps):
            edges = label == value
            ec = self.edge_connectivity()
            ec = ec[edges]
            vertices = np.unique(ec)
            for v in vertices:
                for e in self.vertex_edges_iterator(v):
                    i = self.edge_index(e)
                    label[i] = value
        return label

    def edge_dilate_binary(self, label, steps=1):
        return self.edge_propagate(label, value=True, steps=steps)

    def edge_erode_binary(self, label, steps=1):
        return self.edge_propagate(label, value=False, steps=steps)

    def edge_open_binary(self, label, steps=1):
        label = self.edge_erode_binary(label, steps=steps)
        return self.edge_dilate_binary(label, steps=steps)

    def edge_close_binary(self, label, steps=1):
        label = self.edge_dilate_binary(label, steps=steps)
        return self.edge_erode_binary(label, steps=steps)

    def edge_to_vertex_label(self, edge_label, method='max', as_array=True):
        if isinstance(edge_label, str):
            edge_label = self.edge_property(edge_label)
        if not isinstance(edge_label, gt.PropertyMap):
            edge_label = edge_property_map_from_python(edge_label, self)
        vertex_label = gt.incident_edges_op(self.base, 'in', method, edge_label)
        return vertex_property_map_to_python(vertex_label, as_array=as_array)

    def edge_to_vertex_label_or(self, edge_label):
        label = np.zeros(self.n_vertices, dtype=edge_label.dtype)
        ec = self.edge_connectivity()
        # label[ec[:,0]] = edge_label
        # label[ec[:,1]] = np.logical_or(edge_label, label[ec[:,1]])
        ids = np.unique(ec[edge_label].flatten())
        label[ids] = True
        return label

    def vertex_to_edge_label(self, vertex_label, method=None):
        label = np.zeros(self.n_edges, dtype=vertex_label.dtype)
        ec = self.edge_connectivity()

        if method is None:
            if vertex_label.dtype == bool:
                label = np.mean([vertex_label[ec[:, 0]], vertex_label[ec[:, 1]]], axis=0) == 1
            else:
                label = np.mean([vertex_label[ec[:, 0]], vertex_label[ec[:, 1]]], axis=0)
        else:
            label = method(vertex_label[ec[:, 0]], vertex_label[ec[:, 1]])

        return label

    # ## Geometric manipulation
    def sub_slice(self, slicing, view=False, coordinates=None):
        valid = self.sub_slice_vertex_filter(slicing, coordinates=coordinates)
        return self.sub_graph(vertex_filter=valid, view=view)

    def _slice_coordinates(self, coordinates, slicing, size):
        import ClearMap.IO.IO as io
        slicing = io.slc.unpack_slicing(slicing, self.ndim)
        valid = np.ones(size, dtype=bool)
        for d, s in enumerate(slicing):
            if isinstance(s, slice):
                if s.start is not None:
                    valid = np.logical_and(valid, s.start <= coordinates[:, d])
                if s.stop is not None:
                    valid = np.logical_and(valid, coordinates[:, d] < s.stop)
            elif isinstance(s, int):
                valid = np.logical_and(valid, coordinates[:, d] == s)
            else:
                raise ValueError(f'Invalid slicing {s} in dimension {d} for sub slicing the graph')
        return valid

    def sub_slice_vertex_filter(self, slicing, coordinates=None):
        if coordinates is None:
            coordinates = self.vertex_coordinates()
        elif isinstance(coordinates, str):
            coordinates = self.vertex_property(coordinates)
        valid = self._slice_coordinates(coordinates, slicing, size=self.n_vertices)
        return valid

    def sub_slice_edge_filter(self, slicing, coordinates=None):
        if coordinates is None:
            coordinates = self.edge_coordinates()
        elif isinstance(coordinates, str):
            coordinates = self.edge_property(coordinates)
        valid = self._slice_coordinates(coordinates, slicing, size=self.n_edges)
        return valid

    def transform_properties(self, transformation,
                             vertex_properties=None,
                             edge_properties=None,
                             edge_geometry_properties=None,
                             verbose=False):
        def properties_to_dict(properties):
            if properties is None:
                properties = {}
            if isinstance(properties, list):
                properties = {n: n for n in properties}
            return properties

        vertex_properties = properties_to_dict(vertex_properties)
        edge_properties = properties_to_dict(edge_properties)
        edge_geometry_properties = properties_to_dict(edge_geometry_properties)

        for p in vertex_properties.keys():
            # if p in self.vertex_properties:
            if verbose:
                print(f'Transforming vertex property: {p} -> {vertex_properties[p]}')
            values = self.vertex_property(p)
            values = transformation(values)
            self.define_vertex_property(vertex_properties[p], values)

        for p in edge_properties.keys():
            # if p in self.edge_properties:
            if verbose:
                print(f'Transforming edge property: {p} -> {edge_properties[p]}')
            values = self.edge_property(p)
            values = transformation(values)
            self.define_edge_property(edge_properties[p], values)

        as_list = self.edge_geometry_type != 'graph'
        for p in edge_geometry_properties.keys():
            # if p in self.edge_geometry_properties:
            if verbose:
                print(f'Transforming edge geometry: {p} -> {edge_geometry_properties[p]}')
            values = self.edge_geometry(p, as_list=as_list)
            if as_list:
                values = [transformation(v) for v in values]
            else:
                values = transformation(values)
            self.set_edge_geometry(edge_geometry_properties[p], values=values)

    # ## Annotation

    def vertex_annotation(self, vertex=None):
        return self.vertex_property('annotation', vertex=vertex)

    def set_vertex_annotation(self, annotation, vertex=None, dtype='int32'):
        self.define_vertex_property('annotation', annotation, vertex=vertex, dtype=dtype)

    def edge_annotation(self, edge=None):
        return self.edge_property('annotation', edge=edge)

    def set_edge_annotation(self, annotation, edge=None, dtype='int32'):
        self.define_edge_property('annotation', annotation, edge=edge, dtype=dtype)

    def annotate_properties(self, annotation,
                            vertex_properties=None,
                            edge_properties=None,
                            edge_geometry_properties=None):
        self.transform_properties(annotation,
                                  vertex_properties=vertex_properties,
                                  edge_properties=edge_properties,
                                  edge_geometry_properties=edge_geometry_properties)

    # ## Generic
    def info(self):
        print(self.__str__())
        self._base.list_properties()

    def save(self, filename):
        self._base.save(str(filename))

    def load(self, filename):
        self.path = str(filename)
        self._base = gt.load_graph(self.path)

    def copy(self, from_disk=False, path=''):
        if from_disk:
            return load(path if path else self.path)
        else:
            if self.n_edges <= LARGE_GRAPH_N_EDGES_THRESHOLD:  # Small graph, copy properties
                return Graph(name=copy.copy(self.name), base=gt.Graph(self.base))

            else:  # RAM runs away on direct copy of edge_properties for large graphs
                bare_view = gt.GraphView( self._base, skip_properties=True, skip_vfilt=True, skip_efilt=True)
                # topological copy, no properties
                new_base = gt.Graph(bare_view, prune=(False, False, True))  # keep all V/E
                # vertex properties
                for name, p in self._base.vp.items():
                    new_base.vp[name] = new_base.copy_property(p, g=self._base)

                # graph properties
                for name, p in self._base.gp.items():
                    q = new_base.new_graph_property(p.value_type())
                    q[new_base] = p[self._base]
                    new_base.gp[name] = q

                # edge properties
                for name, p in self._base.ep.items():
                    q = new_base.new_edge_property(p.value_type())
                    q.fa = p.fa.copy()  # one contiguous memcpy
                    new_base.ep[name] = q
                return Graph(name=copy.copy(self.name), base=new_base)


def load(filename):
    g = gt.load_graph(str(filename))
    graph = Graph(base=g)
    graph.path = str(filename)
    return graph


def save(filename, graph):
    graph.save(str(filename))


###############################################################################
# ## Tests
###############################################################################

def _test():
    import numpy as np
    import ClearMap.Analysis.graphs.graph_gt as ggt

    from importlib import reload
    reload(ggt)

    g = ggt.Graph('test')

    g.add_vertex(10)

    el = [[1, 3], [2, 5], [6, 7], [7, 9]]
    g.add_edge(el)

    print(g)
    coords = np.random.rand(10,3)
    g.set_vertex_coordinates(coords)

    g.vertex_coordinates()

    # edge geometry
    elen = [3, 4, 5, 6]
    geometry = [np.random.rand(l, 3) for l in elen]

    g.set_edge_geometry(geometry)

    g.edge_geometry()

    g.add_edge_property('test', [3, 4, 5, 6])

    g2 = ggt.Graph('test2')
    g2.add_vertex(10)
    g2.add_edge([[1, 3], [2, 5], [6, 7], [7, 9]])
    g2.edge_geometry_type = 'edge'

    elen = [3, 4, 5, 6]
    geometry = [np.random.rand(l, 3) for l in elen]
    g2.set_edge_geometry(geometry)

    g2.edge_geometry()

    # graph properties
    reload(ggt)
    g = ggt.Graph('test')
    g.add_vertex(10)
    g.add_edge([[1, 3], [2, 5], [6, 7], [7, 9]])

    # scalar vertex property
    g.add_vertex_property('test', np.arange(g.n_vertices))
    print(g.vertex_property('test') == np.arange(g.n_vertices))

    # vector vertex property
    x = np.random.rand(g.n_vertices, 5)
    g.add_vertex_property('vector', x)
    print(np.all(g.vertex_property('vector') == x))

    # vector vertex property with different lengths
    y = [np.arange(i) for i in range(g.n_vertices)]
    g.define_vertex_property('list', y)
    z = g.vertex_property('list', as_array=False)
    print(z == y)

    # edge properties
    x = 10 * np.arange(g.n_edges)
    g.add_edge_property('test', x)
    assert g.edge_property('test') == x

    g.info()

    # filtering / sub-graphs
    v_filter = [True] * 5 + [False] * 5
    s = g.sub_graph(vertex_filter=v_filter)

    p = s.vertex_property_map('test')
    print(p.a)

    p = s.edge_property_map('test')
    print(p.a)

    print(s.vertex_property('list', as_array=False))

    # views
    v_filter = [False] * 5 + [True] * 5
    v = g.sub_graph(vertex_filter=v_filter, view=True)
    print(v.edge_property('test'))
    print(v.vertex_property('list', as_array=False))

    # sub-graphs and edge geometry
    reload(ggt)

    g = ggt.Graph('edge_geometry')
    g.add_vertex(5)
    g.add_edge([[0, 1], [1, 2], [2, 3], [3, 4]])

    geometry = [np.random.rand(l, 3) for l in [3, 4, 5, 6]]
    g.set_edge_geometry(geometry)

    # note te difference !
    s = g.sub_graph(vertex_filter=[False]*2 + [True]*3)
    s.edge_geometry()
    s.edge_geometry(as_list=False)
    s._edge_geometry_indices_graph()

    v = g.sub_graph(vertex_filter=[False]*2 + [True]*3, view=True)
    v.edge_geometry()
    v.edge_geometry(as_list=False)
    v._edge_geometry_indices_graph()

    # vertex expansion
    reload(ggt)
    g = ggt.Graph()
    g.add_vertex(5)
    g.add_edge([[0, 1], [1, 2], [2, 3], [3, 4]])
    vertex_filter = np.array([False, False, True, False, False], dtype='bool')
    expanded = g.expand_vertex_filter(vertex_filter, steps=1)
    print(expanded)

    # test large arrays in graphs
    import numpy as np
    import ClearMap.IO.IO as io
    import ClearMap.Analysis.graphs.graph_gt as ggt
    reload(ggt)

    g = ggt.Graph('test')
    g.add_vertex(10)

    x = np.zeros(2147483648, dtype='uint8')

    g.define_graph_property('test', x)
    g.save('test.gt')
    # this gives an error when using unmodified graph_tool

    del g
    del x
    import ClearMap.Analysis.graphs.graph_gt as ggt
    f = ggt.load('test.gt')
    f.info()
    print(f.graph_property('test').shape)

    io.delete_file('test.gt')
