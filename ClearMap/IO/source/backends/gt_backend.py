# -*- coding: utf-8 -*-
"""
GT
==

Backend for graph-tool (.gt) graph files.

Graphs are whole-file sources (see :class:`ClearMap.IO.source.Source.GraphSource`):
loaded lazily, read and written whole, written atomically, and pickled as a
lightweight handle so workers reload from disk instead of receiving the graph.

Partial loads are forwarded to :meth:`Graph.partial_load`, e.g.
``io.read(path, exclude_edge_geometry_properties=True)`` loads the topology and
properties without the (large) edge geometry array.

See also
--------
:mod:`ClearMap.Analysis.graphs`
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'https://idisco.info'
__download__  = 'https://www.github.com/ChristophKirst/ClearMap2'


from ClearMap.Analysis.graphs import graph_gt

from ClearMap.IO.source.Source import GraphSource
from ClearMap.IO.source.protocol import Backend


class GraphGtSource(GraphSource):
    """Graph source backed by a graph-tool (.gt) file."""

    backend = Backend.GT
    _graph_types = (graph_gt.Graph,)

    def __init__(self, location=None, graph=None, name=None, mode=None):
        """GT source class constructor.

        Arguments
        ---------
        location : str or None
            The filename of the graph source.
        graph : Graph or None
            The graph object
        """
        if isinstance(location, graph_gt.Graph):  # legacy: GraphGtSource(graph)
            location, graph = None, location
        super().__init__(location=location, graph=graph, mode=mode, name=name)

    @classmethod
    def _load(cls, location, **kwargs):
        if kwargs:
            return graph_gt.Graph.partial_load(location, **kwargs)
        return graph_gt.Graph.load(location)

    @classmethod
    def _dump(cls, graph, location, **kwargs):
        graph.save(location, **kwargs)

    def scan_properties(self):
        """Property names in the file, by scope, without loading the graph."""
        self._require_existing()
        return graph_gt.Graph.scan_gt_properties(self.location, as_dict=True)

    def copy(self):
        """An in-memory source holding a copy of this graph."""
        return GraphGtSource(graph=self.graph.copy())


# class GraphGtVirtualSource(source_mod.VirtualSource):
#     _real_class = GraphGtSource
#
#     def __init__(self, source=None, location=None, name=None, mode=None):
#         if source is not None and location is None:
#             location = source.location
#         super().__init__(location=location, name=name, mode=mode)
#
#     def as_buffer(self):
#         raise NotImplementedError('Cannot convert virtual graph to buffer')
#
#     @property
#     def graph(self):
#         """The underlying graph.
#
#         Returns
#         -------
#         graph : Graph
#             The underlying graph of this source.
#         """
#         if self._graph is None:
#             self._graph = _graph(self.location)
#         return self._graph
#
#     @graph.setter
#     def graph(self, value):
#         raise NotImplementedError("Cannot set virtual graph")
#
#     @property
#     def shape(self):
#         """The shape of the source.
#
#         Returns
#         -------
#         shape : tuple
#             The shape of the source.
#         """
#         return self.graph.shape
#
#     @shape.setter
#     def shape(self, value):
#         raise NotImplementedError("Cannot set shape of virtual graph")


SOURCE_CLASS = GraphGtSource


###############################################################################
### IO Interface
###############################################################################

def open_ro(source_, **kwargs):
    return GraphGtSource.open_ro(source_, **kwargs)


def read(source_, slicing=None, **kwargs):
    return GraphGtSource.read_graph(source_, slicing=slicing, **kwargs)


def write(sink, data=None, slicing=None, overwrite=True, **kwargs):
    return GraphGtSource.write_graph(sink, data, slicing=slicing, overwrite=overwrite, **kwargs)


def create(location=None, shape=None, dtype=None, order=None,
           mode=None, array=None, as_source=True, **kwargs):
    return GraphGtSource.create_graph(location, shape=shape, dtype=dtype, order=order,
                                      mode=mode, array=array, as_source=as_source, **kwargs)


def edit(source_, **kwargs):
    return GraphGtSource.edit(source_, **kwargs)


def is_graph(source):
    """Checks if this source is a graph source."""
    return isinstance(source, GraphGtSource) or (isinstance(source, str) and source.lower().endswith('.gt'))

###############################################################################
### Tests
###############################################################################

def test():    
    """Test GT module"""
    import os
    from ClearMap.Analysis.graphs import graph_gt
    from ClearMap.IO.source.backends.gt_backend import GraphGtSource

    location = 'test.gt'

    g = graph_gt.Graph(n_vertices=10)
 
    s = GraphGtSource(graph=g, location=location)
    print(s)

    s.write()

    r = GraphGtSource(location=location)
    print(r.shape)
    
    os.remove(location)
