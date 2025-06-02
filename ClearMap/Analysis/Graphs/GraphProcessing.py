#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
GraphProcessing
===============

Module providing tools to create and manipulate graphs. Functions include graph
construction from skeletons, simplification, reduction and clean-up of graphs.
"""
__author__ = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__ = 'https://idisco.info'
__download__ = 'https://www.github.com/ChristophKirst/ClearMap2'


import numpy as np

import ClearMap.Analysis.Graphs.type_conversions
import ClearMap.IO.IO as io

import ClearMap.ImageProcessing.Topology.Topology3d as t3d

import ClearMap.ParallelProcessing.DataProcessing.ArrayProcessing as ap

import ClearMap.Analysis.Graphs.GraphGt as ggt
from ClearMap.Analysis.Graphs.fast_graph_reduce import find_degree2_branches

import ClearMap.Utils.Timer as tmr


###############################################################################
# ## Graphs from skeletons
###############################################################################

def graph_from_skeleton(skeleton, points=None, radii=None, vertex_coordinates=True,
                        check_border=True, delete_border=False, verbose=False):
    """Converts a binary skeleton image to a graph-tool graph.

    Arguments
    ---------
    skeleton : array
        Source with 2d/3d binary skeleton.
    points : array
        List of skeleton points as 1d indices of flat skeleton array (optional to save processing time).
    radii  : array
        List of radii associated with each vertex.
    vertex_coordinates : bool
        If True, store coordinates of the vertices / edges.
    check_border : bool
        If True, check if the border is empty. The algorithm requires this.
    delete_border : bool
        If True, delete the border.
    verbose : bool
        If True, print progress information.

    Returns
    -------
    graph : Graph class
        The graph corresponding to the skeleton.

    Note
    ----
    Edges are detected between neighbouring foreground pixels using 26-connectivity.
    """
    skeleton = io.as_source(skeleton)

    if delete_border:
        skeleton = t3d.delete_border(skeleton)
        check_border = False

    if check_border:
        if not t3d.check_border(skeleton):
            raise ValueError('The skeleton array needs to have no points on the border!')

    if verbose:
        timer = tmr.Timer()
        timer_all = tmr.Timer()
        print('Graph from skeleton calculation initialize.!')

    if points is None:
        points = ap.where(skeleton.reshape(-1, order='A')).array

        if verbose:
            timer.print_elapsed_time('Point list generation')
            timer.reset()

    # create graph
    n_vertices = points.shape[0]
    g = ggt.Graph(n_vertices=n_vertices, directed=False)
    g.shape = skeleton.shape

    if verbose:
        timer.print_elapsed_time('Graph initialized with %d vertices' % n_vertices)
        timer.reset()

    # detect edges
    edges_all = np.zeros((0, 2), dtype=int)
    for i, o in enumerate(t3d.orientations()):
        # calculate off set
        offset = np.sum((np.hstack(np.where(o))-[1, 1, 1]) * skeleton.strides)
        edges = ap.neighbours(points, offset)
        if len(edges) > 0:
            edges_all = np.vstack([edges_all, edges])

        if verbose:
            timer.print_elapsed_time('%d edges with orientation %d/13 found' % (edges.shape[0], i+1))
            timer.reset()

    if edges_all.shape[0] > 0:
        g.add_edge(edges_all)

    if verbose:
        timer.print_elapsed_time('Added %d edges to graph' % (edges_all.shape[0]))
        timer.reset()

    if vertex_coordinates:
        vertex_coordinates = np.array(np.unravel_index(points, skeleton.shape, order=skeleton.order)).T
        g.set_vertex_coordinates(vertex_coordinates)

    if radii is not None:
        g.set_vertex_radius(radii)

    if verbose:
        timer_all.print_elapsed_time('Skeleton to Graph')

    return g


def graph_to_skeleton(graph, sink=None, dtype=bool, values=True):
    """Create a binary skeleton from a graph."""
    if not graph.has_edge_geometry():
        coordinates = np.asarray(graph.vertex_coordinates(), dtype=int)
    else:
        coordinates = graph.edge_geometry('coordinates')
        coordinates = np.asarray(np.vstack(coordinates), dtype=int)

    if sink is None:
        shape = graph.shape
        if shape is None:
            shape = tuple(np.max(coordinates, axis=0))
        sink = np.zeros(shape, dtype=dtype)

    coordinates = tuple(coordinates.T)
    sink[coordinates] = values

    return sink


###############################################################################
# ## Graph CleanUp
###############################################################################

def mean_vertex_coordinates(coordinates):
    return np.mean(coordinates, axis=0)


def clean_graph(graph, remove_self_loops=True, remove_isolated_vertices=True,
                vertex_mappings={'coordinates' : mean_vertex_coordinates,
                                 'radii'       : np.max},
                verbose=False):
    """
    Remove all cliques to get pure branch structure of a graph.

    Arguments
    ---------
    graph : Graph
      The graph to clean up.
    verbose : bool
      If True, prin progress information.

    Returns
    -------
    graph : Graph
      A graph removed of all cliques.

    Note
    ----
    cliques are replaced by a single vertex connecting to all non-clique neighbours
    The center coordinate is used for that vertex as the coordinate.
    """

    if verbose:
        timer = tmr.Timer()
        timer_all = tmr.Timer()

    # find branch points
    n_vertices = graph.n_vertices
    degrees = graph.vertex_degrees()

    branches = degrees >= 3
    n_branch_points = branches.sum()

    if verbose:
        timer.print_elapsed_time('Graph cleaning: found %d branch points among %d vertices' % (n_branch_points, n_vertices))
        timer.reset()

    if n_branch_points == 0:
        return graph.copy()

    # detect 'cliques', i.e. connected components of branch points
    gb = graph.sub_graph(vertex_filter=branches, view=True)
    components, counts = gb.label_components(return_vertex_counts=True)

    # note: graph_tools components of a view is a property map of the full graph
    components[branches] += 1

    # group components
    components = _group_labels(components)

    # note: graph_tools components of a view is a property map of the full graph
    # note: remove the indices of the non branch nodes
    components = components[1:]

    clique_ids = np.where(counts > 1)[0]
    n_cliques = len(clique_ids)

    if verbose:
        timer.print_elapsed_time('Graph cleaning: detected %d cliques of branch points' % n_cliques)
        timer.reset()

    # remove cliques
    g = graph.copy()
    g.add_vertex(n_cliques)

    # mappings
    properties = {}
    mappings = {}
    for k in vertex_mappings.keys():
        if k in graph.vertex_properties:
            mappings[k] = vertex_mappings[k]
            properties[k] = graph.vertex_property(k)

    vertex_filter = np.ones(n_vertices + n_cliques, dtype=bool)
    for i, ci in enumerate(clique_ids):
        vi = n_vertices + i
        cc = components[ci]

        # remove clique vertices
        vertex_filter[cc] = False

        # get neighbours
        neighbours = np.hstack([graph.vertex_neighbours(c) for c in cc])
        neighbours = np.setdiff1d(np.unique(neighbours), cc)

        # connect to new node
        g.add_edge([[n, vi] for n in neighbours])

        # map properties
        for k in mappings.keys():
            g.set_vertex_property(k, mappings[k](properties[k][cc]), vertex=vi)

        if verbose and i+1 % 10000 == 0:
            timer.print_elapsed_time(f'Graph cleaning: reducing {i + 1} / {n_cliques}')

    # generate new graph
    g = g.sub_graph(vertex_filter=vertex_filter)

    if remove_self_loops:
        g.remove_self_loops()

    if verbose:
        timer.print_elapsed_time(
          f'Graph cleaning: removed {n_cliques} cliques of branch points from {graph.n_vertices}'
          f' to {g.n_vertices} nodes and {graph.n_edges} to {g.n_edges} edges')
        timer.reset()

    # remove isolated vertices
    if remove_isolated_vertices:
        non_isolated = g.vertex_degrees() > 0
        g = g.sub_graph(vertex_filter=non_isolated)

        if verbose:
            timer.print_elapsed_time(f'Graph cleaning: Removed {np.logical_not(non_isolated).sum()} isolated nodes')
            timer.reset()

        del non_isolated

    if verbose:
        timer_all.print_elapsed_time(
          f'Graph cleaning: cleaned graph has {g.n_vertices} nodes and {g.n_edges} edges')

    return g


def _group_labels(array):
    """Helper grouping labels in a 1d array."""
    idx = np.argsort(array)
    arr = array[idx]
    dif = np.diff(arr)
    res = np.split(idx, np.where(dif > 0)[0]+1)
    return res


###############################################################################
# ## Graph reduction
###############################################################################

def reduce_graph(graph, vertex_to_edge_mappings={'radii': np.max},  # FIXME: use global settings
                 edge_to_edge_mappings={'length': np.sum},
                 edge_geometry=True, edge_length=None,
                 edge_geometry_vertex_properties=('coordinates', 'radii'),
                 edge_geometry_edge_properties=None,
                 return_maps=False, verbose=False, print_period=250000):
    """Reduce graph by replacing all vertices with degree two."""

    if verbose:
        timer = tmr.Timer()
        timer_all = tmr.Timer()
        print('Graph reduction: initialized.')

    check_graph_is_reduce_compatible(graph)

    # copy graph
    g = graph.copy()

    # find non-branching points, i.e. vertices with deg 2
    non_degree_2_vertices_ids = np.where(g.vertex_degrees() != 2)[0]
    n_branch_points = len(non_degree_2_vertices_ids)
    degree_2_vertices_ids = np.where(g.vertex_degrees() == 2)[0]
    n_non_branch_points = len(degree_2_vertices_ids)

    if verbose:
        timer.print_elapsed_time(
            f'Graph reduction: Found {n_branch_points} branching and {n_non_branch_points} non-branching nodes')
        timer.reset()

    # mappings
    edge_to_edge_mappings = edge_to_edge_mappings or {}
    vertex_to_edge_mappings = vertex_to_edge_mappings or {}

    if edge_length:
        edge_to_edge_mappings['length'] = edge_to_edge_mappings.get('length', np.sum)
        if 'length' not in g.edge_properties:
            g.add_edge_property('length', np.ones(g.n_edges, dtype=float))

    edge_to_edge = {}
    edge_properties = {}
    edge_lists = {}
    for k in edge_to_edge_mappings.keys():
        if k in g.edge_properties:
            edge_to_edge[k] = edge_to_edge_mappings[k]
            edge_properties[k] = g.edge_property_map(k)
            edge_lists[k] = []

    vertex_to_edge = {}
    vertex_properties = {}
    vertex_lists = {}
    for k in vertex_to_edge_mappings.keys():
        if k in g.vertex_properties:
            vertex_to_edge[k] = vertex_to_edge_mappings[k]
            vertex_properties[k] = g.vertex_property_map(k)
            vertex_lists[k] = []

    # edge geometry
    reduced_maps = {}
    if edge_geometry:
        reduced_maps['edge_to_vertex'] = []

    if return_maps:
        reduced_maps['vertex_to_vertex'] = []
        reduced_maps['edge_to_edge'] = []

    if edge_geometry_edge_properties is None:
        edge_geometry_edge_properties = []
    else:
        reduced_maps['edge_to_edge'] = []

    if edge_geometry_vertex_properties is None:
        edge_geometry_vertex_properties = []

    chains, edge_descriptors = find_chains(g)

    edge_list = []
    for edges, vertices in chains:
        edges = [edge_descriptors[eid] for eid in edges]  # Convert edge id to edge object
        vertices_ids = [int(vv) for vv in vertices]  # FIXME: check if necessary

        edge_list.append([int(ep) for ep in [vertices[0], vertices[-1]]])  # we just need between branch_points to be considered end points

        # mappings
        for k in edge_to_edge.keys():
            ep = edge_properties[k]
            edge_lists[k].append(edge_to_edge[k]([ep[e] for e in edges]))
        for k in vertex_to_edge.keys():
            v_prop = vertex_properties[k]
            vertex_lists[k].append(vertex_to_edge[k]([v_prop[vv] for vv in vertices]))

        if 'edge_to_vertex' in reduced_maps.keys():
            reduced_maps['edge_to_vertex'].append(vertices_ids)

        if 'edge_to_edge' in reduced_maps.keys():
            reduced_maps['edge_to_edge'].append(edges)

    # REDUCE GRAPH

    # redefine branch edges
    reduced_graph = g.sub_graph(edge_filter=np.zeros(g.n_edges, dtype=bool))  # Delete all edges
    reduced_graph.add_edge(edge_list) #  Add new edges
    # FIXME: is it the same as g.__class__(edges=edge_list)
    # reduced_grpah = g.empty_like()

    # determine edge ordering
    edge_order = reduced_graph.edge_indices()

    # remove degree 2 nodes
    if 'vertex_to_vertex' in reduced_maps.keys():
        reduced_graph.add_vertex_property('_vertex_id_', graph.vertex_indices())
    v_filter = np.zeros(g.n_vertices, dtype=bool)
    v_filter[non_degree_2_vertices_ids] = True
    reduced_graph = reduced_graph.sub_graph(vertex_filter=v_filter)
    if 'vertex_to_vertex' in reduced_maps.keys():
        reduced_maps['vertex_to_vertex'] = reduced_graph.vertex_property('_vertex_id_')
        reduced_graph.remove_vertex_property('_vertex_id_')

    # maps
    if 'edge_to_vertex' in reduced_maps.keys():
        reduced_maps['edge_to_vertex'] = np.array(reduced_maps['edge_to_vertex'], dtype=object)[edge_order]
    if 'edge_to_edge' in reduced_maps.keys():
        reduced_maps['edge_to_edge'] = np.array(reduced_maps['edge_to_edge'], dtype=object)[edge_order]

    # add edge properties
    for k in edge_to_edge.keys():
        reduced_graph.define_edge_property(k, np.array(edge_lists[k])[edge_order])
    for k in vertex_to_edge.keys():
        reduced_graph.define_edge_property(k, np.array(vertex_lists[k])[edge_order])

    if edge_geometry:   # Remap each edge property to the reduced graph (i.e. save as an edge_geometry as f'edge_{prop}' )
        branch_indices = np.hstack(reduced_maps['edge_to_vertex'])
        # Create start and end indices for edge geometry
        indices = np.cumsum([0] + [len(m) for m in reduced_maps['edge_to_vertex']])  # Length of each branch
        indices = np.array([indices[:-1], indices[1:]]).T

        g.edge_geometry_indices_set = False  # FIXME: why
        reduced_graph.set_edge_geometry_vertex_properties(g, edge_geometry_vertex_properties, branch_indices, indices)
        g.edge_geometry_indices_set = True
        reduced_graph.set_edge_geometry_edge_properties(g, edge_geometry_edge_properties, indices, reduced_maps['edge_to_edge'])

    if 'edge_to_edge' in reduced_maps.keys():
        reduced_maps['edge_to_edge'] = [[g.edge_index(e) for e in edge] for edge in reduced_maps['edge_to_edge']]

    if verbose:
        timer_all.print_elapsed_time(f'Graph reduction: Graph reduced from {graph.n_vertices} to {reduced_graph.n_vertices} nodes'
                                     f' and {graph.n_edges} to {reduced_graph.n_edges} edges')

    if return_maps:
        return reduced_graph, (reduced_maps[k] for k in ('vertex_to_vertex', 'edge_to_vertex', 'edge_to_edge'))
    else:
        return reduced_graph


def find_chains(g):
    """
    Find chains (i.e. list of edges between vertices that are either branching points or end points) in a graph.

    Parameters
    ----------
    g: Graph
        The graph to process.

    Returns
    -------
    chains: list of list of edges
        List of chains in the graph.
    """
    connectivity_w_eid = g._base.get_edges(eprops=[g._base.edge_index]).astype(np.uint32)
    connectivity_w_eid = np.roll(connectivity_w_eid, 1, axis=1)  # move edge id to the first column
    v1_degs = g._base.get_out_degrees(connectivity_w_eid[:, 1]).astype(np.uint8)
    v2_degs = g._base.get_out_degrees(connectivity_w_eid[:, 2]).astype(np.uint8)
    end_branches = np.logical_xor(v1_degs == 2, v2_degs == 2)
    end_branches_idx = np.where(end_branches)[0]
    end_branch_ids = connectivity_w_eid[end_branches_idx, 0]
    vertex_degs = g.vertex_degrees().astype(np.uint8)
    chains = find_degree2_branches(np.ascontiguousarray(connectivity_w_eid),  # FIXME: Check if contiguous is necessary
                                   end_branch_ids.astype(np.uint32),
                                   vertex_degs)
    edge_descriptors = list(g._base.edges())
    direct_edges = connectivity_w_eid[np.logical_and(v1_degs != 2, v2_degs != 2)]

    # Add direct edges to chains to process together
    for eid, v1, v2 in direct_edges:
        chains.append(([eid], [v1, v2]))
    return chains, edge_descriptors


def check_graph_is_reduce_compatible(graph):
    # ensure conditions for processing step are fulfilled
    if graph.is_view:
        raise ValueError('Cannot process on graph view, prune graph before graph reduction.')
    if not np.all(np.diff(graph.vertex_indices()) == 1) or int(graph.vertex_iterator().next()) != 0:
        raise ValueError('Graph vertices not ordered!')


def _graph_branch(v0):
    """Helper to follow a branch in a graph consisting of a chain of vertices of degree 2."""

    edges_v0 = [e for e in v0.out_edges()]
    assert len(edges_v0) == 2

    vertices_1, edges_1, endpoint_1 = _follow_branch(edges_v0[0], v0)
    if endpoint_1 != v0:
        vertices_2, edges_2, endpoint_2 = _follow_branch(edges_v0[1], v0)

        vertices_1.reverse()
        vertices_1.append(v0)
        vertices_1.extend(vertices_2)

        edges_1.reverse()
        edges_1.extend(edges_2)
        isolated_loop = False
    else:
        endpoint_2 = v0
        isolated_loop = True

    return vertices_1, edges_1, [int(endpoint_1), int(endpoint_2)], isolated_loop


def _follow_branch(e, v0):
    """
    Helper to follow branch from vertex v0 in direction of edge e.

    .. note::
        v0 is used to prevent infinite loops for isolated closed loops.

    Arguments
    ---------
    e : Edge
        The edge to follow.
    v0 : Vertex
        The starting vertex.


    """
    edges = [e]

    v = e.target()
    vertices = [v]

    while v.out_degree() == 2 and v != v0:
        for e_new in v.out_edges():
            if e_new != e:
                break
        e = e_new
        edges.append(e)
        v = e.target()
        vertices.append(v)

    return vertices, edges, v


def expand_graph_length(graph, length='length', return_edge_mapping=False):
    """Expand a reduced graph to a full graph."""
    # data
    ec = graph.edge_connectivity()

    vertex_lengths = graph.edge_property(length)
    edge_lengths = np.asarray(vertex_lengths, dtype=int) - 1

    # construct graph
    new_edges = edge_lengths > 1  # edges to create
    n_vertices = graph.n_vertices + int(np.sum(vertex_lengths - 2))
    g = ggt.Graph(n_vertices=n_vertices)

    # construct edges
    offset = graph.n_vertices
    new_starts = np.hstack([[0], np.cumsum(edge_lengths[new_edges] - 1)])
    new_ends = new_starts[1:] - 1
    new_starts = new_starts[:-1]

    new_to_new = np.array([np.arange(new_ends[-1]), np.arange(new_ends[-1])+1]).T
    split = np.ones(new_ends[-1], dtype=bool).T
    split[new_ends[:-1]] = False
    new_to_new = new_to_new[split]

    new_starts += offset
    new_ends += offset
    new_to_new += offset

    existing_to_existing = ec[np.logical_not(new_edges)]
    existing_to_new = np.array([ec[new_edges, 0], new_starts]).T
    new_to_existing = np.array([new_ends, ec[new_edges, 1]]).T

    edges = np.vstack([existing_to_existing, existing_to_new, new_to_new, new_to_existing])

    g.add_edge(edges)

    if return_edge_mapping:
        # edge order in new graph
        reorder = g.edge_indices()

        # mapping[new_edge_id] = old_edge_id
        existing_edges_id = np.where(np.logical_not(new_edges))[0]
        new_edges_id = np.where(new_edges)[0]
        mapping = np.hstack(
                  [existing_edges_id,  # existing edges
                   new_edges_id,  # existing to new
                   np.repeat(new_edges_id, new_ends - new_starts),  # new to new
                   new_edges_id])  # new to existing;

        mapping = mapping[reorder]

        return g, mapping
    else:
        return g


def expand_graph(graph):
    raise NotImplementedError()

    # if not graph.has_edge_geometry():
    #     return graph.copy()
    #
    # # calculate nodes of expanded graph
    # lengths = graph.edge_geometry_lengths() - 2
    # n_add_vertices = np.sum(lengths)
    #
    # g = graph.copy()
    # g.add_vertex(n_add_vertices)
    # g.remove_edge_geometry()
    #
    # # re link edges and add vertex properties
    # # current_vertex =
    # # for e in graph.edge_iterator():
    #
    #
    # # for name in graph.edge_geometry_names:
    #
    # graph.edge_geometry('coordinates')


###############################################################################
# ## Label Tracing
###############################################################################

def trace_vertex_label(graph, vertex_label, condition, dilation_steps=1, max_iterations=None, pass_label=False, **condition_args):
    """
    Traces label within a graph.

    Arguments
    ---------
    vertex_label : array
        Start label.
    condition : function(graph, vertex)
        A function determining if the vertex should be added to the labels.
    steps : int
        Number edges to jump to find new neighbours.  Default is 1.

    Returns
    -------
    vertex_label : array
        The traced label.
    """
    label = vertex_label.copy()
    not_checked = np.logical_not(label)

    if max_iterations is None:
        max_iterations = np.inf

    iteration = 0
    while iteration < max_iterations:
        label_dilated = graph.vertex_dilate_binary(label, steps=dilation_steps)
        label_new = np.logical_xor(label, label_dilated)
        label_new = np.logical_and(label_new, not_checked)
        vertices_new = np.where(label_new)[0]
        # print(label.sum(), label_dilated.sum(), label_new.sum())
        # print(vertices_new)
        if len(vertices_new) == 0:
            break
        else:
            if pass_label:
                current_label = label.copy()
                for v in vertices_new:
                    if condition(graph, v, current_label, **condition_args):
                        label[v] = True
                    not_checked[v] = False
            else:
                for v in vertices_new:
                    if condition(graph, v, **condition_args):
                        label[v] = True
                    not_checked[v] = False
        iteration += 1

    return label


def trace_edge_label(graph, edge_label, condition, max_iterations = None, dilation_steps = 1, pass_label = False, **condition_args):
    """Traces label within a graph.

    Arguments
    ---------
    edge_label : array
      Start label.
    condition : function(graph, vertex)
      A function determining if the vertex should be added to the labels.
    steps : int
      Number edges to jump to find new neighbours.  Default is 1.

    Returns
    -------
    edge_label : array
      The traced label.
    """
    edge_graph, edge_map = graph.edge_graph(return_edge_map=True)
    traced = trace_vertex_label(edge_graph, edge_label, condition, max_iterations=max_iterations,
                                dilation_steps=dilation_steps, pass_label=pass_label, **condition_args)
    label = np.zeros(len(traced), dtype=bool)
    label[edge_map] = traced  # TODO: check if initial label need to be reordered too ?
    return label
  
    # label = edge_label.copy()
    # not_checked = np.logical_not(label)
    # while True:
    #    label_dilated = graph.edge_dilate_binary(label, steps=steps)
    #    label_new = np.logical_xor(label, label_dilated)
    #    label_new = np.logical_and(label_new, not_checked)
    #    edges_new = np.where(label_new)[0]
    #    # print(label.sum(), label_dilated.sum(), label_new.sum())
    #    # print(vertices_new)
    #    if len(edges_new) == 0:
    #        break
    #    else:
    #        for e in edges_new:
    #            if condition(graph, e):
    #                label[e] = True
    #            not_checked[e] = False
    #
    # return label


###############################################################################
# ## Tests
###############################################################################

def _test():
    import numpy as np
    import ClearMap.Tests.Files as tf
    import ClearMap.Analysis.Graphs.GraphProcessing as gp
    # reload(gp)

    skeleton = tf.source('skeleton')

    # import ClearMap.Visualization.Plot3d as p3d
    # p3d.plot(skeleton)

    # reload(gp)
    g = gp.graph_from_skeleton(skeleton)

    g.vertex_coordinates()

    s = gp.graph_to_skeleton(g)
    assert np.all(s == skeleton)

    gc = gp.clean_graph(g, verbose=True)

    gr = gp.reduce_graph(gc, verbose=True)

    gr2 = gr.copy()
    gr2.set_edge_geometry_type('edge')

    l = gr.edge_geometry_lengths()
    print(l)

    g = gp.ggt.Graph(n_vertices=10)
    g.add_edge(np.array([[7, 8], [7, 9], [1, 2], [2, 3], [3, 1], [1, 4], [4, 5], [2, 6], [6, 7]]))
    g.set_vertex_coordinates(np.array([
        [10, 10, 10], [0, 0, 0], [1, 1, 1], [1, 1, 0], [5, 0, 0],
        [8, 0, 1], [0, 7, 1], [0, 10, 2], [0, 12, 3], [3, 7, 7]], dtype=float))

    import ClearMap.Visualization.Plot3d as p3d
    p3d.plot_graph_line(g)

    gc = gp.clean_graph(g, verbose=True)
    p3d.plot_graph_line(gc)

    gr = gp.reduce_graph(gc, edge_geometry=True, verbose=True)
    # gr.set_edge_geometry(0.1*np.ones(gr.edge_geometry(as_list=False).shape[0]), 'radii')


    import ClearMap.Visualization.Plot3d as p3d
    vertex_colors = np.random.rand(g.n_vertices, 4)
    vertex_colors[:, 3] = 1
    p3d.plot_graph_mesh(gr, default_radius=1, vertex_colors=vertex_colors)

    eg = gr.edge_geometry(as_list=False)
    egs = 0.5 * eg
    gr.set_edge_geometry(name='coordinates', values=egs)

    # tracing
    import numpy as np
    import ClearMap.Visualization.Plot3d as p3d
    import ClearMap.Analysis.Graphs.GraphProcessing as gp

    g = gp.ggt.Graph(n_vertices=10)
    g.add_edge(np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 0], [0, 5], [5, 6], [6, 7], [0, 8], [8, 9], [9, 0]]))
    g.set_vertex_coordinates(np.array([
        [0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 2, 0], [0, 1, 0],
        [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, -1, 0], [0, -1, -1]], dtype=float))
    g.set_vertex_radii(np.array([10, 6, 4, 6, 7, 4, 2, 2, 5, 5]) * 0.02)
    vertex_colors = np.array([g.vertex_radii()]*4).T
    vertex_colors = vertex_colors / vertex_colors.max()
    vertex_colors[:, 3] = 1
    p3d.plot_graph_mesh(g, default_radius=1, vertex_colors=vertex_colors)

    def condition(graph, vertex):
        r = graph.vertex_radii(vertex=vertex)
        print(f'condition, vertex={vertex}, radius={r}')
        return r >= 5 * 0.02

    label = np.zeros(g.n_vertices, dtype=bool)
    label[0] = True
    traced = gp.trace_vertex_label(g, label, condition=condition, steps=1)
    print(traced)

    vertex_colors = np.array([[1, 0, 0, 1], [1, 1, 1, 1]])[np.asarray(traced, dtype=int)]
    p3d.plot_graph_mesh(g, default_radius=1, vertex_colors=vertex_colors)

    from importlib import reload
    reload(gp)

    # edge tracing
    import ClearMap.Analysis.Graphs.GraphGt as ggt
    edges = [[0, 1], [1, 2], [2, 3], [4, 5], [5, 6], [1, 7]]
    g = ggt.Graph(edges=edges)

    _, _ = g.edge_graph(return_edge_map=True)

    import numpy as np
    label = np.zeros(len(edges), dtype=bool)
    label[1] = True

    import ClearMap.Analysis.Graphs.GraphProcessing as gp

    def condition(graph, edge):
        print(f'condition, edge={edge}')
        return True

    traced = gp.trace_edge_label(g, label, condition=condition)
    print(traced)

    # expansion of edge lengths

    import numpy as np
    import ClearMap.Tests.Files as tf
    import ClearMap.Analysis.Graphs.GraphProcessing as gp

    graph = gp.ggt.Graph(n_vertices=5)
    graph.add_edge([[0, 1], [0, 2], [0, 3], [2, 3], [2, 1], [0, 4]])
    graph.add_edge_property('length', np.array([0, 1, 2, 3, 4, 5]) + 2)

    e, m = gp.expand_graph_length(graph, 'length', True)

    import graph_tool.draw as gd
    from ClearMap.Analysis.Graphs import GraphGt
    pos = GraphGt.vertex_property_map_to_python(gd.sfdp_layout(e.base))
    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.clf()
    import matplotlib.collections as mc

    import ClearMap.Visualization.Color as col
    colors = np.array([col.color(c) for c in ['red', 'blue', 'green', 'black', 'purple', 'orange']])

    ec = e.edge_connectivity()
    lines = pos[ec]
    cols = colors[m]
    lc = mc.LineCollection(lines, linewidths=1, color=cols)

    ax = plt.gca()
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)
    plt.scatter(pos[:, 0], pos[:, 1])
    p = pos[:graph.n_vertices]
    plt.scatter(p[:, 0], p[:, 1], color='red')
