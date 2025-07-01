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

import functools
import multiprocessing
import warnings
from typing import Dict, Callable, List, Sequence

import numpy as np

import ClearMap.IO.IO as io

import ClearMap.ImageProcessing.Topology.Topology3d as t3d

import ClearMap.ParallelProcessing.DataProcessing.ArrayProcessing as ap

from ClearMap.Analysis.graphs.fast_graph_reduce import find_degree2_branches
from ClearMap.Analysis.graphs import graph_gt

import ClearMap.Utils.Timer as tmr
from ClearMap.ParallelProcessing.DataProcessing.ArrayProcessing import initialize_sink


def mean_vertex_coordinates(coordinates):  # REFACTORING: functools.partial(np.mean, axis=0)
    return np.mean(coordinates, axis=0)

DEFAULT_EDGE_TO_EDGE = {'length': np.sum}
DEFAULT_VERTEX_TO_EDGE = {'radii': np.max}
DEFAULT_VERTEX_TO_VERTEX = {
    'coordinates': functools.partial(np.mean, axis=0),
    'coordinates_units': functools.partial(np.mean, axis=0),
    'length': np.sum,
    'radii': np.max
}

###############################################################################
# ## Graphs from skeletons
###############################################################################

def graph_from_skeleton(skeleton, points=None, radii=None, vertex_coordinates=True, spacing=None,
                        distance_unit=None, check_border=True, delete_border=False, verbose=False):  # FIXME: add orientation
    """
    Converts a binary skeleton image to a graph-gt graph.

    Note
    ----
    Edges are detected between neighbouring foreground pixels using 26-connectivity.

    .. rubric:: Algorithm: Graph From Skeleton

    1. Linearise the skeleton to create a list of 1D indices of the skeleton voxels.
    2. Create a graph with the number of vertices equal to the number of skeleton points.
    3. For each orientation (13 in total), calculate the offset 1D to the neighbouring voxel.
    4. For each skeleton point, find its neighbours using the offset and add edges to the graph.
    5. If `vertex_coordinates` is True, store the coordinates of the vertices in the graph. Also
    stores the physical coordinates if `spacing` is provided.
    6. If `radii` is provided, set the vertex radii in the graph.

    Arguments
    ---------
    skeleton: array
        Source with 2d/3d binary skeleton.
    points: array
        List of skeleton points as 1d indices of flat skeleton array (optional to save processing time).
    radii: array
        List of radii associated with each vertex.
    vertex_coordinates : bool
        If True, store coordinates of the vertices / edges.
    spacing: tuple
        Voxel spacing in physical units (e.g. micrometers) for the coordinates.
    distance_unit: str
        Name of the distance unit for the graph, e.g. 'µm', 'micrometer', 'nanometer', etc.
    check_border: bool
        If True, check if the border is empty. The algorithm requires this.
    delete_border: bool
        If True, delete the border.
    verbose: bool
        If True, print progress information.

    Returns
    -------
    graph : Graph class
        The graph corresponding to the skeleton.
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

    # Create the list of indices of the skeleton voxels
    if points is None:
        points = ap.where(skeleton.reshape(-1, order='A')).array
        if verbose:
            timer.print_elapsed_time('Point list generation', reset=True)

    # create graph
    n_vertices = points.shape[0]
    g = graph_gt.Graph(n_vertices=n_vertices, directed=False)
    g.shape = skeleton.shape
    if verbose:
        timer.print_elapsed_time(f'Graph initialized with {n_vertices} vertices', reset=True)

    # detect edges
    n_edges = 0
    for i, ori in enumerate(t3d.orientations()):
        # calculate offset
        ori_coord = np.hstack(np.where(ori))  # coordinate in 3,3,3 cube where True
        centered_ori_coord = (ori_coord - [1, 1, 1])  # center the coordinate to (0,0,0)
        offset = np.sum(centered_ori_coord * skeleton.strides)  # compute 1D offset for that orientation
        edges = ap.neighbours(points, offset)
        if edges.shape[0]:
            g.add_edge(edges)
            n_edges += edges.shape[0]
        if verbose:
            timer.print_elapsed_time(f'{edges.shape[0]} edges with orientation {i + 1}/13 found', reset=True)
    if verbose:
        timer.print_elapsed_time(f'Added {n_edges} edges to graph', reset=True)

    if vertex_coordinates:
        max_dim = max(skeleton.shape)
        coord_dtype = np.uint16 if max_dim < 65536 else np.uint32

        vertex_coordinates = np.array(
            np.unravel_index(points, skeleton.shape, order=skeleton.order)
        ).T.astype(coord_dtype)
        g.set_vertex_coordinates(vertex_coordinates)
        # optional physical coordinates (anisotropy-aware)
        if spacing is not None:  # Add unit and resolution information as graph properties
            spacing = np.asarray(spacing, dtype=np.float32)
            coords_phys = vertex_coordinates.astype(np.float32) * spacing[None, :]
            g.define_vertex_property('coordinates_units', coords_phys)
            g.define_graph_property('spacing', spacing)

    if radii is not None:
        g.set_vertex_radius(None, radii)

    if distance_unit is not None:
        g.define_graph_property('distance_unit', distance_unit)

    # FIXME: add orientation property

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

def drop_degree2_loops(graph):
    """
    Drop all loops in the graph that consist only of vertices with degree 2.

    Parameters
    ----------
    graph: GraphGt.Graph

    Returns
    -------
    GraphGt.Graph
        The graph with all pure degree 2 loops removed.
    """
    # 1) mask & view
    deg = graph.vertex_degrees()
    deg2_mask = (deg == 2)
    if not deg2_mask.any():
        return graph

    view = graph.sub_graph(vertex_filter=deg2_mask, view=True)

    # 2) component labels + counts
    comp_full, comp_counts = view.label_components(return_vertex_counts=True)
    comp_full = np.asarray(comp_full)
    comp_counts = np.array(comp_counts)

    # 2) restrict to the deg-2 vertices only
    # comp_map = comp_full[deg2_mask]

    # 3) count edges per component from the view’s edge_connectivity
    ec = view.edge_connectivity()
    src = ec[:, 0]
    comp_edge_counts = np.bincount(comp_full[src], minlength=len(comp_counts))

    # 4) cycles = components where #edges == #vertices
    cycle_ids = np.where(comp_edge_counts == comp_counts)[0]

    # 5) boolean mask over view-vertices
    comp_view = comp_full[view.vertex_indices()]
    loop_mask = np.isin(comp_view, cycle_ids)

    # 6) map back to original graph indices
    orig_vids = view.vertex_indices()
    to_remove = orig_vids[loop_mask]

    # 7) build the keep mask for the original graph
    v_keep = np.ones(graph.n_vertices, dtype=bool)
    v_keep[to_remove] = False

    print(f"Dropping {len(to_remove)} pure degree-2 loop vertices "
          f"from {graph.n_vertices} total.")
    return graph.sub_graph(vertex_filter=v_keep)


def reduce_graph(graph, vertex_to_edge_mappings={'radii': np.max},  # FIXME: use global settings for defaults
                 edge_to_edge_mappings={'length': np.sum},
                 edge_geometry=True, edge_length=False,
                 edge_geometry_vertex_properties=('coordinates', 'radii'),
                 edge_geometry_edge_properties=None,
                 return_maps=False, drop_pure_degree_2_loops=True,
                 verbose=False, print_period=250000):
    """
    Reduce graph by removing all vertices with degree two.
    Whenever this is done, the edges are merged and properties are aggregated
    The coordinates of the degree 2 vertices are stored in a shared graph property
    called edge_geometry_coordinates. The new edge will only hold the start and end indexing
    into that graph level array of coordinates for efficiency.

    .. warning::
        Currently, existing degree 2 loops are dropped inplace from the source graph.

    Parameters
    ----------
    graph : GraphGt.Graph
        The graph to reduce.  WARNING: currently, existing degree 2 loops are dropped inplace
    vertex_to_edge_mappings : dict
        A dictionary mapping vertex properties to edge properties. The keys are the vertex property names,
        and the values are functions that aggregate the vertex properties to edge properties.
    edge_to_edge_mappings : dict
        A dictionary mapping edge properties to edge properties. The keys are the edge property names,
        and the values are functions that aggregate the edge properties.
    edge_geometry : bool  # FIXME: improve docstring
        If True, compute edge geometry properties (coordinates, radii) for the reduced graph.
    edge_length : bool
        If True, compute the length of edges in the reduced graph.
        Mapping defaults to np.sum, i.e. the length of the edge is the sum of the lengths of
        the edges in the original graph.
    edge_geometry_vertex_properties : tuple  # FIXME: improve docstring
        A tuple of vertex property names to be used for edge geometry.
        These properties will be aggregated and stored in the edge geometry.
    edge_geometry_edge_properties : list  # FIXME: improve docstring
        A list of edge property names to be used for edge geometry.
        These properties will be aggregated and stored in the edge geometry.
        If None, no edge properties are used for edge geometry.
    return_maps : bool
        If True, return mappings of vertex to vertex, edge to vertex, and edge to edge.
        These mappings are useful for further processing of the reduced graph.
    drop_pure_degree_2_loops : bool
        If True, drop all pure degree 2 loops from the graph before reducing it.
        This is useful to avoid issues with loops in the graph that would otherwise
        lead to incorrect results.
    verbose : bool
        If True, print progress information during the reduction process.
    print_period : int
        The period in number of edges after which to print progress information in the chain finding step.

    Returns
    -------
    reduced_graph : GraphGt.Graph
        The reduced graph with degree 2 vertices removed and properties aggregated.
    """

    if verbose:
        timer = tmr.Timer()
        timer_all = tmr.Timer()
        print('Graph reduction: initialized.')

    check_graph_is_reduce_compatible(graph)
    # copy graph
    if drop_pure_degree_2_loops:
        graph = drop_degree2_loops(graph)  # WARNING: modifies graph in place
    g = graph.copy()

    # find non-branching points, i.e. vertices with deg 2
    non_degree_2_vertices_ids = np.where(g.vertex_degrees() != 2)[0]
    n_branch_points = len(non_degree_2_vertices_ids)
    degree_2_vertices_ids = np.where(g.vertex_degrees() == 2)[0]
    n_degree_2_vertices = len(degree_2_vertices_ids)

    if verbose:
        timer.print_elapsed_time(
            f'Graph reduction: Found {n_branch_points} branching and {n_degree_2_vertices} non-branching nodes')
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

    chains, edge_descriptors = find_chains(g) #, return_endpoints_mask=True)  # FIXME: remove return_endpoints_mask=True
    # check_chains(g, chains, degree_2_vertices_ids, non_degree_2_vertices_ids)

    edge_list = []
    # for edges, vertices, _ in chains:  # FIXME: remove _ if not return_endpoints_mask
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

    # Create an empty graph with same properties as g
    reduced_graph = g.sub_graph(edge_filter=np.zeros(g.n_edges, dtype=bool))  # Delete all edges
    reduced_graph.add_edge(edge_list) #  Add new edges

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

    if edge_geometry:   # Remap each edge property to the reduced graph (i.e. save as f'edge_geometry_{prop}' )
        branch_indices = np.hstack(reduced_maps['edge_to_vertex'])
        # Create start and end indices for edge geometry
        indices = np.cumsum([0] + [len(m) for m in reduced_maps['edge_to_vertex']])  # Length of each branch
        indices = np.array([indices[:-1], indices[1:]]).T

        g.edge_geometry_indices_set = False  # FIXME: why
        reduced_graph.set_edge_geometry_vertex_properties(g, edge_geometry_vertex_properties, branch_indices, indices)
        g.edge_geometry_indices_set = True

        if edge_geometry_edge_properties:
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


def check_chains(graph, chains, degree_2_v_ids, non_degree_2_v_ids):
    """
    Check if the chains computation is correct by verifying that all degree 2 vertices are included in the chains
    and that all non-degree 2 vertices are not included in the chains.

    Parameters
    ----------
    chains: list of list of edges
        List of chains in the graph.
    degree_2_v_ids: array-like
        Array of vertex ids with degree 2.
    non_degree_2_v_ids: array-like
        Array of vertex ids with degree not equal to 2.

    Returns
    -------
    bool
        True if the computation is correct, False otherwise.
    """
    degree_2_v_ids = np.unique(degree_2_v_ids)
    non_degree_2_v_ids = np.unique(non_degree_2_v_ids)

    d2_lists, end_lists = [], []

    for _, chain_vs, end_point_mask in chains:
        v_arr = np.asarray(chain_vs, dtype=int)
        d2_lists.append(v_arr[~end_point_mask])  # internal degree-2 vertices
        end_lists.append(v_arr[end_point_mask])  # real endpoints (deg ≠ 2)

    d2s_ids_from_chains = np.unique(np.hstack(d2_lists))
    non_d2s_ids_from_chains = np.unique(np.hstack(end_lists))

    # chains_vertices = [c[1] for c in chains]
    # non_d2s_ids_from_chains = (np.unique(np.hstack([(c[0], c[-1]) for c in chains_vertices]))).astype(int)  # first and last vertex of each chain
    # d2s_ids_from_chains = (np.unique(np.hstack([c[1:-1] for c in chains_vertices]))).astype(int)  # all vertices except first and last

    diff_d2s = set(degree_2_v_ids) ^ set(d2s_ids_from_chains)
    assert len(diff_d2s) == 0, f'Degree 2 vertices not found in chains: {diff_d2s}'

    assert np.all(d2s_ids_from_chains == degree_2_v_ids)
    assert np.all(np.isin(non_degree_2_v_ids, non_d2s_ids_from_chains))  # all non-degree 2 from chains are in non_degree_2_v_ids


def find_chains(graph, return_endpoints_mask=False):
    """
    Find chains (i.e. list of edges between vertices that are either branching points or end points) in a graph.

    This is done in three steps and the results are combined:
    - Compute chains between degree 2 vertices that are connected to non-degree 2 vertices.
    - Add direct edges between non-degree 2 vertices.
    - Add pure degree 2 loops that are not connected to any non-degree 2 vertices.
    (if the graph is not fully connected, this will find isolated loops)

    Parameters
    ----------
    graph: Graph
        The graph to process.
    return_endpoints_mask: bool
        If True, return a mask indicating which vertices are endpoints of the chains.
        This is particularly useful for debugging or further processing,
        especially in large graphs and when there may be pure degree 2 loops.
        where `endpoint_mask` is a 1-D ``bool`` np.NdArray aligned with
        `vertex_ids`.  A value *True* means “this vertex is a real endpoint
        (degree != 2)”, *False* means “internal degree-2 vertex”.
        Default is *False*.

    Returns
    -------
    chains : list
        *Without mask* (default):
            list of 2-tuples (edge_ids, vertex_ids)
        *With mask* (return_endpoint_mask=True):
            list of 3-tuples (edge_ids, vertex_ids, endpoint_mask)
    """
    # compute proper non_d2 --- d2...d2 --- non_d2 chains
    connectivity_w_eid = graph._base.get_edges(eprops=[graph._base.edge_index]).astype(np.uint32)
    connectivity_w_eid = np.roll(connectivity_w_eid, 1, axis=1)  # move edge id to the first column

    v1_degs = graph._base.get_out_degrees(connectivity_w_eid[:, 1]).astype(np.uint8)
    v2_degs = graph._base.get_out_degrees(connectivity_w_eid[:, 2]).astype(np.uint8)

    end_branches = np.logical_xor(v1_degs == 2, v2_degs == 2)
    end_branches_idx = np.where(end_branches)[0]
    end_branch_ids = connectivity_w_eid[end_branches_idx, 0]

    vertex_degs = graph.vertex_degrees().astype(np.uint8)
    chains = find_degree2_branches(
        np.ascontiguousarray(connectivity_w_eid),  # FIXME: Check if contiguous is necessary
        end_branch_ids.astype(np.uint32),
        vertex_degs,
    )

    edge_descriptors = list(graph._base.edges())

    # Add direct edges to chains to process together
    direct_edges = connectivity_w_eid[(v1_degs != 2) & (v2_degs != 2)]
    chains.extend(([eid], [v1, v2]) for eid, v1, v2 in direct_edges)

    # Add pure degree2 loops
    visited = np.zeros(graph.n_vertices, dtype=bool)
    visited_vs_idx = np.concatenate([np.asarray(vs, dtype=int) for _, vs in chains])
    visited[visited_vs_idx] = True

    unvisited_deg2_v_idx = np.where(~visited & (vertex_degs == 2))[0]
    if unvisited_deg2_v_idx:
        print(f'Found {len(unvisited_deg2_v_idx)} degree 2 vertices not visited in chains. '
              f'They will be processed as isolated loops.')
    else:
        raise ValueError('No degree 2 vertices found in the graph! '
                            'This is unexpected, please check the graph structure.')
    for v0 in unvisited_deg2_v_idx:
        if visited[v0]:  # in case marked within this very loop
            continue
        vs, es, ends, _isolated = _graph_branch(graph.vertex(v0))

        vs_idx = [int(v) for v in vs]
        edges_idx = [graph.edge_index(e) for e in es]
        chains.append((edges_idx, vs_idx))
        visited[vs_idx] = True

    if not visited.sum() == graph.n_vertices:
        raise ValueError(f'Not all vertices were visited during chain finding! '
                         f'{visited.sum()} / {graph.n_vertices} visited. '
                         f'Unvisited vertices: {np.where(~visited)[0]}')


    if return_endpoints_mask:
        chains_out = []
        for eids, vids in chains:
            mask = (vertex_degs[vids] != 2)  # True endpoint
            chains_out.append((eids, vids, mask))
        return chains_out, edge_descriptors
    else:
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
    g = graph_gt.Graph(n_vertices=n_vertices)

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
    import ClearMap.Analysis.graphs.graph_processing as gp
    from ClearMap.Analysis.graphs import graph_gt
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

    g = graph_gt.Graph(n_vertices=10)
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
    from ClearMap.Analysis.graphs import graph_gt
    import ClearMap.Analysis.graphs.graph_processing as gp

    g = graph_gt.Graph(n_vertices=10)
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
    from ClearMap.Analysis.graphs import graph_gt
    edges = [[0, 1], [1, 2], [2, 3], [4, 5], [5, 6], [1, 7]]
    g = graph_gt.Graph(edges=edges)

    _, _ = g.edge_graph(return_edge_map=True)

    import numpy as np
    label = np.zeros(len(edges), dtype=bool)
    label[1] = True

    import ClearMap.Analysis.graphs.graph_processing as gp

    def condition(graph, edge):
        print(f'condition, edge={edge}')
        return True

    traced = gp.trace_edge_label(g, label, condition=condition)
    print(traced)

    # expansion of edge lengths

    import numpy as np
    import ClearMap.Tests.Files as tf
    import ClearMap.Analysis.graphs.graph_processing as gp

    graph = graph_gt.Graph(n_vertices=5)
    graph.add_edge([[0, 1], [0, 2], [0, 3], [2, 3], [2, 1], [0, 4]])
    graph.add_edge_property('length', np.array([0, 1, 2, 3, 4, 5]) + 2)

    e, m = gp.expand_graph_length(graph, 'length', True)

    import graph_tool.draw as gd
    from ClearMap.Analysis.graphs import graph_gt
    pos = graph_gt.vertex_property_map_to_python(gd.sfdp_layout(e.base))
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
