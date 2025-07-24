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

import ClearMap.ParallelProcessing.DataProcessing.ArrayProcessing as ap
import ClearMap.ImageProcessing.Topology.Topology3d as t3d
import ClearMap.Utils.Timer as tmr

from ClearMap.Analysis.graphs import graph_gt
from ClearMap.Analysis.graphs.fast_graph_reduce import find_degree2_branches, cy_reduce

SENTINEL = -1


def mean_vertex_coordinates(coordinates):
    return np.mean(coordinates, axis=0)


def medoid_vertex_coordinates(coordinates):
    """
    Compute the vertex coordinates as the median of the coordinates.
    This gives a vertex **on the grid** that is closest to the median of the coordinates.

    .. note::
        The vertex may not be one of the original coordinates, but it is guaranteed to be
        a valid vertex coordinate in the graph.

    Parameters
    ----------
    coordinates

    Returns
    -------

    """
    k = coordinates.shape[0] // 2
    median = np.array([
        np.partition(coordinates[:, i], k)[k]
        for i in range(3)
    ])
    return median


DEFAULT_EDGE_TO_EDGE = {
    'length': np.sum,
    'chain_id': np.mean   # We use mean as a sanity check (float outputs would be a tell-tale sign of a bug)
}
DEFAULT_VERTEX_TO_EDGE = {
    'radii': np.max
}
DEFAULT_VERTEX_TO_VERTEX = {
    'coordinates': medoid_vertex_coordinates,
    'coordinates_units': mean_vertex_coordinates,
    'length': np.sum,
    'radii': np.max,
    # 'chain_id': functools.partial(np.quantile, 0.5, method='nearest', axis=0),
    'chain_id': np.mean,
    '_vertex_id_': np.min
}

###############################################################################
# ## Graphs from skeletons
###############################################################################

def neighbours(indices: np.ndarray, offset: int) -> np.ndarray:  # TODO: check if remove
    """
    indices  – sorted 1-D array of flat voxel indices (int64)
    offset   – neighbour offset in *elements* (signed)
    returns  – (m,2) array with vertex-index pairs
    """
    shifted = indices + offset                      # still sorted
    pos     = np.searchsorted(indices, shifted)     # batched binary search
    # pos is where the shifted indices would be inserted in the original array

    valid = pos < indices.size  # step 1: in_bounds mask
    mask = np.zeros_like(valid, dtype=bool)
    # mask = indices[pos] == shifted i.e. there is an element at the shifted position
    mask[valid] = indices[pos[valid]] == shifted[valid]

    src = np.nonzero(mask)[0].astype(np.int64)
    dst = pos[mask].astype(np.int64)
    return np.vstack((src, dst)).T


def annotate_edge_lengths(graph: graph_gt.Graph, coord_prop, spacing=None):
    """
    Compute the length property for edges in a graph based on the voxel
    coordinates and the resolution of the voxels.

    Parameters
    ----------
    graph: graph_tool.Graph
    coord_prop: VertexPropertyMap
        Array of shape (N,3)   voxel indices (⟨x,y,z⟩)
        Array of shape (N,3)   voxel indices (⟨x,y,z⟩)
    spacing: tuple(float, float, float) | None
        Voxel pitch along each axis. If None, the value is retrieved from the graph's
        `spacing` property.

    .. warning::
        Because this uses adjacent connectivity, it can only be used for graphs
        before reduction, i.e. before `reduce_graph` is called.
        Also, coord_prop must be integer-valued, i.e. voxel indices.


    Notes
    -----
    * Works because every edge links *adjacent* voxels → only 26 displacement vectors.
    * Complexity: **O(|E|)** but with *only* cheap integer arithmetics + one final lookup.
    """
    spacing = spacing if spacing is not None else graph.graph_property('spacing')

    src_vertices, target_vertices = np.hsplit(graph.edge_connectivity(), 2)  # TODO: check indexing
    src_vertices = src_vertices.squeeze()
    target_vertices = target_vertices.squeeze()

    # coord_prop.a flattens, so reshape:
    if isinstance(coord_prop, np.ndarray):
        coords = coord_prop.reshape((-1, 3)).astype(int)
    else:
        coords = coord_prop.a.reshape((-1, 3)).astype(int)

    # convert -1, 0, +1 displacement to lookup index
    diff = coords[target_vertices] - coords[src_vertices]  # triplets of values -1, 0, +1 (in each dimension)
    if diff.max() > 1 or diff.min() < -1:
        raise ValueError(f'Coordinates must be integer-valued voxel indices in the range [-1, 1] for each axis.'
                         f'Found values outside this range: {diff.min()} to {diff.max()}. '
                         f'This graph may have been reduced or the coordinates are not voxel indices.'
                         f'Please use `annotate_edge_lengths` before reducing the graph.')

    # Use the trick below if speed is a constraint
    # Convert to index in [0,26] for the distance lookup table using base 3
    # idx = (diff[:, 0] + 1) * 9 + (diff[:, 1] + 1) * 3 + (diff[:, 2] + 1)
    # return distance_table[idx]

    res = np.asarray(spacing, dtype=np.float64)
    distance_table = get_distance_map_27(res).reshape(3, 3, 3)

    # shifted by 1 to get [0-2] indices for each axis
    xs = diff[:, 0] + 1
    ys = diff[:, 1] + 1
    zs = diff[:, 2] + 1
    return distance_table[xs, ys, zs]


def get_distance_map_27(resolution):
    """
    Create a distance lookup table for the 26 possible displacements in 3D.
    Uses the resolution to convert voxel displacements to physical distances.

    Parameters
    ----------
    resolution: tuple(float, float, float)
        Size of one voxel along each axis (x, y, z).

    Returns
    -------
    distance_table: np.ndarray
        A 1D array of shape (27,) containing the distances for each of the 26
        possible displacements in 3D, plus one entry for the zero displacement.
    """
    # index mapping:   idx = (dx+1)*9 + (dy+1)*3 + (dz+1)
    # produces 3-digit base-3 number in ⟦0,26⟧
    distance_table = np.empty(27, dtype=np.float64)  # Float64 to avoid overflow when summing
    k = 0
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                # physical displacement
                d_xyz = resolution * np.array([dx, dy, dz], dtype=np.float64)
                distance_table[k] = np.linalg.norm(d_xyz)
                k += 1
    return distance_table


def graph_from_skeleton(skeleton, points=None, radii=None, compute_vertex_coordinates=True, compute_edge_length=True,
                        check_border=True, delete_border=False, spacing=None, verbose=False):
    """
    Converts a binary skeleton image to a graph-tool graph.


    .. note::
        Edges are detected between neighbouring foreground pixels using 26-connectivity.

    Arguments
    ---------
    skeleton: array
        Source with 2d/3d binary skeleton.
    points: array
        List of skeleton points as 1d indices of flat skeleton array (optional to save processing time).
    radii: array
        List of radii associated with each vertex.
    compute_vertex_coordinates: bool
        If True, store coordinates of the vertices / edges.
    compute_edge_length: bool
        If True, compute the length of each edge based on the vertex coordinates.
        If spacing is also provided, the length is computed in physical units.
    check_border: bool
        If True, check if the border is empty. The algorithm requires this.
    delete_border: bool
        If True, delete the border (`check_border` is ignored in this case).
    spacing: array
        Spacing of the voxels in the skeleton in physical units (e.g. micrometers) in each dimension.
    verbose: bool
        If True, print progress information.

    Returns
    -------
    graph : Graph class
        The graph corresponding to the skeleton.
    """
    if compute_edge_length and not compute_vertex_coordinates:
        raise ValueError('Activating `compute_edge_length` requires `vertex_coordinates` to be True!')

    skeleton = io.as_source(skeleton)
    if skeleton.dtype not in ('bool', np.uint8):
        raise TypeError('The skeleton array needs to be a boolean array!')

    if delete_border:
        skeleton = t3d.delete_border(skeleton)
        check_border = False

    if check_border:
        if not t3d.check_border(skeleton):
            raise ValueError('The skeleton array needs to have no points on the border!')

    if verbose:
        timer = tmr.Timer()
        timer_all = tmr.Timer()
        print('Graph from skeleton calculation initialized.!')

    if points is None:
        # points = ap.where(skeleton.reshape(-1, order='A')).array  # FIXME: put back to ap.where
        points = np.where(skeleton.reshape(-1, order='A'))[0]

        if verbose: timer.print_elapsed_time('Point list generation', reset=True)

    # create graph
    n_vertices = points.shape[0]
    g = graph_gt.Graph(n_vertices=n_vertices, directed=False)
    if spacing:
        spacing = np.asarray(spacing, dtype=np.float64)
        g.add_graph_property('spacing', spacing)
    g.shape = skeleton.shape

    if verbose: timer.print_elapsed_time(f'Graph initialized with {n_vertices:,} vertices', reset=True)

    # ######################### detect edges #########################
    edges_all = np.zeros((0, 2), dtype=int)  # TODO: list and stack later
    for i, o in enumerate(t3d.orientations()):
        offset = np.sum((np.hstack(np.where(o)) - [1, 1, 1]) * skeleton.strides)
        # edges = ap.neighbours(points, offset)
        edges = neighbours(points, offset)
        if len(edges) > 0:
            edges_all = np.vstack([edges_all, edges])

        if verbose:
            timer.print_elapsed_time(f'{edges.shape[0]:,} edges with orientation {i + 1}/13 found', reset=True)

    if edges_all.shape[0] > 0:
        g.add_edge(edges_all)

    if verbose: timer.print_elapsed_time(f'Added {edges_all.shape[0]:,} edges to graph', reset=True)

    # ################## Compute additional properties ##################

    if compute_vertex_coordinates:
        vertex_coordinates = np.array(np.unravel_index(points, skeleton.shape, order=skeleton.order)).T

        coords_dtype = np.int16   #  unsigned ints not handled by graph-tool
        if vertex_coordinates.max() > np.iinfo(np.int16).max:
            coords_dtype = np.int32
        vertex_coordinates = vertex_coordinates.astype(coords_dtype)
        g.set_vertex_coordinates(vertex_coordinates, dtype=coords_dtype)

        if spacing is not None:
            # Upcast to float because result is in physical units
            coords_phys = vertex_coordinates.astype(np.float32) * spacing[None, :]  # None broadcasts spacing to (1, 3)
            g.define_vertex_property('coordinates_units', coords_phys)

    if radii is not None:
        g.set_vertex_radius(radii)

    if compute_edge_length:
        edge_lengths = annotate_edge_lengths(g, g.vertex_coordinates(), spacing=spacing)
        g.define_edge_property('length', edge_lengths)

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


def clean_graph(graph: graph_gt.Graph, remove_self_loops: bool = True, remove_isolated_vertices: bool = True,
                vertex_mappings: dict[str, Callable] | None = None, verbose: bool = False) -> graph_gt.Graph:
    """
    Remove all cliques to get pure branch structure of a graph.

    A clique is a cluster of vertices, defined as a connected component of branch points

    .. warning::

        This function is meant to be used on graphs **before** reduction, in
        which case clusters of branch points are **immediately adjacent** to each other.

    .. note::

        cliques are replaced by a single vertex connecting to all non-clique neighbours
        The center coordinate is used as the coordinate for that vertex.
        The vertex properties are reduced using the provided `vertex_mappings` functions.
        The edge properties are not reduced. The length edge property is computed as the
        length of the edge connecting the clique center to its neighbours.


    Arguments
    ---------
    graph : Graph
        The graph to clean up.
    remove_self_loops: bool
        If True, remove self-loops from the graph.
    remove_isolated_vertices: bool
        If True, remove isolated vertices from the graph.
    vertex_mappings : dict[str, Callable] | None
        Dictionary of reduction functions to apply to vertex properties within cliques.
    verbose : bool
        If True, prin progress information.

    Returns
    -------
    graph : Graph
        A graph removed of all cliques.
    """
    if vertex_mappings is None:
        vertex_mappings = DEFAULT_VERTEX_TO_VERTEX
    vertex_mappings = {name: fn for name, fn in vertex_mappings.items() if name in graph.vertex_properties}

    if verbose:
        timer = tmr.Timer()
        timer_all = tmr.Timer()
    else:
        timer = None

    # find branch points
    n_vertices = graph.n_vertices
    branch_mask = graph.vertex_degrees() >= 3
    n_branch_points = branch_mask.sum()

    if verbose:
        timer.print_elapsed_time(f'Graph cleaning: found {n_branch_points:,} branch points '
                                 f'among {n_vertices:,} vertices', reset=True)

    if n_branch_points == 0:
        return graph.copy()

    # detect 'cliques', i.e. connected components of branch points
    # First we remove non-branch points from the graph, so that the graph
    # now becomes disjoint between clusters of branch points which we can now label.
    gb = graph.sub_graph(vertex_filter=branch_mask, view=True)
    component_ids, component_sizes = gb.label_components(return_vertex_counts=True)

    # note: graph_tools components of a view is a property map of the full graph
    component_ids[branch_mask] += 1 # shift so non-branch = 0

    # group components
    # note: remove the indices of the non branch nodes (0 label)
    component_ids = _group_labels(component_ids)[1:]

    clique_ids = np.where(component_sizes > 1)[0]  # clusters (>1) of degree > 2 vertices
    n_cliques = len(clique_ids)

    if verbose:
        timer.print_elapsed_time(f'Graph cleaning: detected {n_cliques} cliques of branch points', reset=True)

    if n_cliques == 0:
        return graph.copy()

    # Create temp graph with clique vertices + new vertex for each clique
    # g = graph.copy(from_disk=True)  # WARNING: trick to avoid runaway memory usage
    g = graph.copy()
    g.add_vertex(n_cliques)

    vertex_filter = np.ones(n_vertices + n_cliques, dtype=bool)
    coords = g.vertex_coordinates()
    center_vertices = np.arange(n_vertices, n_vertices + n_cliques, dtype=np.intp)
    neighbours = {}
    for clk_id in clique_ids:
        clique_vertices = component_ids[clk_id]
        neighbours[clk_id] = get_clique_neighbours_set(graph, clique_vertices)

    def _prop_to_buffer(prop_map, n_rows):
        """return an empty (n_rows, …) array whose tail shape matches one value"""
        sample = prop_map[0]  # any vertex will do
        return np.empty((n_rows, *(sample.shape)), dtype=(sample.dtype))

    n_new_edges = sum(neighbours[clik_id].shape[0] for clik_id in clique_ids)
    new_edges = np.empty((n_new_edges, 2), dtype=np.intp)
    if 'length' in g.edge_properties:
        new_lengths = np.zeros((n_new_edges,), dtype=np.float64)
    new_vertices_props = {name: _prop_to_buffer(graph.vertex_property(name), n_cliques) for name in vertex_mappings}
    edge_counter = 0
    for i, clik_id in enumerate(clique_ids):
        center_vertex = center_vertices[i]  # new vertex for the clique
        clique_vertices = component_ids[clik_id]  # TODO check if cached list of arrays faster
        vertex_filter[clique_vertices] = False # Schedule clique vertices for removal
        current_neighbours = neighbours[clik_id]
        if current_neighbours.size == 0:  # unconnected clique
            vertex_filter[center_vertex] = False  # Remove altogether
            continue

        # connect to new node
        edges_to_clique_neighbours = [[n, center_vertex] for n in current_neighbours]
        new_edges[edge_counter: edge_counter + len(edges_to_clique_neighbours)] = edges_to_clique_neighbours

        # Existing median coordinate of the clique vertices
        # center_coord = np.quantile(coords[clique_vertices], 0.5, method='nearest', axis=0)
        center_coord = medoid_vertex_coordinates(coords[clique_vertices])
        if 'length' in g.edge_properties:
            edge_lengths = np.linalg.norm(coords[current_neighbours] - center_coord, axis=1)
            new_lengths[edge_counter: edge_counter + len(edges_to_clique_neighbours)] = edge_lengths

        edge_counter += len(edges_to_clique_neighbours)

        # map vertex properties
        for prop_name, reduce_fn in vertex_mappings.items():
            if prop_name == 'coordinates':  # Already calculated above
                val = center_coord
            else:
                prop = graph.vertex_property(prop_name)
                val = reduce_fn(prop[clique_vertices])
            # g.set_vertex_property(prop_name, val, vertex=center_vertex)
            new_vertices_props[prop_name][i] = val  # Store for later

        if verbose and i+1 % 10000 == 0:
            timer.print_elapsed_time(f'Graph cleaning: reducing {i + 1} / {n_cliques}')

    # map vertex properties to new vertices
    for prop_name in vertex_mappings.keys():
        prop = g.vertex_property(prop_name)  # OPTIMISE: see if we could do in-place using vertex_property_map instead
        prop[n_vertices:] = new_vertices_props[prop_name]  # Add for all the cliques at once
        g.set_vertex_property(prop_name, prop)

    # Add new edges and map e_props
    if 'length' in g.edge_properties:
        length_prop = g.edge_property('length')
    g.add_edge(new_edges)
    if 'length' in g.edge_properties:
        g.set_edge_property('length', np.hstack([length_prop, new_lengths]))

    # Remove vertices belonging to cliques
    g = g.sub_graph(vertex_filter=vertex_filter)

    if remove_self_loops:
        g.remove_self_loops()

    if verbose:
        timer.print_elapsed_time(
            f'Graph cleaning: removed {n_cliques} cliques of branch points from {graph.n_vertices:,} vertices'
            f' to {g.n_vertices:,} nodes and {graph.n_edges:,} to {g.n_edges:,} edges', reset=True)

    # remove isolated vertices
    if remove_isolated_vertices:
        g = _remove_isolated_vertices(g, timer, verbose)  # REFACTORING: part of graph_gt.Graph

    if verbose:
        timer_all.print_elapsed_time(
            f'Graph cleaning: cleaned graph has {g.n_vertices} nodes and {g.n_edges} edges')

    return g


def get_clique_neighbours(graph, clique_vertices):
    neighbours = np.hstack([graph.vertex_neighbours(v) for v in clique_vertices])  # Clique + neighbours
    neighbours = np.setdiff1d(np.unique(neighbours), clique_vertices, assume_unique=True)  # Remove clique vertices (-> neighbours only)
    return neighbours


def get_clique_neighbours_set(g, clique_vertices):
    s_clique = set(clique_vertices)
    s_neigh   = set(np.concatenate([g.vertex_neighbours(v) for v in clique_vertices]).flatten())  # Clique + neighbours
    # for v in clique_vertices:
    #     s_neigh.update(g.vertex_neighbours(v))
    s_neigh.difference_update(s_clique)
    return np.fromiter(s_neigh, dtype=int)


def _remove_isolated_vertices(g, timer, verbose):
    non_isolated = g.vertex_degrees() > 0
    g = g.sub_graph(vertex_filter=non_isolated)
    if verbose:
        timer.print_elapsed_time(f'Graph cleaning: Removed {np.logical_not(non_isolated).sum()} isolated nodes',
                                 reset=True)
    return g


def _group_labels(array):
    """Helper grouping labels in a 1d array."""
    idx = np.argsort(array)
    arr = array[idx]
    dif = np.diff(arr)
    groups = np.split(idx, np.where(dif > 0)[0]+1)
    return groups


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
    comp_view = comp_full[view._base.get_vertices()]  # WARNING: implementation specific
    loop_mask = np.isin(comp_view, cycle_ids)

    # 6) map back to original graph indices
    orig_vids = view._base.get_vertices()  # WARNING: implementation specific
    to_remove = orig_vids[loop_mask]

    # 7) build the keep mask for the original graph
    v_keep = np.ones(graph.n_vertices, dtype=bool)
    v_keep[to_remove] = False

    print(f"Dropping {len(to_remove)} pure degree-2 loop vertices from {graph.n_vertices} total.")
    return graph.sub_graph(vertex_filter=v_keep)


class PropertyAggregator:
    ALLOWED_KINDS = ('edge', 'vertex', 'edge_geometry')
    """
    Collect & aggregate graph-tool properties while you walk chains.

    Parameters
    ----------
    graph: GraphGt.Graph
        The working copy of the graph (`g` in reduce_graph).
    mapping: Dict[str, Callable]
        Keys are property names (e.g. "length"), values are aggregation
        functions (e.g. np.sum, np.max …).
    kind: str
        "edge"  → aggregating edge properties
        "vertex"→ aggregating vertex properties
    """
    def __init__(self, graph: graph_gt.Graph, mapping: Dict[str, Callable], kind: str = "edge"):
        self.offsets = None  # offsets for the chains, (i.e., cumsum of chain lengths -> start/end indices)
        if kind not in self.ALLOWED_KINDS:
            raise ValueError(f"kind must be in {self.ALLOWED_KINDS}")

        self.src_graph = graph
        self.kind  = kind
        self.chain_indices: list[np.ndarray] = []  # collects indices of chains
        self.chain_edges_direction: list[np.ndarray] = []  # collection of boolean arrays indicating the need to reverse an edge direction in the chain

        self.properties:            Dict[str, np.ndarray] = {}
        self.aggregation_functions: Dict[str, Callable] = {}
        self.aggregated_properties: Dict[str, List] = {}

        base = (graph.edge_properties if kind == 'edge' else
                graph.vertex_properties if kind == 'vertex' else
                graph.edge_geometry_property_names)

        for key, func in mapping.items():
            if key in base:  # Ensure the property exists in the graph
                if self.kind == "edge":
                    prop = graph.edge_property_map(key).a
                elif self.kind == "vertex":
                    prop = graph.vertex_property_map(key).a
                else:  # edge_geometry
                    prop = graph.edge_geometry_property(key)  # FIXME: this one is not dynamic
                self.properties[key] = prop
                self.aggregation_functions[key] = func
                self.aggregated_properties[key] = []
            else:
                warnings.warn(f'Property "{key}" not found in graph {self.kind} properties. Skipping aggregation.')

    def __str__(self):
        return f"PropertyAggregator(kind={self.kind}, properties={list(self.properties.keys())})"

    @property
    def property_names(self):
        return list(self.properties.keys())

    @functools.cached_property
    def edge_connectivity(self) -> np.ndarray:
        return self.src_graph.edge_connectivity()

    def accumulate(self, indices: Sequence[int]) -> None:
        """
        Append the result of aggregating (applying the specified function)
        a chain of vertices or edges to the aggregator.

        indices : list/array of edge IDs (kind="edge")
                  or vertex IDs (kind="vertex") that form one chain.
        """
        self.chain_indices.append(np.asarray(indices, dtype=np.intp))

        # if self.kind == 'edge':
        edge_orders = self.edge_connectivity[indices]
        sorted_vertices = np.sort(edge_orders, axis=1)
        diffs       = edge_orders[:, 1] - edge_orders[:, 0]
        first_sign  = np.sign(diffs[0])  # +, - (or 0 for self-loops)
        need_flip   = np.sign(diffs) != first_sign
        is_not_loop = np.all(sorted_vertices[0, :] != sorted_vertices[-1, :])
        need_flip = need_flip & is_not_loop  # no flip for self-loops
        if self.kind == 'edge'and len(need_flip) > 0:
            need_flip = np.hstack([need_flip, need_flip[-1]])  # FIXME: dirty trick for edge/vertex geometry length mismatch
        self.chain_edges_direction.append(need_flip)

    def aggregate(self) -> None:  # FIXME: document (especially cython vs numpy)
        if not self.chain_indices:
            raise ValueError("No chains have been accumulated. Call accumulate() first.")

        idx_stack = np.concatenate(self.chain_indices).astype(np.uint64)           # 1-D
        offsets = np.cumsum([0] + [len(x) for x in self.chain_indices]).astype(np.uint64)
        self.offsets = offsets

        n_procs = multiprocessing.cpu_count() - 2

        for prop_name, arr in self.properties.items():
            reduction_fn = self.aggregation_functions[prop_name]

            out_dtype = np.float64 if reduction_fn is np.sum else arr.dtype

            mapped = np.zeros(len(self.chain_indices), dtype=out_dtype)  # pre-allocate output array
            success = cy_reduce(arr, mapped, idx_stack=idx_stack, offsets=offsets, reducer_fn=reduction_fn,
                                num_threads=n_procs)
            if not success:  # default to pure Python if Cython fails
                starts = offsets[:-1]
                ends = offsets[1:]
                mapped = np.array([reduction_fn(arr[idx_stack[s:e]]) for s, e in zip(starts, ends)], dtype=out_dtype)
            self.aggregated_properties[prop_name] = mapped

    # def reorder_indices(self, edge_order: np.ndarray) -> None:
    #     self.chain_indices = (np.asarray(self.chain_indices, dtype=object)[edge_order]).tolist()
    #     # PERFORMANCE: see if we can skip to_list if we amend stacked_indices() later

    def get_indices_and_ranges(self, reduced_edge_order) -> (np.ndarray, np.ndarray):
        """
        Return a 1D array of all accumulated indices, concatenated from all chains.
        and the start and end indices for each chain as a 2D array.
        """
        start_idx = self.offsets[:-1]
        end_idx = self.offsets[1:]
        edge_geom_ranges = np.array([start_idx, end_idx]).T  # how to chunk edge_agg.chain_indices
        edge_geom_ranges = edge_geom_ranges[reduced_edge_order]  # Reorder chains according to the new edge order
        return np.hstack(self.chain_indices), edge_geom_ranges  # WARNING: reorder only slicing, not array itself

    def apply(self, reduced_graph: graph_gt.Graph, reduced_edge_order: np.ndarray) -> None:
        """
        Attach the aggregated values as edge properties on `graph`.

        .. note::
            This method is typically called once after `reduce_graph` has
            finished processing the chains and before returning the reduced graph.

        Parameters
        ----------

        reduced_graph : GraphGt.Graph
            The graph to which the aggregated properties will be added.
            This is typically the reduced graph returned by `reduce_graph`.
        reduced_edge_order : np.ndarray
            The *new* edge order, i.e. the order in which edges are stored in the reduced graph.
        """
        # Handle existing edge geometry properties  # FIXME: only if self.kind == "edge" + should be part of other instance with != kind
        if self.src_graph.has_edge_geometry() and self.src_graph.edge_geometry_type == "graph":
            if self.kind == 'edge' and not reduced_graph.has_edge_geometry():
                # FIXME: src_ranges from the new graph because we drop the reduced edges ????
                src_ranges = self.src_graph.edge_geometry_indices()  # (E_orig, 2)

                # Reorder the edges according to the reduced edge order
                chains = [self.chain_indices[i] for i in reduced_edge_order]
                flip_masks = [self.chain_edges_direction[i] for i in reduced_edge_order]

                # Build the take index (ordered list of *each* index in the chains concatenated)
                take_idx: List[int] = []
                new_ranges = np.full((len(chains), 2), SENTINEL, dtype=np.intp)
                for i, (edges, flips) in enumerate(zip(chains, flip_masks)):
                    current_start = len(take_idx)
                    for j, (e_id, flip) in enumerate(zip(edges, flips)):
                        start_idx, end_idx = src_ranges[e_id]  # old [start, end]
                        # Build the whole itemwise index for the chain
                        idx = range(start_idx, end_idx, 1)
                        if flip:
                            idx = idx[::-1]

                        if j > 0 and take_idx[-1] == idx[0]:  # Avoid duplicates
                            idx = idx[1:]

                        take_idx.extend(idx)

                    new_ranges[i] = (current_start, len(take_idx))

                reduced_graph.define_edge_geometry_indices_graph(new_ranges)
                take_idx = np.fromiter(take_idx, dtype=np.intp)
                # Reorder the edge_geometry properties according to the new edge order
                for name in self.src_graph.edge_geometry_property_names:
                    arr = self.src_graph.graph_property(name)
                    new_arr = arr.take(take_idx, axis=0)
                    reduced_graph.define_graph_property(name, new_arr)
            else:  # Handle the case where edge geometry is already defined (after edges have been reduced)
                pass  # FIXME: implement case where edge_geometry_vertex_property are defined separately
                # new_ranges = reduced_graph.edge_geometry_indices()
                # take_idx = np.concatenate([np.arange(start, end) for start, end in new_ranges])

        # Handle "regular" aggregated properties
        for prop_name, values in self.aggregated_properties.items():
            reordered_values = values[reduced_edge_order]  # Reorder according to the edge order
            reduced_graph.define_edge_property(prop_name, reordered_values)


def __drop_former_degree_2_nodes(graph, reduced_graph, non_degree_2_vertices_ids, return_maps):
    """remove former degree 2 nodes (now isolated vertices)"""
    v_to_v_map = None
    if return_maps:
        reduced_graph.add_vertex_property('_vertex_id_', graph.vertex_indices())
    v_filter = np.zeros(graph.n_vertices, dtype=bool)
    v_filter[non_degree_2_vertices_ids] = True
    reduced_graph = reduced_graph.sub_graph(vertex_filter=v_filter)
    if return_maps:
        v_to_v_map = reduced_graph.vertex_property('_vertex_id_')
        reduced_graph.remove_vertex_property('_vertex_id_')
    return reduced_graph, v_to_v_map


def add_chain_id(graph, prop_kind):
    # Store int as float as a sentinel for mean
    if prop_kind == 'edge':
        graph.define_edge_property('chain_id', np.full(graph.n_edges, -1, dtype=np.int64))
        prop = graph.edge_property_map('chain_id').a
    elif prop_kind == 'vertex':
        graph.define_vertex_property('chain_id', np.full(graph.n_vertices, -1, dtype=np.int64))
        prop = graph.vertex_property_map('chain_id').a
    else:
        raise ValueError(f'Unknown property kind "{prop_kind}". Expected "edge" or "vertex".')
    return prop


def reduce_graph(graph, vertex_to_edge_mappings=None,
                 edge_to_edge_mappings=None,
                 compute_edge_length=False,
                 compute_edge_geometry=True,
                 edge_geometry_vertex_properties=('coordinates', 'radii', 'chain_id', '_vertex_id_'),
                 edge_geometry_edge_properties=('chain_id', ),
                 return_maps=False, drop_pure_degree_2_loops=True,
                 verbose=False, label_branches=False, save_modified_graph_path=''):
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
    graph: GraphGt.Graph
        The graph to reduce.  WARNING: currently, existing degree 2 loops are dropped inplace
    vertex_to_edge_mappings: dict | None
        A dictionary mapping vertex properties to edge properties. The keys are the vertex property names,
        and the values are functions that aggregate the vertex properties to edge properties.
        Defaults to `DEFAULT_VERTEX_TO_EDGE`
        Supply an empty dictionary to disable vertex to edge mappings.
    edge_to_edge_mappings: dict | None
        A dictionary mapping edge properties to edge properties. The keys are the edge property names,
        and the values are functions that aggregate the edge properties.
        Defaults to `DEFAULT_EDGE_TO_EDGE`
        Supply an empty dictionary to disable edge to edge mappings.
    compute_edge_length: bool
        If True, compute the length of edges in the reduced graph.
        Mapping defaults to np.sum, i.e. the length of the edge is the sum of the lengths of
        the edges in the original graph.
    compute_edge_geometry: bool
        If True, compute edge geometry properties for the reduced graph, i.e.
        store the aggregated selected vertex (see `edge_geometry_vertex_properties`) and
        edge properties (see `edge_geometry_edge_properties`) in the edge geometry.
    edge_geometry_vertex_properties: tuple | list | None
        A tuple of vertex property names that will be aggregated and
        stored in the edge geometry.
        If None, no vertex properties are used for edge geometry.
        Ignored if `compute_edge_geometry` is False.
    edge_geometry_edge_properties: tuple | list | None
        A list of edge property names that will be aggregated and
        stored in the edge geometry.
        If None, no edge properties are used for edge geometry.
        Ignored if `compute_edge_geometry` is False.
    return_maps: bool
        If True, return mappings of vertex to vertex, edge to vertex, and edge to edge.
        These mappings are useful for further processing of the reduced graph.
    drop_pure_degree_2_loops: bool
        If True, drop all pure degree 2 loops from the graph before reducing it.
        This is useful to avoid issues with loops in the graph that would otherwise
        lead to incorrect results.
    save_modified_graph_path: str | PathLike
        If provided, save the original graph after modification (without degree 2 loops and after
        branch labelling) to this path.
    label_branches: bool
        If True, label the branches in the graph with a unique chain ID.
    verbose: bool
        If True, print progress information during the reduction process.

    Returns
    -------
    reduced_graph : GraphGt.Graph
        The reduced graph with degree 2 vertices removed and properties aggregated.
    """
    save_modified_graph_path = str(save_modified_graph_path)  # So we can check if empty string

    if verbose:
        timer = tmr.Timer()
        timer_all = tmr.Timer()
        print('Graph reduction: initialized.')

    check_graph_is_reduce_compatible(graph)

    if drop_pure_degree_2_loops:
        graph = drop_degree2_loops(graph)  # WARNING: modifies graph in place

    # Find potential chains starting points
    non_degree_2_vertices_ids = np.where(graph.vertex_degrees() != 2)[0]

    if verbose: print_graph_info(graph, timer, non_degree_2_vertices_ids)

    # Setup mappings
    vertex_to_edge_mappings = vertex_to_edge_mappings if vertex_to_edge_mappings is not None else DEFAULT_VERTEX_TO_EDGE
    edge_to_edge_mappings = edge_to_edge_mappings if edge_to_edge_mappings is not None else DEFAULT_EDGE_TO_EDGE
    edge_geometry_vertex_properties = edge_geometry_vertex_properties or []
    edge_geometry_edge_properties = edge_geometry_edge_properties or []
    reduced_maps = {k: [] for k in ('vertex_to_vertex', 'edge_to_vertex', 'edge_to_edge')}  # TODO: see if part of egeom aggregators

    if compute_edge_length or edge_to_edge_mappings['length']:
        if 'length' not in graph.edge_properties:
            warnings.warn(f'Reducing a graph without preexisting edge lengths is now deprecated. '
                          f'Please ensure you use "compute_edge_length=True" in skeleton_to_graph() '
                          f'before running clean_graph and reduce_graph.',)
        edge_to_edge_mappings['length'] = edge_to_edge_mappings.get('length', np.sum)

    if label_branches:
        chain_id_prop_arr = add_chain_id(graph, prop_kind='edge')
        chain_id_vertex_prop_arr = add_chain_id(graph, prop_kind='vertex')

    vertex_agg = PropertyAggregator(graph, vertex_to_edge_mappings, kind="vertex")
    edge_agg = PropertyAggregator(graph, edge_to_edge_mappings, kind="edge")
    # vertex_geometry_agg =
    # edge_geometry_agg =

    chains, _ = find_chains(graph) #, return_endpoints_mask=True);  check_chains(g, chains, degree_2_vertices_ids, non_degree_2_vertices_ids)

    direct_edges = []  # Only the direct edges between non-degree 2 vertices

    edge_old_to_new = np.full(graph.n_edges, -1, dtype=np.intp)
    for chain_id, (edges, vertices) in enumerate(chains):
        edge_old_to_new[edges] = chain_id  # Map old edge id to new edge id
        if label_branches:
            chain_id_prop_arr[edges] = chain_id  # Assign chain id to edges
            chain_id_vertex_prop_arr[vertices] = chain_id

        direct_edges.append([vertices[0], vertices[-1]])

        # mappings
        vertex_agg.accumulate(vertices)  # TODO: check if vertices or vertices_ids should be used
        edge_agg.accumulate(edges)

    if save_modified_graph_path: graph.save(save_modified_graph_path)

    # Reduce each property to (typically) a single value per chain
    vertex_agg.aggregate()
    edge_agg.aggregate()

    # Create a copy with all vertices and properties, but no edges
    reduced_graph = graph.sub_graph(edge_filter=np.zeros(graph.n_edges, dtype=bool))  # Delete all edges but keep vertices and properties
    reduced_graph.add_edge(direct_edges) #  Add new direct edges
    # edge_order = edge_old_to_new[reduced_graph.edge_indices()]  # Reorder edges according to the new edge ids
    edge_order = reduced_graph.edge_indices()  # Reorder edges according to the new edge ids

    reduced_graph, v_to_v_map = __drop_former_degree_2_nodes(graph, reduced_graph, non_degree_2_vertices_ids, return_maps)
    if return_maps: reduced_maps['vertex_to_vertex'] = v_to_v_map

    # Add edge properties
    edge_agg.apply(reduced_graph, edge_order)  # Will also apply the edge geometry properties if they exist
    vertex_agg.apply(reduced_graph, edge_order)

    # Concatenate edge_geometry_properties to a graph property and keep ranges as edge geometry indices (edge property)
    if compute_edge_geometry:   # Remap each vertex or edge property to the reduced graph (i.e., save as f'edge_geometry_{prop}' )
        branch_indices, edge_geom_ranges = vertex_agg.get_indices_and_ranges(edge_order)

        reduced_graph.set_edge_geometry_vertex_properties(graph, edge_geometry_vertex_properties,  # REFACTOR: pass aggregator
                                                          branch_indices, edge_geom_ranges)
        if edge_geometry_edge_properties:
            reduced_graph.set_edge_geometry_edge_properties(graph, edge_geometry_edge_properties,  # REFACTOR: pass aggregator
                                                            edge_geom_ranges, edge_agg.chain_indices)

    if return_maps: reduced_maps['edge_to_edge'] = edge_agg.chain_indices

    if verbose:
        timer_all.print_elapsed_time(f'Graph reduction: Graph reduced '
                                     f'from {graph.n_vertices} to {reduced_graph.n_vertices} nodes '
                                     f'and {graph.n_edges} to {reduced_graph.n_edges} edges')

    if return_maps:
        return reduced_graph, list(reduced_maps.values())  # We rely on insertion order of the dict
    else:
        return reduced_graph


def print_graph_info(graph, timer, non_degree_2_vertices_ids):
    n_branch_points = len(non_degree_2_vertices_ids)
    degree_2_vertices_ids = np.where(graph.vertex_degrees() == 2)[0]
    n_degree_2_vertices = len(degree_2_vertices_ids)
    timer.print_elapsed_time(
        f'Graph reduction: Found {n_branch_points} branching and {n_degree_2_vertices} non-branching nodes')
    timer.reset()


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

    edge_descriptors = np.array(list(graph._base.edges()), dtype=object)

    # Add direct edges to chains to process together
    direct_edges = connectivity_w_eid[(v1_degs != 2) & (v2_degs != 2)]
    chains.extend(([eid], np.array([v1, v2], dtype=int)) for eid, v1, v2 in direct_edges)

    # Add pure degree2 loops
    visited = np.zeros(graph.n_vertices, dtype=bool)
    visited_vs_idx = np.hstack([vs for _, vs in chains], dtype=int)
    visited[visited_vs_idx] = True

    unvisited_deg2_v_idx = np.where(~visited & (vertex_degs == 2))[0]
    if unvisited_deg2_v_idx.size:
        print(f'Found {len(unvisited_deg2_v_idx)} degree 2 vertices not visited in chains. '
              f'They will be processed as isolated loops.')
    for v0 in unvisited_deg2_v_idx:
        if visited[v0]:  # in case marked within this very loop
            continue
        vs, es, ends, _isolated = _graph_branch(graph.vertex(v0))

        vs_idx = np.array(vs, dtype=int)
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
    v_idx = graph._base.get_vertices()
    if graph.is_view:
        raise ValueError('Cannot process on graph view, prune graph before graph reduction.')
    if v_idx[0] != 0 or not np.all(np.diff(v_idx) == 1):
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

    gr = gp.reduce_graph(gc, compute_edge_geometry=True, verbose=True)
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
