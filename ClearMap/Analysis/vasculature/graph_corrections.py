import multiprocessing
import os.path
import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from ClearMap.Alignment import Annotation as annotation
from ClearMap.Analysis.vasculature.vasc_graph_utils import remove_surface, vertex_filter_to_edge_filter

print('Loading ClearMap modules, please wait ...')
import ClearMap.Analysis.Graphs.GraphGt as ggt

CPU_COUNT = multiprocessing.cpu_count()


def join_neighbouring_degrees_1(graph, min_radius=5, dest_path=''):
    """
    Join degrees one (empty ended branches) that are very close and likely
    interrupted by a thresholding issue. The distance criterion is defined
    by min_radius.

    .. warning::
        This is meant to be used on the GRAPH BEFORE REDUCTION

    graph : GraphGt.Graph
        the graph to fix
    dest_path : str
        The optional path to save the modified graph
    min_radius : float
        The radius around a degree 1 vertex to search for neighbouring vertices

    returns : GraphGt.Graph
        The modified graph
    """
    df = pd.DataFrame()
    degrees_1_mask = graph.vertex_degrees() == 1
    df[["x", "y", "z"]] = graph.vertex_coordinates()[degrees_1_mask]
    degree_1_indices = np.where(degrees_1_mask)[0]
    df['vertex_id'] = degree_1_indices
    # find the nearest degree 1 for each degree 1 vertex
    knn = NearestNeighbors(n_neighbors=2, radius=min_radius + 1,
                           n_jobs=CPU_COUNT - 2)  # we take the two nearest "neighbors" : itself and the nearest degree1
    knn.fit(df[["x", "y", "z"]])
    dist, indices = knn.kneighbors(df[["x", "y", "z"]], return_distance=True)
    dist = dist[:, 1]  # throw away the vertex itself
    indices = indices[:, 1]  # throw away the vertex itself
    df["neighbor_id"] = -1
    close_neighbors_mask = dist < min_radius
    df.loc[close_neighbors_mask, "neighbor_id"] = degree_1_indices[indices[close_neighbors_mask]]
    df = df[df["neighbor_id"] != -1]
    edges_array = df[['vertex_id', 'neighbor_id']].values
    edges_array.sort(axis=1)
    # remove double edges
    edges = list(set([tuple(e) for e in edges_array]))

    # we reconnect vertices of degree 1 to the nearest degree 1
    graph.add_edge(edges)
    if dest_path:
        graph.save(dest_path)
    return graph


def remove_spurious_branches(graph, r_min=None, min_length=1.0, view=False):  #  WARNING: isolated vertices are not removed
    """
    Removes spurious branches from the graph.
    Spurious branches are defined as small degree 1 branches (with a radius smaller than r_min).

    Parameters
    ----------
    graph : Graph object
        The graph to consider
    r_min : float
        The minimum radius to consider
    min_length : float
        The minimum length of the branch

    Returns
    -------
        The graph with the spurious branches removed
    """
    degrees_filter = graph.vertex_degrees() == 1
    degrees_filter = vertex_filter_to_edge_filter(graph, degrees_filter, operator=np.logical_or)
    if r_min is not None:
        # radii_filter = graph.edge_radii() < r_min
        raise NotImplementedError('Radii are not implemented yet because we need to decide how they combine '
                                  '("or" or "and") with the length')
    length_filter = graph.edge_property('length') < min_length

    edge_filter = np.logical_not(degrees_filter & length_filter)

    return graph.sub_graph(edge_filter=edge_filter, view=view)


def remove_auto_loops(graph, min_length=None):
    """
    Removes auto loops from the graph.
    Auto loops are defined as edges with the same source and destination.

    Parameters
    ----------
    graph : Graph object
        The graph to consider
    min_length : float
        The minimum length to consider

    Returns
    -------
        The graph with the auto loops removed
    """
    connectivity = graph.edge_connectivity()
    auto_loops_mask = ((connectivity[:, 0] - connectivity[:, 1]) == 0)
    if min_length is None:
        bad_length_mask = np.zeros(graph.n_edges, dtype=bool)
    else:
        lengths = graph.edge_property('length')
        bad_length_mask = lengths < min_length

    auto_loops_filter = np.logical_not(auto_loops_mask & bad_length_mask)

    return graph.sub_graph(edge_filter=auto_loops_filter)


def remove_mutual_loops_gt(graph, min_radius, min_length):  # TODO: what are the defaults 5  & 3 ?
    """
    Removes mutual loops from the graph.
    Mutual loops are defined as edges with the same source and destination. In other words, two edges that
        connect the same pair of vertices in the graph form a mutual loop.
    This function uses a parallelized version of mutual_loop_detection.

    .. warning::
        There is a substantial difference between this function and a
        standard parallel edge filter. Contrary to a standard parallel edge filter,
        this function takes into account the length, i.e. the distance between
        the two vertices along the tract of the vessel, and the radius, i.e. the size of the vessel.
        For each pair of vertices where these two conditions are met,
        only the edge with the smallest radius will be removed.

    Parameters
    ----------
    graph : Graph object
        The graph to consider
    min_radius : float
        The minimum radius to consider
    min_length : float
        The minimum length to consider

    Returns
    -------
        The graph with the mutual loops removed
    """
    # mutual_loops = graph_tool.generation.label_parallel_edges(graph, mark_only=True)

    radii = graph.edge_radii()
    lengths = graph.edge_geometry_lengths()

    edge_group_ids = np.sort(graph.edge_connectivity(), axis=1)
    # return the indices of unique values. This labels edges belonging to the same group with the same id
    _, edge_group_ids = np.unique(edge_group_ids, axis=0, return_inverse=True)

    df = pd.DataFrame({
        'edge_id': np.arange(graph.n_edges),
        'edge_group_id': edge_group_ids,
        'radius': radii,
        'length': lengths
    })
    good_edges_indices = (df
                          .sort_values("radius", ascending=False)
                          .groupby('edge_group_id')  # This will only give one element for single edges hence kept
                          .first())["edge_id"].values

    single_edges = df.groupby('edge_group_id').count() == 1
    single_edges = single_edges['edge_id'].values

    parallel_edges = (df.groupby('edge_group_id').count() != 1)['edge_id'].values
    assert single_edges.sum() + parallel_edges.sum() == graph.n_edges   #FIXME: not True

    # Go back to mask with same size as the original graph from indices of the edges to remove
    good_edges = np.zeros(graph.n_edges, dtype=bool)
    good_edges[good_edges_indices] = 1

    large_edges = (radii >= min_radius) | (lengths >= min_length)
    edge_filter = large_edges | good_edges #| (mutual_loops == 0)
    # edge_filter = np.ones(graph.n_edges)
    # edge_filter[bad_edges] = 0
    return graph.sub_graph(edge_filter=edge_filter)


def correct_graph(graph, min_radius=2.9, min_length=13):  # FIXME: reinstate min_length
    """
    This is the main function to correct the graph.
    Corrects the graph by removing
        * spurious branches (small degree 1 branches)
        * surface branches (branches that are too close to the surface)
        * auto loops (edges where edge.source == edge.target)
        * mutual loops (edges share the same source and target and are small(in radius and length))

    Parameters
    ----------
    graph
    min_radius
    min_length

    Returns
    -------
        The corrected graph

    """
    corrected_graph = graph.largest_component()

    print_percent_degree(corrected_graph, 5)  # @Sophie: why 5 ??

    corrected_graph = remove_spurious_branches(corrected_graph, r_min=min_radius)
    corrected_graph = remove_surface(corrected_graph, 2)
    corrected_graph = corrected_graph.largest_component()

    corrected_graph = remove_auto_loops(corrected_graph)
    corrected_graph = remove_mutual_loops_gt(corrected_graph, min_radius, 5)

    print_percent_degree(corrected_graph, 5)  # @Sophie: why 5 ??

    return corrected_graph


def print_percent_degree(graph, degree=4):
    """
    Print the number of nodes with a degree greater or equal to degree.

    Parameters
    ----------
    graph : Graph object
        The graph to consider
    degree : int
        The degree to consider

    Returns
    -------

    """
    percent_greater_degrees = np.sum(graph.vertex_degrees() >= degree) / graph.n_vertices * 100
    print(f'Percent of degree {degree}+ nodes: {percent_greater_degrees:.2f}%')


def test():
    region_list = [(0, 0)]
    mutants = ['7o', '8c']
    save = False
    if len(sys.argv) > 1:
        work_dir = sys.argv[1]
    else:
        work_dir = '/media/sophie.skriabine/sophie/HFD_VASC'
    for sample_id in mutants:
        sample_dir = os.path.join(work_dir, f'{sample_id}')
        print(f'Processing {sample_dir} ...')
        sample_graph = ggt.load(os.path.join(sample_dir, f'{sample_id}_graph.gt'))
        corrected_graph = correct_graph(sample_graph)
        if save:
            file_name = f'data_graph_corrected{annotation.find_name(region_list[0][0], key="id")}.gt'
            corrected_graph.save(os.path.join(sample_dir, file_name))


if __name__ == "__main__":
    test()
