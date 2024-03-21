import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import igraph
import numpy as np

from ClearMap.Alignment import Annotation as annotation, Annotation as ano

"""These are the values of the "nkind" property of the edges in the graph
these are used to have an integer value for the edges that can be used in the linear system"""
n_kinds = {
    'arteries': 2,
    'veins': 3,
    'capillaries': 4
}


MAX_PROCS = multiprocessing.cpu_count() - 1


# ############################## Brain graph generic ########################################


def extract_annotated_region(graph, region):  # TODO: move to annotation ?
    id_, level = region
    print(annotation.annotation.ids_to_names(id_))

    label = graph.vertex_annotation()
    label_leveled = annotation.convert_label(label, key='id', value='id', level=level)
    vertex_filter = label_leveled == id_

    region_graph = graph.sub_graph(vertex_filter=vertex_filter)
    return region_graph, vertex_filter


def get_sub_graph(graph, region_list):
    label = graph.vertex_annotation()
    vertex_filter = np.isin(label, region_list)
    sub_graph = graph.sub_graph(vertex_filter=vertex_filter)
    return sub_graph


def edge_to_vertex_property(graph, edge_property, dtype=None):
    """
    Converts graph_property from edge to vertex property

    .. warning:: This function only works for binary properties

    Parameters
    ----------
    graph : Graph
        The graph to convert the property from
    edge_property : str or np.array
        The name of the edge property to convert or the edge property itself

    Returns
    -------
    The vertex property (as a numpy array)
    """
    edge_property_data = edge_property if not isinstance(edge_property, str) else graph.edge_property(edge_property)
    edge_connectivity = graph.edge_connectivity()
    vertex_property_data = np.zeros(graph.n_vertices)
    for i in range(edge_connectivity.shape[1]):
        vertex_property_data[edge_connectivity[edge_property_data == 1, i]] = 1
    if dtype is not None:
        vertex_property_data = vertex_property_data.astype(dtype)
    return vertex_property_data


def vertex_to_edge_property(graph, graph_property):
    """
    Converts graph_property from vertex to edge property

    .. warning:: This function only works for binary properties

    Parameters
    ----------
    graph : Graph
        The graph to convert the property from
    graph_property : str or np.array
        The name of the vertex property to convert or the vertex property itself

    Returns
    -------
    The edge property (as a numpy array)
    """
    vertex_prop = graph.vertex_property(graph_property) if isinstance(graph_property, str) else graph_property
    connectivity = graph.edge_connectivity()
    edge_prop = np.logical_and(vertex_prop[connectivity[:, 0]], vertex_prop[connectivity[:, 1]])
    return edge_prop


def vertex_filter_to_edge_filter(graph, vertex_filter, operator=np.logical_and):
    """
    Converts a vertex filter to an edge filter

    .. warning::
      The operator is essential. Should both vertices follow the filter or either one.

    Parameters
    ----------
    graph : Graph
        The graph to convert the filter for
    vertex_filter : np.array
        The vertex filter to convert

    Returns
    -------
    The edge filter (as a numpy array)
    """
    connectivity = graph.edge_connectivity()
    start_vertex_follows_filter = vertex_filter[connectivity[:, 0]]
    end_vertex_follows_filter = vertex_filter[connectivity[:, 1]]
    edge_filter = operator(start_vertex_follows_filter, end_vertex_follows_filter)
    return edge_filter


def remove_surface(graph, depth):
    """
    Removes the vessels that are within a certain distance from the surface of the brain
    The filter is done on the 'distance_to_surface' property of the vertices. Hence, the
    depth should be given in the same unit as the property.

    Parameters
    ----------
    graph : Graph
        The graph to filter
    depth : float
        The depth from the surface to remove

    Returns
    -------

    """
    # distance_from_surface = from_v_prop2_eprop(graph, graph.vertex_property('distance_to_surface'))
    # ef=distance_from_surface>width  # TODO: verify if filtering by vertex is OK or if we should filter by edge
    vf = graph.vertex_property('distance_to_surface') > depth
    return graph.sub_graph(vertex_filter=vf)


# PERFORMANCE: could be parallelized
def modularity_measure(graph, modules):  # @Sophie: fill docstring explaination
    """
    Measure of modularity of a subgraph

    Parameters
    ----------
    modules
    graph

    Returns
    -------

    """
    K = graph.n_edges  # @Sophie: maybe we want some explaination for the single letter vars. That could be a latex eq in the docstring
    # trash_clusters = np.unique(partition)[np.where(c<20)]
    Qs = []
    for module_id in np.unique(modules):  # Could call colors instead of modules ?
        cluster = graph.sub_graph(vertex_filter=(modules == module_id))
        ms = cluster.n_edges  # @ Sophie isn't that k of the cluster? (shouldn't this be ks ?)
        ks = np.sum(cluster.vertex_degrees())
        Qs.append((ms / K) - ((ks / (2 * K)) ** 2))
    return sum(Qs)


# ############################## Vasculature Graph generic ########################################


# FIXME: check that all uses should use the same implementation
def set_artery_vein_if_missing(graph, artery_min_radius=4, vein_min_radius=8):  # FIXME: magic number
    """
    Sets the vertex properties for artery and vein if they are missing.
    """
    try:
        edge_to_vertex_property(graph, 'artery')
        edge_to_vertex_property(graph, 'vein')
    except KeyError:  # May be other exception kind, check
        print('No artery vertex properties')
        artery = graph.vertex_radii() >= artery_min_radius
        vein = graph.vertex_radii() >= vein_min_radius
        graph.add_vertex_property('artery', artery)
        graph.add_vertex_property('vein', vein)
        graph.add_edge_property('artery', vertex_to_edge_property(graph, artery))
        graph.add_edge_property('vein', vertex_to_edge_property(graph, vein))


def combine_arteries_and_veins(graph, artery=None, vein=None, mode='arteryvein', min_radius=None):
    """
    Combine the arteries and veins in the graph into a single property

    Parameters
    ----------
    graph : Graph
        The graph to combine the arteries and veins of
    artery : np.array or None
        The artery property of the graph. If None, will be extracted from the graph
    vein : np.array or None
        The vein property of the graph. If None, will be extracted from the graph
    mode : str
        The mode to use for combining the arteries and veins. Can be one of:
            - 'arteryvein': simply combine (OR) the artery and vein properties
            - 'bigvessels': combine the artery and vein properties and add the vessels with a radius above min_radius
    min_radius : float or None
        The minimum radius to use for the 'bigvessels' mode. This is required if mode is 'bigvessels'.

    Returns
    -------
    The combined artery_vein property (as a numpy array)
    """
    if artery is None:
        artery = np.asarray(graph.vertex_property('artery'))
    if vein is None:
        vein = np.asarray(graph.vertex_property('vein'))
    combined = np.logical_or(artery, vein)
    if mode == 'bigvessels':
        big_vessel_filter = edge_to_vertex_property(graph, graph.edge_property('radii') > min_radius)
        # FIXME: could we just do graph.vertex_property('radii') > radius ?
        combined = np.logical_or(combined, big_vessel_filter)
    elif mode != 'arteryvein':
        raise ValueError(f'Unknown mode {mode}, expected one of ("arteryvein", "bigvessels")')
    return combined


def parallel_get_vessels_lengths(graph, edge_coordinates_name='coordinates', clip_below_unity=True,
                                 min_chunk_size=5, n_processes=1):
    """
    Computes the length of each vessel in the graph in parallel.
    Using the geometry, we extract the coordinates of each vessel and compute the length from that.
    This is parallelized because pure numpy is impossible because the vessels have different lengths.

    Parameters
    ----------
    graph : Graph
        The graph to compute the lengths of
    edge_coordinates_name : str
        The name of the property containing the coordinates of the vessels.
        Typically, 'coordinates' or 'coordinates_atlas'
    clip_below_unity : bool
        If True, clips the length below 1 to 1
    min_chunk_size : int
        The minimum chunk size to use for parallel processing
    n_processes : int
        The number of processes to use for the computation

    Returns
    -------
    An array of the lengths of the vessels in the graph
    """
    n_processes = min(n_processes, MAX_PROCS)
    vessels_coordinates = graph.edge_geometry(edge_coordinates_name)
    if n_processes > 1:
        try:  # try parallel processing
            chunk_size = max(min_chunk_size, len(vessels_coordinates) // (2 * n_processes))
            with ProcessPoolExecutor(n_processes) as executor:
                results = executor.map(get_vessel_length, vessels_coordinates, chunksize=chunk_size)
            vessels_lengths = np.array(list(results))
        except BrokenProcessPool:  # Go single threaded
            vessels_lengths = np.array([get_vessel_length(vessel_coords) for vessel_coords in vessels_coordinates])
    else:
        vessels_lengths = np.array([get_vessel_length(vessel_coords) for vessel_coords in vessels_coordinates])
    if clip_below_unity:
        vessels_lengths = np.clip(vessels_lengths, 1, None)
    return vessels_lengths


def get_vessel_length(coordinates):
    """
    Get the length of a single vessel from its coordinates

    Parameters
    ----------
    coordinates : ndarray
        The coordinates of the (single) vessel in the format (n_points, 3)

    Returns
    -------

    """
    diff = np.diff(coordinates, axis=0)
    length = np.sum(np.linalg.norm(diff, axis=1))
    return length


def filter_graph_degrees(graph, min_degree=2, max_degree=4):
    """
    Filter the graph to keep only the edges with a degree between min_degree and max_degree
    (boundaries are included)

    Parameters
    ----------
    graph
    min_degree : int
        The minimum degree to keep
    max_degree : int
        The maximum degree to keep

    Returns
    -------
    The filtered graph
    """
    degrees = graph.vertex_degrees()
    vf = np.logical_and(degrees >= min_degree, degrees <= max_degree)
    graph = graph.sub_graph(vertex_filter=vf)
    return graph


def graph_gt_to_igraph(src_graph):
    """
    Convert graph from GraphGt to igraph format

    .. warning:: This only transfers the properties 'radius', 'diameter' and 'nkind'. nkind is the integer value
        representing the type of vessel (see the n_kinds dictionary for the possible values)

    Parameters
    ----------
    src_graph : GraphGt
        The graph to convert (in GraphGt format)

    Returns
    -------
    The converted graph (in igraph format)
    """
    output_graph = igraph.Graph()
    output_graph.add_vertices(src_graph.n_vertices)
    output_graph.add_edges(src_graph.edge_connectivity())

    output_graph.es["nkind"] = get_edges_n_kinds(src_graph)
    output_graph.vs["nkind"] = get_vertices_n_kinds(src_graph)

    radii = src_graph.edge_property('radii')
    output_graph.es["radius"] = radii
    output_graph.es["diameter"] = 2 * radii
    return output_graph


def get_edges_n_kinds(graph):
    """
    Get the n_kind property (i.e. the integer value representing the type of vessel)
    for each edge in the graph

    Parameters
    ----------
    graph : Graph
        The graph to get the n_kind property of

    Returns
    -------

    n_kind : np.array
        The n_kind property for each edge in the graph
    """
    n_kind = np.full(graph.n_edges, n_kinds['capillaries'], dtype=int)
    arteries = graph.edge_property('artery')
    veins = graph.edge_property('vein')
    n_kind[arteries] = n_kinds['arteries']
    n_kind[veins] = n_kinds['veins']
    return n_kind


def get_vertices_n_kinds(graph):
    """
    Get the n_kind property (i.e. the integer value representing the type of vessel)
    for each vertex in the graph

    Parameters
    ----------
    graph : Graph
        The graph to get the n_kind property of

    Returns
    -------

    n_kind : np.array
        The n_kind property for each vertex in the graph
    """
    n_kind = np.full(graph.n_vertices, n_kinds['capillaries'], dtype=int)
    # REFACTOR: see if can replace the following with np.where
    #   Also, check if this is a binary property (in which case, we could use a boolean array)
    arteries = edge_to_vertex_property(graph, 'artery', int).nonzero()[0]
    veins = edge_to_vertex_property(graph, 'vein', int).nonzero()[0]
    n_kind[arteries] = n_kinds['arteries']
    n_kind[veins] = n_kinds['veins']
    return n_kind


def get_vertex_coordinates(graph, coordinates_name):
    """
    Get the coordinates from the graph in any space.
    The coordinates can be in any space, as long as it is a vertex property of the graph.
    Examples of coordinates spaces are 'coordinates', 'coordinates_atlas', 'coordinates_scaled', 'coordinates_MRI'
    Parameters
    ----------
    graph : GraphGt.Graph
        The graph to get the coordinates from
    coordinates_name : str
        The name of the coordinates space to use. Typically, 'coordinates' or 'coordinates_atlas'

    Returns
    -------

    """
    if coordinates_name == 'coordinates':
        coordinates = graph.vertex_coordinates()  # REFACTOR: check if we could use the vertex property instead to generalize
    else:
        coordinates = graph.vertex_property(coordinates_name)
    return coordinates
