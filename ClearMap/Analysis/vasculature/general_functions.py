import os

from pathlib import Path

from tqdm import tqdm

from ClearMap.Analysis.vasculature.geometry_utils import cartesian_to_polar, angle_between_vectors, compute_grid, \
    interpolate_vectors
from ClearMap.Analysis.vasculature.vasc_graph_utils import set_artery_vein_if_missing, combine_arteries_and_veins, \
    vertex_to_edge_property, edge_to_vertex_property, filter_graph_degrees, \
    parallel_get_vessels_lengths, graph_gt_to_igraph, n_kinds, get_vertex_coordinates, vertex_filter_to_edge_filter

CORRECTED_GRAPH_BASE_NAME = 'data_graph_correcteduniverse'

try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle

import numpy as np

import igraph

import ClearMap.IO.IO as clearmap_io
import ClearMap.Alignment.Annotation as annotation
import ClearMap.Analysis.Graphs.GraphGt as graph_gt

from ClearMap.Analysis.vasculature.flow.linear_system import LinearSystem


# TODO:
#  - Change all orders to ids
#  - Check that group_name is forced to controls whenever appropriate

# FIXME: check cases where group should be hard coded to controls (i.e. when it is the reference group)


N_PROCESSES = 15

VESSEL_PERMEABILITY_CONSTANT = 7.6  # TODO: check value
DEFAULT_UNITS = {'length': 'um', 'mass': 'ug', 'time': 'ms'}


def load_graph(work_dir, group_name):
    """
    Loads the graph from the given work directory and control.
    Tries to load the default corrected graph, if it does not exist, tries to load the default graph.
    """
    graph_dir = os.path.join(work_dir, group_name)
    try:
        graph = graph_gt.load(os.path.join(graph_dir, f'{CORRECTED_GRAPH_BASE_NAME}.gt'))
    except FileNotFoundError:
        graph = graph_gt.load(os.path.join(graph_dir, f'{group_name}_graph.gt'))
    return graph


# FIXME: structures by name (here, cerebellum and hippocampus)
# FIXME: set to ID but won't work because algo seems designed for order
def structure_specific_radius_filter(graph, artery_vein, label=None, specific_structures=((1006, 3), (463, 6))):
    """
    This function reworks the graph so that the radius criterion is applied differently
    depending on the structure (here, cerebellum and hippocampus)

    Parameters
    ----------
    graph
    artery_vein
    label
    specific_structures

    Returns
    -------

    """
    if label is None:
        label = graph.vertex_annotation()
    radii = graph.vertex_property('radii')
    try:
        arts = []  # @Sophie: what is arts ? arteries ? more explicit name
        for id_, level in specific_structures:
            label_leveled = annotation.convert_label(label, key='id', value='id', level=level)
            structure = label_leveled == id_
            arts.append(np.logical_and(radii <= 6, np.logical_and(structure, artery_vein)))  # FIXME: magic numbers
            artery_vein[structure] = 0

        radii = graph.vertex_property('radii')
        pb_art = arts[0]
        for art in arts[1:]:
            pb_art = np.logical_or(pb_art, art)
        artery_vein = np.logical_or(pb_art, np.logical_and(artery_vein, radii >= 3))  # FIXME: magic numbers (6 in streamlines_flow.py)
    except KeyError:
        print(f'Could not find region, check json file')
    return artery_vein


# WARNING: if clean_graph is True, This will use a structure specific radius filter
def get_artery_vein_edge_coords_and_vectors(graph, mode, artery, vein, min_radius,
                                            label=None, clean_graph=True,
                                            distance_to_surface=None, min_dist_to_surface=2, invert=False,
                                            coordinates_name='coordinates', orientation_criterion='distance',
                                            return_filter=False):
    artery_vein = combine_arteries_and_veins(graph, artery, vein, mode, min_radius)

    if clean_graph:
        artery_vein = structure_specific_radius_filter(graph, artery_vein, label=label)

    if distance_to_surface is not None:
        artery_vein = np.asarray(np.logical_and(artery_vein, distance_to_surface >= min_dist_to_surface))

    vertex_filter = artery_vein
    if invert:  # i.e. capillaries
        vertex_filter = np.logical_not(vertex_filter)
    sub_graph = graph.sub_graph(vertex_filter=vertex_filter)  # TODO: see why edge_connectivity is different for view (this would speedup otherwise)
    out = graph_to_edge_coords_and_vectors(sub_graph, coordinates_name=coordinates_name,
                                               orientation_criterion=orientation_criterion)
    if return_filter:
        out = list(out) + [vertex_filter_to_edge_filter(graph, vertex_filter)]
    return out


def compute_flow_f_schmid(work_dir, graph, cont, graph_name='correcteduniverse', dest_base_name='same'):
    """
    Compute the flow and velocity of the graph using the method published in
    Schmid F. et al. https://doi.org/10.1371/journal.pcbi.1005392

    First, make an igraph version of the graph with the necassary properties
    Then add:
    - length (computed from the geometry of the edges)
    - boundary_cap (True if edge is a boundary capillary)
    - nkind (node kind, i.e. integer representing the type of node from
    (0, 1, 2, 3) for (capillaries, arteries, veins, universe)
    - pBC (pressure boundary conditions)

    Parameters
    ----------
    work_dir : str
        The path to the work directory
    graph: GraphGt.Graph
        The graph to compute the flow and velocity from
    cont : str
        The name of the control group
    graph_name : str
        The name of the graph file
    dest_base_name : str
        The base name of the destination file. If 'same', the name will be the same as the graph name

    Returns
    -------
        flow, velocity, pressure : ndarray, ndarray, ndarray
    """
    tmp_graph = graph_gt_to_igraph(graph)
    vessels_lengths = parallel_get_vessels_lengths(graph, n_processes=N_PROCESSES)
    tmp_graph.es["length"] = vessels_lengths

    graph_ml_path = os.path.join(work_dir, cont, f'{CORRECTED_GRAPH_BASE_NAME}.GraphML')  # REFACTORING: could use path from graph ?
    tmp_graph.write_graphml(graph_ml_path)
    tmp_graph = igraph.read(graph_ml_path)  # FIXME: is the read necessary
    tmp_graph['defaultUnits'] = DEFAULT_UNITS

    # Change capillary deadends adjacent to A or V to (n_kinds['arteries'], n_kinds['veins'])
    tmp_graph.es['boundary_cap'] = [0] * tmp_graph.ecount()
    for v in tmp_graph.vs(_degree_eq=1, nkind_eq=n_kinds['capillaries']):
        tmp_graph.es[tmp_graph.incident(v)[0]]['boundary_cap'] = 1  # 0 by default, 1 for (degree1 and capillary)

    # For boundary capillaries with diameter > 10, if vertices are connected to A or V, change to A or V
    for e in tmp_graph.es(boundary_cap_eq=1, diameter_gt=10):  # FIXME: magic number
        for n_kind in (n_kinds['arteries'], n_kinds['veins']):  # @ Sophie: is it correct to be sequential ?
            if n_kind in tmp_graph.vs[e.tuple]['nkind']:
                e['nkind'] = n_kind
                for v in e.tuple:
                    tmp_graph.vs[v]['nkind'] = n_kind

    # pressure fit. pBC = pressure boundary conditions
    p = np.array([2.59476905e-05, -8.91218311e-03, 1.06888397e+00, 2.04594862e+01])  # FIXME: extract and explain where this comes from
    tmp_graph.vs['pBC'] = [None] * tmp_graph.vcount()
    # for all degree1 vertices, if veins, pBC = 10, else pBC = np.polyval(p, diameter)
    tmp_graph.vs(_degree_eq=1, nkind_eq=n_kinds['veins'])['pBC'] = (
            [10] * len(tmp_graph.vs(_degree_eq=1, nkind_eq=n_kinds['veins'])))
    for v in tmp_graph.vs(_degree_eq=1, nkind_ne=n_kinds['veins']):
        diameter = tmp_graph.es[tmp_graph.incident(v)[0]]['diameter']
        v['pBC'] = np.polyval(p, diameter)

    # p_a, _ = fit_for_pressureBC()  # TODO: check if could remove
    # assign_pressureBC(tmp_graph, p_a)

    linear_system = LinearSystem(tmp_graph)
    dest_base_name = graph_name if dest_base_name == 'same' else cont  # REFACTOR: simplify logic
    out_path = os.path.join(work_dir, cont, f'sampledict{dest_base_name}.pkl')  # TODO: see if could extract path and share with load_sample_dict
    sample_dict = linear_system.solve('iterative2', dest_file_path=out_path)

    return [np.asarray(sample_dict[k]) for k in ['flow', 'v', 'pressure']]


def compute_blood_flow(graph, work_dir, sample_name):
    """
    Return the graph with the flow, velocity and pressure properties added
    If this has not been computed before, it will be computed and saved to
    the sample dict

    Parameters
    ----------
    graph : GraphGt.Graph
        The graph to compute the flow, velocity and pressure from
    work_dir : str
        The path to the work directory
    sample_name : str
        The name of the sample (used in the file name)

    Returns
    -------
    graph : GraphGt.Graph
        The graph with the flow, velocity and pressure properties added
    """

    flow, velocity, pressure = compute_flow_f_schmid(work_dir, graph, sample_name)
    graph.add_edge_property('flow', flow)
    graph.add_edge_property('veloc', velocity)
    graph.add_vertex_property('pressure', pressure)
    return graph


def get_nb_radial_vessels(edge_color, radiality_threshold=0.7):  # warning: unused
    """
    Get the number of radial vessels from the edge color

    Parameters
    ----------
    edge_color : ndarray
    radiality_threshold : float
        Threshold to filter out vessels that are not sufficiently radial to the plane.
        The default value is 0.7, which means that the vessel must be at least 70% radial to the plane to be
        counted as radial. This is a good empirical value, but it may need to be adjusted for different
        applications.

    Returns
    -------
    nb_radial : int
        The number of radial vessels
    """
    radial = edge_color[:, 2] / edge_color.sum(axis=1)  # TODO: check that edge_color.shape = (n, 3) and document
    return np.sum(radial > radiality_threshold)


def get_nb_parallel_vessels(edge_color, verbose=True, parallel_threshold=0.7):  # warning: unused
    """
    Get the number of parallel vessels from the edge color

    Parameters
    ----------
    edge_color : ndarray
    parallel_threshold : float
        Threshold to filter out vessels that are not sufficiently parallel to the plane.
        The default value is 0.7, which means that the vessel must be at least 70% parallel to the plane to be
        counted as parallel. This is a good empirical value, but it may need to be adjusted for different
        applications.
    verbose : bool
        Whether to print the shape of the array

    Returns
    -------
    nb_parallel : int
        The number of parallel vessels
    """
    planar = (edge_color[:, 0] + edge_color[:, 1]) / edge_color.sum(
        axis=1)  # TODO: check that edge_color.shape = (n, 3) and document
    if verbose:
        print(planar.shape)
    return np.sum(planar > parallel_threshold)


def graph_to_edge_coords_and_vectors(graph, coordinates_name='coordinates', orientation_criterion='distance'):
    """
    Get the coordinates and vectors of the edges of the graph.
    The edges can be oriented based on the criterion so that the vectors will follow this orientation.

    Parameters
    ----------
    graph : GraphGt.Graph
        The graph to get the edge coordinates and vectors from
    coordinates_name : str
        The coordinates space to use. Typically, 'coordinates' or 'coordinates_atlas'
        IT should be an existing vertex property of the graph
        (that represents the coordinates of the vertices in some space)
    orientation_criterion : str
        The criterion to use to orient the edges. Can be either 'pressure', 'distance_to_surface' or '' (no sorting)

    Returns
    -------

    """
    vertex_coordinates = get_vertex_coordinates(graph, coordinates_name)
    connectivity = sort_connectivity_based_on_vertex_property(graph, orientation_criterion)
    # Compute the edge coordinates as the mean of the coordinates of the vertices of the edge
    edge_coordinates = np.round((vertex_coordinates[connectivity[:, 0]] +
                                 vertex_coordinates[connectivity[:, 1]])
                                / 2)
    edge_vectors = coordinates_to_edge_vectors(vertex_coordinates, connectivity, normalized=False)
    return edge_coordinates, edge_vectors


def coordinates_to_edge_vectors(vertex_coordinates, connectivity, normalized=False):
    """
    Get the edge vectors from the coordinates and connectivity.
    The vectors are computed as the difference between the coordinates of the vertices of the edge.


    .. warning::
        If normalized is True, the vectors will be normalized to unit length.
        If a normalized vector is 0, it will be replaced by a vector of 0s.


    Parameters
    ----------
    vertex_coordinates : ndarray
        The coordinates of the vertices
    connectivity : ndarray
        The connectivity of the graph.
    normalized : bool
        Whether to normalize the vectors to unit length or not

    Returns
    -------
    edge_vectors : ndarray
        The vectors of the edges
    """
    edges_coordinates = vertex_coordinates[connectivity]
    edge_vectors = edges_coordinates[:, 1, :] - edges_coordinates[:, 0, :]
    if normalized:
        norms = np.linalg.norm(edge_vectors, axis=1)[:, None]
        edge_vectors = np.divide(edge_vectors, norms, out=np.zeros_like(edge_vectors), where=norms != 0)
    return edge_vectors


def get_edge_vectors(graph, normed=False, orientation_criterion='distance',
                     coordinates_name='coordinates'):
    """
    Get the edge vectors and sorted connectivity from the graph.
    The edges can be oriented based on the criterion so that the vectors will follow this orientation.
    If no criterion is given (''), the edges will not be oriented.

    Parameters
    ----------
    graph : GraphGt.Graph
        The graph to get the edge vectors from
    normed : bool
        Whether to normalize the vectors to unit length or not
    orientation_criterion : str
        The criterion to use to orient the edges.
        Can be either 'pressure', 'distance_to_surface' or '' (no sorting)
    coordinates_name : str
        The coordinates space to use. Typically, 'coordinates', 'coordinates_atlas',
        'coordinates_scaled' or 'coordinates_MRI'

    Returns
    -------

    """
    connectivity = sort_connectivity_based_on_vertex_property(graph, orientation_criterion)

    vertex_coordinates = get_vertex_coordinates(graph, coordinates_name)
    edge_vectors = coordinates_to_edge_vectors(vertex_coordinates, connectivity, normed)
    return edge_vectors, connectivity


# FIXME: what are x, y, center and where do they come from ?
def get_edge_vectors_spherical(graph, x, y, center, orientation_criterion='distance', normed=False):
    connectivity = sort_connectivity_based_on_vertex_property(graph, orientation_criterion, reversed=True)

    spherical_coord = np.array(cartesian_to_polar(x - center[0], y - center[1])).T
    edge_vectors = coordinates_to_edge_vectors(spherical_coord, connectivity, normed)
    return edge_vectors, connectivity


# FIXME: set criterion defaults but check usage first
def sort_connectivity_based_on_vertex_property(graph, orientation_criterion, reversed=False):
    """
    For each edge, sort the vertices based on the criterion.
    i.e. get the edges direction based on the criterion,
    which can be either 'pressure', 'distance_to_surface' or '' (no sorting)

    Parameters
    ----------
    graph
    orientation_criterion : str
        The criterion to use to sort the edges. Can be either 'pressure', 'distance_to_surface' or '' (no sorting)

    Returns
    -------

    """
    if orientation_criterion.startswith('distance'):
        orientation_criterion = 'distance_to_surface'

    if orientation_criterion == '':
        vertex_property = None
    else:
        try:
            vertex_property = graph.vertex_property(orientation_criterion)
        except KeyError as err:
            print(f'Unknown criteria {orientation_criterion}, typical values are ("pressure", "distance")')
            raise err
    connectivity = graph.edge_connectivity()
    if vertex_property is not None and vertex_property.shape[0] != 0:
        sorted_indices = np.argsort(vertex_property[connectivity], axis=1)
        if reversed:
            sorted_indices = sorted_indices[:, ::-1]
        # Apply the sorted indices to the connectivity
        oriented_connectivity = connectivity[
            np.arange(connectivity.shape[0])[:, None], sorted_indices]  # TODO: TEST: compare to above
        connectivity = oriented_connectivity
    return connectivity


def get_streamline_agv_path(work_dir, group_name, sample_name, mode):
    return os.path.join(work_dir, f'streamline_AGV_{group_name}_{sample_name}_{mode}.npy')


def get_streamline_aec_path(work_dir, group_name, sample_name, mode):
    return os.path.join(work_dir, f'streamline_AEC_{group_name}_{sample_name}_{mode}.npy')


def get_reference_streamlines(work_dir, reference_samples, mode):
    AEC = []
    AGV = []
    for sample in reference_samples:
        streamline_aec_path = get_streamline_aec_path(work_dir, 'controls', sample, mode)
        streamline_agv_path = get_streamline_agv_path(work_dir, 'controls', sample, mode)
        if not os.path.exists(streamline_aec_path) or not os.path.exists(streamline_agv_path):
            take_avg_streamlines(work_dir, sample, mode=mode)
        AEC.append(np.load(streamline_aec_path, allow_pickle=True))
        AGV.append(np.load(streamline_agv_path, allow_pickle=True))
    return AEC, AGV



def take_avg_streamlines(work_dir, sample, mode='bigvessels', group_name='controls', min_radius=4,
                         coordinates_name='coordinates'):

    graph = load_graph(work_dir, sample)  # TODO: check if sample or group_name
    graph = filter_graph_degrees(graph)

    set_artery_vein_if_missing(graph)

    artery = edge_to_vertex_property(graph, 'artery')
    vein = edge_to_vertex_property(graph, 'vein')
    d2s = graph.vertex_property('distance_to_surface')
    art_edge_coordinates, art_graph_vector = (
        get_artery_vein_edge_coords_and_vectors(graph, mode, artery, vein, min_radius=min_radius,
                                                distance_to_surface=d2s, coordinates_name=coordinates_name))

    np.save(get_streamline_aec_path(work_dir, group_name, sample, mode), art_edge_coordinates)
    np.save(get_streamline_agv_path(work_dir, group_name, sample, mode), art_graph_vector)


def avg_streamlines_grid(work_dir, reference_samples, mode='bigvessels', group_name='controls'):  # FIXME: unused
    AEC, AGV = get_reference_streamlines(work_dir, reference_samples, mode)

    all_samples_flow_vectors = []
    for arteries_edge_coordinates, arteries_graph_vector in zip(AEC, AGV):

        grid = compute_grid(arteries_edge_coordinates)

        flow_vectors = interpolate_vectors(arteries_graph_vector, arteries_edge_coordinates, grid)
        all_samples_flow_vectors.append(flow_vectors)

    flow_vectors_avgs = np.nanmean(np.array(all_samples_flow_vectors), axis=0)
    dest_path = Path(work_dir) / f'streamline_grid_avg_{group_name}{mode}.npy'
    np.save(dest_path, flow_vectors_avgs)
    clearmap_io.write(str(dest_path.with_suffix('.tif')), flow_vectors_avgs)


# ############################################################################################################

def get_orientation_from_normal_to_surface_local(graph, region_ids, normal_vector_method='mean',
                                                 coordinates_name='coordinates'):
    """
    Orientation in the cortex relative to the surface normal. (i.e. we fit a plane to the top vertices
    and the bottom vertices and take the normal to this plane as the top-to-bottom direction). Alternatively,
    we can use the average method which takes the mean of the top and bottom vertices.

    Parameters
    ----------
    graph : GraphGt.Graph
        The graph to get the orientation from
    region_ids : list
        The list of the ids of the regions to consider.
    normal_vector_method : str
        Determines the way to compute the normal vector to the plane best fitting the vertices.
        Can be either 'svd' or 'mean'.
        If set to svd, we use Singular Value Decomposition (SVD) to find the normal vector to the plane best fitting
        the top vertices.
        If set to mea, we take the average of the top and bottom vertices.
    coordinates_name : str
        The coordinates space to use. Typically, 'coordinates', 'coordinates_atlas',
         'coordinates_scaled' or 'coordinates_MRI'

    Returns
    -------
        radial_orientations, planar_orientations, reference_norms, edge_lengths
        (all ndarrays)
    """
    radial_orientations = np.zeros(graph.n_edges)
    planar_orientations = np.zeros(graph.n_edges)
    label = graph.vertex_annotation()
    for id_ in region_ids:  # WARNING: could use second element of tuple which is the level
        vertex_filter = label == id_
        sub_graph = graph.sub_graph(vertex_filter=vertex_filter)  # PERFORMANCE: check if we could use view instead of sub_graph
        r, p = get_orientation_from_normal_to_surface_global(sub_graph, sub_graph,  # FIXME: check if sub_graph is correct
                                                             normal_vector_method=normal_vector_method,
                                                             coordinates_name=coordinates_name)
        edge_filter = vertex_to_edge_property(graph, vertex_filter)
        radial_orientations[edge_filter] = r
        planar_orientations[edge_filter] = p
    return abs(radial_orientations), abs(planar_orientations)


def get_orientation_from_normal_to_surface_global(graph, ref_graph, normal_vector_method='mean',
                                                  coordinates_name='coordinates'):
    """
    Get the orientation of the graph relative to the surface normal.

    Parameters
    ----------
    graph : GraphGt.Graph
        The graph to get the orientation from
    ref_graph
    normal_vector_method : str
        The method to use to compute the normal vector to the plane best fitting the vertices.
        Can be either 'svd' or 'mean'.
    coordinates_name : str
        The coordinates space to use. Typically, 'coordinates', 'coordinates_atlas',
            'coordinates_scaled' or 'coordinates_MRI'

    Returns
    -------

    """
    vertex_coordinates = get_vertex_coordinates(graph, coordinates_name=coordinates_name)
    normed_edge_vect = coordinates_to_edge_vectors(vertex_coordinates, graph.edge_connectivity(), normalized=True)
    top_to_bottom_dist = get_top_to_bottom_dist(ref_graph, normal_vector_method=normal_vector_method)
    radial_orientations = np.dot(normed_edge_vect, top_to_bottom_dist)
    planar_orientations = np.sqrt(1 - radial_orientations ** 2)
    # edge_vect = coordinates_to_edge_vectors(vertex_coordinates, connectivity, normalized=False)
    # reference_norms = np.linalg.norm(edge_vect[edge_vect.shape[0] - 1])
    return abs(radial_orientations), abs(planar_orientations)


def get_top_to_bottom_dist(graph, normal_vector_method='svd', coordinates_name='coordinates'):
    """
    Get the top to bottom distance of the graph.
    This can be computed using the minimum distance method or by taking the mean of the top and bottom vertices.

    Parameters
    ----------
    graph : GraphGt.Graph
        The graph to get the top to bottom distance from
    normal_vector_method : str
        The method to use to compute the top to bottom distance.
        Can be either 'svd' or 'mean'.
        If set to 'svd', we identify the vertices in the graph that are closer to the surface than a certain threshold
        (min_dist, which is the minimum distance to the surface plus 1.5).
        We then calculate the centroid of these vertices and uses Singular Value Decomposition (SVD)
        to find the normal vector to the plane best fitting these vertices.
        This normal vector is considered as the top-to-bottom direction.

        If set to 'mean', the function computes the top-to-bottom distance by taking the mean of the coordinates
        of the top and bottom vertices in the graph.
        The top vertices are those with a distance to the surface less than or equal to the minimum distance plus 1,
        and the bottom vertices are those with a distance to the surface greater than or equal
        to the maximum distance minus 1.
        The top-to-bottom direction is then the normalized vector from the mean of the top vertices
        to the mean of the bottom vertices.
    coordinates_name : str
        The coordinates space to use. Typically, 'coordinates', 'coordinates_atlas',
        'coordinates_scaled' or 'coordinates_MRI'

    Returns
    -------

    """
    dist = graph.vertex_property('distance_to_surface')
    coordinates = get_vertex_coordinates(graph, coordinates_name)
    if normal_vector_method == 'svd':
        margin = 1.5
        top_vertices = dist < np.min(dist) + margin
        coords = coordinates[top_vertices]

        centroid = coords.mean(axis=0)
        # run SVD
        _, _, vh = np.linalg.svd(coords - centroid)
        # unitary normal vector
        top_to_bottom_dist = vh[2, :]  # FIXME: why 2 ?
    elif normal_vector_method == 'mean':
        margin = 1  # FIXME: different margin
        top_vertices = dist <= np.min(dist) + margin
        bottom_vertices = dist >= np.max(dist) - margin

        top_to_bottom_dist = (np.mean(coordinates[bottom_vertices], axis=0) -
                              np.mean(coordinates[top_vertices], axis=0))
        top_to_bottom_dist /= np.linalg.norm(top_to_bottom_dist)  # preprocessing.normalize(top_to_bottom_dist, norm='l2')
    else:
        raise ValueError(f'Unknown method {normal_vector_method}, expected "svd" or "mean"')
    return top_to_bottom_dist.T


# FIXME: extract part that modifies the graph to add artery and vein
def generalized_radial_planar_orientation(graph, work_dir, min_radius, reference_samples, corrected=True,
                                          distance_to_surface=True,
                                          mode='arteryvein', average=False, dvlpmt=True, min_dist_to_surface=7,
                                          coordinates_name='coordinates', orientation_criterion='distance'):
    """
    Use Flow interpolation to compute the orientation of the vessels in the graph.
    This is based on the local orientation of the arteries

    Parameters
    ----------
    graph
    work_dir
    min_radius
    reference_samples
    corrected
    distance_to_surface
    mode
    average
    dvlpmt
    min_dist_to_surface
    coordinates_name
    orientation_criterion

    Returns
    -------

    """
    artery = edge_to_vertex_property(graph, 'artery')
    vein = edge_to_vertex_property(graph, 'vein')
    distance_to_surface = graph.vertex_property('distance_to_surface') if distance_to_surface else None

    if average:
        reference_arteries_edge_coords, reference_arteries_edge_vectors = get_reference_streamlines(work_dir, reference_samples, mode)  # FIXME: this could be computed in parent function and passed as argument

        capillaries_edge_coordinates, capillaries_edge_vectors, filter = (
            get_artery_vein_edge_coords_and_vectors(graph, mode, artery, vein,
                                                    min_radius=min_radius,
                                                    distance_to_surface=distance_to_surface, min_dist_to_surface=min_dist_to_surface,
                                                    coordinates_name=coordinates_name, orientation_criterion=orientation_criterion,
                                                    invert=True, return_filter=True))
        tmp_angles = []  # The angles to all the reference samples
        # PERFORMANCE: should use multiple cores
        # For each reference sample, interpolate its arteries vector onto the edges of the tested graph and compute the angle
        for arteries_edge_coordinates, arteries_graph_vector in zip(reference_arteries_edge_coords, reference_arteries_edge_vectors):
            flow_vectors = interpolate_vectors(arteries_graph_vector, arteries_edge_coordinates, capillaries_edge_coordinates)
            tmp_angles.append(angle_between_vectors(flow_vectors, capillaries_edge_vectors))
        sub_angles = np.nanmedian(np.array(tmp_angles), axis=0)
        angles = np.full(graph.n_edges, np.nan)
        indices = np.where(filter)[0]
        angles[indices] = sub_angles  # Cast back to the original graph shape
    else:
        edge_coordinates, edge_vectors = graph_to_edge_coords_and_vectors(graph, coordinates_name=coordinates_name,  # FIXME: is this capillaries ? do we need to invert ?
                                                                          orientation_criterion=orientation_criterion)
        arteries_edge_coordinates, arteries_edge_vectors = (
            get_artery_vein_edge_coords_and_vectors(graph, mode, artery, vein,
                                                    min_radius=min_radius, clean_graph=(corrected and not dvlpmt),
                                                    distance_to_surface=distance_to_surface, min_dist_to_surface=min_dist_to_surface,
                                                    coordinates_name=coordinates_name, orientation_criterion=orientation_criterion))

        flow_vectors = interpolate_vectors(arteries_edge_vectors, arteries_edge_coordinates, edge_coordinates)
        angles = angle_between_vectors(flow_vectors, edge_vectors)

    return angles, graph  # FIXME: why return graph ? It shouldn't be modified
