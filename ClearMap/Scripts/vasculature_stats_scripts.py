import cProfile
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize
import seaborn as sns
from tqdm import tqdm

import graph_tool
import graph_tool.inference as gti

import ClearMap.Analysis.Graphs.GraphGt as ggt
import ClearMap.Alignment.Annotation as ano
from ClearMap.Analysis.vasculature.general_functions import generalized_radial_planar_orientation, \
    compute_blood_flow, get_orientation_from_normal_to_surface_local, VESSEL_PERMEABILITY_CONSTANT
from ClearMap.Analysis.vasculature.vasc_graph_utils import edge_to_vertex_property, \
    filter_graph_degrees, set_artery_vein_if_missing, get_sub_graph
import ClearMap.Settings as settings

layers = ['l1', 'l2/3', 'l4', 'l5', 'l6a', 'l6b']


def _compute_sbm(graph, planar_proportion, n_repeats=5):
    graph.add_edge_property('planar_proportion', planar_proportion)
    # Get edge connectivity and radii
    radii = graph.vertex_property('radii')
    connect_radii = radii[graph.edge_connectivity()]
    param = 1 / (1 + np.abs(1 - (connect_radii[:, 0] / connect_radii[:, 1])))  # Similarity of radii for each edge
    graph.add_edge_property('param', param.astype(float))
    graph.add_edge_property('invert_length', 1 / graph.edge_property('length'))

    g = graph.base
    seeds = np.random.randint(0, 2 ** 32 - 1, n_repeats)
    Qs = []
    Ns = []
    for seed in seeds:
        graph_tool.seed_rng(seed)
        np.random.seed(seed)

        state = gti.minimize_blockmodel_dl(g, state_args={
            'recs': [g.ep.param.a, g.ep.invert_length.a, g.ep.planar_proportion.a],
            'rec_types': ["real-exponential"] * 3}
                                           )
        modules = state.get_blocks().a
        # graph.add_vertex_property('blocks', modules)  # we are overwriting the same property
        try:
            # Q = modularity_measure(graph, modules)
            Q = graph_tool.inference.modularity(g, state.get_blocks())  # TODO: check if we need .a
        except:   # FIXME: too broad
            Q = 0
        Qs.append(Q)
        n_modules = np.unique(modules).shape[0]
        Ns.append(n_modules)
    return np.mean(Ns), np.mean(Qs)  # TODO: check if mean or median is more appropriate


def _compute_arteries_branch_points(graph):  # FIXME: this does not do what it says
    artery = edge_to_vertex_property(graph, 'artery')
    vertex_filter = np.logical_and(artery, graph.vertex_property('artery_binary') > 0)
    art_g = graph.sub_graph(vertex_filter=vertex_filter)
    dist_art = art_g.vertex_property('distance_to_surface')
    return dist_art


def compute_permeability(blood_flow, remove_nans=False):  # FIXME: verify that this is indeed permeability
    flow_sim = np.array(blood_flow)
    e = 1 - np.exp(-(VESSEL_PERMEABILITY_CONSTANT / abs(flow_sim)))  # Permeability ?
    if remove_nans:
        e = e[~np.isnan(e)]
    return e


def angles_to_orientation_parameters(angles, dist, limit_angle):
    """
    Compute the proportion of radial, planar and neutral orientations based on the angles and distances.

    radial_proportion = [dist, radials_mask, neutrals_mask, planars_mask], where:
        - The distance associated with the angle (dist)
        - A boolean value indicating if the angle is considered radial (radials)
        - A boolean value indicating if the angle is considered neutral (neutrals)
        - A boolean value indicating if the angle is considered planar (planars)

    .. warning: this is not a proportion, but a list of arrays with the same length as the input angles

    Parameters
    ----------
    angles
    dist
    limit_angle

    Returns
    -------

    """
    radials_mask = angles < limit_angle
    planars_mask = angles > (90 - limit_angle)
    neutrals_mask = np.logical_not(np.logical_or(radials_mask, planars_mask))
    # radial_proportion = np.concatenate(
    #     (np.expand_dims(dist, axis=1), np.concatenate(
    #             (np.expand_dims(radials_mask, axis=1), np.concatenate(
    #                     (np.expand_dims(neutrals_mask, axis=1), np.expand_dims(planars_mask, axis=1)), axis=1)
    #          ), axis=1)
    #      ), axis=1)
    radial_proportion = np.stack((dist, radials_mask, neutrals_mask, planars_mask), axis=1)  # TEST: check if this is equivalent to the above + what it means
    return planars_mask, radials_mask, radial_proportion


def compute_orientation(work_dir, graph, sub_graph, region_ids, controls, sample_name, average, mode, orientation_method):
    # FIXME: KW args and docstring
    print('Computing orientation ...')
    if orientation_method == 'local_normal':
        radial_orientations, planar_orientations = get_orientation_from_normal_to_surface_local(
            sub_graph, region_ids, normal_vector_method='mean')  # Try with 'svd'
        if np.count_nonzero(
                np.isnan(radial_orientations)) > 0:  # TODO: part of above with a flag but beware since recursive
            print('Nan values in radial orientations, removing them')
        planar_orientations = planar_orientations[~np.isnan(radial_orientations)]
        radial_orientations = radial_orientations[~np.isnan(radial_orientations)]
        planar_proportion = planar_orientations / (radial_orientations + planar_orientations)
        angles = np.degrees(np.arccos(radial_orientations))
        dist = sub_graph.edge_property('distance_to_surface')[~np.isnan(radial_orientations)]
    else:  # 'flow_interpolation':
        if average:
            angles = sub_graph.edge_property('angle')
        else:
            angles, graph = generalized_radial_planar_orientation(graph, work_dir, sample_name,
                                                                  4.5, controls,  # FIXME: magic number
                                                                  mode=mode, average=average)
        dist = sub_graph.edge_property('distance_to_surface')
        planar_proportion = []
    return graph, angles, dist, planar_proportion


# class GraphProps:  # TODO: could this be a DF ?
#     def __init__(self, work_dir, steps):
#         self.work_dir = work_dir
#         self.compute_steps = steps
#         self.edges = []  # FIXME: Edges or edges dist to surface ?
#         # self.branch_points = []  #  branch_poitns or branch_points_distance_to_surface ?
#         self.branch_point_distance_to_surface = []
#         self.arteries_branch_points = []  # FIXME: branch_poitns or branch_points_distance_to_surface ?
#         self.blood_flow = []
#         self.radial_orientation = []  # in flow
#         self.planar_orientation = []  # cross flow
#         self.proportion_radially_oriented = []
#         self.modularity = []
#         self.n_modules = []
#
#     def __get_path(self, base_name):
#         return os.path.join(self.work_dir, f'{base_name}_{self.postfix}')
#
#     def save_array(self, name, attr):
#         np.save(self.__get_path(name), attr)
#
#     def save(self, condition, sample_name, orientation_method):  # FIXME: the names should be saved in one place in the init
#         self.postfix = f'{condition}_{sample_name}.npy'
#         method_short = 'ln' if orientation_method == 'local_normal' else 'fi_bv'
#         self.save_array(f'ORI_{method_short}', self.radial_orientation)
#         self.save_array('ORI_plan', self.planar_orientation)
#         self.save_array(f'PROP_ORI_{method_short}', self.proportion_radially_oriented)
#         if self.compute_steps['arteries_branch_points']:
#             self.save_array('ARTBP', self.arteries_branch_points)
#         if self.compute_steps['sbm']:
#             self.save_array('SBMQ', self.modularity)
#             self.save_array('SBMNB', self.n_modules)
#         if self.compute_steps['flow']:
#             self.save_array('FLOW', self.blood_flow)
#         # self.save_array('BP', self.branch_points)
#         self.save_array('BP_dist', self.branch_point_distance_to_surface)
#         self.save_array('EP', self.edges)
#
#     def load_array(self, name):
#         return np.load(self.__get_path(name), allow_pickle=True)
#
#     def load(self, condition, sample_name, orientation_method):
#         self.postfix = f'{condition}_{sample_name}.npy'
#         method_short = 'ln' if orientation_method == 'local_normal' else 'fi_bv'
#         self.radial_orientation = self.load_array(f'ORI_{method_short}')  # FIXME: ori rad ??
#         self.planar_orientation = self.load_array('ORI_plan')
#         self.proportion_radially_oriented = self.load_array(f'PROP_ORI_{method_short}')
#         if self.compute_steps['arteries_branch_points']:
#             self.arteries_branch_points = self.load_array('ARTBP')
#         if self.compute_steps['sbm']:
#             self.modularity = self.load_array('SBMQ')
#             self.n_modules = self.load_array('SBMNB')
#         if self.compute_steps['flow']:
#             self.blood_flow = self.load_array('FLOW')
#         # self.branch_points = self.load_array('BP')
#         self.branch_point_distance_to_surface = self.load_array('BP_dist')
#         self.edges = self.load_array('EP')


def compute_brain_params_per_region(work_dir, sample_name, regions, artery_min_radius, compute_steps, controls,
                                    average, mode, orientation_method, limit_angle, sample_graph_suffix, vein_min_radius):
    """
    Compute the parameters for each region of the brain.
    The parameters include the
        - orientation
        - modularity
        - number of modules
        - arteries branch points
        - blood flow
        - proportion of radially oriented edges

    The output is saved in a GraphProps object. (which is a collection of lists). Is is essentially a table but
    each cell might be a list.
    This should probably be replaced by a DF with a column for the region ID and a column for each parameter.

    Parameters
    ----------
    work_dir
    sample_name
    regions
    artery_min_radius
    compute_steps
    controls
    average
    mode
    orientation_method
    limit_angle
    sample_graph_suffix
    vein_min_radius

    Returns
    -------

    """
    graph = ggt.load(os.path.join(work_dir, sample_name, f'{sample_name}_{sample_graph_suffix}.gt'))
    graph = filter_graph_degrees(graph)
    set_artery_vein_if_missing(graph, artery_min_radius=artery_min_radius, vein_min_radius=vein_min_radius)

    graph_stats = {
        'edge_props': {'region_id': [], 'distance_to_surface': [], 'dist_to_surface_no_nan': [],
                       'radial_orientation': [], 'planar_orientation': [], 'neutral_orientation': [],
                       'blood_flow': [], 'permeability': []},
        'vertices_props': {'region_id': [], 'distance_to_surface': []}
    }
    if compute_steps['arteries_branch_points']:
        graph_stats['arteries_vertices_props'] = {'region_id': [], 'distance_to_surface': []}
    if compute_steps['sbm']:
        graph_stats['graph_props'] = {'region_id': [], 'modularity': [], 'n_modules': []}

    # Precompute for whole brain
    if compute_steps['flow']:
        graph = compute_blood_flow(graph, work_dir, sample_name)
    if average and orientation_method == 'flow_interpolation':
        angles, graph = generalized_radial_planar_orientation(graph, work_dir, 4.5, controls,
                                                              mode=mode, average=average)  # FIXME: magic number
        graph.add_edge_property('angle', angles)  # Used by compute_orientation (sub_graph.edge_property('angle'))

    # Add stats for each region
    for region_id, region_params in regions.items():
        sub_graph = get_sub_graph(graph, region_params['ids'])

        graph_stats['vertices_props']['region_id'].append(np.repeat(region_id, sub_graph.n_vertices).tolist())
        graph_stats['vertices_props']['distance_to_surface'].append(sub_graph.vertex_property('distance_to_surface'))

        graph_stats['edge_props']['region_id'].append(np.repeat(region_id, sub_graph.n_edges))
        graph_stats['edge_props']['distance_to_surface'].append(sub_graph.edge_property('distance_to_surface'))

        graph, angles, dist, planar_proportion = compute_orientation(work_dir, graph, sub_graph, region_params['ids'],
                                                                     controls, sample_name, average, mode,
                                                                     orientation_method)

        graph_stats['edge_props']['edges_dist_no_nan'].append(dist)
        radials_mask = angles < limit_angle
        planars_mask = angles > (90 - limit_angle)
        graph_stats['edge_props']['radial_orientation'].append(radials_mask)  # TODO: dtype=bool
        graph_stats['edge_props']['planar_orientation'].append(planars_mask)  # TODO: dtype=bool
        neutrals_mask = np.logical_not(np.logical_or(radials_mask, planars_mask))
        graph_stats['edge_props']['neutral_orientation'].append(neutrals_mask)  # TODO: dtype=bool
        if compute_steps['flow']:  # Why not add velocities and pressures ?
            graph_stats['edge_props']['blood_flow'].append(sub_graph.edge_property('flow'))
            graph_stats['edge_props']['permeability'].append(
                graph_stats['edge_props'].groupby('region_id').apply(compute_permeability))

        if compute_steps['arteries_branch_points']:
            artery_distances = _compute_arteries_branch_points(sub_graph)
            graph_stats['arteries_vertices_props']['region_id'].append(np.repeat(region_id, artery_distances.size))
            graph_stats['arteries_vertices_props']['distance_to_surface'].append(artery_distances)

        # this only works if planar_proportion is not empty. i.e. if orientation_method == 'local_normal'
        if compute_steps['sbm'] and orientation_method == 'local_normal':
            graph_stats['graph_props']['region_id'].append(region_id)
            N, Q = _compute_sbm(sub_graph, planar_proportion)
            graph_stats['graph_props']['modularity'].append(Q)
            graph_stats['graph_props']['n_modules'].append(N)

    for k in graph_stats.keys():
        graph_stats[k] = pd.DataFrame(graph_stats[k])  # FIXME: check if we skip emtpy columns
    return graph_stats


def plot_selected_brain_regions(regions, features, norm=False, font_size='x-large'):
    """
    Plot individual values for selected_brain_regions

    Parameters
    ----------
    regions
    features
    norm
    font_size

    Returns
    -------

    """
    gp_names = list(features.keys())
    selected_brain_regions = ['VISp', 'AUDp', 'MOs', 'MOp', 'SSp-n', 'AUD']
    features_to_plot = ['BP', 'PROP_ORI_PLAN', 'PROP_ORI_RAD', 'ORI_RAD']

    colors = {
        gp_names[0]: ['royalblue', 'darkblue', 'forestgreen', 'lightseagreen'],
        gp_names[1]: ['darkred', 'indianred', 'darkgoldenrod', 'darkorange', 'forestgreen', 'lightseagreen']
    }
    line_plot_params = {'x': 'variable', 'y': 'value', 'err_style': 'bars', 'linewidth': 2.5}

    def feature_to_df(features, radial_orientations, f, norm=False):
        radial_features = features[:, radial_orientations, f]
        if norm:
            return pd.DataFrame(normalize(radial_features, norm='l2', axis=1)).melt()
        else:
            return pd.DataFrame(radial_features).melt()

    for radial_orientations in range(features[gp_names[0]].shape[1]):
        region_acronym = ano.find(regions[radial_orientations][0][0], key='id')['acronym']
        selected = region_acronym in selected_brain_regions
        print(f'Region {region_acronym} {"plotted" if selected else "skipped"}')
        if not selected:
            continue
        plt.figure()
        sns.set_style(style='white')

        # 3 ORI
        cpd = {gp_names[i]: feature_to_df(features[gp_names[i]], radial_orientations, 8, norm)  # FIXME: is f the feature index ? Coudn't we use the name ?
               for i in range(len(gp_names))}

        bp = {gp_names[i]: feature_to_df(features[gp_names[i]], radial_orientations, 2, norm)
              for i in range(len(gp_names))}

        for i in range(len(gp_names)):
            sns.lineplot(data=cpd[gp_names[i]], color=colors[gp_names[i]][i+1],
                         label=f'prop rad {gp_names[i]}', **line_plot_params)
        plt.yticks(size=font_size)
        plt.ylabel(features_to_plot[2], size=font_size)
        plt.xlabel('cortical depth (um)', size=font_size)
        plt.xticks(np.arange(0, 10), 25 * np.arange(0, np.max(bins), np.max(bins) / 10, dtype=int),
                   size=font_size, rotation=20)
        plt.twinx()
        for i in range(len(gp_names)):
            sns.lineplot(data=bp[gp_names[i]], color=colors[gp_names[i]][i+2],
                         label=f'bp {gp_names[i]}', **line_plot_params)

        plt.title(f'{region_acronym} {features_to_plot[0]}', size=font_size)
        plt.yticks(size=font_size)
        plt.ylabel(features_to_plot[0], size=font_size)


def bin_props(graph_props, prop_name, n_bins, normed):
    _, bins = graph_props[prop_name].agg({k: [np.histogram] for k in (
        [k for k in graph_props[prop_name].columns if k != 'region_id'])}, bins=n_bins, density=normed)  # density not really needed here
    # Apply the binning to all regions
    df_hist = pd.DataFrame(graph_props[prop_name].groupby("region_id").
                           agg({k: [np.histogram] for k in (
        [k for k in graph_props[prop_name].columns if k != 'region_id'])}, bins=bins, density=normed))
    return df_hist, bins  # FIXME: there must be a better way to store this


def main(work_dir, controls, mutants, regions, orientation_method='flow_interpolation', limit_angle=40, average=True,
         sample_graph_suffix='data_graph_correcteduniverse', artery_min_radius=4, vein_min_radius=8,
         n_bins=10, normed=False):
    """

    Parameters
    ----------
    work_dir : str
        The source directory
    controls : list[str]
        IDs of control samples so that os.path.join(work_dir, id_, '{sample_graph_suffix}.gt')
        is a valid path
    mutants : list[str]
        IDs of mutant samples so that os.path.join(work_dir, id_, '{sample_graph_suffix}.gt')
        is a valid path
    regions : dict
        The regions to consider. Should be a dict with the region main id as key the value being a
        dictionary of the form {'name': ..., 'ids': ...} with ids being all the ids forming the region.
    orientation_method : str
        The method to use for orientation. Should be in ('local_normal', 'flow_interpolation')
    limit_angle : int
        The limit angle to use for the orientation. Should be in (0, 90). It means that angles below this value
        will be considered radial, and angles above (90 - limit_angle) will be considered planar.
    average : bool
        Whether to return the average of the streamlines from the control subjects.
    sample_graph_suffix : str
        The suffix to use to find the graph files. Should be a valid suffix for the graph files.
    artery_min_radius : int
        The minimum radius to consider an edge as an artery.
    vein_min_radius : int
        The minimum radius to consider an edge as a vein.

    Returns
    -------
    """
    groups = {
        'control': controls,
        'mutant': mutants
    }  # TODO: make group names an argument

    mode = 'bigvessels'  # 'clusters', 'layers'

    branch_point_distance_threshold = 0.1  # WARNING: ???  (suggested by copilot)

    compute_steps = {'arteries_branch_points': True, 'sbm': True, 'flow': True}  # Normally False
    if orientation_method not in ('local_normal', 'flow_interpolation'):
        raise ValueError(f'Unrecognised orientation method {orientation_method},'
                         f' should be in ("local_normal", "flow_interpolation")')
    feature = 'vessels'  # 'vessels'  # art_raw_signal  # warning: unused

    for gp_name, brains in groups.items():
        for sample_name in tqdm(brains):
            graph_stats = compute_brain_params_per_region(work_dir, sample_name, regions, artery_min_radius,
                                                          compute_steps, controls, average, mode, orientation_method,
                                                          limit_angle, sample_graph_suffix, vein_min_radius)
            for k in graph_stats.keys():  # TODO: check if we should use gp_name also
                graph_stats[k].to_csv(os.path.join(work_dir, sample_name, f'{k}_{orientation_method}.csv'), sep=';')

    compute_steps['flow'] = False
    results = {}
    for gp_name, brains in groups.items():
        results[gp_name] = {}  # np.array((len(brains), nb_reg))

        for sample_name in brains:
            graph_props = {}
            for k in ('edge_props', 'vertices_props', 'arteries_vertices_props', 'graph_props'):
                df_path = os.path.join(work_dir, sample_name, f'{k}_{orientation_method}.csv')
                if os.path.exists(df_path):
                    graph_props[k] = pd.read_csv(df_path, sep=';')
            # Skip no_labels (ano.find(region_id, key='id')['acronym'] == 'NoL')

            # For each property, calculate the binning that works for all regions
            result_histograms = {k: bin_props(graph_props, k, n_bins, normed) for k in graph_props.keys()}
            orientation_columns = [c for c in graph_props['edge_props'].keys() if c.endswith('_orientation')]
            hist_all_orientations = result_histograms['edge_props'][0][orientation_columns].sum(axis=1)
            for column in orientation_columns:
                result_histograms['edge_props'][0][column] = result_histograms['edge_props'][0][column] / hist_all_orientations
                result_histograms['edge_props'][0]['close_branch_points'] = (
                    graph_props['edge_props'].groupby('region_id').branch_point_distance_to_surface.apply(
                        lambda x: np.histogram(x[x > branch_point_distance_threshold], bins=n_bins, density=normed)))

            for k in result_histograms.keys():
                dest_path = os.path.join(work_dir, sample_name, f'histograms_{orientation_method}.csv')
                result_histograms[k][0].to_csv(dest_path, sep=';')
                dest_path = os.path.join(work_dir, sample_name, f'bins_{orientation_method}.csv')
                result_histograms[k][1].to_csv(dest_path, sep=';')

            results[gp_name][sample_name] = result_histograms

    # plot_selected_brain_regions(regions, results)


def fix_annotation(work_dir, groups, sample_graph_suffix):

    for gp_name, brains in groups.items():
        for brain_id in brains:
            graph_path = os.path.join(work_dir, brain_id, f'{brain_id}_{sample_graph_suffix}.gt')
            graph = ggt.load(graph_path)
            label = graph.vertex_annotation()
            label_corrected = ano.convert_label(label, 'order', 'id')
            graph.set_vertex_annotation(label_corrected)
            graph.save(graph_path)


if __name__ == '__main__':
    work_dir = '/data/buffer/clearmap_tube_map/graph_comparison_experiment/'
    controls = ['1w', '3w', '5w']
    mutants = ['1k', '2k', '3k']
    df = pd.read_csv(os.path.join(settings.atlas_folder, 'regions_lambada.csv'), sep=';')
    structure_ids_path_base = '997/8/567/688/695/315'  # isocortex. TODO: add bulb
    relevant_df = df[df['structure_ids_path'].str.startswith(structure_ids_path_base)]
    lambada_relevant_ids = relevant_df.groupby('lambada_id')['lambada_id'].first().values
    is_development = False
    # get the name column where the id column is in lambada_relevant_ids
    names_map = relevant_df[relevant_df['id'].isin(lambada_relevant_ids)][['id', 'name']]
    # make that an id->name dict
    names_map = names_map.set_index('id')['name'].to_dict()
    if is_development:
        regions = {id_: {'name': name} for id_, name in names_map.items()}
    else:
        regions = {id_: {'name': name,
                         'ids': relevant_df[relevant_df['lambada_id'] == id_]['id'].values}
                   for id_, name in names_map.items()}
    main(work_dir, controls, mutants, regions, orientation_method='flow_interpolation', limit_angle=40, average=True,
         sample_graph_suffix='graph', artery_min_radius=4, vein_min_radius=8, n_bins=10)
