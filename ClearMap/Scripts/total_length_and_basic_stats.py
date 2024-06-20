import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

import ClearMap.Settings as settings
from ClearMap.Analysis.vasculature.vasc_graph_utils import remove_surface
from ClearMap.IO import IO as clearmap_io
import ClearMap.Alignment.Annotation as annotation
import ClearMap.Analysis.Graphs.GraphGt as ggt


MAX_PROCS = 28  # Max number of processes to use for parallel processing
MIN_CHUNK_SIZE = 5
DEV_CUTOFF_TIMEPOINT = 30  # below this is considered as developmental
GET_REG = True

JSON_PATH = os.path.join(settings.atlas_folder, 'ABA_annotation.json')  # FIXME: should use new atlas
ADULT_ATLAS_PATH = os.path.join(settings.atlas_folder, 'ABA_25um_annotation.tif')  # FIXME: replace with new atlas
LAMBADA_ATLAS_BASE_DIR = '/data/elisa.delaunoit/ATLAS_26July'
LAMBADA_BASE_NAME = 'LAMBADA_25um_annotation_p{}.tif'  # FIXME: MRI
WORKDIRS_BASE = '/data/elisa.delaunoit/development_stats/TP'
OUTPUT_DF_BASE_NAME = 'vasculature_stats_p{}.csv'
REF_BRAIN_PATH = os.path.join(WORKDIRS_BASE, 'P5', '5b_graph.gt')  # FIXME: should vary w/ timepoint
SAMPLES_INFO = {
    'P3': {'sample_ids': ['1', '2', '3a', '3', '3c', '3d']},
    'P5': {'sample_ids': ['1', '2', '3', '5a', '5b']},
    'P7': {'sample_ids': ['a', 'b', 'c', 'd', 'e']},
    'P9': {'sample_ids': ['2', '3', '4', '9a', '9c', '9d']},
    'P12': {'sample_ids': ['2', '3', '1', '12a', '12B', '12c', '12d']},
    'P14': {'sample_ids': ['2', '3', '14a', '14b', '14c', '14d']},
    'P21': {'sample_ids': ['2', '3', '21a', '21b', '21e', '21d']},
    'P30': {'sample_ids': ['4', '7', '10', '15']},
    'P60': {'sample_ids': ['3', '1', '4', '5', '6']},
}


# ######################################################################################################################


def get_regions_from_atlas(atlas_path):  # FIXME: add volume
    uniq_ids = np.unique(clearmap_io.read(atlas_path))[1:]  # exclude universe
    regions = {id_: {'name': annotation.find(id_, key='id')['name'],
                     'level': annotation.find(id_, key='id')['level'],
                     'color': annotation.convert_label(id_, key='id', value='rgba')}
               for id_ in uniq_ids}
    return regions


def get_regions_from_graph(graph):
    uniq_ids = np.unique(graph.vertex_annotation())[1:]  # exclude universe
    regions = {id_: {'name': annotation.find(id_, key='id')['name'],
                     'level': annotation.find(id_, key='id')['level'],
                     'color': annotation.convert_label(id_, key='id', value='rgba')}
               for id_ in uniq_ids}
    return regions


def get_regions_auto(graph):
    order, level = (0, 0)

    # I think this fuses the structures that are at a level below the one specified
    vertex_filter = np.zeros(graph.n_vertices)
    label_leveled = annotation.convert_label(graph.vertex_annotation(), key='id', value='id', level=level)
    vertex_filter[label_leveled == order] = 1
    graph = graph.sub_graph(vertex_filter=vertex_filter)

    return graph, get_regions_from_graph(graph)


def extract_annotated_region(graph, id_, region, state='id', return_graph=True):
    # print(state, region['level'], id_, annotation.find(id_, key='id')['name'])

    label = graph.vertex_annotation()

    if state == 'order':
        print('order')
        label_leveled = annotation.convert_label(label, key='order', value='order', level=region['level'])
        order = annotation.find(id_, key='id')['order']
        vertex_filter = label_leveled == order
    else:
        try:
            label[label < 0] = 0
            label_leveled = annotation.convert_label(label, key='id', value='id', level=region['level'])
            vertex_filter = label_leveled == id_
        except KeyError:
            print('could not extract region')

    if return_graph:
        return graph.sub_graph(vertex_filter=vertex_filter)
    else:
        return vertex_filter


def get_length(coordinates):
    diff = np.diff(coordinates, axis=0)
    return np.sum(np.linalg.norm(diff, axis=1))


def get_total_length(graph, coordinates_type):
    coordinates_chunks = get_coordinate_chunks(graph, coordinates_type)
    try:  # try parallel processing
        n_chunks_per_core = int((len(coordinates_chunks) / MAX_PROCS) / 2)
        n_chunks_per_core = max(MIN_CHUNK_SIZE, n_chunks_per_core)
        with ProcessPoolExecutor(MAX_PROCS) as executor:
            results = executor.map(get_length, coordinates_chunks, chunksize=n_chunks_per_core)
        results = list(results)
        lengths = np.array(results)
    except BrokenProcessPool:  # Go single threaded
        lengths = np.array([get_length(chunk) for chunk in coordinates_chunks])
    return lengths


def compute_stat_on_radii(graph, stat_func=np.std):
    radii_chunks = get_radii_chunks(graph)
    try:  # try parallel processing
        n_chunks_per_core = int((len(radii_chunks) / MAX_PROCS) / 2)
        n_chunks_per_core = max(MIN_CHUNK_SIZE, n_chunks_per_core)
        with ProcessPoolExecutor(MAX_PROCS) as executor:
            results = executor.map(stat_func, radii_chunks, chunksize=n_chunks_per_core)
        results = list(results)
        stats = np.array(results)
    except BrokenProcessPool:  # Go single threaded
        stats = np.array([stat_func(chunk) for chunk in radii_chunks])
    return stats


def get_annotation_chunks(graph):
    annotation_chunks = graph.edge_geometry_property('annotation')
    indices = graph.edge_property('edge_geometry_indices')
    annotation_chunks = [annotation_chunks[ind[0]:ind[1]] for ind in indices]
    return annotation_chunks


def vote_annotation(annotations):
    return np.argmax(np.bincount(annotations))


def get_edge_annotations(graph):
    annotation_chunks = get_annotation_chunks(graph)
    try:  # try parallel processing
        n_chunks_per_core = int((len(annotation_chunks) / MAX_PROCS) / 2)
        n_chunks_per_core = max(MIN_CHUNK_SIZE, n_chunks_per_core)
        with ProcessPoolExecutor(MAX_PROCS) as executor:
            results = executor.map(vote_annotation, annotation_chunks, chunksize=n_chunks_per_core)
        results = list(results)
        results = np.array(results)
    except BrokenProcessPool:  # Go single threaded
        annotations = np.array([vote_annotation(chunk) for chunk in annotation_chunks])
    return annotations


def get_coordinate_chunks(graph, coordinates_type):  # REFACTOR: improve docstring (are those pairs ?)
    coordinates = graph.edge_geometry_property(coordinates_type)
    indices = graph.edge_property('edge_geometry_indices')
    coordinates_chunks = [coordinates[ind[0]:ind[1]] for ind in indices]
    return coordinates_chunks


def get_radii_chunks(graph):
    radii = graph.edge_geometry_property('radii')
    indices = graph.edge_property('edge_geometry_indices')
    radii_chunks = [radii[ind[0]:ind[1]] for ind in indices]
    return radii_chunks


def update_samples_info(samples_info, workdirs_base, lambada_atlas_base_dir, lambada_base_name, adult_atlas_path):
    for time_point in samples_info.keys():
        samples_info[time_point]['work_dir'] = os.path.join(workdirs_base, time_point)
        num_time_point = int(time_point[1:])
        if num_time_point in DEV_TIME_POINTS:
            samples_info[time_point]['atlas_path'] = os.path.join(lambada_atlas_base_dir, time_point,
                                                                  lambada_base_name.format(num_time_point))
        else:
            samples_info[time_point]['atlas_path'] = adult_atlas_path
    return samples_info


def plot_length(df, regions):
    for reg_name in [r['name'] for r in regions.values()]:
        plt.figure()
        sns.lineplot(x="timepoint", y='n_vertices/length', err_style='bars', color='indianred',
                     data=df[df['region'] == reg_name])
        plt.ylim(5000, 15000)
        plt.twinx()
        sns.lineplot(x="timepoint", y='n_vertices/length', err_style='bars', color='cadetblue',
                     data=df)
        plt.ylim(5000, 15000)
        plt.title(reg_name)

        plt.figure()
        sns.lineplot(x="timepoint", y='bp', err_style='bars', color='indianred',
                     data=df[df['region'] == reg_name])
        plt.ylim(0, 5000)
        plt.twinx()
        sns.lineplot(x="timepoint", y='nbdeg1/length', err_style='bars', color='cadetblue',
                     data=df)
        plt.ylim(0, 5000)
        plt.title(reg_name)


def uncrust_graph(time_point, work_dir, samples):
    print(time_point, work_dir, samples)
    for sample_id in tqdm(samples):
        graph = ggt.load(os.path.join(work_dir, f'{sample_id}_graph.gt'))
        graph_uncrusted = remove_surface(graph, 2)
        graph_uncrusted.save(os.path.join(work_dir, f'{sample_id}_graph_uncrusted.gt'))

# ######################################################################################################################


def compute_stats(samples_info, get_reg=False, regions=None, uncrust=False, coordinates_type='MRI_coordinates'):
    """
    Extract basic stats for all the previously selected brains and timepoints

    Parameters
    ----------

    samples_info : dict
        dict of dicts containing the info for each timepoint
    get_reg : bool
        whether to extract the regions from the reference brain or not
    regions : dict
        dict of dicts containing the info for each region
    uncrust: bool
        whether to remove the surface or not
    coordinates_type: str
        type of coordinates to use for the length computation (one of ('MRI_coordinates', 'coordinates'))
    """

    out_df = pd.DataFrame()
    for time_point, info in samples_info.items():
        print(time_point, info['work_dir'], info['sample_ids'])
        time_point_int = int(time_point[1:])

        result_dict = {
            'time_point': [],
            'brain_ID': [],
            'region': [],
            'total_length': [],
            'n_edges': [],
            'n_vertices': [],
            'nb_deg1': [],
            'deg_1_ratio': []
        }

        if uncrust:
            uncrust_graph(time_point, info['work_dir'], info['sample_ids'])  # TODO: check order w/ Elisa

        annotation.initialize(label_file=JSON_PATH, extra_label=None, annotation_file=(info['atlas_path']))  # WARNING: always extra_label=None

        stage = "dev" if time_point_int in DEV_TIME_POINTS else "adult"
        print(f'Processing {stage} brain')

        for sample_index, sample_id in enumerate(tqdm(info['sample_ids'], unit='sample n')):
            try:
                graph = ggt.load(os.path.join(info['work_dir'], f'{sample_id}_graph.gt'))
            except FileNotFoundError:
                print('graph is missing !')
                continue
            graph = graph.largest_component()

            if (regions is None or get_reg) and sample_index == 0:
                # graph, regions = get_regions_auto(graph)  # TODO: check why graph is returned
                # regions = get_regions_from_graph(graph)
                regions = get_regions_from_atlas(info['atlas_path'])
            # print(f'IDs: {regions.keys()}')

            for id_, region in (region_pbar := tqdm(regions.items(), position=0, leave=True, colour='green', ncols=120)):  # Get stat for each region
                region_pbar.set_postfix_str(f'region: {region["name"]}')

                try:
                    region_graph = extract_annotated_region(graph, id_, region)
                except IndexError:  # No index for subgraph
                    print(f'Skipping empty region {region["name"]}')
                    continue

                n_deg1 = np.sum(region_graph.vertex_degrees() == 1)

                length = get_total_length(region_graph, coordinates_type)

                result_dict['time_point'].append(time_point_int)
                result_dict['brain_ID'].append(sample_id)
                result_dict['region'].append(region['name'])
                result_dict['total_length'].append(np.sum(length) * 1e3)
                result_dict['n_edges'].append(region_graph.n_edges)
                result_dict['n_vertices'].append(region_graph.n_vertices)
                result_dict['nb_deg1'].append(n_deg1)
                result_dict['deg_1_ratio'].append(n_deg1 / region_graph.n_vertices)

        df = pd.DataFrame(result_dict)
        df.to_csv(os.path.join(WORKDIRS_BASE, OUTPUT_DF_BASE_NAME.format(time_point_int)), index=False)

        out_df.append(df, ignore_index=True)

    return out_df


if __name__ == '__main__':
    DEV_TIME_POINTS = (int(tp[1:]) for tp in SAMPLES_INFO.keys() if
                       int(tp[1:]) < DEV_CUTOFF_TIMEPOINT)  # (3, 5, 7, 9, 12, 14, 21)
    SAMPLES_INFO = update_samples_info(SAMPLES_INFO, WORKDIRS_BASE,
                                       LAMBADA_ATLAS_BASE_DIR, LAMBADA_BASE_NAME, ADULT_ATLAS_PATH)

    if not GET_REG:
        ref_brain_graph = ggt.load(REF_BRAIN_PATH)
        regions = get_regions_from_graph(ref_brain_graph)
    else:
        regions = None
    df = compute_stats(SAMPLES_INFO, get_reg=GET_REG, regions=regions, coordinates_type='coordinates_atlas')
    plot_length(df, regions)
