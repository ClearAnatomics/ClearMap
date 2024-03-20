import os
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize

import seaborn as sns

import graph_tool.inference as gti

import ClearMap.Analysis.Graphs.GraphGt as ggt
import ClearMap.Alignment.Annotation as ano
from ClearMap.Scripts.vasculature_stats_general_fonctions import from_e_prop_to_v_prop, from_v_prop_to_e_prop, \
    generalized_radial_planar_orientation, get_radial_planar_orientation, \
    computeFlowFranca, ps, modularity_measure, load_sample_dict


def get_sub_regions(reg_list, region_list):
    """
    Get regions that share part of the name

    Parameters
    ----------
    reg_list
    region_list

    Returns
    -------

    """
    regions = []
    main_region_name = ano.find(region_list[0][0], key='order')['name']
    for region in reg_list.keys():
        reg_name = ano.find_name(region, key='order')
        if main_region_name in reg_name:
            for se in reg_list[region]:
                reg_name = ano.find_name(se, key='order')
                regions.append(reg_name)
    return regions


def compute_arteries_and_veins_if_missing(graph):
    try:
        graph.vertex_property('artery')
        graph.vertex_property('vein')
    except:  # FIXME: too broad
        try:
            from_e_prop_to_v_prop(graph, 'artery')
            from_e_prop_to_v_prop(graph, 'vein')
        except:  # FIXME: too broad
            print('no artery vertex properties')
            artery = np.logical_and(graph.vertex_radii() >= 4.8, graph.vertex_radii() <= 8)  # 4
            vein = graph.vertex_radii() >= 8
            graph.add_vertex_property('artery', artery)
            graph.add_vertex_property('vein', vein)
            artery = from_v_prop_to_e_prop(graph, artery)
            graph.add_edge_property('artery', artery)
            vein = from_v_prop_to_e_prop(graph, vein)
            graph.add_edge_property('vein', vein)


def compute_blood_flow(work_dir, sample_name, graph):
    try:
        sample_dict = load_sample_dict(work_dir, sample_name)
        f = np.asarray(sample_dict['flow'][0])
        v = np.asarray(sample_dict['v'][0])
    except KeyError:
        f, v = computeFlowFranca(work_dir, graph, sample_name)
        sample_dict = load_sample_dict(work_dir, sample_name)
    graph.add_edge_property('flow', f)
    graph.add_edge_property('veloc', v)
    pressure = np.asarray(sample_dict['pressure'][0])
    graph.add_vertex_property('pressure', pressure)
    return graph


def _compute_sbm(graph, p, r):
    print('sbm ... ')
    Qs = []
    N = []
    graph.add_edge_property('rp', p / (r + p))
    g = graph.base
    for i in range(5):
        state = gti.minimize_blockmodel_dl(g, state_args=dict(recs=[g.ep.param, g.ep.invL, g.ep.rp],
                                                              rec_types=["real-exponential"] * 3))
        modules = state.get_blocks().a
        graph.add_vertex_property('blocks', modules)
        try:
            Q, Qbis = modularity_measure(modules, graph, 'blocks')
        except:   # FIXME: too broad
            Q = 0
        s = np.unique(modules).shape[0]
        Qs.append(Q)
        N.append(s)
    return N, Qs


def _compute_art_bp(graph):
    artery = from_e_prop_to_v_prop(graph, 'artery')
    vertex_filter = np.logical_and(artery, graph.vertex_property('artery_binary') > 0)  # np.logical_and()
    art_g = graph.sub_graph(vertex_filter=vertex_filter)
    dist_art = art_g.vertex_property('distance_to_surface')
    return dist_art


def group_region(regis, struct_name, features, extra_reg_ids, struct_acronyms=None):
    print(struct_name)
    inds = []
    for i, r in enumerate(regis):
        order, level = r[0]
        n = ano.find(order, key='order')['acronym']
        if struct_acronyms is None:
            struct_acronyms = [struct_name.upper()]
        for acro in struct_acronyms:
            if acro in n:
                print(n)
                inds.append(i)
    grouped_regions = np.mean(features[:, inds, :], axis=1)
    grouped_regions = np.expand_dims(grouped_regions, axis=1)
    features = np.delete(features, inds, axis=1)
    features = np.concatenate((features, grouped_regions), axis=1)
    regis = np.delete(regis, inds, axis=0)
    regis = np.concatenate((regis, np.expand_dims(np.array(extra_reg_ids), axis=0)))
    return features, regis


def get_sub_graph(graph, region_list):
    vertex_filter = np.zeros(graph.n_vertices)
    for order, level in region_list:
        print(level, order, ano.find(order, key='order')['name'])
        label = graph.vertex_annotation()
        label_leveled = ano.convert_label(label, key='order', value='order', level=level)
        vertex_filter[label_leveled == order] = 1
    gss4_t = graph.sub_graph(vertex_filter=vertex_filter)
    gss4 = gss4_t.copy()
    return gss4, gss4_t


class GraphProps:
    def __init__(self, work_dir, steps):
        self.work_dir = work_dir
        self.compute_steps = steps
        self.edges = []
        self.branch_points = []
        self.branch_point_distance_to_surface = []
        self.radial_orientation = []  # in flow
        self.planar_orientation = []  # cross flow
        self.modularity = []
        self.n_modules = []
        self.arteries_branch_points = []
        self.blood_flow = []
        self.proportion_radialy_oriented  # FIXME: = what ?? [] ??

    def get_path(self, base_name):
        return os.path.join(self.work_dir, f'{base_name}_{self.postfix}')

    def save_array(self, name, attr):
        np.save(self.get_path(name), attr)

    def save(self, condition, sample_name, orientation_method):
        self.postfix = f'{condition}_{sample_name}.npy'
        if orientation_method == 'flow_interpolation':
            np.save(self.get_path('ORI_fi_bv'), self.radial_orientation)
            np.save(self.get_path('ORI_plan'), self.planar_orientation)
            np.save(self.get_path('PROP_ORI_fi_bv'), self.proportion_radialy_oriented)
        elif orientation_method == 'local_normal':
            np.save(self.get_path('ORI_ln_test'), self.radial_orientation)
            np.save(self.get_path('ORI_plan'), self.planar_orientation)
            np.save(self.get_path('PROP_ORI_ln_test'), self.proportion_radialy_oriented)

        if self.compute_steps['art_bp']:
            np.save(self.get_path('ARTBP'), self.arteries_branch_points)
        if self.compute_steps['sbm']:
            np.save(self.get_path('SBMQ'), self.modularity)
            np.save(self.get_path('SBMNB'), self.n_modules)
        if self.compute_steps['flow']:
            np.save(self.get_path('FLOW'), self.blood_flow)
        np.save(self.get_path('BP'), self.branch_points)
        np.save(self.get_path('EP'), self.edges)


def main(work_dir, controls, mutants):
    """

    Parameters
    ----------
    work_dir str:
        The source directory
    controls list:
        list of IDs
    mutants list:
        list of IDs

    Returns
    -------
    """
    groups = [controls, mutants]

    condition = 'isocortex'
    mode = 'bigvessels'  # 'clusters', 'layers'

    compute_steps = {'art_bp': False, 'sbm': False, 'flow': False}
    orientation_method = 'flow_interpolation'  # in ('local_normal', 'flow_interpolation')
    feature = 'vessels'  # 'vessels'  # art_raw_signal  # FIXME: unused
    limit_angle = 40
    average = True

    reg_list = {}   # FIXME: all regions read from json

    #  Fuse structures
    regions_lists = {
        'Aud_p': [(142, 8)],
        'Aud': [(127, 7)],
        'Aud_po': [(149, 8)],
        'Aud_d': [(128, 8)],
        'Aud_v': [(156, 8)],
        'Ssp': [(40, 8)],
        'barrels': [(54, 9)],
        'nose': [(47, 9)],
        'mouth': [(75, 9)]
    }
    hard_coded_regions_lists = {
        'Auditory_regions': [(142, 8), (149, 8), (128, 8), (156, 8)],
        'barrel_region': [(54, 9), (47, 9)],
        'l2 barrels': [(56, 10), (49, 10), (77, 10)],
        'l4 barrels': [(58, 10), (51, 10), (79, 10)]
    }
    if condition in regions_lists.keys():
        regions = get_sub_regions(reg_list, regions_lists[condition])
    elif condition in hard_coded_regions_lists.keys():
        regions = hard_coded_regions_lists[condition]
    elif condition == 'isocortex':
        regions = []  # FIXME: all cortical regions folded by layers ()

    for brains in groups:
        for sample_name in brains:
            print(sample_name)
            graph_properties = GraphProps(work_dir, steps=compute_steps)

            graph = ggt.load(os.path.join(work_dir, sample_name, 'data_graph_correcteduniverse.gt'))
            degrees = graph.vertex_degrees()
            vf = np.logical_and(degrees > 1, degrees <= 4)
            graph = graph.sub_graph(vertex_filter=vf)

            compute_arteries_and_veins_if_missing(graph)

            if compute_steps['flow']:
                print("Computing flow...")
                graph = compute_blood_flow(work_dir, sample_name, graph)
                print('done')

            if orientation_method == 'flow_interpolation' and average:
                angle, graph = generalized_radial_planar_orientation(graph, sample_name, 4.5, controls,
                                                                     mode=mode, average=average)
                graph.add_edge_property('angle', angle)

            for region_list in regions:
                gss4, gss4_t = get_sub_graph(graph, region_list)

                graph_properties.branch_point_distance_to_surface.append(gss4.vertex_property('distance_to_surface'))
                graph_properties.branch_points.append(gss4.vertex_property('distance_to_surface'))
                graph_properties.edges.append(gss4.edge_property('distance_to_surface'))

                print('Computing orientation ...')
                orientation_computation_failed = False
                if orientation_method == 'local_normal':
                    try:
                        r, p, _, _ = get_radial_planar_orientation(gss4, gss4_t, local_normal=True, calc_art=False)  # , calc_art=True)
                        p = p[~np.isnan(r)]  # WARNING: before removing from r
                        r = r[~np.isnan(r)]  # TODO: check if ~ is not deprecated
                        angle = np.array([math.acos(r[i]) for i in range(r.shape[0])]) * 180 / math.pi

                        dist = gss4.edge_property('distance_to_surface')[~np.isnan(r)]
                    except:  # FIXME: too broad
                        orientation_computation_failed = True
                elif orientation_method == 'flow_interpolation':
                    try:  # FIXME: missing r
                        if average:
                            angle = gss4.edge_property('angle')
                        else:
                            angle, graph = generalized_radial_planar_orientation(graph, sample_name, 4.5, controls,
                                                                                 mode=mode, average=average)

                        dist = gss4.edge_property('distance_to_surface')
                    except:  # FIXME: too broad
                        orientation_computation_failed = True
                else:
                    raise NotImplementedError(f'Unrecognised orientation method {orientation_method}')

                if orientation_computation_failed:  # set defaults
                    print(f'IndexError - error in {orientation_method}')
                    r, p, dist, angle = [np.zeros(gss4.n_edges) for i in range(4)]  # Ignore this graph

                radiality = angle < limit_angle  # 40
                planarity = angle > 90 - limit_angle  # 60
                neutral = np.logical_not(np.logical_or(radiality, planarity))

                ori_prop = np.concatenate(
                    (np.expand_dims(dist, axis=1), np.concatenate((np.expand_dims(radiality, axis=1), np.concatenate(
                        (np.expand_dims(neutral, axis=1), np.expand_dims(planarity, axis=1)), axis=1)), axis=1)),
                    axis=1)
                graph_properties.proportion_radialy_oriented.append(ori_prop)

                graph_properties.planar_orientation.append(angle > 90 - limit_angle)
                graph_properties.radial_orientation.append(angle < limit_angle)
                print('done')

                if compute_steps['art_bp']:
                    graph_properties.arteries_branch_points.append(_compute_art_bp(gss4_t))

                if compute_steps['flow']:
                    graph_properties.blood_flow.append(gss4.edge_property('flow'))

                rad = gss4.vertex_property('radii')
                connect = gss4.edge_connectivity()
                param = [1 / (1 + abs(1 - (rad[connect[i][0]] / rad[connect[i][1]]))) for i in np.arange(gss4.n_edges)]
                gss4.add_edge_property('param', np.array(param).astype(float))
                gss4.add_edge_property('invL', 1 / gss4.edge_property('length'))  # FIXME: rename

                if compute_steps['sbm']:
                    N, Qs = _compute_sbm(gss4, p, r)
                    graph_properties.modularity.append(np.mean(np.array(Qs)))
                    graph_properties.n_modules.append(np.mean(np.array(N)))
                    print('done')

                graph_properties.save()

    compute_steps['flow'] = False
    for brains in groups:
        normed = False
        regis = []
        features = []  # np.array((len(brains), nb_reg))

        grouped = True
        aud = True
        vis = True
        rsp = True
        nose = True
        trunk = True

        reg_lis = True

        for sample_name in brains:  # FIXME: grpah_props.load
            postfix = f'{condition}_{sample_name}.npy'
            try:
                SBM_modules = np.load(os.path.join(work_dir, f'SBMNB_{postfix}'))
                SBM_modularity = np.load(os.path.join(work_dir, f'SBMQ_{postfix}'))
            except FileNotFoundError:
                print('No sbm were computed')
                SBM_modules = np.zeros(4)
                SBM_modularity = np.zeros(4)

            if compute_steps['flow']:
                flow_simulation = np.load(os.path.join(work_dir, f'FLOW_{postfix}'), allow_pickle=True)

            vess_rad_control = np.load(os.path.join(work_dir, f'ORI_fi_bv{postfix}'), allow_pickle=True)
            prop_ori_control = np.load(os.path.join(work_dir, f'PROP_ORI_fi_bv{postfix}'), allow_pickle=True)
            # prop_ori_control = np.load(os.path.join(work_dir, f'PROP_ORI_{postfix}'),allow_pickle=True)
            bp_dist_2_surface_control = np.load(os.path.join(work_dir, f'BP_{postfix}'), allow_pickle=True)
            if compute_steps['art_bp']:
                art_ep_dist_2_surface_control = np.load(os.path.join(work_dir, f'ARTBP_{postfix}'), allow_pickle=True)
            ve_ep_dist_2_surface_control = np.load(os.path.join(work_dir, f'EP_{postfix}'), allow_pickle=True)

            features_brains = []
            for reg in range(len(regions)):
                order, level = regions[reg][0]
                n = ano.find(order, key='order')['acronym']
                id_ = ano.find(order, key='order')['id']
                print(id_, n, (n != 'NoL' or id_ == 182305712))
                if n != 'NoL' or id_ == 182305712:
                    if '6' not in n:
                        if reg_lis:
                            print(regions[reg])
                            regis.append(regions[reg])
                        try:
                            if compute_steps['flow']:
                                flow_sim = np.array(flow_simulation[reg])
                                e = 1 - np.exp(-(ps / abs(flow_sim)))
                                e = e[~np.isnan(e)]
                            try:
                                sbm_nb = np.array(SBM_modules[reg])
                                sbm_q = np.array(SBM_modularity[reg])
                            except:  # FIXME: broad
                                sbm_nb = np.zeros(10)
                                sbm_q = np.zeros(10)
                            bp_dist = np.array(bp_dist_2_surface_control[reg])
                            if compute_steps['art_bp']:
                                art_ep = np.array(art_ep_dist_2_surface_control[reg])
                            ve_ep = np.array(ve_ep_dist_2_surface_control[reg])
                            orientation = np.array(vess_rad_control[reg])
                            # radial_ori = ori[:int(len(ori)/2)]
                            # radial_depth = ori[int(len(ori)/2):]
                            radial_depth = ve_ep[orientation > 0.6]
                            if compute_steps['flow']:
                                hist_flow, bins_flow = np.histogram(e, bins=bin2, normed=normed)
                            if compute_steps['art_bp']:
                                hist_art_ep, bins_art_ep = np.histogram(art_ep, bins=bin, normed=normed)
                            hist_ve_ep, bins_ve_ep = np.histogram(ve_ep, bins=bin, normed=normed)
                            hist_bp_dist, bins_bp_dist = np.histogram(bp_dist[bp_dist > thresh], bins=bin, normed=normed)
                            # , np.sum(np.mean(H, axis=1)))
                            hist_ori, bins_ori_dist = np.histogram(radial_depth, bins=bin, normed=normed)

                            dist = prop_ori_control[reg][:, 0]
                            ori_rad = prop_ori_control[reg][:, 1]
                            ori_neutral = prop_ori_control[reg][:, 2]
                            ori_plan = prop_ori_control[reg][:, 3]
                            histrad, bins = np.histogram(dist[ori_rad.astype(bool)], bins=10)
                            histneut, bins = np.histogram(dist[ori_neutral.astype(bool)], bins=bins)
                            histplan, bins = np.histogram(dist[ori_plan.astype(bool)], bins=bins)
                            R = histrad / (histrad + histneut + histplan)
                            N = histneut / (histrad + histneut + histplan)
                            P = histplan / (histrad + histneut + histplan)

                            tmp_features = []
                            if compute_steps['art_bp']:
                                tmp_features.append(hist_art_ep)
                            else:
                                tmp_features.append(np.zeros(10))
                            tmp_features.extend([hist_ve_ep, hist_bp_dist, hist_ori, sbm_q, sbm_nb])
                            if compute_steps['flow']:
                                print('features with flow sim')
                                tmp_features.append(hist_flow)
                            else:
                                print('features without flow sim')
                            features_brains.append(tmp_features + [P, N, R])
                            print(reg, 'works!')
                        except:  # FIXME: broad
                            print(reg, 'no data')
                            features_brains.append([np.zeros(10) for _ in range(10)])

            features.append(features_brains)
            reg_lis = False

        shape = (len(brains), len(features_brains), len(features_brains[0]), bin)
        F = np.zeros(shape).astype(float)
        F = F.astype(float)
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    F[i, j, k, :] = features[i][j][k].astype(float)
        # features = np.array(features)
        features = np.nan_to_num(F)
        regis = np.array(regis)

        if grouped:
            for proceed, structure_name, extra_reg_ids, acronyms in [(aud, 'aud', [[127, 7]], None),
                                                                    (vis, 'vis', [[163, 7]], None),
                                                                    (rsp, 'rsp', [[303, 7]], None),
                                                                    (nose, 'nose', [[47, 9]], ['bfd', 'SSp-n']),
                                                                    (trunk, 'trunk', [[89, 9]], ['SSp-ll', 'SSp-tr'])]:
                print(features.shape, regis.shape)
                if proceed:
                    features, regis = group_region(regis, structure_name, features, extra_reg_ids, struct_acronyms=acronyms)

        if brains == controls:
            features_avg = np.mean(features, axis=0)
            features_avg_c = features_avg  # FIXME: unused
            features_c = features
        if brains == mutants:
            features_avg = np.mean(features, axis=0)
            features_avg_m = features_avg  # FIXME: unused
            features_m = features

    # ############### plot individual values for ROI ###############
    # ROI = ['Alp', 'VIS', 'AUDd', 'SSp-ul', 'SSp-ll', 'SSp-bfd', 'SSs', 'ILA', 'RSP', 'SSp-m', 'SSp-tr']
    ROI = ['VISp', 'AUDp', 'MOs', 'MOp', 'SSp-n', 'AUD']

    colors_c = ['royalblue', 'darkblue', 'forestgreen', 'lightseagreen']
    colors_m = ['darkred', 'indianred', 'darkgoldenrod', 'darkorange', 'forestgreen', 'lightseagreen']
    norm = False
    # bins = bins_bp_dist
    feat = ['BP', 'PROP_ORI_PLAN', 'PROP_ORI_RAD', 'ORI_RAD']  # ['ART EP', 'VE EP', 'BP', 'ORI']  # 'SP len', 'SP step'

    for r in range(features.shape[1]):
        region_acronym = ano.find(regis[r][0][0], key='order')['acronym']
        selected = region_acronym in ROI
        print(f'Region {region_acronym} {"plotted" if selected else "skipped"}')
        if selected:
            plt.figure()
            sns.set_style(style='white')

        f = 8  # 3 ORI
        if norm:
            Cpd_c = pd.DataFrame(normalize(features_c[:, r, f], norm='l2', axis=1)).melt()
            Cpd_m = pd.DataFrame(normalize(features_m[:, r, f], norm='l2', axis=1)).melt()
        else:
            Cpd_c = pd.DataFrame(features_c[:, r, f]).melt()
            Cpd_m = pd.DataFrame(features_m[:, r, f]).melt()

        f = 2  # BP
        if norm:
            bp_c = pd.DataFrame(normalize(features_c[:, r, f], norm='l2', axis=1)).melt()
            bp_m = pd.DataFrame(normalize(features_m[:, r, f], norm='l2', axis=1)).melt()
        else:
            bp_c = pd.DataFrame(features_c[:, r, f]).melt()
            bp_m = pd.DataFrame(features_m[:, r, f]).melt()

        line_polt_params = {'x': 'variable', 'y': 'value', 'err_style': 'bars', 'linewidth': 2.5}

        sns.lineplot(data=Cpd_c, color=colors_c[0], **line_polt_params)
        sns.lineplot(data=Cpd_m, color=colors_m[1], **line_polt_params)
        plt.legend(['prop rad control', 'prop rad mutant'])  # (['prop rad', 'prop plan'])
        plt.yticks(size='x-large')
        plt.ylabel(feat[2], size='x-large')
        plt.xlabel('cortical depth (um)', size='x-large')
        plt.xticks(size='x-large')
        plt.xticks(np.arange(0, 10), 25*np.arange(0, np.max(bins), np.max(bins) / 10).astype(int),
                   size='x-large', rotation=20)
        plt.twinx()
        sns.lineplot(data=bp_c, color=colors_c[2], **line_polt_params)
        sns.lineplot(data=bp_m, color=colors_m[3], **line_polt_params)
        plt.title(f'{region_acronym} {feat[0]}', size='x-large')
        plt.legend(['bp control', 'bp mutant'])  # (['bp'])#['control', 'deprived'])
        plt.yticks(size='x-large')
        plt.ylabel(feat[0], size='x-large')
