import ClearMap.Alignment.Annotation as ano

import ClearMap.IO.IO as io
import ClearMap.Visualization.Plot3d as p3d
import graph_tool.inference as gti
import os
print('TEST')
import math
import matplotlib.pyplot as plt
import ClearMap.Analysis.Graphs.GraphGt as ggt
import graph_tool.centrality as gtc
import graph_tool.generation as gtg
print('loading...')
import numpy as np
import numexpr as ne
import graph_tool.topology as gtt
from sklearn import preprocessing
# from ClearMap.Visualization.Vispy.sbm_plot import *
import seaborn as sns
from scipy.stats import ttest_ind
import math
pi=math.pi
import pickle
from sklearn.linear_model import LinearRegression
work_dir='/data_SSD_2to/191122Otof'#'/data_SSD_2to/whiskers_graphs'
graph_nb=['1R','2R','3R','5R','6R','7R', '8R', '4R']#['44R', '30R']#, '39L']
from scipy.stats import wilcoxon
from scipy.stats import ks_2samp
from scipy.stats import brunnermunzel
import pandas as pd
import random
# from random import random


def weighted_sample(population, weights, k):
    """
    This function draws a random sample (without repeats)
    of length k     from the sequence 'population' according
    to the list of weights
    """
    sample = []

    while len(sample) < k:
        choice = random.choices(population, weights, k=1)
        sample.append(choice[0])
        weights[choice]=0

    return np.array(sample)



def modularity_measure(partition, graph, vertex_prop):
    u, c= np.unique(partition, return_counts=True)
    vp=graph.vertex_property(vertex_prop)
    K=graph.n_edges
    # trash_clusters=u[np.where(c<20)]
    Q=0
    Qs=[]
    for e in u:
        vf=np.zeros(graph.n_vertices)
        vf[np.where(vp==e)[0]]=1
        cluster= graph.sub_graph(vertex_filter=vf)
        ms=cluster.n_edges
        ks=np.sum(cluster.vertex_degrees())
        Q=Q+(ms/K)-((ks/(2*K))**2)
        Qs.append((ms/K)-((ks/(2*K))**2))
    print(Q)
    return Q, Qs




work_dir='/data_SSD_2to/whiskers_graphs/new_graphs'
graph_nb = ['138L', '141L', '142L', '158L', '163L', '162L', '164L']

controls=['142L','158L','162L', '164L']
mutants=['138L','141L', '163L', '165L']
import pickle
try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle

with open('/data_SSD_2to/181002_4/reg_list.p', 'rb') as fp:
  reg_list = pickle.load(fp)

with open('/data_SSD_2to/191122Otof/reg_list_full.p', 'rb') as fp:
  reg_list = pickle.load(fp)

with open('/data_SSD_2to/181002_4/atlas_volume_list.p', 'rb') as fp:
  atlas_list = pickle.load(fp)

regions=['Inferior colliculus','lateral lemniscus', 'Superior olivary complex, lateral part', 'Cochlear nuclei']
# regions=[]

controls=['142L','158L','162L', '164L']
mutants=['138L','141L', '163L', '165L']

colors = ['cadetblue', 'indianred', 'darkgoldenrod', 'darkorange', 'royalblue', 'blueviolet', 'forestgreen',
          'lightseagreen']


controls=['2R','3R','5R', '8R']#['2R','3R','5R', '8R']
mutants=['1R','7R', '6R', '4R']
work_dir='/data_SSD_2to/191122Otof'


mutants=['2R','3R','5R', '1R']
controls=['7R','8R', '6R']
work_dir='/data_SSD_1to/otof6months'



# aging pruning experiment
controls=['2R','3R','5R', '8R']
mutants=['7R','8R', '6R']

states=[controls, mutants]
# states=[controls]

ps=7.3
condition= 'Auditory_regions'
# control='8R'
ratios=np.arange(0,50,5)/100#[1.0/7.8]
eff_ratios=[]
ratio_name='0_50'
# param='small_length'#'small_radii'#'big_length'#'big_radii'#multi
compute_flow = False

pressure_driven = False
compute_naive = True
prune_graph = True

compute_non_modified = True

compute_sbm = True
keep_sbm_partition=False

largest_component=True

alphas=[0.01, 0.2, 1.0, 1.5, 10]
# alpha=0.5
if keep_sbm_partition:
    compute_sbm=False

if condition == 'Auditory_regions':
    regions = [[(142, 8), (149, 8), (128, 8), (156, 8)]]
    sub_region = True
elif condition == 'barrel_region':
    regions = [[(54, 9), (47, 9)]]  # , (75, 9)]  # barrels

params=['small_radii','d2s']#['d2s','flow', 'small_length','small_radii','big_length','big_radii']#,multi]

for param in params:
    for alpha in alphas:
        Qmutant=[]
        NbModulesM=[]

        Qcontrol=[]
        NbModulesC=[]

        Qcontrol_mod=[]
        NbModulesC_mod=[]

        Qcontrol_mod_rad=[]
        NbModulesC_mod_rad=[]

        ConnexionDensity=[]
        if prune_graph:
            Qcontrol_NbRMedges=[]
            Qcontrol_mod_NbRMedges=[]

            Qcontrol_NbRMvertices=[]
            Qcontrol_mod_NbRMvertices=[]

        for state in states:
            for control in state:
                if state==controls:
                    work_dir = '/data_SSD_2to/191122Otof'
                else:
                    work_dir = '/data_SSD_1to/otof6months'
                print(control, param, condition, alpha)
                N=1
                graph = ggt.load(work_dir + '/' + control + '/' + 'data_graph_correcteduniverse.gt')  # data_graph_corrected_Isocortex.gt')#/data_graph_correcteduniverse.gt')#
                degrees = graph.vertex_degrees()
                vf = np.logical_and(degrees > 1, degrees <= 4)
                graph = graph.sub_graph(vertex_filter=vf)

                if pressure_driven:
                    try:
                        with open(work_dir + '/' + control + '/sampledict' + control + '.pkl', 'rb') as fp:
                            sampledict = pickle.load(fp)
                    except:
                        f, v = computeFlowFranca_graph(work_dir, graph, control, '')

                    flow = np.asarray(sampledict['flow'][0])
                    e = 1 - np.exp(-(ps / abs(flow)))
                    graph.add_edge_property('extracted_frac', e)
                # diff = np.load(work_dir + '/' + control + '/sbm/' + 'diffusion_penetrating_vessel_overlap_end_point_cluster_per_region_iteration_Isocortex_graph_corrected' + '.npy')
                # graph.add_vertex_property('overlap', diff)

                for region_list in regions:

                    vertex_filter = np.zeros(graph.n_vertices)
                    for i, rl in enumerate(region_list):
                        order, level = region_list[i]
                        print(level, order, ano.find(order, key='order')['name'])
                        label = graph.vertex_annotation();
                        label_leveled = ano.convert_label(label, key='order', value='order', level=level)
                        vertex_filter[label_leveled == order] = 1;
                    # gss4_t = graph.sub_graph(vertex_filter=vertex_filter)

                gss4=graph.copy()
                gss4 = gss4.sub_graph(vertex_filter=vertex_filter)



                if largest_component:
                    gss4=gss4.largest_component()

                conn_density_init=(2*gss4.n_edges)/(gss4.n_vertices*(gss4.n_vertices-1))
                print('conn_density_init: ',conn_density_init)
                n_vert=np.sum(vertex_filter)
                n_edge=gss4.n_edges

                if largest_component:
                    gss4=gss4.largest_component()
                n_vert = np.sum(vertex_filter)
                n_edge = gss4.n_edges
                n_vert =gss4.n_vertices


                if keep_sbm_partition:
                    g = gss4.base
                    state_sbm = gti.minimize_blockmodel_dl(g)
                    modules = state_sbm.get_blocks().a
                    s = np.unique(modules).shape[0]
                    gss4.add_vertex_property('blocks', modules)
                    # Q=get_modularity(gss4, modules)
                    Q, Qs = modularity_measure(modules, gss4, 'blocks')
                    print(s, Q, state_sbm.get_B())
                    N=0

                if state==controls:
                    for ratio in ratios:
                        print(ratio)
                        nbE2rm=(n_edge*ratio) #removing 1 edge removes 2 vrtices becaus ewe keep the degree=3
                        print('nb of edge 2 remove: ', nbE2rm)
                        connectivity=gss4.edge_connectivity()
                        import random

                        if compute_naive:
                            ## NAIVE EDGE SELECTION
                            print('sbm naive control modified')
                            # weight=from_v_prop2_eprop(gss4,vertex_filter)
                            weight = np.ones(gss4.n_edges)
                            e2rm=random.sample(range(gss4.n_edges), k=int(np.round(nbE2rm)))
                            e2rm=np.array(e2rm)
                            print(e2rm.shape)
                            edge_filter=np.ones(gss4.n_edges)
                            # edge_filter=np.logical_not(np.array([e in e2rm for e in range(gss4.n_edges)])).astype(int)
                            edge_filter[e2rm.astype(int)]=0
                            print(edge_filter.shape[0]-np.sum(edge_filter))

                            connectivity = gss4.edge_connectivity()
                            deg2 = np.zeros(gss4.n_vertices)
                            gss4_mod = gss4.sub_graph(edge_filter=edge_filter)
                            if e2rm.shape[0]>=1:
                                deg2[connectivity[e2rm].flatten()] = 1
                                gss4.add_vertex_property('deg2', deg2)
                                gss4_mod = gss4.sub_graph(edge_filter=edge_filter)

                                if prune_graph:
                                    connectivity = gss4_mod.edge_connectivity()
                                    label = gss4_mod.vertex_annotation();
                                    label_leveled = ano.convert_label(label, key='order', value='order', level=level)
                                    audi = label_leveled == order

                                    # deg2 = np.logical_and(np.asarray(gss4_mod.vertex_degrees() == 2), )
                                    deg2 = gss4_mod.vertex_property('deg2').nonzero()[0]
                                    print(np.sum(deg2))
                                    edges = []
                                    for d2 in deg2:
                                        edge = connectivity[np.asarray(
                                            np.logical_or(connectivity[:, 0] == d2, connectivity[:, 1] == d2)).nonzero()[0],
                                               :]
                                        print(edge)
                                        edge = edge[edge != d2]
                                        print(edge)
                                        if edge.shape[0] == 2:
                                            edges.append(edge)
                                    print(gss4_mod, np.sum(gss4_mod.vertex_degrees() == 2))
                                    gss4_mod.remove_vertex(deg2)
                                    print(gss4_mod, np.sum(gss4_mod.vertex_degrees() == 2))
                                    deg0 = np.asarray(gss4_mod.vertex_degrees() == 0).nonzero()[0]
                                    gss4_mod.remove_vertex(deg0)
                                    print(gss4_mod, (gss4.n_edges - gss4_mod.n_edges) / gss4.n_edges)
                            Qcontrol_NbRMedges.append((gss4.n_edges-gss4_mod.n_edges)/gss4.n_edges)
                            Qcontrol_NbRMvertices.append((gss4.n_vertices-gss4_mod.n_vertices)/gss4.n_vertices)

                            if largest_component:
                                gss4_mod = gss4_mod.largest_component()

                            import ClearMap.Analysis.Graphs.GraphProcessing as gp

                            # g = reduce_graph(gss4, edge_geometry=False, verbose=True)
                            print(gss4_mod)


                            print(ratio, gss4, gss4_mod)

                            g = gss4_mod.base
                            mod = []
                            bloc = []

                            if keep_sbm_partition:
                                # if N==1:
                                #
                                #     state_sbm = gti.minimize_blockmodel_dl(g)
                                #     modules = state_sbm.get_blocks().a
                                #     s = np.unique(modules).shape[0]
                                #     dls = []  # description length history
                                #     bs = []  # partitions
                                #
                                #     gss4_mod.add_vertex_property('blocks', modules)
                                #     # gss4.add_vertex_property('indices', indices)
                                #     Q, Qs = modularity_measure(modules, gss4_mod, 'blocks')
                                #     print(s, Q, state_sbm.get_B())
                                #     bloc.append(state_sbm.get_B())
                                #     mod.append(Q)
                                # if N == 0:
                                modules=gss4_mod.vertex_property('blocks')
                                s=np.unique(modules).shape[0]
                                Q, Qs = modularity_measure(modules, gss4_mod, 'blocks')
                                # Q=get_modularity(gss4_mod, gss4_mod.vertex_property('blocks'))
                                print(s, Q, state_sbm.get_B())
                                bloc.append(state_sbm.get_B())
                                mod.append(Q)

                                N = 0
                                Qcontrol_mod.append(np.median(np.array(mod)))
                                # ersmutant.append(M)
                                NbModulesC_mod.append(np.median(np.array(bloc)))


                            if compute_sbm:



                                for i in range(5):
                                    # g.ep.artery, g.ep.vein , "discrete-binomial","discrete-binomial"
                                    # state = gti.minimize_blockmodel_dl(g, state_args=dict(recs=[g.ep.param, g.ep.invL, g.ep.rp],
                                    #                                                                      rec_types=["real-exponential", "real-exponential", "real-exponential"]), overlap = False, deg_corr = True)
                                    # state = gti.minimize_blockmodel_dl(g,  state_args = dict(recs=[g.ep.param, g.ep.invL],
                                    #                   rec_types=["real-exponential", "real-exponential"]), overlap = False, deg_corr = True)
                                    # state = gti.minimize_blockmodel_dl(g, state_args=dict(recs=[g.ep.param],
                                    #                                       rec_types=["real-exponential"]), overlap=False, deg_corr=True)
                                    # state = gti.minimize_blockmodel_dl(g, state_args=dict(recs=[g.ep.flow],
                                    #                                       rec_types=["real-exponential"]), overlap=False, deg_corr=True)
                                    state_sbm = gti.minimize_blockmodel_dl(g)
                                    modules = state_sbm.get_blocks().a
                                    s = np.unique(modules).shape[0]
                                    dls = []  # description length history
                                    bs = []  # partitions

                                    gss4_mod.add_vertex_property('blocks', modules)
                                    # gss4.add_vertex_property('indices', indices)
                                    Q, Qs = modularity_measure(modules, gss4_mod, 'blocks')
                                    # Q = get_modularity(gss4_mod, gss4_mod.vertex_property('blocks'))
                                    print(s, Q, state_sbm.get_B())
                                    bloc.append(state_sbm.get_B())
                                    mod.append(Q)

                                Qcontrol_mod.append(np.median(np.array(mod)))
                                # ersmutant.append(M)
                                NbModulesC_mod.append(np.median(np.array(bloc)))

                            if compute_flow:
                                print("compute flow...")
                                try:
                                    with open(work_dir+'/'+control+'/sampledict'+'naive_modified'+'.pkl', 'rb') as fp:
                                        sampledict = pickle.load(fp)

                                    # f = np.asarray(sampledict['flow'][0])
                                    # v = np.asarray(sampledict['v'][0])
                                    # graph.add_edge_property('flow', f)
                                    # graph.add_edge_property('veloc', v)
                                except:
                                    f, v = computeFlowFranca_graph(work_dir, gss4_mod, control, 'naive_modified')
                                    gss4_mod.save(work_dir + '/' + control + '/' + 'data_graph_correcteduniverse_naive_mod.gt')
                                    gss4_mod.add_edge_property('flow', f)
                                    gss4_mod.add_edge_property('veloc', v)
                                print('done')


                        ## SMART LENGTH BASED EDGE SELECTION
                        print('sbm smart control modified ', param)
                        # weight=from_v_prop2_eprop(gss4,vertex_filter)/(gss4.edge_property('radii')*gss4.edge_property('length'))
                        if param=='small_length':
                            print(param)
                            weight = np.ones(gss4.n_edges) / gss4.edge_property('length')
                            weight = np.nan_to_num(weight)
                            e2rm = np.random.choice(range(gss4.n_edges), int(np.round(nbE2rm)),
                                                    p=weight / np.sum(weight), replace=False)
                        elif param=='small_radii':
                            print(param)
                            weight = np.ones(gss4.n_edges) / gss4.edge_property('radii')
                            weight=np.power(weight, alpha)
                            weight = np.nan_to_num(weight)
                            e2rm = np.random.choice(range(gss4.n_edges), int(np.round(nbE2rm)),
                                                    p=weight / np.sum(weight), replace=False)
                        elif param == 'big_length':
                            print(param)
                            weight = gss4.edge_property('length')
                            weight = np.power(weight, alpha)
                            weight = np.nan_to_num(weight)
                            e2rm = np.random.choice(range(gss4.n_edges), int(np.round(nbE2rm)),
                                                    p=weight / np.sum(weight), replace=False)
                        elif param == 'big_radii':
                            print(param)
                            weight = gss4.edge_property('radii')
                            weight = np.power(weight, alpha)
                            weight = np.nan_to_num(weight)
                            e2rm = np.random.choice(range(gss4.n_edges), int(np.round(nbE2rm)),
                                                    p=weight / np.sum(weight), replace=False)
                        elif param == 'multi':
                            print(param)
                            weight = np.ones(gss4.n_edges)/(gss4.edge_property('radii')*gss4.edge_property('length'))
                            weight = np.nan_to_num(weight)
                            e2rm = np.random.choice(range(gss4.n_edges), int(np.round(nbE2rm)),
                                                    p=weight / np.sum(weight), replace=False)
                        elif param == 'flow':
                            print(param)
                            weight = gss4.edge_property('extracted_frac')
                            weight = np.power(weight, alpha)
                            weight = np.nan_to_num(weight)
                            e2rm = np.random.choice(range(gss4.n_edges), int(np.round(nbE2rm)),
                                                    p=weight / np.sum(weight), replace=False)
                        elif param == 'd2s':
                            print(param)
                            weight = gss4.edge_property('distance_to_surface')
                            # weight[weight<17]=0
                            weight = np.power(weight, alpha)
                            weight = np.nan_to_num(weight)
                            # e2rm = weighted_sample(range(gss4.n_edges), weights=weight, k=int(np.round(nbE2rm)))
                            try:
                                e2rm = np.random.choice(range(gss4.n_edges), int(np.round(nbE2rm)),
                                                        p=weight / np.sum(weight), replace=False)
                            except:
                                weight = gss4.edge_property('distance_to_surface')
                                e2rm = np.random.choice(range(gss4.n_edges), int(np.round(nbE2rm)),
                                                        p=weight / np.sum(weight),
                                                        replace=False)

                        e2rm = np.array(e2rm)
                        print(e2rm.shape)
                        edge_filter = np.ones(gss4.n_edges)
                        # edge_filter=np.logical_not(np.array([e in e2rm for e in range(gss4.n_edges)])).astype(int)
                        edge_filter[e2rm.astype(int)] = 0
                        print(edge_filter.shape[0] - np.sum(edge_filter))
                        # edge_filter = np.ones(gss4.n_edges)
                        # edge_filter[e2rm] = 0
                        # # print(gss4_t)
                        connectivity = gss4.edge_connectivity()
                        deg2 = np.zeros(gss4.n_vertices)
                        deg2[connectivity[e2rm].flatten()] = 1
                        gss4.add_vertex_property('deg2', deg2)
                        gss4_mod = gss4.sub_graph(edge_filter=edge_filter)





                        mod = []
                        bloc = []
                        if e2rm.shape[0] >= 1:
                            if prune_graph:
                                connectivity = gss4_mod.edge_connectivity()
                                label = gss4_mod.vertex_annotation();
                                label_leveled = ano.convert_label(label, key='order', value='order', level=level)
                                audi = label_leveled == order

                                # deg2 = np.logical_and(np.asarray(gss4_mod.vertex_degrees() == 2), )
                                deg2 = gss4_mod.vertex_property('deg2').nonzero()[0]
                                print(np.sum(deg2))
                                edges = []
                                for d2 in deg2:
                                    edge = connectivity[np.asarray(
                                        np.logical_or(connectivity[:, 0] == d2, connectivity[:, 1] == d2)).nonzero()[0],
                                           :]
                                    print(edge)
                                    edge = edge[edge != d2]
                                    print(edge)
                                    if edge.shape[0] == 2:
                                        edges.append(edge)
                                print(gss4_mod, np.sum(gss4_mod.vertex_degrees() == 2))
                                gss4_mod.remove_vertex(deg2)
                                print(gss4_mod, np.sum(gss4_mod.vertex_degrees() == 2))
                                deg0 = np.asarray(gss4_mod.vertex_degrees() == 0).nonzero()[0]
                                gss4_mod.remove_vertex(deg0)
                                print(gss4_mod, (gss4.n_edges - gss4_mod.n_edges) / gss4.n_edges)
                        Qcontrol_mod_NbRMedges.append((gss4.n_edges - gss4_mod.n_edges) / gss4.n_edges)
                        Qcontrol_mod_NbRMvertices.append((gss4.n_vertices - gss4_mod.n_vertices) / gss4.n_vertices)

                        if largest_component:
                            gss4_mod = gss4_mod.largest_component()

                        print(gss4)
                        conn_density = (2 * gss4_mod.n_edges) / (gss4_mod.n_vertices * (gss4_mod.n_vertices - 1))
                        print('conn_density: ', conn_density)
                        ConnexionDensity.append(conn_density)


                        g = gss4_mod.base

                        import ClearMap.Analysis.Graphs.GraphProcessing as gp

                        # g = reduce_graph(gss4, edge_geometry=False, verbose=True)
                        print(gss4_mod)

                        if keep_sbm_partition:
                            # if N == 1:
                            #     state_sbm = gti.minimize_blockmodel_dl(g)
                            #     modules = state_sbm.get_blocks().a
                            #     s = np.unique(modules).shape[0]
                            #     dls = []  # description length history
                            #     bs = []  # partitions
                            #
                            #     gss4_mod.add_vertex_property('blocks', modules)
                            #     # gss4.add_vertex_property('indices', indices)
                            #     Q, Qs = modularity_measure(modules, gss4_mod, 'blocks')
                            #     print(s, Q, state_sbm.get_B())
                            #     bloc.append(state_sbm.get_B())
                            #     mod.append(Q)
                            # if N == 0:
                            modules = gss4_mod.vertex_property('blocks')
                            s = np.unique(modules).shape[0]
                            Q, Qs = modularity_measure(modules, gss4_mod, 'blocks')
                            # Q=get_modularity(gss4_mod, gss4_mod.vertex_property('blocks'))
                            print(s, Q, state_sbm.get_B())
                            bloc.append(state_sbm.get_B())
                            mod.append(Q)

                            N = 0
                            Qcontrol_mod_rad.append(np.median(np.array(mod)))
                            # ersmutant.append(M)
                            NbModulesC_mod_rad.append(np.median(np.array(bloc)))

                        if compute_sbm:


                            for i in range(5):
                                # g.ep.artery, g.ep.vein , "discrete-binomial","discrete-binomial"
                                # state = gti.minimize_blockmodel_dl(g, state_args=dict(recs=[g.ep.param, g.ep.invL, g.ep.rp],
                                #                                                                      rec_types=["real-exponential", "real-exponential", "real-exponential"]), overlap = False, deg_corr = True)
                                # state = gti.minimize_blockmodel_dl(g,  state_args = dict(recs=[g.ep.param, g.ep.invL],
                                #                   rec_types=["real-exponential", "real-exponential"]), overlap = False, deg_corr = True)
                                # state = gti.minimize_blockmodel_dl(g, state_args=dict(recs=[g.ep.param],
                                #                                       rec_types=["real-exponential"]), overlap=False, deg_corr=True)
                                # state = gti.minimize_blockmodel_dl(g, state_args=dict(recs=[g.ep.flow],
                                #                                       rec_types=["real-exponential"]), overlap=False, deg_corr=True)
                                state_sbm = gti.minimize_blockmodel_dl(g)
                                modules = state_sbm.get_blocks().a
                                s = np.unique(modules).shape[0]
                                dls = []  # description length history
                                bs = []  # partitions

                                gss4_mod.add_vertex_property('blocks_sbm', modules)
                                # gss4.add_vertex_property('indices', indices)
                                Q, Qs = modularity_measure(modules, gss4_mod, 'blocks_sbm')
                                # Q = get_modularity(gss4_mod, gss4_mod.vertex_property('blocks'))
                                print(s, Q, state_sbm.get_B())
                                bloc.append(state_sbm.get_B())
                                mod.append(Q)

                            Qcontrol_mod_rad.append(np.median(np.array(mod)))
                            # ersmutant.append(M)
                            NbModulesC_mod_rad.append(np.median(np.array(bloc)))

                        if compute_flow:
                            print("compute flow...")
                            try:
                                with open(work_dir+'/'+control+'/sampledict'+'smart_modified'+'.pkl', 'rb') as fp:
                                    sampledict = pickle.load(fp)

                                # f = np.asarray(sampledict['flow'][0])
                                # v = np.asarray(sampledict['v'][0])
                                # graph.add_edge_property('flow', f)
                                # graph.add_edge_property('veloc', v)
                            except:
                                f, v = computeFlowFranca_graph(work_dir, gss4_mod, control, 'smart_modified')
                                gss4_mod.save(work_dir + '/' + control + '/' + 'data_graph_correcteduniverse_smart_mod.gt')
                                # graph.add_edge_property('flow', f)
                                # graph.add_edge_property('veloc', v)
                            print('done')


                import ClearMap.Analysis.Graphs.GraphProcessing as gp
                # g = reduce_graph(gss4, edge_geometry=False, verbose=True)
                print(gss4)
                if compute_non_modified:
                    if compute_flow:
                        print("compute flow...")
                        try:
                            with open(work_dir + '/' + control + '/sampledict' + control + '.pkl', 'rb') as fp:
                                sampledict = pickle.load(fp)


                        except:
                            f, v = computeFlowFranca_graph(work_dir, gss4_mod, control, 'normal')

                        print('done')

                    if keep_sbm_partition:
                        print(N)
                        if N == 1:
                            state_sbm = gti.minimize_blockmodel_dl(g)
                            modules = state_sbm.get_blocks().a
                            s = np.unique(modules).shape[0]
                            dls = []  # description length history
                            bs = []  # partitions

                            gss4_mod.add_vertex_property('blocks', modules)
                            # gss4.add_vertex_property('indices', indices)
                            Q, Qs = modularity_measure(modules, gss4_mod, 'blocks')
                            print(s, Q, state_sbm.get_B())
                            bloc.append(state_sbm.get_B())
                            mod.append(Q)
                        if N == 0:
                            modules = gss4_mod.vertex_property('blocks')
                            Q, Qs = modularity_measure(modules, gss4_mod, 'blocks')
                            print(s, Q, state_sbm.get_B())
                            bloc.append(state_sbm.get_B())
                            mod.append(Q)

                        N = 0
                        Qcontrol_mod.append(np.median(np.array(mod)))
                        # ersmutant.append(M)
                        NbModulesC_mod.append(np.median(np.array(bloc)))

                    if compute_sbm:
                        g = gss4.base

                        print('sbm')
                        mod = []
                        bloc = []

                        for i in range(5):
                        #g.ep.artery, g.ep.vein , "discrete-binomial","discrete-binomial"
                            # state = gti.minimize_blockmodel_dl(g, state_args=dict(recs=[g.ep.param, g.ep.invL, g.ep.rp],
                            #                                                                      rec_types=["real-exponential", "real-exponential", "real-exponential"]), overlap = False, deg_corr = True)
                            # state = gti.minimize_blockmodel_dl(g,  state_args = dict(recs=[g.ep.param, g.ep.invL],
                            #                   rec_types=["real-exponential", "real-exponential"]), overlap = False, deg_corr = True)
                            # state = gti.minimize_blockmodel_dl(g, state_args=dict(recs=[g.ep.param],
                            #                                       rec_types=["real-exponential"]), overlap=False, deg_corr=True)
                            # state = gti.minimize_blockmodel_dl(g, state_args=dict(recs=[g.ep.flow],
                            #                                       rec_types=["real-exponential"]), overlap=False, deg_corr=True)
                            state_sbm = gti.minimize_blockmodel_dl(g)
                            modules=state_sbm.get_blocks().a
                            s = np.unique(modules).shape[0]
                            dls = []  # description length history
                            bs = []  # partitions

                            gss4.add_vertex_property('blocks', modules)
                            # gss4.add_vertex_property('indices', indices)
                            Q, Qs = modularity_measure(modules, gss4, 'blocks')
                            # Q = get_modularity(gss4_mod, gss4_mod.vertex_property('blocks'))
                            print(s, Q, state_sbm.get_B())
                            bloc.append(state_sbm.get_B())
                            mod.append(Q)

                        if state==mutants:
                            Qmutant.append(np.median(np.array(mod)))
                            # ersmutant.append(M)
                            NbModulesM.append(np.median(np.array(bloc)))
                        elif state==controls:
                            Qcontrol.append(np.median(np.array(mod)))
                            # ersmutant.append(M)
                            NbModulesC.append(np.median(np.array(bloc)))



        ## saving
        work_dir='/data_SSD_1to/aging'
        if compute_sbm:
            np.save(work_dir + '/' + 'Qcontrol_ratios' + ratio_name + param +'_prun_'+str(prune_graph)+'_lc_'+str(largest_component)+'_alpha_'+str(alpha) + '_gx_sbm.npy', Qcontrol)
            np.save(work_dir + '/' + 'Qcontrol_mod_ratios' + ratio_name + param+'_prun_'+str(prune_graph)+'_lc_'+str(largest_component)+'_alpha_'+str(alpha) + '_gx_sbm.npy', Qcontrol_mod)
            # np.save(work_dir + '/' +'Qmutant_ratios.npy', Qmutant)
            np.save(work_dir + '/' + 'Qcontrol_mod_rad_ratios' + ratio_name + param+'_prun_'+str(prune_graph)+'_lc_'+str(largest_component)+'_alpha_'+str(alpha) + '_gx_sbm.npy', Qcontrol_mod_rad)

            np.save(work_dir + '/' + 'NbModulesC_ratios' + ratio_name + param +'_prun_'+str(prune_graph)+'_lc_'+str(largest_component)+'_alpha_'+str(alpha)+ '_gx_sbm.npy', NbModulesC)
            np.save(work_dir + '/' + 'NbModulesC_mod_ratios' + ratio_name + param +'_prun_'+str(prune_graph)+'_lc_'+str(largest_component)+'_alpha_'+str(alpha)+ '_gx_sbm.npy', NbModulesC_mod)
            # np.save(work_dir + '/' +'NbModulesM_ratios.npy', NbModulesM)
            np.save(work_dir + '/' + 'NbModulesC_mod_rad_ratios' + ratio_name + param +'_prun_'+str(prune_graph)+'_lc_'+str(largest_component)+'_alpha_'+str(alpha)+ '_gx_sbm.npy', NbModulesC_mod_rad)
            np.save(work_dir + '/' + 'ConnexionDensity_ratios' + ratio_name + param +'_prun_'+str(prune_graph)+'_lc_'+str(largest_component)+'_alpha_'+str(alpha)+ '_gx_sbm.npy',
                    np.array(ConnexionDensity))

            if prune_graph:
                np.save(work_dir + '/' + 'effectivelyRMedges_randomRM' + ratio_name + param + '_prun_' + str(
                    prune_graph) + '_lc_'+str(largest_component)+'_alpha_'+str(alpha)+'_gx_sbm.npy', Qcontrol_NbRMedges)
                np.save(work_dir + '/' + 'effectivelyRMedges_weightedRM' + ratio_name + param + '_prun_' + str(
                    prune_graph) +'_lc_'+str(largest_component)+'_alpha_'+str(alpha)+ '_gx_sbm.npy', Qcontrol_mod_NbRMedges)

                np.save(work_dir + '/' + 'effectivelyRMvertices_randomRM' + ratio_name + param + '_prun_' + str(
                    prune_graph) + '_lc_' + str(largest_component) + '_alpha_' + str(alpha) + '_gx_sbm.npy',
                        Qcontrol_NbRMvertices)
                np.save(work_dir + '/' + 'effectivelyRMvertices_weightedRM' + ratio_name + param + '_prun_' + str(
                    prune_graph) + '_lc_' + str(largest_component) + '_alpha_' + str(alpha) + '_gx_sbm.npy',
                        Qcontrol_mod_NbRMvertices)
        else:
            np.save(work_dir + '/' +'Qcontrol_ratios'+ratio_name+param+'_prun_'+str(prune_graph)+'_lc_'+str(largest_component)+'_alpha_'+str(alpha)+'_gx.npy', Qcontrol)
            np.save(work_dir + '/' +'Qcontrol_mod_ratios'+ratio_name+param+'_prun_'+str(prune_graph)+'_lc_'+str(largest_component)+'_alpha_'+str(alpha)+'_gx.npy', Qcontrol_mod)
            # np.save(work_dir + '/' +'Qmutant_ratios.npy', Qmutant)
            np.save(work_dir + '/' +'Qcontrol_mod_rad_ratios'+ratio_name+param+'_prun_'+str(prune_graph)+'_lc_'+str(largest_component)+'_alpha_'+str(alpha)+'_gx.npy', Qcontrol_mod_rad)

            np.save(work_dir + '/' +'NbModulesC_ratios'+ratio_name+param+'_prun_'+str(prune_graph)+'_lc_'+str(largest_component)+'_alpha_'+str(alpha)+'_gx.npy', NbModulesC)
            np.save(work_dir + '/' +'NbModulesC_mod_ratios'+ratio_name+param+'_prun_'+str(prune_graph)+'_lc_'+str(largest_component)+'_alpha_'+str(alpha)+'_gx.npy', NbModulesC_mod)
            # np.save(work_dir + '/' +'NbModulesM_ratios.npy', NbModulesM)
            np.save(work_dir + '/' +'NbModulesC_mod_rad_ratios'+ratio_name+param+'_prun_'+str(prune_graph)+'_lc_'+str(largest_component)+'_alpha_'+str(alpha)+'_gx.npy', NbModulesC_mod_rad)
            np.save(work_dir + '/' +'ConnexionDensity_ratios'+ratio_name+param+'_prun_'+str(prune_graph)+'_lc_'+str(largest_component)+'_alpha_'+str(alpha)+'_gx.npy',np.array(ConnexionDensity))
            if prune_graph:
                np.save(work_dir + '/' + 'effectivelyRMedges_randomRM' + ratio_name + param + '_prun_' + str(
                    prune_graph) + '_lc_'+str(largest_component)+'_alpha_'+str(alpha)+'_gx.npy', Qcontrol_NbRMedges)
                np.save(work_dir + '/' + 'effectivelyRMedges_weightedRM' + ratio_name + param + '_prun_' + str(
                    prune_graph) +'_lc_'+str(largest_component)+'_alpha_'+str(alpha)+ '_gx.npy', Qcontrol_mod_NbRMedges)

    ## loading
    # controls=['2R','3R','5R', '8R']
    # mutants=['1R','7R', '6R', '4R']
    # work_dir='/data_SSD_2to/191122Otof'
    # condition = 'Auditory_regions'
    # ratio_name = '0_100'
    # params = ['small_length','small_radii','big_length','big_radii','multi']#,multi]
    # param = params[0]
    # compute_sbm=False
controls=np.array(controls)
mutants=np.array(mutants)
params=['small_radii','d2s']#, 'small_length','small_radii','big_length','big_radii','d2s',]
param='small_radii'
alpha=10
alphas=[1.0]
prune_graph=True
compute_sbm=True
work_dir='/data_SSD_2to/191122Otof'
ConnexionDensitymut=np.load(work_dir + '/' + 'MutantConnexionDensity_ratios0_50small_radii_prun_True_lc_True_alpha_0.5' + '_gx_sbm.npy')
# work_dir='/data_SSD_2to/191122Otof/test'
work_dir='/data_SSD_1to/aging'#'/data_SSD_1to/otof6months/rewiring'
for alpha in alphas:
    if compute_sbm:
        Qcontrol = np.load(work_dir + '/' + 'Qcontrol_ratios' + ratio_name + param +'_prun_'+str(prune_graph)+'_lc_'+str(largest_component)+'_alpha_'+str(alpha)+ '_gx_sbm.npy')
        Qcontrol_mod = np.load(work_dir + '/' + 'Qcontrol_mod_ratios' + ratio_name + param +'_prun_'+str(prune_graph)+'_lc_'+str(largest_component)+'_alpha_'+str(alpha)+ '_gx_sbm.npy')
        # Qmutant=np.save(work_dir + '/' +'Qmutant_ratios.npy')
        Qcontrol_mod_rad = np.load(work_dir + '/' + 'Qcontrol_mod_rad_ratios' + ratio_name + param +'_prun_'+str(prune_graph)+'_lc_'+str(largest_component)+'_alpha_'+str(alpha)+ '_gx_sbm.npy')

        NbModulesC = np.load(work_dir + '/' + 'NbModulesC_ratios' + ratio_name + param +'_prun_'+str(prune_graph)+'_lc_'+str(largest_component)+'_alpha_'+str(alpha)+ '_gx_sbm.npy')
        NbModulesC_mod = np.load(work_dir + '/' + 'NbModulesC_mod_ratios' + ratio_name + param +'_prun_'+str(prune_graph)+'_lc_'+str(largest_component)+'_alpha_'+str(alpha)+  '_gx_sbm.npy')
        # NbModulesM=np.save(work_dir + '/' +'NbModulesM_ratios.npy')
        NbModulesC_mod_rad = np.load(work_dir + '/' + 'NbModulesC_mod_rad_ratios' + ratio_name + param +'_prun_'+str(prune_graph)+'_lc_'+str(largest_component)+'_alpha_'+str(alpha)+  '_gx_sbm.npy')
        ConnexionDensity = np.load(work_dir + '/' + 'ConnexionDensity_ratios' + ratio_name + param +'_prun_'+str(prune_graph)+'_lc_'+str(largest_component)+'_alpha_'+str(alpha)+ '_gx_sbm.npy')
        if prune_graph:
            Qcontrol_NbRMedges=np.load(work_dir + '/' + 'effectivelyRMedges_randomRM' + ratio_name + param + '_prun_' + str(
                prune_graph) + '_lc_'+str(largest_component)+'_alpha_'+str(alpha)+ '_gx_sbm.npy')
            Qcontrol_mod_NbRMedges=np.load(work_dir + '/' + 'effectivelyRMedges_weightedRM' + ratio_name + param + '_prun_' + str(
                prune_graph) +'_lc_'+str(largest_component)+'_alpha_'+str(alpha)+ '_gx_sbm.npy')

        prune_graph = True
        alpha=1.0
        # work_dir = '/data_SSD_1to/otof6months'
        Qcontrol_np = np.load(
            work_dir + '/' + 'Qcontrol_ratios' + ratio_name + param + '_prun_' + str(prune_graph) + '_lc_' + str(
                largest_component) + '_alpha_' + str(alpha) + '_gx_sbm.npy')
        Qcontrol_mod_np= np.load(
            work_dir + '/' + 'Qcontrol_mod_ratios' + ratio_name + param + '_prun_' + str(prune_graph) + '_lc_' + str(
                largest_component) + '_alpha_' + str(alpha) + '_gx_sbm.npy')
        # Qmutant=np.save(work_dir + '/' +'Qmutant_ratios.npy')
        Qcontrol_mod_rad_np = np.load(work_dir + '/' + 'Qcontrol_mod_rad_ratios' + ratio_name + param + '_prun_' + str(
            prune_graph) + '_lc_' + str(largest_component) + '_alpha_' + str(alpha) + '_gx_sbm.npy')

        NbModulesC_np = np.load(
            work_dir + '/' + 'NbModulesC_ratios' + ratio_name + param + '_prun_' + str(prune_graph) + '_lc_' + str(
                largest_component) + '_alpha_' + str(alpha) + '_gx_sbm.npy')
        NbModulesC_mod_np = np.load(
            work_dir + '/' + 'NbModulesC_mod_ratios' + ratio_name + param + '_prun_' + str(prune_graph) + '_lc_' + str(
                largest_component) + '_alpha_' + str(alpha) + '_gx_sbm.npy')
        # NbModulesM=np.save(work_dir + '/' +'NbModulesM_ratios.npy')
        NbModulesC_mod_rad_np = np.load(work_dir + '/' + 'NbModulesC_mod_rad_ratios' + ratio_name + param + '_prun_' + str(
            prune_graph) + '_lc_' + str(largest_component) + '_alpha_' + str(alpha) + '_gx_sbm.npy')

    else:
        Qcontrol=np.load(work_dir + '/' + 'Qcontrol_ratios' + ratio_name + param +'_prun_'+str(prune_graph)+ '_lc_'+str(largest_component)+'_alpha_'+str(alpha)+ '_gx.npy')
        Qcontrol_mod=np.load(work_dir + '/' + 'Qcontrol_mod_ratios' + ratio_name + param +'_prun_'+str(prune_graph)+'_lc_'+str(largest_component)+'_alpha_'+str(alpha)+  '_gx.npy')
        # Qmutant=np.save(work_dir + '/' +'Qmutant_ratios.npy')
        Qcontrol_mod_rad=np.load(work_dir + '/' + 'Qcontrol_mod_rad_ratios' + ratio_name + param +'_prun_'+str(prune_graph)+ '_lc_'+str(largest_component)+'_alpha_'+str(alpha)+ '_gx.npy')

        NbModulesC= np.load(work_dir + '/' + 'NbModulesC_ratios' + ratio_name + param +'_prun_'+str(prune_graph)+'_lc_'+str(largest_component)+'_alpha_'+str(alpha)+  '_gx.npy')
        NbModulesC_mod=np.load(work_dir + '/' + 'NbModulesC_mod_ratios' + ratio_name + param +'_prun_'+str(prune_graph)+'_lc_'+str(largest_component)+ '_alpha_'+str(alpha)+ '_gx.npy')
        # NbModulesM=np.save(work_dir + '/' +'NbModulesM_ratios.npy')
        NbModulesC_mod_rad=np.load(work_dir + '/' + 'NbModulesC_mod_rad_ratios' + ratio_name + param +'_prun_'+str(prune_graph)+'_lc_'+str(largest_component)+'_alpha_'+str(alpha)+  '_gx.npy' )
        ConnexionDensity=np.load(work_dir + '/' + 'ConnexionDensity_ratios' + ratio_name + param +'_prun_'+str(prune_graph)+'_lc_'+str(largest_component)+'_alpha_'+str(alpha)+  '_gx.npy')
        if prune_graph:
            Qcontrol_NbRMedges=np.load(work_dir + '/' + 'effectivelyRMedges_randomRM' + ratio_name + param + '_prun_' + str(
                prune_graph) + '_lc_'+str(largest_component)+'_alpha_'+str(alpha)+ '_gx.npy')
            Qcontrol_mod_NbRMedges=np.load(work_dir + '/' + 'effectivelyRMedges_weightedRM' + ratio_name + param + '_prun_' + str(
                prune_graph) +'_lc_'+str(largest_component)+'_alpha_'+str(alpha)+  '_gx.npy')


    ## ratio
    prune_graph = True

    nb_c = np.array(NbModulesC_mod).reshape(controls.shape[0], ratios.shape[0])  # .transpose()
    mod_c = np.array(Qcontrol_mod).reshape(controls.shape[0], ratios.shape[0])  # .transpose()

    nb_c_smart = np.array(NbModulesC_mod_rad).reshape(controls.shape[0], ratios.shape[0])  # .transpose()
    mod_c_smart = np.array(Qcontrol_mod_rad).reshape(controls.shape[0], ratios.shape[0])  # .transpose()

    if prune_graph:
        mod_nb_edge_c = np.array(Qcontrol_mod_NbRMedges).reshape(controls.shape[0], ratios.shape[0])  # .transpose()
        nb_edge_c = np.array(Qcontrol_NbRMedges).reshape(controls.shape[0], ratios.shape[0])  # .transpose()
        Q_C_edges = pd.DataFrame(nb_edge_c).melt()
        mod_Q_C_edges = pd.DataFrame(mod_nb_edge_c).melt()

    work_dir = '/data_SSD_2to/191122Otof'
    nb_m = np.load(work_dir + '/' + 'NbModulesM.npy')
    mod_m = np.load(work_dir + '/' + 'Qmutant.npy')

    work_dir = '/data_SSD_1to/otof6months/rewiring'
    import pandas as pd

    Q_C = pd.DataFrame(mod_c).melt()
    Q_C_smart = pd.DataFrame(mod_c_smart).melt()

    Nb_C = pd.DataFrame(nb_c).melt()
    Mb_C_smart = pd.DataFrame(nb_c_smart).melt()

    Nb_M = [nb_m.tolist(), nb_m.tolist(), nb_m.tolist(), nb_m.tolist(), nb_m.tolist(), nb_m.tolist(), nb_m.tolist(),
            nb_m.tolist(), nb_m.tolist(), nb_m.tolist(), nb_m.tolist(), nb_m.tolist(), nb_m.tolist(), nb_m.tolist(),
            nb_m.tolist(), nb_m.tolist(), nb_m.tolist(), nb_m.tolist(), nb_m.tolist(), nb_m.tolist()]
    Nb_M = np.array(Nb_M).transpose()
    Nb_M = pd.DataFrame(Nb_M).melt()

    Q_M = [mod_m.tolist(), mod_m.tolist(), mod_m.tolist(), mod_m.tolist(), mod_m.tolist(), mod_m.tolist(),
           mod_m.tolist(), mod_m.tolist(), mod_m.tolist(), mod_m.tolist(), mod_m.tolist(), mod_m.tolist(),
           mod_m.tolist(), mod_m.tolist(), mod_m.tolist(), mod_m.tolist(), mod_m.tolist(), mod_m.tolist(), mod_m.tolist(), mod_m.tolist()]
    Q_M = np.array(Q_M).transpose()
    Q_M = pd.DataFrame(Q_M).melt()

    CD_C=np.array(ConnexionDensity).reshape(controls.shape[0], ratios.shape[0])  #
    CD_C=pd.DataFrame(CD_C).melt()

    ## mom pruning

    nb_c_np = np.array(NbModulesC_mod_np).reshape(controls.shape[0], ratios.shape[0])  # .transpose()
    mod_c_np = np.array(Qcontrol_mod_np).reshape(controls.shape[0], ratios.shape[0])  # .transpose()

    nb_c_smart_np = np.array(NbModulesC_mod_rad_np).reshape(controls.shape[0], ratios.shape[0])  # .transpose()
    mod_c_smart_np = np.array(Qcontrol_mod_rad_np).reshape(controls.shape[0], ratios.shape[0])  # .transpose()

    work_dir = '/data_SSD_2to/191122Otof'
    nb_m_np = np.load(work_dir + '/' + 'NbModulesM.npy')
    mod_m_np = np.load(work_dir + '/' + 'Qmutant.npy')
    work_dir = '/data_SSD_1to/otof6months/rewiring'

    import pandas as pd

    Q_C_np = pd.DataFrame(mod_c_np).melt()
    Q_C_smart_np = pd.DataFrame(mod_c_smart_np).melt()

    Nb_C_np = pd.DataFrame(nb_c_np).melt()
    Mb_C_smart_np = pd.DataFrame(nb_c_smart_np).melt()

    Nb_M_np = [nb_m_np.tolist(), nb_m_np.tolist(), nb_m_np.tolist(), nb_m_np.tolist(), nb_m_np.tolist(), nb_m_np.tolist(), nb_m_np.tolist(),
            nb_m_np.tolist(), nb_m_np.tolist(), nb_m_np.tolist(), nb_m_np.tolist(), nb_m_np.tolist(), nb_m_np.tolist(), nb_m_np.tolist(),
            nb_m_np.tolist(), nb_m_np.tolist(), nb_m_np.tolist(), nb_m_np.tolist(), nb_m_np.tolist(), nb_m_np.tolist()]
    Nb_M_np = np.array(Nb_M_np).transpose()
    Nb_M_np = pd.DataFrame(Nb_M_np).melt()

    Q_M_np = [mod_m_np.tolist(), mod_m_np.tolist(), mod_m_np.tolist(), mod_m_np.tolist(), mod_m_np.tolist(), mod_m_np.tolist(),
           mod_m_np.tolist(), mod_m_np.tolist(), mod_m_np.tolist(), mod_m_np.tolist(), mod_m_np.tolist(), mod_m_np.tolist(),
           mod_m_np.tolist(), mod_m_np.tolist(), mod_m_np.tolist(), mod_m_np.tolist(), mod_m_np.tolist(), mod_m_np.tolist(), mod_m_np.tolist(), mod_m_np.tolist()]
    Q_M_np = np.array(Q_M_np).transpose()
    Q_M_np = pd.DataFrame(Q_M_np).melt()

    CD_C=np.array(ConnexionDensity).reshape(controls.shape[0], ratios.shape[0])  #
    CD_C=pd.DataFrame(CD_C).melt()

    ###########

    plt.figure()
    sns.set_style('white')
    sns.lineplot(x="variable", y="value", err_style='bars', data=CD_C, color='darkorange')
    plt.hlines(np.mean(ConnexionDensitymut), 0, 10, colors='k', linestyles='solid')
    plt.hlines(np.mean(ConnexionDensitymut)+np.std(ConnexionDensitymut), 0, 10, colors='k', linestyles='dotted', label='')
    plt.hlines(np.mean(ConnexionDensitymut)-+np.std(ConnexionDensitymut), 0, 10, colors='k', linestyles='dotted', label='')
    plt.title('connexion density '+param, size='x-large')
    plt.xticks(np.arange(ratios.shape[0]), ratios, size='x-large')
    plt.xlabel('% removed edges', size='x-large')
    plt.ylabel('connexion density', size='x-large')
    sns.despine()
    plt.tight_layout()




    # plt.figure()
    # sns.set_style('white')
    # sns.lineplot(x="variable", y="value", err_style='bars', data=Nb_C, color='darkorange')
    # sns.lineplot(x="variable", y="value", err_style='bars', data=Mb_C_smart, color='indianred')
    # sns.lineplot(x="variable", y="value", data=Nb_M, color='cadetblue')
    #
    # plt.title('Nb modules '+param,size='x-large')
    # plt.xticks(np.arange(ratios.shape[0]), ratios,size='x-large')
    # plt.xlabel('% removed edges', size='x-large')
    # plt.ylabel('nb modules', size='x-large')
    # sns.despine()
    # plt.legend(['random removal', 'radii-length driven removal', 'mutant baseline'])
    # plt.tight_layout()

    if prune_graph:
        # nb_edge_c=np.insert(nb_edge_c.T, 0, np.zeros(4), axis=0).T
        # mod_nb_edge_c=np.insert(mod_nb_edge_c.T, 0, np.zeros(4), axis=0).T
        from scipy.interpolate import interp1d

        # x= np.mean(np.insert(nb_edge_c.T, 0, np.zeros(4), axis=0).T, axis=0)
        x = np.mean(nb_edge_c, axis=0)#*(40704/32355)
        y = np.mean(nb_c, axis=0)

        f2 = interp1d(x, y, kind='cubic')

        x2 = np.mean(np.insert(mod_nb_edge_c.T, 0, np.zeros(controls.shape[0]), axis=0).T, axis=0)
        # nb_c_smart=
        x2 = np.mean(mod_nb_edge_c, axis=0)
        y2 = np.mean(nb_c_smart, axis=0)

        f2_mod = interp1d(x2, y2, kind='cubic')

        plt.figure()
        sns.set_style('white')
        Nb_M2plot = Nb_M[Nb_M['variable'] <= len(ratios) + 5]
        Nb_M2plot['variable'] = Nb_M2plot['variable'] * 5 / 100
        sns.lineplot(x="variable", y="value", data=Nb_M2plot, color='cadetblue')
        # plt.scatter(nb_edge_c.flatten(), nb_c.flatten(), color='darkorange')
        plt.scatter(mod_nb_edge_c.flatten(), nb_c_smart.flatten(), color='darkorange')
        Nb_C_np=Nb_C_np.replace({'variable': {0: ratios[0], 1: ratios[1], 2: ratios[2], 3: ratios[3], 4: ratios[4], 5: ratios[5], 6: ratios[6], 7: ratios[7], 8: ratios[8], 9: ratios[9]}})
        sns.lineplot(x="variable", y="value", err_style='bars', data=Nb_C_np, color='forestgreen')
        # plt.plot(x, f2(x), '--', color='darkorange' )
        plt.plot(x2, f2_mod(x2), '--', color='indianred')
        # modules = pd.DataFrame(np.stack([nb_edge_c.flatten(),nb_c.flatten()], axis=1)).melt()
        # modules_mod = pd.DataFrame(np.stack([mod_nb_edge_c.flatten(), nb_c.flatten()])).melt()
        # sns.lmplot(x="variable", y="value", data=modules,
        #            order=2, ci=None, scatter_kws={"s": 80}, palette='darkorange');
        # sns.lmplot(x="variable", y="value", data=modules_mod,
        #             order = 2, ci = None, scatter_kws = {"s": 80}, palette = 'indianred');
        plt.title('Nb modules ' + param, size='x-large')
        # plt.xticks(np.arange(ratios.shape[0]), ratios, size='x-large')
        plt.xlabel('% removed edges', size='x-large')
        plt.ylabel('nb modules', size='x-large')
        sns.despine()
        plt.legend(['mutant baseline', 'random removal no pruning','random removal pruning'])#, 'radii-length driven removal'])
        plt.tight_layout()


        plt.figure()
        x = np.mean(nb_edge_c, axis=0)
        y = np.mean(mod_c, axis=0)

        f2 = interp1d(x, y, kind='cubic')

        x2 = np.mean(mod_nb_edge_c, axis=0)
        y2 = np.mean(mod_c_smart, axis=0)

        f2_mod = interp1d(x2, y2, kind='cubic')

        sns.set_style('white')
        Q_M2plot=Q_M[Q_M['variable'] <= len(ratios)+5]
        Q_M2plot['variable']=Q_M2plot['variable']*5/100
        # plt.scatter(nb_edge_c.flatten(), mod_c.flatten(), color='darkorange')
        plt.scatter(mod_nb_edge_c.flatten(), mod_c_smart.flatten(), color='darkorange')
        sns.lineplot(x="variable", y="value", data=Q_M2plot, color='cadetblue')
        Q_C_np=Q_C_np.replace({'variable': {0: ratios[0], 1: ratios[1], 2: ratios[2], 3: ratios[3], 4: ratios[4], 5: ratios[5], 6: ratios[6], 7: ratios[7], 8: ratios[8], 9: ratios[9]}})
        sns.lineplot(x="variable", y="value", err_style='bars', data=Q_C_np, color='forestgreen')
        # plt.plot(x, f2(x), '--', color='darkorange')
        plt.plot(x2, f2_mod(x2), '--', color='indianred')
        plt.title('modularity ' + param+ ' with pruning '+'alpha '+str(alpha), size='x-large')
        # plt.xticks(np.arange(ratios.shape[0]), ratios, size='x-large')
        plt.xlabel('% removed edges', size='x-large')
        plt.ylabel('modularity', size='x-large')
        sns.despine()
        plt.legend(['mutant baseline', 'random removal no pruning','random removal pruning'])#, 'radii-length driven removal'])
        plt.tight_layout()
        # plt.xticks(np.arange(np.max(ratios)/100, 10), ratios, size='x-large')
    else:

        plt.figure()
        sns.set_style('white')
        sns.lineplot(x="variable", y="value", err_style='bars', data=Nb_C, color='darkorange')
        sns.lineplot(x="variable", y="value", err_style='bars', data=Mb_C_smart, color='indianred')
        sns.lineplot(x="variable", y="value", data=Nb_M, color='cadetblue')

        plt.title('Nb modules '+param+ ' alpha '+str(alpha), size='x-large')
        plt.xticks(np.arange(ratios.shape[0]), ratios,size='x-large')
        plt.xlabel('% removed edges', size='x-large')
        plt.ylabel('nb modules', size='x-large')
        sns.despine()
        plt.legend(['random removal', 'radii-length driven removal', 'mutant baseline'])
        plt.tight_layout()


        plt.figure()
        sns.set_style('white')
        sns.lineplot(x="variable", y="value", err_style='bars', data=Q_C, color='darkorange')
        sns.lineplot(x="variable", y="value", err_style='bars', data=Q_C_smart, color='indianred')
        sns.lineplot(x="variable", y="value", data=Q_M, color='cadetblue')

        plt.title('modularity ' + param + ' alpha '+str(alpha), size='x-large')
        plt.xticks(np.arange(ratios.shape[0]), ratios, size='x-large')
        plt.xlabel('% removed edges', size='x-large')
        plt.ylabel('modularity', size='x-large')
        sns.despine()
        plt.legend(['random removal', 'radii-length driven removal', 'mutant baseline'])
        plt.tight_layout()

    Qcontrol_f = np.array(Qcontrol).tolist()  # np.array(Qcontrol)/np.array(NbModulesC).tolist()#np.array(r_Qcontrol).tolist()
    Qcontrol_mod_f = np.array(Qcontrol_mod).tolist()  # np.array(Qcontrol)/np.array(NbModulesC).tolist()#np.array(r_Qcontrol).tolist()
    Qmutant_f = np.array(Qmutant).tolist()  # np.array(Qmutant)/np.array(NbModulesM).tolist()#np.array(r_Qmutant).tolist()
    Qcontrol_mod_rad_f= np.array(Qcontrol_mod_rad).tolist()
    plt.figure()
    # sns.set_style(style='white')
    sns.set_style(style="whitegrid")
    dfc = pd.DataFrame(np.array([Qcontrol_f, Qcontrol_mod_f, Qcontrol_mod_rad_f, Qmutant_f]).transpose()).melt()
    sns.boxplot(x='variable', y='value', data=dfc)
    sns.despine()
    plt.xticks([0, 1, 2, 3], ['controls','naive', 'smart', 'otof-/-'])
    plt.ylabel('modularity')
    print(ks_2samp(Qcontrol_f, Qmutant_f))
    print(wilcoxon(Qcontrol_f, Qmutant_f))
    print(brunnermunzel(Qcontrol_f, Qmutant_f))

    NbModulesC_f = np.array(NbModulesC).tolist()  # np.array(NbModulesC)/np.array(Nb_V_C).tolist()#np.array(r_NbModulesC).tolist()
    NbModulesC_mod_f = np.array(NbModulesC_mod).tolist()  # np.array(NbModulesC)/np.array(Nb_V_C).tolist()#np.array(r_NbModulesC).tolist()
    NbModulesM_f = np.array(NbModulesM).tolist()  # np.array(NbModulesM)/np.array(Nb_V_M).tolist()#np.array(r_NbModulesM).tolist()
    NbModulesC_mod_rad_f = np.array(NbModulesC_mod_rad).tolist()
    plt.figure()
    # sns.set_style(style='white')
    sns.set_style(style="whitegrid")
    dfc = pd.DataFrame(np.array([NbModulesC_f, NbModulesC_mod_f,NbModulesC_mod_rad_f, NbModulesM_f]).transpose()).melt()
    sns.boxplot(x='variable', y='value', data=dfc)
    sns.despine()
    plt.xticks([0, 1, 2, 3], ['controls','naive', 'smart', 'otof-/-'])
    plt.ylabel('Nb blocks')
    print(ks_2samp(NbModulesC_f, NbModulesM_f))
    print(wilcoxon(NbModulesC_f, NbModulesM_f))
    print(brunnermunzel(np.array(NbModulesC_f).astype(float),np.array(NbModulesM_f).astype(int),nan_policy='omit'))

## voxelization flow modified graphs

import ClearMap.Analysis.Measurements.Voxelization as vox

template_shape = (320, 528, 228)
vox_shape = (320, 528, 228, len(controls))
vox_ori_control_rad = np.zeros(vox_shape)
vox_ori_mutant_rad = np.zeros(vox_shape)

radius = 5
ps=7.3

for i, g in enumerate(controls):
    print(g)
    graph = ggt.load(
        work_dir + '/' + g + '/' + '/data_graph_correcteduniverse.gt')  # data_graph_correcteduniverse.gt')
    degrees = graph.vertex_degrees()
    vf = np.logical_and(degrees > 1, degrees <= 4)
    graph = graph.sub_graph(vertex_filter=vf)
    label = graph.vertex_annotation();
    # art_tree = graph.sub_graph(vertex_filter=vertex_filter)
    with open(work_dir + '/' + g + '/sampledict' + g + '.pkl', 'rb') as fp:
        sampledict = pickle.load(fp)

    flow = np.asarray(sampledict['flow'][0])
    # flow_reg = np.clip(flow, 0, 200)
    e = 1 - np.exp(-(ps / abs(flow)))
    e = np.nan_to_num(e)

    connectivity = graph.edge_connectivity()
    coordinates = graph.vertex_property('coordinates_atlas')  # *1.625/25
    edges_centers = np.array(
        [(coordinates[connectivity[i, 0]] + coordinates[connectivity[i, 1]]) / 2 for i in range(connectivity.shape[0])])

    # print('artBP')#artBP
    # v = vox.voxelize(art_coordinates[:, :3], shape=template_shape, weights=None, radius=(radius, radius, radius), method='sphere');
    # vox_art_control[:, :, :, i] = v
    vox_data = np.concatenate((edges_centers, np.expand_dims(e, axis=1)), axis=1)
    v = vox.voxelize(vox_data[:, :3], shape=template_shape, weights=vox_data[:, 3], radius=(radius, radius, radius),
                     method='sphere');
    # w = vox.voxelize(vox_data[:, :3], shape=template_shape, weights=None, radius=(radius, radius, radius),
    #                  method='sphere');
    vox_ori_control_rad[:, :, :, i] = v.array #/ w.array

io.write(work_dir + '/' + 'vox_extr_frac_sim_control_unormalized' + str(radius) + '.tif', vox_ori_control_rad.astype('float32'))
vox_ori_control_rad_avg = np.mean(vox_ori_control_rad, axis=3)
io.write(work_dir + '/' + 'vox_extr_frac_sim_control_unormalized_avg_' + str(radius) + '.tif', vox_ori_control_rad_avg.astype('float32'))


for i, g in enumerate(controls):
    print(g)
    graph = ggt.load(work_dir + '/' + g + '/' + 'data_graph_correcteduniverse_smart_mod.gt')  # data_graph_correcteduniverse.gt')
    # degrees = graph.vertex_degrees()
    # vf = np.logical_and(degrees > 1, degrees <= 4)
    # graph = graph.sub_graph(vertex_filter=vf)
    # label = graph.vertex_annotation();
    # art_tree = graph.sub_graph(vertex_filter=vertex_filter)
    with open(work_dir + '/' + g + '/sampledict' + 'smart_modified' + '.pkl', 'rb') as fp:
        sampledict = pickle.load(fp)

    flow = np.asarray(sampledict['flow'][0])
    # flow_reg = np.clip(flow, 0, 200)
    e = 1 - np.exp(-(ps / abs(flow)))
    e = np.nan_to_num(e)

    connectivity = graph.edge_connectivity()
    coordinates = graph.vertex_property('coordinates_atlas')  # *1.625/25
    edges_centers = np.array(
        [(coordinates[connectivity[i, 0]] + coordinates[connectivity[i, 1]]) / 2 for i in range(connectivity.shape[0])])

    # print('artBP')#artBP
    # v = vox.voxelize(art_coordinates[:, :3], shape=template_shape, weights=None, radius=(radius, radius, radius), method='sphere');
    # vox_art_control[:, :, :, i] = v
    vox_data = np.concatenate((edges_centers, np.expand_dims(e, axis=1)), axis=1)
    v = vox.voxelize(vox_data[:, :3], shape=template_shape, weights=vox_data[:, 3], radius=(radius, radius, radius),
                     method='sphere');
    # w = vox.voxelize(vox_data[:, :3], shape=template_shape, weights=None, radius=(radius, radius, radius),
    #                  method='sphere');
    vox_ori_control_rad[:, :, :, i] = v.array #/ w.array

io.write(work_dir + '/' + 'vox_extr_frac_sim_control_smart_mod_unormalized' + str(radius) + '.tif', vox_ori_control_rad.astype('float32'))
vox_ori_control_rad_avg = np.mean(vox_ori_control_rad, axis=3)
io.write(work_dir + '/' + 'vox_extr_frac_sim_control_avg_smart_mod_unormalized' + str(radius) + '.tif', vox_ori_control_rad_avg.astype('float32'))



for i, g in enumerate(mutants):
    print(g)
    graph = ggt.load(
        work_dir + '/' + g + '/' + '/data_graph_correcteduniverse.gt')  # 'data_graph_correcteduniverse.gt')
    degrees = graph.vertex_degrees()
    vf = np.logical_and(degrees > 1, degrees <= 4)
    graph = graph.sub_graph(vertex_filter=vf)
    label = graph.vertex_annotation();
    # vertex_filter = from_e_prop2_vprop(graph, 'artery')
    # art_tree = graph.sub_graph(vertex_filter=vertex_filter)
    # art_tree = graph.sub_graph(vertex_filter=vertex_filter)
    with open(work_dir + '/' + g + '/sampledict' + g + '.pkl', 'rb') as fp:
        sampledict = pickle.load(fp)

    flow = np.asarray(sampledict['flow'][0])
    # flow_reg = np.clip(flow, 0, 200)
    e = 1 - np.exp(-(ps / abs(flow)))
    e = np.nan_to_num(e)

    connectivity = graph.edge_connectivity()
    coordinates = graph.vertex_property('coordinates_atlas')  # *1.625/25
    edges_centers = np.array(
        [(coordinates[connectivity[i, 0]] + coordinates[connectivity[i, 1]]) / 2 for i in range(connectivity.shape[0])])

    # print('artBP')  # artBP
    # v = vox.voxelize(art_coordinates[:, :3], shape=template_shape, weights=None, radius=(radius, radius, radius), method='sphere');
    # vox_art_mutant[:, :, :, i] = v

    vox_data = np.concatenate((edges_centers, np.expand_dims(e, axis=1)), axis=1)
    v = vox.voxelize(vox_data[:, :3], shape=template_shape, weights=vox_data[:, 3], radius=(radius, radius, radius),
                     method='sphere');
    # w = vox.voxelize(vox_data[:, :3], shape=template_shape, weights=None, radius=(radius, radius, radius),
    #                  method='sphere');
    vox_ori_mutant_rad[:, :, :, i] = v.array #/ w.array


io.write(work_dir + '/' + 'vox_extr_frac_sim_mutant_unormalized' + str(radius) + '.tif', vox_ori_mutant_rad.astype('float32'))
vox_ori_mutant_rad_avg = np.mean(vox_ori_mutant_rad, axis=3)
io.write(work_dir + '/' + 'vox_extr_frac_sim_mutant_unormalized_avg_' + str(radius) + '.tif', vox_ori_mutant_rad_avg.astype('float32'))

for i in range(len(controls)):
    # np.save(work_dir + '/' + controls[i] + '/' + 'vox_art_'+str(radius)+'.npy', vox_art_control[:, :, :, i])
    io.write(work_dir + '/' + controls[i] + '/' + 'vvox_flow_sim_' + controls[i] + '.tif',
             vox_ori_control_rad[:, :, :, i].astype('float32'))

for i in range(len(mutants)):
    # np.save(work_dir + '/' + controls[i] + '/' + 'vox_art_'+str(radius)+'.npy', vox_art_mutant[:, :, :, i])
    io.write(work_dir + '/' + mutants[i] + '/' + 'vox_flow_sim_' + mutants[i] + '.tif',
             vox_ori_mutant_rad[:, :, :, i].astype('float32'))

from scipy import stats
pcutoff = 0.05

vox_ori_control_rad=io.read(work_dir + '/' + 'vox_extr_frac_sim_control_smart_mod_unormalized' + str(radius) + '.tif')
tvals, pvals = stats.ttest_ind(vox_ori_control_rad, vox_ori_mutant_rad, axis=3, equal_var=True);

pi = np.isnan(pvals);
pvals[pi] = 1.0;
tvals[pi] = 0;

pvals2 = pvals.copy();
pvals2[pvals2 > pcutoff] = pcutoff;
psign = np.sign(tvals)

## from sagital to coronal view
pvals2_f = np.swapaxes(np.swapaxes(pvals2, 0, 2), 1, 2)
psign_f = np.swapaxes(np.swapaxes(psign, 0, 2), 1, 2)
# pvals = self.cutoffPValues(pvals, pcutoff = pcutoff);

# pvals, psign = tTestVoxelization(vox_control_avg, vox_mutant_avg, pcutoff = None, signed = True);
pvalscol = colorPValues(pvals2_f, psign_f, positive=[255, 0, 0], negative=[0, 255, 0])

# io.write('/data_SSD_2to/191122Otof/pvalcolors.tif', np.moveaxis(pvalscol, -1, 0).astype('float32'))
# io.write('/data_SSD_2to/191122Otof/pvalcolors.tif', pvalscol.astype('uint8'), photometric='rgb')


tifffile.imsave(work_dir + '/pvalcolors_extr_frac_unormalized' + str(radius) + '.tif', np.swapaxes(pvalscol, 2, 0).astype('uint8'),
                photometric='rgb', imagej=True)




## get radius distribution/ length

radius_C=[]
radius_M=[]

for state in states:
    for control in state:
        print(control)
        graph = ggt.load(work_dir + '/' + control + '/' + 'data_graph_correcteduniverse.gt')  # data_graph_corrected_Isocortex.gt')#/data_graph_correcteduniverse.gt')#
        degrees = graph.vertex_degrees()
        vf = np.logical_and(degrees > 1, degrees <= 4)
        graph = graph.sub_graph(vertex_filter=vf)
        # diff = np.load(work_dir + '/' + control + '/sbm/' + 'diffusion_penetrating_vessel_overlap_end_point_cluster_per_region_iteration_Isocortex_graph_corrected' + '.npy')
        # graph.add_vertex_property('overlap', diff)

        for region_list in regions:

            vertex_filter = np.zeros(graph.n_vertices)
            for i, rl in enumerate(region_list):
                order, level = region_list[i]
                print(level, order, ano.find(order, key='order')['name'])
                label = graph.vertex_annotation();
                label_leveled = ano.convert_label(label, key='order', value='order', level=level)
                vertex_filter[label_leveled == order] = 1;
            gss4_t = graph.sub_graph(vertex_filter=vertex_filter)
        if state==controls:
            radius_C.append(gss4_t.edge_property('length'))
        elif state==mutants:
            radius_M.append(gss4_t.edge_property('length'))

import pandas as pd
normed=False
bin2=10
feature_C=[]
feature_M=[]
binned=False

for state in states:

    if state == controls:
        radius=radius_C
        feature=[]
    elif state == mutants:
        radius=radius_M
        feature = []
    for i in range(len(state)):
        hist_rad, bins_rad = np.histogram(radius[i], bins=bin2, normed=normed)
        if not binned:
            bin2=bins_rad
            binned=True
        feature.append(hist_rad)
    if state == controls:
        feature_C = feature
    elif state == mutants:
        feature_M = feature

Rad_C = pd.DataFrame(np.array(feature_C)).melt()
Rad_M = pd.DataFrame(np.array(feature_M)).melt()

plt.figure()
sns.set_style(style='white')
sns.lineplot(x="variable", y="value", err_style='bars', data=Rad_C, color='cadetblue')
sns.lineplot(x="variable", y="value", err_style='bars', data=Rad_M, color='indianred')
plt.xticks(np.linspace(0,10, 10), ((bins_rad[:-1]+bins_rad[1:])/2).astype(int))
plt.xlabel('edge radii')
plt.ylabel('count')
plt.legend(['controls', 'mutants'])
sns.despine()




np.save(work_dir + '/' +'Qcontrol_mod_ratios.npy', Qcontrol_mod)
np.save(work_dir + '/' +'Qcontrol_mod_rad_ratios.npy', Qcontrol_mod_rad)


np.save(work_dir + '/' +'NbModulesC_mod_ratios.npy', NbModulesC_mod)
np.save(work_dir + '/' +'NbModulesC_mod_rad_ratios.npy', NbModulesC_mod_rad)







nb_c=np.array(NbModulesC_mod).reshape(4,20)#.transpose()
mod_c=np.array(Qcontrol_mod).reshape(4, 20)#.transpose()

nb_c_smart=np.array(NbModulesC_mod_rad).reshape(4,20)#.transpose()
mod_c_smart=np.array(Qcontrol_mod_rad).reshape(4,20)#.transpose()

nb_m=np.load(work_dir + '/' +'NbModulesM.npy')
mod_m=np.load(work_dir + '/' +'Qmutant.npy')

import pandas as pd
Q_C = pd.DataFrame(mod_c).melt()
Q_C_smart = pd.DataFrame(mod_c_smart).melt()

Nb_C = pd.DataFrame(nb_c).melt()
Mb_C_smart = pd.DataFrame(nb_c_smart).melt()

Nb_M=[nb_m.tolist(),nb_m.tolist(),nb_m.tolist(),nb_m.tolist(),nb_m.tolist(),nb_m.tolist(),nb_m.tolist(),nb_m.tolist(),nb_m.tolist(),nb_m.tolist(),nb_m.tolist(),nb_m.tolist(),nb_m.tolist(),nb_m.tolist(),nb_m.tolist()]
Nb_M=np.array(Nb_M).transpose()
Nb_M= pd.DataFrame(Nb_M).melt()

Q_M=[mod_m.tolist(),mod_m.tolist(),mod_m.tolist(),mod_m.tolist(),mod_m.tolist(),mod_m.tolist(),mod_m.tolist(),mod_m.tolist(),mod_m.tolist(),mod_m.tolist(),mod_m.tolist(),mod_m.tolist(),mod_m.tolist(),mod_m.tolist(),mod_m.tolist()]
Q_M=np.array(Q_M).transpose()
Q_M= pd.DataFrame(Q_M).melt()

plt.figure()
sns.set_style('white')
sns.lineplot(x="variable", y="value", err_style='bars', data=Q_C, color='darkorange')
sns.lineplot(x="variable", y="value", err_style='bars', data=Q_C_smart, color='indianred')
sns.lineplot(x="variable", y="value", data=Q_M, color='cadetblue')

plt.title('modularity')
plt.xticks(np.arange(ratios.shape[0]), ratios)
plt.xlabel('% removed edges', size='x-large')
plt.ylabel('modularity',size='x-large')
sns.despine()
plt.legend(['random removal', 'radii-length driven removal', 'mutant baseline'])
plt.tight_layout()

plt.figure()
sns.set_style('white')
sns.lineplot(x="variable", y="value", err_style='bars', data=Nb_C, color='darkorange')
sns.lineplot(x="variable", y="value", err_style='bars', data=Mb_C_smart, color='indianred')
sns.lineplot(x="variable", y="value", data=Nb_M, color='cadetblue')

plt.title('Nb modules')
plt.xticks(np.arange(ratios.shape[0]), ratios)
plt.xlabel('% removed edges',size='x-large')
plt.ylabel('nb modules',size='x-large')
sns.despine()
plt.legend(['random removal', 'radii-length driven removal', 'mutant baseline'])
plt.tight_layout()













ConnexionDensitymut = []


for mut in mutants:


    graph = ggt.load(
        work_dir + '/' + mut + '/' + 'data_graph_correcteduniverse.gt')  # data_graph_corrected_Isocortex.gt')#/data_graph_correcteduniverse.gt')#
    degrees = graph.vertex_degrees()
    vf = np.logical_and(degrees > 1, degrees <= 4)
    graph = graph.sub_graph(vertex_filter=vf)


    for region_list in regions:

        vertex_filter = np.zeros(graph.n_vertices)
        for i, rl in enumerate(region_list):
            order, level = region_list[i]
            print(level, order, ano.find(order, key='order')['name'])
            label = graph.vertex_annotation();
            label_leveled = ano.convert_label(label, key='order', value='order', level=level)
            vertex_filter[label_leveled == order] = 1;
        # gss4_t = graph.sub_graph(vertex_filter=vertex_filter)

    gss4 = graph.copy()
    gss4 = gss4.sub_graph(vertex_filter=vertex_filter)
    conn_density_init = (2 * gss4.n_edges) / (gss4.n_vertices * (gss4.n_vertices - 1))
    print('conn_density_init: ', conn_density_init)
    ConnexionDensitymut.append(conn_density_init)

np.save(work_dir + '/' + 'MutantConnexionDensity_ratios' + ratio_name + param + '_prun_' + str(prune_graph) + '_lc_' + str(
    largest_component) + '_alpha_' + str(alpha) + '_gx_sbm.npy',
        np.array(ConnexionDensitymut))


controls=['2R','3R','5R', '8R']#['2R','3R','5R', '8R']
mutants=['1R','7R', '6R', '4R']
work_dir='/data_SSD_2to/191122Otof'
ps=7.3
# ratio=0.15#0.25

ratios=np.arange(0,50,5)/100#[1.0/7.8]
control='5R'
import graph_tool.draw as gtd

graph = ggt.load(
    work_dir + '/' + control + '/' + 'data_graph_correcteduniverse.gt')  # data_graph_corrected_Isocortex.gt')#/data_graph_correcteduniverse.gt')#
degrees = graph.vertex_degrees()
vf = np.logical_and(degrees > 1, degrees <= 4)
graph = graph.sub_graph(vertex_filter=vf)

with open(work_dir + '/' + control + '/sampledict' + control + '.pkl', 'rb') as fp:
    sampledict = pickle.load(fp)

flow = np.asarray(sampledict['flow'][0])
e = 1 - np.exp(-(ps / abs(flow)))
graph.add_edge_property('extracted_frac', e)
# diff = np.load(work_dir + '/' + control + '/sbm/' + 'diffusion_penetrating_vessel_overlap_end_point_cluster_per_region_iteration_Isocortex_graph_corrected' + '.npy')
# graph.add_vertex_property('overlap', diff)

for region_list in regions:

    vertex_filter = np.zeros(graph.n_vertices)
    for i, rl in enumerate(region_list):
        order, level = region_list[i]
        print(level, order, ano.find(order, key='order')['name'])
        label = graph.vertex_annotation();
        label_leveled = ano.convert_label(label, key='order', value='order', level=level)
        vertex_filter[label_leveled == order] = 1;
    # gss4_t = graph.sub_graph(vertex_filter=vertex_filter)

gss4 = graph.copy()
gss4 = gss4.sub_graph(vertex_filter=vertex_filter)
conn_density_init = (2 * gss4.n_edges) / (gss4.n_vertices * (gss4.n_vertices - 1))
print('conn_density_init: ', conn_density_init)
n_vert = np.sum(vertex_filter)
n_edge = gss4.n_edges


gss4 = gss4.largest_component()

### SBM
g=gss4.base
state_sbm = gti.minimize_blockmodel_dl(g)
b = state_sbm.b
pos = gtd.sfdp_layout(g)
pos_m=pos.get_2d_array(range(gss4.n_vertices))

bn= b.get_array()
new_cmap, randRGBcolors = rand_cmap(np.unique(bn).shape[0], type='bright', first_color_black=False,
                                    last_color_black=False, verbose=True)
n = gss4.n_vertices
colorval = np.zeros((n, 3));
for i in range(bn.size):
    colorval[i] = randRGBcolors[int(bn[i])]
colorval=np.array(colorval)

gss4.add_vertex_property('vertex_fill_color', colorval)
gss4.add_vertex_property('pos',pos_m[:2, :].T)#.astype(int)
g=gss4.base
pos = g.vp.pos
# vertex_fill_color=g.vp.vertex_fill_color
gtd.graph_draw(g, pos=pos,vertex_fill_color=b, vertex_size=3, output="/home/sophie.skriabine/Pictures/graphRewiring/graph-draw-sfdp_50"+str(control)+"_0.pdf")


### HSBM
g=gss4.base
# state1 = gti.minimize_nested_blockmodel_dl(g, deg_corr=True)
state_sbm = gti.minimize_blockmodel_dl(g)
state=state_sbm
b = state.b
# levels = state1.get_levels()
# b=levels[0].get_blocks()
# b = state1.levels[0].b
pos = gtd.sfdp_layout(g)
pos_m=pos.get_2d_array(range(gss4.n_vertices))
# gti.get_hierarchy_tree(state)[0]
# pos = gtd.radial_tree_layout(t, t.vertex(t.num_vertices() - 1), weighted=True)
# cts = gtd.get_hierarchy_control_points(g, t, tpos)
# pos = g.own_property(tpos)

bn= b.get_array()
new_cmap, randRGBcolors = rand_cmap(np.unique(bn).shape[0], type='bright', first_color_black=False,
                                    last_color_black=False, verbose=True)
n = gss4.n_vertices
colorval = np.zeros((n, 3));
for i in range(bn.size):
    colorval[i] = randRGBcolors[int(bn[i])]
colorval=np.array(colorval)

gss4.add_vertex_property('vertex_fill_color',b)#colorval
# g=gss4.base
#vprop={"fill_color":g.vp.vertex_fill_color},
# hpos, t , tpos=gtd.draw_hierarchy(state1, output="/home/sophie.skriabine/Pictures/graphRewiring/hierarchy_graph-draw-sfdp_50"+str(control)+"_0.pdf")# vprops={"fill_color":g.vp.vertex_fill_color}
# hpos_m=hpos.get_2d_array(range(gss4.n_vertices))
# b = state.levels[0].b
# shape = b.copy()
# shape.a %= 14
b = g.vp.vertex_fill_color#state.levels[0].b


gtd.graph_draw(g, pos=pos, vertex_fill_color=b, vertex_shape=shape, edge_control_points=cts,
              edge_color=[0, 0, 0, 0.3], vertex_anchor=0,
               output="/home/sophie.skriabine/Pictures/graphRewiring/hierarchy_graph-draw-sfdp_50"+str(control)+"_0.pdf")


gss4.add_vertex_property('pos',pos_m[:2, :].T)#.astype(int)

largest_component=False
ratios=[0.25]#[0.04, 0.2, 0.25,0.38, 0.5]
for ratio in ratios:
    # ratio=0.38#0.2#0.25

    nbE2rm=(n_edge*ratio)
    print(nbE2rm)
    weight = np.ones(gss4.n_edges)
    e2rm=random.sample(range(gss4.n_edges), k=int(np.round(nbE2rm)))
    e2rm=np.array(e2rm)
    print(e2rm.shape)
    edge_filter=np.ones(gss4.n_edges)
    # edge_filter=np.logical_not(np.array([e in e2rm for e in range(gss4.n_edges)])).astype(int)
    edge_filter[e2rm.astype(int)]=0
    print(edge_filter.shape[0]-np.sum(edge_filter))

    connectivity = gss4.edge_connectivity()
    deg2=np.zeros(gss4.n_vertices)
    deg2[connectivity[e2rm].flatten()]=1
    gss4.add_vertex_property('deg2', deg2)
    gss4_mod = gss4.sub_graph(edge_filter=edge_filter)



    if prune_graph:
        connectivity = gss4_mod.edge_connectivity()
        label = gss4_mod.vertex_annotation();
        label_leveled = ano.convert_label(label, key='order', value='order', level=level)
        audi = label_leveled == order

        # deg2 = np.logical_and(np.asarray(gss4_mod.vertex_degrees() == 2), )
        deg2=gss4_mod.vertex_property('deg2').nonzero()[0]
        print(np.sum(deg2))
        edges = []
        for d2 in deg2:
            edge = connectivity[np.asarray(np.logical_or(connectivity[:, 0] == d2, connectivity[:, 1] == d2)).nonzero()[0],
                   :]
            print(edge)
            edge = edge[edge != d2]
            print(edge)
            if edge.shape[0] == 2:
                edges.append(edge)
        print(gss4_mod, np.sum(gss4_mod.vertex_degrees()==2))
        gss4_mod.remove_vertex(deg2)
        print(gss4_mod, np.sum(gss4_mod.vertex_degrees() == 2))
        deg0 = np.asarray(gss4_mod.vertex_degrees() == 0).nonzero()[0]
        gss4_mod.remove_vertex(deg0)
        print(gss4_mod,(gss4.n_edges - gss4_mod.n_edges) / gss4.n_edges)

    import ClearMap.Analysis.Graphs.GraphProcessing as gp
    if largest_component:
        gss4_mod = gss4_mod.largest_component()
    # g = reduce_graph(gss4, edge_geometry=False, verbose=True)
    print(gss4_mod)
    print(gss4_mod.n_edges/gss4.n_edges)
    g = gss4_mod.base
    pos = g.vp.pos#gss4.vertex_property('pos')
    b = g.vp.vertex_fill_color#gss4.vertex_property('vertex_fill_color')
    # state = gti.minimize_nested_blockmodel_dl(g, deg_corr=True)
    # state=state1
    # state.levels[0].b=b
    # state.levels[0]=b

    # gtd.draw_hierarchy(state, overlap=False, pos=pos,output="/home/sophie.skriabine/Pictures/graphRewiring/hierarchy_graph-draw-sfdp_50"+ str(control) +"_"+ str(ratio)+".pdf")#vprops={"fill_color": g.vp.vertex_fill_color}

    gtd.graph_draw(g, pos=pos, vertex_fill_color=b, vertex_size=3,  output="/home/sophie.skriabine/Pictures/graphRewiring/graph-draw_50_"+str(control)+"-sfdp_"+str(ratio)+".pdf")

