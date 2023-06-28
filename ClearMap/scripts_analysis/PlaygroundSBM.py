
import numpy as np

import ClearMap.Settings as settings

import ClearMap.Alignment.Annotation as ano


import ClearMap.Visualization.Plot3d as p3d
import graph_tool.inference as gti

print('TEST')
import math
import matplotlib.pyplot as plt
import ClearMap.Analysis.Graphs.GraphGt as ggt
import graph_tool.centrality as gtc
print('loading...')
import numpy as np
import numexpr as ne
import graph_tool.topology as gtt
from sklearn import preprocessing
# from ClearMap.Visualization.Vispy.sbm_plot import *
from ClearMap.DiffusionPenetratingArteriesCortex import getExtendedOverlaps, getOverlaps, cart2sph, rand_cmap
import math
from ClearMap.RandomWalk import stochasticPathEmbedding, getColorMap_from_vertex_prop
import random
import graph_tool.generation as gtg
import graph_tool.draw as gtd
import graph_tool.collection as gtc
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon
from scipy.stats import ks_2samp

import ClearMap.Gt2Nx as Gt2Nx
import networkx.algorithms.non_randomness as nxnr
import networkx.algorithms.smallworld as nxsw

def deg_sample():
   if np.random.random() > 0.5:
       return np.random.poisson(4)#, np.random.poisson(4)
   else:
       return np.random.poisson(20)#, np.random.poisson(20)


def sample_k(max):
    accept = False
    while not accept:
        k = np.random.randint(1,max+1)
        accept = np.random.random() < 1.0/k
    return k

from ClearMap.vi import variation_of_information, comparePartition

########################### Unweighted Graphs
# graph 1
i=4
g1 = ggt.Graph(n_vertices=i, directed=False)

# graph 2
i=4
g2 = ggt.Graph(n_vertices=i, directed=False)
edges_all = np.zeros((0, 2), dtype=int)
edge=(0,1)
edges_all = np.vstack((edges_all, edge))
edge=(2,3)
edges_all = np.vstack((edges_all, edge))
g2.add_edge(edges_all)

# graph 3 random graphs
g30 = gtg.random_graph(20, deg_sample, directed=False)

g31 = gtg.random_graph(20, lambda:sample_k(3), directed=False)

# g32=gtg.random_graph(20, directed=False)

# graph 4 fully connected
i=4
g4=ggt.Graph(n_vertices=i, directed=False)
edges_all = np.zeros((0, 2), dtype=int)

for e in range(i):
    for f in range(i):
        if f<e:
            edge=(e,f)
            edges_all = np.vstack((edges_all, edge))
g4.add_edge(edges_all)


# graph 5 fully connected
i=10
g5=ggt.Graph(n_vertices=i, directed=False)
edges_all = np.zeros((0, 2), dtype=int)

for e in range(5):
    for f in range(5):
        if f<e:
            edge=(e,f)
            edges_all = np.vstack((edges_all, edge))

for e in np.arange(5,10):
    for f in np.arange(5,10):
        if f<e:
            edge=(e,f)
            edges_all = np.vstack((edges_all, edge))
g5.add_edge(edges_all)
g5bis=g5.copy()

edges_all = np.zeros((0, 2), dtype=int)
edge = (0, 7)
edges_all = np.vstack((edges_all, edge))
g5.add_edge(edges_all)

# graph 6


########################### Weighted Graphs
# graph 1
i=4
g1 = ggt.Graph(n_vertices=i, directed=False)

# graph 2
i=4
g2 = ggt.Graph(n_vertices=i, directed=False)
edges_all = np.zeros((0, 2), dtype=int)
edge=(0,1)
edges_all = np.vstack((edges_all, edge))
edge=(2,3)
edges_all = np.vstack((edges_all, edge))
g2.add_edge(edges_all)

# graph 3 random graphs
g30 = gtg.random_graph(20, deg_sample, directed=False)

g31 = gtg.random_graph(20, lambda:sample_k(3), directed=False)

# g32=gtg.random_graph(20, directed=False)

# graph 4 fully connected
i=4
g4=ggt.Graph(n_vertices=i, directed=False)
edges_all = np.zeros((0, 2), dtype=int)

for e in range(i):
    for f in range(i):
        if f<e:
            edge=(e,f)
            edges_all = np.vstack((edges_all, edge))
g4.add_edge(edges_all)


# graph 5 fully connected
i=10
g5=ggt.Graph(n_vertices=i, directed=False)
edges_all = np.zeros((0, 2), dtype=int)

for e in range(5):
    for f in range(5):
        if f<e:
            edge=(e,f)
            edges_all = np.vstack((edges_all, edge))

for e in np.arange(5,10):
    for f in np.arange(5,10):
        if f<e:
            edge=(e,f)
            edges_all = np.vstack((edges_all, edge))
g5.add_edge(edges_all)
g5bis=g5.copy()

edges_all = np.zeros((0, 2), dtype=int)
edge = (0, 7)
edges_all = np.vstack((edges_all, edge))
g5.add_edge(edges_all)

# graph 6
p=0.01
def prob(a, b):
   if a == b:
       return 1-p
   else:
       return p
def deg_sampler():
    return 3
# g = random_graph(1000,deg_sampler,verbose=True)
g = gtg.random_graph(30000, lambda: deg_sampler(), directed=False)
g = gtg.random_graph(10000, lambda: deg_sampler(), directed=False,
                         model="blockmodel",
                         block_membership=lambda: np.random.randint(2),
                         edge_probs=prob)
state = gti.minimize_blockmodel_dl(g)
sbm=state.get_blocks().a
s = np.unique(sbm).shape[0]
print(s)
Q, Qs = modularity_measure(modules, gss4, 'blocks')


print(np.unique(bm.a))
plt.figure()
plt.hist(g.degree_property_map('total').a, bins=10)
gtd.graph_draw(g, vertex_fill_color=bm, edge_color="black", output="/home/sophie.skriabine/Pictures/playgroundsbm/blockmodel10000.pdf")
g6=g.copy()


# karate_graph
g7 = gtc.data["karate"]


#get log likelihood values
ps=np.array([0.5, 0.4, 0.3, 0.2, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
lambdas=np.arange(1,10)

entropies=np.zeros((ps.size, lambdas.size))
Blocks=np.zeros((ps.size, lambdas.size))
effective_blocks=np.zeros((ps.size, lambdas.size))
V=np.zeros((ps.size, lambdas.size))

for pi, p in enumerate(ps):
    def prob(a, b):
        if a == b:
            return 1 - p
        else:
            return p
    for li, l in enumerate(lambdas):
        print(pi, li)
        g, bm = gtg.random_graph(100, lambda: np.random.poisson(l), directed=False,
                                 model="blockmodel",
                                 block_membership=lambda: np.random.randint(2),
                                 edge_probs=prob)
        S_ent=[]
        S_B=[]
        S_Be=[]
        Vis=[]
        for i in range(10):
            state = gti.minimize_blockmodel_dl(g.base)
            S_ent.append(state.entropy())
            S_Be.append(state.get_Be())
            S_B.append(state.get_B())
            Vis.append(comparePartition(state.get_blocks().a, bm))
        S_ent =np.array(S_ent)
        S_B = np.array(S_B)
        S_Be = np.array(S_Be)
        entropies[pi, li]=np.mean(S_ent)
        Blocks[pi, li]=np.mean(S_B)
        effective_blocks[pi, li]=np.mean(S_Be)
        V[pi, li]=np.mean(Vis)

plt.figure()
plt.imshow(V, cmap='viridis')
plt.colorbar()
plt.xlabel('inter/intra connectivity')
plt.ylabel('poisson param')


def getPropertyDiff(a, b):
    ua=np.unique(a)
    diff=0
    for u in ua:
        upos=np.asarray(a==u).nonzero()[0]
        b_clus=b[upos]
        ub, cb=np.unique(b_clus, return_counts=True)
        ubm=ub[0]
        diff=diff+np.linalg.norm(np.asarray(a==u).astype(int)- np.asarray(b==ubm).astype(int))
    return diff


plt.figure()
l=3
n=50000
b=50
ps=np.array([ 0.4,  0.3, 0.2,  1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
B=[]
E=[]
for p in ps:
    print(ps)
    def prob(a, b):
        if a == b:
            return 1 - p
        else:
            return p
    g, bm = gtg.random_graph(n, lambda: np.random.poisson(l), directed=False,
                             model="blockmodel",
                             block_membership=lambda: np.random.randint(b),
                             edge_probs=prob)
    bm=bm.get_array()
    S_B = []
    err=[]
    for i in range(1):
        print(i)
        state = gti.minimize_blockmodel_dl(g)
        sbm=state.get_blocks().a
        S_B.append(state.get_B())
        err.append(comparePartition(state.get_blocks().a, bm))
    B.append(S_B)
    E.append(err)
    print(B)
    print(E)
plt.figure()
dfc = pd.DataFrame(np.array(E).transpose()).melt()
sns.lineplot(x="variable", y="value", err_style="bars", ci='sd', data=dfc)
sns.despine()

plt.xticks(np.arange(ps.size), ps)
plt.ylabel('Nb blocks')
plt.xlabel('inter/intra edges prob')



#test on deprived animals data
import graph_tool.generation as gtg
import networkx as nx
import pickle

cases=[('whiskers', 'nose', True)]#,('otof', 'auditory')#('whiskers', 'nose'),

for case in cases:
    expe=case[0]#'whiskers'
    condition=case[1]#'sss'
    print(expe, condition)
    directed=case[2]

    Nbart=690
    Nbvein=246

    if expe == 'otof':
        work_dir = '/data_SSD_2to/191122Otof'
        controls = ['2R', '3R', '5R', '8R']
        mutants = ['1R', '7R', '6R', '4R']
    elif expe == 'whiskers':
        work_dir = '/data_SSD_2to/whiskers_graphs/new_graphs'
        controls = ['142L', '158L', '162L', '164L']
        mutants = ['138L', '141L', '163L', '165L']

    states = [controls, mutants]

    if condition == 'Aud_p':
        region_list = [(142, 8)]  # auditory
        order, level = region_list[0]
    elif condition == 'barrels':
        region_list = [(54, 9)]  # barrels
        order, level = region_list[0]
    elif condition == 'cortex':
        region_list = [[(6, 6)]]
    elif condition == 'snout':
        region_list = [[(54, 9), (47, 9)]]
    elif condition == 'auditory':
        region_list = [[(142, 8), (149, 8), (128, 8), (156, 8)]]
    elif condition == 'sss':
        region_list = [[(103, 8)]]
    elif condition == 'nose':
        region_list = [[(47, 9)]]
    elif condition == 'mop':
        region_list = [[(19, 8)]]


    Qcontrol=[]
    Qmutant=[]
    erscontrol=[]
    ersmutant=[]
    NbModulesC=[]
    NbModulesM = []

    r_Qcontrol=[]
    r_Qmutant=[]
    r_erscontrol=[]
    r_ersmutant=[]
    r_NbModulesC=[]
    r_NbModulesM = []

    Nb_V_C=[]
    Nb_V_M=[]

    Sigmas_C=[]
    Randomness_C=[]

    Sigmas_M=[]
    Randomness_M=[]

    for st in states:

        for cont in st:
            print(cont)
            graph_i = ggt.load(work_dir + '/' + cont + '/data_graph_correcteduniverse_directed.gt')#data_graph_correcteduniverse
            if not directed:
                degrees = graph_i.vertex_degrees()
                vf = np.logical_and(degrees > 1, degrees <= 4)
                graph_i = graph_i.sub_graph(vertex_filter=vf)
            with open(work_dir + '/' + cont + '/sampledict' + cont + '.pkl', 'rb') as fp:
                sampledict = pickle.load(fp)

            f = np.asarray(sampledict['flow'][0])
            v = np.asarray(sampledict['v'][0])
            graph_i.add_edge_property('flow', f)
            graph_i.add_edge_property('veloc', v)

            # vein = from_e_prop2_vprop(graph_i, 'vein')
            # vein = graph_i.expand_vertex_filter(vein, steps=1)
            # graph_i.add_vertex_property('vein', vein)

            label = graph_i.vertex_annotation();
            vertex_filter = np.zeros(graph_i.n_vertices)
            for i, rl in enumerate(region_list[0]):
                order, level = region_list[0][i]
                print(level, order, ano.find(order, key='order')['name'])
                label_leveled = ano.convert_label(label, key='order', value='order', level=level)
                vertex_filter[label_leveled == order] = 1;

            gss4 = graph_i.sub_graph(vertex_filter=vertex_filter)
            Nb_vertices=gss4.n_vertices
            rad = gss4.vertex_property('radii')
            connect = gss4.edge_connectivity()
            leng = gss4.edge_property('length')
            param = [1 / (1 + abs(1 - (rad[connect[i][0]] / rad[connect[i][1]]))) for i in np.arange(gss4.n_edges)]
            gss4.add_edge_property('param', np.array(param).astype(float))
            gss4.add_edge_property('invL', 1 / leng)

            r, p, n, l = getRadPlanOrienttaion(gss4, graph_i)#, local_normal=True)
            r = r[~np.isnan(r)]
            p = p[~np.isnan(r)]

            gss4.add_edge_property('rp', p/(r+p))
            # gss4.add_edge_property('p', p)

            g=gss4.base

            mod=[]
            bloc=[]
            Rand=[]
            Sig=[]

            mod_r=[]
            bloc_r=[]
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
                if directed:
                    g.is_directed()
                state = gti.minimize_blockmodel_dl(g)
                modules=state.get_blocks().a
                s = np.unique(modules).shape[0]
                dls = []  # description length history
                bs = []  # partitions

                gss4.add_vertex_property('blocks', modules)
                # gss4.add_vertex_property('indices', indices)
                Q, Qs = modularity_measure(modules, gss4, 'blocks')
                m=state.get_matrix()
                M=np.asarray(m.todense().flatten())
                bloc.append(state.get_B())
                mod.append(Q)
                # if i==0:
                #     for module in np.unique(modules):
                #         gg=gss4.sub_graph(vertex_filter=modules==module)
                #         gg=Gt2Nx.gt2nx(gg)
                #         Gcc = sorted(nx.connected_components(gg), key=len, reverse=True)
                #         gg = gg.subgraph(Gcc[0])
                #         try:
                #             nr=nxnr(gg, 1)
                #         except:
                #             print('error duroing nonrandomness computation !')
                #             nr=0
                #         si=1#nxsw.sigma(gg)
                #         print('non randomness : ', nr,'small world sigma : ', si)
                #         Rand.append(nr)
                #         # Sig.append(si)



                # ret = gtg.random_rewire(g, "configuration")
                # gss4_r = ggt.Graph(base=g)
                # state = gti.minimize_blockmodel_dl(g)
                # modules = state.get_blocks().a
                # s = np.unique(modules).shape[0]
                # dls = []  # description length history
                # bs = []  # partitions
                #
                # gss4_r.add_vertex_property('blocks', modules)
                # # gss4.add_vertex_property('indices', indices)
                # Q, Qs = modularity_measure(modules, gss4_r, 'blocks')
                # m = state.get_matrix()
                # M = np.asarray(m.todense().flatten())
                #
                # bloc_r.append(state.get_B())
                # mod_r.append(Q)

            if st==controls:
                Qcontrol.append(np.mean(np.array(mod)))
                # erscontrol.append(M)
                NbModulesC.append(np.mean(np.array(bloc)))
                Nb_V_C.append(Nb_vertices)
                # Sigmas_C.append(Sig)
                # Randomness_C.append(Rand)
            else:
                Qmutant.append(np.mean(np.array(mod)))
                # ersmutant.append(M)
                NbModulesM.append(np.mean(np.array(bloc)))
                Nb_V_M.append(Nb_vertices)
                # Sigmas_M.append(Sig)
                # Randomness_M.append(Rand)

            # if st == controls:
            #     r_Qcontrol.append(Q)
            #     r_erscontrol.append(M)
            #     r_NbModulesC.append(state.get_B())
            # else:
            #     r_Qmutant.append(Q)
            #     r_ersmutant.append(M)
            #     r_NbModulesM.append(state.get_B())

        b = state.get_blocks().a
        print(np.unique(b))
        bn = np.zeros(b.shape)
        for i in np.unique(b):
            # print(i)
            bn[np.where(b == i)] = np.where(np.unique(b) == i)
        new_cmap, randRGBcolors = rand_cmap(np.unique(bn).shape[0], type='bright', first_color_black=False,
                                            last_color_black=False, verbose=True)
        n = gss4.n_vertices
        colorval = np.zeros((n, 3));
        for i in range(b.size):
            colorval[i] = randRGBcolors[int(bn[i])]
        # colorval = getColorMap_from_vertex_prop(g2plot.vertex_property('artterr'))
        # p3d.plot_graph_mesh(gss4, vertex_colors=colorval, n_tube_points=3);
        if st==controls:
            if directed:
                np.save('/home/sophie.skriabine/Pictures/playgroundsbm/' + expe + '_' + condition + 'control' + 'directed_blockmodel.npy',b)
            else:
                np.save('/home/sophie.skriabine/Pictures/playgroundsbm/'+expe+'_'+condition+'control'+'blockmodel.npy', b)
        else:
            if directed:
                np.save('/home/sophie.skriabine/Pictures/playgroundsbm/' + expe + '_' + condition + 'mutant' + 'directed_blockmodel.npy',b)
            else:
                np.save('/home/sophie.skriabine/Pictures/playgroundsbm/'+expe+'_'+condition+'mutant'+'blockmodel.npy', b)

    Qcontrol_f=np.array(Qcontrol).tolist()#np.array(Qcontrol)/np.array(NbModulesC).tolist()#np.array(r_Qcontrol).tolist()
    Qmutant_f=np.array(Qmutant).tolist()#np.array(Qmutant)/np.array(NbModulesM).tolist()#np.array(r_Qmutant).tolist()
    plt.figure()
    dfc = pd.DataFrame(np.array([Qcontrol_f,Qmutant_f]).transpose()).melt()
    sns.boxplot(x='variable', y='value', data=dfc)
    sns.despine()
    plt.xticks([0,1], ['controls', 'otof-/-'])
    plt.ylabel('modularity')
    print(ks_2samp(Qcontrol_f, Qmutant_f))
    print(wilcoxon(Qcontrol_f, Qmutant_f))


    NbModulesC_f=np.array(NbModulesC).tolist()#np.array(NbModulesC)/np.array(Nb_V_C).tolist()#np.array(r_NbModulesC).tolist()
    NbModulesM_f=np.array(NbModulesM).tolist()#np.array(NbModulesM)/np.array(Nb_V_M).tolist()#np.array(r_NbModulesM).tolist()
    plt.figure()
    dfc = pd.DataFrame(np.array([NbModulesC_f,NbModulesM_f]).transpose()).melt()
    sns.boxplot(x='variable', y='value', data=dfc)
    sns.despine()
    plt.xticks([0,1], ['controls', 'otof-/-'])
    plt.ylabel('Nb blocks')
    print(ks_2samp(NbModulesC_f, NbModulesM_f))
    print(wilcoxon(NbModulesC_f, NbModulesM_f))

    for i in range(len(Randomness_C)):
        for r in range(len(Randomness_C[i])):
            if Randomness_C[i][r]==0:
                Randomness_C[i][r]=(0,0)

        q=np.array(np.array(Randomness_C)[i])[:, 1]
        q = q[~np.isnan(q)]
        bin=np.arange(-150, 0, 20)
        if i==0:
            histc, bins_c = np.histogram(q, bins=bin, normed=True)
            qc=histc.reshape((histc.size, 1))
        else:
            histc, bins_c = np.histogram(q, bins=bins_c, normed=True)
            qc=np.concatenate((qc, histc.reshape((histc.size, 1))), axis=1)

    for i in range(len(Randomness_M)):
        try:
            q = np.array(np.array(Randomness_M)[i])[:, 1]
            q = q[~np.isnan(q)]
            histm, bins_m = np.histogram(q, bins=bins_c, normed=True)
            if i==0:
                qm=histm.reshape((histc.size, 1))
            else:
                qm=np.concatenate((qm, histm.reshape((histm.size, 1))), axis=1)
        except:
            print('error of nr')

    plt.figure()
    sns.set_style(style='white')
    # histc, bins_c = np.histogram(qc, bins=10, normed=True)
    # histm, bins_m = np.histogram(qm, bins=bins_c, normed=True)
    # dfc = pd.DataFrame(np.array(histc.reshape((histc.size, 1))).transpose()).melt()
    # dfm = pd.DataFrame(np.array(histm.reshape((histm.size, 1))).transpose()).melt()
    dfc = pd.DataFrame(qc.transpose()).melt()
    dfm = pd.DataFrame(qm.transpose()).melt()
    sns.lineplot(x="variable", y="value", err_style="bars", ci='sd',data=dfc)
    sns.lineplot(x="variable", y="value", err_style="bars", ci='sd',data=dfm)
    # plt.yscale('log')
    sns.despine()
    plt.xticks(np.arange(bins_m.size)-0.5, bins_m.astype(int), rotation=45)
    plt.ylabel('non randomness values distribution')
    plt.xlabel('non randomness values distribution')


    # erscontrol=np.array(erscontrol)#/np.array(r_erscontrol).tolist()
    # ersmutant=np.array(ersmutant)#/np.array(r_ersmutant).tolist()
    for i, q in enumerate(erscontrol):

        if i==0:
            histc, bins_c = np.histogram(q, bins=10, normed=True)
            qc=histc.reshape((histc.size, 1))
        else:
            histc, bins_c = np.histogram(q, bins=bins_c, normed=True)
            qc=np.concatenate((qc, histc.reshape((histc.size, 1))), axis=1)

    for i, q in enumerate(ersmutant):
        histm, bins_m = np.histogram(q, bins=bins_c, normed=True)
        if i==0:
            qm=histm.reshape((histc.size, 1))
        else:
            qm=np.concatenate((qm, histm.reshape((histm.size, 1))), axis=1)

    plt.figure()
    # histc, bins_c = np.histogram(qc, bins=10, normed=True)
    # histm, bins_m = np.histogram(qm, bins=bins_c, normed=True)
    # dfc = pd.DataFrame(np.array(histc.reshape((histc.size, 1))).transpose()).melt()
    # dfm = pd.DataFrame(np.array(histm.reshape((histm.size, 1))).transpose()).melt()
    dfc = pd.DataFrame(qc.transpose()).melt()
    dfm = pd.DataFrame(qm.transpose()).melt()
    sns.lineplot(x="variable", y="value", err_style="bars", ci='sd',data=dfc)
    sns.lineplot(x="variable", y="value", err_style="bars", ci='sd',data=dfm)
    plt.yscale('log')
    sns.despine()
    plt.xticks(np.arange(bins_m.size), bins_m.astype(int), rotation=45)
    plt.ylabel('krs values distribution')
    plt.xlabel('krs values distribution')


    # b=state.get_blocks().a
    # print(np.unique(b))
    # bn = np.zeros(b.shape)
    # for i in np.unique(b):
    #     # print(i)
    #     bn[np.where(b == i)] = np.where(np.unique(b) == i)
    # new_cmap, randRGBcolors = rand_cmap(np.unique(bn).shape[0], type='bright', first_color_black=False,
    #                                     last_color_black=False, verbose=True)
    # n = gss4.n_vertices
    # colorval = np.zeros((n, 3));
    # for i in range(b.size):
    #     colorval[i] = randRGBcolors[int(bn[i])]
    # # colorval = getColorMap_from_vertex_prop(g2plot.vertex_property('artterr'))
    # p = p3d.plot_graph_mesh(gss4, vertex_colors=colorval, n_tube_points=3);
    #

gss4.save('/home/sophie.skriabine/Pictures/playgroundsbm/4Raudi.gt')





plt.figure()
ns=np.arange(10,300, 50)
p=1e-4
ls=np.arange(3,10, 1)

for l in ls:
    ef = []
    for n in ns:
        S_B=[]

        g, bm = gtg.random_graph(n, lambda: np.random.poisson(l), directed=False,
                                 model="blockmodel",
                                 block_membership=lambda: np.random.randint(2),
                                 edge_probs=prob)
        for i in range(20):
            state = gti.minimize_blockmodel_dl(g)
            S_B.append(state.get_B())
        ef.append(S_B)

    dfc = pd.DataFrame(np.array(ef).transpose()).melt()

    sns.lineplot(x="variable", y="value", err_style="bars", ci='sd', data=dfc)
sns.despine()
plt.legend(ls)
plt.xticks(np.arange(np.arange(10,300, 50).size), np.arange(10,300, 50))
plt.ylabel('Nb blocks')
plt.xlabel('Nb vertices')


#get entropies
g_name=['g1', 'g2', 'g30', 'g31', 'g4','g5bis', 'g5', 'g6', 'gkarate']
for i, g in enumerate([g1, g2, g30, g31, g4,g5bis, g5, g6, g7]):
    plt.figure()
    if g !=g7:
        g=g.base
    for j in range(20):
        state=gti.BlockState(g, B=2)

        bs=[]
        k=0
        S=0
        def collect_partitions(s):
            global bs, k
            k=k+1
            if k % 100==0:
                bs.append(s.entropy())


        # Now we collect partitions for exactly 100,000 sweeps, at intervals
        # of 10 sweeps:
        for r in range(10):
            dS, na, nm=state.mcmc_sweep(beta=1, c=1, niter=100)
            S=S+dS
            print(dS, na, nm)
            bs.append(S)
        # gti.mcmc_equilibrate(state, force_niter=1000, mcmc_args=dict(niter=2),
        #                     callback=collect_partitions)
        # state = gti.minimize_blockmodel_dl(g)
        print(state.get_blocks().a)


        plt.plot(bs)
    plt.title('entropy '+g_name[i])
    state.draw(output="/home/sophie.skriabine/Pictures/playgroundsbm/"+'g'+g_name[i]+"sbm_test.svg")
    
    
    
## COMPARE SBM ART TERITORIES
work_dir = '/data_SSD_2to/191122Otof'
g='7R'
# region_list = [[(54, 9), (47, 9)]]
region_list = [[(142, 8), (149, 8), (128, 8), (156, 8)]]

from ClearMap.vi import variation_of_information
graph_i = ggt.load(work_dir + '/' + g + '/data_graph_correctedIsocortex.gt')

diff = np.load(
    work_dir + '/' + g + '/sbm/' + 'diffusion_penetrating_vessel_overlap_end_point_cluster_per_region_iteration_Isocortex_graph_corrected' + '.npy')
art_terr = np.load(
    work_dir + '/' + g + '/sbm/' + 'diffusion_penetrating_vessel_art_end_point_cluster_per_region_iteration_Isocortex_graph_corrected' + '.npy')
vein_terr=np.load(
    work_dir + '/' + g + '/sbm/' + 'diffusion_penetrating_vessel_vein_end_point_cluster_per_region_iteration_Isocortex_graph_corrected' + '.npy')

vein = from_e_prop2_vprop(graph_i, 'vein')
vein = graph_i.expand_vertex_filter(vein, steps=1)
graph_i.add_vertex_property('vein', vein)

graph_i.add_vertex_property('art_terr', art_terr)
graph_i.add_vertex_property('vein_terr', vein_terr)
graph_i.add_vertex_property('diff', diff)
label = graph_i.vertex_annotation();
vertex_filter = np.zeros(graph_i.n_vertices)
for i, rl in enumerate(region_list[0]):
    order, level = region_list[0][i]
    print(level, order, ano.find(order, key='order')['name'])
    label_leveled = ano.convert_label(label, key='order', value='order', level=level)
    vertex_filter[label_leveled == order] = 1;

gss4 = graph_i.sub_graph(vertex_filter=vertex_filter)
# gss4=graph_i

Nb_vertices = gss4.n_vertices
rad = gss4.vertex_property('radii')
connect = gss4.edge_connectivity()
leng = gss4.edge_property('length')
param = [1 / (1 + abs(1 - (rad[connect[i][0]] / rad[connect[i][1]]))) for i in np.arange(gss4.n_edges)]
gss4.add_edge_property('param', np.array(param).astype(float))
gss4.add_edge_property('invL', 1 / leng)

print('orientation ... ')
r, p, l = getRadPlanOrienttaion(gss4, graph_i)#,local_normal=True
print('... done')
r = r[~np.isnan(r)]
p = p[~np.isnan(r)]
gss4.add_edge_property('rp', p / (r + p))
g=gss4.base
print('sbm ... ')
state = gti.minimize_blockmodel_dl(g, state_args=dict(recs=[g.ep.param, g.ep.invL, g.ep.rp],rec_types=["real-exponential", "real-exponential", "real-exponential"]))
print('... done')

modules=state.get_blocks().a
s = np.unique(modules).shape[0]

art_terr = gss4.vertex_property('art_terr')
vein_terr = gss4.vertex_property('vein_terr')
diff = gss4.vertex_property('diff')

print(work_dir, g, 'Nb modules SBM : ',s )
print('Nb art territories : ', np.unique(art_terr).shape)
print('Nb vein territories : ', np.unique(vein_terr).shape)
v=comparePartition(modules, art_terr)
print('variation of info SBM vs artery terr : ', v)
v=comparePartition(modules, vein_terr)
print('variation of info SBM vs vein terr : ', v)

b=art_terr
print(np.unique(b))
bn = np.zeros(b.shape)
for i in np.unique(b):
    # print(i)
    bn[np.where(b == i)] = np.where(np.unique(b) == i)
new_cmap, randRGBcolors = rand_cmap(np.unique(bn).shape[0], type='bright', first_color_black=False,
                                    last_color_black=False, verbose=True)
n = gss4.n_vertices
colorval = np.zeros((n, 3));
for i in range(b.size):
    colorval[i] = randRGBcolors[int(bn[i])]
# colorval = getColorMap_from_vertex_prop(g2plot.vertex_property('artterr'))
p = p3d.plot_graph_mesh(gss4, vertex_colors=colorval, n_tube_points=3);

b=modules
print(np.unique(b))
bn = np.zeros(b.shape)
for i in np.unique(b):
    # print(i)
    bn[np.where(b == i)] = np.where(np.unique(b) == i)
new_cmap, randRGBcolors = rand_cmap(np.unique(bn).shape[0], type='bright', first_color_black=False,
                                    last_color_black=False, verbose=True)
n = gss4.n_vertices
colorval = np.zeros((n, 3));
for i in range(b.size):
    colorval[i] = randRGBcolors[int(bn[i])]
# colorval = getColorMap_from_vertex_prop(g2plot.vertex_property('artterr'))
q_mod = p3d.plot_graph_mesh(gss4, vertex_colors=colorval, n_tube_points=3);

vein_terr_modify=mergePartition(gss4, modules, vein_terr)
np.unique(vein_terr_modify, return_counts=True)


b=vein_terr_modify
print(np.unique(b))
bn = np.zeros(b.shape)
for i in np.unique(b):
    # print(i)
    bn[np.where(b == i)] = np.where(np.unique(b) == i)
new_cmap, randRGBcolors = rand_cmap(np.unique(bn).shape[0], type='bright', first_color_black=False,
                                    last_color_black=False, verbose=True)
n = gss4.n_vertices
colorval = np.zeros((n, 3));
for i in range(b.size):
    colorval[i] = randRGBcolors[int(bn[i])]
# colorval = getColorMap_from_vertex_prop(g2plot.vertex_property('artterr'))
q2 = p3d.plot_graph_mesh(gss4, vertex_colors=colorval, n_tube_points=3);





p=0.001
def prob(a, b):
   if a == b:
       return 1-p
   else:
       return p
def deg_sampler():
    return 3

g = gtg.random_graph(10000, lambda: deg_sampler(), directed=False,
                        model="blockmodel",
                        block_membership=lambda: np.random.randint(20),
                        edge_probs=prob)



modules=g[1].a
partition=getpartition(g[1].a)

gnx=gt2nx(g[0])
# nx.set_node_attributes(gnx, modules, "modules")
res=modularity(gnx,partition)
print(res)

# nx.set_node_attributes(gnx, modules, "modules")
for i in range(10000):
    gnx.nodes[i]['modules'] = modules[i]


connectivity = [(e[0], e[1]) for e in gnx.edges()]
ratio=0.5
print(ratio)
n_edge=len(gnx.edges)
nbE2rm=(n_edge*ratio)
e2rm_indices=random.sample(range(len(connectivity)), k=int(np.round(nbE2rm)))
e2rm_indices=np.array(e2rm_indices)
print(e2rm_indices.shape)
connectivity=np.array(connectivity)
e2rm=connectivity[e2rm_indices]
gnx_rm=gnx.copy()

for e in e2rm:
    gnx_rm.remove_edge(*e)
    # print(len(gnx_rm.edges))
# gnx_rm=gnx_rm.remove_edges_from(e2rm.tolist())
modules_rm=[gnx_rm.nodes[i]['modules'] for i in range(len(gnx_rm.nodes))]

partition_rm=getpartition(modules_rm)
res_rm=modularity(gnx_rm,partition_rm)
print(res_rm)




import ClearMap.Analysis.Graphs.GraphGt_old as ggto
n_vertices=len(gnx_rm.nodes)
gnx_gt=ggto.Graph(n_vertices=n_vertices-1, directed=False)
print(gnx_gt)
edges_all=np.zeros((0,2), dtype=int)
for e in gnx_rm.edges:
    edge=(int(e[0]), int(e[1]))
    edges_all=np.vstack((edges_all, edge))

# print(edges_all)
gnx_gt.add_edge(edges_all)
#radii=np.ones(edges_all.shape[0])
print(gnx_gt)

g=gnx_gt.base

state_sbm = gti.minimize_blockmodel_dl(g)
modules = state_sbm.get_blocks().a
s = np.unique(modules).shape[0]
dls = []  # description length history
bs = []  # partitions

gnx_gt.add_vertex_property('blocks', modules)
# gss4.add_vertex_property('indices', indices)
Q, Qs = modularity_measure(modules, gnx_gt, 'blocks')
# Q = get_modularity(gss4_mod, gss4_mod.vertex_property('blocks'))
print(s, Q, state_sbm.get_B())






