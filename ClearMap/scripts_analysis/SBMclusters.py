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
from numpy import arctan2, sqrt

from sklearn import preprocessing
import math
import multiprocessing as mp #Semaphore
from scipy import stats
import graph_tool.topology as gtt
from scipy.special import gammaln

# from ClearMap.DiffusionPenetratingArteriesCortex import rand_cmap

def from_e_prop2_vprop(graph, property):
    e_prop = graph.edge_property(property)
    v_prop=np.zeros(graph.n_vertices)
    connectivity = graph.edge_connectivity()
    v_prop[connectivity[e_prop==1,0]]=1
    v_prop[connectivity[e_prop == 1,1]] = 1
    # graph.add_vertex_property(property, v_prop)
    return v_prop

def extract_AnnotatedRegion(graph, region):
    order, level = region
    print(level, order, ano.find_name(order, key='order'))

    label = graph.vertex_annotation();
    label_leveled = ano.convert_label(label, key='order', value='order', level=level)
    vertex_filter = label_leveled == order;

    gss4 = graph.sub_graph(vertex_filter=vertex_filter)
    return gss4


def sbm_test(graph, brainnb, work_dir, condition, region=None, s=None, param=None , param_name=None, title=None, save=True, overlap=False, deg_corr=True):

    if region ==None:
        graph = graph
    else:
        order, level = region
        graph=extract_AnnotatedRegion(graph, region)
    if param_name !=None:
        graph.add_edge_property('param', param)
    # graph.add_edge_property('_Rad', 1/graph.edge_radii())

    print(graph)
    base=graph.base

    if s == None:
        if param_name != None:
            state = gti.minimize_blockmodel_dl(base, state_args=dict(recs=[base.ep.param], rec_types=["real-exponential"]), overlap=overlap, deg_corr=deg_corr)
        else:
            state = gti.minimize_blockmodel_dl(base, overlap=overlap,deg_corr=deg_corr)
    elif param_name != None:
        state = gti.minimize_blockmodel_dl(base, B_min=s, B_max=s + 10,  state_args=dict(recs=[base.ep.param], rec_types=["real-exponential"]), overlap=overlap,deg_corr=deg_corr)
    else:
        state = gti.minimize_blockmodel_dl(base, B_min=s, B_max=s+10, overlap=overlap, deg_corr=deg_corr)#,  state_args=dict(recs=[base.ep._Rad], rec_types=["real-normal"]) #, base.ep.Rad] ,"real-normal"
    # levels = state.get_levels()
    n = 0
    # for l in levels:
    blocks_leveled = state.get_blocks().a
    if save:
        if s == None:
            if title!=None:
                np.save(
                    work_dir + '/' + brainnb + '/sbm/blockstate_'+condition+'_corrected_graph_' + 'unconstrained' +title+ '.npy',
                    blocks_leveled)  # weighted_radiii_length
            else:
                np.save(work_dir + '/' + brainnb + '/sbm/blockstate_'+condition+'_corrected_graph_' + 'unconstrained' + '.npy', blocks_leveled)  # weighted_radiii_length
            if overlap:
                np.save(work_dir + '/' + brainnb + '/sbm/blockstate_' + condition + '_corrected_graph_overlap_' + 'unconstrained' + '.npy',blocks_leveled)  # weighted_radiii_length
            if param_name == None:
                np.save(work_dir + '/' + brainnb+'/sbm/blockstate_'+condition+'_corrected_graph_' + '.npy', blocks_leveled)#weighted_radiii_length
            else:
                np.save(work_dir + '/' + brainnb + '/sbm/blockstate_'+condition+'_corrected_graph_' + param_name+'.npy',blocks_leveled)  # weighted_radiii_length
        else:
            np.save(work_dir + '/' + brainnb + '/sbm/blockstate_' + condition + '_corrected_graph_' + s + '.npy',blocks_leveled)  # weighted_radiii_length
    n = n + 1
    print(n)
    return state, blocks_leveled


# sbm_1_rad= sbm_test(gss4, region, s)

def plot_sbm(graph, brainnb, region, modules_path, n, slice=False):
    modules=[]
    graph = extract_AnnotatedRegion(graph, region)
    if n==0:
        modules=np.load(modules_path)

    else:
        for i in range(n):
            print(i)
            # blocks=np.load('/mnt/dnata_SSD_2to/'+brain+'/sbm/blockstate_full_brain_levelled_' + str(i) + '.npy')
            blocks=np.load('/data_SSD_2to/'+brainnb+'/sbm/blockstate_full_brain_levelled_radii_' + str(i) + '.npy')
            if i==0:
                modules=blocks
            else:
                modules=np.array([blocks[b] for b in modules])
    # print(np.unique(blocks))

    graph.add_vertex_property('blocks', modules)
    # print('get arteries...')
    # label = gts.vertex_annotation();
    # label_leveled = ano.convert_label(label, key='order', value='order', level=6)
    # vertex_filter = label_leveled ==6;
    # gss4 = gts.sub_graph(vertex_filter=vertex_filter)
    # radii=gts.edge_geometry_property('radii')
    # edge_filter = radii>0.1;
    # gss4 = gss4.sub_graph(edge_filter=edge_filter)
    if slice:
        # gss4 = gts.sub_slice((slice(1,300), slice(300,308), slice(1,480)))
        # gss4 = gts.sub_slice((slice(1, 300), slice(40, 500), slice(65, 85)))
        gss4_sub = graph.sub_slice((slice(0, 4000), slice(0, 7000), slice(2000, 2300)))
        # gss4 = gts.sub_slice((slice(0, 4000), slice(0, 7000), slice(2000, 2150)));
    else:
        gss4_sub=graph
    b = gss4_sub.vertex_property('blocks')
    print(np.unique(b))
    bn=np.zeros(b.shape)
    for i in np.unique(b):
        # print(i)
        bn[np.where(b==i)]=np.where(np.unique(b)==i)
    new_cmap, randRGBcolors = rand_cmap(len(np.unique(bn)), type='bright', first_color_black=False, last_color_black=False, verbose=True)
    print(len(np.unique(bn)), len(np.unique(b)))
    cmax = np.max(bn)
    cmin = np.min(bn)
    n = gss4_sub.n_vertices
    colors = np.zeros((n, 3));
    for i in range(b.size):
        colors[i] = randRGBcolors[int(bn[i])]
    colors=np.insert(colors, 3, 1.0, axis=1)
    # vertex_filter = label_leveled == 146;
    # gss4 = g.sub_graph(vertex_filter=vertex_filter)
    # colors4=colors[vertex_filter]
    edge_artery_label = gss4_sub.edge_property('artery')
    connectivity = gss4_sub.edge_connectivity();
    edge_colors = (colors[connectivity[:, 0]] + colors[connectivity[:, 1]]) / 2.0;
    edge_colors[edge_artery_label > 0] = [1., 1.0, 1.0, 1.0]
    # p = p3d.plot_graph_mesh(gts, vertex_colors=colors, n_tube_points=3);
    print(gss4_sub)
    p = p3d.plot_graph_mesh(gss4_sub, edge_colors=edge_colors, n_tube_points=3);



def get_length_ep(gss4):#brain_list,region_list
    # get brain vessels total length ep
    Ls = []
    # for brain in brain_list:
    #     B = []
        # gts = ggt.load('/mnt/data_SSD_2to/' + brain + '/data_graph.gt')
        # for region in region_list:
        #     order, level = region
        #     print(level, order, ano.find_name(order, key='order'))
        #
        #     label = gts.vertex_annotation();
        #     label_leveled = ano.convert_label(label, key='order', value='order', level=level)
        #     vertex_filter = label_leveled == order;
        #
        #     gss4 = gts.sub_graph(vertex_filter=vertex_filter)
    coordinates = gss4.edge_geometry_property('coordinates')
    indices = gss4.edge_property('edge_geometry_indices')

    L = 0
    for i, ind in enumerate(indices):
        diff = np.diff(coordinates[ind[0]:ind[1]], axis=0)
        Ls.append(np.sum(np.linalg.norm(diff, axis=1)))
    # print(L)
    # B.append(L * 25 / 1.6)
    # Ls.append(B)
    Ls = np.array(Ls)
    return Ls





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


def getflowEP(graph):
    Ls=get_length_ep(graph)
    flowEP=np.zeros(graph.n_edges)
    connectivity = graph.edge_connectivity();
    r=graph.edge_property('radii')
    degrees=graph.vertex_degrees()
    for i in range(len(connectivity)):
        print(i)
        flowEP[i]=(r[i]**2)/(Ls[i]*(degrees[connectivity[i][0]]+degrees[connectivity[i][1]]))
    graph.add_edge_property('flowEP',flowEP)
    np.save('/mnt/data_SSD_2to/'+brainnb+'/sbm/flowEP.npy',flowEP)
    return graph




if __name__ == "__main__":

    import random

    expe='whiskers'
    condition='snout'

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
    cont='142L'

    Qcontrol=[]
    Qmutant=[]
    for st in states:

        for cont in st:
            print(cont)
            graph_i = ggt.load(work_dir + '/' + cont + '/data_graph_correctedIsocortex.gt')
            vein = from_e_prop2_vprop(graph_i, 'vein')
            vein = graph_i.expand_vertex_filter(vein, steps=1)
            graph_i.add_vertex_property('vein', vein)
            # graph_i = ggt.load('/data_SSD_2to/'+brainnb+'/data_graph.gt')
            # graph_i=ggt.load('/data_SSD_2to/whiskers_graphs/data_graph_corrected_30R.gt')
            # vesseltype = 'overlap'


            # for r in range(5):
            #     #create mock arteries and vein seeds:
            #     MockArtSeed=random.choices(vertex_range, k=Nbart)
            #     MockVeinSeed=random.choices(vertex_range, k=Nbvein)
            #
            #     MockArtLabel=np.zeros(graph.n_vertices)
            #     for i,a in enumerate(MockArtSeed):
            #         MockArtLabel[a]=a
            #         node=a
            #         for j in range(5):
            #             try:
            #                 neighbours = graph.vertex_neighbours(node)[MockArtLabel[graph.vertex_neighbours(node)] == 0][0]
            #                 MockArtLabel[neighbours] = a
            #                 node=neighbours
            #             except IndexError:
            #                 print(j)
            #
            #
            #     MockVeinLabel=np.zeros(graph.n_vertices)
            #     for i,a in enumerate(MockVeinSeed):
            #         MockVeinLabel[a]=a
            #         node=a
            #         for j in range(5):
            #             try:
            #                 neighbours = graph.vertex_neighbours(node)[MockVeinLabel[graph.vertex_neighbours(node)] == 0][0]
            #                 MockVeinLabel[neighbours] = a
            #                 node=neighbours
            #             except IndexError:
            #                 print(j)
            #
            #     graph.add_vertex_property('MockVeinLabel',MockVeinLabel)
            #     graph.add_vertex_property('MockArtLabel',MockArtLabel)
            #     artterr=diffusion_labelled(graph, MockArtLabel, vesseltype='mockArt')
            #     veinterr = diffusion_labelled(graph, MockVeinLabel, vesseltype='mockVein')
            #     indices = get_art_vein_overlap(graph, veinterr, artterr)
            #     np.save('/data_SSD_2to/' + brainnb + '/sbm/diffusion_penetrating_vessel_mock_overlap_end_point_cluster_per_region_iteration_Isocortex.npy',indices)
            #
            #
            #     s=np.unique(indices).shape[0]
            label = graph_i.vertex_annotation();
            vertex_filter = np.zeros(graph_i.n_vertices)
            for i, rl in enumerate(region_list[0]):
                order, level = region_list[0][i]
                print(level, order, ano.find(order, key='order')['name'])
                label_leveled = ano.convert_label(label, key='order', value='order', level=level)
                vertex_filter[label_leveled == order] = 1;

            gss4 = graph_i.sub_graph(vertex_filter=vertex_filter)
            state_pp = gti.PPBlockState(gss4.base)

            state_pp.multiflip_mcmc_sweep(beta=np.inf, niter=1000)

            state_pp.draw(pos=gss4.base.vp.pos,
                      output="/data_SSD_2to/whiskers_graphs/new_graphs/142L/sbm/blockstate_pp.svg")

            def collect_partitions(st):
                global bs, dls
                bs.append(st.get_state().a.copy())
                # bs.append(st.get_blocks().a.copy())
                dls.append(st.entropy())

            Qlist=[]
            #modularity
            print(cont)
            print('##############################')
            print('modules simple')
            for deg_corr in [True, False]:
                start = time.time()

                print('deg_corr : ', deg_corr)
                state,modules = sbm_test(gss4, cont, work_dir, condition)
                s=np.unique(modules).shape[0]
                dls = []  # description length history
                bs = []  # partitions

                gss4.add_vertex_property('blocks', modules)
                # gss4.add_vertex_property('indices', indices)
                Q, Qs = modularity_measure(modules, gss4, 'blocks')
                # Qart, Qsart = modularity_measure(indices, graph, 'indices')
                print(level, order, ano.find_name(order, key='order'), ' sbm : ', Q, 'number of cluster : ', s)
                # Now we collect 2000 partitions; but the larger this is, the
                # more accurate will be the calculation

                gti.mcmc_equilibrate(state, force_niter=200, mcmc_args=dict(niter=10),
                                    callback=collect_partitions)

                # Infer partition modes
                pmode = gti.partition_modes.ModeClusterState(bs)

                # Minimize the mode state itself
                gti.mcmc_equilibrate(pmode, wait=1, mcmc_args=dict(niter=1, beta=np.inf))

                # Posterior entropy
                H = pmode.posterior_entropy()

                # log(B!) term
                logB = np.mean(gammaln(np.array([len(np.unique(b)) for b in bs]) + 1))

                # Evidence
                L = -np.mean(dls) + logB + H

                print(f"Model log-evidence for deg_corr = {deg_corr}: {L}")
                Qlist.append((Q, s, pmode))
                elapsed = time.time()
                elapsed = elapsed - start
                print("Time spent in (function name) is : ", elapsed)

            # print('##############################')
            # print('modules simple overlap')
            # modules_overlap = sbm_test(gss4, cont, work_dir, condition, overlap=True)
            # s = np.unique(modules_overlap).shape[0]
            # gss4.add_vertex_property('blocks', modules_overlap)
            # # gss4.add_vertex_property('indices', indices)
            # Q, Qs = modularity_measure(modules_overlap, gss4, 'blocks')
            # # Qart, Qsart = modularity_measure(indices, graph, 'indices')
            # print(level, order, ano.find_name(order, key='order'), ' sbm : ', Q, 'number of cluster : ', s)
            # Qlist.append((Q, s))

            radius
            print('##############################')
            print('modules radius')
            for deg_corr in [True, False]:
                print('deg_corr : ', deg_corr)
                # Apply the weight transformation
                # # y = gss4.edge_property('radii')
                rad=gss4.vertex_property('radii')
                connect=gss4.edge_connectivity()
                len=gss4.edge_property('length')
                param=[1/(len[i]*(1+abs(1-(rad[connect[i][0]]/rad[connect[i][1]])))) for i in np.arange(gss4.n_edges)]
                y = np.log(param)

                state, modules_rad = sbm_test(gss4, cont, work_dir, condition, param=np.array(param), param_name='radlength', deg_corr=deg_corr)
                s = np.unique(modules_rad).shape[0]
                dls = []  # description length history
                bs = []  # partitions
                gss4.add_vertex_property('blocks', modules_rad)
                # gss4.add_vertex_property('indices', indices)
                Q, Qs = modularity_measure(modules_rad, gss4, 'blocks')
                # Qart, Qsart = modularity_measure(indices, graph, 'indices')
                print(level, order, ano.find_name(order, key='order'), ' sbm : ', Q, 'number of cluster : ', s)
                gti.mcmc_equilibrate(state, force_niter=2000, mcmc_args=dict(niter=10),
                                     callback=collect_partitions)

                # Infer partition modes
                pmode = gti.ModeClusterState(bs)

                # Minimize the mode state itself
                gti.mcmc_equilibrate(pmode, wait=1, mcmc_args=dict(niter=1, beta=np.inf))

                # Posterior entropy
                H = pmode.posterior_entropy()

                # log(B!) term
                logB = np.mean(gammaln(np.array([len(np.unique(b)) for b in bs]) + 1))

                # Evidence
                L = -np.mean(dls) + logB + H

                print(f"Model log-evidence for deg_corr = {deg_corr}: {L}")
                Qlist.append((Q, s))
            #
            # # length
            # print('##############################')
            # print('modules length')
            # for deg_corr in [True, False]:
            #     print('deg_corr : ', deg_corr)
            #     state, modules_length = sbm_test(gss4, cont, work_dir, condition, param= gss4.edge_property('length'),param_name='length')
            #     s = np.unique(modules_length).shape[0]
            #     dls = []  # description length history
            #     bs = []  # partitions
            #     gss4.add_vertex_property('blocks', modules_length)
            #     # gss4.add_vertex_property('indices', indices)
            #     Q, Qs = modularity_measure(modules_length, gss4, 'blocks')
            #     # Qart, Qsart = modularity_measure(indices, graph, 'indices')
            #     print(level, order, ano.find_name(order, key='order'), ' sbm : ', Q, 'number of cluster : ', s)
            #     gti.mcmc_equilibrate(state, force_niter=2000, mcmc_args=dict(niter=10),
            #                          callback=collect_partitions)
            #
            #     # Infer partition modes
            #     pmode = gti.ModeClusterState(bs)
            #
            #     # Minimize the mode state itself
            #     gti.mcmc_equilibrate(pmode, wait=1, mcmc_args=dict(niter=1, beta=np.inf))
            #
            #     # Posterior entropy
            #     H = pmode.posterior_entropy()
            #
            #     # log(B!) term
            #     logB = np.mean(gammaln(np.array([len(np.unique(b)) for b in bs]) + 1))
            #
            #     # Evidence
            #     L = -np.mean(dls) + logB + H
            #
            #     print(f"Model log-evidence for deg_corr = {deg_corr}: {L}")
            #     Qlist.append((Q, s))

            if st==controls:
                Qcontrol.append(Qlist)
            if st==mutants:
                Qmutant.append(Qlist)



rad = gss4.vertex_property('radii')
connect = gss4.edge_connectivity()
len = gss4.edge_property('length')
param = [1 / (1 + abs(1 - (rad[connect[i][0]] / rad[connect[i][1]]))) for i in np.arange(gss4.n_edges)]
gss4.add_edge_property('param', np.array(param).astype(float))
gss4.add_edge_property('invL', 1/len)
base=gss4.base
overlap=False
deg_corr=True

rec_param={'alpha': 1.0, 'beta': 3.0}
state_args=dict(recs=[base.ep.param], rec_types=["real-exponential"], rec_params=[rec_param])
# st_weigthe=gti.BlockState(base, state_args=state_args, overlap=overlap,deg_corr=False)
# st_weigthe.get_rec_params()
# st_weigthe.set_rec_params(rec_param)
# st_weigthe.get_rec_params()
state = gti.minimize_blockmodel_dl(base,  state_args=dict(recs=[base.ep.param], rec_types=["real-exponential"],rec_params=[rec_param]), overlap=overlap,deg_corr=deg_corr)
state.get_rec_params()

st_weigthe=gti.BlockState(base, recs=[base.ep.param], rec_types=["discrete-binomial"], overlap=overlap,deg_corr=False)
state = gti.minimize_blockmodel_dl(base,  state_args=dict(recs=[base.ep.param,base.ep.invL], rec_types=["real-exponential","real-exponential"]), overlap=overlap,deg_corr=deg_corr)
state.draw(output="/data_SSD_2to/whiskers_graphs/new_graphs/142L/sbm/blockstate_nested_radlength_barrels.pdf")
S1 = state.entropy()
print(S1)
# we will pad the hierarchy with another four empty levels, to
# give it room to potentially increase

state = state.copy(bs=state.get_bs() + [np.zeros(1)] * 4,
                   sampling = True)

for i in range(100):
   ret = state.multiflip_mcmc_sweep(niter=10, beta=np.inf)

S2 = state.entropy()

print("Improvement:", S2 - S1)
import graph_tool.draw as gtd
import matplotlib
state.draw(output="/data_SSD_2to/whiskers_graphs/new_graphs/142L/sbm/blockstate_nested_barrels_enhanced.pdf")
state.draw(edge_color=gtd.prop_to_size(base.ep.param, power=1, log=True), ecmap=(matplotlib.cm.inferno, .6),
           eorder=g.ep.weight, edge_pen_width=gtd.prop_to_size(base.ep.param, 1, 4, power=1, log=True),
           edge_gradient=[], output="foodweb-wsbm.pdf")



modularities=[]
entropies=[]
for i in range(np.array(state.get_levels()).shape[0]):
    print(i)
    if i==0:
        modules = state.get_levels()[i].get_blocks().a
    else:
        for j in range(i):
            print(j)
            # blocks=np.load('/mnt/dnata_SSD_2to/'+brain+'/sbm/blockstate_full_brain_levelled_' + str(i) + '.npy')
            blocks=state.get_levels()[j].get_blocks().a
            if j==0:
                modules=blocks
            else:
                modules=np.array([blocks[b] for b in modules])
    s = np.unique(modules).shape[0]
    dls = []  # description length history
    bs = []  # partitions
    gss4.add_vertex_property('blocks', modules)
    # gss4.add_vertex_property('indices', indices)
    Q, Qs = modularity_measure(modules, gss4, 'blocks')
    # Qart, Qsart = modularity_measure(indices, graph, 'indices')
    print(level, order, ano.find_name(order, key='order'), ' sbm : ', Q, 'number of cluster : ', s)
    modularities.append(Q)


plt.figure()
plt.plot(modularities)

modules=state.get_blocks().a
s=np.unique(modules)
np.save(work_dir + '/' + cont + '/sbm/exponential_weigthedSBM.npy',modules)
gss4.add_vertex_property('blocks', modules)
Q, Qs = modularity_measure(modules, gss4, 'blocks')
# Qart, Qsart = modularity_measure(indices, graph, 'indices')
print(' sbm : ', Q, 'number of cluster : ', s)
b=state.get_blocks().a
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






########################################################################################################################
g=gss4.base
overlaps=np.load(work_dir+'/'+cont+'/sbm/diffusion_penetrating_vessel_overlap_end_point_cluster_per_region_iteration_Isocortex_graph_corrected.npy')
overlaps_snout=overlaps[vertex_filter.astype(bool)]
bs=np.zeros((int(np.log2(overlaps_snout.shape[0])),overlaps_snout.shape[0] ))
bs[0]=overlaps_snout
# state = gti.BlockState(g,b=overlaps_snout)   # By default this creates a state with an initial single-group
state = gti.NestedBlockState(g, bs=bs)  # hierarchy of depth ceil(log2(g.num_vertices()).

# Now we run 1000 sweeps of the MCMC

dS, nmoves = 0, 0
for i in range(100):
    ret = state.multiflip_mcmc_sweep(niter=10)
    dS += ret[0]
    nmoves += ret[1]

print("Change in description length:", dS)
print("Number of accepted vertex moves:", nmoves)


#check modularities if simple SBM ( non nested )
modules=state.get_blocks().a
np.unique(modules).shape[0]
gss4.add_vertex_property('blocks', modules)
# gss4.add_vertex_property('indices', indices)
Q, Qs = modularity_measure(modules, gss4, 'blocks')
# Qart, Qsart = modularity_measure(indices, graph, 'indices')
print(level, order, ano.find_name(order, key='order'), ' sbm : ', Q, 'number of cluster : ', s)


#check the modularities if hierarchical clustering
for i in range(np.array(state.get_levels()).shape[0]):
    print(i)
    if i==0:
        modules = state.get_levels()[i].get_blocks().a
    else:
        for j in range(i):
            print(j)
            # blocks=np.load('/mnt/dnata_SSD_2to/'+brain+'/sbm/blockstate_full_brain_levelled_' + str(i) + '.npy')
            blocks=state.get_levels()[j].get_blocks().a
            if j==0:
                modules=blocks
            else:
                modules=np.array([blocks[b] for b in modules])
    s = np.unique(modules).shape[0]
    dls = []  # description length history
    bs = []  # partitions
    gss4.add_vertex_property('blocks', modules)
    # gss4.add_vertex_property('indices', indices)
    Q, Qs = modularity_measure(modules, gss4, 'blocks')
    # Qart, Qsart = modularity_measure(indices, graph, 'indices')
    print(level, order, ano.find_name(order, key='order'), ' sbm : ', Q, 'number of cluster : ', s)




# We will first equilibrate the Markov chain
gti.mcmc_equilibrate(state, wait=100, mcmc_args=dict(niter=10))

# collect nested partitions
bs = []
state = state.copy(bs=bs)
h = [np.zeros(g.num_vertices() + 1) for s in state.get_levels()]
# h = np.zeros(g.num_vertices() + 1)

def collect_partitions(s):
   global bs
   # bs.append(s.b.a.copy())
   bs.append(s.get_bs())
   # B = s.get_nonempty_B()
   # h[B] += 1
   for l, sl in enumerate(s.get_levels()):
       B = sl.get_nonempty_B()
       h[l][B] += 1

# Now we collect the marginals for exactly 100,000 sweeps
gti.mcmc_equilibrate(state, force_niter=1000, mcmc_args=dict(niter=10),
                    callback=collect_partitions)

B=h[h.nonzeros[0]]

# Disambiguate partitions and obtain marginals
pmode = gti.PartitionModeState(bs, nested=True, converge=True)

pmode = gti.ModeClusterState(bs, nested=True)
pv = pmode.get_marginal(g)

# Get consensus estimate
bs = pmode.get_max_nested()
# Infer partition modes
pmode = gti.ModeClusterState(bs, nested=True)

# Minimize the mode state itself
gt.mcmc_equilibrate(pmode, wait=1, mcmc_args=dict(niter=1, beta=np.inf))

# Get inferred modes
modes = pmode.get_modes()

for i, mode in enumerate(modes):
    b = mode.get_max_nested()    # mode's maximum
    pv = mode.get_marginal(g)    # mode's marginal distribution

    print(f"Mode {i} with size {mode.get_M()/len(bs)}")
    state = state.copy(bs=b)
    state.draw(vertex_shape="pie", vertex_pie_fractions=pv,
               output="lesmis-partition-mode-%i.svg" % i)