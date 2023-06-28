import numpy as np

import ClearMap.Settings as settings
import ClearMap.IO.IO as io

import ClearMap.ImageProcessing.Skeletonization.Skeletonization as skl

import ClearMap.Analysis.Graphs.GraphProcessing as gp

import ClearMap.Alignment.Annotation as ano

import ClearMap.Analysis.Measurements.MeasureRadius as mr
import ClearMap.Analysis.Measurements.MeasureExpression as me
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
import numexpr as ne
from sklearn import preprocessing
import math
import multiprocessing as mp #Semaphore
from scipy import stats
mutex = None
import graph_tool.topology as gtt
# from ClearMap.DiffusionPenetratingArteriesCortex import rand_cmap

def extract_AnnotatedRegion(graph, region):
    order, level = region
    print(level, order, ano.find(order, key='order')['name'])

    label = graph.vertex_annotation();
    label_leveled = ano.convert_label(label, key='order', value='order', level=level)
    vertex_filter = label_leveled == order;

    gss4 = graph.sub_graph(vertex_filter=vertex_filter)
    return gss4


def sbm_test(graph, region=None, s=None, param=None , param_name=None, title=None):
    if region ==None:
        graph = graph
    else:
        order, level = region
        graph=extract_AnnotatedRegion(graph, region)
    if param_name !=None:
        graph.add_edge_property('param', 1/param)
    # graph.add_edge_property('_Rad', 1/graph.edge_radii())

    print(graph)
    base=graph.base

    if s == None:
        state = gti.minimize_blockmodel_dl(base)
    elif param_name != None:
        state = gti.minimize_blockmodel_dl(base, B_min=s, B_max=s + 10,  state_args=dict(recs=[base.ep.param], rec_types=["real-normal"]))
    else:
        state = gti.minimize_blockmodel_dl(base, B_min=s, B_max=s+10)#,  state_args=dict(recs=[base.ep._Rad], rec_types=["real-normal"]) #, base.ep.Rad] ,"real-normal"
    # levels = state.get_levels()
    n = 0
    # for l in levels:
    blocks_leveled = state.get_blocks().a
    if s == None:
        if title!=None:
            np.save(
                '/data_SSD_2to/' + brainnb + '/sbm/blockstate_full_brain_levelled_corrected_graph_' + 'unconstrained' +title+ '.npy',
                blocks_leveled)  # weighted_radiii_length
        else:
            np.save('/data_SSD_2to/' + brainnb + '/sbm/blockstate_full_brain_levelled_corrected_graph_' + 'unconstrained' + '.npy', blocks_leveled)  # weighted_radiii_length
    elif param_name == None:
        np.save('/data_SSD_2to/'+brainnb+'/sbm/blockstate_full_brain_levelled_corrected_graph_' + str(n) + '.npy', blocks_leveled)#weighted_radiii_length
    else:
        np.save('/data_SSD_2to/' + brainnb + '/sbm/blockstate_full_brain_levelled_corrected_graph_' + str(n) + param_name+'.npy',blocks_leveled)  # weighted_radiii_length
    n = n + 1
    print(n)
    return blocks_leveled


# sbm_1_rad= sbm_test(gss4, region, s)

def plot_sbm(graph, region, brain, n):
    modules=[]
    graph = extract_AnnotatedRegion(graph, region)
    if n==0:
        modules=np.load('/data_SSD_2to/' + brainnb + '/sbm/blockstate_full_brain_levelled_1_over_radii_' + str(n) + '.npy')

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
    # gss4 = gts.sub_slice((slice(1,300), slice(300,308), slice(1,480)))
    # gss4 = gts.sub_slice((slice(1, 300), slice(40, 500), slice(65, 85)))
    gss4_sub = graph.sub_slice((slice(0, 4000), slice(0, 7000), slice(2000, 2300)))
    # gss4 = gts.sub_slice((slice(0, 4000), slice(0, 7000), slice(2000, 2150)));
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
    n = len(gss4_sub.vertices)
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
    # execute only if run as a script
    brain_list = ['190408_44R']  # 190912_SC3
    brainnb = brain_list[0]
    region_list = []
    region_list = [(1006, 3), (580, 5), (650, 5), (724, 5), (811, 4), (875, 4), (6, 6), (463, 6), (388, 6)]
    reg_colors = {1006: 'gold', 580: 'skyblue', 650: 'indianred', 724: 'violet', 811: 'darkorchid',
                  875: 'mediumslateblue', 6: 'forestgreen', 463: 'lightgreen', 388: 'turquoise'}
    region_list = [(6, 6)]

    graph = ggt.load('/data_SSD_2to/whiskers_graphs/data_graph_corrected_39L.gt')
    graph = extract_AnnotatedRegion(graph, (6, 6))
    print(graph)
    graph = graph[0]  # .largest_component()
    ####### test
    order = 6
    indices = np.load(
        '/data_SSD_2to/' + brainnb + '/sbm/diffusion_penetrating_vessel_overlap_end_point_cluster_per_region_iteration_' + ano.find_name(
            order, key='order') + '_graph_corrected' + '.npy')
    graph.add_vertex_property('overlap', indices)


    import graph_tool.spectral as gts
    import scipy
    L = gts.laplacian(g2plot.base)
    ew, ev = scipy.sparse.linalg.eigs(L, k=3)
    vals = np.real(ew[1:])
    vecs = np.real(ev[:, 1:])

    pos= vecs * vals

    plt.figure()
    plt.scatter(pos[:,0], pos[:,1])


    state=gti.minimize_nested_blockmodel_dl(g2plot.base)
    levels = state.get_levels()
    n = 3
    modules = []
    # for l in levels:
    #     blocks_leveled = l.get_blocks().a
    for i in range(n):
        print(i)
        blocks_leveled = levels[i].get_blocks().a
        if i == 0:
            modules = blocks_leveled
        else:
            modules = np.array([blocks_leveled[b] for b in modules])

    print(np.unique(modules))
    g.add_vertex_property('blocks', modules)

    bl1 = sbm_test(g, title='test1')
    u, c = np.unique(bl1, return_counts=True)
    print(u, c)

    p3d.plot_graph_mesh(g)




    Ls=get_length_ep(brain_list,region_list)

    s=3127
    g=g.largest_component()

    Ls = get_length_ep(giso)
    Ls=giso.edge_radii()

    sbm_test(g,region_list[0], s, Ls, 'radii')

    # plot_sbm(g,region_list[0], brainnb, 0)

    sbm_test(g, region_list[0])
    sbm = np.load('/data_SSD_2to/' + brainnb + '/sbm/blockstate_full_brain_levelled_corrected_graph_' + 'unconstrained' + '.npy')
    g.add_vertex_property('sbm', sbm)
    Q, Qs = modularity_measure(sbm, g, 'sbm')
    print('modularity : ', Q)



    # plot SBM with random colors
    print(np.unique(sbm))
    bn = np.zeros(sbm.shape)
    for i in np.unique(sbm):
        # print(i)
        bn[np.where(sbm == i)] = np.where(np.unique(sbm) == i)
    new_cmap, randRGBcolors = rand_cmap(len(np.unique(sbm)), type='bright', first_color_black=False,
                                        last_color_black=False, verbose=True)
    n = len(g.vertices)
    colorval = np.zeros((n, 3));
    for i in range(sbm.size):
        colorval[i] = randRGBcolors[int(bn[i])]
    # colorval = getColorMap_from_vertex_prop(g2plot.vertex_property('artterr'))
    p = p3d.plot_graph_line(g, color=colorval);

    sbm=np.load('/data_SSD_2to/' + brainnb + '/sbm/blockstate_full_brain_levelled_corrected_graph_' + str(0) + 'radii'+ '.npy')
    giso = extract_AnnotatedRegion(g, region_list[0])
    giso.add_vertex_property('sbm', sbm)
    Q, Qs = modularity_measure(sbm, giso, 'sbm')
    print('modularity : ',Q)

    graph=getflowEP(graph)
    sbm_test(gss4, region_list[0], brainnb)#, Ls)
    flowEPsbm=np.load('/data_SSD_2to/'+brainnb+'/sbm/blockstate_full_brain_levelled_690_clusteres_flowEP' + str(0) + '.npy')
    gss4.add_vertex_property('flowEPsbm', flowEPsbm)
    gss4_sub = gss4.sub_slice((slice(0, 4000), slice(0, 7000), slice(2000, 2300)));
    b = gss4_sub.vertex_property('flowEPsbm')
    print(np.unique(b))
    bn = np.zeros(b.shape)
    for i in np.unique(b):
        # print(i)
        bn[np.where(b == i)] = np.where(np.unique(b) == i)
    new_cmap, randRGBcolors = rand_cmap(len(np.unique(bn)), type='bright', first_color_black=False,
                                        last_color_black=False, verbose=True)
    n = len(graph.vertices)
    colorval = np.zeros((n, 3));
    for i in range(b.size):
        colorval[i] = randRGBcolors[int(bn[i])]
    # colorval = getColorMap_from_vertex_prop(g2plot.vertex_property('artterr'))
    p = p3d.plot_graph_mesh(gss4_sub, vertex_colors=colorval, n_tube_points=3);


    import os
    from ClearMap.DiffusionPenetratingArteriesCortex import diffusion_through_penetrating_arteries
    from ClearMap.DiffusionPenetratingArteriesCortex import get_penetration_arteries_dustance_surface

    brain_list = ['190408_44R', '190506_44R', '190506_6R', '190506_4R', '190506_3R']#'190408_44R',
    region_list = [(6, 6)]
    import math

    pi = math.pi

    for brainnb in brain_list:
        for region in region_list:
            order, level = region
            try:
                graph = ggt.load('/mnt/data_SSD_2to/' + brainnb + '/data_graph_reduced_transformed.gt')
            except FileNotFoundError:
                try:
                    graph = ggt.load('/mnt/data_SSD_2to/' + brainnb + '/data_graph_reduced.gt')
                except FileNotFoundError:
                    graph = ggt.load('/mnt/data_SSD_2to/' + brainnb + '/data_graph.gt')
            graph=extract_AnnotatedRegion(graph, region)

            for root, dirs, files in os.walk('/mnt/data_SSD_2to/' + brainnb + '/sbm'):
                if 'diffusion_penetrating_art_end_point_cluster_per_region_iteration_' + ano.find_name(order, key='order') + '.npy' in files:
                    artterr = np.load('/mnt/data_SSD_2to/' + brainnb + '/sbm/diffusion_penetrating_art_end_point_cluster_per_region_iteration_' + ano.find_name(order, key='order') + '.npy')
                else:
                    try:
                        artterr=diffusion_through_penetrating_arteries(graph, get_penetration_arteries_dustance_surface, get_penetrating_arteries_labels,vesseltype='art')
                    except:
                        artterr = diffusion_through_penetrating_arteries(graph, get_penetration_arteries,get_penetrating_arteries_labels,vesseltype='art')

                if 'diffusion_penetrating_vein_end_point_cluster_per_region_iteration_' + ano.find_name(order,key='order') + '.npy' in files:
                    veinterr = np.load('/mnt/data_SSD_2to/' + brainnb + '/sbm/diffusion_penetrating_vein_end_point_cluster_per_region_iteration_' + ano.find_name(order, key='order') + '.npy')
                else:
                    try:
                        veinterr = diffusion_through_penetrating_arteries(graph, get_penetration_veins_dustance_surface,
                                                                         get_penetrating_veins_labels, vesseltype='vein')
                    except:
                        veinterr = diffusion_through_penetrating_arteries(graph, get_penetration_veins,
                                                                         get_penetrating_veins_labels, vesseltype='vein')

            # if 'blockstate_full_brain_levelled_690_clusteres_flowEP0.npy' in files:
            #     modules = np.load('/mnt/data_SSD_2to/' + brainnb + '/sbm/blockstate_full_brain_levelled_690_clusteres_flowEP0.npy')
            # else:
            indices = get_art_vein_overlap(graph, veinterr, artterr)
            np.save('/mnt/data_SSD_2to/' + brainnb + '/sbm/diffusion_penetrating_vessel_overlap_end_point_cluster_per_region_iteration_Isocortex.npy',indices)
            s=np.unique(indices).shape[0]

            modules=sbm_test(graph, region, s)
            # modules=sbm_test(graph, region, brainnb)
            graph.add_vertex_property('blocks', modules)
            graph.add_vertex_property('indices', indices)
            Q, Qs = modularity_measure(modules, graph, 'blocks')
            Qart, Qsart = modularity_measure(indices, graph, 'indices')
            print(level, order, ano.find_name(order, key='order'), ' sbm : ', Q, ' indices : ', Qart, 'number of cluster : ', s)
            # Q, Qs= modularity_measure(indices, graph, 'artveinoverlap')

plt.figure()
plt.hist(Qsart, bins=100)
plt.title('modularity per cluster')



brain_list = ['190408_44R']  # 190912_SC3
brainnb = brain_list[0]
Nbart=690
Nbvein=246
graph=graph.largest_component()
vertex_range=range(graph.n_vertices)

import random

for r in range(5):
    #create mock arteries and vein seeds:
    MockArtSeed=random.choices(vertex_range, k=Nbart)
    MockVeinSeed=random.choices(vertex_range, k=Nbvein)

    MockArtLabel=np.zeros(graph.n_vertices)
    for i,a in enumerate(MockArtSeed):
        MockArtLabel[a]=a
        node=a
        for j in range(5):
            try:
                neighbours = graph.vertex_neighbours(node)[MockArtLabel[graph.vertex_neighbours(node)] == 0][0]
                MockArtLabel[neighbours] = a
                node=neighbours
            except IndexError:
                print(j)


    MockVeinLabel=np.zeros(graph.n_vertices)
    for i,a in enumerate(MockVeinSeed):
        MockVeinLabel[a]=a
        node=a
        for j in range(5):
            try:
                neighbours = graph.vertex_neighbours(node)[MockVeinLabel[graph.vertex_neighbours(node)] == 0][0]
                MockVeinLabel[neighbours] = a
                node=neighbours
            except IndexError:
                print(j)

    graph.add_vertex_property('MockVeinLabel',MockVeinLabel)
    graph.add_vertex_property('MockArtLabel',MockArtLabel)
    artterr=diffusion_labelled(graph, MockArtLabel, vesseltype='mockArt')
    veinterr = diffusion_labelled(graph, MockVeinLabel, vesseltype='mockVein')
    indices = get_art_vein_overlap(graph, veinterr, artterr)
    np.save('/data_SSD_2to/' + brainnb + '/sbm/diffusion_penetrating_vessel_mock_overlap_end_point_cluster_per_region_iteration_Isocortex.npy',indices)


    s=np.unique(indices).shape[0]

    modules=sbm_test(graph, region, s)
    graph.add_vertex_property('blocks', modules)
    graph.add_vertex_property('indices', indices)
    Q, Qs = modularity_measure(modules, graph, 'blocks')
    Qart, Qsart = modularity_measure(indices, graph, 'indices')
    print(level, order, ano.find_name(order, key='order'), ' sbm : ', Q, ' indices : ', Qart, 'number of cluster : ', s)

Nboverlaps=3127
res=[]
for r in range(5):
    #create mock arteries and vein seeds:
    MockOverlapSeed=random.choices(vertex_range, k=Nboverlaps)
    # MockVeinSeed=random.choices(vertex_range, k=Nbvein)
    MockOverlapLabel = np.zeros(graph.n_vertices)
    for i, a in enumerate(MockOverlapSeed):
        MockOverlapLabel[a] = a
    graph.add_vertex_property('MockOverlapSeed', MockOverlapLabel)
    randomterr = diffusion_labelled(graph, MockOverlapLabel, vesseltype='mockOver')
    np.save(
        '/data_SSD_2to/' + brainnb + '/sbm/diffusion_penetrating_vessel_mock_overlap_terr_cluster_per_region_iteration_Isocortex.npy',
        indices)

    s = np.unique(randomterr).shape[0]

    modules = sbm_test(graph, region, s)
    graph.add_vertex_property('blocks', modules)
    graph.add_vertex_property('randomterr', randomterr)
    Q, Qs = modularity_measure(modules, graph, 'blocks')
    Qart, Qsart = modularity_measure(randomterr, graph, 'randomterr')
    print(level, order, ano.find_name(order, key='order'), ' sbm : ', Q, ' randomterr : ', Qart, 'number of cluster : ', s)
    res.append(str(ano.find_name(order, key='order')+' '+' sbm : ' + str(Q)+ ' randomterr : '+ str(Qart)+ ' number of cluster : '+str(s)))
print(res)