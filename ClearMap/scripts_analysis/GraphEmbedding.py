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


import seaborn
pi=math.pi
import pickle
brainnb = 'whiskers_graphs/39L'
#
# graph = ggt.load('/data_SSD_2to/whiskers_graphs/data_graph_corrected_39L.gt')
# graph = extract_AnnotatedRegion(graph, (6, 6))
# print(graph)
# graph = graph[0]  # .largest_component()
# ####### test
# order = 6
# indices = np.load(
#     '/data_SSD_2to/' + brainnb + '/sbm/diffusion_penetrating_vessel_overlap_end_point_cluster_per_region_iteration_' + ano.find_name(ano
#         order, key='order') + '_graph_corrected' + '.npy')
# graph.add_vertex_property('overlap', indices)
# u, c = np.unique(indices, return_counts=True)

def from_v_prop2_eprop(graph, vprop):
    # v_prop = graph.vertex_property(property)
    # e_prop=np.zeros(graph.n_edges)
    connectivity = graph.edge_connectivity()
    e_prop = np.logical_and(vprop[connectivity[:, 0]], vprop[connectivity[:, 1]])
    return e_prop


def from_e_prop2_vprop(graph, property):
    e_prop = graph.edge_property(property)
    v_prop=np.zeros(graph.n_vertices)
    connectivity = graph.edge_connectivity()
    v_prop[connectivity[e_prop==1,0]]=1
    v_prop[connectivity[e_prop == 1,1]] = 1
    # graph.add_vertex_property(property, v_prop)
    return v_prop


def getVesselOrientation(subgraph, graph):
    x = subgraph.vertex_coordinates()[:, 0]
    y = subgraph.vertex_coordinates()[:, 1]
    z = subgraph.vertex_coordinates()[:, 2]

    x_g = graph.vertex_coordinates()[:, 0]
    y_g = graph.vertex_coordinates()[:, 1]
    z_g = graph.vertex_coordinates()[:, 2]

    center = np.array([np.mean(x_g), np.mean(y_g), np.mean(z_g)])
    x = x - np.mean(x_g)
    y = y - np.mean(y_g)
    z = z - np.mean(z_g)

    spherical_coord = np.array(cart2sph(x, y, z, ceval=ne.evaluate)).T
    connectivity = subgraph.edge_connectivity()

    x_s = spherical_coord[:, 0]
    y_s = spherical_coord[:, 1]
    z_s = spherical_coord[:, 2]

    spherical_ori = np.array(
        [x_s[connectivity[:, 1]] - x_s[connectivity[:, 0]], y_s[connectivity[:, 1]] - y_s[connectivity[:, 0]],
         z_s[connectivity[:, 1]] - z_s[connectivity[:, 0]]]).T
    # orientations=preprocessing.normalize(orientations, norm='l2')

    # edge_colors = (vertex_colors[connectivity[:, 0]] + vertex_colors[connectivity[:, 1]]) / 2.0;

    # spherical_ori=np.array(cart2sph(orientations[:, 0],orientations[:, 1],orientations[:, 2], ceval=ne.evaluate)).T
    spherical_ori = preprocessing.normalize(spherical_ori, norm='l2')
    spherical_ori = np.abs(spherical_ori)

    return spherical_ori

def get_nb_radial_vessels(edge_color):
  radial=edge_color[:,2]/(edge_color[:,0]+edge_color[:,1]+edge_color[:,2])
  return(np.sum(radial>0.7))


def get_nb_parrallel_vessels(edge_color):
  planar=(edge_color[:,0]+edge_color[:,1])/(edge_color[:,2]+edge_color[:,0]+edge_color[:,1])
  print(planar.shape)
  return(np.sum(planar>0.7))

def CreateSimpleBasis(n, m):
    simpla_basis=[]
    cycles = np.arange(n,m)#[3,4,5,6,7,8]#,9
    for i in cycles:
        print('cycle', i)
        g = ggt.Graph(n_vertices=i, directed=False)
        edges_all = np.zeros((0, 2), dtype=int)
        for j in range(i):
            if j+1<i:
                edge = (j, j+1)
            else:
                edge = (j, 0)
            edges_all = np.vstack((edges_all, edge))

        # print(edges_all)
        g.add_edge(edges_all)
        simpla_basis.append(g)

    return simpla_basis


def checkUnicityOfElement(liste):
    result=[]
    for i, r in enumerate(liste):
        if i == 0:
            result.append(r.a)
        else:
            bool=False
            for j, elem in enumerate(result):
                if set(elem) == set(r.a):
                    bool=True
                    break
            if not bool:
                result.append(r.a)
                # print(len(result))
    return result

def extractCycles(graph, basis, NbLoops, cyclesEPVect, i, Nb_loops_only=False):
    # print('extractFeaturesSubgraphs : ', i)
    NbLoops=0
    cyclesEP = np.zeros((graph.n_edges, len(basis)))
    n = 1

    evect_tot = np.zeros(graph.n_edges)
    # for l in layers:
    #     order=l
    #     level=10
    #     label = g_i.vertex_annotation();
    #     label_leveled = ano.convert_label(label, key='order', value='order', level=level)
    #     vf = label_leveled == order;
    #     g_j=g_i.sub_graph(vertex_filter=vf)
    #     print(level, order, ano.find_name(order, key='order'))
    # cycLen2=[]
    # lengths = g_j.edge_property('length')
    # print(g_j)
    g_j = graph
    print(g_j)
    for i, b in enumerate(basis):
        # if i >= 3:
        res = gtt.subgraph_isomorphism(b.base, g_j.base, induced=True)
        res=checkUnicityOfElement(res)
        evect = np.zeros(g_j.n_edges)
        NbLoops = NbLoops + len(res)
        # print(i, len(res), NbLoops)
        # cycLen=[]
        if not Nb_loops_only:
            for j, r in enumerate(res):
                coordinates = g_j.vertex_property('coordinates_atlas')
                CyclePos.append(np.mean(coordinates[r], axis=0).tolist())
                t = np.zeros(g_j.n_vertices)
                t[r] = 1
                tem = from_v_prop2_eprop(g_j, t)
                # cycLen.append(np.sum(lengths[tem]))
                # print(np.sum(t),np.sum(tem))
                if np.sum(t) == np.sum(tem):
                    evect[tem] = j
                    evect_tot[tem] = 1
                else:
                    print('there is a pb !')
                # n=n+1
            # cycLen2.append(cycLen)
            # cyclesLength30.append(cycLen2)
            # print(len(cyclesLength30))

            # print(evect.shape)
            # print(cyclesEP.shape)
        if not Nb_loops_only:
            cyclesEP[:, i] = (evect)


    if not Nb_loops_only:
        cyclesEPVect.append(cyclesEP)

    if Nb_loops_only:
        return NbLoops
    else:
        return cyclesEPVect, evect_tot, NbLoops

def plotGraph(graphs, path, plot=False):
    g, g_i=graphs
    color = g.vertex_property('plot_color')
    pos = gtd.sfdp_layout(g.base)

    v=g.vertex_property('vein')
    col=g.vertex_property('plot_color')
    col[np.asarray(v==1).nonzero()[0]]=[1,1,1]
    g.add_vertex_property('plot_color', col)

    if plot:
        gtd.graph_draw(g.base, pos=pos, vertex_fill_color=g.base.vertex_properties['plot_color'], output=path)
    artery = from_e_prop2_vprop(g_i, 'artery')
    # vein = from_e_prop2_vprop(g_i, 'vein')
    vein = g_i.vertex_property('vein')
    print(np.sum(artery),np.sum(vein) )
    V, L = stochasticPathEmbedding(g, 100, 30, np.asarray(g.vertex_property('artery')).nonzero()[0])
    if plot:
        colorval = getColorMap_from_vertex_prop(V, norm=None, cmn=0, cmx=50)
        g.add_vertex_property('temp', colorval)
        gtd.graph_draw(g.base, pos=pos, vertex_fill_color=g.base.vertex_properties['temp'], output=path[:-4]+'_temp.pdf')
    if plot:
        p = p3d.plot_graph_mesh(g_i, vertex_colors=g_i.vertex_property('colorval'), n_tube_points=3);
    return V,L


def CreateEmbeddedGraph(g_i, compf, cyclesEPVect, plot=False):
    bn = np.zeros(compf.shape)
    PosBigAgg=[]
    for i in np.unique(compf):
        # print(i)
        bn[np.where(compf == i)] = np.where(np.unique(compf) == i)
    new_cmap, randRGBcolors = rand_cmap(len(np.unique(bn)), type='bright', first_color_black=False,
                                        last_color_black=False, verbose=False)

    artery = from_e_prop2_vprop(g_i, 'artery')
    # vein = from_e_prop2_vprop(g_i, 'vein')
    vein=g_i.vertex_property('vein')
    print(np.sum(artery), np.sum(vein))
    n = len(g_i.vertices)
    colorval = np.zeros((n, 3));
    for i in range(compf.shape[0] - 1):
        colorval[i] = randRGBcolors[int(bn[i])]
        if hist[compf[i]] == 1:
            colorval[i] = [0, 0, 0]
        if artery[i]:
            colorval[i] = [1, 0, 0]
        if vein[i]:
            colorval[i] = [0, 0, 1]
    g_i.add_vertex_property('colorval', colorval)

    # colorval = getColorMap_from_vertex_prop(g2plot.vertex_property('artterr'))
    if plot:
        p = p3d.plot_graph_mesh(g_i, vertex_colors=colorval, n_tube_points=3);
    u, indices = np.unique(colorval, return_inverse=True, axis=0)  # diffusion_through_penetrating_arteries_vector

    Nbloops_array=[]
    for col in u:
        # print(col)
        vf=np.asarray((colorval==col).all(axis=1))
        temp=g_i.sub_graph(vertex_filter=vf)
        # print(temp)
        loopsAgg = extractCycles(temp, simpla_basis, 0, [], i, Nb_loops_only=True)
        print(Nbloops_array)
        # if loopsAgg!=0:
        Nbloops_array.append(loopsAgg)
        # if loopsAgg>=5:
        PosBigAgg.append(np.mean(temp.vertex_property('distance_to_surface')))

    g = ggt.Graph(n_vertices=u.shape[0], directed=False)
    g.add_vertex_property('NbLoops', np.array(Nbloops_array))
    print(g.vertex_property('NbLoops'))

    conn = g_i.edge_connectivity()
    edges_all = [[indices[conn[i, 0]], indices[conn[i, 1]]] for i in range(conn.shape[0])]
    edges_all = np.array(edges_all)

    g.add_vertex_property('plot_color', u)
    vein = np.array([np.array(u[i] == [0, 0, 1]).all() for i in range(len(u))]).astype(int)
    artery = np.array([np.array(u[i] == [1, 0, 0]).all() for i in range(len(u))]).astype(int)
    backbone = np.array([np.array(u[i] == [0, 0, 0]).all() for i in range(len(u))]).astype(int)
    g.add_vertex_property('vein', vein)
    g.add_vertex_property('artery', artery)
    g.add_vertex_property('backbone', backbone)
    g.add_vertex_property('clusterLoc', np.array(PosBigAgg))
    print(g)
    # histCycles = []
    # for i, uc in enumerate(u):
    #     vf = np.asarray(colorval == uc)
    #     ef = from_v_prop2_eprop(g_i, vf)
    #     ef = np.asarray([ef[i].all() for i in range(ef.shape[0])]).nonzero()[0]
    #     # cluster_size[uc] = c[i]
    #     temp_hist = []
    #     if ef.shape[0] != 0:
    #         for j in range(len(simpla_basis)):
    #             temp_hist.append(np.unique(cyclesEPVect[-1][ef, j]).tolist())
    #     else:
    #         for j in range(len(simpla_basis)):
    #             temp_hist.append(np.unique(np.zeros(5)).tolist())
    #     histCycles.append(temp_hist)
    # histCycles = np.array(histCycles)


    ## BUILDING THE EMBEDDED GRAPH
    inter_edges = np.asarray([edges_all[i, 0] != edges_all[i, 1] for i in range(edges_all.shape[0])])  # .nonzero()[0]
    inter_edges_t = np.asarray([inter_edges[i].all() for i in range(inter_edges.shape[0])]).nonzero()[0]
    inter_edges_t_f=[edges_all[i] for i in inter_edges_t]
    # inter_edges_t_f = np.asarray(
    #     [[np.asarray(u == edges_all[ie][0]).nonzero()[0][-1], np.asarray(u == edges_all[ie][1]).nonzero()[0][-1]] for ie
    #      in inter_edges_t])

    try:
        # eu, ec = np.unique(inter_edges_t, return_counts=True, axis=0)
        g.add_edge(inter_edges_t_f)


    except:
        print('could not embed graph')
    print(g)
    return (g, g_i)


def EmbeddedPath(EmbeddedGraph):
    L_f=[]
    V_f=[]
    C_f=[]
    NV_f=[]
    BCPos=[]
    b=0
    for i, egs in enumerate(EmbeddedGraph):
        L_i=[]
        b=0
        # V_i = []
        # C_i = []
        NV_i = []
        for j, g in enumerate(egs):
            print(i, j)
            g = EmbeddedGraph[i][j][0]
            mask=np.ones(g.n_vertices)
            color = g.vertex_property('plot_color')
            artery=np.asarray(g.vertex_property('artery')).nonzero()[0]
            vein = np.asarray(g.vertex_property('vein')).nonzero()[0]
            backbone=np.asarray(g.vertex_property('plot_color')==[0,0,0]).nonzero()[0]
            mask[artery]=0
            mask[vein]=0
            mask[backbone]=0
            mask=mask.astype('bool')
            if np.sum(artery)>0:
                V, L = stochasticPathEmbedding(g, 100, 30, artery)
                if not np.isnan(np.mean(L)):
                    L_i.append(np.mean(L))
                if b==0:
                    V_i=V[mask]
                    C_i=g.vertex_property('NbLoops')
                    BC_i=g.vertex_property('clusterLoc')#[np.asarray(C_i>=6).nonzero()[0]]
                    b=1
                else:
                    V_i=np.concatenate((V_i, V[mask]))
                    C_i = np.concatenate((C_i, g.vertex_property('NbLoops')))
                    BC_i = np.concatenate((BC_i,g.vertex_property('clusterLoc')))#[np.asarray(g.vertex_property('NbLoops')>=6).nonzero()[0]]))

                NV_i.append(g.n_vertices-2)
        print(len(C_i), len(BC_i))
        L_f.append(L_i)
        V_f.append(V_i)
        C_f.append(C_i)
        NV_f.append(NV_i)
        BCPos.append(BC_i)
    return L_f, V_f, C_f, NV_f, BCPos


def checkProblematicVertices(graph):
    probvertices=np.asarray(graph.vertex_property('degrees')>=4).nonzeros()[0]
    depthprobVertics=graph.vertex_property('distance_to_surface')[probvertices]
    return depthprobVertics

if __name__ == "__main__":
    # brain_list = ['whiskers_graphs/30R']#['190408_44R']#['190408_44R']#
    # brainnb = brain_list[0]
    controlsEmbeddedGraphs=[]
    mutantsEmbeddedGraphs = []
    mutantAggloCycles=[]
    controlAggloCycles=[]
    mutantsNbloops=[]
    controlsNbloops=[]
    controlsCyclesperAggregate=[]
    mutantsCyclesperAggregate=[]
    expe='whiskers'#'otof'#'whiskers
    condition = 'snout'#auditory#snout
    # BigAggLoc_C=[]
    # BigAggLoc_M=[]
    if expe=='otof':
        work_dir = '/data_SSD_2to/191122Otof'
        controls = ['2R' ,'3R', '5R', '8R']
        mutants = ['1R', '7R', '6R', '4R']
    elif expe=='whiskers':
        work_dir='/data_SSD_2to/whiskers_graphs/new_graphs'
        controls=['142L','158L','162L', '164L']
        mutants=['138L','141L', '163L', '165L']
    cluster_max_size=100000#5000
    cluster_min_size=200#100
    simpla_basis = CreateSimpleBasis(3, 7)



    cluster_type='art'#'vein', 'overlap'
    states=[controls, mutants]

    for state in states:

        for cont in state:
            print(cont)
            graph_i = ggt.load(work_dir + '/' + cont + '/data_graph_correctedIsocortex.gt')
            vein = from_e_prop2_vprop(graph_i, 'vein')
            vein = graph_i.expand_vertex_filter(vein, steps=1)
            graph_i.add_vertex_property('vein', vein)
            # graph_i = ggt.load('/data_SSD_2to/'+brainnb+'/data_graph.gt')
            # graph_i=ggt.load('/data_SSD_2to/whiskers_graphs/data_graph_corrected_30R.gt')

            # region_list = [(6, 6)]
            order, level = (6, 6)
            vesseltype = 'overlap'

            print(level, order, ano.find_name(order, key='order'))

            label = graph_i.vertex_annotation();
            label_leveled = ano.convert_label(label, key='order', value='order', level=level)
            vertex_filter = label_leveled == order;

            # gss4 = graph.sub_graph(vertex_filter=vertex_filter)
            # gss4=graph.largest_component()
            gss4 = graph_i.sub_graph(vertex_filter=vertex_filter)

            # diff=np.load(work_dir+'/'+cont+'/sbm/diffusion_penetrating_vessel_overlap_end_point_cluster_per_region_iteration_Isocortex_graph_corrected.npy')
            # diff = np.load('/data_SSD_2to/' + brainnb + '/sbm/diffusion_penetrating_vessel_' + vesseltype + '.npy')  # + '_end_point_cluster_per_region_iteration_' + ano.find_name(order, key='order')
            # diff=np.load('/data_SSD_2to/whiskers_graphs/30R/sbm/diffusion_penetrating_vessel_overlap_end_point_cluster_per_region_iteration_Isocortex_graph_corrected.npy')
            # gss4.add_vertex_property('overlap', diff)
            print(gss4)
            if cluster_type=='overlap':
                clusters = getOverlaps(work_dir, cont, gss4)
            elif cluster_type=='art':
                clusters=np.load(work_dir+'/'+cont+'/sbm/diffusion_penetrating_vessel_art_end_point_cluster_per_region_iteration_Isocortex_graph_corrected.npy')

            gss4.add_vertex_property('overlap', clusters)

            if condition=='Aud_p':
                region_list = [(142, 8)]  # auditory
                order, level = region_list[0]
            elif condition=='barrels':
                region_list = [(54, 9)] # barrels
                order, level = region_list[0]
            elif condition == 'cortex':
                region_list = [[(6, 6)]]
            elif condition == 'snout':
                region_list = [[(54, 9), (47, 9)]]
            elif condition == 'auditory':
                region_list = [[(142, 8), (149, 8), (128, 8), (156, 8)]]




            # print(level, order, ano.find_name(order, key='order'))

            label = gss4.vertex_annotation();
            vertex_filter = np.zeros(gss4.n_vertices)
            for i, rl in enumerate(region_list[0]):
                order, level = region_list[0][i]
                print(level, order, ano.find(order, key='order')['name'])
                label_leveled = ano.convert_label(label, key='order', value='order', level=level)
                vertex_filter[label_leveled == order] = 1;

            gss4 = gss4.sub_graph(vertex_filter=vertex_filter)

            graph=gss4.copy()
            indices = gss4.vertex_property('overlap')
            u, c = np.unique(indices, return_counts=True, axis=0)


            medium_cluster = np.asarray(np.logical_and(c >= cluster_min_size, c <= cluster_max_size)).nonzero()[0]

            # medium_cluster=np.asarray(range(c.shape[0]))
            import random
            medium_cluster
            import graph_tool.draw as gtd


            # r = u[random.choice(medium_cluster)]
            NbAggloCycle = 0
            NbLoops=0
            HU=[]
            HC=[]
            heatmap=np.zeros((6000, 7))#121
            CyclePos=[]
            NbCycleperAggregate_All = []
            cyclesEPVect = []
            Embeded=[]
            # bcl=[]
            layers=[55,56,57,58,59,60]
            print('Number of cluster in brain '+cont+' : ',medium_cluster.shape)
            for k, r in enumerate(medium_cluster):
                print(k, ':',  r)
                r=int(r)
                print('cluster number : ', r, u[r])
                # vf = np.zeros(graph.n_vertices)
                # vf[np.asarray(indices == r).nonzero()[0]] = 1
                if cluster_type == 'overlap':
                    vf=getExtendedOverlaps(gss4, indices, u[r])
                elif cluster_type == 'art':
                    vf=np.asarray(indices == u[r])
                print('containing ', np.sum(vf>0), ' vertices')
                g2plot = gss4.sub_graph(vertex_filter=vf>0)
                g2plot = g2plot.largest_component()
                if g2plot.n_vertices>3000:
                    print('too big break')
                    # break
                if g2plot.n_vertices<100:
                    print('too small break')
                    # break
                else:
                    print(g2plot)
                    # g2plot=graph.copy()
                    # bl=sbm_test(g2plot, title='test0')
                    # u, c=np.unique(bl, return_counts=True)
                    # print(u, c)
                    #
                    # r = u[random.choice(u)]
                    # print('cluster number : ', r)

                    ## random choice of cluster

                    # r = random.choice(medium_cluster)
                    # print('cluster number : ', r)
                    # vf = np.zeros(graph.n_vertices)
                    # vf[np.asarray(indices == r).nonzero()[0]] = 1
                    # print('containing ', np.sum(vf), ' vertices')
                    #
                    # g2plot = graph.sub_graph(vertex_filter=vf)
                    g = g2plot.copy()

                    # g = graph.copy()
                    print(g)



                    ####  GRAPH EMBEDDING METHOD +++

                    gl=[g]

                    # cyclesLength44 = []
                    cyclesLength30=[]

                    for i, g_i in enumerate(gl):
                        NbCycleperAggregate=[]
                        cyclesEPVect, evect_tot, NbLoops=extractCycles(g_i, simpla_basis, NbLoops, cyclesEPVect, i)
                        print('NbLoops: ',NbLoops)
                        g_s = g_i.sub_graph(edge_filter=evect_tot)
                        comp, hist = gtt.label_components(g_s.base)
                        # print(hist)
                        compf = comp.a
                        NbAggloCycle=NbAggloCycle+np.asarray(hist>1).nonzero()[0].shape[0]-1 #remove skeleton part
                        uagg, cagg = np.unique(compf, return_counts=True)
                        uagg = uagg[np.asarray(cagg >= 2).nonzero()[0]]
                        for agg in uagg:
                            aggregate=g_s.sub_graph(vertex_filter=compf==agg)
                            loopsAgg=extractCycles(aggregate, simpla_basis, 0, [], i, Nb_loops_only=True)
                            print(loopsAgg)
                            NbCycleperAggregate.append(loopsAgg)


                        artery = from_e_prop2_vprop(g_i, 'artery')
                        vein = from_e_prop2_vprop(g_i, 'vein')
                        print(np.sum(artery))
                        print(np.sum(vein))
                        # compf_filter = [(compf[i] in uagg) for i in range(compf.shape[0])]
                        eg=CreateEmbeddedGraph(g_i, compf,cyclesEPVect)
                        Embeded.append(eg)
                        # bcl.append(eg[1])
                    NbCycleperAggregate_All.append(NbCycleperAggregate)
            if state==mutants:
                mutantAggloCycles.append(NbAggloCycle)
                mutantsNbloops.append(NbLoops)
                print(NbLoops)
                mutantsCyclesperAggregate.append(NbCycleperAggregate_All)
                mutantsEmbeddedGraphs.append(Embeded)
                # BigAggLoc_M.append(bcl)
            if state==controls:
                controlAggloCycles.append(NbAggloCycle)
                controlsNbloops.append(NbLoops)
                print(NbLoops)
                controlsCyclesperAggregate.append(NbCycleperAggregate_All)
                controlsEmbeddedGraphs.append(Embeded)
                # BigAggLoc_C.append(bcl)

            np.save(work_dir + '/' + cont + '/cyclesPos'+condition+'all_cortex_Extended_Meth.npy', np.array(CyclePos))

            import pickle

            with open(work_dir + '/' + cont + '/cyclesEPVect_' + condition + '_all_cortex_Extended_Meth.p', 'wb') as fp:
                pickle.dump(cyclesEPVect, fp, protocol=pickle.HIGHEST_PROTOCOL)

            with open(work_dir + '/' + cont + '/evect_tot_' + condition + '_all_cortex_Extended_Meth.p', 'wb') as fp:
                pickle.dump(evect_tot, fp, protocol=pickle.HIGHEST_PROTOCOL)

    print(mutantAggloCycles,controlAggloCycles )
    print(mutantsNbloops, controlsNbloops)
    print(mutantsCyclesperAggregate, controlsCyclesperAggregate)

    controlAggloCycles=np.array(controlAggloCycles)
    mutantAggloCycles=np.array(mutantAggloCycles)

    controlsNbloops=np.array(controlsNbloops)
    mutantsNbloops=np.array(mutantsNbloops)

    controlsCyclesperAggregate=np.array(controlsCyclesperAggregate)
    mutantsCyclesperAggregate = np.array(mutantsCyclesperAggregate)

    if expe=='otof':
        if condition == 'snout':
            mutantsEmbeddedGraphsOtofSnout = mutantsEmbeddedGraphs
            controlsEmbeddedGraphsOtofSnout = controlsEmbeddedGraphs
        elif condition == 'auditory':
            mutantsEmbeddedGraphsOtofAudi = mutantsEmbeddedGraphs
            controlsEmbeddedGraphsOtofAudi = controlsEmbeddedGraphs
    elif expe=='whiskers':
        if condition == 'snout':
            mutantsEmbeddedGraphsWhiskSnout = mutantsEmbeddedGraphs
            controlsEmbeddedGraphsWhiskSnout = controlsEmbeddedGraphs
        elif condition == 'auditory':
            mutantsEmbeddedGraphsWhiskAudi = mutantsEmbeddedGraphs
            controlsEmbeddedGraphsWhiskAudi = controlsEmbeddedGraphs


    ## save data
    np.save(work_dir + '/'+cluster_type+'_'+'controlsNbloops_all_' + condition + '_Extended_Meth.npy', np.array(controlsNbloops))
    np.save(work_dir + '/'+cluster_type+'_'+'mutantsNbloops_all_' + condition + '_Extended_Meth.npy', np.array(mutantsNbloops))
    np.save(work_dir + '/'+cluster_type+'_'+'controlAggloCycles_all_' + condition + '_Extended_Meth.npy', np.array(controlAggloCycles))
    np.save(work_dir + '/'+cluster_type+'_'+'mutantAggloCycles_all_' + condition + '_Extended_Meth.npy', np.array(mutantAggloCycles))
    np.save(work_dir + '/'+cluster_type+'_'+'controlsCyclesperAggregate_all_' + condition + '_Extended_Meth.npy', np.array(controlsCyclesperAggregate))
    np.save(work_dir + '/'+cluster_type+'_'+'mutantsCyclesperAggregate_all_' + condition + '_Extended_Meth.npy', np.array(mutantsCyclesperAggregate))
    np.save(work_dir + '/' + cluster_type + '_' + 'mutantsEmbeddedGraphs_' + condition + '_Extended_Meth.npy',np.array(mutantsEmbeddedGraphs))
    np.save(work_dir + '/' + cluster_type + '_' + 'controlsEmbeddedGraphs_' + condition + '_Extended_Meth.npy',np.array(controlsEmbeddedGraphs))

    condition = 'auditory'
    expe == 'otof'
    Mpaths_L, Mpaths_V, Mpaths_C, Mpath_NV, Mpaths_BCP=EmbeddedPath(mutantsEmbeddedGraphsOtofAudi)
    Cpaths_L, Cpaths_V, Cpaths_C, Cpath_NV, Cpaths_BCP = EmbeddedPath(controlsEmbeddedGraphsOtofAudi)
    # C=[np.mean(Cpaths[i]) for i in range(4)]
    # M=[np.mean(Mpaths[i]) for i in range(4)]
    plt.figure()
    sns.despine()
    sns.set_style(style='white')
    for i in range(len(Cpaths_L)):
        if i == 0:
            H=Cpaths_L[i]
        else:
            H = np.concatenate((H, Cpaths_L[i]))
    i=0
    box1 = plt.boxplot(H, positions=[1], patch_artist=True, widths=1)
    sns.despine()
    for i in range(len(Mpaths_L)):
        if i == 0:
            H = Mpaths_L[i]
        else:
            H = np.concatenate((H, Mpaths_L[i]))

    box2 = plt.boxplot(H, positions=[4], patch_artist=True,widths=1)
    sns.despine()
    for patch in box1['boxes']:
        patch.set_facecolor('cadetblue')
    for patch in box2['boxes']:
        patch.set_facecolor('indianred')

    plt.xticks([1,4], ['controls','deprived'], size='x-large')
    plt.title('average path length from artery to vein in embedded graph')
    # plt.tight_layout()
    plt.xlim([-1, 7])
    sns.despine()

    norm=False
    condition='auditory'
    ###
    bin = 15
    plt.figure()
    sns.despine()
    sns.set_style(style='white')
    nb=0

    for i in range(len(Cpaths_BCP)):
        if i == 0:
            hist, bins_len = np.histogram(Cpaths_BCP[i][np.asarray(Cpaths_C[i]>=nb).nonzero()[0]],  bins = bin, normed=norm)  # ,bins=bin[Cpaths_C[i]>=6]
            H = hist.reshape((hist.shape[0], 1))
        else:
            hist, bins = np.histogram(Cpaths_BCP[i][np.asarray(Cpaths_C[i]>=nb).nonzero()[0]], bins=bins_len, normed=norm)#[Cpaths_C[i]>=6]
            H = np.concatenate((H, hist.reshape((hist.shape[0], 1))), axis=1)
    Hpd = pd.DataFrame(np.array(H).transpose()).melt()
    sns.lineplot(x="variable", y="value", err_style='bars', data=Hpd, color='cadetblue')
    # plt.xticks(np.arange(bin+1), np.arange(0, np.max(bins_len), 1+np.max(bins_len) / bin).astype(int))

    for i in range(len(Mpaths_BCP)):
        hist, bins = np.histogram(Mpaths_BCP[i][np.asarray(Mpaths_C[i]>=nb).nonzero()[0]], bins=bins_len, normed=norm)#[Mpaths_C[i]>=6]
        if i == 0:
            H = hist.reshape((hist.shape[0], 1))
        else:
            H = np.concatenate((H, hist.reshape((hist.shape[0], 1))), axis=1)
    Hpd = pd.DataFrame(np.array(H).transpose()).melt()
    sns.lineplot(x="variable", y="value", err_style='bars', data=Hpd, color='indianred')
    plt.xticks(np.arange(bin+1) , np.arange(0, np.max(bins_len), np.max(bins_len) / bin).astype(int))

    # sns.distplot(Cpaths_V, kde=False, color='cadetblue')
    # sns.distplot(Mpaths_V, kde=False, color='indianred')
    plt.yscale('linear')
    plt.title('distribution of the Position of Aggregates '+condition)
    plt.legend(['control', 'deprived'])
    # plt.xlim([0, 60])
    sns.despine()


    ###
    norm = True
    plt.figure()
    sns.despine()
    sns.set_style(style='white')
    nb=14
    bin=np.arange(12)
    for i in range(len(Cpaths_C)):
        if i == 0:
            hist, bins_len = np.histogram(Cpaths_C[i][np.asarray(Cpaths_BCP[i] >= nb).nonzero()[0]], bins=bin,
                                          normed=norm)  # ,bins=bin[Cpaths_C[i]>=6]
            H = hist.reshape((hist.shape[0], 1))
        else:
            hist, bins = np.histogram(Cpaths_C[i][np.asarray(Cpaths_BCP[i] >= nb).nonzero()[0]], bins=bins_len,
                                      normed=norm)  # [Cpaths_C[i]>=6]
            H = np.concatenate((H, hist.reshape((hist.shape[0], 1))), axis=1)
    Hpd = pd.DataFrame(np.array(H).transpose()).melt()
    sns.lineplot(x="variable", y="value", err_style='bars', data=Hpd, color='cadetblue')
    plt.xticks(np.arange(np.max(bin)+1), np.arange(0, np.max(bins_len)).astype(int))

    for i in range(len(Mpaths_BCP)):
        hist, bins = np.histogram(Mpaths_C[i][np.asarray(Mpaths_BCP[i] >= nb).nonzero()[0]], bins=bins_len,
                                  normed=norm)  # [Mpaths_C[i]>=6]
        if i == 0:
            H = hist.reshape((hist.shape[0], 1))
        else:
            H = np.concatenate((H, hist.reshape((hist.shape[0], 1))), axis=1)
    Hpd = pd.DataFrame(np.array(H).transpose()).melt()
    sns.lineplot(x="variable", y="value", err_style='bars', data=Hpd, color='indianred')
    # plt.xticks(np.arange(bin+1) , np.arange(0, np.max(bins_len), np.max(bins_len) / bin).astype(int))

    # sns.distplot(Cpaths_V, kde=False, color='cadetblue')
    # sns.distplot(Mpaths_V, kde=False, color='indianred')
    plt.yscale('linear')
    plt.title('Nb of loops per aggregate at the surface at ' + condition)
    plt.legend(['control', 'deprived'])
    # plt.xlim([0, 60])
    sns.despine()


    ###
    norm=False
    bin = 20
    plt.figure()
    sns.despine()
    sns.set_style(style='white')

    for i in range(len(Cpaths_V)):
        if i == 0:
            hist, bins_len = np.histogram(Cpaths_V[i], bins=bin, normed=norm)#,bins=bin
            H = hist.reshape((hist.shape[0], 1))
        else:
            hist, bins = np.histogram(Cpaths_V[i], bins=bins_len, normed=norm)
            H = np.concatenate((H, hist.reshape((hist.shape[0], 1))), axis=1)
    Hpd = pd.DataFrame(np.array(H).transpose()).melt()
    sns.lineplot(x="variable", y="value", err_style='bars', data=Hpd)
    plt.xticks(np.arange(bin), np.arange(0, np.max(bins_len), np.max(bins_len) / bin).astype(int))

    for i in range(len(Mpaths_V)):
        hist, bins = np.histogram(Mpaths_V[i], bins=bins_len, normed=norm)
        if i == 0:
            H = hist.reshape((hist.shape[0], 1))
        else:
            H = np.concatenate((H, hist.reshape((hist.shape[0], 1))), axis=1)
    Hpd = pd.DataFrame(np.array(H).transpose()).melt()
    sns.lineplot(x="variable", y="value", err_style='bars', data=Hpd)
    # plt.xticks(np.arange(bin), np.arange(0, np.max(bins_len), np.max(bins_len) / bin).astype(int))

    # sns.distplot(Cpaths_V, kde=False, color='cadetblue')
    # sns.distplot(Mpaths_V, kde=False, color='indianred')
    plt.yscale('log')
    plt.title('distribution of the vertices passage values using random Walk')
    plt.legend(['control', 'deprived'])
    # plt.xlim([0, 60])
    sns.despine()

    ###
    norm=True
    bin=np.arange(12)
    plt.figure()
    sns.despine()
    sns.set_style(style='white')
    for i in range(len(Cpaths_C)):
        if i == 0:
            hist, bins_len = np.histogram(Cpaths_C[i][Cpaths_C[i]>0], bins=bin, normed=norm)#, bins=bin
            H = hist.reshape((hist.shape[0], 1))
        else:
            hist, bins_len = np.histogram(Cpaths_C[i][Cpaths_C[i]>0], bins=bins_len, normed=norm)
            H = np.concatenate((H, hist.reshape((hist.shape[0], 1))), axis=1)
    Hpd = pd.DataFrame(np.array(H).transpose()).melt()
    sns.lineplot(x="variable", y="value", err_style='bars', data=Hpd, color='cadetblue')
    # plt.xticks(np.arange(bin), np.arange(0, np.max(bins_len), np.max(bins_len) / bin).astype(int))

    for i in range(len(Mpaths_C)):
        hist, bins = np.histogram(Mpaths_C[i][Mpaths_C[i]>0], bins=bins_len, normed=norm)
        if i == 0:
            H = hist.reshape((hist.shape[0], 1))
        else:
            H = np.concatenate((H, hist.reshape((hist.shape[0], 1))), axis=1)
    Hpd = pd.DataFrame(np.array(H).transpose()).melt()
    sns.lineplot(x="variable", y="value", err_style='bars', data=Hpd, color='indianred')
    plt.xticks(np.arange(np.max(bin)), np.arange(np.max(bins_len+1)).astype(int))

    # sns.distplot(Cpaths_C[Cpaths_C!=0], bins=np.arange(12), kde=False, color='cadetblue')
    # sns.distplot(Mpaths_C[Mpaths_C!=0], bins=np.arange(12), kde=False, color='indianred')
    plt.yscale('linear')
    plt.title('distribution of the number of loops per aggregate')
    plt.yscale('linear')
    plt.legend(['control', 'deprived'])
    # plt.xlim([0, 12])
    sns.despine()


    ###
    bin = 30
    plt.figure()
    sns.despine()
    sns.set_style(style='white')
    for i in range(len(Cpath_NV)):

        if i == 0:
            hist, bins_len = np.histogram(Cpath_NV[i], normed=norm)  # ,bins=bin
            H = hist.reshape((hist.shape[0], 1))
        else:
            hist, bins = np.histogram(Cpath_NV[i], bins=bins_len, normed=norm)
            H = np.concatenate((H, hist.reshape((hist.shape[0], 1))), axis=1)
    Hpd = pd.DataFrame(np.array(H).transpose()).melt()
    sns.lineplot(x="variable", y="value", err_style='bars', data=Hpd)
    plt.xticks(np.arange(bin), np.arange(0, np.max(bins_len), np.max(bins_len) / bin).astype(int))

    for i in range(len(Mpath_NV)):
        hist, bins = np.histogram(Mpath_NV[i], bins=bins_len, normed=norm)
        if i == 0:
            H = hist.reshape((hist.shape[0], 1))
        else:
            H = np.concatenate((H, hist.reshape((hist.shape[0], 1))), axis=1)
    Hpd = pd.DataFrame(np.array(H).transpose()).melt()
    sns.lineplot(x="variable", y="value", err_style='bars', data=Hpd)
    # plt.xticks(np.arange(bin), np.arange(0, np.max(bins_len), np.max(bins_len) / bin).astype(int))
    # sns.distplot(Cpath_NV, bins=50, kde=False, color='cadetblue')
    # sns.distplot(Mpath_NV, bins=50, kde=False, color='indianred')
    plt.yscale('linear')
    plt.title('distribution of the number of aggregates')
    sns.despine()
    plt.legend(['control' ,'deprived'])
    # plt.yscale('log')

    ### plot depth vs nb loops per cluster
    for i in range(len(Cpaths_C)):
        if i == 0:
            C_C = Cpaths_C[i]
        else:
            C_C = np.concatenate((C_C, Cpaths_C[i][Cpaths_C[i]>0]))
    for i in range(len(Mpaths_C)):
        if i == 0:
            M_C = Mpaths_C[i]
        else:
            M_C = np.concatenate((M_C, Mpaths_C[i][Mpaths_C[i]>0]))
    for i in range(len(Mpaths_BCP)):
        if i == 0:
            M_BCP = Mpaths_BCP[i]
        else:
            M_BCP = np.concatenate((M_BCP, Mpaths_BCP[i][Mpaths_C[i]>0]))
    for i in range(len(Cpaths_BCP)):
        if i == 0:
            C_BCP = Cpaths_BCP[i]
        else:
            C_BCP = np.concatenate((C_BCP, Cpaths_BCP[i][Cpaths_C[i]>0]))
    data = {'depth': np.concatenate((C_BCP, M_BCP), axis=0), 'path': np.concatenate((C_C, M_C), axis=0),
            'condition': np.concatenate((np.zeros(C_C.shape[0]), np.ones(M_C.shape[0])), axis=0)}
    df = pd.DataFrame(data)
    control = df.loc[df.condition == 0.0]
    mutant = df.loc[df.condition == 1.0]
    # sns.set_style(style='white')
    # sns.despine()
    g = sns.JointGrid('depth', 'path', data=df)
    # plt.figure()
    sns.set_style(style='white')
    sns.despine()
    ax = sns.kdeplot(mutant.depth, mutant.path, cmap="Reds",
                     shade=False, shade_lowest=False, ax=g.ax_joint, bw=1)
    sns.distplot(mutant.depth, color="r", ax=g.ax_marg_x, bins=50)
    ax = sns.distplot(mutant.path, kde=False,bins=np.arange(12), color="r", ax=g.ax_marg_y, vertical=True)
    ax.set_yscale('linear')
    # sns.scatterplot(mutant.depth,mutant.path,size=0.1, alpha=0.001,color="r")
    # sns.distplot(Dm, kde=True, hist=False, color="r", ax=g.ax_marg_x)
    # sns.distplot(Pm, kde=True, hist=False, color="r", ax=g.ax_marg_y, vertical=True)
    ax = sns.kdeplot(control.depth, control.path, cmap="Blues",
                     shade=False, shade_lowest=False, ax=g.ax_joint, bw=1)
    sns.distplot(control.depth, color="b", ax=g.ax_marg_x, bins=50)
    ax = sns.distplot(control.path, kde=False, bins=np.arange(12), color="b", ax=g.ax_marg_y, vertical=True)
    g.ax_marg_y.set_xscale('log')
    g.ax_marg_x.set_yscale('linear')
    # g.ax_marg_x.set_ylim(0, 10)
    g.ax_marg_y.set_ylim(0, 7)
    g.set_axis_labels('depth', 'Nb Loops per aggregate')
    # sns.scatterplot(control.depth,control.path,size=0.1, alpha=0.0001,color="b")
    # sns.distplot(Dc, kde=True, hist=False, color="b", ax=g.ax_marg_x)
    # sns.distplot(Pc, kde=True, hist=False, color="b", ax=g.ax_marg_y, vertical=True)
    sns.set_style(style='white')
    sns.despine()




    ## load data
    condition='barrels'
    controlsNbloops=np.load(work_dir + '/'+cluster_type+'_'+'controlsNbloops_all_' + condition + '_Extended_Meth.npy')
    mutantsNbloops=np.load(work_dir + '/'+cluster_type+'_'+'mutantsNbloops_all_' + condition + '_Extended_Meth.npy')
    controlAggloCycles=np.load(work_dir + '/'+cluster_type+'_'+'controlAggloCycles_all_' + condition + '_Extended_Meth.npy')
    mutantAggloCycles=np.load(work_dir + '/'+cluster_type+'_'+'mutantAggloCycles_all_' + condition + '_Extended_Meth.npy')
    controlsCyclesperAggregate=np.load(work_dir + '/'+cluster_type+'_'+'controlsCyclesperAggregate_all_' + condition + '_Extended_Meth.npy')
    mutantsCyclesperAggregate=np.load(work_dir + '/'+cluster_type+'_'+'mutantsCyclesperAggregate_all_' + condition + '_Extended_Meth.npy')

    from scipy.stats import ttest_ind
    st, pval = ttest_ind(controlsNbloops, mutantsNbloops)
    print(pval)
    st, pval = ttest_ind(controlAggloCycles, mutantAggloCycles)
    print(pval)


    ## PLOT DISTRIBUTION nB OF CYCLES PER AGGREGATE
    step=10
    max=200
    normed=False
    for a, m in enumerate(controlsCyclesperAggregate):
        if a==0:
            hist,bins = np.histogram(np.array(m),bins = np.arange(0, max, step), normed=normed)
            C=hist.reshape((hist.shape[0], 1))
        else:
            hist, bins = np.histogram(np.array(m), bins=np.arange(0, max, step), normed=normed)
            C=np.concatenate((C, hist.reshape((hist.shape[0], 1))), axis=1)

    for a, m in enumerate(mutantsCyclesperAggregate[1:]):
        if a == 0:
            hist, bins = np.histogram(np.array(m), bins=np.arange(0, max, step), normed=normed)
            M = hist.reshape((hist.shape[0], 1))
        else:
            hist, bins = np.histogram(np.array(m), bins=np.arange(0, max, step), normed=normed)
            M = np.concatenate((M, hist.reshape((hist.shape[0], 1))), axis=1)
    #
    # M = M.reshape((M.shape[0], 1))
    # C = C.reshape((C.shape[0], 1))
    # data=[C, M]

    # data=pd.DataFrame(np.array(data).transpose()).melt()
    C=pd.DataFrame(np.array(C).transpose()).melt()
    M=pd.DataFrame(np.array(M).transpose()).melt()


    plt.figure()
    import pandas as pd
    import seaborn as sns

    sns.set_style(style='white')
    sns.lineplot(x="variable", y="value", ci='sd', data=C)#, y="normalized count"
    sns.lineplot(x="variable", y="value", ci='sd', data=M)#, err_style="bars"

    sns.despine()

    plt.legend(['controls', 'mutants'])
    plt.title('Nb of loops per aggregate in ' + condition, size='x-large')
    plt.yscale('log')
    plt.xlabel("number of loops per aggregate")
    plt.ylabel("normalized count")
    plt.xticks(range(np.arange(0, max, step).shape[0]),np.arange(0, max, step))
    plt.tight_layout()





    ## PLOT BOXPLOT NB LOOPS NB AGGREGATES
    plt.figure()
    sns.despine()
    sns.set_style(style='white')


    box1 = plt.boxplot(controlsNbloops, positions=[1], patch_artist=True, widths=2,
                       showfliers=False, showmeans=True,
                       autorange=True, meanline=True)
    box2 = plt.boxplot(mutantsNbloops[1:], positions=[4], patch_artist=True, widths=2,
                       showfliers=False, showmeans=True,
                       autorange=True, meanline=True)
    for patch in box1['boxes']:
        patch.set_facecolor(colors[0])
    for patch in box2['boxes']:
        patch.set_facecolor(colors[1])


    plt.yticks(size='x-large')
    plt.xticks(np.arange(0, 7), np.arange(3, 10), size='x-large')
    # plt.legend(['controls', 'mutants'])

    # plt.xlabel('cycle size in BP')
    plt.xticks([1, 4], ['controls', 'mutants'], size='x-large')
    plt.ylabel('count', size='x-large')
    plt.title('Nb of loops in '+condition+' per condition', size='x-large')
    plt.xlim([-2, 8])
    # plt.ylim([0, 600])
    plt.tight_layout()



    plt.figure()

    sns.set_style(style='white')


    box1 = plt.boxplot(controlAggloCycles, positions=[1], patch_artist=True, widths=2,
                       showfliers=False, showmeans=True,
                       autorange=True, meanline=True)
    box2 = plt.boxplot(mutantAggloCycles[1:], positions=[4], patch_artist=True, widths=2,
                       showfliers=False, showmeans=True,
                       autorange=True, meanline=True)
    for patch in box1['boxes']:
        patch.set_facecolor(colors[0])
    for patch in box2['boxes']:
        patch.set_facecolor(colors[1])

    sns.despine()
    plt.yticks(size='x-large')
    plt.xticks(np.arange(0, 7), np.arange(3, 10), size='x-large')
    # plt.legend(['controls', 'mutants'])

    # plt.xlabel('cycle size in BP')
    plt.xticks([1, 4], ['controls', 'mutants'], size='x-large')
    plt.ylabel('count', size='x-large')
    plt.title('Nb of aggregates in '+condition+' per condition', size='x-large')
    plt.xlim([-2, 8])
    # plt.ylim([0, 60000])
    plt.tight_layout()

    # import matplotlib.patches as mpatches
    # layerNb=['l1', 'l2/3', 'l4', 'l5', 'l6a', 'l6b']
    # plt.figure()
    # labels=[]
    # for i, l in enumerate(layers):
    #     if i>0:
    #         if i<5:
    #             print(i, l, layerNb[i])
    #             # box2 = plt.violinplot(cyclesLength44, positions=np.arange(len(cyclesLength)) + 0.85)
    #             box3 = plt.violinplot(cyclesLength30[i], positions=np.arange(len(cyclesLength30[i])) +0.6+ (0.15*i))
    #             plt.xticks(np.arange(1, 8, 1), np.arange(3, 10, 1), size='large')
    #             plt.yticks(size='large')
    #             plt.title('layer'+layerNb[i])
    #             col=box3['bodies'][0].get_facecolor().flatten()
    #             labels.append((mpatches.Patch(color=col), layerNb[i]))
    #             # for pc in box3['bodies']:
    #             #     pc.set_facecolor(colors[i])
    #             #     pc.set_edgecolor('black')
    #             #     pc.set_alpha(0.3)
    # plt.legend(*zip(*labels), loc=2)
    # plt.ylabel('cycle length')
    # plt.xlabel('cycle bp')

    # plt.figure()
    # box2 = plt.boxplot(cyclesLength44, positions=np.arange(len(cyclesLength))+0.85, patch_artist=True, widths=0.3,
    #                    showfliers=True, showmeans=True, autorange=True, meanline=True)
    # box3 = plt.boxplot(cyclesLength30, positions=np.arange(len(cyclesLength))+1.15, patch_artist=True, widths=0.3,
    #                    showfliers=True, showmeans=True, autorange=True, meanline=True)
    #
    # plt.xticks(np.arange(1.15, 8.15, 1), np.arange(3, 10, 1))

    # plt.figure()
    # box2 = plt.violinplot(cyclesLength44, positions=np.arange(len(cyclesLength)) + 0.85)
    # box3 = plt.violinplot(cyclesLength30, positions=np.arange(len(cyclesLength)) + 1.15)
    # plt.xticks(np.arange(1, 8, 1), np.arange(3, 10, 1))
    #
    # plt.figure()
    # positions44 = np.arange(len(cyclesLength44)) + 0.85
    # positions30 = np.arange(len(cyclesLength30)) + 1.15
    # for i,p in enumerate(positions):
    #     p2=positions30[i]
    #     plt.plot(np.ones(len(cyclesLength44[i]))*p, cyclesLength44[i], alpha=0.3, color='k')
    #     plt.plot(np.ones(len(cyclesLength30[i])) * p2, cyclesLength30[i], alpha=0.3, color='k')

    # work_dir = '/data_SSD_2to/whiskers_graphs/30R'
    # with open(work_dir + '/cyclesLength30.p', 'wb') as fp:
    #     pickle.dump(cyclesLength30, fp, protocol=pickle.HIGHEST_PROTOCOL)
    # work_dir = '/data_SSD_2to/whiskers_graphs/44R'
    # with open(work_dir + '/cyclesLength44.p', 'wb') as fp:
    #     pickle.dump(cyclesLength44, fp, protocol=pickle.HIGHEST_PROTOCOL)



    ## GET DISCONNECTED SUB COMPONENT
    # work_dir = '/data_SSD_2to/whiskers_graphs/30R'
    import pickle

    with open(work_dir + '/' + cont + '/cyclesEPVect_'+condition+'.p', 'wb') as fp:
        pickle.dump(cyclesEPVect, fp, protocol=pickle.HIGHEST_PROTOCOL)

    with open(work_dir + '/' + cont + '/evect_tot_'+condition+'.p', 'wb') as fp:
        pickle.dump(evect_tot, fp, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # with open(work_dir + '/cyclesEPVect_barrels.p', 'rb') as fp:
    #     cyclesEPVect = pickle.load(fp)
    #
    # with open(work_dir + '/evect_tot_barrels.p', 'rb') as fp:
    #     evect_tot = pickle.load(fp)

    g_i=gl[0]
    g_s=g_i.sub_graph(edge_filter=evect_tot)
    comp, hist = gtt.label_components(g_s.base)
    print(hist)
    compf = comp.a

    artery = from_e_prop2_vprop(g_i, 'artery')
    vein = from_e_prop2_vprop(g_i, 'vein')
    print(np.sum(artery))
    print(np.sum(vein))




    # u, c = np.unique(compf, return_counts=True)
    # medium_comp = np.asarray(c >= 8).nonzero()[0]
    #
    # r = u[random.choice(medium_comp)]
    # print('comp number : ', r)
    # vf = np.zeros(g_s.n_vertices)
    # vf[np.asarray(compf == r).nonzero()[0]] = 1
    # print('containing ', np.sum(vf), ' vertices')
    # gf = g_s.sub_graph(vertex_filter=vf)
    # e = gf.n_edges
    # v = gf.n_vertices
    # print(gf)
    # p3d.plot_graph_mesh(gf)


    # pos = gtd.sfdp_layout(gf.base)
    # gtd.graph_draw(gf.base, pos=pos, output="/data_SSD_2to/whiskers_graphs/teststuckedcycles.pdf")
    from ClearMap.DiffusionPenetratingArteriesCortex import rand_cmap
    bn = np.zeros(compf.shape)
    for i in np.unique(compf):
        # print(i)
        bn[np.where(compf == i)] = np.where(np.unique(compf) == i)
    new_cmap, randRGBcolors = rand_cmap(len(np.unique(bn)), type='bright', first_color_black=False,
                                        last_color_black=False, verbose=True)

    artery=from_e_prop2_vprop(g_i, 'artery')
    vein=from_e_prop2_vprop(g_i, 'vein')

    n = len(g_i.vertices)
    colorval = np.zeros((n, 3));
    for i in range(compf.shape[0]-1):
        colorval[i] = randRGBcolors[int(bn[i])]
        if hist[compf[i]]==1:
            colorval[i]=[0, 0, 0]
        if artery[i]:
            colorval[i] = [1, 0, 0]
        if vein[i]:
            colorval[i] = [0, 0, 1]
    # colorval = getColorMap_from_vertex_prop(g2plot.vertex_property('artterr'))

    p = p3d.plot_graph_mesh(g_i, vertex_colors=colorval, n_tube_points=3);


    backbon_vf=np.zeros(g_i.n_vertices)
    n = len(g_i.vertices)
    colorval = np.zeros((n, 3));
    for i in range(compf.shape[0] - 1):
        if hist[compf[i]] == 1:
            backbon_vf[i] = 1
            colorval[i] = [1, 1, 1]
        if artery[i]:
            backbon_vf[i] = 1
            colorval[i] = [1, 0, 0]
        if vein[i]:
            backbon_vf[i] = 1
            colorval[i] = [0, 0, 1]
    backbon = g_i.sub_graph(vertex_filter=backbon_vf)
    backbon=backbon.largest_component()
    colorval = np.zeros((n, 3));
    artery=from_e_prop2_vprop(backbon, 'artery')
    vein=from_e_prop2_vprop(backbon, 'vein')
    for i in range(backbon.n_vertices):
        if artery[i]:
            colorval[i] = [1, 0, 0]
        elif vein[i]:
            colorval[i] = [0, 0, 1]
        else:
            colorval[i] = [0, 0, 0]
    # p = p3d.plot_graph_mesh(backbon,vertex_colors=colorval, n_tube_points=3);
    backbon.add_vertex_property('plot_color',colorval)
    pos = gtd.sfdp_layout(backbon.base)
    gtd.graph_draw(backbon.base, pos=pos, vertex_fill_color=backbon.base.vertex_properties['plot_color'],  output=work_dir+"/testBackbon_color.pdf")



    colorval2=np.concatenate((colorval, np.ones((colorval.shape[0], 1))), axis=1)
    g_i.add_vertex_property('plot_color',colorval2)
    # pos = gtd.sfdp_layout(g_i.base)
    # gtd.graph_draw(g_i.base, pos=pos, vertex_fill_color=g_i.base.vertex_properties['plot_color'],  output="/data_SSD_2to/whiskers_graphs/teststuckedcycles_color.pdf")
    # gtd.graph_draw(g_i.base, pos=module.base.vertex_properties['pos'],
    #                    vertex_fill_color=g_i.base.vertex_properties['plot_color'],
    #                    output="/data_SSD_2to/whiskers_graphs/teststuckedcycles_color_embedded_diff_map_pos" + str(
    #                        id) + ".pdf")

    ## GRAPH EMBEDDED CONSTRUCTION



    u, c = np.unique(colorval, return_counts=True, axis=0)#diffusion_through_penetrating_arteries_vector
    conn = g_i.edge_connectivity()
    edges_all=[[colorval[conn[i, 0]], colorval[conn[i,1]]] for i in range(conn.shape[0])]
    edges_all=np.array(edges_all)
    g = ggt.Graph(n_vertices=u.shape[0], directed=False)
    # radii = np.zeros((0, 1), dtype=int)
    # cluster_size = np.zeros(g.n_vertices)
    print(g)
    histCycles=[]
    for i, uc in enumerate(u):
        vf=np.asarray(colorval==uc)
        ef=from_v_prop2_eprop(g_i, vf)
        ef = np.asarray([ef[i].all() for i in range(ef.shape[0])]).nonzero()[0]
        # cluster_size[uc] = c[i]
        temp_hist=[]
        if ef.shape[0]!=0:
            for j in range(len(simpla_basis)):
                temp_hist.append(np.unique(cyclesEPVect[0][ef,j]).tolist())
        else:
            for j in range(len(simpla_basis)):
                temp_hist.append(np.unique(np.zeros(5)).tolist())
        histCycles.append(temp_hist)
    histCycles=np.array(histCycles)

    ## plot une sub graph cycle cluster to see if it matches with its corresponding histogram
    # u2plt=u[0]
    # vf=np.asarray(colorval==u2plt)
    # vf=np.asarray([vf[i].all() for i in range(vf.shape[0])])
    #
    # g2plt=g_i.sub_graph(vertex_filter=vf)
    # p3d.plot_graph_mesh(g2plt)

    ## BUILDING THE EMBEDDED GRAPH
    inter_edges = np.asarray([edges_all[i, 0]!=edges_all[i, 1] for i in range(edges_all.shape[0])])#.nonzero()[0]
    inter_edges_t=np.asarray([inter_edges[i].all() for i in range(inter_edges.shape[0])]).nonzero()[0]

    inter_edges_t_f = np.asarray([[np.asarray(u==edges_all[ie][0]).nonzero()[0][-1], np.asarray(u==edges_all[ie][1]).nonzero()[0][-1]] for ie in inter_edges_t])

    eu, ec=np.unique(inter_edges_t_f, return_counts=True, axis=0)
    g.add_edge(eu)
    g.add_vertex_property('plot_color', u)
    print(g)


    pos = gtd.sfdp_layout(g.base)
    gtd.graph_draw(g.base, pos=pos, vertex_fill_color=g.base.vertex_properties['plot_color'],  output="/data_SSD_2to/whiskers_graphs/teststuckedcycles_color_embedded2.pdf")

    ## OVERLAP BOW HISTOGRAM

    histVect=[]
    H=[]
    for i in range(len(histCycles)-1):
        hi=histCycles[i]
        temp=0
        for h in hi:
            hu=np.unique(np.asarray(h)).shape[0]-1
            H.append(hu)
            temp=temp+hu
        for j,h in enumerate(hi):
            heatmap[temp, j]=heatmap[temp, j]+len(h)
        histVect.append(temp)

    hu, hc=np.unique(histVect, return_counts=True)
    HU.append(hu)
    HC.append(hc)
    # plt.figure()
    # plt.bar(hu, hc)
    np.save(work_dir + '/' + cont + '/histCycle_'+condition+'.npy', histVect)
    np.save(work_dir + '/' + cont + '/heatmap_'+condition+'.npy', heatmap)

    import matplotlib as mpl
    # plt.imshow(heatmap, norm = mpl.colors.LogNorm())



    ## plot giant loop:

    giantloop=u[np.asarray(c==np.max(c)).nonzero()[0]]
    vf_GL=np.asarray(colorval==giantloop)
    vf_GL=np.asarray([vf_GL[i].all() for i in range(vf_GL.shape[0])])
    # giso=ggt.load('/data_SSD_2to/whiskers_graphs/data_graph_44R.gt')
    # brainnb='whiskers_graphs/39L'

    # g, barrels_filter = extract_AnnotatedRegion(giso, (54,9))
    art_filter = from_e_prop2_vprop(g_i, 'artery')

    vertex_filter=np.logical_or(vf_GL, art_filter)

    GL_art_graph= g_i.sub_graph(vertex_filter=vertex_filter)

    gl4, barrels_l_4_filter = extract_AnnotatedRegion(GL_art_graph, (57, 10))
    gl23, barrels_l_23_filter = extract_AnnotatedRegion(GL_art_graph, (56, 10))
    gl5, barrels_l_5_filter = extract_AnnotatedRegion(GL_art_graph, (58, 10))

    l23=[186/255, 252/255,3/255, 1]
    l4=[252/255, 186/255,3/255, 1]
    l5=[3/255, 252/255, 232/255, 1]
    artery_color = np.array([[1, 0, 0, 1], [252/255, 186/255,3/255, 1]]).astype('float')
    vertexcolors=artery_color[np.asarray(barrels_l_4_filter, dtype=int)]

    for i in np.asarray(barrels_l_23_filter, dtype=int).nonzero()[0]:
        vertexcolors[i]=l23

    for i in np.asarray(barrels_l_5_filter, dtype=int).nonzero()[0]:
        vertexcolors[i] = l5

    print(GL_art_graph)

    p3d.plot_graph_mesh(GL_art_graph, vertex_colors=vertexcolors)


    ## get orientation loopd

    backbone = u==[1, 1, 1, 1]
    backbone = np.asarray([colorval2[i]==[1,1,1,1] for i in range(colorval2.shape[0])])
    backbone=np.array([backbone[i].all()for i in range(backbone.shape[0])])
    BB_graph= g_i.sub_graph(vertex_filter=backbone)#, np.logical_not(art_filter)))

    r, p, l=getRadPlanOrienttaion(BB_graph)

    # rad=np.sum(l[(r/(r+p))>0.5])
    # plan=np.sum(l[(p/(r+p))>0.6])
    rad=np.sum((r/(r+p))>0.5)
    plan=np.sum((p/(r+p))>0.6)
    print(rad)
    print(plan)
    print(rad/plan)

    vess_ori_backbone=getVesselOrientation(BB_graph, graph_i)

    print('orientation backbone')
    print(get_nb_parrallel_vessels(vess_ori_backbone) / BB_graph.n_edges)
    print(get_nb_radial_vessels(vess_ori_backbone) / BB_graph.n_edges)

    loops=np.logical_not(backbone)
    loop_graph= g_i.sub_graph(vertex_filter=loops)#np.logical_and(loops, np.logical_not(art_filter)))

    print('orientation loops')
    r, p, l=getRadPlanOrienttaion(loop_graph)

    # rad=np.sum(l[(r/(r+p))>0.5])
    # plan=np.sum(l[(p/(r+p))>0.6])
    rad=np.sum((r/(r+p))>0.5)
    plan=np.sum((p/(r+p))>0.6)
    print(rad)
    print(plan)
    print(rad/plan)

    print('orientation gra[uj')
    r, p, l=getRadPlanOrienttaion(g_i)
    # rad=np.sum(l[(r/(r+p))>0.5])
    # plan=np.sum(l[(p/(r+p))>0.6])
    rad=np.sum((r/(r+p))>0.5)
    plan=np.sum((p/(r+p))>0.6)
    print(rad)
    print(plan)
    print(rad/plan)
    #
    # vess_ori_loops=getVesselOrientation(loop_graph, graph_i)
    #
    # print('orientation loops')
    # print(get_nb_parrallel_vessels(vess_ori_loops) / loop_graph.n_edges)
    # print(get_nb_radial_vessels(vess_ori_loops) / loop_graph.n_edges)

    art_graph= g_i.sub_graph(vertex_filter=art_filter)
    p3d.plot_graph_mesh(art_graph)
    # vess_ori_art=getVesselOrientation(art_graph, graph_i)

    distance_from_suface = art_graph.vertex_property('distance_to_surface')
    plt.figure()
    plt.hist(distance_from_suface)
    vf = distance_from_suface > 5
    g = art_graph.sub_graph(vertex_filter=vf)

    r, p, l=getRadPlanOrienttaion(g)
    #
    artery_color = np.array([[1, 0, 0, 1], [0,1,0, 1]]).astype('float')
    edgevolor=artery_color[np.asarray((r/(r+p))>0.7, dtype=int)]
    edgevolor[np.asarray((p/(r+p))>0.7, dtype=bool)]=[0,0,1,1]

    p3d.plot_graph_mesh(g, edge_colors=edgevolor)

    rad=np.sum((r/(r+p))>0.5)
    plan=np.sum((p/(r+p))>0.6)
    print(rad)
    print(plan)
    print(rad/plan)


    print(np.mean(r),np.mean(p))

    print('orientation art')
    print(get_nb_parrallel_vessels(vess_ori_art) / art_graph.n_edges)
    print(get_nb_radial_vessels(vess_ori_art) / art_graph.n_edges)

    loops_ori=[]
    for i, uc in enumerate(u):

        vf = np.asarray(colorval == uc)
        vf=np.asarray([vf[i].all() for i in range(vf.shape[0])])
        print(i)
        if np.sum(vf)>=2:
            g_t=g_i.sub_graph(vertex_filter=vf)
            print(np.sum(vf), g_t)
            if g_t.n_edges>0:
                try:
                    r, p, l = getRadPlanOrienttaion(g_t)
                    print(r.shape, p.shape, l.shape)
                    rad = np.sum(l[(r / (r + p)) > 0.5])
                    plan = np.sum(l[(p / (r + p))> 0.6])
                    # print(rad)
                    # print(plan)
                    # print(rad / plan)
                    loops_ori.append(rad / plan)
                except:
                    print('!!!1')

    plt.figure()
    # loops_ori=np.array(loops_ori)
    loops_ori=loops_ori[~np.isnan(loops_ori)]
    loops_ori=loops_ori[~np.isinf(loops_ori)]
    plt.hist(loops_ori, bins=500)
    plt.xscale('log')




    ## heatmaps

    # work_dir = '/data_SSD_2to/191122Otof'
    # controls = ['2R' ,'3R', '5R', '8R']
    # mutants = ['1R', '7R', '6R', '4R']


    work_dir='/data_SSD_2to/whiskers_graphs/new_graphs'
    controls=['142L','158L','162L', '164L']
    mutants=['138L','141L', '163L', '165L']

    heatmap_mutants=np.zeros((6000, 7, len(mutants)))
    heatmap_controls=np.zeros((6000, 7, len(controls)))

    for i,cont in enumerate(mutants):
        h_temp=np.load(work_dir+'/'+cont+'/heatmap_auditory_mutants.npy')
        heatmap_mutants[:,:,i]=h_temp


    for i,cont in enumerate(controls):
        try:
            h_temp=np.load(work_dir+'/'+cont+'/heatmap_auditory.npy')
        except:
            h_temp = np.load(work_dir + '/' + cont + '/heatmap_auditory_controls.npy')
        heatmap_controls[:,:,i]=h_temp


    ## loops histograms
    import pandas as pd
    import seaborn as sns

    totNbLopps_c=np.sum(heatmap_controls, axis=0)
    print(totNbLopps_c.shape)

    totNbLopps_m=np.sum(heatmap_mutants, axis=0)
    print(totNbLopps_c.shape)

    plt.figure()

    sns.set_style(style='white')

    # df = pd.DataFrame(np.array(totNbLopps_c)[:,:].transpose()).melt()
    # p1=sns.lineplot(x="variable", y="value", err_style="bars",data=df)
    #
    #
    # df = pd.DataFrame(np.array(totNbLopps_m)[:,:].transpose()).melt()
    # p1=sns.lineplot(x="variable", y="value", err_style="bars",data=df)

    box1 = plt.boxplot(np.sum(heatmap_controls.reshape(42000, 4), axis=0),positions=[1], patch_artist=True, widths=2, showfliers=False, showmeans=True,
                       autorange=True, meanline=True)
    box2 = plt.boxplot(np.sum(heatmap_mutants.reshape(42000, 4), axis=0), positions=[4], patch_artist=True, widths=2, showfliers=False, showmeans=True,
                       autorange=True, meanline=True)
    for patch in box1['boxes']:
        patch.set_facecolor(colors[0])
    for patch in box2['boxes']:
        patch.set_facecolor(colors[1])


    sns.despine()
    plt.yticks(size='x-large')
    plt.xticks(np.arange(0,7), np.arange(3,10), size='x-large')
    # plt.legend(['controls', 'mutants'])

    # plt.xlabel('cycle size in BP')
    plt.xticks([1,4], ['controls', 'mutants'], size='x-large')
    plt.ylabel('count', size='x-large')
    plt.title('Nb of loops in Aud_p per condition',size='x-large')
    plt.tight_layout()
    ## agglomerated loops clusters histograms

    condition='auditory_mutants'
    for i,cont in enumerate(mutants):
        hv=np.load(work_dir + '/' + cont + '/evect_tot_'+condition+'.p')
        print(hv.shape)

    condition='auditory_mutants'
    for i,cont in enumerate(controls):
        hv=np.load(work_dir + '/' + cont + '/histCycle_' + condition + '.npy')
        print(hv.shape)


    plt.figure()

    df = pd.DataFrame(np.array(totNbLopps_c)[:,:].transpose()).melt()
    p1=sns.lineplot(x="variable", y="value", err_style="bars",data=df)


    df = pd.DataFrame(np.array(totNbLopps_m)[:,:].transpose()).melt()
    p1=sns.lineplot(x="variable", y="value", err_style="bars",data=df)

    plt.yticks(size='x-large')
    plt.xticks(np.arange(0,7), np.arange(3,10), size='x-large')
    plt.legend(['controls', 'mutants'])
    plt.tight_layout()
    plt.xlabel('cycle size in BP')



    for i,cont in enumerate(controls):
        print(cont)




    ## p values heatmaps


    from scipy import stats
    pcutoff = 0.05

    tvals, pvals = stats.ttest_ind(heatmap_controls, heatmap_mutants, axis = 2, equal_var = True);

    pi = np.isnan(pvals);
    pvals[pi] = 1.0;
    tvals[pi] = 0;

    pvals2 = pvals.copy();
    pvals2[pvals2 > pcutoff] = pcutoff;
    psign=np.sign(tvals)



    # pvals, psign = tTestVoxelization(vox_control_avg, vox_mutant_avg, pcutoff = None, signed = True);
    pvalscol = colorPValues(pvals2, psign, positive = [255,0,0], negative = [0,255,0])
    pv=pvalscol.astype('uint8')
    tuples=np.transpose(np.nonzero(pvalscol!=[0,0,0]))
    print(np.unique(np.reshape(pv, (42000, 3)), axis=0))

    col=np.zeros(pvalscol.shape[:2])
    print(col.shape)
    for i in range(col.shape[0]):
        for j in range(col.shape[1]):
            if (pv[i, j]==[4,0,0]).all():
                print('pos')
                col[i, j]=2
            elif (pv[i, j]==[8,0,0]).all():
                print('neg')
                col[i, j]=1

    # print(np.unique(np.reshape(pv, (42000, 3)), axis=0))
    flatui = ['black', 'indianred', "forestgreen"]

    # Draw heatmap with the custom palette color
    sns.heatmap(col[:42, :], cmap=sns.color_palette(flatui))
    # sns.heatmap(pvalscol)

    heatmap_mutants_avg=np.mean(heatmap_mutants, axis=2)
    heatmap_controls_avg=np.mean(heatmap_controls, axis=2)

    plt.figure()
    heatmap_mutants_avg=heatmap_mutants_avg/heatmap_mutants_avg.max()+np.ones(heatmap_mutants_avg.shape)*0.01
    heatmap_mutants_avg=heatmap_mutants_avg[1:80, :]#M
    sns.heatmap(heatmap_mutants_avg,norm=mpl.colors.LogNorm())

    plt.figure()
    heatmap_controls_avg=heatmap_controls_avg/heatmap_controls_avg.max()+np.ones(heatmap_controls_avg.shape)*0.01
    heatmap_controls_avg=heatmap_controls_avg[1:80, :]#M
    sns.heatmap(heatmap_controls_avg,norm=mpl.colors.LogNorm())


    plt.figure(3)
    plt.xticks(np.arange(0,7)+0.5, np.arange(3,10))
    plt.title('Otof -/- Barrels') #Primary Auditory
    plt.xlabel('cycle size in BP')
    plt.ylabel('number of agglomerated cycles')


    plt.figure(4)
    plt.xticks(np.arange(0,7)+0.5, np.arange(3,10))
    plt.title('Controls Barrels') #Primary Auditory
    plt.xlabel('cycle size in BP')
    plt.ylabel('number of agglomerated cycles')


    maxs=[]
    for i in range(len(HU)):
        maxs.append(np.max(np.asarray(HU[i])))
    M=np.max(np.asarray(maxs))
    print(M)

    ## heatmaps
    import seaborn as sns
    heatmap=heatmap/heatmap.max()+np.ones(heatmap.shape)*0.01
    heatmap_t=heatmap[1:80, :]#M
    sns.heatmap(heatmap_t,norm=mpl.colors.LogNorm())

    plt.xticks(np.arange(0,7), np.arange(3,10))
    plt.yticks(np.arange(0,80), np.arange(1,81))

    # mat=np.zeros((len(HU), M+1))
    mat=np.zeros((len(HU), M+1))

    for i in range(len(HU)):
        for k, j in enumerate(HU[i]):
            mat[i, j]=HC[i][k]

    import seaborn as sns
    ax=sns.boxplot(data=mat[1:])
    # ax=ax = sns.swarmplot(data=mat, color=".25")
    plt.xticks(np.arange(0,M+1), np.arange(1,M+2))


    u_ind, c_ind=np.unique(indices, return_counts=True)
    plt.figure()
    plt.hist(c_ind,bins=100)
    plt.yscale('log')

    plt.figure()
    for i in range(len(HU)):
        plt.scatter(np.asarray(range(M+1)), mat[i], c='k')



    ## Number of cycles clusters per region
    uni, com = np.unique(colorval2, return_counts=True, axis=0)#diffusion_through_penetrating_arteries_vector
    for e in reg_list.keys():
          reg_name=ano.find_name(e, key='order')
          if 'barrel' in reg_name:#'Primary auditory'#barrel
            print(reg_name)
            label_leveled = ano.convert_label(label, key='order', value='order', level=ano.find_level(e, key='order'))
            vertex_filter = label_leveled == e  # 54;
            vess_tree=g_i.sub_graph(vertex_filter=vertex_filter)
            V=[]
            nbcyV=[]
            for se in reg_list[e]:
                print(ano.find_name(se, key='order'))
                label_se = vess_tree.vertex_annotation();
                label_leveled_se = ano.convert_label(label_se, key='order', value='order',level=ano.find_level(se, key='order'))
                vertex_filter = label_leveled_se == se
                vess_tree_se=vess_tree.sub_graph(vertex_filter=vertex_filter)
                col=vess_tree_se.vertex_property('plot_color')
                uv=np.unique(col, axis=0)
                V.append(uv.shape[0])
                n=np.zeros((uv.shape[0],7))
                for l,co in enumerate(uv):
                    i = np.asarray(uni == co)
                    i=np.asarray([i[j].all() for j in range(i.shape[0])]).nonzero()[0][0]
                    print(i)
                    for k in range(7):
                        r=np.unique(np.asarray(histCycles[i,k])).shape[0]-1
                        print(i, k,r)
                        n[l,k]=r

                # nbcyV.append(np.mean(n, axis=0))
                nbcyV.append(np.mean(np.sum(n, axis=1)))

    nbcyV=np.asarray(nbcyV)

    plt.figure()
    for nbv in nbcyV:
        plt.plot(range(7), nbv/np.max(nbv), marker='o')
    plt.xticks(np.arange(6),['3', '4','5','6','7','8', '9'])
    plt.legend(['l1', 'l2/3','l4','l5','l6a','l6b'])

    ## avg number of cycles per cluster cycle
    plt.figure(3)
    b30R=np.array([6.02, 0.2, 6.65, 9.04, 5.6, 0.93])
    b44R=[65.5, 739.27, 1247.4, 1273.8, 419.3, 1.5]

    nbcyV_t=nbcyV+[0, 0, 0, 0, 0, 1]
    plt.plot(range(6), nbcyV_t,  marker='o')#/np.max(nbcyV)
    plt.plot(range(6),b30R,  marker='o')# /np.max(b30R)

    plt.xticks(np.arange(6),['l1', 'l2/3','l4','l5','l6a','l6b'])
    plt.legend(['44R', '30R'])
    plt.yscale('log')

    #nb of cluster cycle per layer
    plt.figure(4)
    b30R=np.array([1.9, 1.01, 6.9, 13.9, 8.9, 2.98])
    b44R=[1.96, 6.9, 3.9, 3.9, 11.9, 0.98]

    plt.plot(range(6), V,  marker='o')
    plt.plot(range(6), b30R,  marker='o')

    plt.xticks(np.arange(6),['l1', 'l2/3','l4','l5','l6a','l6b'])
    plt.legend(['44R', '30R'])

    ## betweenesss centrality edges removal Newman Griman algorithm implementation
    import graph_tool.centrality as gtc
    import graph_tool.topology as gtt
    import graph_tool.inference as gtii



    u, c = np.unique(indices, return_counts=True)

    medium_cluster = np.asarray(np.logical_and(c >= 600, c <= 1200)).nonzero()[0]
    r = u[random.choice(medium_cluster)]

    print('cluster number : ', r)
    vf = np.zeros(graph.n_vertices)
    vf[np.asarray(indices == r).nonzero()[0]] = 1
    print('containing ', np.sum(vf), ' vertices')
    g2plot = gss4.sub_graph(vertex_filter=vf)
    g2plot = g2plot.largest_component()

    g=g2plot.copy()

    modularities = []
    comp, hist = gtt.label_components(g2plot.base)
    n = 1
    Nb_mod = 0
    boolean = True
    while boolean:
        while np.unique(comp.a).shape[0] == Nb_mod + 1:
            vp, ep = gtc.betweenness(g2plot.base)
            ep = ep.a
            # plt.figure()
            # plt.hist(ep)
            # plt.yscale('log')
            # ep=np.around(ep, decimals=10)
            max = np.max(ep)
            ef = ep != max
            e2rm = np.asarray(ep == max).nonzero()[0]
            e2rm = e2rm.astype(int)
            print('Nb edge 2 delete ', np.sum(ep == max), e2rm)
            print('e2rm', e2rm)
            # g=g2plot
            i = 0
            if len(e2rm) < 30:
                for e in e2rm:
                    print(int(e) - i)
                    g2plot.remove_edge(g2plot.edge(int(e)))
                    i = i + 1
                    print(g2plot)
            else:
                g2plot.remove_edge(g2plot.edge(int(e2rm[0])))
                if modularity > 0.9:
                    boolean = False
            n = n + 1
            comp, hist = gtt.label_components(g2plot.base)
            print(g2plot, g)
        print(n)
        g.add_vertex_property('mod', comp.a)
        modularity = gtii.modularity(g.base, g.base.vp.mod)
        print('modularity', modularity)
        if modularities != []:
            if modularity >= modularities[-1]:
                modularities.append(modularity)
                Nb_mod = Nb_mod + 1
                print(modularities)
                print('nb clusters ', np.unique(comp.a).shape[0])
            else:
                print('modularity starts decreasing !')
                boolean = False
        else:
            modularities.append(modularity)
            Nb_mod = Nb_mod + 1
            print(modularities)
            print('nb clusters ', np.unique(comp.a).shape[0])

    compf, histf = gtt.label_components(g2plot.base)
    print(histf)
    compf = compf.a
    e = -1
    v = 0
    u, c = np.unique(compf, return_counts=True)
    medium_comp = np.asarray(c >= 6).nonzero()[0]
    while e < v:
        r = u[random.choice(medium_comp)]
        print('comp number : ', r)
        vf = np.zeros(g.n_vertices)
        vf[np.asarray(compf == r).nonzero()[0]] = 1
        print('containing ', np.sum(vf), ' vertices')
        gf = g.sub_graph(vertex_filter=vf)
        e = gf.n_edges
        v = gf.n_vertices
    print(gf)
    p3d.plot_graph_mesh(gf)

    vp, ep = gtc.betweenness(g.base)
    plt.figure()
    plt.hist(ep.get_array())
    plt.yscale('log')
    max = np.max(ep)
    ef = ep != max
    print('Nb edge 2 delete ', np.sum(ep == max))
    g = g.sub_graph(edge_filter=ef)
    print(g)