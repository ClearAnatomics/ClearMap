
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
import pandas as pd
import seaborn as sns

def extract_AnnotatedRegion(graph, region):
    order, level = region
    print(level, order, ano.find_name(order, key='id'))

    label = graph.vertex_annotation();
    label_leveled = ano.convert_label(label, key='id', value='id', level=level)
    vertex_filter = label_leveled == order;

    gss4 = graph.sub_graph(vertex_filter=vertex_filter)
    return gss4, vertex_filter


def removeAutoLoops(graph):
    connectivity=graph.edge_connectivity()
    autoloops=np.asarray((connectivity[:, 0]-connectivity[:,1])==0).nonzero()[0]
    print(autoloops)
    ef=np.ones(graph.n_edges)
    for edge in autoloops:
        ef[edge]=0
    g=graph.sub_graph(edge_filter=ef)
    return g

def f_min(X,p):
    plane_xyz = p[0:3]
    distance = (plane_xyz*X.T).sum(axis=1) + p[3]
    return distance / np.linalg.norm(plane_xyz)

def residuals(params, signal, X):
    return f_min(X, params)

from scipy.optimize import leastsq


def get_edges_from_vertex_filter(prev_graph,vertex_filter):
  connectivity=prev_graph.edge_connectivity()
  edges=np.logical_and(vertex_filter[connectivity[:,0]], vertex_filter[connectivity[:,1]])
  return(edges)

def cleandeg4nodes(graph):
    degrees_m=graph.vertex_degrees()
    deg4=np.asarray(degrees_m==4).nonzero()[0]
    graph_cleaned=graph
    conn = graph.edge_connectivity()
    e_g = np.array(graph.edge_geometry())
    for j, d in enumerate(deg4):
        ns=graph.vertex_neighbours(d)
        vf=np.zeros(graph.n_vertices)

        vf[ns]=1
        vf[d]=1
        ef = get_edges_from_vertex_filter(graph, vf)
        conn_f=conn[ef]

        gs=graph.sub_graph(vertex_filter=vf)

        es=np.array(gs.edge_geometry())
        cs=gs.edge_connectivity()
        ds = gs.vertex_degrees()

        d4 = np.asarray(ds == 4).nonzero()[0]
        if isinstance(d4, (list, tuple, np.ndarray)):
            d4=d4[0]
        pts=[]
        pts_index=[]
        for i,c in enumerate(cs):
            # id=np.asarray(c==d4).nonzero()[0][0]
            if c[0]==d4:
                pts.append(np.array(es[i])[1])
                pts_index.append(i)
                if i==0:
                    pts.append(np.array(es[i])[0])
                    pts_index.append(100)
            elif c[1]==d4:
                pts.append(np.array(es[i])[-2])
                pts_index.append(i)
                if i==0:
                    pts.append(np.array(es[i])[-1])
                    pts_index.append(100)
            else:
                print('...')

        XYZ=np.array(pts).transpose()
        p0=[0,0,1,np.mean(XYZ[2])]
        sol = leastsq(residuals, p0, args=(None, XYZ))[0]
        x=sol[0:3]
        new_error=(f_min(XYZ, sol) ** 2).sum()
        old_error=(f_min(XYZ, p0) ** 2).sum()

        if new_error<1:
            z=np.array([0, 0, 1])
            costheta=np.dot(x,z)/(np.linalg.norm(x)*np.linalg.norm(z))
            # print(np.dot(x,z), (np.linalg.norm(x)*np.linalg.norm(z)))
            if abs(np.arccos(costheta))<0.3:
                print(j)
                print("Solution: ", x / np.linalg.norm(x), sol[3])
                print("Old Error: ", old_error)
                print("New Error: ", new_error)

                dn4 = np.asarray(ds != 4).nonzero()[0]
                coord=gs.vertex_coordinates()

                pos=[]
                neg=[]
                pos_e=[]
                neg_e = []
                e_i = np.asarray(ef == 1).nonzero()[0]
                v_i=np.asarray(vf == 1).nonzero()[0]
                for i, co in enumerate(np.array(pts)):
                    if i != d4:
                        res=sol[0]*co[0]+sol[1]*co[1]+sol[2]*co[2]+sol[3]
                        if res<0:
                            # neg.append((v_i[i], d))
                            neg.append((i, d4))
                        if res>=0:
                            # pos.append((v_i[i], d))
                            pos.append((i, d4))


                for p in pos:
                    for i,cf in enumerate(conn_f):
                        if p[0] in cf:
                            print(i)
                            pos_e.append(e_i[i])
                            break

                for n in neg:
                    for i,cf in enumerate(conn_f):
                        if n[0] in cf:
                            print(i)
                            neg_e.append(e_i[i])
                            break

                graph_cleaned.remove_vertex(d)
                newpos_edge=[]
                new_conn=[]
                for i, p in enumerate(pos_e):
                    newpos_edge.append(e_g[p])
                    new_conn.append(pos[i][0])
                newpos_edge=np.array(newpos_edge).ravel()


                graph_cleaned.add_edge((new_conn[0],new_conn[1]))

                newpos_edge = []
                new_conn = []
                for i, p in enumerate(neg_e):
                    newpos_edge.append(e_g[p])
                    new_conn.append(neg[i][0])
                newpos_edge = np.array(newpos_edge).ravel()

                graph_cleaned.add_edge((new_conn[0], new_conn[1]))



def removeSpuriousBranches(gss ,rmin=1, length=5):
    radii = gss.vertex_radii()
    connectivity=gss.edge_connectivity()
    degrees_m = gss.vertex_degrees()
    deg1 = np.asarray(degrees_m <= 1).nonzero()[0]
    rad1 = np.asarray(radii <= rmin).nonzero()[0]
    lengths = gss.edge_geometry_lengths()
    # print(rad1.shape)
    # conn1=np.asarray(radii == 1).nonzero()[0]#conn[rad1]
    vertex2rm=[]
    for i in rad1:
        if i in deg1:
            # if lengths[i]<=length:
            vertex2rm.append(i)
        # if c[1] in deg1:
        #     if lengths[i]<=length:
        #         vertex2rm.append(i)
    # for i in deg1:
    #     print(i,'/',deg1.shape[0])
    #     es=np.asarray(graph.edge_connectivity()==i).nonzero()[0]
    #     # print(es)
    #     for e in es:
    #         if lengths[e]<=length:
    #             vertex2rm.append(i)
    #             break
    
    deg1 = gss.vertex_degrees() == 1
    d1edges = np.asarray(np.logical_or(deg1[connectivity[:, 0]], deg1[connectivity[:, 1]])).nonzero()[0]
    for i in d1edges:
        if lengths[i]<=length:
            vertex2rm.append(i)
                
        
    vertex2rm=np.array(vertex2rm)
    print(vertex2rm)
    print(vertex2rm.shape)
    ef=np.ones(gss.n_vertices)
    ef[vertex2rm]=0
    graph=gss.sub_graph(vertex_filter=ef)
    return graph

#
# def removeMutualLoop(graph, rmin=3, length=5):
#     radii = graph.edge_radii()
#     conn = graph.edge_connectivity()
#     rad1 = np.asarray(radii <= rmin).nonzero()[0]
#     # print(rad1.shape)
#     edge2rm = []
#     lengths=graph.edge_geometry_lengths()
#     print(graph, np.asarray(radii <= rmin).shape)
#     graph.add_edge_property('rad1', np.asarray(radii <= rmin))
#     n=0
#     for j in range(rad1.shape[0]):
#         n=False
#         rad1=graph.edge_property('rad1')
#         i=rad1.nonzero()[0][0]
#         print(i)
#         co=conn[i]
#         n0=graph.vertex_neighbours(co[0])
#         n1=graph.vertex_neighbours(co[1])
#         # print('n0, n1 ', n0, n1)
#         u, c =np.unique(n0, return_counts=True)
#         # print('u, c ',u, c)
#         if 2 in c:
#             # print('l ', lengths[i])
#             if lengths[i]<=length:
#                 edge2rm.append(i)
#                 # print('rm ', i)
#                 ef = np.ones(graph.n_edges)
#                 ef[i] = 0
#                 rad1[i]=0
#                 graph.add_edge_property('rad1', np.asarray(rad1).astype('bool'))
#                 graph = graph.sub_graph(edge_filter=ef)
#                 n = True
#
#         else:
#             u, c = np.unique(n1, return_counts=True)
#             # print(u, c)
#             if 2 in c:
#                 if lengths[i] <= length:
#                     edge2rm.append(i)
#                     # print('rm ', i)
#                     ef = np.ones(graph.n_edges)
#                     ef[i] = 0
#                     rad1[i] = 0
#                     graph.add_edge_property('rad1', np.asarray(rad1).astype('bool'))
#                     graph = graph.sub_graph(edge_filter=ef)
#                     n = True
#
#         # print(rad1)
#         if n == False:
#             rad1[i] = 0
#             graph.add_edge_property('rad1', np.asarray(rad1).astype('bool'))
#
#     edge2rm=np.array(edge2rm)
#     print(edge2rm.shape)
#     # ef = np.ones(graph.n_edges)
#     # ef[edge2rm] = 0
#     # graph = graph.sub_graph(edge_filter=ef)
#     return graph


def mutualLoopDetection(args):
    res=0
    ind, i, rmin, length, conn, radii, lengths = args
    co = conn[i]
    # print(ind)
    similaredges = np.logical_or(np.logical_and(conn[:, 0] == co[0], conn[:, 1] == co[1]),
                                 np.logical_and(conn[:, 1] == co[0], conn[:, 0] == co[1]))
    # print(similaredges.shape)
    similaredges = np.asarray(similaredges == True).nonzero()[0]

    if similaredges.shape[0] >= 2:
        rs = radii[similaredges]
        # print(rs)
        imin = np.argmin(rs)
        if rs[imin] <= rmin:
            if lengths[imin] <= length:
                print('adding edge to remove ', similaredges[imin])
                # e2rm.append(similaredges[imin])
                res=similaredges[imin]
    return res


def removeMutualLoop(graph, rmin=3, length=5):
    radii = graph.edge_radii()
    conn = graph.edge_connectivity()
    rad1 = np.asarray(radii <= rmin).nonzero()[0]
    print(rad1.shape)
    edge2rm = []
    lengths=graph.edge_geometry_lengths()
    # print(graph, np.asarray(radii <= rmin).shape, rmin, length)
    # graph.add_edge_property('rad1', np.asarray(radii <= rmin))
    n=0
    for i in rad1:
        # n=False
        # rad1=graph.edge_property('rad1')
        # i=rad1.nonzero()[0][0]
        # print(i)
        co=conn[i]
        # n0=graph.vertex_neighbours(co[0])
        # n1=graph.vertex_neighbours(co[1])
        # print('n0, n1 ', co[0], co[1])

        similaredges=np.logical_or(np.logical_and(conn[:,0]==co[0], conn[:, 1]==co[1]), np.logical_and(conn[:,1]==co[0], conn[:, 0]==co[1]))
        # print(similaredges.shape)
        similaredges=np.asarray(similaredges ==True).nonzero()[0]


        if similaredges.shape[0]>=2:
            rs=radii[similaredges]
            # print(rs)
            imin=np.argmin(rs)
            if rs[imin]<=rmin:
                if lengths[imin]<=length:
                    # print('adding edge to remove ', imin)
                    edge2rm.append(similaredges[imin])
                    # n = True

        # u, c =np.unique(n0, return_counts=True)
        # # print('u, c ',u, c)
        # if 2 in c:
        #     # print('l ', lengths[i])
        #     if lengths[i]<=length:
        #         edge2rm.append(i)
        #         # print('rm ', i)
        #         ef = np.ones(graph.n_edges)
        #         ef[i] = 0
        #         rad1[i]=0
        #         graph.add_edge_property('rad1', np.asarray(rad1).astype('bool'))
        #         graph = graph.sub_graph(edge_filter=ef)
        #         n = True
        #
        # else:
        #     u, c = np.unique(n1, return_counts=True)
        #     # print(u, c)
        #     if 2 in c:
        #         if lengths[i] <= length:
        #             edge2rm.append(i)
        #             # print('rm ', i)
        #             ef = np.ones(graph.n_edges)
        #             ef[i] = 0
        #             rad1[i] = 0
        #             graph.add_edge_property('rad1', np.asarray(rad1).astype('bool'))
        #             graph = graph.sub_graph(edge_filter=ef)
        #             n = True
        #
        # # print(rad1)
        # if n == False:
        #     rad1[i] = 0
        #     graph.add_edge_property('rad1', np.asarray(rad1).astype('bool'))

    edge2rm=np.array(edge2rm)
    print(edge2rm.shape)
    ef = np.ones(graph.n_edges)
    if edge2rm.shape[0] !=0:
        ef[edge2rm] = 0
        graph = graph.sub_graph(edge_filter=ef)
    return graph



def remove_surface(graph, width):
    distance_from_suface = graph.edge_property('distance_to_surface')
    ef=distance_from_suface>width
    g=graph.sub_graph(edge_filter=ef)
    return g

def graphCorrection(graph, graph_dir, region, save=True):
    artery = graph.edge_property('artery');
    vein = graph.edge_property('vein');

    u, c = np.unique(vein, return_counts=True)
    print('vein init : ', u, c / graph.n_edges)
    vi = c[1] / graph.n_edges

    u, c = np.unique(artery, return_counts=True)
    print('artery init : ', u, c / graph.n_edges)
    ai = c[1] / graph.n_edges

    graph, vf=extract_AnnotatedRegion(graph, region)
    #
    print(graph)
    art = graph.edge_property('artery')
    vein = graph.edge_property('vein')

    u, c = np.unique(art, return_counts=True)
    print('vein init : ', u, c / graph.n_edges)
    vi = c[1] / graph.n_edges

    u, c = np.unique(vein, return_counts=True)
    print('artery init : ', u, c / graph.n_edges)
    ai = c[1] / graph.n_edges

    gss = graph.largest_component()

    degrees_m = gss.vertex_degrees()
    deg4 = np.asarray(degrees_m >= 4).nonzero()[0]
    print('deg4 init ', deg4.shape[0] / gss.n_vertices)
    d4i = [deg4.shape[0] / gss.n_vertices, gss.vertex_property('distance_to_surface')]

    gss = removeSpuriousBranches(gss, rmin=2.9, length=7)
    gss = remove_surface(gss, 2)
    gss = gss.largest_component()

    gss = removeAutoLoops(gss)
    # gss.remove_self_loops()
    # gss=removeMutualLoop(gss, rmin=2.9, length=13)

    rmin = 2.9
    length = 13
    radii = gss.edge_radii()
    lengths = gss.edge_geometry_lengths()

    conn = gss.edge_connectivity()
    rad1 = np.asarray(radii <= rmin)  # .nonzero()[0]
    len1 = np.asarray(lengths <= length)  # .nonzero()[0]
    l = np.logical_and(len1, rad1).nonzero()[0]

    print(l.shape)
    # edge2rm = []

    # global r2min
    # r2min=np.ones(gss.n_edges)
    from multiprocessing import Pool
    p = Pool(20)
    import time
    start = time.time()

    e2rm = np.array(
        [p.map(mutualLoopDetection, [(ind, i, rmin, length, conn, radii, lengths) for ind, i in enumerate(l)])])

    end = time.time()
    print(end - start)

    print(gss)
    degrees_m = gss.vertex_degrees()
    deg4 = np.asarray(degrees_m >= 4).nonzero()[0]
    print('deg4 exit ', deg4.shape)

    u = np.unique(e2rm[0].nonzero())
    ef = np.ones(gss.n_edges)
    ef[u] = 0
    g = gss.sub_graph(edge_filter=ef)
    # g.save('/data_SSD_2to/mesospim/data_graph_corrected_30R.gt')

    degrees_m = g.vertex_degrees()
    deg4 = np.asarray(degrees_m >= 4).nonzero()[0]
    print('deg4 exit ', deg4.shape[0] / g.n_vertices)
    d4e = [deg4.shape[0] / gss.n_vertices, gss.vertex_property('distance_to_surface')]

    artery = g.edge_property('artery');
    vein = g.edge_property('vein');

    u, c = np.unique(vein, return_counts=True)
    print('vein exit : ', u, c / g.n_edges)
    ve = c[1] / g.n_edges

    u, c = np.unique(artery, return_counts=True)
    print('artery exit : ', u, c / g.n_edges)
    ae = c[1] / g.n_edges

    if save:
        g.save(graph_dir+'/data_graph_corrected_v2'+ano.find_name(region[0], key='order')+'.gt')
    return g, [ai, vi, d4i, ae, ve, d4e]



def checkProblematicVertices(graph):
    probvertices=np.asarray(graph.vertex_degrees()>=4).nonzero()[0]
    depthprobVertics=graph.vertex_property('distance_to_surface')[probvertices]
    depthAll=graph.vertex_property('distance_to_surface')
    return depthprobVertics, depthAll


if __name__ == "__main__":
    bin=10
    norm=False
    work_dir = '/data_SSD_2to/whiskers_graphs/fluoxetine'# '/data_SSD_2to/191122Otof'
    controls = ['2R' +',3R', '5R', '8R']
    mutants = ['1R', '7R', '6R', '4R']

    brains=['138L', '141L', '142L', '158L', '163L', '162L', '164L','165L']
    brains = ['36']#['21', '22', '23']#, ['1', '2', '3', '4', '6', '18', '26']
    Tab=[]
    work_dir='/data_SSD_1to/10weeks'
    brains=['1L', '2L', '3L', '4L', '6L', '8L', '9L']
    brains = ['7R', '8R', '6R','2R', '3R', '5R', '1R']
    work_dir = '/data_SSD_1to/otof6months'
    brains=['1', '2', '4', '7', '8', '9']
    work_dir= '/data_2to/DBA2J'
    work_dir = '/data_SSD_2to/whiskers_graphs/fluoxetine2'
    work_dir = '/data_SSD_2to/fluoxetine2'
    brains=['1', '2', '3', '4','5', '7', '8', '9', '10', '11']

    work_dir='/data_2to/earlyDep_ipsi'
    brains=['3', '4', '6', '7','10', '11', '15', '16']

    work_dir='/data_2to/otof1M'
    brains=['1k', '1w', '2k', '3k','3w', '4k', '5w', '6w', '7w']

    work_dir='/data_SSD_2to/otofRECUP/'
    brains=['17L', '17R', '18L', '18R', '19L', '19R', '20R', '20L', '21L', '21R', '22L', '22R']

    region_list = [(0, 0)]
    for c in brains:
        # graph=ggt.load(work_dir+'/'+c+'/data_graph.gt')
        graph=ggt.load(work_dir+'/'+c+'/'+str(c)+'_graph.gt')
        giso= graphCorrection(graph, work_dir+'/'+c, region_list[0], save=True)
        Tab.append(T)

        giso = ggt.load(work_dir+'/'+c+'/data_graph_corrected_Isocortex.gt')

        graph, vf = extract_AnnotatedRegion(graph, (6, 6))
        ### GET CLUSTERS
        # # diffusion_through_penetrating_arteries(giso, get_penetration_veins_dustance_surface,
        # #                                        get_penetrating_veins_labels, vesseltype='vein', graph_dir=work_dir+'/'+c, feature='distance')
        # diffusion_through_penetrating_arteries(giso, get_penetration_arteries_dustance_surface,
        #                                        get_penetrating_arteries_labels, vesseltype='art',graph_dir=work_dir+'/'+c)
        # # vein_clusters = np.load(
        # #     work_dir+'/'+c+'/sbm/diffusion_penetrating_vessel_vein_end_point_cluster_per_region_iteration_Isocortex_graph_corrected.npy')
        # art_clusters = np.load(
        #     work_dir+'/'+c+'/sbm/diffusion_penetrating_vessel_art_end_point_cluster_per_region_iteration_Isocortex_graph_corrected.npy')
        #
        # print('art ', np.unique(art_clusters).shape)
        # print('vein ', np.unique(vein_clusters).shape)
        # indices = get_art_vein_overlap(giso, vein_clusters, art_clusters)
        # np.save(
        #     work_dir+'/'+c+'/sbm/diffusion_penetrating_vessel_overlap_end_point_cluster_per_region_iteration_Isocortex_graph_corrected.npy',
        #     indices)
        depth_p, depth = checkProblematicVertices(graph)
        depth_corr_p, depth_corr = checkProblematicVertices(giso)

        # if i == 0:
        #     probVert = depth
        #     probVertCorr = depth_corr
        # else:
        #     probVert = np.concatenate((probVert, depth))
        #     probVertCorr = np.concatenate((probVertCorr, depth_corr), axis=1)

        hist, bins = np.histogram(depth, bins=bin, normed=norm)
        hist_p, bins = np.histogram(depth_p, bins=bins, normed=norm)

        hist = hist_p / hist
        if i == 0:
            H = hist.reshape((hist.shape[0], 1))
        else:
            H = np.concatenate((H, hist.reshape((hist.shape[0], 1))), axis=1)

        hist, bins = np.histogram(depth_corr, bins=bins, normed=norm)
        hist_p, bins = np.histogram(depth_corr_p, bins=bins, normed=norm)

        hist = hist_p / hist
        if i == 0:
            H_corr = hist.reshape((hist.shape[0], 1))
        else:
            H_corr = np.concatenate((H, hist.reshape((hist.shape[0], 1))), axis=1)

    plt.figure()

    Hpd = pd.DataFrame(np.array(H).transpose()[5:, 1:]).melt()
    Hpd_corr = pd.DataFrame(np.array(H_corr).transpose()[5:, 1:]).melt()
    sns.lineplot(x="variable", y="value", err_style='bars', data=Hpd)
    sns.lineplot(x="variable", y="value", err_style='bars', data=Hpd_corr)
    plt.yscale('linear')
    plt.xticks(np.arange(bin - 1), np.arange(1, np.max(depth_corr), np.max(depth_corr) / bin).astype(int))

    plt.title('distribution of the depth of degree 4 or + vertices')
    plt.legend(['before', 'after'])
    # plt.xlim([0, 60])
    sns.despine()

    I=[]
    E=[]
    for t in Tab:
        ai=t[2][0]
        ae=t[5][0]
        I.append(ai)
        E.append(ae)

    plt.figure()
    plt.plot(I)
    plt.plot(E)


    expe='whiskers'

    if expe == 'otof':
        work_dir = '/data_SSD_2to/191122Otof'
        controls = ['2R', '3R', '5R', '8R']
        mutants = ['1R', '7R', '6R', '4R']
    elif expe == 'whiskers':
        work_dir = '/data_SSD_2to/whiskers_graphs/new_graphs'
        controls = ['142L', '158L', '162L', '164L']
        mutants = ['138L', '141L', '163L', '165L']


    cluster_type = 'art'  # 'vein', 'overlap'
    states = [controls, mutants]

    probVert=[]
    probVertCorr=[]

    bin=30
    norm=False

    for state in states:

        for i, cont in enumerate(state):
            print(cont)

            graph=ggt.load(work_dir + '/' + cont + '/data_graph.gt')
            order, level = (6, 6)
            print(level, order, ano.find_name(order, key='order'))
            label = graph.vertex_annotation();
            label_leveled = ano.convert_label(label, key='order', value='order', level=level)
            vertex_filter = label_leveled == order;
            graph = graph.sub_graph(vertex_filter=vertex_filter)

            graph_corr = ggt.load(work_dir + '/' + cont + '/data_graph_corrected_Isocortex.gt')

            depth_p, depth=checkProblematicVertices(graph)
            depth_corr_p, depth_corr = checkProblematicVertices(graph_corr)

            # if i == 0:
            #     probVert = depth
            #     probVertCorr = depth_corr
            # else:
            #     probVert = np.concatenate((probVert, depth))
            #     probVertCorr = np.concatenate((probVertCorr, depth_corr), axis=1)

            hist, bins = np.histogram(depth, bins=bin, normed=norm)
            hist_p, bins = np.histogram(depth_p, bins=bins, normed=norm)

            # hist=hist_p/hist
            if i == 0:
                H = hist.reshape((hist.shape[0], 1))
            else:
                H = np.concatenate((H, hist.reshape((hist.shape[0], 1))), axis=1)

            hist, bins = np.histogram(depth_corr, bins=bins, normed=norm)
            hist_p, bins = np.histogram(depth_corr_p, bins=bins, normed=norm)

            # hist = hist_p / hist
            if i == 0:
                H_corr = hist.reshape((hist.shape[0], 1))
            else:
                H_corr = np.concatenate((H, hist.reshape((hist.shape[0], 1))), axis=1)

        plt.figure()

        Hpd = pd.DataFrame(np.array(H).transpose()[:, 1:]).melt()
        Hpd_corr = pd.DataFrame(np.array(H_corr).transpose()[:, 1:]).melt()
        sns.lineplot(x="variable", y="value", err_style='bars', data=Hpd)
        sns.lineplot(x="variable", y="value", err_style='bars', data=Hpd_corr)
        plt.yscale('linear')
        plt.xticks(np.arange(bin - 1), np.arange(1, np.max(depth_corr), np.max(depth_corr)/bin).astype(int))

        plt.title('distribution of the depth of degree 4 or + vertices')
        plt.legend(['before', 'after'])
        # plt.xlim([0, 60])
        sns.despine()



graph=ggt.load('/data_SSD_2to/whiskers_graphs/new_graphs/162L/data_graph_correcteduniverse.gt')
deg1=np.asarray(graph.vertex_degrees()).nonzero()[0]
depthdeg1=graph.vertex_property('distance_to_surface')[deg1]
plt.figure()
plt.hist(depthdeg1, bins=10)

nb = 5
depthdeg1 = graph.vertex_property('distance_to_surface')[deg1]
plt.figure()
max_depth = np.max(graph.vertex_property('distance_to_surface'))
bin = np.arange(max_depth, nb)
print(bin)
# deg1 = np.asarray(graph.vertex_degrees()==1).nonzero()[0]

#check length of edges belonging to degree 1 verices
connectivity=graph.edge_connectivity()
deg1 = graph.vertex_degrees() == 1
d1edges=np.logical_or(deg1[connectivity[:, 0]], deg1[connectivity[:, 1]])
Ls=[]
lengths=graph.edge_property('length')
# for n, i in enumerate(deg1):
#     print(n,'/',deg1.shape[0],  ' ' , len(Ls))
#     es = np.asarray(graph.edge_connectivity() == i).nonzero()[0]
#     # print(es)
#     for e in es:
#         Ls.append(lengths[e])
Ls=lengths[np.asarray(d1edges).nonzero()[0]]
plt.figure()
plt.hist(Ls, bins=100)
plt.yscale('log')

depthdeg1 = graph.vertex_property('distance_to_surface')[deg1]
deg1=graph.vertex_degrees()==1
for i in range(nb):
    ran = [int(i * max_depth / nb), int((i + 1) * max_depth / nb)]
    print(ran)
    slice = np.logical_and(graph.vertex_property('distance_to_surface') >= ran[0],
                           graph.vertex_property('distance_to_surface') < ran[1])
    d1 = np.logical_and(deg1, slice)
    plt.bar(i, np.sum(d1) / np.sum(slice))

import random
v = random.choice(deg1.nonzero()[0])
print(v)
c=graph.vertex_coordinates(v)
# dict_color = np.array([[1, 0, 0, 1], [0,1,0, 1]]).astype('float')
# colors=dict_color[np.asarray(graph.vertex_degrees()==1, dtype=int)]
# graph.add_vertex_property('colors' , colors)
print(graph.vertex_property('distance_to_surface')[v])
s=50
gs=extractSubGraph(graph,c-np.array([s,s,s]), c+np.array([s, s, s]))
colors=gs.vertex_property('colors')
p3d.plot_graph_mesh(gs, vertex_colors=colors)
