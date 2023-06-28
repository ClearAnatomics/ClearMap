
import numpy as np
import ClearMap.Alignment.Annotation as ano
from matplotlib import cm
import ClearMap.Visualization.Plot3d as p3d
import ClearMap.Analysis.Graphs.GraphGt as ggt
print('loading...')
import numpy as np
import graph_tool as gt
# import graph_tool.draw as gtd
import graph_tool.spectral as gts
import seaborn as sns; sns.set()
#
# import ClearMap.Analysis.Measurements.MeasureRadius as mr
# import ClearMap.Analysis.Measurements.MeasureExpression as me
#
# import graph_tool.inference as gti
#
#
# import math
# import matplotlib.pyplot as plt
#
#
#
#
# import ClearMap.Settings as settings
# import ClearMap.IO.IO as io
#
# import ClearMap.ImageProcessing.Skeletonization.Skeletonization as skl

# import ClearMap.Analysis.Graphs.GraphProcessing as gp


# import scipy.cluster.hierarchy as hierarchy
# import seaborn as sns

# import ClearMap.Gt2Nx as Gt2Nx
# import networkx as nx

# import networkx.algorithms.similarity as nxas
# from scipy.cluster.hierarchy import ward, dendrogram, linkage
import random
# from ClearMap.DiffusionFromArteries import from_e_prop2_vprop


def from_e_prop2_vprop(graph, property):
    e_prop = graph.edge_property(property)
    v_prop=np.zeros(graph.n_vertices)
    connectivity = graph.edge_connectivity()
    v_prop[connectivity[e_prop==1,0]]=1
    v_prop[connectivity[e_prop == 1,1]] = 1
    # graph.add_vertex_property(property, v_prop)
    return v_prop



def getRsDistMatrix(graph):
    print(graph)
    L=gts.laplacian(graph.base, weight=graph.base.ep['length'])
    L=graph.laplacian()
    gamma = np.linalg.pinv(L)

    rows_mat=np.zeros(gamma.shape)
    for i in range(gamma.shape[0]):
        rows_mat[i, :] = gamma[i, i]

    col_mat=np.zeros(gamma.shape)
    for i in range(gamma.shape[0]):
        col_mat[:, i] = gamma[i,i]

    Omega=rows_mat+col_mat -2*gamma
    print('gamma')
    # print(gamma)
    print('rows gamma')
    # print(rows_mat)
    print('cols gamma')
    # print(col_mat)

    return Omega


def labelNodeIntensities(graph):
    arteries=from_e_prop2_vprop(graph,'artery')
    veins = from_e_prop2_vprop(graph, 'vein')
    degree=graph.vertex_degrees()
    surf_dist=graph.vertex_property('distance_to_surface')
    v=np.multiply(surf_dist, arteries)
    v = np.ma.masked_equal(v, 0)
    i=np.argmin(v)
    I=np.zeros(graph.n_vertices)
    I[i]=100
    v=np.multiply(surf_dist, veins)
    v = np.ma.masked_equal(v, 0)
    j=np.argmin(v)
    I[j]=-100
    return I



def getColorMap_from_vertex_prop(vp):
    colors = np.zeros((vp.shape[0], 4));
    print(colors.shape)
    # for i in range(b.size):
    i = 0

    # for index, b in enumerate(np.unique(diff)):
    cmax = np.max(vp)
    cmin = np.min(vp)
    # for i in range(gss4.n_vertices):
    #     colors[i, :]=[x[i],0, 1-x[i], 1]
    jet = cm.get_cmap('jet_r')#jet_r#viridis
    import matplotlib.colors as col
    cNorm = col.Normalize(vmin=cmin, vmax=cmax)
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)
    print(scalarMap.get_clim())
    colorVal = scalarMap.to_rgba(vp)
    return colorVal

def getRIV(graph, graph_dir, region, ps=7.3, selectRegion=True):

    diff = np.load(
       graph_dir + '/sbm/'+'diffusion_penetrating_vessel_overlap_end_point_cluster_per_region_iteration_Isocortex_graph_corrected' + '.npy')#+ '_end_point_cluster_per_region_iteration_' + ano.find_name(order, key='order')
    # diff=np.load('/mnt/data_SSD_2to/' + brainnb + '/sbm/diffusion_penetrating_art_end_point_cluster_per_region_'+ ano.find_name(order,key='order') + '.npy')
    graph.add_vertex_property('overlap', diff)

    order, level = region

    print(level, order, ano.find_name(order, key='order'))

    label = graph.vertex_annotation();
    label_leveled = ano.convert_label(label, key='order', value='order', level=level)
    vertex_filter = label_leveled == order;

    # gss4 = graph.sub_graph(vertex_filter=vertex_filter)
    # gss4=graph.largest_component()

    graph = graph.sub_graph(vertex_filter=vertex_filter)

    overlap=graph.vertex_property('overlap')

    u, c = np.unique(overlap, return_counts=True)

    tension_nodale=[]
    omegas=[]
    currents=[]
    tension_branche=[]
    currents_branche=[]
    extracted_fraction=[]
    u_=u.copy()
    c_=c.copy()
    ind2rm=[]

    for i, e in enumerate(u):
        vp=overlap==e
        module=graph.sub_graph(vertex_filter=vp)
        module=module.largest_component()
        if module.n_vertices<=3000 and module.n_edges>=5:
            bool=True
            try:
                resistance_distance=getRsDistMatrix(module)
            except:
                print('deketed', i)
                ind2rm.append(i)
                bool=False

            if bool:
                omegas.append(resistance_distance)
                adj=module.adjacency()
                Lw=-np.multiply(resistance_distance, adj)
                for i in range(Lw.shape[0]):
                    Lw[i, i]=np.sum(-Lw[:, i])#module.vertex_degrees()[i]
                I=np.expand_dims(labelNodeIntensities(module), 1)
                V= np.linalg.pinv(Lw)*I
                tension_nodale.append(V)
                print(V.shape,resistance_distance.shape)
                # plt.figure(1)
                # plt.hist(V, bins=100)

                connectivity = module.edge_connectivity()
                V_br = np.squeeze(np.asarray((V[connectivity[:, 0]]+ V[connectivity[:, 1]])/2), axis=1)
                R_br=np.array([resistance_distance[connectivity[i, 0], connectivity[i, 1]] for i in range(connectivity.shape[0])])
                tension_branche.append(V_br)
                I_br=V_br/R_br
                # print('shapes !',V_br.shape, R_br.shape, I_br.shape)
                currents_branche.append(I_br)
                e=1-np.exp(-(ps/abs(I_br)))
                extracted_fraction.append(e)
                print('shapes !', e.shape)

        else:
            print('neglectable graph')
            ind2rm.append(i)

    u_ = np.delete(u_, ind2rm)
    for i,e in enumerate(u_):
        vp=overlap==e
        module=graph.sub_graph(vertex_filter=vp)
        module = module.largest_component()
        if module.n_vertices <= 3000 and module.n_edges >= 5:
            V=tension_nodale[i]
            R=omegas[i]
            connectivity=module.edge_connectivity()
            current=np.zeros(module.n_edges)
            print(i, module.n_vertices, module.n_edges, V.shape,R.shape)
            for j, c in enumerate(connectivity):
                print(i, module.n_vertices, module.n_edges, V.shape, c[0], c[1])
                ans=abs(V[c[1]]-V[c[0]])/R[c[0], c[1]]
                if np.isnan(ans):
                    print('isnan !!!!!!')
                    print(c[1], c[0], V[c[1]], V[c[0]], R[c[0], c[1]])
                    current[j] = 0
                else:
                    current[j]=ans
            currents.append(current)

    edge_degs=[]
    for i, e in enumerate(u_):
        vp = overlap == e
        edge_deg=[]
        module = graph.sub_graph(vertex_filter=vp)
        module = module.largest_component()
        if module.n_vertices <= 3000 and module.n_edges >= 5:
            V = tension_nodale[i]
            R = omegas[i]
            degrees = module.vertex_degrees()
            connectivity = module.edge_connectivity()
            for j, c in enumerate(connectivity):
                edge_deg.append(degrees[c[0]]+degrees[c[1]])

            edge_degs.append(edge_deg)

    return omegas, currents, tension_nodale, tension_branche, edge_degs, currents_branche,extracted_fraction


if __name__ == "__main__":
    # execute only if run as a script
    brain_list = ['190408_44R']
    brainnb=brain_list[0]
    region_list = []
    region_list = [(1006, 3), (580, 5), (650, 5), (724, 5), (811, 4), (875, 4), (6, 6), (463, 6), (388, 6)]
    reg_colors = {1006: 'gold', 580: 'skyblue', 650: 'indianred', 724: 'violet', 811: 'darkorchid',
                  875: 'mediumslateblue', 6: 'forestgreen', 463: 'lightgreen', 388: 'turquoise'}

    # graph=initialize_brain_graph(brainnb)
    graph=ggt.load('/data_SSD_2to/190408_44R/data_graph.gt')
    region_list=[(6, 6)]
    order, level=region_list[0]
    vesseltype='overlap'

    print(level, order, ano.find_name(order, key='order'))

    label = graph.vertex_annotation();
    label_leveled = ano.convert_label(label, key='order', value='order', level=level)
    vertex_filter = label_leveled == order;

    gss4 = graph.sub_graph(vertex_filter=vertex_filter)
    # gss4=graph.largest_component()

    omegas, currents, tension_nodale = getRIV(gss4, '/data_SSD_2to/' + brainnb, region)
    # diff = np.load(
    #     '/data_SSD_2to/' + brainnb + '/sbm/diffusion_penetrating_vessel_' + vesseltype + '.npy')#+ '_end_point_cluster_per_region_iteration_' + ano.find_name(order, key='order')
    # # diff=np.load('/mnt/data_SSD_2to/' + brainnb + '/sbm/diffusion_penetrating_art_end_point_cluster_per_region_'+ ano.find_name(order,key='order') + '.npy')
    # gss4.add_vertex_property('overlap', diff)
    #
    # overlap=gss4.vertex_property('overlap')
    #
    # u, c = np.unique(overlap, return_counts=True)
    #
    # tension_nodale=[]
    # omegas=[]
    # currents=[]
    #
    # for e in u:
    #     vp=overlap==e
    #     module=gss4.sub_graph(vertex_filter=vp)
    #     resistance_distance=getRsDistMatrix(module)
    #     omegas.append(resistance_distance)
    #     adj=module.adjacency()
    #     Lw=-np.multiply(resistance_distance, adj)
    #     for i in range(Lw.shape[0]):
    #         Lw[i, i]=np.sum(-Lw[:, i])#module.vertex_degrees()[i]
    #     I=np.expand_dims(labelNodeIntensities(module), 1)
    #     V= np.linalg.pinv(Lw)*I
    #     tension_nodale.append(V)
    #     print(V.shape,resistance_distance.shape)
    #     # plt.figure(1)
    #     # plt.hist(V, bins=100)
    #
    #
    # for i,e in enumerate(u):
    #     vp=overlap==e
    #     module=gss4.sub_graph(vertex_filter=vp)
    #     V=tension_nodale[i]
    #     R=omegas[i]
    #     connectivity=module.edge_connectivity()
    #     current=np.zeros(module.n_edges)
    #     for j, c in enumerate(connectivity):
    #         ans=abs(V[c[1]]-V[c[0]])/R[c[0], c[1]]
    #         if np.isnan(ans):
    #             print('isnan !!!!!!')
    #             print(c[1], c[0], V[c[1]], V[c[0]], R[c[0], c[1]])
    #             current[j] = 0
    #         else:
    #             current[j]=ans
    #     currents.append(current)
    #
    # edge_degs=[]
    # for i, e in enumerate(u):
    #     vp = overlap == e
    #     edge_deg=[]
    #     module = gss4.sub_graph(vertex_filter=vp)
    #     V = tension_nodale[i]
    #     R = omegas[i]
    #     degrees = module.vertex_degrees()
    #     connectivity = module.edge_connectivity()
    #     for j, c in enumerate(connectivity):
    #         edge_deg.append(degrees[c[0]]+degrees[c[1]])
    #
    #     edge_degs.append(edge_deg)
    #
    #
    import pickle

    with open('/data_SSD_2to/' + brainnb + '/sbm/currents_100.p', 'wb') as fp:
        pickle.dump(currents, fp, protocol=pickle.HIGHEST_PROTOCOL)
    with open('/data_SSD_2to/' + brainnb + '/sbm/tension_nodale_100.p', 'wb') as fp:
        pickle.dump(tension_nodale, fp, protocol=pickle.HIGHEST_PROTOCOL)
    with open('/data_SSD_2to/' + brainnb + '/sbm/omegas_100.p', 'wb') as fp:
        pickle.dump(omegas, fp, protocol=pickle.HIGHEST_PROTOCOL)


    with open('/data_SSD_2to/' + brainnb + '/sbm/omegas_100.p', 'rb') as fp:
        omegas = pickle.load(fp)
    with open('/data_SSD_2to/' + brainnb + '/sbm/tension_nodale_100.p', 'rb') as fp:
        tension_nodale = pickle.load(fp)
    with open('/data_SSD_2to/' + brainnb + '/sbm/currents_100.p', 'rb') as fp:
        currents = pickle.load(fp)

    import random
    v=0
    while v<800:
        id = random.choice(range(u.shape[0]))
        print(id)
        v=c[id]
    e=u[id]
    vp = overlap == e
    module = gss4.sub_graph(vertex_filter=vp)
    # id=1820
    V = tension_nodale[id]
    R = omegas[id]
    I=currents[id]

    # plt.figure()
    # T=np.ravel(R)
    # T=T[T!=0]
    # plt.hist(T, bins=100)
    #
    # plt.figure()
    # plt.hist(V, bins=100)
    # plt.yscale('log')
    # plt.title('vertex tension')
    #
    # plt.figure()
    # x = I[~np.isnan(I)]
    # plt.hist(x, bins=100)
    # plt.yscale('log')
    # plt.title('intensity')
    #
    # plt.figure()
    # for i in range(10):
    #     v=0
    #     while v < 300:
    #         id = random.choice(range(u.shape[0]))
    #         v = c[id]
    #     print(id)
    #     plt.scatter(edge_degs[id], currents[id])
    # plt.ylabel('current intensity')
    # plt.xlabel('edge vertex degree')


    module.add_edge_property('I', I)
    colorval = getColorMap_from_vertex_prop(module.edge_property('I'))
    v_arteries = from_e_prop2_vprop(module, 'artery')
    v_veins = from_e_prop2_vprop(module, 'vein')
    v_arteries = module.edge_property('artery')
    v_veins = module.edge_property('vein')
    colorval[v_arteries == 1] = [1., 0.0, 0.0, 1.0]
    colorval[v_veins == 1] = [0.0, 0.0, 1.0, 1.0]
    p = p3d.plot_graph_mesh(module, edge_colors=colorval, n_tube_points=3);# #edge_colors=colorval [:,0, :]

    v_arteries = from_e_prop2_vprop(module, 'artery')
    v_veins = from_e_prop2_vprop(module, 'vein')

    #compute sparse eig values
    from scipy.sparse.linalg import eigs
    from sklearn.preprocessing import normalize

    normed_R = normalize(R, axis=1, norm='l1')
    vals, vecs = eigs(np.asarray(normed_R), k=3)
    vals=np.real(vals[1:])
    vecs = np.real(vecs[:, 1:])

    module.add_vertex_property('pos', vecs * vals)

    module.add_vertex_property('v_veins', v_veins.astype('bool'))
    module.add_vertex_property('v_arteries', v_arteries.astype('bool'))

    pos = gt.draw.sfdp_layout(module.base)

    red_blue_map = {0: [1.0, 0, 0, 1.0], 1: [0, 0, 1.0, 1.0], 2: [0, 1.0, 1.0]}

    plot_color = module.base.new_vertex_property('vector<double>')

    veins= module.base.vertex_properties['v_veins']
    arteries = module.base.vertex_properties['v_arteries']

    # j = 0
    for v in module.base.vertices():
        res = 2
        if veins[v]:
            res = 1
        if arteries[v]:
            res = 0
        # print(res)
        plot_color[v] = red_blue_map[res]
        # j = j + 1


    module.base.vertex_properties['plot_color'] = plot_color


    gt.draw.graph_draw(module.base, pos=pos, vertex_fill_color=module.base.vertex_properties['plot_color'], output="/home/sophie.skriabine/Pictures/GeraphAnalysisTries/graph-draw-sfdp"+str(id)+".pdf")
    gt.draw.graph_draw(module.base, pos=module.base.vertex_properties['pos'], vertex_fill_color=module.base.vertex_properties['plot_color'], output="/data_SSD_2to/whiskers_graphs/teststuckedcycles_color_embedded_diff_map_pos"+str(id)+".pdf")




