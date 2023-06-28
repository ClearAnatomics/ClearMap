import numpy as np

import ClearMap.Settings as settings

import ClearMap.Analysis.Graphs.GraphProcessing as gp

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
import random

import math
pi=math.pi
import pickle
from matplotlib import cm

def cart2sph(x,y,z, ceval=ne.evaluate):
    """ x, y, z :  ndarray coordinates
        ceval: backend to use:
              - eval :  pure Numpy
              - numexpr.evaluate:  Numexpr """
    r = ceval('sqrt(x**2+y**2+z**2)')#sqrt(x * x + y * y + z * z)
    theta = ceval('arccos(z/r)*180')/pi#acos(z / r) * 180 / pi  # to degrees
    phi = ceval('arctan2(y,x)*180')/pi#*180/3.4142
    # azimuth = ceval('arctan2(y,x)')
    # xy2 = ceval('x**2 + y**2')
    # elevation = ceval('arctan2(z, sqrt(xy2))')
    # r = ceval('sqrt(xy2 + z**2)')
    rmax=np.max(r)
    return phi/180, theta/180, r/rmax#, theta/180, phi/180

def get_spherical_orientation(graph, all_graph):

    x = graph.vertex_coordinates()[:, 0]
    y = graph.vertex_coordinates()[:, 1]
    z = graph.vertex_coordinates()[:, 2]

    x_g = all_graph.vertex_coordinates()[:, 0]
    y_g = all_graph.vertex_coordinates()[:, 1]
    z_g = all_graph.vertex_coordinates()[:, 2]

    center = np.array([np.mean(x_g), np.mean(y_g), np.mean(z_g)])
    x = x - np.mean(x_g)
    y = y - np.mean(y_g)
    z = z - np.mean(z_g)

    spherical_coord = np.array(cart2sph(x, y, z, ceval=ne.evaluate)).T
    connectivity = graph.edge_connectivity()

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



def initialize_brain_graph(brain):
    # gri = ggt.load('/mnt/data_SSD_2to/' + brain + '/data_graph_reduced.gt')
    grti = ggt.load('/mnt/data_SSD_2to/' + brain + '/data_graph_reduced_transformed.gt')
    # grti=ggt.load('/mnt/data_SSD_2to/181002_4/data_graph_reduced_transformed.gt')

    level = 1
    order = 1
    print('decomposition ...')
    # gr = gri.largest_component()
    gt = grti.largest_component()
    print('gt vertex degree 2 ', np.sum(gt.vertex_degrees() == 2))
    print('grti vertex degree 2 ', np.sum(grti.vertex_degrees() == 2))
    label = gt.vertex_annotation();
    label_leveled = ano.convert_label(label, key='order', value='order', level=level)
    vertex_filter = label_leveled == order;
    print('vertex filter', vertex_filter.shape)
    ####################################################################################################################
    # Extract big vessels
    # sub_vertices = vs.get_sub_vertices(graph)
    # print(gr)
    print(gt)

    # g = gr.sub_graph(vertex_filter=vertex_filter)
    # print(g)
    gts = gt.sub_graph(vertex_filter=vertex_filter)
    print('gts vertex degree 2 ', np.sum(gts.vertex_degrees() == 2))
    print(gts)
    # vertex_filter = g.vertex_degrees() >= 1
    # g = gt.sub_graph(vertex_filter=vertex_filter)
    # gts = gts.sub_graph(vertex_filter=vertex_filter)
    # print(g)
    base = gt.getBase()
    return gts,gts, base


def get_penetration_arteries(graph, all_graph):
    arteries=graph.edge_property('artery')
    orientations=get_spherical_orientation(graph, all_graph)
    print('orientations ; ', orientations.shape)
    radials=(orientations[:,2]/(orientations[:,0]+orientations[:,1]+orientations[:,2]))>0.5
    print('radials ; ', radials.shape)
    penetrating_arteries=np.logical_and(arteries, radials)
    print(penetrating_arteries.shape)
    return penetrating_arteries

def get_penetrating_arteries_labels(graph, all_graph, function):
    p_a=function(graph, all_graph)



    vertex_arteries = np.zeros(graph.n_vertices)
    art_grt=graph.sub_graph(edge_filter=p_a)
    print(graph)
    print(art_grt)
    connectivity = graph.edge_connectivity();
    artery_connectivity = connectivity[p_a]
    print('edge_arteries ', p_a, artery_connectivity.shape)
    vertex_arteries[connectivity[:, 0]] = 1
    vertex_arteries[connectivity[:, 1]] = 1

    p_a_labels=gtt.label_components(art_grt.base)
    p_a_labels_hist = p_a_labels[1]
    p_a_labels=p_a_labels[0]

    p_a_labels=ggt.vertex_property_map_to_python(p_a_labels, as_array=True)
    u_l=np.unique(p_a_labels)
    for v in u_l:
        # print(i, ' / ', len(u_l))
        if p_a_labels_hist[v]<=3:
            p_a_labels[np.where(p_a_labels==v)]=-1

    print('nb of labelled componenets ; ',np.unique(p_a_labels), p_a_labels.shape)
    art_end_pt = art_grt.vertices

    art_end_pt_labels=p_a_labels#*art_end_pt*
    # np.save('/mnt/data_SSD_2to/' + brainnb + '/sbm/penetrating_art_end_point_cluster_per_region_' + ano.find_name(order, key='order') + '.npy', p_a_labels)

    # p_a_labels=np.load('/mnt/data_SSD_2to/' + brainnb + '/sbm/penetrating_art_end_point_cluster_per_region_' + ano.find_name(order, key='order') + '.npy')
    graph.add_vertex_property('p_a', p_a_labels)
    gss4_sub = graph.sub_slice((slice(0, 4000), slice(0, 7000), slice(2000, 2300)))
    b = gss4_sub.vertex_property('p_a')
    bn = np.zeros(b.shape)
    for i in np.unique(b):
        # print(i)
        bn[np.where(b == i)] = np.where(np.unique(b) == i)
    new_cmap, randRGBcolors = rand_cmap(len(np.unique(bn)), type='bright', first_color_black=False,
                                        last_color_black=False, verbose=True)
    n = len(gss4_sub.edges)
    colors = np.zeros((n, 3));

    edge_artery_label = gss4_sub.edge_property('artery')

    for i in range(b.size):
        colors[i] = randRGBcolors[int(bn[i])]
        if int(bn[i])==-1 and edge_artery_label[i] :
            colors[i] = [1., 1.0, 1.0]
    colors = np.insert(colors, 3, 1.0, axis=1)
    p = p3d.plot_graph_mesh(gss4_sub, vertex_colors=colors, n_tube_points=3);

    return p_a_labels




def diffusion_model(region_list, brain_list):
    for brainnb in brain_list:
        print(brainnb)
        # g, gts, base = initialize_brain_graph(brainnb)
        gts=ggt.load('/mnt/data_SSD_2to/' + brainnb + '/data_graph.gt')
        for region in region_list:
            order, level = region
            print(level, order, ano.find_name(order, key='order'))

            label = gts.vertex_annotation();
            label_leveled = ano.convert_label(label, key='order', value='order', level=level)
            vertex_filter = label_leveled == order;

            gss4 = gts.sub_graph(vertex_filter=vertex_filter)
            edge_arteries = gss4.edge_property('artery')
            print(edge_arteries, np.sum(edge_arteries > 0))
            art_grt = gss4.sub_graph(edge_filter=edge_arteries)
            vertex_arteries = np.zeros(gss4.n_vertices)
            vertices = gss4.vertex_degrees() < 0

            connectivity = gss4.edge_connectivity();
            artery_connectivity = connectivity[edge_arteries]
            print(artery_connectivity, artery_connectivity.shape)
            vertex_arteries[connectivity[:, 0]] = 1
            vertex_arteries[connectivity[:, 1]] = 1
            # vertex_arteries[artery_connectivity[:,0]]=1
            # vertex_arteries[artery_connectivity[:,1]] = 1
            print(vertex_arteries, np.sum(vertex_arteries))
            visited_vertices = np.array(vertex_arteries)  # .copy()
            # neighbours=vertex_arteries.copy()
            new_sum = 0
            n = 0
            while 0 in visited_vertices:
                print(n)
                neigh = np.zeros(gss4.n_vertices)
                for i in range(gss4.n_vertices):
                    if visited_vertices[i] > 0:
                        # print(gss4.vertex_neighbours(i))
                        neigh[gss4.vertex_neighbours(i)] = n
                        # if visited_vertices[gss4.vertex_neighbours(i)]==0:
                update = np.logical_and(neigh > 0, visited_vertices == 0)
                visited_vertices[update] = n
                # visited_vertices = np.minimum(visited_vertices, neigh)
                print(visited_vertices, neigh)
                print(np.sum(visited_vertices > 0), np.sum(neigh))
                if len(np.unique(neigh)) == 0:
                    print('no arteries detected')
                    break
                if not new_sum < np.sum(visited_vertices > 0):
                    print('converged', new_sum, np.sum(visited_vertices > 0))
                    break
                new_sum = np.sum(visited_vertices > 0)
                n = n + 1

            np.save(
                '/mnt/data_SSD_2to/' + brainnb + '/sbm/diffusion_cluster_per_region_' + 'iteration_' + ano.find_name(
                    order, key='order') + '.npy', visited_vertices)
    return visited_vertices


def from_e_prop2_vprop(graph, property):
    e_prop = graph.edge_property(property)
    v_prop=np.zeros(graph.n_vertices)
    connectivity = graph.edge_connectivity()
    v_prop[connectivity[e_prop==1,0]]=1
    v_prop[connectivity[e_prop == 1,1]] = 1
    # graph.add_vertex_property(property, v_prop)
    return v_prop


def stochasticPath(graph, Nb, startPt):
    N=Nb
    visited_vertices = np.zeros(graph.n_vertices)
    visited_vertices[startPt] = 1
    sP_vect=np.zeros(graph.n_vertices)
    sP_vect[startPt] = 1
    pathOK=True
    while N!=0:
        N = N - 1
        print(N)
        node = startPt
        currdiffVal = 0
        pathOK=True
        prev_node=-1
        visited_vertices_temp = np.zeros(graph.n_vertices)
        # visited_vertices[startPt] = visited_vertices[startPt] + 1
        while currdiffVal!=1000 and pathOK:
            neighbours=graph.vertex_neighbours(node)[diff_val[graph.vertex_neighbours(node)]>=diff_val[node]]
            temp=np.zeros(graph.n_vertices)
            temp[neighbours]=1
            neighbours=np.where(np.logical_and(temp, np.logical_not((visited_vertices_temp>0)==1)))[0]
            if prev_node in neighbours:
                i=np.where(neighbours==prev_node)[0]
                np.delete(neighbours, i)
            nbn=neighbours.shape[0]
            print('node ', node, neighbours)
            if nbn<1:
                pathOK=False
                break
            if nbn>=1:
                n=random.choice(range(nbn))
                visited_vertices_temp[neighbours[n]] =  1
                visited_vertices[neighbours[n]] = visited_vertices[neighbours[n]] + 1
                prev_node=node
                node = neighbours[n]
            else:
                visited_vertices_temp[neighbours[0]] = 1
                visited_vertices[neighbours[0]] = visited_vertices[neighbours[0]] + 1
                prev_node = node
                node=neighbours[0]
            currdiffVal=diff_val[prev_node]
            print(currdiffVal)
            # visited_vertices[(visited_vertices_temp>0).astype(bool)]=1
    graph.add_vertex_property('color', visited_vertices)
    graph.add_vertex_property('sP_vect', sP_vect)

    return visited_vertices



def getColorMap_from_vertex_prop(vp):
    colors = np.zeros((vp.shape[0], 4));
    print(colors.shape)
    # for i in range(b.size):
    i = 0

    # for index, b in enumerate(np.unique(diff)):
    cmax = 0.75#np.max(vp)
    cmin = np.min(vp)
    print(cmin, cmax)
    # for i in range(gss4.n_vertices):
    #     colors[i, :]=[x[i],0, 1-x[i], 1]
    jet = cm.get_cmap('viridis')#jet_r
    import matplotlib.colors as col
    cNorm = col.Normalize(vmin=cmin, vmax=cmax)
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)
    print(scalarMap.get_clim())
    colorVal = scalarMap.to_rgba(vp)
    return colorVal


def plot_diff():
    from matplotlib import cm
    for brainnb in brain_list:
        print(brainnb)
        for region in region_list:
            order, level = region
            print(level, order, ano.find_name(order, key='order'))

            label = gts.vertex_annotation();
            label_leveled = ano.convert_label(label, key='order', value='order', level=level)
            vertex_filter = label_leveled == order;

            gss4 = gts.sub_graph(vertex_filter=vertex_filter)
            diff=np.load('/mnt/data_SSD_2to/' + brainnb + '/sbm/diffusion_cluster_per_region_iteration_'+ ano.find_name(order,key='order') + '.npy')
            gss4.add_vertex_property('diff_val', diff)

            # gss4_sub = gss4.sub_slice((slice(1, 300), slice(1, 700), slice(185, 205)))
            label = gss4.vertex_annotation();
            label_leveled = ano.convert_label(label, key='order', value='order', level=10)
            vertex_filter = label_leveled == 57;
            gss4_sub = gss4.sub_graph(vertex_filter=vertex_filter)

            diff=gss4_sub.vertex_property('diff_val')
            art = gss4_sub.edge_property('artery')
            # plt.hist(diff, bins=100)
            # diff=np.array([min(d, 15) for d in diff])
            colors = np.zeros((gss4_sub.n_vertices, 4));
            print(colors.shape)
            # for i in range(b.size):
            i=0

            # for index, b in enumerate(np.unique(diff)):
            cmax=np.max(diff)
            cmin=np.min(diff)
            x=(diff.astype(float)-cmin)/cmax
            # for i in range(gss4.n_vertices):
            #     colors[i, :]=[x[i],0, 1-x[i], 1]
            jet=cm.get_cmap('jet_r')
            import matplotlib.colors as col
            cNorm = col.Normalize(vmin=cmin, vmax=cmax)
            scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)
            print(scalarMap.get_clim())
            colorVal = scalarMap.to_rgba(diff)
            connectivity = gss4_sub.edge_connectivity();
            edge_colors = (colorVal[connectivity[:, 0]] + colorVal[connectivity[:, 1]]) / 2.0;
            edge_colors[art > 0] = [1., 0.0, 0.0, 1.0]
            # colormpa = cm.jet(np.linspace(0,cmax,10))
            # print(colormpa)
            # colors=np.array([colormpa[d] for d in diff])
            print(colorVal)
            from matplotlib import pyplot as plt
            fig, ax = plt.subplots(1, 1)
            colo=scalarMap.colorbar
            from matplotlib import colors, colorbar


            cb = colorbar.ColorbarBase(ax, cmap=jet, norm=cNorm, spacing='proportional', ticks=None,
                                       format='%1i', orientation=u'horizontal')
            plt.show()
            # p = p3d.plot_graph_mesh(gss4_sub, edge_colors=edge_colors, n_tube_points=3);

if __name__ == "__main__":
    # execute only if run as a script
    brain_list = ['190408_44R']#190912_SC3
    brainnb = brain_list[0]
    region_list = []
    region_list = [(1006, 3), (580, 5), (650, 5), (724, 5), (811, 4), (875, 4), (6, 6), (463, 6), (388, 6)]
    reg_colors = {1006: 'gold', 580: 'skyblue', 650: 'indianred', 724: 'violet', 811: 'darkorchid',
                  875: 'mediumslateblue', 6: 'forestgreen', 463: 'lightgreen', 388: 'turquoise'}
    region_list=[(6, 6)]
    # graph = initialize_brain_graph(brainnb)
    diffusion_model(region_list, brain_list)
    # diffusion_model_from_arteries(region_list, brain_list)
    plot_diff()

    v_veins = from_e_prop2_vprop(gss4, 'vein')
    v_arteries=from_e_prop2_vprop(gss4, 'artery')
    v_art_id = np.where(v_arteries == 1)[0]
    diff_val=np.load('/mnt/data_SSD_2to/'+brainnb+'/sbm/diffusion_cluster_per_region_iteration_Isocortex.npy')

    startPt = random.choice(v_art_id)

    visited_vertices=stochasticPath(gss4, 100, startPt)
    print(startPt)
    stocha_path = gss4.sub_graph(vertex_filter=visited_vertices>0)
    colorval=getColorMap_from_vertex_prop(stocha_path.vertex_property('color'))
    v_arteries=from_e_prop2_vprop(stocha_path, 'artery')
    v_veins = from_e_prop2_vprop(stocha_path, 'vein')
    colorval[v_arteries==1]= [1., 0.0, 0.0, 1.0]
    colorval[v_veins==1] = [0.0, 1.0,0.0,1.0]
    colorval[stocha_path.vertex_property('sP_vect') == 1] = [0.0, 0.0, 0.0, 1.0]
    p = p3d.plot_graph_mesh(stocha_path, vertex_colors=colorval, n_tube_points=3);

    indices = gss4.vertex_property('indices')
    u, c = np.unique(indices, return_counts=True)

    index = random.choice(u)
    print(index)
    vp = indices == index
    g2plot = gss4.sub_graph(vertex_filter=vp)
    p = p3d.plot_graph_mesh(g2plot, n_tube_points=3);

    #sbm VS art terr

    for brainnb in brain_list:
        print(brainnb)
        for region in region_list:
            order, level = region
            print(level, order, ano.find_name(order, key='order'))

            label = gts.vertex_annotation();
            label_leveled = ano.convert_label(label, key='order', value='order', level=level)
            vertex_filter = label_leveled == order;

            gss4 = gts.sub_graph(vertex_filter=vertex_filter)
            artterr=np.load('/data_SSD_2to/' + brainnb + '/sbm/diffusion_penetrating_vessel_overlap_end_point_cluster_per_region_iteration_Isocortex.npy')
            # artterr=np.load('/mnt/data_SSD_2to/' + brainnb + '/sbm/diffusion_penetrating_art_end_point_cluster_per_region_iteration_' + ano.find_name(order, key='order') + '.npy')
            # sbm = np.load('/mnt/data_SSD_2to/' + brainnb + '/sbm/diffusion_cluster_per_region_iteration_' + ano.find_name(order, key='order') + '.npy')
            #
            # modules = []
            # n = 1
            # for i in range(n):
            #     print(i)
            #     blocks = np.load(
            #         '/mnt/data_SSD_2to/' + brainnb + '/sbm/blockstate_full_brain_levelled_' + str(i) + '.npy')
            #     if i == 0:
            #         modules = blocks
            #     else:
            #         modules = np.array([blocks[b] for b in modules])
            i=0
            modules = np.load('/mnt/data_SSD_2to/' + brainnb + '/sbm/blockstate_full_brain_levelled_690_clusteres' + str(i) + '.npy')
            modules = np.load('/mnt/data_SSD_2to/' + brainnb + '/sbm/blockstate_full_brain_levelled_690_clusteres_Ls'+ str(i) + '.npy')
            gss4.add_vertex_property('blocks', modules)
            gss4.add_vertex_property('artterr', artterr)

            # u_sbm = np.unique(modules)
            u_arterr = np.unique(artterr)


            cluster=np.random.choice(u_arterr, 1)[0]
            print(cluster)
            vp = artterr==cluster
            # g2plot=gss4.sub_graph(vertex_filter=vp)
            g2plot = gss4.sub_slice((slice(0, 4000), slice(0, 7000), slice(2000, 2300)))
            print(g2plot)
            b = g2plot.vertex_property('blocks')
            print(np.unique(b))
            bn = np.zeros(b.shape)
            for i in np.unique(b):
                # print(i)
                bn[np.where(b == i)] = np.where(np.unique(b) == i)
            new_cmap, randRGBcolors = rand_cmap(len(np.unique(bn)), type='bright', first_color_black=False,last_color_black=False, verbose=True)
            n = len(g2plot.vertices)
            colorval = np.zeros((n, 3));
            for i in range(b.size):
                colorval[i] = randRGBcolors[int(bn[i])]
            # colorval = getColorMap_from_vertex_prop(g2plot.vertex_property('artterr'))
            p = p3d.plot_graph_mesh(g2plot, vertex_colors=colorval, n_tube_points=3);


mockoverlap=np.load('/data_SSD_2to/' + brainnb + '/sbm/diffusion_penetrating_vessel_mock_overlap_terr_cluster_per_region_iteration_Isocortex.npy')
gss4.add_vertex_property('mockoverlap', mockoverlap)
mockoverlap_u, c = np.unique(mockoverlap, return_counts=True)


a_terr_u, c = np.unique(artterr, return_counts=True)
print(a_terr_u.shape)
print(c)
plt.figure()
plt.hist(c, bins=200, alpha = 0.5)
# plt.xlim([0,1500])
# sbm_u, c = np.unique(modules, return_counts=True)
# print(sbm_u.shape)
# print(c)
# plt.hist(c, bins=50, alpha = 0.5)
# plt.xlim([0,1500])


# from ClearMap.DiffusionPenetratingArteriesCortex import get_penetrating_arteries_labels, get_penetration_veins_dustance_surface
# from ClearMap.DiffusionPenetratingArteriesCortex import get_penetration_arteries_dustance_surface, get_penetrating_veins_labels


sbm_edges=[]
sbm_art=[]
sbm_vein=[]
sbm_density=[]
sbm_degree=[]
sbm_color=[]
sbm_avg_deg=[]


a_terr_edges=[]
a_terr_art=[]
a_terr_vein=[]
a_terr_density=[]
a_terr_degree=[]
a_terr_color=[]
a_terr_avg_deg=[]

function=get_penetration_arteries_dustance_surface
function2=get_penetration_veins_dustance_surface

for e in mockoverlap_u:
    vp=modules==e
    g=gss4.sub_graph(vertex_filter=vp)
    if g.n_vertices>0:
        sbm_edges.append(g.n_edges)
        # aepl = get_penetrating_arteries_labels(g, gss4, function)
        # sbm_art.append(np.unique(aepl).shape[0])
        # vepl=get_penetrating_veins_labels(g, gss4, function2)
        # sbm_vein.append(np.unique(vepl).shape[0])
        sbm_density.append(g.n_edges/(g.n_vertices*(g.n_vertices+1)))
        d_u, d_c = np.unique(g.vertex_degrees(), return_counts=True)
        sbm_degree.append([d_u,d_c])
        a_t=g.vertex_property('artterr')
        sbm_color.append(np.unique(a_t).shape[0])
        sbm_avg_deg.append(np.sum(g.vertex_degrees())/g.n_vertices)



for e in a_terr_u:
    vp=artterr==e
    g=gss4.sub_graph(vertex_filter=vp)
    if g.n_vertices>0:
        a_terr_edges.append(g.n_edges)
        # aepl = get_penetrating_arteries_labels(g, gss4, function, 'art')
        # a_terr_art.append(np.unique(aepl).shape[0])
        # vepl=get_penetrating_veins_labels(g, gss4, function2)
        # a_terr_vein.append(np.unique(vepl).shape[0])
        a_terr_density.append(g.n_edges/(g.n_vertices*(g.n_vertices+1)))
        d_u, d_c = np.unique(g.vertex_degrees(), return_counts=True)
        a_terr_degree.append([d_u,d_c])
        # a_t=g.vertex_property('blocks')
        # a_terr_color.append(np.unique(a_t).shape[0])
        a_terr_avg_deg.append(np.sum(g.vertex_degrees()) / g.n_vertices)
#
plt.figure()
plt.hist(gss4.vertex_degrees(), bins=50)
plt.title('isocortex vertex degree histogram')

plt.figure()
# plt.hist(a_terr_edges, bins=100, alpha = 0.5, density=True)
# plt.hist(sbm_edges, bins=20, alpha = 0.5, density=True)
results, edges = np.histogram(a_terr_edges, density=True, bins=100)
binWidth = edges[1] - edges[0]
plt.bar(edges[:-1], results*binWidth, binWidth, alpha = 0.5)

results, edges = np.histogram(sbm_edges, density=True, bins=20)
binWidth = edges[1] - edges[0]
plt.bar(edges[:-1], results*binWidth, binWidth, alpha = 0.5)

plt.title('real VS mock overlap edge per cluster')

# plt.figure()
# plt.hist(a_terr_art-np.ones(3127), bins=10, alpha = 0.5)
# plt.hist(sbm_art-np.ones(600), alpha = 0.5)
# plt.title('real VS mock overlap arteries per cluster')
#
# plt.figure()
# plt.hist(a_terr_vein-np.ones(3127),bins=10, alpha = 0.5)
# plt.hist(sbm_vein-np.ones(600), alpha = 0.5)
# plt.title('real VS mock overlap vein per cluster')

plt.figure()
# plt.hist(a_terr_density, bins=100, alpha = 0.5, density=True)
# plt.hist(sbm_density, bins=100, alpha = 0.5, density=True)
results, edges = np.histogram(a_terr_density, density=True)
binWidth = edges[1] - edges[0]
plt.bar(edges[:-1], results*binWidth, binWidth, alpha = 0.5)

results, edges = np.histogram(sbm_density, density=True)
binWidth = edges[1] - edges[0]
plt.bar(edges[:-1], results*binWidth, binWidth, alpha = 0.5)

plt.title('real VS mock overlap densty per cluster')
#
# plt.figure()
# plt.hist(a_terr_color, bins=100, alpha = 0.5)
# plt.hist(sbm_color, bins=100, alpha = 0.5)
# plt.title('real VS mock overlap overlap per cluster')

plt.figure()
# plt.hist(a_terr_avg_deg, bins=100, alpha = 0.5, density=True)
# plt.hist(sbm_avg_deg, bins=100, alpha = 0.5, density=True)
results, edges = np.histogram(a_terr_avg_deg, density=True)
binWidth = edges[1] - edges[0]
plt.bar(edges[:-1], results*binWidth, binWidth, alpha = 0.5)

results, edges = np.histogram(sbm_avg_deg, density=True)
binWidth = edges[1] - edges[0]
plt.bar(edges[:-1], results*binWidth, binWidth, alpha = 0.5)

plt.title('real VS mock overlap avg vertex degree per cluster')

a_terr_degree=np.array(a_terr_degree)
# sbm_degree=np.array(sbm_degree)

import seaborn as sns

mat=np.zeros((10, sbm_degree.shape[0]))
for j,l in enumerate(sbm_degree[:,0]):
    for i, c in enumerate(l):
        if c<10:
            mat[c, j]=sbm_degree[j,1][i]
# plt.figure()
# hm = sns.heatmap(mat)#, row_cluster=True

plt.figure()
hm = sns.clustermap(mat, row_cluster=True)

mat=np.zeros((10, a_terr_degree.shape[0]))
for j,l in enumerate(a_terr_degree[:,0]):
    for i, c in enumerate(l):
        if c<10:
            mat[c, j]=a_terr_degree[j,1][i]

plt.figure()
hm = sns.clustermap(mat, row_cluster=True)