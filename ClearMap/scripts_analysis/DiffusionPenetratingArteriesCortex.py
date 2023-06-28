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
# from ClearMap.KirschoffAnalysis import from_e_prop2_vprop
import math
pi=math.pi
import pickle
region_list=[(6, 6)]

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


def get_penetration_veins(graph, all_graph):
    veins=graph.edge_property('vein')

    orientations=get_spherical_orientation(graph, all_graph)
    print('orientations ; ', orientations.shape)
    radials=(orientations[:,2]/(orientations[:,0]+orientations[:,1]+orientations[:,2]))>0.5
    print('radials ; ', radials.shape)
    penetrating_veins=np.logical_and(veins, radials)
    print(penetrating_veins.shape)
    return penetrating_veins

def get_penetration_veins_dustance_surface(graph, all_graph):
    veins=graph.edge_property('vein')
    distance_from_suface=graph.edge_property('distance_to_surface')
    print('distance_to_surface ; ', distance_from_suface.shape)

    penetrating_veins=np.logical_and(veins, distance_from_suface>5)
    print(penetrating_veins.shape)
    return penetrating_veins

def get_penetrating_veins_labels(graph, all_graph, function):
    p_a = function(graph, all_graph)

    vertex_arteries = np.zeros(graph.n_vertices)
    art_grt = graph.sub_graph(edge_filter=p_a)
    print(graph)
    print(art_grt)
    connectivity = graph.edge_connectivity();
    artery_connectivity = connectivity[p_a]
    print('edge_arteries ', p_a, artery_connectivity.shape)
    vertex_arteries[connectivity[:, 0]] = 1
    vertex_arteries[connectivity[:, 1]] = 1

    p_a_labels = gtt.label_components(art_grt.base)
    p_a_labels_hist = p_a_labels[1]
    p_a_labels = p_a_labels[0]

    p_a_labels = ggt.vertex_property_map_to_python(p_a_labels, as_array=True)
    u_l = np.unique(p_a_labels)
    for v in u_l:
        # print(i, ' / ', len(u_l))
        if p_a_labels_hist[v] <= 3:
            p_a_labels[np.where(p_a_labels == v)] = -1

    print('nb of labelled componenets ; ', np.unique(p_a_labels), p_a_labels.shape)
    art_end_pt = art_grt.vertices

    art_end_pt_labels = p_a_labels  # *art_end_pt*
    # np.save('/data_SSD_2to/' + brainnb + '/sbm/penetrating_vein_end_point_cluster_per_region_' + ano.find_name(order, key='order') + '.npy', p_a_labels)
    return p_a_labels

def remove_surface(graph, width):
    distance_from_suface = graph.edge_property('distance_to_surface')
    ef=distance_from_suface>width
    g=graph.sub_graph(edge_filter=ef)
    return g


def get_penetration_arteries_dustance_surface(graph, all_graph):
    arteries=graph.edge_property('artery')
    distance_from_suface=graph.edge_property('distance_to_surface')
    print('distance_to_surface ; ', distance_from_suface.shape)

    penetrating_arteries=np.logical_and(arteries, distance_from_suface>5)
    print(penetrating_arteries.shape)
    return penetrating_arteries

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
    # np.save('/mnt/data_SSD_2to/' + brainnb + '/sbm/diffusion_penetrating_art_end_point_cluster_per_region_iteration_' + ano.find_name(order, key='order') + '.npy', p_a_labels)

    # p_a_labels=np.load('/mnt/data_SSD_2to/' + brainnb + '/sbm/penetrating_art_end_point_cluster_per_region_' + ano.find_name(order, key='order') + '.npy')
    graph.add_vertex_property('p_a', p_a_labels)
    # gss4_sub = graph.sub_slice((slice(0, 4000), slice(0, 7000), slice(2000, 2300)))
    # b = gss4_sub.vertex_property('p_a')
    # bn = np.zeros(b.shape)
    # for i in np.unique(b):
    #     # print(i)
    #     bn[np.where(b == i)] = np.where(np.unique(b) == i)
    # new_cmap, randRGBcolors = rand_cmap(len(np.unique(bn)), type='bright', first_color_black=False,
    #                                     last_color_black=False, verbose=True)
    # n = len(gss4_sub.edges)
    # colors = np.zeros((n, 3));
    #
    # edge_artery_label = gss4_sub.edge_property('artery')
    #
    # for i in range(b.size):
    #     colors[i] = randRGBcolors[int(bn[i])]
    #     if int(bn[i])==-1 and edge_artery_label[i] :
    #         colors[i] = [1., 1.0, 1.0]
    # colors = np.insert(colors, 3, 1.0, axis=1)
    # p = p3d.plot_graph_mesh(gss4_sub, vertex_colors=colors, n_tube_points=3);

    return p_a_labels

def diffusion_through_penetrating_arteries(graph, function, vesselfunction, vesseltype, graph_dir, feature='cluster'):
    region_list=[(1,1)]
    for regions in region_list:
        order, level = regions
        print(level, order, ano.find_name(order, key='order'))

        label = graph.vertex_annotation();
        label_leveled = ano.convert_label(label, key='order', value='order', level=level)
        vertex_filter = label_leveled == order;

        # with mutex:
        gss4 = graph.sub_graph(vertex_filter=vertex_filter)
        #callable #get_penetration_arteries_dustance_surface OR get_penetration_arteries
        aepl=vesselfunction(gss4, graph, function)
        u, c=np.unique(aepl, return_counts=True)
        print('aepl : ', u, c)
        # print(brainnb)
        visited_vertices = np.zeros(gss4.n_vertices)  # .copy()
        m = 0
        for i in range(len(aepl)):
            if aepl[i] != -1:
                # print(i)
                m = m + 1
                visited_vertices[i] = aepl[i]
        # neighbours=vertex_arteries.copy()
        artveinVess=from_e_prop2_vprop(gss4, 'artery')#np.logical_or(from_e_prop2_vprop(gss4, 'vein'), from_e_prop2_vprop(gss4, 'artery'))
        veinVertices=from_e_prop2_vprop(gss4, 'vein')
        print('visited_vertices ', visited_vertices, np.sum(visited_vertices))
        if feature=='cluster':
            new_sum = 0
            n = 0
            while 0 in visited_vertices:
                print(n)
                neigh = np.zeros(gss4.n_vertices)
                for i in range(gss4.n_vertices):
                    if visited_vertices[i] > 0:
                        if veinVertices[i] < 1:
                            nonstopingvertex=np.asarray(artveinVess[gss4.vertex_neighbours(i)]==0).nonzero()[0]
                            # print(gss4.vertex_neighbours(i))
                            neigh[gss4.vertex_neighbours(i)[nonstopingvertex]] = visited_vertices[i]
                            # if visited_vertices[gss4.vertex_neighbours(i)]==0:
                update = np.logical_and(neigh > 0, visited_vertices == 0)
                visited_vertices[update] = neigh[update]
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
            np.save(graph_dir+'/sbm/diffusion_penetrating_vessel_'+vesseltype+'_end_point_cluster_per_region_iteration_' + ano.find_name(order, key='order') +'_graph_corrected' +'.npy', visited_vertices)

        elif feature=='distance':
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
            np.save(graph_dir + '/sbm/diffusion_penetrating_vessel_' + vesseltype + '_end_point_cluster_per_region_iteration_' + ano.find_name(order, key='order') + '_distance_graph_corrected' + '.npy', visited_vertices)

    return visited_vertices


def diffusion_labelled(graph, label, vesseltype):
        new_sum = 0
        n = 0
        visited_vertices = label
        print('visited_vertices ', visited_vertices, np.sum(visited_vertices))
        while 0 in visited_vertices:
            print(n)
            neigh = np.zeros(graph.n_vertices)
            for i in range(graph.n_vertices):
                if visited_vertices[i] > 0:
                    # print(gss4.vertex_neighbours(i))
                    neigh[graph.vertex_neighbours(i)] = visited_vertices[i]
                    # if visited_vertices[gss4.vertex_neighbours(i)]==0:
            update = np.logical_and(neigh > 0, visited_vertices == 0)
            visited_vertices[update] = neigh[update]
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
        np.save('/data_SSD_2to/' + brainnb + '/sbm/diffusion_penetrating_'+vesseltype+'_end_point_cluster_per_region_iteration_' + ano.find_name(order, key='order') + '.npy', visited_vertices)
        return visited_vertices

# diffusion_through_penetrating_arteries_vector= np.load('/mnt/data_SSD_2to/190408_44R/sbm/diffusion_penetrating_art_end_point_cluster_per_region_iteration_Isocortex.npy')
# u, c = np.unique(diffusion_through_penetrating_arteries_vector, return_counts=True)



# poisson function, parameter lamb is the fit parameter
from scipy.optimize import curve_fit
from scipy.special import factorial

def poisson(k, lamb):
    return (lamb**k/factorial(k)) * np.exp(-lamb)


#
# plt.figure()
# entries, bin_edges, patches=plt.hist(c, bins=500, color='#1F9D5A')
# # calculate binmiddles
# bin_middles = 0.5*(bin_edges[1:] + bin_edges[:-1])
# # fit with curve_fit
# parameters, cov_matrix = curve_fit(poisson, bin_middles, entries, bounds=(150, 300))
# print(parameters)
# # plot poisson-deviation with fitted parameter
# x_plot = np.linspace(0, 1400, 50)
# plt.plot(x_plot, poisson(x_plot, *parameters), 'r-', lw=2)
# plt.xlim([0,1400])



def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=True):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    from matplotlib.colors import LinearSegmentedColormap
    import colorsys
    import numpy as np


    if type not in ('bright', 'soft'):
        print ('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in xrange(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        # cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
        #                            boundaries=bounds, format='%1i', orientation=u'horizontal')

    return random_colormap, randRGBcolors



def plot_penetrating_artery(work_dir, control):
    graph = ggt.load(work_dir + '/' + control + '/' + 'data_graph_correctedIsocortex.gt')
    diff = np.load(
        work_dir + '/' + control + '/sbm/' + 'diffusion_penetrating_vessel_overlap_end_point_cluster_per_region_iteration_Isocortex_graph_corrected' + '.npy')
    graph.add_vertex_property('overlap', diff)
    cluster = getOverlaps(work_dir, control, graph, save_overlap=False)
    graph.add_vertex_property('overlap_art', cluster[:, 0])
    artery = from_e_prop2_vprop(graph, 'artery')
    art_g = graph.sub_graph(vertex_filter=artery)
    b = art_g.vertex_property('overlap_art')
    print(np.unique(b))
    bn = np.zeros(b.shape)
    for i in np.unique(b):
        # print(i)
        bn[np.where(b == i)] = np.where(np.unique(b) == i)
    new_cmap, randRGBcolors = rand_cmap(len(np.unique(bn)), type='bright', first_color_black=False,
                                        last_color_black=False, verbose=True)
    print(len(np.unique(bn)), len(np.unique(b)))
    cmax = np.max(bn)
    cmin = np.min(bn)
    n = len(art_g.vertices)
    colors = np.zeros((n, 3));
    for i in range(b.size):
        colors[i] = randRGBcolors[int(bn[i])]
    colors = np.insert(colors, 3, 1.0, axis=1)
    # vertex_filter = label_leveled == 146;
    # gss4 = g.sub_graph(vertex_filter=vertex_filter)
    # colors4=colors[vertex_filter]
    # p = p3d.plot_graph_mesh(gts, vertex_colors=colors, n_tube_points=3);
    print(art_g)
    p = p3d.plot_graph_mesh(art_g, vertex_colors=colors, n_tube_points=3);


def plot_vertex_hist(gts, brainnb):
    from matplotlib import cm
    # for brainnb in brain_list:
    print(brainnb)
    for region in region_list:
        order, level = region
        print(level, order, ano.find_name(order, key='order'))

        label = gts.vertex_annotation();
        label_leveled = ano.convert_label(label, key='order', value='order', level=level)
        vertex_filter = label_leveled == order;

        gss4 = gts.sub_graph(vertex_filter=vertex_filter)
        diff=np.load('/data_SSD_2to/' + brainnb + '/sbm/diffusion_penetrating_vessel_overlap_end_point_cluster_per_region_iteration_' + ano.find_name(order, key='order') +'_graph_corrected'+ '.npy')
        # diff=np.load('/mnt/data_SSD_2to/' + brainnb + '/sbm/diffusion_penetrating_art_end_point_cluster_per_region_'+ ano.find_name(order,key='order') + '.npy')
        gss4.add_vertex_property('diff_val', diff)
        diff=gss4.vertex_property('diff_val')
        u, c = np.unique(diff, return_counts=True)#diffusion_through_penetrating_arteries_vector
        nb_edge_per_art_terr=np.zeros(u.shape[0])#gss4.n_vertices
        nb_vvertex_per_art_terr=np.zeros(u.shape[0])
        density_vect=np.zeros(u.shape[0])
        orientation_vect = np.zeros(u.shape[0])

        x = gss4.vertex_coordinates()[:, 0]
        y = gss4.vertex_coordinates()[:, 1]
        z = gss4.vertex_coordinates()[:, 2]
        x_g = gts.vertex_coordinates()[:, 0]
        y_g = gts.vertex_coordinates()[:, 1]
        z_g = gts.vertex_coordinates()[:, 2]
        center = np.array([np.mean(x_g), np.mean(y_g), np.mean(z_g)])
        x = x - np.mean(x_g)
        y = y - np.mean(y_g)
        z = z - np.mean(z_g)
        spherical_coord = np.array(cart2sph(x, y, z, ceval=ne.evaluate)).T
        connectivity = gss4.edge_connectivity()
        x_s = spherical_coord[:, 0]
        y_s = spherical_coord[:, 1]
        z_s = spherical_coord[:, 2]
        spherical_ori = np.array([x_s[connectivity[:, 1]] - x_s[connectivity[:, 0]], y_s[connectivity[:, 1]] - y_s[connectivity[:, 0]],z_s[connectivity[:, 1]] - z_s[connectivity[:, 0]]]).T
        spherical_ori = preprocessing.normalize(spherical_ori, norm='l2')
        spherical_ori = np.abs(spherical_ori)
        gss4.add_edge_property('spherical_ori', spherical_ori)



        i=0
        for i, iu in enumerate(u):
            print(i)
            vf=diff==i
            g = gss4.sub_graph(vertex_filter=vf)
            nb_edge_per_art_terr[i]=g.n_edges
            nb_vvertex_per_art_terr[i]=g.n_vertices
            oris=g.edge_property('spherical_ori')
            ori=np.sum(oris, axis=0)

            # orientation_vect[vf]=(ori[0]+ori[1])/(ori[0]+ori[1]+ori[2])
            if g.n_edges==0:
                nedg=1
                orientation_vect[i] = 0
            else:
                nedg=g.n_edges
                orientation_vect[i] = (ori[0] + ori[1]) / (ori[0] + ori[1] + ori[2])
            density=g.n_vertices/nedg
            density_vect[i]=density
            i=i+1

    # nb_edge_per_art_terr=np.array(nb_edge_per_art_terr)
    # nb_vvertex_per_art_terr=np.array(nb_vvertex_per_art_terr)
    return nb_edge_per_art_terr, density_vect, nb_vvertex_per_art_terr, orientation_vect





def plot_diff_from_penetrating_arteries(gts, vesseltype):
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
            diff=np.load('/data_SSD_2to/' + brainnb + '/sbm/diffusion_penetrating_vessel_'+vesseltype+'_end_point_cluster_per_region_iteration_' + ano.find_name(order, key='order') +'_graph_corrected' +'.npy')
            # diff=np.load('/mnt/data_SSD_2to/' + brainnb + '/sbm/diffusion_penetrating_art_end_point_cluster_per_region_'+ ano.find_name(order,key='order') + '.npy')
            gss4.add_vertex_property('diff_val', diff)

            # gss4 = gts.sub_slice((slice(1,300), slice(300,308), slice(1,480)))
            # gss4 = gts.sub_slice((slice(1, 300), slice(40, 500), slice(65, 85)))
            gss4_sub = gss4#gss4.sub_slice((slice(1, 300), slice(1, 700), slice(175, 185)))
            gss4_sub = gss4.sub_slice((slice(0, 4000), slice(0, 7000), slice(2000, 2300)));
            b = gss4_sub.vertex_property('diff_val')
            print(np.unique(b))
            bn = np.zeros(b.shape)
            for i in np.unique(b):
                # print(i)
                bn[np.where(b == i)] = np.where(np.unique(b) == i)
            new_cmap, randRGBcolors = rand_cmap(len(np.unique(bn)), type='bright', first_color_black=False,
                                                last_color_black=False, verbose=True)
            print(len(np.unique(bn)), len(np.unique(b)))
            cmax = np.max(bn)
            cmin = np.min(bn)
            n = len(gss4_sub.vertices)
            colors = np.zeros((n, 3));
            for i in range(b.size):
                colors[i] = randRGBcolors[int(bn[i])]
            colors = np.insert(colors, 3, 1.0, axis=1)
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


def get_art_vein_overlap(graph, vein_clusters, art_clusters):
    overlap=np.zeros(graph.n_vertices)
    cluster=np.zeros((graph.n_vertices, 2))
    cluster[:, 0]=art_clusters
    cluster[:, 1]=vein_clusters
    u, indices=np.unique(cluster, axis=0, return_inverse=True)
    overlap=indices
    return indices

def extract_AnnotatedRegion(graph, region):
    order, level = region
    print(level, order, ano.find_name(order, key='order'))

    label = graph.vertex_annotation();
    label_leveled = ano.convert_label(label, key='order', value='order', level=level)
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



def removeSpuriousBranches(graph ,rmin=1, length=5):
    radii = graph.vertex_radii()
    conn=graph.edge_connectivity()
    degrees_m = graph.vertex_degrees()
    deg1 = np.asarray(degrees_m <= 1).nonzero()[0]
    rad1 = np.asarray(radii <= rmin).nonzero()[0]
    lengths = graph.edge_geometry_lengths()
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

    vertex2rm=np.array(vertex2rm)
    print(vertex2rm.shape)
    ef=np.ones(graph.n_vertices)
    ef[vertex2rm]=0
    graph=graph.sub_graph(vertex_filter=ef)
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


def createHighLevelGraph(gss4):
    # for brainnb in brain_list:
    #     print(brainnb)
    #     for region in region_list:
    #         order, level = region
    #         print(level, order, ano.find_name(order, key='order'))
    #
    #         label = gts.vertex_annotation();
    #         label_leveled = ano.convert_label(label, key='order', value='order', level=level)
    #         vertex_filter = label_leveled == order;
    #
    #         gss4 = gts.sub_graph(vertex_filter=vertex_filter)
    diff=np.load('/data_SSD_2to/' + brainnb + '/sbm/diffusion_penetrating_vessel_overlap_end_point_cluster_per_region_iteration_' + ano.find_name(order, key='order') +'_graph_corrected'+ '.npy')
    # diff=np.load('/mnt/data_SSD_2to/' + brainnb + '/sbm/diffusion_penetrating_art_end_point_cluster_per_region_'+ ano.find_name(order,key='order') + '.npy')
    gss4.add_vertex_property('diff_val', diff)
    diff=gss4.vertex_property('diff_val')
    coordinates=gss4.vertex_coordinates()
    u, c = np.unique(diff, return_counts=True)#diffusion_through_penetrating_arteries_vector
    high_lev_coord=np.zeros((u.shape[0], 3))
    conn = gss4.edge_connectivity()
    edges_all=[[diff[conn[i, 0]], diff[conn[i,1]]] for i in range(conn.shape[0])]
    edges_all=np.array(edges_all)
    g = ggt.Graph(n_vertices=u.shape[0], directed=False)
    # radii = np.zeros((0, 1), dtype=int)
    cluster_size = np.zeros(g.n_vertices)
    print(g)
    for i, uc in enumerate(u):
        vf=diff==uc
        high_lev_coord[i]=np.sum(coordinates[vf], axis=0)/c[i]
        cluster_size[uc] = c[i]

    intra_edges=np.asarray([edges_all[i, 0]==edges_all[i, 1] for i in range(edges_all.shape[0])]).nonzero()
    intra_edges=edges_all[intra_edges]

    inter_edges = np.asarray([edges_all[i, 0]!=edges_all[i, 1] for i in range(edges_all.shape[0])]).nonzero()
    inter_edges = edges_all[inter_edges]

    eu, ec=np.unique(inter_edges, return_counts=True, axis=0)
    g.add_edge(eu)
    # radii=np.ones(edges_all.shape[0])
    print(g)
    g.set_edge_geometry(name='radii', values=ec)
    g.set_vertex_coordinates(high_lev_coord)


    eu, ec=np.unique(intra_edges, return_counts=True, axis=0)

    inter_connectivity=np.zeros(g.n_vertices)

    for i, u in enumerate(eu):
        inter_connectivity[u]=ec[i]


    g.add_vertex_property('inter_connectivity', inter_connectivity)
    g.add_vertex_property('cluster_size', cluster_size)
    return g



def graphCorrection(graph, graph_dir, region, save=True):
    # artery = graph.edge_property('artery');
    # vein = graph.edge_property('vein');
    #
    # u, c = np.unique(vein, return_counts=True)
    # print(u, c / graph.n_edges)
    #
    # u, c = np.unique(artery, return_counts=True)
    # print(u, c / graph.n_edges)
    #
    # graph, vf=extract_AnnotatedRegion(graph, region)
    # #
    # print(graph)
    # art = graph.edge_property('artery')
    # vein = graph.edge_property('vein')
    #
    # u, c = np.unique(art, return_counts=True)
    # print(u, c / graph.n_edges)
    #
    # u, c = np.unique(vein, return_counts=True)
    # print(u, c / graph.n_edges)

    gss = graph.largest_component()

    degrees_m = gss.vertex_degrees()
    deg4 = np.asarray(degrees_m >= 5).nonzero()[0]
    print('deg4 init ', deg4.shape[0] / gss.n_vertices)

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
    deg4 = np.asarray(degrees_m >= 5).nonzero()[0]
    print('deg4 exit ', deg4.shape[0] / g.n_vertices)

    # artery = g.edge_property('artery');
    # vein = g.edge_property('vein');
    #
    # u, c = np.unique(vein, return_counts=True)
    # print(u, c / g.n_edges)
    #
    # u, c = np.unique(artery, return_counts=True)
    # print(u, c / g.n_edges)
    if save:
        g.save(graph_dir+'/data_graph_corrected'+ano.find_name(region[0], key='id')+'.gt')
    return g

def getArteryCluster(workdir, graph_dir):
    art_clusters = np.load(
        workdir + '/' + graph_dir + '/sbm/diffusion_penetrating_vessel_art_end_point_cluster_per_region_iteration_Isocortex_graph_corrected.npy')
    return art_clusters

def getOverlaps(workdir, graph_dir, graph, save_overlap=False):
    vein_clusters = np.load(
        workdir+'/'+graph_dir+'/sbm/diffusion_penetrating_vessel_vein_end_point_cluster_per_region_iteration_brain_distance_graph_corrected.npy')
    art_clusters = np.load(
         workdir+'/'+graph_dir+'/sbm/diffusion_penetrating_vessel_art_end_point_cluster_per_region_iteration_brain_graph_corrected.npy')
    indices = get_art_vein_overlap(graph, vein_clusters, art_clusters)
    if save_overlap:
        np.save(
             workdir+'/'+graph_dir+'/sbm/diffusion_penetrating_vessel_overlap_end_point_cluster_per_region_iteration_Isocortex_graph_corrected.npy',
            indices)

    overlap = np.zeros(graph.n_vertices)
    cluster = np.zeros((graph.n_vertices, 2))
    cluster[:, 0] = art_clusters
    cluster[:, 1] = vein_clusters
    u, c = np.unique(cluster, axis=0, return_counts=True)
    return cluster

def getExtendedOverlaps(graph, cluster, clus, plot=False):
    arteries = from_e_prop2_vprop(graph, 'artery')
    veins = from_e_prop2_vprop(graph, 'vein')
    # u, c = np.unique(cluster, axis=0, return_counts=True)
    #
    # # import random
    # # b_c = np.asarray(np.logical_and(c > 600, c <= 6000)).nonzero()[0]
    # # r = random.choice(b_c)
    # r=int(np.asarray(u==id).nonzero()[0][0])
    #
    # print(r, c[r])
    vf = np.zeros(graph.n_vertices)

    capi = np.logical_and(np.asarray(cluster[:, 0] == clus[0]), np.asarray(cluster[:, 1] == clus[1]))
    capi = graph.expand_vertex_filter(capi, steps = 2)
    vf[capi] = 3
    art=np.logical_and(cluster[:, 0] == clus[0], arteries)
    ve=np.logical_and(cluster[:, 1] == clus[1], veins)

    if np.sum(art)<=300:
        vf[art] = 1
    if np.sum(ve) <= 300:
        vf[ve] = 2

    vertex_filter = vf > 0
    graph.add_vertex_property('vf', vf)
    g_clus = graph.sub_graph(vertex_filter=vertex_filter)

    v = g_clus.vertex_property('vf')
    colorVal = np.zeros((v.shape[0], 4))
    red_blue_map = {1: [1.0, 0, 0, 1.0], 3: [0.0, 1.0, 0.0, 1.0], 2: [0.0, 0, 1.0, 1.0]}
    for i, c in enumerate(v):
        # print(j)
        if c == 1:
            colorVal[i] = red_blue_map[1]
        elif c == 2:
            colorVal[i] = red_blue_map[2]
        elif c == 3:
            colorVal[i] = red_blue_map[3]
    if plot:
        print('plotting')
        p = p3d.plot_graph_mesh(g_clus, vertex_colors=colorVal)
    else:
        print('values: 2: vein, 1:arteries, 3:capillaries', g_clus)
        return vf


if __name__ == "__main__":

    from ClearMap.Visualization.visualize import *

    # ####################################################################################################################
    # #%% Plot graph and corresponding image superimposed and image aside
    # ####################################################################################################################
    #
    source ='/media/sophie.skriabine/Elements/190408_44R/data_stitched.npy'
    source='/data_2to/201120/1L/201120-1L_stitched_vasc.tif'
    import tifffile
    # image=tifffile.imread(source)
    image=io.read(source)




    g=ggt.load('/data_2to/201120/1L/data_graph_correcteduniverse.gt')
    degrees_m=g.vertex_degrees()
    deg4=np.asarray(degrees_m==1).nonzero()[0]

    conn = g.edge_connectivity()

    d=deg4[56]
    center=g.vertex_coordinates()[d]
    s=30
    mins=center-np.array([s, s, s])
    maxs=center+np.array([s, s, s])

    import ClearMap.IO.IO as io

    patch = extractSubGraph(g, mins=mins, maxs=maxs)
    impatch=image[int(center[0])-s:int(center[0])+s, int(center[1])-s:int(center[1])+s, int(center[2])-s:int(center[2])+s]
    # impatch = image.patch(center[0], center[1], center[2], size=(40, 40, 40))
    from ClearMap.Visualization.visualUtils import *
    # vb1, vb2 = get_two_views()
    vb1 = get_view()
    # center=graph.vertex_coordinates()[d]
    # center = im_view_center((2800., 1000., 1600.))
    # center=(center[0], center[1], center[2])
    plotVessels(patch, radiusScaling=600, view=vb1,  color_map=get_radius_color_map(transparency=0))#center=center,
    # plot3d(impatch, colormap=FireMap(), view=vb2)
    plot3d(impatch, colormap=FireMap(), view=vb1)
    show()

    s=50
    center=g.vertex_coordinates()[d]
    mins=center-np.array([s, s, s])
    maxs=center+np.array([s, s, s])
    patch = extractSubGraph(g, mins=mins, maxs=maxs)
    p3d.plot_graph_mesh(patch)

    ns = g.vertex_neighbours(d)
    vf = np.zeros(g.n_vertices)

    vf[ns] = 1
    vf[d] = 1
    ef = get_edges_from_vertex_filter(g, vf)
    conn_f = conn[ef]

    patch = g.sub_graph(vertex_filter=vf)
    p3d.plot_graph_mesh(patch)
    p3d.plot(impatch)

    p3d.plot_graph_mesh(patch)
    patch.edge_radii()
    test=removeSpuriousBranches(test,  rmin=2.9, length=7)
    test=removeMutualLoop(patch,  rmin=2.9, length=14)

    p3d.plot_graph_mesh(test)
    test.edge_radii()
    test.edge_geometry_lengths()

    vein_clusters = np.load('/data_SSD_2to/AlbaNA_06/sbm/diffusion_penetrating_vessel_vein_end_point_cluster_per_region_iteration_Isocortex_graph_corrected.npy')
    art_clusters = np.load('/data_SSD_2to/AlbaNA_06/sbm/diffusion_penetrating_vessel_art_end_point_cluster_per_region_iteration_Isocortex_graph_corrected.npy')
    indices = get_art_vein_overlap(giso, vein_clusters, art_clusters)
    np.save('/data_SSD_2to/AlbaNA_06/sbm/diffusion_penetrating_vessel_overlap_end_point_cluster_per_region_iteration_Isocortex_graph_corrected.npy',indices)

    overlap=np.zeros(giso.n_vertices)
    cluster=np.zeros((giso.n_vertices, 2))
    cluster[:, 0]=art_clusters
    cluster[:, 1]=vein_clusters

    arteries=from_e_prop2_vprop(giso,'artery')
    veins=from_e_prop2_vprop(giso,'vein')

    u, c=np.unique(cluster, axis=0, return_counts=True)

    import random
    # b_c=np.asarray(np.logical_and(c>150, c<=300)).nonzero()[0]
    b_c=np.asarray(np.logical_and(c>600, c<=6000)).nonzero()[0]
    r=random.choice(b_c)
    print(r, c[r])
    vf=np.zeros(giso.n_vertices)

    vf[np.logical_and(np.asarray(cluster[:, 0]==u[r][0]), np.asarray(cluster[:, 1]==u[r][1]))]=3
    vf[np.asarray(np.logical_and(cluster[:, 0]==u[r][0], arteries))]=1
    vf[np.asarray(np.logical_and(cluster[:, 1]==u[r][1], veins))]=2

    vertex_filter=vf>0
    giso.add_vertex_property('vf', vf)
    g_clus=giso.sub_graph(vertex_filter=vertex_filter)

    vf=g_clus.vertex_property('vf')
    colorVal=np.zeros((vf.shape[0], 4))
    red_blue_map = {1: [1.0, 0, 0, 1.0], 3: [0.0, 1.0, 0.0, 1.0], 2: [0.0, 0, 1.0, 1.0]}
    for i, c in enumerate(vf):
        # print(j)
        if c==1:
            colorVal[i] = red_blue_map[1]
        elif c==2:
            colorVal[i] = red_blue_map[2]
        elif c == 3:
            colorVal[i] = red_blue_map[3]

    p = p3d.plot_graph_mesh(g_clus,vertex_colors=colorVal)

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN
    # nb_edge_per_art_terr, density_vect, nb_vvertex_per_art_terr, orientation_vect
    featVect=np.zeros((g_test.n_vertices, 4))
    featVect[:, 0]=orientation_vect
    featVect[:, 1]=inter_conn
    featVect[:, 2]=clust_size
    featVect[:, 3]=g_test.vertex_degrees()

    X = StandardScaler().fit_transform(featVect)
    X_tofit=X
    colors_dict = {0: 'deepskyblue', 1: 'forestgreen', 2: 'firebrick', 3: 'royalblue'}

    db = DBSCAN(eps=0.32, min_samples=10).fit(X_tofit)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    u, c =np.unique(labels, return_counts=True )
    print(c)
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)


    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    X_proj_transformed = pca.fit_transform(X_tofit)#X[:,[0,1,3,4,15,16]])

    fig = plt.figure()
    ax = fig.add_subplot(111)  # , projection='3d')

    for i, n in enumerate(np.unique(labels)):
        indtoplot = np.where(labels == n)[0]
        ax.scatter(X_proj_transformed[indtoplot, 0], X_proj_transformed[indtoplot, 1],
                   color=colors_dict[i], alpha=0.5)  # X_proj_transformed[indtoplot,2]
    # ax.view_init(30, 185)
    plt.show()

    colorVal=np.zeros((g_test.n_vertices, 4))
    red_blue_map = {1: [1.0, 0, 0, 1.0], 3: [0.0, 1.0, 0.0, 1.0], 2: [0.0, 0, 1.0, 1.0]}
    for i, c in enumerate(vf):
        # print(j)
        if c==-1:
            colorVal[i] = red_blue_map[1]
        elif c==0:
            colorVal[i] = red_blue_map[2]
        elif c == 1:
            colorVal[i] = red_blue_map[3]

    p3d.plot_graph_line(g_test, color=colorVal)




    # if __name__ == "__main__":
    # execute only if run as a script
    brain_list = ['/data_SSD_2to/whiskers_graphs/30R']#['AlbaNA_06']
    brainnb=brain_list[0]
    region_list = []
    region_list = [(1006, 3), (580, 5), (650, 5), (724, 5), (811, 4), (875, 4), (6, 6), (463, 6), (388, 6)]
    reg_colors = {1006: 'gold', 580: 'skyblue', 650: 'indianred', 724: 'violet', 811: 'darkorchid',
                  875: 'mediumslateblue', 6: 'forestgreen', 463: 'lightgreen', 388: 'turquoise'}

    # graph=initialize_brain_graph(brainnb)
    # graph44=ggt.load('/data_SSD_2to/190408_44R/data_graph.gt')
    graph=ggt.load('/data_SSD_2to/whiskers_graphs/data_graph_30R.gt')
    region_list=[(6, 6)]
    region_list = [(1, 1)]

    # graph=ggt.load('/data_SSD_2to/AlbaNA_06/data_graph_reduced_transformed_t.gt')

    # graph = ggt.load('/data_SSD_2to/191122Otof/0_9NA/data_graph_annotated.gt')

    gisocortex=ggt.load('/data_SSD_2to/whiskers_graphs/data_graph_corrected_30R.gt')
    # g=ggt.load('/data_SSD_2to/190408_44R/data_graph_corrected.gt')
    g_test = createHighLevelGraph(gisocortex)
    print(g_test)
    g_test.save("/data_SSD_2to/"+brainnb+"/HL_graph.gt")

    p3d.plot_graph_line(g_test)




    colorVal=np.zeros((gss.n_edges, 4))
    artery=gss.edge_property('artery')
    vein=gss.edge_property('vein')
    arteries=np.asarray(artery>0).nonzero()[0]
    veins=np.asarray(vein>0).nonzero()[0]
    red_blue_map = {1: [1.0, 0, 0, 1.0], 0: [0.0, 1.0, 0.0, 1.0], 2: [0.0, 0, 1.0, 1.0]}

    for i in range(gss.n_edges):
        # print(j)
        if i in arteries:
            colorVal[i] = red_blue_map[1]
        elif i in veins:
            colorVal[i] = red_blue_map[0]
        else:
            colorVal[i] = red_blue_map[2]

    p = p3d.plot_graph_mesh(gss, edge_colors=colorVal)
    p=p3d.plot_graph_line(gss)

    # giso=ggt.load('/data_SSD_2to/whiskers_graphs/data_graph_44R.gt')
    # brainnb='whiskers_graphs/39L'

    # g, barrels_filter = extract_AnnotatedRegion(giso, (54,9))
    # art_filter = from_e_prop2_vprop(g, 'artery')
    # g_art_barrel = g.sub_graph(vertex_filter=art_filter)
    # gl, barrels_l_4_filter = extract_AnnotatedRegion(g_art_barrel, (57, 10))
    #
    # artery_color = np.array([[1, 0, 0, 1], [0, 1, 0, 1]])
    # colorVal=artery_color[np.asarray(barrels_l_4_filter, dtype=int)]
    # p3d.plot_graph_mesh(g_art_barrel, vertex_colors=colorVal)

    work_dir = '/data_SSD_2to/191122Otof'#'/data_SSD_2to/whiskers_graphs/new_graphs'# 
    controls = ['2R', '3R', '5R', '8R']
    mutants = ['1R', '7R', '6R', '4R']
    region_list = [(0, 0)]
    # mutants=['138L', '141L', '142L', '158L', '163L', '162L', '164L','165L']
    mutants=['2R', '3R', '5R', '8R','1R', '7R', '6R', '4R']
    work_dir = '/data_SSD_2to/whiskers_graphs/fluoxetine'
    mutants = ['36','30' ,'21', '23','1', '2', '3', '4', '6', '18']#21

    work_dir = '/data_SSD_1to/otof1month'
    # mutants=['5R', '6R', '7R', '8R','1R', '2R', '3R']
    mutants = ['7', '9', '11', '14', '17', '18']
    mutants=['1', '2', '4', '7', '8', '9']
    work_dir= '/data_2to/DBA2J'

    work_dir = '/data_SSD_2to/whiskers_graphs/fluoxetine2'
    mutants = ['1']

    work_dir = '/data_2to/fluoxetine'
    mutants = ['8']#['1', '2', '3', '4', '5', '7', '8']

    work_dir = '/data_SSD_2to/fluoxetine2'
    mutants=['1', '2', '3', '4','5', '7', '8', '9', '10', '11']

    work_dir = '/data_SSD_2to/earlyDep'
    mutants=['16']#'3', '4', '6', '7','9', '10', '11', '12', '15',
    region_list = [(0, 0)]

    work_dir='/data_2to/p0'
    controls=['2', '3']


    work_dir='/data_2to/201120'
    controls=['1', '5', '8']
    mutants=[ '5', '8', '2', '3','4', '6', '7']
    states=[controls, mutants]
    region_list = [(0, 0)]

    work_dir='/data_2to/otof3M/new_vasc'
    mutants=['1k', '1w', '2w', '3k', '4k', '4w','5k', '5w', '6k', '6w']
    states=[controls, mutants]
    region_list = [(0, 0)]

    work_dir='/data_2to/otof1M'
    mutants=['1k', '1w', '2k', '3k','3w', '4k', '5w', '6w', '7w']
    mutants=['7w']
    region_list = [(0, 0)]


    work_dir='/data_2to/whiskers5M/R'
    mutants=['456', '458','467', '468', '469']# 457, 433, 468, 469
    mutants=[ '433', '468', '469', '457']
    mutants=['457']
    region_list = [(0, 0)]

    work_dir='/data_SSD_2to/211019_otof_10m'
    mutants=['1k', '2k','3k', '6k', '7w', '9w', '10w', '12w', '13w']# 457, 433, 468, 469
    mutants=[ '433', '468', '469', '457']
    mutants=['457']
    region_list = [(0, 0)]


    work_dir='/data_SSD_2to/elisahfd'
    mutants=['1h', '1o', '2c', '2h', '2o', '3h', '3o', '4c', '5c', '5h', '5o', '7o']
    region_list = [(0, 0)]

def remove_surface(graph, width):
    distance_from_suface = graph.vertex_property('distance_to_surface')
    ef=distance_from_suface>width
    g=graph.sub_graph(vertex_filter=ef)
    return g


for c in mutants:
        # graph=ggt.load(work_dir+'/'+c+'/'+str(c)+'_graph.gt')
        try:
            graph = ggt.load(work_dir + '/' + c + '/' + 'data_graph.gt')
        except:
            graph = ggt.load(work_dir + '/' + c + '/' + str(c)+'_graph.gt')
        giso=graphCorrection(graph, work_dir+'/'+c, region_list[0])

        giso = ggt.load(work_dir+'/'+c+'/data_graph_corrected_Isocortex.gt')
        diffusion_through_penetrating_arteries(giso, get_penetration_veins_dustance_surface,
                                               get_penetrating_veins_labels, vesseltype='vein', graph_dir=work_dir+'/'+c, feature='distance')
        diffusion_through_penetrating_arteries(giso, get_penetration_arteries_dustance_surface,
                                               get_penetrating_arteries_labels, vesseltype='art',graph_dir=work_dir+'/'+c)
        vein_clusters = np.load(work_dir+'/'+c+'/sbm/diffusion_penetrating_vessel_vein_end_point_cluster_per_region_iteration_brain_distance_graph_corrected.npy')
        art_clusters = np.load(work_dir+'/'+c+'/sbm/diffusion_penetrating_vessel_art_end_point_cluster_per_region_iteration_brain_graph_corrected.npy')

        print('art ', np.unique(art_clusters).shape)
        print('vein ', np.unique(vein_clusters).shape)
        indices = get_art_vein_overlap(giso, vein_clusters, art_clusters)
        np.save(
            work_dir+'/'+c+'/sbm/diffusion_penetrating_vessel_overlap_end_point_cluster_per_region_iteration_Isocortex_graph_corrected.npy',
            indices)





    giso=ggt.load('/data_SSD_2to/whiskers_graphs/data_graph_corrected_30R.gt')

    diffusion_through_penetrating_arteries(giso, get_penetration_veins_dustance_surface, get_penetrating_veins_labels,vesseltype='vein')
    diffusion_through_penetrating_arteries(giso, get_penetration_arteries_dustance_surface, get_penetrating_arteries_labels, vesseltype='art')

    vein_clusters = np.load('/data_SSD_2to/whiskers_graphs/30R/sbm/diffusion_penetrating_vessel_vein_end_point_cluster_per_region_iteration_Isocortex_graph_corrected.npy')
    art_clusters = np.load('/data_SSD_2to/whiskers_graphs/30R/sbm/diffusion_penetrating_vessel_art_end_point_cluster_per_region_iteration_Isocortex_graph_corrected.npy')

    print('art ',np.unique(art_clusters).shape )
    print('vein ', np.unique(vein_clusters).shape)

    for regions in region_list:
        order, level = regions
    print(level, order, ano.find_name(order, key='order'))

    label = giso.vertex_annotation();
    label_leveled = ano.convert_label(label, key='order', value='order', level=level)
    vertex_filter = label_leveled == order;

    # with mutex:
    gss4 = giso.sub_graph(vertex_filter=vertex_filter)


    indices = get_art_vein_overlap(gss4, vein_clusters, art_clusters)
    np.save('/data_SSD_2to/'+brainnb+'/sbm/diffusion_penetrating_vessel_overlap_end_point_cluster_per_region_iteration_Isocortex_graph_corrected.npy',indices)

    s = np.unique(indices).shape[0]
    print(s)
    # modules=sbm_test(graph, region, brainnb)
    giso.add_vertex_property('indices', indices)
    Qart, Qsart = modularity_measure(indices, giso, 'indices')

    plot_diff_from_penetrating_arteries(giso, 'vein')

    gs = gss.sub_slice((slice(1, 320), slice(1, 528), slice(170, 180)));
    # %% plotting
    artery_label = gs.edge_property('artery_binary');
    artery_color = np.array([[1, 0, 0, 1], [0, 0, 1, 1]])
    edge_colors = artery_color[np.asarray(artery_label, dtype=int)];
    # p3d.plot_graph_line(graph_reduced, edge_color = artery_color)
    p3d.plot_graph_mesh(gs, edge_colors=edge_colors)

    import random
    u, c=np.unique(indices, return_counts=True)
    r=u[random.choice(np.asarray(c>=500).nonzero()[0])]
    print(r)
    vf=np.zeros(giso.n_vertices)
    vf[np.asarray(indices==r).nonzero()[0]]=1
    g2plot=giso.sub_graph(vertex_filter=vf)
    p3d.plot_graph_mesh(g2plot)

    # callable #get_penetration_arteries_dustance_surface OR get_penetration_arteries

    diffusion_through_penetrating_arteries(graph, get_penetration_veins_dustance_surface, get_penetrating_veins_labels, vesseltype='vein')
    diffusion_through_penetrating_arteries(graph, get_penetration_arteries_dustance_surface, get_penetrating_arteries_labels,vesseltype='art')

    plot_diff_from_penetrating_arteries(graph, 'overlap')

    nb_edge_per_art_terr, density_vect, nb_vvertex_per_art_terr, orientation_vect=plot_vertex_hist(gisocortex, brainnb)

    epat, cepat = np.unique(nb_vvertex_per_art_terr, return_counts=True)
    plt.figure()
    val=nb_vvertex_per_art_terr[np.asarray(np.logical_and(nb_vvertex_per_art_terr<=2000, nb_vvertex_per_art_terr>5)).nonzero()[0]]
    entries, bin_edges, patches =  plt.hist(val, bins=500, color='#1F9D5A')#plt.bar(epat,cepat)#
    plt.xlim([0, 1400])

    b = np.array(density_vect)
    n=density_vect.shape[0]
    red_blue_map = {1: [1.0, 0, 0, 1.0], 0: [0.0, 1.0, 0.0, 1.0], 2: [0.0, 0, 1.0, 1.0]}
    colors = np.zeros((n, 4));
    j=0
    for i in b:
        print(j)
        if i<0.81:
            colors[b==i, :] = red_blue_map[1]
        elif i>0.85:
            colors[b==i, :] = red_blue_map[0]
        else:
            colors[b == i, :] = red_blue_map[2]
        j=j+1

    from matplotlib import cm
    # colors = np.zeros((gss4_sub.n_vertices, 4));
    # print(colors.shape)
    # for i in range(b.size):
    i = 0

    # for index, b in enumerate(np.unique(diff)):
    inter_conn=g_test.vertex_property('inter_connectivity')
    clust_size = g_test.vertex_property('cluster_size')
    clust_size[np.asarray(clust_size==0).nonzero()[0]]=1000

    ori=orientation_vect/(1-orientation_vect)

    vector2map=inter_conn/clust_size#clust_size#g_test.vertex_degrees()#orientation_vect#clust_size#g_test.vertex_degrees()#inter_conn/clust_size#g_test.vertex_degrees()#clust_size#g_test.vertex_degrees()#

    cmax = np.max(vector2map)
    # vector2map[np.asarray(vector2map ==cmax).nonzero()[0]] = np.mean(vector2map)



    plt.figure()
    plt.hist(vector2map, bins=500)

    cmax = np.max(vector2map)
    cmin = np.min(vector2map)
    x = (vector2map.astype(float) - cmin) / cmax
    # for i in range(gss4.n_vertices):
    #     colors[i, :]=[x[i],0, 1-x[i], 1]
    jet = cm.get_cmap('jet')
    import matplotlib.colors as col

    cNorm = col.Normalize(vmin=cmin, vmax=cmax)
    scalarMap = cm.ScalarMappable(norm=col.LogNorm(), cmap=jet)#norm=col.LogNorm()
    print(scalarMap.get_clim())
    colorVal = scalarMap.to_rgba(vector2map)

    p3d.plot_graph_line(g_test, color=colorVal)
    # connectivity = gss4_sub.edge_connectivity();
    # edge_colors = (colorVal[connectivity[:, 0]] + colorVal[connectivity[:, 1]]) / 2.0;
    # edge_colors[art > 0] = [1., 0.0, 0.0, 1.0]

    # colors = np.insert(colors, 3, 1.0, axis=1)
    # vertex_filter = label_leveled == 146;
    # gss4 = g.sub_graph(vertex_filter=vertex_filter)
    # colors4=colors[vertex_filter]
    # gss4_sub.add_vertex_property('colors_clusters', colors)
    gss4.add_vertex_property('colorVal', colorVal)

    gs = gss.sub_slice((slice(0, 4000), slice(0, 7000), slice(2000, 2300)));
    edge_vein_label = gs.edge_property('vein');
    edge_artery_label = gs.edge_property('artery');

    vertex_colors = ano.convert_label(gs.vertex_annotation(), key='order', value='rgba');

    connectivity = gs.edge_connectivity();
    edge_colors = (vertex_colors[connectivity[:, 0]] + vertex_colors[connectivity[:, 1]]) / 2.0;
    edge_colors[edge_artery_label > 0] = [0.8, 0.0, 0.0, 1.0]
    edge_colors[edge_vein_label > 0] = [0.0, 0.0, 0.8, 1.0]

    p = p3d.plot_graph_mesh(gs, edge_colors=edge_colors, n_tube_points=3);




    diffusion_through_penetrating_arteries(gisocortex, get_penetration_veins_dustance_surface, get_penetrating_veins_labels,vesseltype='vein')
    diffusion_through_penetrating_arteries(gisocortex, get_penetration_arteries_dustance_surface, get_penetrating_arteries_labels,vesseltype='art')
    vein_clusters=np.load('/data_SSD_2to/190408_44R/sbm/diffusion_penetrating_vein_end_point_cluster_per_region_iteration_Isocortex.npy')
    art_clusters=np.load('/data_SSD_2to/190408_44R/sbm/diffusion_penetrating_art_end_point_cluster_per_region_iteration_Isocortex.npy')
    indices=get_art_vein_overlap(graph, vein_clusters, art_clusters)
    np.save('/mnt/data_SSD_2to/190408_44R/sbm/diffusion_penetrating_vessel_overlap_end_point_cluster_per_region_iteration_Isocortex.npy',indices)

    s = np.unique(indices).shape[0]

    graph.add_vertex_property('indices', indices)
    Qart, Qsart = modularity_measure(indices, graph, 'indices')

    plot_diff_from_penetrating_arteries(graph, 'vessel_overlap')


mutants=['1c', '1h', '1o','2c', '2h', '2o', '3o', '4c', '5c', '5h', '7o', '8c']
work_dir='/media/sophie.skriabine/sophie/HFD_VASC'
percent_non_corr=[]
percent_corr=[]
nb_vertex=[]
nb_vertex_corrected=[]
degsup5=[]
degsup5_corr=[]
for c in mutants:
    graph = ggt.load(work_dir + '/' + str(c) + '/' + str(c)+'_graph.gt')
    giso=ggt.load(work_dir + '/' + str(c) + '/data_graph_correcteduniverse.gt')
    nb_vertex.append(graph.n_vertices)
    nb_vertex_corrected.append(giso.n_vertices)
    deg4=np.sum(graph.vertex_degrees()>=5)
    deg4_corrected=np.sum(giso.vertex_degrees()>=5)
    print(c, ': ', deg4/graph.n_vertices)
    percent_non_corr.append(deg4/graph.n_vertices)
    degsup5.append(deg4/graph.n_vertices)
    print(c, ': ', deg4_corrected/giso.n_vertices)
    percent_corr.append(deg4_corrected/giso.n_vertices)
    degsup5_corr.append(deg4/giso.n_vertices)


df=pd.DataFrame(columns=['deg5', 'cat'])
for e in range(len(mutants)):
    df = df.append({'deg5': degsup5_corr[e]+percent_non_corr[e], 'cat':  'non_corr'}, ignore_index=True)
    df = df.append({'deg5': degsup5[e]+percent_corr[e], 'cat':  'corr'}, ignore_index=True)
    # df=df.append([percent_non_corr[e], 'non_corr'])
    # df=df.append([percent_corr[e], 'corr'])
plt.figure()
sns.boxplot(data=df, x='cat', y='deg5')
sns.despine()