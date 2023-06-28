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
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import Rbf

from sklearn.linear_model import LinearRegression
work_dir='/data_SSD_2to/191122Otof'#'/data_SSD_2to/whiskers_graphs'
brains=['1R','2R','3R','5R','6R','7R', '8R', '4R']#['44R', '30R']#, '39L']
brains=['2R','3R','5R','6R','7R', '8R', '4R']
control='2R'
controls=['2R','3R','5R', '8R']
# work_dir = '/data_SSD_2to/whiskers_graphs/new_graphs'
# brains = ['142L', '158L', '162L', '164L','138L', '141L', '163L', '165L']



graph = ggt.load(work_dir + '/' + control + '/' + 'data_graph_correcteduniverse.gt')#data_graph_corrected_Isocortex.gt')#/data_graph_correcteduniverse.gt')#
degrees = graph.vertex_degrees()
vf = np.logical_and(degrees > 1, degrees <= 4)
graph = graph.sub_graph(vertex_filter=vf)


with open(work_dir + '/' + control + '/sampledict' + control + '.pkl', 'rb') as fp:
    sampledict = pickle.load(fp)

pressure = np.asarray(sampledict['pressure'][0])
graph.add_vertex_property('pressure', pressure)

pi = math.pi
G=[]
for control in controls:
    graph = ggt.load(
        work_dir + '/' + control + '/' + 'data_graph_correcteduniverse.gt')  # data_graph_corrected_Isocortex.gt')#/data_graph_correcteduniverse.gt')#
    degrees = graph.vertex_degrees()
    vf = np.logical_and(degrees > 1, degrees <= 4)
    graph = graph.sub_graph(vertex_filter=vf)

    with open(work_dir + '/' + control + '/sampledict' + control + '.pkl', 'rb') as fp:
        sampledict = pickle.load(fp)

    pressure = np.asarray(sampledict['pressure'][0])
    graph.add_vertex_property('pressure', pressure)

    artery = from_e_prop2_vprop(graph, 'artery')
    vein = from_e_prop2_vprop(graph, 'vein')
    radii = graph.vertex_property('radii')
    d2s = graph.vertex_property('distance_to_surface')



    # region_list = [(142, 8), (149, 8), (128, 8), (156, 8)]
    # vertex_filter=np.zeros(graph.n_vertices)
    # for i, rl in enumerate(region_list):
    #     order, level = region_list[i]
    #     print(level, order, ano.find(order, key='order')['name'])
    #     label = graph.vertex_annotation();
    #     label_leveled = ano.convert_label(label, key='order', value='order', level=level)
    #     vertex_filter[label_leveled == order] = 1;
    # graph=graph.sub_graph(vertex_filter=vertex_filter)

    label =  graph.vertex_annotation()

    artery_vein = np.asarray(np.logical_or(artery, vein))
    # order, level=1006,3
    # label_leveled = ano.convert_label(label, key='order', value='order', level=level)
    # cerebellum = label_leveled == order;
    # # radii = graph.edge_property('radii')
    # # cerebellum_art=from_e_prop2_vprop(graph, np.logical_and(radii <=3, np.logical_and(from_v_prop2_eprop(graph, cerebellum),from_v_prop2_eprop(graph,artery_vein))))
    # cerebellum_art = np.logical_and(radii <= 6,np.logical_and( cerebellum,artery_vein))
    # # cg=graph.sub_graph(vertex_property=cerebellum_art)
    #
    #
    # order, level = 463, 6
    # label_leveled = ano.convert_label(label, key='order', value='order', level=level)
    # hippocampus = label_leveled == order;
    # # radii = graph.edge_property('radii')
    # # hippocampus_art=from_e_prop2_vprop(graph, np.logical_and(radii <=3, np.logical_and(from_v_prop2_eprop(graph, hippocampus),from_v_prop2_eprop(graph,artery_vein))))
    # hippocampus_art = np.logical_and(radii <= 6, np.logical_and(hippocampus, artery_vein))
    # # hg = graph.sub_graph(vertex_filter=hippocampus_art)
    # # p3d.plot_graph_mesh(hg)
    #
    # artery_vein[hippocampus] = 0
    # artery_vein[cerebellum] = 0
    #
    # radii = graph.vertex_property('radii')
    # pb_art=np.logical_or(hippocampus_art,cerebellum_art)
    # artery_vein=np.logical_or(pb_art,np.logical_and(artery_vein, radii >= 6))
    # # artery_vein = np.asarray(np.logical_and(artery_vein, radii >= 6))  # .nonzero()[0]
    #
    # artery_vein = np.asarray(np.logical_and(artery_vein, d2s >= 4))#7  # .nonzero()[0]
    # # capi = np.logical_not(np.logical_or(artery, vein)).nonzero()[0]
    #
    # # big_radii = graph.edge_property('radii') > 4
    # # artery_vein_edge = np.asarray(np.logical_or(graph.edge_property('artery'), graph.edge_property('vein')))
    # # artery_vein_edge = np.asarray(np.logical_and(artery_vein_edge, big_radii)).nonzero()[0]
    # # capi_edge = np.logical_not(np.logical_or(graph.edge_property('artery'), graph.edge_property('vein'))).nonzero()[0]
    # # try:
    # art_graph = graph.sub_graph(vertex_filter=artery_vein)  # np.logical_or(artery, vein))
    # graph=graph.sub_graph(vertex_filter=np.logical_not(artery_vein))
    # art_graph_vector, art_press_conn = getEdgeVector(art_graph, control)

    graph_vector, press_conn = getEdgeVector(graph,control, normed=True, criteria='distance')

    coordinates = graph.vertex_property('coordinates_atlas')# art_graph.vertex_coordinates()
    edge_coordinates = np.array(
        [np.round((coordinates[press_conn[i, 0]] + coordinates[press_conn[i, 1]]) / 2) for i in
         range(press_conn.shape[0])])




    angle = GeneralizedRadPlanorientation(graph, control)
    angle_range=np.arange(0, 100, 10)
    region_list=[(144,9)]
    region_list = [(142, 8), (149, 8), (128, 8), (156, 8)]
    vertex_filter=np.zeros(graph.n_vertices)
    for i, rl in enumerate(region_list):
        order, level = region_list[i]
        print(level, order, ano.find(order, key='order')['name'])
        label = graph.vertex_annotation();
        label_leveled = ano.convert_label(label, key='order', value='order', level=level)
        vertex_filter[label_leveled == order] = 1;
        edge_filter=from_v_prop2_eprop(graph,vertex_filter)

    angle_t=angle[edge_filter]

    plt.figure()
    hist, bins=np.histogram(angle_t,bins=angle_range)
    bins=[(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
    plt.plot(bins,hist/np.sum(hist))
    # def prop_radial(limit_angle):
    #     A_tot=4*pi
    #     A_rad=2*2*pi*(1-np.cos(limit_angle))
    #     A_plan=(4*pi)-(2*2*pi*(1-np.cos(90 - limit_angle)))
    #     return A_rad/A_tot, A_plan/A_tot
    #
    # angle_range=np.arange(0, 100, 10)
    # angle_range=angle_range*pi/180
    # rad_prop=[]
    # plan_prop=[]
    # for theta in angle_range:
    #     R, P=prop_radial(theta)
    #     rad_prop.append(R)
    #     plan_prop.append(P)
    isotropicAngledist=hist/np.sum(hist)#np.diff(rad_prop)





    # coordinates = graph.vertex_coordinates()
    # edge_coordinates = np.array(
    #     [np.round((coordinates[press_conn[i, 0]] + coordinates[press_conn[i, 1]]) / 2) for i in
    #      range(press_conn.shape[0])])

    from scipy.interpolate import LinearNDInterpolator,NearestNDInterpolator
    # from scipy.interpolate import Rbf
    # selection=np.random.choice(np.arange(edge_coordinates.shape[0]), int(edge_coordinates.shape[0]/10),replace=False)
    # NNDI_x = LinearNDInterpolator(edge_coordinates[selection], graph_vector[selection,0])
    # NNDI_y = LinearNDInterpolator(edge_coordinates[selection], graph_vector[selection, 1])
    # NNDI_z = LinearNDInterpolator(edge_coordinates[selection], graph_vector[selection, 2])

    # local_flow_X = NNDI_x(edge_coordinates)
    # local_flow_Y = NNDI_y(edge_coordinates)
    # local_flow_Z = NNDI_z(edge_coordinates)
    size=31#21
    grid_x = np.linspace(0, np.max(edge_coordinates[:, 0]), size)#21
    grid_y = np.linspace(0, np.max(edge_coordinates[:, 1]), size)
    grid_z = np.linspace(0, np.max(edge_coordinates[:, 2]), size)

    # bins_inds = np.digitize(edge_coordinates, bins=(grid_x, grid_y, grid_z))
    # H, edges  = np.histogramdd(graph_vector, bins=(grid_x, grid_y, grid_z))

    # def custom_function(array):
    #     hist,bins=np.histogram(array, bins=10)
    #     bins_center=[(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
    #     elem = bins_center[np.argmax(np.array(hist))]
    #     return elem

    # def custom_function(array):
    #     # array=array.T
    #     print(array)
    #     a=array
    #     # print(a,a.shape)
    #     # coord=array[1:]
    #     # print(coord, coord.shape)
    #     hist,bins=np.histogram(a, bins=np.arange(0, 100, 10))
    #     bins_center=[(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
    #     elem = bins_center[np.argmax(np.array(hist)-isotropicAngledist)]
    #     # m=np.nanmedian(coord[np.asarray(np.logical_and(a>elem-5,a<elem+5))], axis=0)
    #
    #     m=np.asarray(np.logical_and(a>elem-5,a<elem+5)).nonzero()[0]
    #     print(m)
    #     return (m)
    #
    #
    #
    # from scipy import stats
    #
    # array = np.concatenate((angle[:, np.newaxis], graph_vector), axis=1)
    #
    # ret = stats.binned_statistic_dd(edge_coordinates, array,
    #                                  bins=(grid_x, grid_y, grid_z),
    #                                  statistic=custom_function)#'mean'#custom_function

    import random
    from scipy.sparse.linalg import eigs
    def multiproc_function(args):
        edges=[]
        i, j, k=args
        # print(i, j, k)
        mins=[grid_x[i], grid_y[j], grid_z[k]]
        maxs=[grid_x[i+1], grid_y[j+1], grid_z[k+1]]
        # print(mins, maxs)
        isOver = (edge_coordinates >= mins).all(axis=1)
        isUnder = (edge_coordinates < maxs).all(axis=1)

        edgeFilter=np.logical_and(isOver, isUnder)
        edgeFilterIndices=np.asarray(edgeFilter).nonzero()[0]
        coord=edge_coordinates[edgeFilter]
        a=angle[edgeFilter]
        # print(a.shape)
        if a.shape[0]==0:
            # m=[np.NaN,np.NaN,np.NaN]
            # elem=np.NaN
            # return [np.NaN,np.NaN,np.NaN]#, np.NaN]
            res=np.asarray([])
        else:
            # correlation=coord.T.dot(coord)
            # vals, vecs = eigs(correlation, k=1, which='LM')
            # m=vecs.real
            # m=[m[0][0], m[1][0], m[2][0]]
            hist,bins=np.histogram(a, bins=np.arange(0, 100, 10))
            # bins_center=[(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
            elem = bins[np.argmax((hist/np.sum(hist))-isotropicAngledist)]
            print(elem)
            try:
                res=edgeFilterIndices[np.asarray(np.logical_and(a>elem-1,a<elem+1)).nonzero()[0]]
                print(res)
                # index=random.choice(np.asarray(np.logical_and(a>elem-2,a<elem+2)).nonzero()[0])
                # m=coord[index].tolist()
            except:
                # m=[np.NaN,np.NaN,np.NaN]
                # elem=np.NaN
                res=np.asarray([])

            # m=np.asarray(np.logical_and(a>elem-5,a<elem+5)).nonzero()[0]

        # res=m#.append(elem)
        return res

    import multiprocessing
    import itertools
    args=itertools.product(range(grid_x.shape[0]-1), range(grid_y.shape[0]-1), range(grid_z.shape[0]-1))
    with multiprocessing.Pool(processes=15) as pool:
        res=[pool.map(multiproc_function, args)]

    resarray=np.array(res)[0]
    np.save(  work_dir + '/' + control + '/' + 'resarray'+str(size)+'.npy', resarray)

    resarray=np.load( work_dir + '/' + control + '/' + 'resarray'+str(size)+'.npy', allow_pickle=True)

    filter=np.zeros(graph.n_edges)

    for ar in resarray:
        if ar.shape[0]>0:
            for a in ar:
                filter[a]=1

    filter=from_e_prop2_vprop(graph, filter)
    art_graph = graph.sub_graph(vertex_filter=filter)



    art_graph_vector, art_press_conn = getEdgeVector(art_graph, control, criteria='distance')

    graph_vector, press_conn = getEdgeVector(graph,control, criteria='distance')

    art_coordinates = art_graph.vertex_property('coordinates_atlas')# art_graph.vertex_coordinates()
    art_edge_coordinates = np.array(
        [np.round((art_coordinates[art_press_conn[i, 0]] + art_coordinates[art_press_conn[i, 1]]) / 2) for i in
         range(art_press_conn.shape[0])])

    # coordinates = graph.vertex_coordinates()
    # edge_coordinates = np.array(
    #     [np.round((coordinates[press_conn[i, 0]] + coordinates[press_conn[i, 1]]) / 2) for i in
    #      range(press_conn.shape[0])])

    from scipy.interpolate import LinearNDInterpolator,NearestNDInterpolator
    # from scipy.interpolate import Rbf
    NNDI_x = LinearNDInterpolator(art_edge_coordinates, art_graph_vector[:,0])
    NNDI_y = LinearNDInterpolator(art_edge_coordinates, art_graph_vector[:, 1])
    NNDI_z = LinearNDInterpolator(art_edge_coordinates, art_graph_vector[:, 2])

    # local_flow_X = NNDI_x(edge_coordinates)
    # local_flow_Y = NNDI_y(edge_coordinates)
    # local_flow_Z = NNDI_z(edge_coordinates)


    resolution=20
    grid_x = np.linspace(0, np.max(art_edge_coordinates[:, 0]), resolution)#100
    grid_y = np.linspace(0, np.max(art_edge_coordinates[:, 1]), resolution)
    grid_z = np.linspace(0, np.max(art_edge_coordinates[:, 2]), resolution)


    grid=np.array(np.meshgrid(grid_x, grid_y, grid_z)).reshape((3, np.power(resolution, 3))).T

    grid_flow_X = NNDI_x(grid)
    grid_flow_Y = NNDI_y(grid)
    grid_flow_Z = NNDI_z(grid)

    grid_vector=np.stack([grid_flow_X, grid_flow_Y,grid_flow_Z]).T












#     grid_x = np.linspace(0, np.max(edge_coordinates[:, 0]), 100)#100
#     grid_y = np.linspace(0, np.max(edge_coordinates[:, 1]), 100)
#     grid_z = np.linspace(0, np.max(edge_coordinates[:, 2]), 100)
#     grid=np.array(np.meshgrid(grid_x, grid_y, grid_z)).reshape((3, 125000)).T
#     #
#     # grid_flow_X = NNDI_x(grid)
#     # grid_flow_Y = NNDI_y(grid)
#     # grid_flow_Z = NNDI_z(grid)
#     # grid_flow_X=retX.statistic.reshape(1000000)
#     # grid_flow_Y=retY.statistic.reshape(1000000)
#     # grid_flow_Z=retZ.statistic.reshape(1000000)
#
#
#     grid_flow_X=resarray[:, 0]
#     grid_flow_Y=resarray[:, 1]
#     grid_flow_Z=resarray[:, 2]
#
#     grid_vector=np.stack([grid_flow_X, grid_flow_Y,grid_flow_Z]).T
# #     G=[]
#     G.append(grid_vector)
#
#
#     # except:
#     #     print('problem interpolation ...')
#
# Gmean=np.nanmean(np.array(G), axis=0)
# grid_vector = Gmean

sl=228
grid_coordinates_2plot = grid[grid[:, 1] > sl]
grid_vector_2plot = grid_vector[grid[:, 1] > sl]
# art_graph_vector_spher_2plot = spher_ori[art_edge_coordinates[:, 1] > slice]
grid_vector_2plot = grid_vector_2plot[grid_coordinates_2plot[:, 1] < sl + 30]  # 100
# art_graph_vector_spher_2plot= art_graph_vector_spher_2plot[art_edge_coordinates_2plot[:, 1] < slice+100]
grid_coordinates_2plot = grid_coordinates_2plot[grid_coordinates_2plot[:, 1] < sl + 30]  # 100



center = (np.median(grid_coordinates_2plot[:, 0]), np.max(grid_coordinates_2plot[:, 2]) - 30)  # -500#(3500, 2000)
print(center)
grid_coordinates_2plot = np.array(
    [grid_coordinates_2plot[i, [0, 2]] - center for i in range(grid_coordinates_2plot.shape[0])])
# grid_vector_spher_2plot = [np.dot(preprocessing.normalize(np.nan_to_num(grid_coordinates_2plot), norm='l2')[i],
#                                   preprocessing.normalize(np.nan_to_num(grid_vector_2plot[:, [0, 2]]))[i]) for i
#                            in range(grid_coordinates_2plot.shape[0])]
X = grid_coordinates_2plot[:, 1]  # -center[0]
Y = grid_coordinates_2plot[:, 0]  # -center[1]
grid_vector_2plot = preprocessing.normalize(np.nan_to_num(grid_vector_2plot[:, [0, 2]]))
U = grid_vector_2plot[:, 1]
V = grid_vector_2plot[:, 0]

# M = abs(np.array(grid_vector_spher_2plot))


plt.figure()
# plt.quiver(X[M>3], Y[M>3], U[M>3], V[M>3], M[M>3], pivot='mid')
U = -U
V = -V
plt.quiver(X, Y, U, V, pivot='mid')  # [M_normed[:,0]>1]
plt.title(str(sl))
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()

from scipy.interpolate import griddata

xi = np.linspace(X.min(), X.max(), resolution)
yi = np.linspace(Y.min(), Y.max(), resolution)

# an (nx * ny, 2) array of x,y coordinates to interpolate at
# ipts = np.vstack(a.ravel() for a in np.meshgrid(yi, xi)[::-1]).T
pts = np.vstack((X, Y)).T
vals = np.vstack((U, V)).T
ipts = np.vstack(a.ravel() for a in np.meshgrid(xi, yi)).T
ivals = griddata(pts, vals, ipts, method='cubic')

ui, vi = preprocessing.normalize(ivals).T
ui.shape = vi.shape = (resolution, resolution)


import tifffile
annotation=tifffile.imread( '/home/sophie.skriabine/Pictures/Reslice_of_annotation_25_HeadLightOrientation_coronal.tif')
annotation=np.swapaxes(annotation, 0, 2)
annotation=np.flip(annotation, 2)
annotation=np.flip(annotation, 0)
plt.figure()
with plt.style.context('seaborn-white'):
    # plt.rcParams['axes.facecolor'] = 'black'
    from skimage import feature
    edges2 = feature.canny(annotation[:228, :, sl].T, sigma=0.1).astype(int)
    # mask=annotation[:228, :, slice].T>0
    mask = np.ma.masked_where(annotation[:228, :, sl].T > 0, annotation[:228, :, sl].T)
    xi = xi - np.min(xi)
    yi = yi - np.min(yi)
    plt.streamplot(xi, yi, -ui, -vi, density=15, arrowstyle='-', color='k', zorder=2)  # color=ui+vi#color=colors_rgb,
    # plt.quiver(X +180, Y +140, U, V,pivot='mid',zorder=2)  # [M_normed[:,0]>1]

    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.imshow(mask, cmap='Greys', zorder=10)
    plt.imshow(edges2, cmap='Greys', zorder=1)  # jet#copper#
    edges2 = np.logical_not(edges2)
    # edges2[edges2==0]=np.nan
    masked_data = np.ma.masked_where(edges2 > 0, edges2)
    plt.imshow(masked_data, cmap='cool_r', zorder=11)  # jet#copper#
    # plt.axis('off')
    plt.title(str(sl))




work_dir='/data_SSD_2to/capillariesStream/'
edges2 = feature.canny(annotation[:228, :, 200].T, sigma=0.1)
plt.figure()
plt.imshow(edges2.astype(int), cmap='gray')

from skimage import io

sl=228
im = annotation[:228, :, sl].T> 10
im=np.logical_not(im)
io.imsave(work_dir+'autofluo'+str(sl)+'.png', im.astype(int))


work_dir='/data_SSD_2to/191122Otof'
control='2R'
limit_angle=40

angles=[]
for control in controls:
    graph = ggt.load(
        work_dir + '/' + control + '/' + 'data_graph_correcteduniverse.gt')

    degrees = graph.vertex_degrees()
    vf = np.logical_and(degrees > 1, degrees <= 4)
    graph = graph.sub_graph(vertex_filter=vf)


    with open(work_dir + '/' + control + '/sampledict' + control + '.pkl', 'rb') as fp:
        sampledict = pickle.load(fp)

    f = np.asarray(sampledict['flow'][0])
    v = np.asarray(sampledict['v'][0])
    graph.add_edge_property('flow', f)
    graph.add_edge_property('veloc', v)
    pressure = np.asarray(sampledict['pressure'][0])
    graph.add_vertex_property('pressure', pressure)

    region_list = [(142, 8), (149, 8), (128, 8), (156, 8)]
    # region_list = [(144, 9), (149, 8), (128, 8), (156, 8)]
    vertex_filter = np.zeros(graph.n_vertices)



    region_list=[(0,0)]
    for i, rl in enumerate(region_list):
        order, level = region_list[i]
        print(level, order, ano.find(order, key='order')['name'])
        label = graph.vertex_annotation();
        label_leveled = ano.convert_label(label, key='order', value='order', level=level)
        vertex_filter[label_leveled == order] = 1;
        edge_filter=from_v_prop2_eprop(graph, vertex_filter)

    gss4 = graph.sub_graph(vertex_filter=vertex_filter)

    angle = GeneralizedRadPlanorientation(gss4, control)
    gss4.add_edge_property('angle',angle)

    artery=from_e_prop2_vprop(gss4, 'artery')
    vein=from_e_prop2_vprop(gss4, 'vein')
    radii=gss4.edge_property('radii')
    plt.figure()
    plt.hist(radii[np.logical_or(gss4.edge_property('artery'), gss4.edge_property('vein'))], bins=50)
    gss4 = gss4.sub_graph(vertex_filter=np.logical_not(np.logical_or(artery, vein)))
    angle=gss4.edge_property('angle')

    angle_t=angle[~np.isnan(angle)]
    angle_range=np.arange(0, 100, 10)
    hist, bins=np.histogram(angle_t,bins=angle_range)
    bins=[(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
    angles.append(hist)

import pandas as pd
import seaborn as sns
from sklearn.preprocessing import normalize
angles_bis=angles.copy()
for i, angle in enumerate(angles):
    angles_bis[i]=angles[i]/np.sum(angles[i])
plt.figure()
angle_range=np.arange(0, 100, 10)
hist, bins=np.histogram(angle_t,bins=angle_range)
bins=[(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
angles_dist = pd.DataFrame(angles_bis).melt()
angles_dist['variable']=angles_dist['variable']*10+5
sns.lineplot(x="variable", y="value", data=angles_dist, color='darkorange')#,  err_style='bars', linewidth=1.5)
plt.plot(bins, np.diff(rad_prop))







plt.figure()
angle_range=np.arange(0, 100, 10)
hist, bins=np.histogram(angle_t,bins=angle_range)
bins=[(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
plt.plot(bins,hist/np.sum(hist))
plt.plot(bins, np.diff(rad_prop))

dist = gss4.edge_property('distance_to_surface')


radiality=angle_t < limit_angle#40
planarity=angle_t > 90-limit_angle#60


neutral = np.logical_not(np.logical_or(radiality, planarity))

ori_prop = np.concatenate((np.expand_dims(dist, axis=1), np.concatenate((np.expand_dims(radiality, axis=1), np.concatenate(
    (np.expand_dims(neutral, axis=1), np.expand_dims(planarity, axis=1)), axis=1)), axis=1)), axis=1)
prop_ori.append(ori_prop)

limit_angle=40
rad = angle < limit_angle  # 40
plan = angle > 90 - limit_angle  # 60

## get isotropic orientation distribution


def prop_radial(limit_angle):
    A_tot=4*pi
    A_rad=2*2*pi*(1-np.cos(limit_angle))
    A_plan=(4*pi)-(2*2*pi*(1-np.cos(90 - limit_angle)))
    return A_rad/A_tot, A_plan/A_tot

angle_range=np.arange(0, 100, 10)
angle_range=angle_range*pi/180
rad_prop=[]
plan_prop=[]
for theta in angle_range:
    R, P=prop_radial(theta)
    rad_prop.append(R)
    plan_prop.append(P)


plt.figure()
plt.plot(rad_prop)
plt.plot(np.diff(rad_prop))

plt.figure()
plt.plot(plan_prop)
plt.plot(np.diff(plan_prop))

plt.figure()
angle_range=np.arange(0, 100, 10)
hist, bins=np.histogram(angle_t,bins=angle_range)
bins=[(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
plt.plot(bins,hist/np.sum(hist))
plt.plot(bins, np.diff(rad_prop))
plt.legend(['whole brain angle distribution', 'isotropic distribution'])


region_list=[(145,9)]#[(144, 9)]
# region_list = [(142, 8), (149, 8), (128, 8), (156, 8)]
vertex_filter=np.zeros(gss4.n_vertices)
for i, rl in enumerate(region_list):
    order, level = region_list[i]
    print(level, order, ano.find(order, key='order')['name'])
    label = gss4.vertex_annotation();
    label_leveled = ano.convert_label(label, key='order', value='order', level=level)
    vertex_filter[label_leveled == order] = 1;
    edge_filter=from_v_prop2_eprop(gss4,vertex_filter)

angle_t=angle[edge_filter]

# plt.figure()
hist, bins=np.histogram(angle_t,bins=angle_range)
bins=[(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
# plt.plot(bins,hist/np.sum(hist))

plt.figure()
hist, bins=np.histogram(angle_t,bins=angle_range)
bins=[(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
plt.plot(bins,isotropicAngledist)
plt.plot(bins,(hist/np.sum(hist)))
# plt.plot(bins,(hist/np.sum(hist))-isotropicAngledist)

elem = bins[np.argmax((hist/np.sum(hist))-isotropicAngledist)]
print(elem)
a=angle_t
edgeFilterIndices=np.asarray(edge_filter).nonzero()[0]
res=edgeFilterIndices[np.asarray(np.logical_and(a>elem-5,a<elem+5)).nonzero()[0]]

filter=np.zeros(gss4.n_edges)
filter[res]=1
filter=from_e_prop2_vprop(gss4, filter)
gss4_t = gss4.sub_graph(vertex_filter=filter)
p3d.plot_graph_mesh(gss4_t)