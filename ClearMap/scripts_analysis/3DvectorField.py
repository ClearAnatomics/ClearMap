import plotly.graph_objects as go


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


graph = ggt.load(work_dir + '/' + control + '/' + 'data_graph_correcteduniverse.gt')#data_graph_corrected_Isocortex.gt')#/data_graph_correcteduniverse.gt')#
degrees = graph.vertex_degrees()
vf = np.logical_and(degrees > 1, degrees <= 4)
graph = graph.sub_graph(vertex_filter=vf)


with open(work_dir + '/' + control + '/sampledict' + control + '.pkl', 'rb') as fp:
    sampledict = pickle.load(fp)

pressure = np.asarray(sampledict['pressure'][0])
graph.add_vertex_property('pressure', pressure)

from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator, CloughTocher2DInterpolator



def radialVectorField(graph, slice, zmin, zmax, dim, sampling, function):
    pi = math.pi
    artery = from_e_prop2_vprop(graph, 'artery')
    vein = from_e_prop2_vprop(graph, 'vein')
    radii = graph.vertex_property('radii')
    d2s = graph.vertex_property('distance_to_surface')

    artery_vein = np.asarray(np.logical_or(artery, vein))
    artery_vein = np.asarray(np.logical_and(artery_vein, radii >= 6))  # .nonzero()[0]
    # artery_vein = np.asarray(np.logical_or(artery_vein, radii >= 4))  # .nonzero()[0]
    artery_vein = np.asarray(np.logical_and(artery_vein, d2s >= 4))#7  # .nonzero()[0]
    # capi = np.logical_not(np.logical_or(artery, vein)).nonzero()[0]

    # big_radii = graph.edge_property('radii') > 4
    # artery_vein_edge = np.asarray(np.logical_or(graph.edge_property('artery'), graph.edge_property('vein')))
    # artery_vein_edge = np.asarray(np.logical_and(artery_vein_edge, big_radii)).nonzero()[0]
    # capi_edge = np.logical_not(np.logical_or(graph.edge_property('artery'), graph.edge_property('vein'))).nonzero()[0]
    try:
        art_graph = graph.sub_graph(vertex_filter=artery_vein)  # np.logical_or(artery, vein))

        art_graph_vector, art_press_conn = getEdgeVector(art_graph)

        graph_vector, press_conn = getEdgeVector(graph)

        art_coordinates = art_graph.vertex_property('coordinates_atlas')# art_graph.vertex_coordinates()
        art_edge_coordinates = np.array(
            [np.round((art_coordinates[art_press_conn[i, 0]] + art_coordinates[art_press_conn[i, 1]]) / 2) for i in
             range(art_press_conn.shape[0])])

        # coordinates = graph.vertex_coordinates()
        # edge_coordinates = np.array(
        #     [np.round((coordinates[press_conn[i, 0]] + coordinates[press_conn[i, 1]]) / 2) for i in
        #      range(press_conn.shape[0])])

        from scipy.interpolate import NearestNDInterpolator
        # from scipy.interpolate import Rbf
        NNDI_x = LinearNDInterpolator(art_edge_coordinates, art_graph_vector[:,0])
        NNDI_y = LinearNDInterpolator(art_edge_coordinates, art_graph_vector[:, 1])
        NNDI_z = LinearNDInterpolator(art_edge_coordinates, art_graph_vector[:, 2])

        # local_flow_X = NNDI_x(edge_coordinates)
        # local_flow_Y = NNDI_y(edge_coordinates)
        # local_flow_Z = NNDI_z(edge_coordinates)

        grid_x = np.linspace(np.min(art_edge_coordinates[:, 0]), np.max(art_edge_coordinates[:, 0]), 100)#100
        grid_y = np.linspace(np.min(art_edge_coordinates[:, 1]), np.max(art_edge_coordinates[:, 1]), 100)
        grid_z = np.linspace(np.min(art_edge_coordinates[:, 2]), np.max(art_edge_coordinates[:, 2]), 100)


        grid=np.array(np.meshgrid(grid_x, grid_y, grid_z)).reshape((3, 1000000)).T

        grid_flow_X = NNDI_x(grid)
        grid_flow_Y = NNDI_y(grid)
        grid_flow_Z = NNDI_z(grid)

        grid_vector=np.stack([grid_flow_X, grid_flow_Y,grid_flow_Z]).T


    except:
        print('problem interpolation ...')

    # spher_ori=getVesselOrientation(art_graph, graph)
    from sklearn.preprocessing import normalize
    import cmath
    from scipy.interpolate import griddata
    X=grid[:, 0]
    Y=grid[:, 1]
    Z=grid[:, 2]

    U = grid_vector[:, 0]
    V = grid_vector[:, 1]
    W = grid_vector[:, 2]

    xi = X#np.linspace(X.min(), X.max(), 100)
    yi = Y#np.linspace(Y.min(), Y.max(), 100)
    zi = Z#np.linspace(Z.min(), Z.max(), 100)

    # an (nx * ny, 2) array of x,y coordinates to interpolate at
    # ipts = np.vstack(a.ravel() for a in np.meshgrid(yi, xi)[::-1]).T
    pts = np.vstack((X, Y, Z)).T
    vals = np.vstack((U, V, W)).T

    ipts = np.vstack(a.ravel() for a in np.meshgrid(xi, yi, zi)).T
    ivals = griddata(pts, vals, ipts, method='linear')
    ui, vi, wi = ivals.T
    # ui.shape = vi.shape = wi.shape = (100, 100)

    ui=U
    vi=V
    wi=W

    mins=[150,150, 150]
    maxs=[250, 250, 250]


    mins=[90,150, 90]
    maxs=[190, 250, 190]
    isOver = (pts > mins).all(axis=1)
    isUnder = (pts < maxs).all(axis=1)

    pt2plot=np.logical_and(isOver, isUnder)

    import plotly.io as pio
    pio.renderers.default = "browser"

    fig = go.Figure(data=go.Streamtube(x=xi[pt2plot], y=yi[pt2plot], z=zi[pt2plot],
                                       u=-ui[pt2plot], v=vi[pt2plot], w=-wi[pt2plot]))
    fig.show()

    slices=[400,600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800,3000, 3200, 3400, 3600, 3800, 4000,  4500]
    slices=[300, 400, 450, 500]#[100 ,150, 200, 250, 350]
    # slice=300
    slices_sagital=[210, 165]
    sxe='coronal'#'sagital'
    for slice in slices:
        if sxe=='sagital':
            # art_graph_vector_spher, art_press_conn = getEdgeVectorSpheric(art_graph)

            # grid=art_edge_coordinates
            # grid_vector=art_graph_vector

            grid_coordinates_2plot = grid[grid[:, 2] > slice]
            grid_vector_2plot = grid_vector[grid[:, 2] > slice]
            # art_graph_vector_spher_2plot = spher_ori[art_edge_coordinates[:, 1] > slice]
            grid_vector_2plot = grid_vector_2plot[grid_coordinates_2plot[:, 2] < slice+2.5]#100
            # art_graph_vector_spher_2plot= art_graph_vector_spher_2plot[art_edge_coordinates_2plot[:, 1] < slice+100]
            grid_coordinates_2plot = grid_coordinates_2plot[grid_coordinates_2plot[:, 2] < slice+2.5]#100

            # import random
            # e2plot=random.sample(range(grid_coordinates_2plot.shape[0]), k=int(grid_coordinates_2plot.shape[0]/1))
            # grid_coordinates_2plot=grid_coordinates_2plot[e2plot]
            # grid_vector_2plot=grid_vector_2plot[e2plot]

            center=(np.median(grid_coordinates_2plot[:,0]), np.max(grid_coordinates_2plot[:,1])-30)#-500#(3500, 2000)
            print(center)
            grid_coordinates_2plot=np.array([grid_coordinates_2plot[i, [0,1]]-center for i in range(grid_coordinates_2plot.shape[0])])
            grid_vector_spher_2plot= [np.dot(preprocessing.normalize(np.nan_to_num(grid_coordinates_2plot), norm='l2')[i],preprocessing.normalize(np.nan_to_num(grid_vector_2plot[:, [0,1]]))[i]) for i in range(grid_coordinates_2plot.shape[0])]
            X = grid_coordinates_2plot[:, 1]#-center[0]
            Y = grid_coordinates_2plot[:, 0]#-center[1]
            grid_vector_2plot=preprocessing.normalize(np.nan_to_num(grid_vector_2plot[:, [0,1]]))
            U = grid_vector_2plot[:, 1]
            V = grid_vector_2plot[:, 0]

            M=abs(np.array(grid_vector_spher_2plot))



            plt.figure()
            # plt.quiver(X[M>3], Y[M>3], U[M>3], V[M>3], M[M>3], pivot='mid')
            U=-U
            V=-V
            plt.quiver(X, Y, U, V, M, pivot='mid')#[M_normed[:,0]>1]
            plt.title(str(slice))
            plt.gca().invert_yaxis()
            plt.gca().invert_xaxis()

            from scipy.interpolate import griddata


            xi = np.linspace(X.min(), X.max(), 100)
            yi = np.linspace(Y.min(), Y.max(), 100)

            # an (nx * ny, 2) array of x,y coordinates to interpolate at
            # ipts = np.vstack(a.ravel() for a in np.meshgrid(yi, xi)[::-1]).T
            pts = np.vstack((X, Y)).T
            vals = np.vstack((U, V)).T
            ipts = np.vstack(a.ravel() for a in np.meshgrid(xi, yi)).T
            ivals = griddata(pts, vals, ipts, method='cubic')

            ui, vi = ivals.T
            ui.shape = vi.shape = (100, 100)
            colors_rgb = M.reshape(ui.shape)

            # an (nx * ny, 2) array of interpolated u, v values

            plt.figure()
            with plt.style.context('dark_background'):
                # plt.rcParams['axes.facecolor'] = 'black'
                edges2 = feature.canny(annotation[:, :, slice].T, sigma=0.1).astype(int)
                # mask=annotation[:228, :, slice].T>0
                mask=np.ma.masked_where(annotation[:, :, slice].T >0, annotation[:, :, slice].T)
                xi=xi-np.min(xi)
                yi=yi-np.min(yi)
                # plt.streamplot(xi, yi, -ui, -vi, color=colors_rgb, density=10,arrowstyle='-', zorder=2)#color=ui+vi
                plt.streamplot(xi, yi, -ui, -vi, density=15, arrowstyle='-', zorder=2)  # color=ui+vi
                # plt.quiver(X, Y, -U, -V,pivot='mid')  # [M_normed[:,0]>1]

                plt.gca().invert_yaxis()
                plt.gca().invert_xaxis()
                plt.imshow(mask, cmap='gray', zorder=10)
                plt.imshow(edges2, cmap='gray', zorder=1)  # jet#copper#
                edges2 = np.logical_not(edges2)
                # edges2[edges2==0]=np.nan
                masked_data = np.ma.masked_where(edges2 >0, edges2)
                plt.imshow(masked_data,cmap='cool_r', zorder=11)#jet#copper#
                # plt.axis('off')
                plt.title(str(slice))

        elif sxe=='coronal':#'sagital':
            grid_coordinates_2plot = grid[grid[:, 1] > slice]
            grid_vector_2plot = grid_vector[grid[:, 1] > slice]
            # art_graph_vector_spher_2plot = spher_ori[art_edge_coordinates[:, 1] > slice]
            grid_vector_2plot = grid_vector_2plot[grid_coordinates_2plot[:, 1] < slice + 5]  # 100
            # art_graph_vector_spher_2plot= art_graph_vector_spher_2plot[art_edge_coordinates_2plot[:, 1] < slice+100]
            grid_coordinates_2plot = grid_coordinates_2plot[grid_coordinates_2plot[:, 1] < slice + 5]  # 100



            center = (
            np.median(grid_coordinates_2plot[:, 0]), np.max(grid_coordinates_2plot[:, 2]) - 30)  # -500#(3500, 2000)
            print(center)
            grid_coordinates_2plot = np.array(
                [grid_coordinates_2plot[i, [0, 2]] - center for i in range(grid_coordinates_2plot.shape[0])])
            grid_vector_spher_2plot = [np.dot(preprocessing.normalize(np.nan_to_num(grid_coordinates_2plot), norm='l2')[i],
                                              preprocessing.normalize(np.nan_to_num(grid_vector_2plot[:, [0, 2]]))[i]) for i
                                       in range(grid_coordinates_2plot.shape[0])]
            X = grid_coordinates_2plot[:, 1]  # -center[0]
            Y = grid_coordinates_2plot[:, 0]  # -center[1]
            grid_vector_2plot = preprocessing.normalize(np.nan_to_num(grid_vector_2plot[:, [0, 2]]))
            U = grid_vector_2plot[:, 1]
            V = grid_vector_2plot[:, 0]

            M = abs(np.array(grid_vector_spher_2plot))


            plt.figure()
            # plt.quiver(X[M>3], Y[M>3], U[M>3], V[M>3], M[M>3], pivot='mid')
            U = -U
            V = -V
            plt.quiver(X, Y, U, V, M, pivot='mid')  # [M_normed[:,0]>1]
            plt.title(str(slice))
            plt.gca().invert_yaxis()
            plt.gca().invert_xaxis()

            from scipy.interpolate import griddata

            xi = np.linspace(X.min(), X.max(), 100)
            yi = np.linspace(Y.min(), Y.max(), 100)

            # an (nx * ny, 2) array of x,y coordinates to interpolate at
            # ipts = np.vstack(a.ravel() for a in np.meshgrid(yi, xi)[::-1]).T
            pts = np.vstack((X, Y)).T
            vals = np.vstack((U, V)).T
            ipts = np.vstack(a.ravel() for a in np.meshgrid(xi, yi)).T
            ivals = griddata(pts, vals, ipts, method='cubic')

            ui, vi = ivals.T
            ui.shape = vi.shape = (100, 100)
            colors_rgb = M.reshape(ui.shape)

            # an (nx * ny, 2) array of interpolated u, v values

            plt.figure()
            with plt.style.context('dark_background'):
                # plt.rcParams['axes.facecolor'] = 'black'
                edges2 = feature.canny(annotation[:228, :, slice].T, sigma=0.1).astype(int)
                # mask=annotation[:228, :, slice].T>0
                mask = np.ma.masked_where(annotation[:228, :, slice].T > 0, annotation[:228, :, slice].T)
                xi = xi - np.min(xi)
                yi = yi - np.min(yi)
                plt.streamplot(xi, yi, -ui, -vi, color=colors_rgb, density=15, arrowstyle='-', zorder=2)  # color=ui+vi
                # plt.quiver(X, Y, -U, -V,pivot='mid')  # [M_normed[:,0]>1]

                plt.gca().invert_yaxis()
                plt.gca().invert_xaxis()
                plt.imshow(mask, cmap='gray', zorder=10)
                plt.imshow(edges2, cmap='gray', zorder=1)  # jet#copper#
                edges2 = np.logical_not(edges2)
                # edges2[edges2==0]=np.nan
                masked_data = np.ma.masked_where(edges2 > 0, edges2)
                plt.imshow(masked_data, cmap='cool_r', zorder=11)  # jet#copper#
                plt.axis('off')
                plt.title(str(slice))
