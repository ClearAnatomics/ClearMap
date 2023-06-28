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
# work_dir='/data_SSD_2to/191122Otof'#'/data_SSD_2to/whiskers_graphs'
# brains=['1R','2R','3R','5R','6R','7R', '8R', '4R']#['44R', '30R']#, '39L']
# brains=['2R','3R','5R','6R','7R', '8R', '4R']
# control='2R'
# controls=['2R','3R','5R', '8R']
# work_dir = '/data_SSD_2to/whiskers_graphs/new_graphs'
# brains = ['142L', '158L', '162L', '164L','138L', '141L', '163L', '165L']
#
# for i, g in enumerate(brains):
#     print(g)
#     graph = ggt.load(
#         work_dir + '/' + g + '/' + '/data_graph_correcteduniverse.gt')  # 'data_graph_correcteduniverse.gt')
#     degrees = graph.vertex_degrees()
#     vf = np.logical_and(degrees > 1, degrees <= 4)
#     graph = graph.sub_graph(vertex_filter=vf)
#     label = graph.vertex_annotation();
#     # vertex_filter = from_e_prop2_vprop(graph, 'artery')
#     # art_tree = graph.sub_graph(vertex_filter=vertex_filter)
#     # art_tree = graph.sub_graph(vertex_filter=vertex_filter)
#     with open(work_dir + '/' + g + '/sampledict' + g + '.pkl', 'rb') as fp:
#         sampledict = pickle.load(fp)
#
#     pressure = np.asarray(sampledict['pressure'][0])
#     conn = graph.edge_connectivity()
#
#     presu_conn=np.array([conn[k,np.argsort([pressure[conn[k,0]],pressure[conn[k,1]]]).tolist()] for k in range(conn.shape[0])])
#
#     graph_directed=ggt.Graph()
#     graph_directed.add_vertex(graph.n_vertices)
#     graph_directed.add_edge(presu_conn)
#
#     graph_directed.add_vertex_property('radii', graph.vertex_property('radii'))
#     graph_directed.add_vertex_property('coordinates', graph.vertex_property('coordinates'))
#     graph_directed.add_vertex_property('distance_to_surface', graph.vertex_property('distance_to_surface'))
#     graph_directed.add_vertex_property('coordinates_atlas', graph.vertex_property('coordinates_atlas'))
#     graph_directed.add_vertex_property('annotation', graph.vertex_property('annotation'))
#     graph_directed.add_vertex_property('radii_atlas', graph.vertex_property('radii_atlas'))
#
#     graph_directed.add_edge_property('radii', graph.edge_property('radii'))
#     graph_directed.add_edge_property('length', graph.edge_property('length'))
#     graph_directed.add_edge_property('distance_to_surface', graph.edge_property('distance_to_surface'))
#     graph_directed.add_edge_property('edge_geometry_indices', graph.edge_property('edge_geometry_indices'))
#     graph_directed.add_edge_property('radii_atlas', graph.edge_property('radii_atlas'))
#
#     graph_directed.save(work_dir + '/' + g + '/' + '/data_graph_correcteduniverse_directed.gt')
#
#
# work_dir = '/data_SSD_2to/191122Otof'
# control='2R'
# graph_i = ggt.load(work_dir + '/' + control + '/data_graph_correcteduniverse.gt')#data_graph_correcteduniverse
#
# degrees = graph_i.vertex_degrees()
# vf = np.logical_and(degrees > 1, degrees <= 4)
# graph_i = graph_i.sub_graph(vertex_filter=vf)
# label = graph_i.vertex_annotation();
# # vertex_filter = from_e_prop2_vprop(graph, 'artery')
# # art_tree = graph.sub_graph(vertex_filter=vertex_filter)
# # art_tree = graph.sub_graph(vertex_filter=vertex_filter)
# with open(work_dir + '/' + control + '/sampledict' + control + '.pkl', 'rb') as fp:
#     sampledict = pickle.load(fp)
#
# pressure = np.asarray(sampledict['pressure'][0])
# graph_i.add_vertex_property('pressure', pressure)
#
# #hippocampus
# level=6
# order=463
#
# #barrels
# level=9
# order=54
#
# label = graph_i.vertex_annotation();
# vertex_filter = np.zeros(graph_i.n_vertices)
#
# print(level, order, ano.find(order, key='order')['name'])
# label_leveled = ano.convert_label(label, key='order', value='order', level=level)
# vertex_filter[label_leveled == order] = 1;
#
# graph=graph_i.sub_graph(vertex_filter=vertex_filter)


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def cart2sph(x,y, ceval=ne.evaluate):
    """ x, y, z :  ndarray coordinates
        ceval: backend to use:
              - eval :  pure Numpy
              - numexpr.evaluate:  Numexpr """
    if x.shape[0]!=0:
        r = ceval('sqrt(x**2+y**2)')#sqrt(x * x + y * y + z * z)
        # theta = ceval('arccos(z/r)*180')/pi#acos(z / r) * 180 / pi  # to degrees
        phi = ceval('arctan2(y,x)*180')/pi#*180/3.4142
        # azimuth = ceval('arctan2(y,x)')
        # xy2 = ceval('x**2 + y**2')
        # elevation = ceval('arctan2(z, sqrt(xy2))')
        # r = ceval('sqrt(xy2 + z**2)')
        rmax=np.max(r)
    else:
        print('no orientation to compute')
        r=np.array([0])
        # theta=np.array([0])
        phi=np.array([0])
        rmax=1
    return phi/180, r/rmax#, theta/180, phi/180



def getEdgeVector(graph, cont, normed=False, oriented=True, criteria='distance'):
    x = graph.vertex_coordinates()[:, 0]
    y = graph.vertex_coordinates()[:, 1]
    z = graph.vertex_coordinates()[:, 2]
    # cont='4'
    if criteria=='pressure':
        try:
            pressure=graph.vertex_property('pressure')
        except:
            f, v = computeFlowFranca(work_dir, graph,cont)
            graph.add_edge_property('flow', f)
            graph.add_edge_property('veloc', v)

        with open(work_dir + '/' + cont + '/sampledict' + cont + '.pkl', 'rb') as fp:
            sampledict = pickle.load(fp)

        pressure = np.asarray(sampledict['pressure'][0])
        criterion=pressure
    elif criteria=='distance':
        criterion=graph.vertex_property('distance_to_surface')

    if oriented:
        conn = graph.edge_connectivity()

        presu_conn = np.array(
            [conn[k, np.argsort([criterion[conn[k, 0]], criterion[conn[k, 1]]]).tolist()] for k in range(conn.shape[0])])

        connectivity=presu_conn
    else:
        connectivity = graph.edge_connectivity()
    print(x.shape)
    print(connectivity.shape)
    edge_vect = np.array(
        [x[connectivity[:, 1]] - x[connectivity[:, 0]], y[connectivity[:, 1]] - y[connectivity[:, 0]],
         z[connectivity[:, 1]] - z[connectivity[:, 0]]]).T
    if normed:
        normed_edge_vect = np.array([edge_vect[i] / np.linalg.norm(edge_vect[i]) for i in range(edge_vect.shape[0])])
        return normed_edge_vect,connectivity
    else:
        return edge_vect,connectivity






#
#
# spherical_ori = np.array(
#     [x_s[connectivity[:, 1]] - x_s[connectivity[:, 0]], y_s[connectivity[:, 1]] - y_s[connectivity[:, 0]],
#      z_s[connectivity[:, 1]] - z_s[connectivity[:, 0]]]).T
# # orientations=preprocessing.normalize(orientations, norm='l2')
#
# # edge_colors = (vertex_colors[connectivity[:, 0]] + vertex_colors[connectivity[:, 1]]) / 2.0;
#
# # spherical_ori=np.array(cart2sph(orientations[:, 0],orientations[:, 1],orientations[:, 2], ceval=ne.evaluate)).T
# spherical_ori = preprocessing.normalize(spherical_ori, norm='l2')
# spherical_ori = np.abs(spherical_ori)


def getEdgeVectorSpheric(x, y, center, normed=False, oriented=True):
    spherical_coord = np.array(cart2sph(x-center[0], y-center[1], ceval=ne.evaluate)).T
    x = spherical_coord[:, 0]
    y = spherical_coord[:, 1]
    z = spherical_coord[:, 2]
    try:
        pressure=graph.vertex_property('pressure')
    except:
        f, v = computeFlowFranca(work_dir, graph)
        graph.add_edge_property('flow', f)
        graph.add_edge_property('veloc', v)

    if oriented:
        conn = graph.edge_connectivity()

        presu_conn = np.array(
            [conn[k, np.argsort([pressure[conn[k, 0]], pressure[conn[k, 1]]]).tolist()[::-1]] for k in range(conn.shape[0])])

        connectivity=presu_conn

    else:
        connectivity = graph.edge_connectivity()
    print(x.shape)
    print(connectivity.shape)
    edge_vect = np.array(
        [x[connectivity[:, 1]] - x[connectivity[:, 0]], y[connectivity[:, 1]] - y[connectivity[:, 0]],
         z[connectivity[:, 1]] - z[connectivity[:, 0]]]).T
    if normed:
        normed_edge_vect = np.array([edge_vect[i] / np.linalg.norm(edge_vect[i]) for i in range(edge_vect.shape[0])])
        return normed_edge_vect,connectivity
    else:
        return edge_vect,connectivity





def takeAvgStreamlines2(graph, controls, workdir, mode='bigvessels'):
    pi = math.pi
    G=[]
    # mode='bigvessels'
    average=False
    for control in controls:#[np.array([0,2,3,4])]:#mutants[np.array([0,2,3,4])]:
        graph = ggt.load(
            work_dir + '/' + control + '/' + 'data_graph_correcteduniverse.gt')  # data_graph_corrected_Isocortex.gt')#/data_graph_correcteduniverse.gt')#
        degrees = graph.vertex_degrees()
        vf = np.logical_and(degrees > 1, degrees <= 4)
        graph = graph.sub_graph(vertex_filter=vf)

        # with open(work_dir + '/' + control + '/sampledict' + control + '.pkl', 'rb') as fp:
        #     sampledict = pickle.load(fp)
        #
        # pressure = np.asarray(sampledict['pressure'][0])
        # graph.add_vertex_property('pressure', pressure)

        artery = from_e_prop2_vprop(graph, 'artery')
        vein = from_e_prop2_vprop(graph, 'vein')
        radii = graph.vertex_property('radii')
        d2s = graph.vertex_property('distance_to_surface')

        # artery =  graph.edge_property('artery')
        # vein =  graph.edge_property('vein')
        # adii = graph.edge_property('radii')
        # d2s = graph.edge_property('distance_to_surface')


        label = graph.vertex_annotation()
        if mode=='arteryvein':
            artery_vein = np.asarray(np.logical_or(artery, vein))
        elif mode=='bigvessels':
            artery_vein = np.asarray(np.logical_or(artery, vein))
            artery_vein = np.logical_or(artery_vein,from_e_prop2_vprop(graph, graph.edge_property('radii')>4))

        order, level = 1006, 3
        label_leveled = ano.convert_label(label, key='order', value='order', level=level)
        cerebellum = label_leveled == order;
        # radii = graph.edge_property('radii')
        # cerebellum_art=from_e_prop2_vprop(graph, np.logical_and(radii <=3, np.logical_and(from_v_prop2_eprop(graph, cerebellum),from_v_prop2_eprop(graph,artery_vein))))
        cerebellum_art = np.logical_and(radii <= 6, np.logical_and(cerebellum, artery_vein))
        # cg=graph.sub_graph(vertex_property=cerebellum_art)

        order, level = 463, 6
        label_leveled = ano.convert_label(label, key='order', value='order', level=level)
        hippocampus = label_leveled == order;
        # radii = graph.edge_property('radii')
        # hippocampus_art=from_e_prop2_vprop(graph, np.logical_and(radii <=3, np.logical_and(from_v_prop2_eprop(graph, hippocampus),from_v_prop2_eprop(graph,artery_vein))))
        hippocampus_art = np.logical_and(radii <= 6, np.logical_and(hippocampus, artery_vein))
        # hg = graph.sub_graph(vertex_filter=hippocampus_art)
        # p3d.plot_graph_mesh(hg)

        artery_vein[hippocampus] = 0
        artery_vein[cerebellum] = 0

        radii = graph.vertex_property('radii')
        pb_art = np.logical_or(hippocampus_art, cerebellum_art)
        artery_vein = np.logical_or(pb_art, np.logical_and(artery_vein, radii >= 3))#6
        # artery_vein = np.asarray(np.logical_and(artery_vein, radii >= 6))  # .nonzero()[0]

        artery_vein = np.asarray(np.logical_and(artery_vein, d2s >= 2))


        art_graph=graph.sub_graph(vertex_filter=artery_vein)#np.logical_or(artery, vein))
        art_graph_vector,art_press_conn=getEdgeVector(art_graph, control,criteria='distance')
        art_coordinates=art_graph.vertex_coordinates()
        art_edge_coordinates=np.array(
            [np.round((art_coordinates[art_press_conn[i, 0]] + art_coordinates[art_press_conn[i, 1]]) / 2) for i in range(art_press_conn.shape[0])])


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

        grid_x = np.linspace(0, np.max(art_edge_coordinates[:, 0]), 100)#100
        grid_y = np.linspace(0, np.max(art_edge_coordinates[:, 1]), 100)
        grid_z = np.linspace(0, np.max(art_edge_coordinates[:, 2]), 100)


        grid=np.array(np.meshgrid(grid_x, grid_y, grid_z)).reshape((3, 1000000)).T

        grid_flow_X = NNDI_x(grid)
        grid_flow_Y = NNDI_y(grid)
        grid_flow_Z = NNDI_z(grid)

        grid_vector=np.stack([grid_flow_X, grid_flow_Y,grid_flow_Z]).T
        G.append(grid_vector)

        # except:
        #     print('problem interpolation ...')
    Gmean=np.nanmean(np.array(G), axis=0)
    grid_vector = Gmean
    np.save(workdir+'/streamline_grid_avg_controls'+mode+'.npy',grid_vector )
    io.write(workdir+'/streamline_grid_avg_controls'+mode+'.tif',grid_vector )


def takeAvgStreamlines(controls, work_dir, mode='bigvessels'):
    pi = math.pi
    AEC=[]
    AGV=[]
    # mode='bigvessels'
    average=False
    for control in controls:#[np.array([0,2,3,4])]:#mutants[np.array([0,2,3,4])]:
        print(control)
        graph = ggt.load(
            work_dir + '/' + control + '/' + 'data_graph_correcteduniverse.gt')  # data_graph_corrected_Isocortex.gt')#/data_graph_correcteduniverse.gt')#
        degrees = graph.vertex_degrees()
        vf = np.logical_and(degrees > 1, degrees <= 4)
        graph = graph.sub_graph(vertex_filter=vf)
        print(graph)
        # with open(work_dir + '/' + control + '/sampledict' + control + '.pkl', 'rb') as fp:
        #     sampledict = pickle.load(fp)
        #
        # pressure = np.asarray(sampledict['pressure'][0])
        # graph.add_vertex_property('pressure', pressure)

        try:
            artery=graph.vertex_property('artery')
            vein=graph.vertex_property('vein')
        except:
            print('no artery vertex properties')
            artery=graph.vertex_radii()>=4
            vein=graph.vertex_radii()>=8
            graph.add_vertex_property('artery', artery)
            graph.add_vertex_property('vein', vein)
            artery=from_v_prop2_eprop(graph, artery)
            graph.add_edge_property('artery', artery)
            vein=from_v_prop2_eprop(graph, vein)
            graph.add_edge_property('vein', vein)

        artery = from_e_prop2_vprop(graph, 'artery')
        vein = from_e_prop2_vprop(graph, 'vein')
        radii = graph.vertex_property('radii')
        d2s = graph.vertex_property('distance_to_surface')

        # artery =  graph.edge_property('artery')
        # vein =  graph.edge_property('vein')
        # adii = graph.edge_property('radii')
        # d2s = graph.edge_property('distance_to_surface')



        label = graph.vertex_annotation()
        if mode=='arteryvein':
            artery_vein = np.asarray(np.logical_or(artery, vein))
        elif mode=='bigvessels':
            artery_vein = np.asarray(np.logical_or(artery, vein))
            artery_vein = np.logical_or(artery_vein,from_e_prop2_vprop(graph, graph.edge_property('radii')>4))

        order, level = 1006, 3
        label_leveled = ano.convert_label(label, key='order', value='order', level=level)
        cerebellum = label_leveled == order;
        # radii = graph.edge_property('radii')
        # cerebellum_art=from_e_prop2_vprop(graph, np.logical_and(radii <=3, np.logical_and(from_v_prop2_eprop(graph, cerebellum),from_v_prop2_eprop(graph,artery_vein))))
        cerebellum_art = np.logical_and(radii <= 6, np.logical_and(cerebellum, artery_vein))
        # cg=graph.sub_graph(vertex_property=cerebellum_art)

        order, level = 463, 6
        label_leveled = ano.convert_label(label, key='order', value='order', level=level)
        hippocampus = label_leveled == order;
        # radii = graph.edge_property('radii')
        # hippocampus_art=from_e_prop2_vprop(graph, np.logical_and(radii <=3, np.logical_and(from_v_prop2_eprop(graph, hippocampus),from_v_prop2_eprop(graph,artery_vein))))
        hippocampus_art = np.logical_and(radii <= 6, np.logical_and(hippocampus, artery_vein))
        # hg = graph.sub_graph(vertex_filter=hippocampus_art)
        # p3d.plot_graph_mesh(hg)

        artery_vein[hippocampus] = 0
        artery_vein[cerebellum] = 0

        radii = graph.vertex_property('radii')
        pb_art = np.logical_or(hippocampus_art, cerebellum_art)
        artery_vein = np.logical_or(pb_art, np.logical_and(artery_vein, radii >= 3))#6
        # artery_vein = np.asarray(np.logical_and(artery_vein, radii >= 6))  # .nonzero()[0]

        artery_vein = np.asarray(np.logical_and(artery_vein, d2s >= 2))


        art_graph=graph.sub_graph(vertex_filter=artery_vein)#np.logical_or(artery, vein))
        art_graph_vector,art_press_conn=getEdgeVector(art_graph, control,criteria='distance')
        art_coordinates=art_graph.vertex_coordinates()
        art_edge_coordinates=np.array(
            [np.round((art_coordinates[art_press_conn[i, 0]] + art_coordinates[art_press_conn[i, 1]]) / 2) for i in range(art_press_conn.shape[0])])



        print(art_edge_coordinates)
        print(art_graph_vector)
        print(art_edge_coordinates.shape)
        print(art_graph_vector.shape)


        AEC.append(art_edge_coordinates)
        AGV.append(art_graph_vector)

    np.save(work_dir+'/streamline_AEC_controls'+mode+'.npy',AEC )
    np.save(work_dir+'/streamline_AGVcontrols'+mode+'.npy',AGV )
    print(work_dir+'/streamline_AGVcontrols'+mode+'.npy', art_edge_coordinates.shape, art_graph_vector.shape)
    # io.write(workdir+'/streamline_grid_avg_controls'+mode+'.tif',grid_vector )


def getLocalNormaleOrienttaion(graph, ref_graph, regions_ids, local_normal=False, calc_art=False, verbose=False):

    # if local_normal:
    #     rad_f = np.zeros(graph.n_edges)
    #     plan_f = np.zeros(graph.n_edges)
    #     lengths_f = graph.edge_property('length')
    #     norm_f = np.zeros(graph.n_edges)
    #     label = graph.vertex_annotation();
    #     for r in regions_ids:
    #         o = r#ano.find(r, key='id')['order']
    #         l = ano.find(r, key='id')['level']
    #
    #         label_leveled = ano.convert_label(label, key='id', value='id', level=l)
    #         vf = label_leveled == o  # 54;
    #         ef=from_v_prop2_eprop(graph, vf)
    #         try:
    #             # sub_graph=graph.sub_graph(edge_filter=ef)
    #             sub_graph = graph.sub_graph(vertex_filter=vf)
    #             if verbose:
    #                 print(ano.find(o, key='id')['name'], sub_graph)
    #             r, p, n,l=getRadPlanOrienttaion(sub_graph, sub_graph, local_normal=False, calc_art=calc_art)
    #
    #             rad_f[ef]=r
    #             plan_f[ef] = p
    #             norm_f[ef] = n
    #         except:
    #             if verbose:
    #                 print('problem', ano.find(o, key='id')['name'], ef.shape)
    #
    #     rad=rad_f
    #     plan=plan_f
    #     lengths=lengths_f
    #     N=norm_f
    #
    #
    # else:
    if verbose:
        print(graph, ref_graph)
    x = ref_graph.vertex_coordinates()[:, 0]
    y = ref_graph.vertex_coordinates()[:, 1]
    z = ref_graph.vertex_coordinates()[:, 2]
    dist = ref_graph.vertex_property('distance_to_surface')
    if calc_art:

        m=np.min(dist)+1.5

        x_surface=x[dist<m]

        y_surface = y[dist < m]

        z_surface = z[dist < m]

        coords=np.zeros((np.sum(dist<m), 3))
        coords[:, 0]=x_surface
        coords[:, 1]=y_surface
        coords[:,2]=z_surface
        if verbose:
            print(coords.shape)
        G = coords.sum(axis=0) / coords.shape[0]
        # run SVD
        u, s, vh = np.linalg.svd(coords - G)
        # unitary normal vector
        top2bot = vh[2, :]
        if verbose:
            print(top2bot, np.linalg.norm(top2bot))
    else:
        x = ref_graph.vertex_coordinates()[:, 0]
        y = ref_graph.vertex_coordinates()[:, 1]
        z = ref_graph.vertex_coordinates()[:, 2]

        dist_min=np.min(dist)
        dist_max=np.max(dist)
        top_vertices=dist<=dist_min+2
        bottom_vertices=dist>= dist_max-1

        top2bot=np.array([np.mean(x[bottom_vertices])-np.mean(x[top_vertices]), np.mean(y[bottom_vertices])-np.mean(y[top_vertices]), np.mean(z[bottom_vertices])-np.mean(z[top_vertices])])
        top2bot=top2bot / np.linalg.norm(top2bot)#preprocessing.normalize(top2bot, norm='l2')
        if verbose:
            print(top2bot)

    x = graph.vertex_coordinates()[:, 0]
    y = graph.vertex_coordinates()[:, 1]
    z = graph.vertex_coordinates()[:, 2]

    connectivity = graph.edge_connectivity()
    lengths = graph.edge_property('length')
    edge_vect = np.array(
        [x[connectivity[:, 1]] - x[connectivity[:, 0]], y[connectivity[:, 1]] - y[connectivity[:, 0]],
         z[connectivity[:, 1]] - z[connectivity[:, 0]]]).T

    normed_edge_vect=np.array([edge_vect[i] / np.linalg.norm(edge_vect[i]) for i in range(edge_vect.shape[0])])
    N=np.linalg.norm(edge_vect[i])
    print(N)
    # normed_edge_vect=normed_edge_vect[~np.isnan(normed_edge_vect)]
    rad=np.array([np.dot(top2bot.transpose(), normed_edge_vect[i]) for i in range(edge_vect.shape[0])])
    plan=np.sqrt(1-rad**2)

    # lengths=lengths[np.asarray(~np.isnan(rad))]
    # rad = rad[~np.isnan(rad)]
    # plan = plan[~np.isnan(plan)]
    return abs(rad), abs(plan), N, lengths


def GeneralizedRadPlanorientation(graph, cont, rad, controls, corrected=True, d2s=True, mode='arteryvein', average=False, dvlpmt=True):
    print(mode, average)
    pi = math.pi
    artery=from_e_prop2_vprop(graph, 'artery')
    vein=from_e_prop2_vprop(graph, 'vein')
    radii=graph.vertex_property('radii')
    d2s = graph.vertex_property('distance_to_surface')
    print(np.sum(artery), np.sum(vein))
    label = graph.vertex_annotation()


    # from scipy.interpolate import RegularGridInterpolator

    connectivity = graph.edge_connectivity()
    coordinates = graph.vertex_property('coordinates_atlas')  # *1.625/25
    edges_centers = np.array(
        [(coordinates[connectivity[i, 0]] + coordinates[connectivity[i, 1]]) / 2 for i in range(connectivity.shape[0])])


    graph_vector, press_conn = getEdgeVector(graph, cont,criteria='distance')
    coordinates = graph.vertex_coordinates()
    edge_coordinates = np.array(
        [np.round((coordinates[press_conn[i, 0]] + coordinates[press_conn[i, 1]]) / 2) for i in
         range(press_conn.shape[0])])


    from scipy.interpolate import NearestNDInterpolator,LinearNDInterpolator

    if not average:
        print('not averaged')
        if not corrected:
            print('not corrected')

            if mode=='arteryvein':
                artery_vein = np.asarray(np.logical_or(artery, vein))
            elif mode=='bigvessels':
                artery_vein = np.asarray(np.logical_or(artery, vein))
                artery_vein = np.logical_or(artery_vein,from_e_prop2_vprop(graph, graph.edge_property('radii')>rad))#6
                # artery_vein = np.logical_or(artery_vein,np.logical_and(from_e_prop2_vprop(graph, graph.edge_property('radii')<8),from_e_prop2_vprop(graph, graph.edge_property('radii')>5.5)))

            # artery_vein=np.asarray(np.logical_and(artery_vein, radii>=rad))#.nonzero()[0]
            artery_vein = np.asarray(np.logical_or(artery_vein, radii >= rad))  # .nonzero()[0]
            # if d2s:
            d2s = graph.vertex_property('distance_to_surface')
            artery_vein = np.asarray(np.logical_and(artery_vein, d2s >= 7))  # 7  .nonzero()[0]
            capi = np.logical_not(np.logical_or(artery, vein)).nonzero()[0]

            big_radii = graph.edge_property('radii')>4
            artery_vein_edge = np.asarray(np.logical_or(graph.edge_property('artery'), graph.edge_property('vein')))
            artery_vein_edge = np.asarray(np.logical_and(artery_vein_edge, big_radii)).nonzero()[0]
            capi_edge = np.logical_not(np.logical_or( graph.edge_property('artery'),  graph.edge_property('vein'))).nonzero()[0]


        else:
            print('corrected')
            label = graph.vertex_annotation()
            if mode=='arteryvein':
                artery_vein = np.asarray(np.logical_or(artery, vein))
            elif mode=='bigvessels':
                artery_vein = np.asarray(np.logical_or(artery, vein))
                artery_vein = np.logical_or(artery_vein,from_e_prop2_vprop(graph, graph.edge_property('radii')>rad))
                # artery_vein = np.logical_or(artery_vein,np.logical_and(from_e_prop2_vprop(graph, graph.edge_property('radii')<8),from_e_prop2_vprop(graph, graph.edge_property('radii')>5.5)))

            try:
                if not dvlpmt:
                    order, level = 1006, 3
                    label_leveled = ano.convert_label(label, key='order', value='order', level=level)
                    cerebellum = label_leveled == order;
                    # radii = graph.edge_property('radii')
                    # cerebellum_art=from_e_prop2_vprop(graph, np.logical_and(radii <=3, np.logical_and(from_v_prop2_eprop(graph, cerebellum),from_v_prop2_eprop(graph,artery_vein))))
                    cerebellum_art = np.logical_and(radii <= 6, np.logical_and(cerebellum, artery_vein))
                    # cg=graph.sub_graph(vertex_property=cerebellum_art)

                    order, level = 463, 6
                    label_leveled = ano.convert_label(label, key='order', value='order', level=level)
                    hippocampus = label_leveled == order;
                    # radii = graph.edge_property('radii')
                    # hippocampus_art=from_e_prop2_vprop(graph, np.logical_and(radii <=3, np.logical_and(from_v_prop2_eprop(graph, hippocampus),from_v_prop2_eprop(graph,artery_vein))))
                    hippocampus_art = np.logical_and(radii <= 6, np.logical_and(hippocampus, artery_vein))
                    # hg = graph.sub_graph(vertex_filter=hippocampus_art)
                    # p3d.plot_graph_mesh(hg)

                    artery_vein[hippocampus] = 0
                    artery_vein[cerebellum] = 0

                    radii = graph.vertex_property('radii')
                    pb_art = np.logical_or(hippocampus_art, cerebellum_art)
                    artery_vein = np.logical_or(pb_art, np.logical_and(artery_vein, radii >= 3))#6
                    # artery_vein = np.asarray(np.logical_and(artery_vein, radii >= 6))  # .nonzero()[0]
            except:
                print('could not find the region, check json file')
            # artery_vein = np.asarray(np.logical_and(artery_vein, d2s >= 2))



        art_graph=graph.sub_graph(vertex_filter=artery_vein)#np.logical_or(artery, vein))
        art_graph_vector,art_press_conn=getEdgeVector(art_graph, cont,criteria='distance')
        art_coordinates=art_graph.vertex_coordinates()
        art_edge_coordinates=np.array(
            [np.round((art_coordinates[art_press_conn[i, 0]] + art_coordinates[art_press_conn[i, 1]]) / 2) for i in range(art_press_conn.shape[0])])


        # try:
        # if not average:
        # from scipy.interpolate import Rbf
        # print(art_edge_coordinates)
        # print(art_graph_vector)
        # print(art_edge_coordinates.shape)
        # print(art_graph_vector.shape)
        NNDI_x = LinearNDInterpolator(art_edge_coordinates, art_graph_vector[:,0])
        NNDI_y = LinearNDInterpolator(art_edge_coordinates, art_graph_vector[:, 1])
        NNDI_z = LinearNDInterpolator(art_edge_coordinates, art_graph_vector[:, 2])

        local_flow_X = NNDI_x(edge_coordinates)
        local_flow_Y = NNDI_y(edge_coordinates)
        local_flow_Z = NNDI_z(edge_coordinates)

        angle2normal=np.array([angle_between(graph_vector[i],[local_flow_X[i], local_flow_Y[i], local_flow_Z[i]] ) for i in range(graph.n_edges)])*180/pi
        cosinus2normal=abs(np.cos(angle2normal*pi/180))
        angle = np.array([math.acos(cosinus2normal[i]) for i in range(cosinus2normal.shape[0])]) * 180 / pi
        print('return here')
        return angle,graph

    elif average:

        # grid_x = np.linspace(0, np.max(art_edge_coordinates[:, 0])+100, 100)#100
        # grid_y = np.linspace(0, np.max(art_edge_coordinates[:, 1])+100, 100)
        # grid_z = np.linspace(0, np.max(art_edge_coordinates[:, 2])+100, 100)
        # grid=np.array(np.meshgrid(grid_x, grid_y, grid_z)).reshape((3, 1000000)).T
        #
        # sample=np.random.choice(1000000, 100000)
        label = graph.vertex_annotation()
        if mode=='arteryvein':
            artery_vein = np.asarray(np.logical_or(artery, vein))
        elif mode=='bigvessels':
            artery_vein = np.asarray(np.logical_or(artery, vein))
            artery_vein = np.logical_or(artery_vein,from_e_prop2_vprop(graph, graph.edge_property('radii')>rad))
            # artery_vein = np.logical_or(artery_vein,from_e_prop2_vprop(graph, graph.edge_property('radii')<8))

        try:
            order, level = 1006, 3
            label_leveled = ano.convert_label(label, key='order', value='order', level=level)
            cerebellum = label_leveled == order;
            # radii = graph.edge_property('radii')
            # cerebellum_art=from_e_prop2_vprop(graph, np.logical_and(radii <=3, np.logical_and(from_v_prop2_eprop(graph, cerebellum),from_v_prop2_eprop(graph,artery_vein))))
            cerebellum_art = np.logical_and(radii <= 6, np.logical_and(cerebellum, artery_vein))
            # cg=graph.sub_graph(vertex_property=cerebellum_art)

            order, level = 463, 6
            label_leveled = ano.convert_label(label, key='order', value='order', level=level)
            hippocampus = label_leveled == order;
            # radii = graph.edge_property('radii')
            # hippocampus_art=from_e_prop2_vprop(graph, np.logical_and(radii <=3, np.logical_and(from_v_prop2_eprop(graph, hippocampus),from_v_prop2_eprop(graph,artery_vein))))
            hippocampus_art = np.logical_and(radii <= 6, np.logical_and(hippocampus, artery_vein))
            # hg = graph.sub_graph(vertex_filter=hippocampus_art)
            # p3d.plot_graph_mesh(hg)

            artery_vein[hippocampus] = 0
            artery_vein[cerebellum] = 0

            radii = graph.vertex_property('radii')
            pb_art = np.logical_or(hippocampus_art, cerebellum_art)
            artery_vein = np.logical_or(pb_art, np.logical_and(artery_vein, radii >= 3))#6
        # artery_vein = np.asarray(np.logical_and(artery_vein, radii >= 6))  # .nonzero()[0]
        except:
            print('could not find the region, check json file')

        artery_vein = np.asarray(np.logical_and(artery_vein, d2s >= 2))

        graph=graph.sub_graph(vertex_filter=np.logical_not(artery_vein))#np.logical_or(artery, vein))
        graph_vector, press_conn = getEdgeVector(graph, cont,criteria='distance')
        coordinates = graph.vertex_coordinates()
        edge_coordinates = np.array(
        [np.round((coordinates[press_conn[i, 0]] + coordinates[press_conn[i, 1]]) / 2) for i in
         range(press_conn.shape[0])])


        try:
            # grid_vector=np.load(work_dir+'/streamline_grid_avg_controls'+mode+'.npy')
            AEC=np.load(work_dir+'/streamline_AEC_controls'+mode+'.npy', allow_pickle=True)
            AGV=np.load(work_dir+'/streamline_AGVcontrols'+mode+'.npy', allow_pickle=True )
        except:
            takeAvgStreamlines(controls, work_dir, mode=mode)
            # grid_vector=np.load(work_dir+'/streamline_grid_avg_controls'+mode+'.npy')
            AEC=np.load(work_dir+'/streamline_AEC_controls'+mode+'.npy', allow_pickle=True)
            AGV=np.load(work_dir+'/streamline_AGVcontrols'+mode+'.npy', allow_pickle=True )
        # grid_vector = grid_vector.reshape((100, 100, 100, 3))
        # L_X=[]
        # L_Y=[]
        # L_Z=[]

        # for i in range(len(controls)):
        #     print(i)
        #     if i==0:
        #         grid=AEC[i]
        #         grid_vector=AGV[i]
        #     else:
        #         grid=np.concatenate((grid, AEC[i]),axis=0)
        #         grid_vector=np.concatenate((grid_vector, AGV[i]),axis=0)
        #
        angles=[]
        for i in range(len(controls)):
            # print(i, AEC[i])
            # print(AGV[i])
            grid=AEC[i]
            grid_vector=AGV[i]

            NNDI_x = LinearNDInterpolator(grid, grid_vector[:,0])
            NNDI_y = LinearNDInterpolator(grid, grid_vector[:,1])
            NNDI_z = LinearNDInterpolator(grid, grid_vector[:,2])

            local_flow_X = NNDI_x(edge_coordinates)
            local_flow_Y = NNDI_y(edge_coordinates)
            local_flow_Z = NNDI_z(edge_coordinates)

            # L_X.append(local_flow_X)
            # L_Y.append(local_flow_Y)
            # L_Z.append(local_flow_Z)

            # local_flow_X=np.median(np.array(L_X), axis=0)
            # local_flow_Y=np.median(np.array(L_Y), axis=0)
            # local_flow_Z=np.median(np.array(L_Z), axis=0)

            angle2normal=np.array([angle_between(graph_vector[i],[local_flow_X[i], local_flow_Y[i], local_flow_Z[i]] ) for i in range(graph.n_edges)])*180/pi
            cosinus2normal=abs(np.cos(angle2normal*pi/180))
            angle = np.array([math.acos(cosinus2normal[i]) for i in range(cosinus2normal.shape[0])]) * 180 / pi
            # return angle, graph
            # rad = angle <= limit_angle  # 40
            # planarity = angle > (90 - limit_angle)  # 60
            #
            #
            # # art_coordinates = art_tree.vertex_property('coordinates_atlas')  # *1.625/25
            # # print('artBP')  # artBP
            # # v = vox.voxelize(art_coordinates[:, :3], shape=template_shape, weights=None, radius=(radius, radius, radius), method='sphere');
            # # vox_art_mutant[:, :, :, i] = v
            # print('rad')
            # v = vox.voxelize(edges_centers[rad, :3], shape=template_shape, weights=None, radius=(radius, radius, radius), method='sphere');
            # w = vox.voxelize(edges_centers[:, :3], shape=template_shape, weights=None, radius=(radius, radius, radius), method='sphere');
            # io.write(work_dir + '/' +'vox_ori_fi_'+suffixe+'_control_rad'+str(radius)+str(average)+str(i)+'test1_4.tif', (v.array / w.array).astype('float32'))


            angles.append(angle)
        angle=np.nanmedian(np.array(angles), axis=0)
        return angle, graph

    ###### test to plot data
    # radiality = angle < 45
    # planarity = angle >= 45

    # artery_color = np.array([[0.82, 0.71, 0.55, 1], [1, 0, 0, 1]]).astype('float')
    # edge_colors = artery_color[np.asarray(radiality, dtype=int)]
    # # edge_colors=artery_color[np.asarray(rad>0.6, dtype=int)]
    # # edge_colors[np.asarray((p/(r+p))>0.6, dtype=bool)]=[0,1,0,1]
    # # edge_colors[np.asarray((p/(r+p))>0.75, dtype=bool)]=[0,1,0,1]
    #
    # edge_colors[np.asarray(planarity, dtype=bool)] = [0, 1, 0, 1]
    # graph.add_edge_property('color', edge_colors)
    # gs = graph.sub_slice((slice(1, 5000), slice(4795, 4955), slice(1, 7000)))#, coordinates='coordinates_atlas');#317#hippopocampus
    # # gs = graph.sub_slice((slice(1, 5000), slice(3000, 5000), slice(1, 7000)))#barrels
    # col = gs.edge_property('color')
    # q1 = p3d.plot_graph_mesh(gs, n_tube_points=3, edge_colors=col)
    #
    # # raw_coord=art_graph.edge_geometry_property('coordinates_atlas').astype('float16')#.transpose()
    # # transformer = KernelPCA(n_components=2, kernel='cosine')
    # # raw_coord_transformed = transformer.fit_transform(graph_vector[:])
    # # raw_coord_transformed.shape
    # #
    # # # coord=graph.edge_geometry_property('coordinates_atlas')
    # #
    # # capi_coord_transformed=transformer.transform(graph_vector[capi_edge])
    # # fig=plt.figure()
    # # ax = fig.add_subplot(111)#, projection='3d')
    # # ax.scatter(raw_coord_transformed[capi_edge, 0], raw_coord_transformed[capi_edge, 1], c="blue",
    # #             s=20, edgecolor='k')
    # # plt.scatter(capi_coord_transformed[graph.edge_property('artery').nonzero()[0], 0], capi_coord_transformed[graph.edge_property('artery').nonzero()[0], 1], c="red",
    # #             s=20, edgecolor='b')
    # # xs=raw_coord_transformed[:, 0]
    # # ys=raw_coord_transformed[:, 1]
    # # zs=raw_coord_transformed[:, 2]
    # # ax.scatter(xs, ys, zs, marker='o', color='red')
    # #
    # # xs = capi_coord_transformed[:, 0]
    # # ys = capi_coord_transformed[:, 1]
    # # zs = capi_coord_transformed[:, 2]
    # # ax.scatter(xs, ys, zs, marker='o', color='blue')
    #
    # # plt.title("Projection by KPCA")
    # # plt.xlabel(r"1st principal component in space induced by $\phi$")
    # # plt.ylabel("2nd component")
    # return angle
    # except:
    #     print('problem')
    #     return np.zeros(graph.n_edges)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri

def create_mesh_from_artery(graph):
    min_radius = 1000
    pressure_connectivity=art_graph.edge_property('pressure_connectivity')
    x = art_graph.edge_coordinates()[:, 0]
    y = art_graph.edge_coordinates()[:, 1]
    z = art_graph.edge_coordinates()[:, 2]
    art_graph_vector = getEdgeVector(art_graph, cont)


    triang = mtri.Triangulation(x, y)

    print(triang.triangles.shape)

    xmid = x[triang.triangles].mean(axis=1)
    ymid = y[triang.triangles].mean(axis=1)
    mask = np.where(xmid ** 2 + ymid ** 2 < 0.8e7, 1, 0)
    print(np.sum(mask))
    triang.set_mask(mask)
    print(triang.triangles.shape)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_trisurf(triang, z, cmap=plt.cm.CMRmap)
    # ax.scatter(x, y, z)
    plt.show()

#
# def from_e_prop2_vprop(graph, e_prop):
#     # e_prop = graph.edge_property(property)
#     v_prop=np.zeros(graph.n_vertices)
#     connectivity = graph.edge_connectivity()
#     v_prop[connectivity[e_prop==1,0]]=1
#     v_prop[connectivity[e_prop == 1,1]] = 1
#     # graph.add_vertex_property(property, v_prop)
#     return v_prop
#
# def from_v_prop2_eprop(graph, vprop):
#     # vprop = graph.vertex_property(property)
#     # e_prop=np.zeros(graph.n_edges)
#     connectivity = graph.edge_connectivity()
#     e_prop = np.logical_and(vprop[connectivity[:, 0]], vprop[connectivity[:, 1]])
#     return e_prop
#
#
# def from_v_prop2_eprop(graph, property):
#     vprop = graph.vertex_property(property)
#     # e_prop=np.zeros(graph.n_edges)
#     connectivity = graph.edge_connectivity()
#     e_prop = np.logical_and(vprop[connectivity[:, 0]], vprop[connectivity[:, 1]])
#     return e_prop
#
# def from_e_prop2_vprop(graph, property):
#     e_prop = graph.edge_property(property)
#     v_prop=np.zeros(graph.n_vertices)
#     connectivity = graph.edge_connectivity()
#     v_prop[connectivity[e_prop==1,0]]=1
#     v_prop[connectivity[e_prop == 1,1]] = 1
#     # graph.add_vertex_property(property, v_prop)
#     return v_prop
#



def radialVectorField(graph, slice, zmin, zmax, dim, sampling, function, mode='arteryvein'):
    # graph=graph_i
    pi = math.pi
    G=[]
    mode='bigvessels'
    average=True
    for control in mutants:#[np.array([0,2,3,4])]:#mutants[np.array([0,2,3,4])]:
        try:
            graph = ggt.load(work_dir + '/' + control + '/' + 'data_graph_correcteduniverse.gt')
        except:
            graph = ggt.load(work_dir + '/' + control + '/' + str(control)+'_graph.gt')

        degrees = graph.vertex_degrees()
        vf = np.logical_and(degrees > 1, degrees <= 4)
        graph = graph.sub_graph(vertex_filter=vf)

        try:
            with open(work_dir + '/' + control + '/sampledict' + control + '.pkl', 'rb') as fp:
                sampledict = pickle.load(fp)

            pressure = np.asarray(sampledict['pressure'][0])
            graph.add_vertex_property('pressure', pressure)
        except:
            print('no sample dict found for pressure and flow modelisation')

        # artery = from_e_prop2_vprop(graph, 'artery')
        # vein = from_e_prop2_vprop(graph, 'vein')
        try:
            artery = from_e_prop2_vprop(graph, 'artery')
            vein = from_e_prop2_vprop(graph, 'vein')
        except:
            print('no artery vertex properties')
            artery=graph.vertex_radii()>=4
            vein=graph.vertex_radii()>=8
            graph.add_vertex_property('artery', artery)
            graph.add_vertex_property('vein', vein)
            artery=from_v_prop2_eprop(graph, artery)
            graph.add_edge_property('artery', artery)
            vein=from_v_prop2_eprop(graph, vein)
            graph.add_edge_property('vein', vein)

        radii = graph.vertex_property('radii')
        d2s = graph.vertex_property('distance_to_surface')

        # artery =  graph.edge_property('artery')
        # vein =  graph.edge_property('vein')
        # adii = graph.edge_property('radii')
        # d2s = graph.edge_property('distance_to_surface')


        label =  graph.vertex_annotation()


        if mode=='arteryvein':
            artery_vein = np.asarray(np.logical_or(artery, vein))
        elif mode=='bigvessels':
            artery_vein = graph.vertex_property('radii')>4


        order, level=1006,3
        label_leveled = ano.convert_label(label, key='order', value='order', level=level)
        cerebellum = label_leveled == order;
        # radii = graph.edge_property('radii')
        # cerebellum_art=from_e_prop2_vprop(graph, np.logical_and(radii <=3, np.logical_and(from_v_prop2_eprop(graph, cerebellum),from_v_prop2_eprop(graph,artery_vein))))
        cerebellum_art = np.logical_and(radii <= 6,np.logical_and( cerebellum,artery_vein))
        # cg=graph.sub_graph(vertex_property=cerebellum_art)


        order, level = 463, 6
        label_leveled = ano.convert_label(label, key='order', value='order', level=level)
        hippocampus = label_leveled == order;
        # radii = graph.edge_property('radii')
        # hippocampus_art=from_e_prop2_vprop(graph, np.logical_and(radii <=3, np.logical_and(from_v_prop2_eprop(graph, hippocampus),from_v_prop2_eprop(graph,artery_vein))))
        hippocampus_art = np.logical_and(radii <= 6, np.logical_and(hippocampus, artery_vein))
        # hg = graph.sub_graph(vertex_filter=hippocampus_art)
        # p3d.plot_graph_mesh(hg)

        artery_vein[hippocampus] = 0
        artery_vein[cerebellum] = 0

        radii = graph.vertex_property('radii')
        pb_art=np.logical_or(hippocampus_art,cerebellum_art)
        artery_vein=np.logical_or(pb_art,np.logical_and(artery_vein, radii >= 6))
        # artery_vein = np.asarray(np.logical_and(artery_vein, radii >= 6))  # .nonzero()[0]

        artery_vein = np.asarray(np.logical_and(artery_vein, d2s >= 2))#4  # .nonzero()[0]
        # capi = np.logical_not(np.logical_or(artery, vein)).nonzero()[0]

        # big_radii = graph.edge_property('radii') > 4
        # artery_vein_edge = np.asarray(np.logical_or(graph.edge_property('artery'), graph.edge_property('vein')))
        # artery_vein_edge = np.asarray(np.logical_and(artery_vein_edge, big_radii)).nonzero()[0]
        # capi_edge = np.logical_not(np.logical_or(graph.edge_property('artery'), graph.edge_property('vein'))).nonzero()[0]
        # try:
        art_graph = graph.sub_graph(vertex_filter=artery_vein)  # np.logical_or(artery, vein))

        art_graph_vector, art_press_conn = getEdgeVector(art_graph, control, criteria='distance')

        graph_vector, press_conn = getEdgeVector(graph,control)

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

        grid_x = np.linspace(0, np.max(art_edge_coordinates[:, 0]), 100)#100
        grid_y = np.linspace(0, np.max(art_edge_coordinates[:, 1]), 100)
        grid_z = np.linspace(0, np.max(art_edge_coordinates[:, 2]), 100)


        grid=np.array(np.meshgrid(grid_x, grid_y, grid_z)).reshape((3, 1000000)).T

        grid_flow_X = NNDI_x(grid)
        grid_flow_Y = NNDI_y(grid)
        grid_flow_Z = NNDI_z(grid)

        grid_vector=np.stack([grid_flow_X, grid_flow_Y,grid_flow_Z]).T
        G.append(grid_vector)

        if average:
            # except:
            #     print('problem interpolation ...')
            Gmean=np.nanmean(np.array(G), axis=0)
            grid_vector = Gmean
            np.save(work_dir +'/streamline_grid_avg_controls'+mode+'.npy',grid_vector )

        if not average:
            print('not average')

            # from dipy.tracking.local_tracking import LocalTracking
            # from dipy.tracking.streamline import Streamlines
            # from dipy.tracking import utils
            # from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
            # from dipy.viz import window, actor, colormap, has_fury
            # from dipy.io.stateful_tractogram import Space, StatefulTractogram
            # from dipy.io.streamline import save_trk
            #
            # annotation=io.read('/home/sophie.skriabine/Projects/clearvessel-custom/ClearMap/Resources/Atlas/annotation_25_cropped.nrrd')
            # mask = np.ma.masked_where(annotation > 0, annotation)
            # seeds=utils.seeds_from_mask(mask, affine=static_affine, density=[2,2, 2])
            # # Initialization of LocalTracking. The computation happens in the next step.
            # stopping_criterion = ThresholdStoppingCriterion(np.mean(Gmean, axis=1).reshape((100, 100, 100)), .25)
            # streamlines_generator = LocalTracking(Gmean, stopping_criterion, seeds=seeds, affine=static_affine, step_size=.5)
            # # Generate streamlines object
            # streamlines = Streamlines(streamlines_generator)
            # # sft = StatefulTractogram(streamlines, hardi_img, Space.RASMM)
            # # save_trk(sft, "tractogram_probabilistic_thresh_all.trk")
            #
            # if has_fury:
            #     scene = window.Scene()
            #     scene.add(actor.line(streamlines, colormap.line_colors(streamlines)))
            #     window.record(scene, out_path='tractogram_deterministic_thresh_all.png',
            #                   size=(800, 800))
            #     if interactive:
            #         window.show(scene)
            # spher_ori=getVesselOrientation(art_graph, graph)
    from sklearn.preprocessing import normalize
    import cmath
    slices=[400,600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800,3000, 3200, 3400, 3600, 3800, 4000,  4500]
    slices=[300,  500]#[100 ,150, 200, 250, 350]
    # slice=3400
    slices_sagital=[100 ,150, 200, 250, 300,350,400, 450]#[210, 165]
    # slices_sagital = [50, 75, 100, 125, 150, 175, 210]
    slices_sagital=[329]#200
    sxe='coronal'#'coronal'#'sagital'
    for sl in slices_sagital:
        if sxe=='sagital':
            annotation = io.read(
                '/home/sophie.skriabine/Documents/ClearMap_Ressources/annotation_25_HeadLightOrientation_sagital_rotated.tif')

            # art_graph_vector_spher, art_press_conn = getEdgeVectorSpheric(art_graph)

            # grid=art_edge_coordinates
            # grid_vector=art_graph_vector
            # grid=grid_vector
            grid_coordinates_2plot = grid[grid[:, 2] > sl]
            grid_vector_2plot = grid_vector[grid[:, 2] > sl]
            # art_graph_vector_spher_2plot = spher_ori[art_edge_coordinates[:, 1] > sl]
            grid_vector_2plot = grid_vector_2plot[grid_coordinates_2plot[:, 2] < sl+2.5]#100
            # art_graph_vector_spher_2plot= art_graph_vector_spher_2plot[art_edge_coordinates_2plot[:, 1] < sl+100]
            grid_coordinates_2plot = grid_coordinates_2plot[grid_coordinates_2plot[:, 2] < sl+2.5]#100

            # import random
            # e2plot=random.sample(range(grid_coordinates_2plot.shape[0]), k=int(grid_coordinates_2plot.shape[0]/1))
            # grid_coordinates_2plot=grid_coordinates_2plot[e2plot]
            # grid_vector_2plot=grid_vector_2plot[e2plot]

            center=(np.median(grid_coordinates_2plot[:,0]), np.max(grid_coordinates_2plot[:,1])-30)#-500#(3500, 2000)
            print(center)
            grid_coordinates_2plot=np.array([grid_coordinates_2plot[i, [0,1]]-center for i in range(grid_coordinates_2plot.shape[0])])
            # grid_vector_spher_2plot= [np.dot(preprocessing.normalize(np.nan_to_num(grid_coordinates_2plot), norm='l2')[i],preprocessing.normalize(np.nan_to_num(grid_vector_2plot[:, [0,1]]))[i]) for i in range(grid_coordinates_2plot.shape[0])]
            X = grid_coordinates_2plot[:, 1]#-center[0]
            Y = grid_coordinates_2plot[:, 0]#-center[1]
            grid_vector_2plot=preprocessing.normalize(np.nan_to_num(grid_vector_2plot[:, [0,1]]))
            U = grid_vector_2plot[:, 1]
            V = grid_vector_2plot[:, 0]

            # M=abs(np.array(grid_vector_spher_2plot))



            # plt.figure()
            # # plt.quiver(X[M>3], Y[M>3], U[M>3], V[M>3], M[M>3], pivot='mid')
            # U=-U
            # V=-V
            # plt.quiver(X, Y, U, V, M, pivot='mid')#[M_normed[:,0]>1]
            # plt.title(str(sl))
            # plt.gca().invert_yaxis()
            # plt.gca().invert_xaxis()

            from scipy.interpolate import griddata


            xi = np.linspace(X.min(), X.max(), 100)
            yi = np.linspace(Y.min(), Y.max(), 100)

            # an (nx * ny, 2) array of x,y coordinates to interpolate at
            # ipts = np.vstack(a.ravel() for a in np.meshgrid(yi, xi)[::-1]).T
            pts = np.vstack((X, Y)).T
            vals = np.vstack((U, V)).T
            ipts = np.vstack(a.ravel() for a in np.meshgrid(xi, yi)).T
            ivals = griddata(pts, vals, ipts, method='cubic')

            ui, vi = preprocessing.normalize(ivals).T
            ui.shape = vi.shape = (100, 100)
            # colors_rgb = M.reshape(ui.shape)

            # an (nx * ny, 2) array of interpolated u, v values

            plt.figure()
            with plt.style.context('seaborn-white'):#dark_background
                # plt.rcParams['axes.facecolor'] = 'black'
                edges2 = feature.canny(annotation[:, :, sl].T, sigma=0.1).astype(int)
                # mask=annotation[:228, :, sl].T>0
                mask=np.ma.masked_where(annotation[:, :, sl].T >0, annotation[:, :, sl].T)
                xi=xi-np.min(xi)
                yi=yi-np.min(yi)
                # plt.streamplot(xi, yi, -ui, -vi, color=colors_rgb, density=10,arrowstyle='-', zorder=2)#color=ui+vi
                plt.streamplot(xi, yi, -ui, -vi, density=15, arrowstyle='-', color='k', zorder=2)  # color=ui+vi
                # plt.quiver(X, Y, -U, -V,pivot='mid')  # [M_normed[:,0]>1]

                # plt.gca().invert_yaxis()
                plt.gca().invert_xaxis()
                plt.imshow(mask, cmap='Greys', zorder=10)
                plt.imshow(edges2, cmap='Greys', zorder=1)  # jet#copper#
                edges2 = np.logical_not(edges2)
                # edges2[edges2==0]=np.nan
                masked_data = np.ma.masked_where(edges2 >0, edges2)
                plt.imshow(masked_data,cmap='cool_r', zorder=11)#jet#copper#
                # plt.axis('off')
                plt.title(str(sl))

        elif sxe=='coronal':#'sagital':
            import tifffile
            annotation=tifffile.imread( '/home/sophie.skriabine/Pictures/Reslice_of_annotation_25_HeadLightOrientation_coronal.tif')
            # annotation = io.read('/home/sophie.skriabine/Pictures/Reslice_of_annotation_25_HeadLightOrientation_coronal.tif')
            annotation=np.swapaxes(annotation, 0, 2)
            annotation=np.flip(annotation, 2)
            # annotation=np.flip(annotation, 1)

            grid_coordinates_2plot = grid[grid[:, 1] > sl]
            grid_vector_2plot = grid_vector[grid[:, 1] > sl]
            # art_graph_vector_spher_2plot = spher_ori[art_edge_coordinates[:, 1] > slice]
            grid_vector_2plot = grid_vector_2plot[grid_coordinates_2plot[:, 1] < sl + 5]  # 100
            # art_graph_vector_spher_2plot= art_graph_vector_spher_2plot[art_edge_coordinates_2plot[:, 1] < slice+100]
            grid_coordinates_2plot = grid_coordinates_2plot[grid_coordinates_2plot[:, 1] < sl + 5]  # 100



            center = (
            np.median(grid_coordinates_2plot[:, 0]), np.max(grid_coordinates_2plot[:, 2]) - 30)  # -500#(3500, 2000)
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


            # plt.figure()
            # # plt.quiver(X[M>3], Y[M>3], U[M>3], V[M>3], M[M>3], pivot='mid')
            # U = -U
            # V = -V
            # plt.quiver(X, Y, U, V, M, pivot='mid')  # [M_normed[:,0]>1]
            # plt.title(str(slice))
            # plt.gca().invert_yaxis()
            # plt.gca().invert_xaxis()

            from scipy.interpolate import griddata

            xi = np.linspace(X.min(), X.max(), 100)
            yi = np.linspace(Y.min(), Y.max(), 100)

            # an (nx * ny, 2) array of x,y coordinates to interpolate at
            # ipts = np.vstack(a.ravel() for a in np.meshgrid(yi, xi)[::-1]).T
            pts = np.vstack((X, Y)).T
            vals = np.vstack((U, V)).T
            ipts = np.vstack(a.ravel() for a in np.meshgrid(xi, yi)).T
            ivals = griddata(pts, vals, ipts, method='cubic')

            ui, vi = preprocessing.normalize(ivals).T
            ui.shape = vi.shape = (100, 100)

            # plt.figure()
            # with plt.style.context('seaborn-white'):
            #     # plt.rcParams['axes.facecolor'] = 'black'
            #     from skimage import feature
            #
            #     # mask = np.ma.masked_where(autofluo[:, :, sl]> 1000, autofluo[:, :, sl])
            #
            #     xi = xi - np.min(xi)
            #     yi = yi - np.min(yi)
            #     plt.streamplot(xi, yi,-ui, -vi, density=15, arrowstyle='-', color='k',
            #                    zorder=2)  # color=ui+vi#color=colors_rgb,
            #     # plt.quiver(X, Y, -U, -V,pivot='mid')  # [M_normed[:,0]>1]
            #
            #     # plt.gca().invert_yaxis()
            #     plt.gca().invert_xaxis()
            #     # plt.imshow(np.flip(mask, 1), cmap='Greys', zorder=10)
            #     # plt.imshow(np.flip(np.flip(mask, 0),1), cmap='Greys', zorder=10)
            #     # plt.axis('off')
            #     plt.title(str(sl))
            #     colors_rgb = M.reshape(ui.shape)

                # an (nx * ny, 2) array of interpolated u, v values

            plt.figure()
            with plt.style.context('seaborn-white'):
                # plt.rcParams['axes.facecolor'] = 'black'
                from skimage import feature
                edges2 = feature.canny(annotation[:228, :, 528-sl].T, sigma=0.1).astype(int)
                # mask=annotation[:228, :, slice].T>0
                mask = np.ma.masked_where(annotation[:228, :, 528-sl].T > 0, annotation[:228, :, 528-sl].T)
                xi = xi - np.min(xi)
                yi = yi - np.min(yi)
                plt.streamplot(xi, yi, -ui, -vi, density=15, arrowstyle='-', color='k', zorder=2)  # color=ui+vi#color=colors_rgb,
                # plt.quiver(X, Y, -U, -V,pivot='mid')  # [M_normed[:,0]>1]

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


graph = ggt.load(work_dir + '/' + control + '/' + 'data_graph_correcteduniverse.gt')#data_graph_corrected_Isocortex.gt')#/data_graph_correcteduniverse.gt')#
degrees = graph.vertex_degrees()
vf = np.logical_and(degrees > 1, degrees <= 4)
graph = graph.sub_graph(vertex_filter=vf)


with open(work_dir + '/' + control + '/sampledict' + control + '.pkl', 'rb') as fp:
    sampledict = pickle.load(fp)

pressure = np.asarray(sampledict['pressure'][0])
graph.add_vertex_property('pressure', pressure)




############################################################################
#  CloughTocher2DInterpolator NearestNDInterpolator
############################################################################
#
#
# radialVectorField(graph, 205,2500, 2600,  [0,2], 50, LinearNDInterpolator)
#
#
# ## plot artery graph
# import ClearMap.Analysis.Graphs.GraphGt_old as ggto
# graph = ggto.load(work_dir + '/' + control + '/' + 'data_graph_correcteduniverse.gt')#data_graph_corrected_Isocortex.gt')#/data_graph_correcteduniverse.gt')#
#
# arteries=graph.edge_property('artery')
#
# gs=graph.sub_graph(edge_filter=arteries)
#
# gs = gs.sub_slice((slice(1,320), slice(430,470), slice(1,228)),coordinates='coordinates_atlas');
# # gs = graph.sub_slice((slice(1,320), slice(267,277), slice(1,228)),coordinates='coordinates_atlas');
# # gs = graph.sub_slice((slice(1,320), slice(228,243), slice(1,228)),coordinates='coordinates_atlas');
# # gs = graph.sub_slice((slice(1, 320), slice(327, 337), slice(1, 228)), coordinates='coordinates_atlas');
# # vertex_colors = ano.convert_label(gs.vertex_annotation(), key='order', value='rgba');
#
#
# q1 = p3d.plot_graph_mesh(gs, n_tube_points=3,fov=0);# vertex_colors=vertex_colors,






controls=['2R','3R','5R', '8R']#['2R','3R','5R', '8R']
mutants=['1R','7R', '6R', '4R']
work_dir='/data_SSD_2to/191122Otof'


mutants=['2R','3R','5R', '1R']
controls=['7R','8R', '6R']
work_dir='/data_SSD_1to/otof6months'



# work_dir='/data_2to/p7'
# controls=['4']
work_dir='/data_SSD_2to/capillariesStream/'
from scipy import ndimage as ndi
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator, CloughTocher2DInterpolator
from skimage import feature
annotation=io.read('/home/sophie.skriabine/Pictures/Reslice of annotation_25_HeadLightOrientation_coronal.tif')
annotation=io.read('/home/sophie.skriabine/Documents/ClearMap_Ressources/annotation_25_HeadLightOrientation_sagital_rotated.tif')

edges2 = feature.canny(annotation[:228, :, 200].T, sigma=0.1)
plt.figure()
plt.imshow(edges2.astype(int), cmap='gray')