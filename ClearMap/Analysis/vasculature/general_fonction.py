import matplotlib.pyplot as plt
import numpy as np
import ClearMap.Settings as settings
import os
import ClearMap.Alignment.Annotation as ano
import graph_tool.inference as gti
import pandas as pd
import graph_tool.topology as gtt
import ClearMap.Analysis.Graphs.GraphGt as ggt
import ClearMap.Alignment.Annotation as ano
import ClearMap.IO.IO as io
import ClearMap.Visualization.Plot3d as p3d
import graph_tool.inference as gti
import os
import math
import matplotlib.pyplot as plt
import ClearMap.Analysis.Graphs.GraphGt as ggt
import graph_tool.centrality as gtc
import numpy as np
import numexpr as ne
import graph_tool.topology as gtt
from sklearn import preprocessing
import seaborn as sns
from scipy.stats import ttest_ind
import math
pi=math.pi
import pandas as pd
import json
import numexpr as ne

try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle


with open('/home/sophie.skriabine/Projects/clearVessel_New/ClearMap/ClearMap/Resources/Atlas/annotation.json') as json_data:
    data_dict = json.load(json_data)['msg']
    print(data_dict)



def get_volume_region(region_leaves, atlas):
    val=0
    for l in region_leaves:
      val=val+np.sum(atlas==l[0])
    return val


def from_v_prop2_eprop(graph, property):
    if isinstance(property, ''.__class__):
        vprop = graph.vertex_property(property)#vprop = property#graph.vertex_property(property)
    else:
        vprop=property
    # e_prop=np.zeros(graph.n_edges)
    connectivity = graph.edge_connectivity()
    e_prop = np.logical_and(vprop[connectivity[:, 0]], vprop[connectivity[:, 1]])
    return e_prop

def from_e_prop2_vprop(graph, property):
    if isinstance(property, ''.__class__):
        e_prop=graph.edge_property(property)#e_prop = property#else:
    else:
        e_prop=property
    v_prop=np.zeros(graph.n_vertices)
    connectivity = graph.edge_connectivity()
    v_prop[connectivity[e_prop==1,0]]=1
    v_prop[connectivity[e_prop == 1,1]] = 1
    # graph.add_vertex_property(property, v_prop)
    return v_prop

def get_nb_radial_vessels(edge_color):
  radial=edge_color[:,2]/(edge_color[:,0]+edge_color[:,1]+edge_color[:,2])
  return(np.sum(radial>0.7))


def get_nb_parrallel_vessels(edge_color):
  planar=(edge_color[:,0]+edge_color[:,1])/(edge_color[:,2]+edge_color[:,0]+edge_color[:,1])
  print(planar.shape)
  return(np.sum(planar>0.7))




## orientation in the cortex relative to the surface

def getRadPlanOrienttaion(graph, ref_graph, local_normal=False, calc_art=False, verbose=False):

    if local_normal:
        rad_f = np.zeros(graph.n_edges)
        plan_f = np.zeros(graph.n_edges)
        lengths_f = graph.edge_property('length')
        norm_f = np.zeros(graph.n_edges)
        label = graph.vertex_annotation();
        for r in reg_list.keys():
            o = r#ano.find(r, key='id')['order']
            l = ano.find(r, key='order')['level']

            label_leveled = ano.convert_label(label, key='order', value='order', level=l)
            vf = label_leveled == o  # 54;
            ef=from_v_prop2_eprop(graph, vf)
            try:
                # sub_graph=graph.sub_graph(edge_filter=ef)
                sub_graph = graph.sub_graph(vertex_filter=vf)
                if verbose:
                    print(ano.find(o, key='order')['name'], sub_graph)
                r, p, n,l=getRadPlanOrienttaion(sub_graph, sub_graph, local_normal=False, calc_art=calc_art)

                rad_f[ef]=r
                plan_f[ef] = p
                norm_f[ef] = n
            except:
                if verbose:
                    print('problem', ano.find(o, key='order')['name'], ef.shape)

        rad=rad_f
        plan=plan_f
        lengths=lengths_f
        N=norm_f


    else:
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
            top_vertices=dist<=dist_min+1
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



def cart2sph(x,y,z, ceval=ne.evaluate):
    """ x, y, z :  ndarray coordinates
        ceval: backend to use:
              - eval :  pure Numpy
              - numexpr.evaluate:  Numexpr """
    if x.shape[0]!=0:
        r = ceval('sqrt(x**2+y**2+z**2)')#sqrt(x * x + y * y + z * z)
        theta = ceval('arccos(z/r)*180')/pi#acos(z / r) * 180 / pi  # to degrees
        phi = ceval('arctan2(y,x)*180')/pi#*180/3.4142
        # azimuth = ceval('arctan2(y,x)')
        # xy2 = ceval('x**2 + y**2')
        # elevation = ceval('arctan2(z, sqrt(xy2))')
        # r = ceval('sqrt(xy2 + z**2)')
        rmax=np.max(r)
    else:
        print('no orientation to compute')
        r=np.array([0])
        theta=np.array([0])
        phi=np.array([0])
        rmax=1
    return phi/180, theta/180, r/rmax#, theta/180, phi/180







#### specific for orientation




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



        artery = from_e_prop2_vprop(graph, 'artery')
        vein = from_e_prop2_vprop(graph, 'vein')
        radii = graph.vertex_property('radii')
        d2s = graph.vertex_property('distance_to_surface')




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




        from scipy.interpolate import LinearNDInterpolator,NearestNDInterpolator
        # from scipy.interpolate import Rbf
        NNDI_x = LinearNDInterpolator(art_edge_coordinates, art_graph_vector[:,0])
        NNDI_y = LinearNDInterpolator(art_edge_coordinates, art_graph_vector[:, 1])
        NNDI_z = LinearNDInterpolator(art_edge_coordinates, art_graph_vector[:, 2])



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



            angle2normal=np.array([angle_between(graph_vector[i],[local_flow_X[i], local_flow_Y[i], local_flow_Z[i]] ) for i in range(graph.n_edges)])*180/pi
            cosinus2normal=abs(np.cos(angle2normal*pi/180))
            angle = np.array([math.acos(cosinus2normal[i]) for i in range(cosinus2normal.shape[0])]) * 180 / pi
         

            angles.append(angle)
        angle=np.nanmedian(np.array(angles), axis=0)
        return angle, graph


