import ClearMap.Analysis.Measurements.Voxelization as vox
import ClearMap.IO.IO as io
import numpy as np
import ClearMap.Analysis.Graphs.GraphGt_old as ggto
import ClearMap.Analysis.Graphs.GraphGt as ggt
import pickle
from sklearn import preprocessing
import matplotlib.pyplot as plt

# from ClearMap.OtoferliinGraphs import from_e_prop2_vprop, getRadPlanOrienttaion
template_shape=(320,528,228)
vox_shape_m=(320,528,228, 1)
vox_shape_c=(320,528,228, 1)
vox_control=np.zeros(vox_shape_c)
vox_mutant=np.zeros(vox_shape_m)
condition='all_brain'
work_dir='/data_2to/p0/new'
radius=10
limit_angle=40
controls=['0c', '0d']#['2', '3']
# control='2_1'

for control in controls:
    graph=ggt.load(work_dir+'/'+control+'/'+control+'_graph.gt')
    graph=remove_surface(graph, 2)
    graph=graph.largest_component()
    print(work_dir + '/' + control + '/' + control+'_graph_correcteduniverse.gt')
    graph.save(work_dir + '/' + control + '/' + control+'_graph_correcteduniverse.gt')


    with open(work_dir + '/' + control + '/sampledict' + control + '.pkl', 'rb') as fp:
        sampledict = pickle.load(fp)

    # graph = ggt.load('/data_2to/p0/4/data_graph.gt')
    graph = ggt.load(work_dir + '/' + control + '/' + 'data_graph_correcteduniverse.gt')
    pressure = np.asarray(sampledict['pressure'][0])
    graph.add_vertex_property('pressure', pressure)

### distance
# graph = ggt.load('/data_2to/p0/4/data_graph.gt')
# autofluo=io.read('/data_2to/p0/4_resampled_autofluorescence.tif')
# distance=io.read('/data_2to/p0/4_distances.tif')
autofluo=io.read(work_dir + '/' + control +str(control)+'_resampled_autofluorescence.tif')
distance=io.read(work_dir + '/' + control +str(control)+'_distances.tif')
distance=io.read('/data_2to/p3/test_file/3b_resampled_autofluorescence_distance_32.tif')

coordinates = graph.vertex_property('coordinates_atlas') # coordinates_atlas
coordinates[:, :2] = coordinates[:, :2] * 1.625 / 25
coordinates[:, 2] = coordinates[:, 2] * 2 / 25

np.max(coordinates[:, 0])
np.max(coordinates[:, 1])
np.max(coordinates[:, 2])

# coordinates[:, 2] = coordinates[:, 2]-(np.max(coordinates[:, 2])-distance.shape[2]-1)
coordinates[:, 1] = coordinates[:, 1]-(np.max(coordinates[:, 1])-distance.shape[1]+1)
coordinates[:, 0] = coordinates[:, 0]-(np.max(coordinates[:, 0])-distance.shape[0]+1)

distance2surface=[distance[int(coordinates[i, 0]),int(coordinates[i, 1]),int(coordinates[i, 2])] for i in range(coordinates.shape[0])]
graph.add_vertex_property('distance_to_surface', np.array(distance2surface))
graph.add_vertex_property('coordinates_atlas', coordinates)
graph.add_vertex_property('coordinates', coordinates)
###     BP voxelization

# graph = ggt.load('/data_2to/p0/4_graph2.gt')
# graph = ggt.load('/data_2to/p7/4/data_graph_correcteduniverse.gt')
for i,control in enumerate(controls):
    print(control)
    coordinates = graph.vertex_property('coordinates_atlas') # coordinates_atlas
    coordinates[:, :2] = coordinates[:, :2] #* 1.625 / 25
    coordinates[:, 2] = coordinates[:, 2] #* 2 / 25

    v = vox.voxelize(coordinates[:, :3], shape=template_shape, weights=None, radius=(radius, radius, radius), method='sphere');
    vox_control[:,:,:,i]=v

vox_control_avg=np.mean(vox_control[:, :, :, :], axis=3)
io.write(work_dir + '/' str(control)+'/'+'voxP0_4'+str(radius)+'.tif', vox_control_avg.astype('float32'))

###     deg1 voxelization
for i,control in enumerate(controls):
    print(control)
    graph=ggt.load(work_dir + '/' + control + '/' + str(control)+'_graph_correcteduniverse.gt')
    coordinates = graph.vertex_property('coordinates_atlas') # coordinates_atlas
    # coordinates[:, :2] = coordinates[:, :2] * 1.625 / 25
    # coordinates[:, 2] = coordinates[:, 2] * 2 / 25

    deg0=graph.vertex_degrees()==1
    v = vox.voxelize(coordinates[deg0, :3], shape=template_shape, weights=None, radius=(radius, radius, radius),method='sphere');
    vox_control[:,:,:,i]=v

vox_control_avg=np.mean(vox_control[:, :, :, :], axis=3)
io.write(work_dir + '/' +'vox_avg_deg1'+str(radius)+'.tif', vox_control_avg.astype('float32'))


###     orientation voxelization
work_dir='/data_2to/p0'
controls=['2', '3']
mode='bigvessels'
average=True

for control in controls:
    graph=ggt.load(work_dir+'/'+control+'/data_graph.gt')
    # f, v=computeFlowFranca(work_dir, graph,control)
    #
    # work_dir='/data_2to/p0'
    # try:
    #     with open(work_dir + '/' + control + '/sampledict' + control + '.pkl', 'rb') as fp:
    #         sampledict = pickle.load(fp)
    #
    #     f = np.asarray(sampledict['flow'][0])
    #     v = np.asarray(sampledict['v'][0])
    #     p = np.asarray(sampledict['pressure'][0])
    #     graph.add_edge_property('flow', f)
    #     graph.add_edge_property('veloc', v)
    #     graph.add_vertex_property('pressure', p)
    # except:
    #     f, v=computeFlowFranca(work_dir, graph,control)
    #
    #     with open(work_dir + '/' + control + '/sampledict' + control + '.pkl', 'rb') as fp:
    #         sampledict = pickle.load(fp)
    #
    #         f = np.asarray(sampledict['flow'][0])
    #         v = np.asarray(sampledict['v'][0])
    #         p = np.asarray(sampledict['pressure'][0])
    #         graph.add_edge_property('flow', f)
    #         graph.add_edge_property('veloc', v)
    #         graph.add_vertex_property('pressure', p)
    #
    # print('done')

    angle,graph  = GeneralizedRadPlanorientation(graph, control, controls, mode=mode, average=average)#, corrected=False, distance=True)
    np.save(work_dir +'/'+  control + '/ORI_' + condition +  '_'+control+'.npy', angle)

    angle=np.load(work_dir +'/'+ control + '/ORI_' + condition +   '_'+control+'.npy')
    rad = angle <= limit_angle  # 40
    planarity = angle >= (90 - limit_angle)  # 60
    connectivity = graph.edge_connectivity()

    coordinates = graph.vertex_property('coordinates_atlas') #
    # coordinates[:, :2] = coordinates[:, :2] * 1.625 / 25
    # coordinates[:, 2] = coordinates[:, 2] * 2 / 25

    # coordinates = graph.vertex_property('coordinates_atlas')  # *1.625/25
    edges_centers = np.array(
        [(coordinates[connectivity[i, 0]] + coordinates[connectivity[i, 1]]) / 2 for i in range(connectivity.shape[0])])
    # art_coordinates = art_tree.vertex_property('coordinates_atlas')  # *1.625/25
    # print('artBP')#artBP
    # v = vox.voxelize(art_coordinates[:, :3], shape=template_shape, weights=None, radius=(radius, radius, radius), method='sphere');
    # vox_art_control[:, :, :, i] = v
    print('rad')
    v = vox.voxelize(edges_centers[rad, :3], shape=template_shape, weights=None, radius=(radius, radius, radius), method='sphere');
    w = vox.voxelize(edges_centers[:, :3], shape=template_shape, weights=None, radius=(radius, radius,radius), method='sphere');
    ori=v.array/ w.array
    io.write(work_dir + '/' +'vox_ori_fi_'+control+'_control_rad_avg_'+str(radius)+str(average)+'.tif', ori.astype('float32'))



### streamlines
import math
pi = math.pi
G = []
mode='bigvessels'
average=True
for control in controls:#[np.array([0,2,3,4])]:#mutants[np.array([0,2,3,4])]:
    try:
        graph = ggt.load(work_dir + '/' + control + '/' + 'data_graph_correcteduniverse.gt')
    except:
        graph = ggt.load(work_dir + '/' + control + '/' + str(control)+'_graph_correcteduniverse.gt')

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

    artery = from_e_prop2_vprop(graph, 'artery')
    vein = from_e_prop2_vprop(graph, 'vein')
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
    radii = graph.vertex_property('radii')


    artery_vein = np.logical_or(artery, vein)
    artery_vein = np.logical_or(artery_vein, from_e_prop2_vprop(graph, graph.edge_property('radii')>4))

    radii = graph.vertex_property('radii')

    artery_vein = np.logical_and(artery_vein, radii >= 6)

    art_graph = graph.sub_graph(vertex_filter=artery_vein)  # np.logical_or(artery, vein))

    art_graph_vector, art_press_conn = getEdgeVector(art_graph, control, criteria='distance')

    graph_vector, press_conn = getEdgeVector(graph, control,criteria='distance')

    art_coordinates = art_graph.vertex_property('coordinates_atlas')  # art_graph.vertex_coordinates()
    art_edge_coordinates = np.array(
        [np.round((art_coordinates[art_press_conn[i, 0]] + art_coordinates[art_press_conn[i, 1]]) / 2) for i in
         range(art_press_conn.shape[0])])

    # coordinates = graph.vertex_coordinates()
    # edge_coordinates = np.array(
    #     [np.round((coordinates[press_conn[i, 0]] + coordinates[press_conn[i, 1]]) / 2) for i in
    #      range(press_conn.shape[0])])

    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator,RegularGridInterpolator

    # from scipy.interpolate import Rbf
    NNDI_x = LinearNDInterpolator(art_edge_coordinates, art_graph_vector[:, 0])
    NNDI_y = LinearNDInterpolator(art_edge_coordinates, art_graph_vector[:, 1])
    NNDI_z = LinearNDInterpolator(art_edge_coordinates, art_graph_vector[:, 2])


    grid_x = np.linspace(0, np.max(art_edge_coordinates[:, 0]), 100)  # 100
    grid_y = np.linspace(0, np.max(art_edge_coordinates[:, 1]), 100)
    grid_z = np.linspace(0, np.max(art_edge_coordinates[:, 2]), 100)

    grid = np.array(np.meshgrid(grid_x, grid_y, grid_z)).reshape((3, 1000000)).T

    grid_flow_X = NNDI_x(grid)
    grid_flow_Y = NNDI_y(grid)
    grid_flow_Z = NNDI_z(grid)

    grid_vector = np.stack([grid_flow_X, grid_flow_Y, grid_flow_Z]).T
    G.append(grid_vector)

    # except:
    #     print('problem interpolation ...')
Gmean = np.nanmean(np.array(G), axis=0)
grid_vector = Gmean


slices = [1200, 1600, 2000, 2400, 2800, 3200]

# slice=3400
# slices_coronal=[100 ,150, 200, 250, 300,350,400, 450]#[210, 165]
slices_sagital = [1200, 1600, 2000, 2400, 2800, 3200]
slices_sagital = [1600]
slices_sagital=[200]

slices_sagital=[50,150,250,300, 350, 400]#
sxe = 'coronal'  # 'coronal'#'sagital'
for sl in slices_sagital:
    if sxe == 'sagital':

        grid_coordinates_2plot = grid[grid[:, 2] > sl]
        grid_vector_2plot = grid_vector[grid[:, 2] > sl]
        # art_graph_vector_spher_2plot = spher_ori[art_edge_coordinates[:, 1] > sl]
        grid_vector_2plot = grid_vector_2plot[grid_coordinates_2plot[:, 2] < sl + 25]  # 100
        # art_graph_vector_spher_2plot= art_graph_vector_spher_2plot[art_edge_coordinates_2plot[:, 1] < sl+100]
        grid_coordinates_2plot = grid_coordinates_2plot[grid_coordinates_2plot[:, 2] < sl + 25]  # 100


        center = (
        np.median(grid_coordinates_2plot[:, 0]), np.max(grid_coordinates_2plot[:, 1]) - 300)  # -500#(3500, 2000)
        print(center)
        grid_coordinates_2plot = np.array(
            [grid_coordinates_2plot[i, [0, 1]] - center for i in range(grid_coordinates_2plot.shape[0])])
        grid_vector_spher_2plot = [np.dot(preprocessing.normalize(np.nan_to_num(grid_coordinates_2plot), norm='l2')[i],
                                          preprocessing.normalize(np.nan_to_num(grid_vector_2plot[:, [0, 1]]))[i]) for i
                                   in range(grid_coordinates_2plot.shape[0])]
        X = grid_coordinates_2plot[:, 1]  # -center[0]
        Y = grid_coordinates_2plot[:, 0]  # -center[1]
        grid_vector_2plot = preprocessing.normalize(np.nan_to_num(grid_vector_2plot[:, [0, 1]]))
        U = grid_vector_2plot[:, 1]
        V = grid_vector_2plot[:, 0]

        M = abs(np.array(grid_vector_spher_2plot))



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
        colors_rgb = M.reshape(ui.shape)

        # an (nx * ny, 2) array of interpolated u, v values

        plt.figure()
        with plt.style.context('seaborn-white'):
            # plt.rcParams['axes.facecolor'] = 'black'
            from skimage import feature

            mask = np.ma.masked_where(autofluo[:, :, sl] > 1000, autofluo[:, :, sl])

            # xi = xi - np.min(xi)
            # yi = yi - np.min(yi)
            plt.streamplot(xi, yi, ui, vi, density=15, arrowstyle='-', color='k',
                           zorder=2)  # color=ui+vi#color=colors_rgb,
            # plt.quiver(X, Y, -U, -V,pivot='mid')  # [M_normed[:,0]>1]

            plt.gca().invert_yaxis()
            plt.gca().invert_xaxis()
            # plt.imshow(mask, cmap='Greys', zorder=10)
            plt.imshow(np.flip(np.flip(mask, 0), 1), cmap='Greys', zorder=10)
            # plt.axis('off')
            plt.title(str(sl))

    elif sxe == 'coronal':  # 'sagital':
        # autofluo = np.flip(io.read( work_dir + '/' + control + '/' + '3_resampled_autofluorescence_coronal.tif'), 1)
        grid_coordinates_2plot = grid[grid[:, 1] > sl]
        grid_vector_2plot = grid_vector[grid[:, 1] > sl]
        # art_graph_vector_spher_2plot = spher_ori[art_edge_coordinates[:, 1] > slice]
        grid_vector_2plot = grid_vector_2plot[grid_coordinates_2plot[:, 1] < sl + 50]  # 100
        # art_graph_vector_spher_2plot= art_graph_vector_spher_2plot[art_edge_coordinates_2plot[:, 1] < slice+100]
        grid_coordinates_2plot = grid_coordinates_2plot[grid_coordinates_2plot[:, 1] < sl + 50]  # 100

        center = (
            np.median(grid_coordinates_2plot[:, 0]), np.max(grid_coordinates_2plot[:, 2]) - 300)  # -500#(3500, 2000)
        print(center)
        grid_coordinates_2plot = np.array(
            [grid_coordinates_2plot[i, [0, 2]] - center for i in range(grid_coordinates_2plot.shape[0])])
        # grid_vector_spher_2plot = [np.dot(preprocessing.normalize(np.nan_to_num(grid_coordinates_2plot), norm='l2')[i],
        #                                   preprocessing.normalize(np.nan_to_num(grid_vector_2plot[:, [0, 2]]))[i]) for i
        #                            in range(grid_coordinates_2plot.shape[0])]
        X = grid_coordinates_2plot[:, 0]  # -center[0]
        Y = grid_coordinates_2plot[:, 1]  # -center[1]
        grid_vector_2plot = preprocessing.normalize(np.nan_to_num(grid_vector_2plot[:, [0, 2]]))
        U = grid_vector_2plot[:, 0]
        V = grid_vector_2plot[:, 1]

        # M = abs(np.array(grid_vector_spher_2plot))

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
        with plt.style.context('seaborn-white'):
            # plt.rcParams['axes.facecolor'] = 'black'
            from skimage import feature

            # mask = np.ma.masked_where(autofluo[:, :, sl]> 1000, autofluo[:, :, sl])

            xi = xi - np.min(xi)
            yi = yi - np.min(yi)
            plt.streamplot(xi, yi,-ui, -vi, density=15, arrowstyle='-', color='k',
                           zorder=2)  # color=ui+vi#color=colors_rgb,
            # plt.quiver(X, Y, -U, -V,pivot='mid')  # [M_normed[:,0]>1]

            # plt.gca().invert_yaxis()
            plt.gca().invert_xaxis()
            # plt.imshow(np.flip(mask, 1), cmap='Greys', zorder=10)
            # plt.imshow(np.flip(np.flip(mask, 0),1), cmap='Greys', zorder=10)
            # plt.axis('off')
            plt.title(str(sl))
plt.figure()
plt.imshow(np.flip(np.flip(mask, 0),1), cmap='Greys', zorder=10)
# plt.imshow(mask, cmap='Greys', zorder=10)
plt.imshow(np.flip(mask, 0), cmap='Greys', zorder=10)


from skimage import io

sl=262
im = autofluo[:, :, sl]> 1000
im=np.logical_not(im)
io.imsave('/data_2to/p14/3/streamlines/autofluo'+str(sl)+'.png', im.astype(int))

work_dir='/data_SSD_2to/191122Otof'
control='5R'
graph = ggt.load(work_dir + '/' + control + '/' + str(control)+'_graph.gt')
graph=remove_surface(graph, 2)
graph=graph.largest_component()
print(average)
angle,graph = GeneralizedRadPlanorientation(graph, g, 4.5, controls, mode=mode, average=average, dvlpmt=True)
rad = angle <= limit_angle  # 40
graph.add_edge_property('rad', rad)
brad=np.logical_and(graph.edge_property('radii')>6, graph.edge_property('radii')<=8)
graph.add_edge_property('brad', brad)
gs = graph.sub_slice((slice(1,5000), slice(2000,2100), slice(1,5000)));
# edge_artery_label = gs.edge_property('brad')
# edge_filter=edge_artery_label
# gsrt = gs.sub_graph(edge_filter=edge_filter)
gsrt=gs
edge_artery_label = gsrt.edge_property('rad')
vertex_colors =np.zeros((gsrt.n_vertices, 4))
vertex_colors[:, -1]=1
vertex_colors[:, 1]=0.8
#
connectivity = gsrt.edge_connectivity();
edge_colors = (vertex_colors[connectivity[:,0]] + vertex_colors[connectivity[:,1]])/2.0;
edge_colors[edge_artery_label>0] = [0.8,0.0,0.0,1.0]
#edge_colors[edge_vein_label>0] = [0.0,0.0,0.8,1.0]
print('plotting...')
p = p3d.plot_graph_mesh(gsrt, edge_colors=edge_colors, n_tube_points=5);




coordinates = graph.edge_geometry_property('coordinates_atlas')
indices = graph.edge_property('edge_geometry_indices')

bp = graph.n_vertices

L = 0
for i, ind in enumerate(indices):
    diff = np.diff(coordinates[ind[0]:ind[1]], axis=0)
    L = L + np.sum(np.linalg.norm(diff, axis=1))
Lmm = L * 0.000025
print(Lmm)  # m


from skimage import feature
import tifffile
annotation=tifffile.imread('/home/sophie.skriabine/Projects/ClearMap3/ClearMap/ClearMap/Resources/Atlas/annotationADMBA_thresholded.tif')
annotation=np.swapaxes(annotation, 1, 2)
# annotation=np.flip(annotation, 2)
annotation=np.flip(annotation, 1)

sl=150
work_dir='/data_2to/dvlpIMG/'
edges2 = feature.canny(annotation[:, :, sl].T, sigma=0.1)
plt.figure()
plt.imshow(edges2.astype(int), cmap='gray')

from skimage import io


im = annotation[:200, :, sl].T> 10
im=np.logical_not(im)
plt.figure()
plt.imshow(im.astype(int), cmap='gray')
io.imsave(work_dir+'autofluo'+str(sl)+'.png', im.astype(int))
