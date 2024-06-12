import math
pi = math.pi
import pickle

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.interpolate import LinearNDInterpolator, griddata
from skimage import feature

import graph_tool.all as ggt

import ClearMap.Alignment.Annotation as ano
import ClearMap.IO.IO as io
import ClearMap.Analysis.Graphs.GraphGt as ggt


def load_graph(work_dir, control):
    """
    Loads the graph from the given work directory and control.
    """
    try:
        graph = ggt.load(work_dir + '/' + control + '/' + 'data_graph_correcteduniverse.gt')
    except:
        graph = ggt.load(work_dir + '/' + control + '/' + str(control)+'_graph.gt')

    # Filter out vertices with degrees <= 1 or > 4
    degrees = graph.vertex_degrees()
    vf = np.logical_and(degrees > 1, degrees <= 4)
    graph = graph.sub_graph(vertex_filter=vf)

    return graph


def load_sampledict(work_dir, control):
    """
    Loads the sampledict for the given work directory and control.
    """
    try:
        with open(work_dir + '/' + control + '/sampledict' + control + '.pkl', 'rb') as fp:
            sampledict = pickle.load(fp)

        pressure = np.asarray(sampledict['pressure'][0])
        graph.add_vertex_property('pressure', pressure)
    except:
        print('no sample dict found for pressure and flow modelisation')


def set_vertex_properties(graph):
    """
    Sets the vertex properties for artery and vein.
    """
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



def get_filtered_graph(graph, label, mode='arteryvein'):
    artery = np.asarray(graph.vertex_property('artery'))
    vein = np.asarray(graph.vertex_property('vein'))
    radii = graph.edge_property('radii')
    d2s = graph.vertex_property('d2s')

    artery_vein = None
    if mode == 'arteryvein':
        artery_vein = np.asarray(np.logical_or(artery, vein))
    elif mode == 'bigvessels':
        artery_vein = graph.vertex_property('radii') > 4

    order, level = 1006, 3
    label_leveled = ano.convert_label(label, key='order', value='order', level=level)
    cerebellum = label_leveled == order
    cerebellum_art = np.logical_and(radii <= 6, np.logical_and(cerebellum, artery_vein))

    order, level = 463, 6
    label_leveled = ano.convert_label(label, key='order', value='order', level=level)
    hippocampus = label_leveled == order
    hippocampus_art = np.logical_and(radii <= 6, np.logical_and(hippocampus, artery_vein))

    artery_vein[hippocampus] = 0
    artery_vein[cerebellum] = 0
    artery_vein = np.asarray(np.logical_and(artery_vein, d2s >= 2))

    art_graph = graph.sub_graph(vertex_filter=artery_vein)
    return art_graph


def get_edge_vector(graph, control, normed=False, oriented=True, criteria='distance'):
    x = graph.vertex_coordinates()[:, 0]
    y = graph.vertex_coordinates()[:, 1]
    z = graph.vertex_coordinates()[:, 2]
    if criteria=='pressure':
        try:
            pressure=graph.vertex_property('pressure')
        except:
            f, v = computeFlowFranca(work_dir, graph, cont)
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


def get_interpolated_vector(control, graph, mode='arteryvein', average=True, work_dir=None):
    """
    Creates a grid of interpolated streamlines from the given graph.

    Parameters:
    -----------
    graph : Graph object
        The graph to create streamlines from.
    control : list of str
        The names of the control subjects to use in the interpolation.
    mode : str, optional
        The mode to use for artery and vein filtering. Default is 'arteryvein'.
    average : bool, optional
        Whether to return the average of the streamlines from the control subjects.
        Default is True.

    Returns:
    --------
    grid_vector : ndarray
        The grid of interpolated streamlines from the graph.
    """
    G = []
    for g in graph:
        art_graph = get_filtered_graph(g, label, mode)
        art_graph_vector, art_press_conn = get_edge_vector(art_graph, control)
        art_coordinates = art_graph.vertex_property('coordinates_atlas')
        art_edge_coordinates = np.array(
            [np.round((art_coordinates[art_press_conn[i, 0]] + art_coordinates[art_press_conn[i, 1]]) / 2) for i in
             range(art_press_conn.shape[0])])

        NNDI_x = LinearNDInterpolator(art_edge_coordinates, art_graph_vector[:, 0])
        NNDI_y = LinearNDInterpolator(art_edge_coordinates, art_graph_vector[:, 1])
        NNDI_z = LinearNDInterpolator(art_edge_coordinates, art_graph_vector[:, 2])

        grid_x = np.linspace(0, np.max(art_edge_coordinates[:, 0]), 100)
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
         
        return grid, grid_vector




def plot_vector_field(grid, grid_vector, sl, sxe='sagital'):
    if sxe=='sagital':
        grid_coordinates_2plot = grid[grid[:, 2] > sl]
        grid_vector_2plot = grid_vector[grid[:, 2] > sl]
        grid_vector_2plot = grid_vector_2plot[grid_coordinates_2plot[:, 2] < sl + 2.5]
        grid_coordinates_2plot = grid_coordinates_2plot[grid_coordinates_2plot[:, 2] < sl + 2.5]
        center=(np.median(grid_coordinates_2plot[:,0]), np.max(grid_coordinates_2plot[:,1])-30)
        grid_coordinates_2plot = np.array([grid_coordinates_2plot[i, [0, 1]] - center for i in range(grid_coordinates_2plot.shape[0])])
    elif sxe=='coronal':
        grid_coordinates_2plot = grid[grid[:, 1] > sl]
        grid_vector_2plot = grid_vector[grid[:, 1] > sl]
        grid_vector_2plot = grid_vector_2plot[grid_coordinates_2plot[:, 1] < sl + 5] 
        grid_coordinates_2plot = grid_coordinates_2plot[grid_coordinates_2plot[:, 1] < sl + 5] 
        center = (np.median(grid_coordinates_2plot[:, 0]), np.max(grid_coordinates_2plot[:, 2]) - 30) 
        grid_coordinates_2plot = np.array([grid_coordinates_2plot[i, [0, 2]] - center for i in range(grid_coordinates_2plot.shape[0])])
    
    X = grid_coordinates_2plot[:, 1]
    Y = grid_coordinates_2plot[:, 0]
    grid_vector_2plot = preprocessing.normalize(np.nan_to_num(grid_vector_2plot[:, [0, 1]]))
    U = grid_vector_2plot[:, 1]
    V = grid_vector_2plot[:, 0]
    xi = np.linspace(X.min(), X.max(), 100)
    yi = np.linspace(Y.min(), Y.max(), 100)
    pts = np.vstack((X, Y)).T
    vals = np.vstack((U, V)).T
    ipts = np.vstack(a.ravel() for a in np.meshgrid(xi, yi)).T
    ivals = griddata(pts, vals, ipts, method='cubic')
    ui, vi = preprocessing.normalize(ivals).T
    ui.shape = vi.shape = (100, 100)
    
        
    return X, Y, U, V, ui, vi


def main():
    global work_dir, mode, pi, graph, label
    work_dir = ''
    control = ''
    slice_no = 205
    zmin = 2500
    zmax = 2600
    dim = [0, 2]
    sampling = 50
    function = LinearNDInterpolator
    mode = 'arteryvein'
    pi = math.pi
    G = []
    mode = 'bigvessels'
    average = True
    # Load the graph
    graph = load_graph(work_dir, control)
    # Load the sampledict
    load_sampledict(work_dir, control)
    # Set the vertex properties for artery and vein
    set_vertex_properties(graph)
    # Get the radii and distance_to_surface vertex properties
    radii = graph.vertex_property('radii')
    d2s = graph.vertex_property('distance_to_surface')
    # Get graph labels
    label = graph.vertex_annotation()
    # Get artery graph
    grid, grid_vector = get_interpolated_vector(control, graph, mode='arteryvein', average=True, work_dir=None)
    slices = [400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600, 3800,
              4000, 4500]
    slices = [329]  # [210, 165]
    sxe = 'coronal'  # 'coronal' # 'sagital'
    for sl in slices:
        if sxe == 'sagital':
            annotation = io.read(
                '/home/sophie.skriabine/Documents/ClearMap_Ressources/annotation_25_HeadLightOrientation_sagital_rotated.tif')
        elif sxe == 'coronal':
            import tifffile
            annotation = tifffile.imread(
                '/home/sophie.skriabine/Pictures/Reslice_of_annotation_25_HeadLightOrientation_coronal.tif')
            annotation = np.swapaxes(annotation, 0, 2)
            annotation = np.flip(annotation, 2)

        X, Y, U, V, ui, vi = plot_vector_field(grid, grid_vector, sl, sxe=sxe)

        # Plot streamline on specfy slice
        plt.figure()
        with plt.style.context('seaborn-white'):

            edges2 = feature.canny(annotation[:228, :, 528 - sl].T, sigma=0.1).astype(int)
            mask = np.ma.masked_where(annotation[:228, :, 528 - sl].T > 0, annotation[:228, :, 528 - sl].T)
            xi = xi - np.min(xi)
            yi = yi - np.min(yi)
            plt.streamplot(xi, yi, -ui, -vi, density=15, arrowstyle='-', color='k',
                           zorder=2)  # color=ui+vi#color=colors_rgb,

            plt.gca().invert_yaxis()
            plt.gca().invert_xaxis()
            plt.imshow(mask, cmap='Greys', zorder=10)
            plt.imshow(edges2, cmap='Greys', zorder=1)  # jet#copper#
            edges2 = np.logical_not(edges2)

            masked_data = np.ma.masked_where(edges2 > 0, edges2)
            plt.imshow(masked_data, cmap='cool_r', zorder=11)  # jet#copper#

            plt.title(str(sl))


if __name__ == '__main__':
    main()
