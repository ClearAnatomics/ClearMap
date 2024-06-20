import os

import numpy as np
import matplotlib.pyplot as plt

from skimage import feature
from sklearn import preprocessing

from scipy.interpolate import griddata
import tifffile

from ClearMap.Settings import atlas_folder
import ClearMap.IO.IO as clearmap_io

from ClearMap.Analysis.vasculature.vasc_graph_utils import set_artery_vein_if_missing, filter_graph_degrees
from ClearMap.Analysis.vasculature.general_functions import (graph_to_edge_coords_and_vectors, get_artery_vein_edge_coords_and_vectors,
                                                             compute_blood_flow, load_graph)
from ClearMap.Analysis.vasculature.geometry_utils import compute_grid, interpolate_vectors


def create_interpolated_streamline_grid_and_vectors(graph, work_dir, mode='arteryvein', average=True):
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
    grid, grid_vector : ndarray, ndarray
        The grid of interpolated streamlines and the average of the streamlines from the control subjects.
    """
    G = []  # @Sophie: rename
    label = graph.vertex_annotation()
    for g in graph:
        # FIXME: shouldn't criterion be pressure ?
        arteries_edge_coordinates, arteries_graph_vector = get_artery_vein_edge_coords_and_vectors(g, mode, artery=None, vein=None,
                                                                                                   min_radius=4, label=label,  # FIXME: magic numbers
                                                                                                   distance_to_surface=graph.vertex_property('d2s'),
                                                                                                   coordinates_name='coordinates_atlas',
                                                                                                   orientation_criterion='distance')

        grid = compute_grid(arteries_edge_coordinates)

        grid_vector = interpolate_vectors(arteries_graph_vector, arteries_edge_coordinates, grid)
        G.append(grid_vector)

        if average:
            grid_vector = np.nanmean(np.array(G), axis=0)
            np.save(os.path.join(work_dir, f'streamline_grid_avg_controls{mode}.npy'), grid_vector)

        if not average:
            print('No average')

        return grid, grid_vector


def plot_vector_field(grid, grid_vector, slice_, orientation_name='sagittal'):
    axes = list(range(3))
    slice_values = {0: 2.5, 1: 5, 2: 2.5}  # Define the dictionary mapping each axis to a slice value
    operations = ('horizontal', 'coronal', 'sagittal')

    if orientation_name not in operations:
        raise ValueError(f'Unknown sxe {orientation_name}, expected one of {operations}')

    operation_to_main_axis = {name: i for i, name in enumerate(operations)}
    main_axis = operation_to_main_axis[orientation_name]
    axes.remove(main_axis)

    def filter_func(x, axes, slice_):
        return x[x[:, axes[1]] > slice_]

    def center_func(x, axes):
        return np.median(x[:, axes[0]]), np.max(x[:, axes[1]]) - 30

    grid_axes_2plot = filter_func(grid, axes, slice_values[axes[0]])
    grid_axes_2plot = grid_axes_2plot[:, axes] - center_func(grid_axes_2plot, axes)

    grid_vector_2plot = grid_vector[filter_func(grid[:, axes[0]], axes, slice_values[axes[0]])]
    grid_vector_2plot = preprocessing.normalize(np.nan_to_num(grid_vector_2plot[:, [0, 1]]))

    Y, X = np.hsplit(grid_axes_2plot, 2)
    V, U = np.hsplit(grid_vector_2plot, 2)

    xi = np.linspace(X.min(), X.max(), 100)
    yi = np.linspace(Y.min(), Y.max(), 100)
    pts = np.vstack((X, Y)).T
    vals = np.vstack((U, V)).T
    ipts = np.vstack(a.ravel() for a in np.meshgrid(xi, yi)).T  # explain
    ivals = griddata(pts, vals, ipts, method='cubic')
    ui, vi = preprocessing.normalize(ivals).T
    ui.shape = vi.shape = (100, 100)

    return X, Y, U, V, ui, vi


def main():
    work_dir = ''
    control = ''
    slice_no = 205
    zmin, zmax = (2500, 2600)
    dim = [0, 2]
    sampling = 50

    mode = 'arteryvein'  #  'bigvessels'
    average = True
    graph = load_graph(work_dir, control)
    graph = filter_graph_degrees(graph)
    set_artery_vein_if_missing(graph)
    graph = compute_blood_flow(graph, work_dir, control)

    grid, grid_vector = create_interpolated_streamline_grid_and_vectors(graph, work_dir, control, mode=mode, average=True)
    slices = [329]  # [210, 165]
    sxe = 'coronal'  # 'coronal' # 'sagittal'
    for sl in slices:
        if sxe == 'sagittal':
            annotation = clearmap_io.read(os.path.join(atlas_folder, 'annotation_25_HeadLightOrientation_sagital_rotated.tif'))  # FIXME: add this file to atlas folder
        elif sxe == 'coronal':
            annotation = tifffile.imread(os.path.join(atlas_folder, 'Reslice_of_annotation_25_HeadLightOrientation_coronal.tif'))  # FIXME: add this file to atlas folder
            annotation = np.swapaxes(annotation, 0, 2)
            annotation = np.flip(annotation, 2)

        X, Y, U, V, ui, vi = plot_vector_field(grid, grid_vector, sl, orientation_name=sxe)

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
