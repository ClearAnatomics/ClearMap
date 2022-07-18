# -*- coding: utf-8 -*-
"""
PlotUtils Module
================

Plotting routines for ClearMap based on matplotlib.

Note
----
    This module is using matplotlib.
"""
__author__ = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__ = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__ = 'https://idisco.info'
__download__ = 'https://www.github.com/ChristophKirst/ClearMap2'

import math
import numpy as np
import pandas as pd

import scipy.stats as st

import matplotlib as mpl  # analysis:ignore
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from mpl_toolkits.mplot3d import Axes3D  # analysis:ignore
import matplotlib.pyplot as plt


from ClearMap.Alignment import Annotation as annotation

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams["axes.axisbelow"] = False


###############################################################################
# ## Density and Countour plots
###############################################################################


def plot_density(points, plot_points=True, plot_contour=True, plot_contour_lines=10, n_bins=100,
                 color=None, cmap=plt.cm.gray, xlim=None, ylim=None, verbose=False):
    """Plot point distributions."""

    if xlim is None:
        xlim = (np.min(points[:, 0]), np.max(points[:, 0]))
    if ylim is None:
        ylim = (np.min(points[:, 1]), np.max(points[:, 1]))
    xmin, xmax = xlim
    ymin, ymax = ylim

    # kernel density estimates
    dx = float(xmax - xmin) / n_bins
    dy = float(ymax - ymin) / n_bins
    xx, yy = np.mgrid[xmin:xmax:dx, ymin:ymax:dy]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    kernel = st.gaussian_kde(points.T)
    if verbose:
        print('plot_density: kernel estimation done.')
    density = kernel(positions)
    density = np.reshape(density, xx.shape)
    if verbose:
        print('plot_density: density estimation done.')

    # figure setup
    ax = plt.gca()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # contour plot
    if plot_contour:
        cfset = ax.contourf(xx, yy, density, cmap=cmap)
    if plot_points:
        plt.scatter(points[:, 0], points[:, 1], c=color, cmap=cmap, s=1)
    if plot_contour_lines is not None:
        cset = ax.contour(xx, yy, density, plot_contour_lines, cmap=cmap)

    return kernel, (xx, yy, density)


def plot_curve(coordinates, **kwargs):
    """Plot a curve in 3d.

    Arguments
    ---------
    coordinates : nx3 array.
        The coordinates of the curve.
    kwargs
        Matplotlib parameter.

    Returns
    -------
    ax : ax
        3d axes object.
    """
    ax = plt.gca(projection='3d')
    x, y, z = coordinates.T
    ax.plot(x, y, z, **kwargs)
    return ax


def subplot_tiling(n, tiling=None):
    """Finds a good tiling to arrange subplots.

    Arguments
    ---------
    n : int
        Number of subplots.
    tiling : None, 'automatic, int or tuple
        The tiling to use. If None or 'automatic' calculate automatically.
        If number use this for the number of subplots along the horizontal axis.
        If tuple, (nx,ny) nx and ny can be numbes or None to indicate the number
        of sub-plots in each axis. Iif one of them is None, it will be
        determined automatically to fit the total number of plots.

    Returns
    -------
    tiling : tuple of int
        The subplot tiling.
    """
    if tiling is None:
        tiling = 'automatic'
    if tiling == "automatic":
        nx = math.floor(math.sqrt(n))
        ny = int(math.ceil(n / nx))
        nx = int(nx)
    else:
        if not isinstance(tiling, tuple):
            tiling = (tiling, None)
        if tiling[0] is None:
            ny = tiling[1]
            if ny is None:
                return subplot_tiling(n)
            nx = int(math.ceil(n / ny))
        if tiling[1] is None:
            nx = tiling[0]
            if nx is None:
                return subplot_tiling(n)
            ny = int(math.ceil(n / nx))

    return nx, ny


def plot_histogram(stats_df, aba_df, n_rows=20, sort_order=False, save_path=None,
                   fold_threshold=5, density_cutoff=0.05, fold_regions=False):
    stats_df = sanitize_df(stats_df)

    if fold_regions:
        stats_df = fold_dataframe(stats_df, aba_df, fold_threshold)

    # Compute density after fold
    stats_df['density'] = stats_df['Cell counts'] / stats_df['Structure volume']
    stats_df['color'] = [annotation.find(_id, key='id')['rgb'] for _id in stats_df['Structure ID']]
    stats_df = stats_df.sort_values('density', ascending=False)


    high_density_ids = np.unique(stats_df[stats_df['density'] >= density_cutoff]['Structure ID'])
    stats_df = stats_df[stats_df['Structure ID'].isin(high_density_ids)]

    fig, axes = plt.subplots(figsize=(10, 40), facecolor='#eaeaf2', ncols=2, sharey=True, gridspec_kw={'wspace': 0.2})
    # fig.set_constrained_layout_pads(w_pad=2)
    # fig.tight_layout()

    plt.xlabel('Cell density')

    titles = ['Left', 'Right']
    for i, hem_id in enumerate((0, 255)):  # Split and plot
        hem_df = stats_df[stats_df['Hemisphere'] == hem_id]
        if sort_order:
            hem_df = hem_df.sort_values('Structure order', ascending=False)
        ax = axes[i]
        ax.barh(hem_df['Structure name'], hem_df['density'], align='center', color=hem_df['color'], zorder=10)
        ax.set_title(titles[i], fontsize=18, pad=15)
        ax.tick_params(left=False)
        ax.tick_params(right=False)
        if i == 0:
            # ax.set(yticks=hem_df['Structure name'])
            pass
        else:
            # ax.set(yticks=hem_df['Structure name'], yticklabels=hem_df['Acronym'])
            ax.set(yticklabels=hem_df['Acronym'])
        ax.tick_params(axis='y', labelsize=15, zorder=0)

        if i == 0:
            ax.spines['left'].set_position(('axes', 1))
            # ax.spines['bottom'].set_position(('axes', 1))
            ax.tick_params(axis='y', direction='in')

        # Eliminate upper and right axes
        ax.spines['right'].set_color('none')

        ax.spines['top'].set_color('none')

    axes[0].invert_xaxis()

    # plt.gca().invert_yaxis()    # To show data from highest to lowest

    plt.subplots_adjust(wspace=0, top=0.85, bottom=0.1, left=0.18, right=0.95)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


def fold_dataframe(df, aba_df, fold_threshold):
    df['fold_parent'] = [get_parent_id(aba_df, _id, fold_threshold) for _id in df['Structure ID']]
    df = df.sort_values('Structure name')
    output = pd.DataFrame()
    for hem_id in (0, 255):
        grouped = df[df['Hemisphere'] == hem_id].groupby('fold_parent')
        half_df = pd.DataFrame()
        half_df['Cell counts'] = grouped['Cell counts'].sum()
        half_df['Structure volume'] = grouped['Structure volume'].sum()
        half_df['Average cell size'] = grouped['Average cell size'].mean()
        half_df['Structure ID'] = grouped['fold_parent'].first()

        ids = half_df['Structure ID'].values

        half_df['Structure order'] = [aba_df[aba_df['id'] == _id]['graph_order'].values[0] for _id in ids]
        half_df['Structure name'] = [aba_df[aba_df['id'] == _id]['name'].values[0] for _id in ids]
        half_df['Acronym'] = [aba_df[aba_df['id'] == _id]['acronym'].values[0] for _id in ids]
        half_df['Hemisphere'] = hem_id
        output = pd.concat((output, half_df))  # , ignore_index=True)
    return output


def sanitize_df(df):
    df = df[np.logical_and(df['Structure ID'] > 0, df['Structure ID'] < 2 ** 16)]
    df = df[df['Structure ID'] != 997]  # Not "brain"
    return df


def get_parent_structure(df, _id, level):
    current_structure = df[df['id'] == _id]
    current_level = current_structure['custom_level'].values[0]

    if current_level <= level:
        return current_structure
    else:
        parent_id = current_structure['parent_structure_id'].values[0]
        parent_structure = df[df['id'] == parent_id]
        try:
            parent_level = parent_structure['custom_level'].values[0]
        except IndexError:
            raise ValueError(f'Failed to get parent level for {parent_structure}, with ID: {_id} and level {level}')
        if parent_level <= level:
            return parent_structure
        else:
            return get_parent_structure(df, parent_id, level)


def get_parent_id(df, _id, level):
    return get_parent_structure(df, _id, level)['id'].values[0]


def get_parent_name(df, _id, level):
    return get_parent_structure(df, _id, level)['name'].values[0]


def volcano(df, group_names, save_path='', p_cutoff=0.05):
    mean_0 = df[f'mean_{group_names[0]}'].values
    mean_1 = df[f'mean_{group_names[1]}'].values
    ratio = np.log2(mean_1 / mean_0)

    p_values = df['p_value'].values
    ids = df['id'].values
    colors = np.array([annotation.find(_id, key='id')['rgb'] for _id in ids])
    for i, col in enumerate(colors):
        hsv = rgb_to_hsv(col)
        if p_values[i] > p_cutoff:
            hsv[1] *= 0.3  # desaturate
        colors[i] = hsv_to_rgb(hsv)
    # colors[p_values > p_cutoff] = (0.8, 0.8, 0.8)

    plt.scatter(ratio, -np.log10(p_values), marker='o', linestyle='None', color=colors, edgecolors=(0.9, 0.9, 0.9))
    plt.xlim((-5, 5))
    plt.hlines(-np.log10(p_cutoff), -5, 5, linestyles='dashed', color='grey')
    plt.xlabel('log2(fold change)')
    plt.ylabel('- log10(p value)')

    if not save_path:
        plt.show()
    else:
        plt.savefig(save_path)
