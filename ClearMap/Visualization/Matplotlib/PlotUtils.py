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

from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # analysis:ignore

from ClearMap.Alignment import Annotation as annotation
from ClearMap.Analysis.Statistics.data_frame_operations import sanitize_df, normalise_df_column_names

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


def handle_fig_fate(fig, show, save_path):
    if save_path:
        plt.savefig(save_path)
    else:
        if show:
            plt.show()
        else:
            return fig


def plot_sample_stats_histogram(stats_df, aba_df, sort_by_order=False, split_criterion='hemisphere',
                                metric_name='density', value_cutoff=0.05, fold_threshold=5, fold_regions=False,
                                save_path=None, show=True):
    stats_df = sanitize_df(stats_df)
    is_gp_stats = is_gp_stats_df(stats_df)
    if is_gp_stats:
        group_names = [k.replace('mean_', '') for k in stats_df.columns if k.startswith('mean_')]
        stats_df = gp_stats_df_to_counts(stats_df)  # Convert to same format for simplicity
    stats_df = normalise_df_column_names(stats_df)

    if fold_regions:
        stats_df = fold_sample_stats_dataframe(stats_df, aba_df, fold_threshold)

    stats_df['color'] = get_structure_colors(stats_df)

    # Compute metric, sort and filter after fold
    if metric_name == 'density':
        stats_df['density'] = stats_df['cell_counts'] / stats_df['structure_volume']
    stats_df = stats_df.sort_values(metric_name, ascending=False)
    high_values = np.unique(stats_df[stats_df[metric_name] >= value_cutoff]['structure_id'])
    stats_df = stats_df[stats_df['structure_id'].isin(high_values)]

    fig, axes = plt.subplots(figsize=(10, 40), facecolor='#eaeaf2', ncols=2, sharey=True, gridspec_kw={'wspace': 0.2})
    plt.xlabel(f'Cell {metric_name}')

    if split_criterion == 'hemisphere':
        titles = ['Left', 'Right']
    elif split_criterion == 'group_id':
        titles = group_names
    else:
        raise NotImplementedError(f'Split criterion {split_criterion} is not recognised yet, '
                                  f'supported values are "hemisphere" and "group_id"')
    for i, split_val in enumerate(np.sort(np.unique(stats_df[split_criterion]))):  # Split and plot
        sub_df = stats_df[stats_df[split_criterion] == split_val]
        plot_sub_df(sub_df, i, axes, titles, metric_name, sort_by_order)

    axes[0].invert_xaxis()  # FIXME: check that correct axis

    plt.subplots_adjust(wspace=0, top=0.85, bottom=0.1, left=0.18, right=0.95)

    return handle_fig_fate(fig, show, save_path)


def is_gp_stats_df(df):
    return 'p_value' in df.columns


def gp_stats_df_to_counts(df):
    """
    stack the df by group name
    The only counts are the average for the group
    A new column identifies the group. It is aimed to make it interchangeable with the DF of an individual sample

    Parameters
    ----------
    df pd.DataFrame

    Returns pd.DataFrame
    -------

    """
    group_names = [k.replace('mean_', '') for k in df.columns if k.startswith('mean_')]
    # gp1_name, gp2_name = group_names
    out = pd.DataFrame({
        'structure_id': np.tile(df['id'].values, 2),
        'structure_name': np.tile(df['name'].values, 2),
        'structure_volume': np.tile(df['volume'].values, 2),
        'hemisphere': np.tile(df['hemisphere'].values, 2),
        'group_id': np.hstack([np.repeat(group_names[i], df.shape[0]) for i in range(len(group_names))]),
        'cell_counts': np.hstack([df[df[f'mean_{group_names[i]}']][f'mean_{group_names[i]}'].values for i in range(len(group_names))])
    })
    return out


def plot_sub_df(sub_df, i, axes, titles, criterion_name, sort_by_order=False, struct_name_str='Structure name',
                struct_acronym_str='Acronym', struct_color_str='color', struct_order_str='Structure order'):
    if sort_by_order:
        sub_df = sub_df.sort_values(struct_order_str, ascending=False)
    ax = axes[i]
    ax.barh(sub_df[struct_name_str], sub_df[criterion_name], align='center', color=sub_df[struct_color_str], zorder=10)
    ax.set_title(titles[i], fontsize=18, pad=15)
    ax.tick_params(left=False, right=False)
    ax.tick_params(axis='y', labelsize=15, zorder=0)
    if i != 0:
        ax.set(yticklabels=sub_df[struct_acronym_str])
    else:
        ax.spines['left'].set_position(('axes', 1))
        # ax.spines['bottom'].set_position(('axes', 1))
        ax.tick_params(axis='y', direction='in')
    # Eliminate upper and right axes
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')


def get_structure_colors(stats_df, id_col_name='Structure ID'):
    return [annotation.find(_id, key='id')['rgb'] for _id in stats_df[id_col_name]]


def fold_sample_stats_dataframe(df, aba_df, fold_threshold):
    df['fold_parent'] = [get_parent_id(aba_df, _id, fold_threshold) for _id in df['structure_id']]  # TODO: extract
    df = df.sort_values('structure_name')
    output = pd.DataFrame()
    for hem_id in np.sort(np.unique(df['hemisphere'])):
        sub_df = df[df['hemisphere'] == hem_id]
        grouped = sub_df.groupby('fold_parent')

        half_df = pd.DataFrame()

        half_df['cell_counts'] = grouped['cell_counts'].sum()
        half_df['structure_volume'] = grouped['structure_volume'].sum()
        if 'average_cell_size' in sub_df.columns:
            half_df['average_cell_size'] = grouped['average_cell_size'].mean()
        half_df['structure_id'] = grouped['fold_parent'].first()

        ids = half_df['structure_id'].values

        half_df['structure_order'] = get_prop_from_aba_df(aba_df, ids, 'graph_order')
        half_df['structure_name'] = get_prop_from_aba_df(aba_df, ids, 'name')
        half_df['acronym'] = get_prop_from_aba_df(aba_df, ids, 'acronym')
        half_df['hemisphere'] = hem_id
        output = pd.concat((output, half_df))  # , ignore_index=True)
    return output


def get_prop_from_aba_df(aba_df, ids, prop_name):
    return [aba_df[aba_df['id'] == _id][prop_name].values[0] for _id in ids]


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


def plot_volcano(df, group_names, p_cutoff=0.05, show=False, save_path=''):
    mean_0 = df[f'mean_{group_names[0]}'].values
    mean_1 = df[f'mean_{group_names[1]}'].values
    ratio = np.log2(mean_1 / mean_0)

    p_values = df['p_value'].values
    ids = df['id'].values
    colors = np.array([annotation.find(_id, key='id')['rgb'] for _id in ids])
    for i, col in enumerate(colors):
        hsv = rgb_to_hsv(col)
        if p_values[i] > p_cutoff:
            hsv[1] *= 0.3  # desaturate non significant p values
        colors[i] = hsv_to_rgb(hsv)
    # colors[p_values > p_cutoff] = (0.8, 0.8, 0.8)
    fig = plt.figure()
    plt.scatter(ratio, -np.log10(p_values), marker='o', linestyle='None', color=colors, edgecolors=(0.9, 0.9, 0.9))
    plt.xlim((-5, 5))
    plt.hlines(-np.log10(p_cutoff), -5, 5, linestyles='dashed', color='grey')
    plt.xlabel('log2(fold change)')
    plt.ylabel('- log10(p value)')

    return handle_fig_fate(fig, show, save_path)
