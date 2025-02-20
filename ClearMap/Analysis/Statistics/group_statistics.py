# -*- coding: utf-8 -*-
"""
Statistics
==========

Create some statistics to test significant changes in voxelized and labeled 
data.
"""
__author__ = 'Christoph Kirst <christoph.kirst.ck@gmail.com>, Sophie Skriabine <sophie.skriabine@icm-institute.org>, Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__ = 'https://idisco.info'
__download__ = 'https://github.com/ClearAnatomics/ClearMap'

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import ClearMap.Analysis.Statistics.StatisticalTests as clearmap_stat_tests
from ClearMap.Alignment import Annotation as annotation
from ClearMap.Alignment.utils import get_all_structs
from ClearMap.Analysis.Statistics import MultipleComparisonCorrection as clearmap_FDR
from ClearMap.IO import IO as clearmap_io
from ClearMap.Utils.exceptions import GroupStatsError
from ClearMap.Utils.path_utils import is_density_file, find_density_file, find_cells_df, dir_to_sample_id
from ClearMap.Utils.utilities import make_abs
from ClearMap.config.atlas import ATLAS_NAMES_MAP
from ClearMap.processors.sample_preparation import init_sample_manager_and_processors, SampleManager

colors = {  # REFACTOR: move to visualisation module
    'red': [255, 0, 0],
    'green': [0, 255, 0],
    'blue': [0, 0, 255]
}


def t_test_voxelization(group1, group2, signed=False, remove_nan=True, p_cutoff=None):
    """
    t-Test on differences between the individual voxels in group1 and group2

    Arguments
    ---------
    group1, group2 : array of arrays
        The group of voxelizations to compare.
    signed : bool
        If True, return also the direction of the changes as +1 or -1.
    remove_nan : bool
        Remove Nan values from the data.
    p_cutoff : None or float
        Optional cutoff for the p-values.

    Returns
    -------
    p_values : array
        The p values for the group wise comparison.
    """
    group1 = read_group(group1)
    group2 = read_group(group2)

    t_vals, p_vals = stats.ttest_ind(group1, group2, axis=0, equal_var=True)

    if remove_nan:
        p_vals, t_vals = remove_p_val_nans(p_vals, t_vals)

    if p_cutoff is not None:
        p_vals = np.clip(p_vals, None, p_cutoff)

    if signed:
        return p_vals, np.sign(t_vals)
    else:
        return p_vals


# WARNING: needs clean up
def t_test_region_counts(counts1, counts2, signed=False, remove_nan=True, p_cutoff=None, equal_var=False):
    """t-Test on differences in counts of points in labeled regions"""

    # ids, p1 = countPointsGroupInRegions(pointGroup1, labeledImage = labeledImage, withIds = True);
    # p2 = countPointsGroupInRegions(pointGroup2,  labeledImage = labeledImage, withIds = False);

    t_vals, p_vals = stats.ttest_ind(counts1, counts2, axis=1, equal_var=equal_var)

    if remove_nan:
        p_vals, t_vals = remove_p_val_nans(p_vals, t_vals)

    if p_cutoff is not None:
        p_vals = np.clip(p_vals, None, p_cutoff)

    # p_vals.shape = (1,) + p_vals.shape;
    # ids.shape = (1,) + ids.shape;
    # p_vals = numpy.concatenate((ids.T, p_vals.T), axis = 1);

    if signed:
        return p_vals, np.sign(t_vals)
    else:
        return p_vals


# TODO: group sources in IO
def read_group(sources, combine=True, **args):
    """Turn a list of sources for data into a numpy stack.

    Arguments
    ---------
    sources : list of str or sources
        The sources to combine.
    combine : bool
        If true combine the sources to ndarray, oterhwise return a list.

    Returns
    -------
    group : array or list
        The group data.
    """

    # check if stack already:
    if isinstance(sources, np.ndarray):
        return sources

    # read the individual files
    group = []
    for f in sources:
        data = clearmap_io.as_source(f, **args).array
        data = np.reshape(data, (1,) + data.shape)
        group.append(data)

    if combine:
        return np.vstack(group)
    else:
        return group


def weights_from_precentiles(intensities, percentiles=[25, 50, 75, 100]):
    perc = np.percentiles(intensities, percentiles)
    weights = np.zeros(intensities.shape)
    for p in perc:
        ii = intensities > p
        weights[ii] = weights[ii] + 1

    return weights


def __prepare_cumulative_data(data, offset):  # FIXME: use better variable names
    # fill up the low count data
    n = np.array([x.size for x in data])
    nm = n.max()
    m = np.array([x.max() for x in data])
    mm = m.max()
    k = n.size
    # print nm, mm, k
    if offset is None:
        # assume data starts at 0 !
        offset = mm / nm  # ideal for all statistics this should be mm + eps to have as little influence as possible.
    datac = [x.copy() for x in data]
    return datac, k, m, mm, n, nm, offset


def __plot_cumulative_test(datac, m, plot):
    # test by plotting
    if plot:
        import matplotlib.pyplot as plt
        for i in range(m.size):
            datac[i].sort()
            plt.step(datac[i], np.arange(datac[i].size))


def __run_cumulative_test(datac, k, method):  # FIXME: remove one letter variable names
    if method in ('KolmogorovSmirnov', 'KS'):
        if k == 2:
            s, p = stats.ks_2samp(datac[0], datac[1])
        else:
            raise RuntimeError('KolmogorovSmirnov only for 2 samples not %d' % k)

    elif method in ('CramervonMises', 'CM'):
        if k == 2:
            s, p = clearmap_stat_tests.test_cramer_von_mises_2_sample(datac[0], datac[1])
        else:
            raise RuntimeError('CramervonMises only for 2 samples not %d' % k)

    elif method in ('AndersonDarling', 'AD'):
        s, a, p = stats.anderson_ksamp(datac)
    return p, s


def test_completed_cumulatives(data, method='AndersonDarling', offset=None, plot=False):
    """Test if data sets have the same number / intensity distribution by adding max intensity counts
    to the smaller sized data sets and performing a distribution comparison test"""

    # idea: fill up data points to the same numbers at the high intensity values and use KS test
    # cf. work in progress on thoroughly testing the differences in histograms

    datac, k, m, mm, n, nm, offset = __prepare_cumulative_data(data, offset)
    for i in range(m.size):
        if n[i] < nm:
            datac[i] = np.concatenate((datac[i], np.ones(nm-n[i], dtype=datac[i].dtype) * (mm + offset)))  # + 10E-5 * numpy.random.rand(nm-n[i])));

    __plot_cumulative_test(datac, m, plot)

    return __run_cumulative_test(datac, k, method)


def test_completed_inverted_cumulatives(data, method='AndersonDarling', offset=None, plot=False):
    """Test if data sets have the same number / intensity distribution by adding zero intensity counts
    to the smaller sized data sets and performing a distribution comparison test on the reversed cumulative distribution"""

    # idea: fill up data points to the same numbers at the high intensity values and use KS test
    # cf. work in progress on thoroughly testing the differences in histograms

    datac, k, m, mm, n, nm, offset = __prepare_cumulative_data(data, offset)
    for i in range(m.size):
        if n[i] < nm:
            datac[i] = np.concatenate((-datac[i], np.ones(nm-n[i], dtype=datac[i].dtype) * (offset)))  # + 10E-5 * numpy.random.rand(nm-n[i])));  # FIXME: only different lines with function above
        else:
            datac[i] = -datac[i]

    __plot_cumulative_test(datac, m, plot)

    return __run_cumulative_test(datac, k, method)


def remove_p_val_nans(p_vals, t_vals):
    invalid_idx = np.isnan(p_vals)
    p_vals_c = p_vals.copy()
    t_vals_c = t_vals.copy()
    p_vals_c[invalid_idx] = 1.0
    t_vals_c[invalid_idx] = 0
    return p_vals_c, t_vals_c


def stack_voxelizations(directory, f_list, suffix, channel=None):
    """
    Regroup voxelizations to simplify further processing

    Parameters
    ----------
    directory
    f_list
    suffix

    Returns
    -------

    """
    for i, file_name in enumerate(f_list):
        img = clearmap_io.read(make_abs(directory, file_name))
        if i == 0:  # init on first image
            stacked_voxelizations = img[:, :, :, np.newaxis]
        else:
            stacked_voxelizations = np.concatenate((stacked_voxelizations, img[:, :, :, np.newaxis]), axis=3)
    stacked_voxelizations = stacked_voxelizations.astype(np.float32)
    try:
        clearmap_io.write(directory /f'{channel}_stacked_density_{suffix}.tif', stacked_voxelizations, bigtiff=True)
    except ValueError:
        pass
    return stacked_voxelizations


def average_voxelization_groups(stacked_voxelizations, directory, suffix, channel='', compute_sd=False):
    avg_voxelization = np.mean(stacked_voxelizations, axis=3)
    clearmap_io.write(directory / f'{channel}_avg_density_{suffix}.tif', avg_voxelization)

    if compute_sd:
        sd_voxelization = np.std(stacked_voxelizations, axis=3)
        clearmap_io.write(directory / f'{channel}_sd_density_{suffix}.tif', sd_voxelization)


# REFACTOR: move to visualisation module
def __validate_colors(positive_color, negative_color):
    if len(positive_color) != len(negative_color):
        raise ValueError(f'Length of positive and negative colors do not match, '
                         f'got {len(positive_color)} and {len(negative_color)}')


  # REFACTOR: move to visualisation module
def color_p_values(p_vals, p_sign, positive_color=[1, 0], negative_color=[0, 1], p_cutoff=None,
                   positive_trend=[0, 0, 1, 0], negative_trend=[0, 0, 0, 1], p_max=None):
    """

    Parameters
    ----------
    p_vals np.array:
    p_sign np.array:
    positive list:
    negative list:
    p_cutoff float:
    positive_trend list:
    negative_trend list:
    p_max float:

    Returns
    -------

    """
    if p_max is None:
        p_max = p_vals.max()

    p_vals_inv = p_max - p_vals

    if p_cutoff is None:  # color given p values
        __validate_colors(positive_color, negative_color)

        # 3D + color output array
        d = len(positive_color)  # 3D
        output_shape = p_vals.shape + (d,)  # 3D + color
        colored_p_vals = np.zeros(output_shape)

        # coloring
        for neg, col in ((False, positive_color), (True, negative_color)):
            if neg:
                ids = p_sign < 0
            else:
                ids = p_sign > 0
            p_vals_i = p_vals_inv[ids]
            for i, channel_value in enumerate(col):  # [i] on R, G, B components
                colored_p_vals[ids, i] = p_vals_i * channel_value
    # else:  # split p_values according to cutoff
    #     if any([len(positive_color) != len(v) for v in (negative_color, positive_trend, negative_trend)]):
    #         raise ValueError('color_p_values: positive, negative, positive_trend and '
    #                          'negative_trend option must be equal length !')
    #     output_shape = p_vals.shape + (len(positive_color),)
    #     colored_p_vals = np.zeros(output_shape)
    #
    #     idc = p_vals < p_cutoff
    #     ids = p_sign > 0
    #     # significant positive, non sig positive, sig neg, non sig neg
    #     for id_sign, idc_sign, w in ((1, 1, positive_color), (1, -1, positive_trend),
    #                                  (-1, 1, negative_color), (-1, -1, negative_trend)):
    #         if id_sign < 0:
    #             ids = np.logical_not(ids)
    #         if idc_sign < 0:
    #             idc = np.logical_not(idc)
    #         ii = np.logical_and(ids, idc)
    #         p_vals_i = p_vals_inv[ii]
    #         for i in range(len(w)):
    #             colored_p_vals[ii, i] = p_vals_i * w[i]

    return colored_p_vals

# REFACTOR: move to visualisation module
def get_colored_p_vals(p_vals, t_vals, significance, color_names):
    p_vals_f = np.clip(p_vals, None, significance)
    p_sign = np.sign(t_vals)
    return color_p_values(p_vals_f, p_sign,
                          positive_color=colors[color_names[0]],
                          negative_color=colors[color_names[1]])


def dirs_to_density_files(directory, f_list, channel):
    out = []
    for i, f_name in enumerate(f_list):
        f_name = make_abs(directory, f_name)
        if not is_density_file(f_name):
            f_name = find_density_file(f_name, channel)
        out.append(f_name)
    return out


# def get_p_vals_f(p_vals, t_vals, p_cutoff):
#     p_vals2 = np.clip(p_vals, None, p_cutoff)
#     p_sign = np.sign(t_vals)
#     return p_vals2, p_sign


def group_cells_counts(struct_ids, group_cells_dfs, sample_ids, volume_map):
    """

    Parameters
    ----------
    struct_ids list:
    group_cells_dfs: list(pd.DataFrame)
    sample_ids: list
    volume_map: dict
        maps each id from structure_ids to the corresponding structure's volume (in pixel)

    Returns
    -------

    """
    all_ints = False
    if all_ints:
        output = pd.DataFrame(columns=['id', 'hemisphere'] + [f'counts_{str(sample_ids[i]).zfill(2)}' for i in range(len(group_cells_dfs))])
    else:
        output = pd.DataFrame(columns=['id', 'hemisphere'] + [f'counts_{sample_ids[i]}' for i in range(len(group_cells_dfs))])

    output['id'] = np.tile(struct_ids, 2)  # for each hemisphere
    output['name'] = np.tile([annotation.find(id_, key='id')['name'] for id_ in struct_ids], 2)
    output['hemisphere'] = np.repeat((0, 255), len(struct_ids))  # FIXME: translate hemisphere to plain text
    output['volume'] = output.set_index(['id', 'hemisphere']).index.map(volume_map.get)
    output = output[output['volume'].notna()]

    for multiplier, hem_id in zip((1, 2), (0, 255)):
        for j, sample_df in enumerate(group_cells_dfs):
            if all_ints:
                col_name = f'counts_{str(sample_ids[j]).zfill(2)}'  # TODO: option with f'counts_{j}'
            else:
                col_name = f'counts_{sample_ids[j]}'

            hem_sample_df = sample_df[sample_df['hemisphere'] == hem_id]
            # FIXME: replace loop (slow)
            for i, struct_id in enumerate(struct_ids):
                row_idx = output[(output['id'] == struct_id) & (output['hemisphere'] == hem_id)].index
                output.loc[row_idx, col_name] = len(hem_sample_df[hem_sample_df['id'] == struct_id])
    return output


def generate_summary_table(cells_dfs, p_cutoff=None):
    gp_names = list(cells_dfs.keys())

    grouped_counts = []

    total_df = pd.DataFrame({k: cells_dfs[gp_names[0]][k] for k in ('id', 'name', 'volume', 'hemisphere')})
    for i, gp_name in enumerate(gp_names):
        grouped_counts.append(pd.DataFrame())
        for col_name in cells_dfs[gp_name].columns:
            if 'count' in col_name:
                col = cells_dfs[gp_name][col_name]
                new_col_name = f'{gp_names[i]}_{col_name}'
                total_df[new_col_name] = col
                grouped_counts[i][new_col_name] = col
        total_df[f'mean_{gp_name}'] = grouped_counts[i].mean(axis=1).astype(float)  # To avoid "object" type
        total_df[f'sd_{gp_name}'] = grouped_counts[i].std(axis=1).astype(float)  # To avoid "object" type

    total_df, grouped_counts = sanitize_df(gp_names, grouped_counts, total_df)

    gp1 = grouped_counts[0].values.astype(int)
    gp2 = grouped_counts[1].values.astype(int)
    p_vals, p_signs = t_test_region_counts(gp1, gp2, p_cutoff=p_cutoff, signed=True)
    total_df['p_value'] = p_vals
    total_df['q_value'] = clearmap_FDR.estimate_q_values(p_vals)
    total_df['p_sign'] = p_signs.astype(int)
    return total_df


def sanitize_df(gp_names, grouped_counts, total_df):
    """
    Remove rows with all 0 or NaN in at least 1 group
    Args:
        gp_names:
        grouped_counts:
        total_df:

    Returns:

    """
    bad_idx = total_df[f'mean_{gp_names[0]}'] == 0  # FIXME: check that either not and
    bad_idx = np.logical_or(bad_idx, total_df[f'mean_{gp_names[1]}'] == 0)
    bad_idx = np.logical_or(bad_idx, np.isnan(total_df[f'mean_{gp_names[0]}']))
    bad_idx = np.logical_or(bad_idx, np.isnan(total_df[f'mean_{gp_names[1]}']))

    return total_df[~bad_idx], [grouped_counts[0][~bad_idx], grouped_counts[1][~bad_idx]]


def dirs_to_cells_dfs(directory, dirs):
    out = []
    for i, f_name in enumerate(dirs):
        f_name = make_abs(directory, f_name)
        if not f_name.endswith('cells.feather'):  # FIXME: per channel
            f_name = find_cells_df(f_name)
        out.append(pd.read_feather(f_name))
    return out


def get_volume_map(folder, channel=None):
    res = init_sample_manager_and_processors(folder)
    sample_manager = res['sample_manager']
    registration_manager = res['registration_processor']

    if channel is None:
        annotator = registration_manager.annotators[sample_manager.alignment_reference_channel]
    else:
        annotator = registration_manager.annotators[channel]
    atlas_id = registration_manager.config['atlas']['id']
    atlas_scale = [ATLAS_NAMES_MAP[atlas_id]['resolution']] * 3
    return annotator.get_lateralised_volume_map(atlas_scale)


# REFACTOR: move to separate module
def make_summary(directory, gp1_name, gp2_name, gp1_dirs, gp2_dirs, channel=None, output_path=None, save=True):
    directory = Path(directory)

    dfs = {}
    if channel is None:
        sample_manager = SampleManager()
        sample_manager.setup(src_dir=directory / gp1_dirs[0])
        channels = sample_manager.channels_to_detect
    else:
        channels = [channel]

    for channel_ in channels:
        gp1_dfs = dirs_to_cells_dfs(directory, gp1_dirs)
        gp2_dfs = dirs_to_cells_dfs(directory, gp2_dirs)
        gp_cells_dfs = [gp1_dfs, gp2_dfs]
        structs = get_all_structs(gp1_dfs + gp2_dfs)

        gp1_sample_ids = [dir_to_sample_id(folder) for folder in gp1_dirs]
        gp2_sample_ids = [dir_to_sample_id(folder) for folder in gp2_dirs]
        sample_ids = [gp1_sample_ids, gp2_sample_ids]

        volume_map = get_volume_map(gp1_dirs[0], channel=channel)  # WARNING Hacky

        aggregated_dfs = {gp_name: group_cells_counts(structs, gp_cells_dfs[i], sample_ids[i], volume_map)
                          for i, gp_name in enumerate((gp1_name, gp2_name))}
        total_df = generate_summary_table(aggregated_dfs)

        if output_path is None and save:
            output_path = directory / f'{channel}_statistics_{gp1_name}_{gp2_name}.csv'  # FIXME: per channel
        if save:
            total_df.to_csv(output_path)
        dfs[channel_] = total_df
    return dfs


def density_files_are_comparable(directory, gp1_dirs, gp2_dirs):
    gp1_f_list = dirs_to_density_files(directory, gp1_dirs, channel)  # FIXME: channel
    gp2_f_list = dirs_to_density_files(directory, gp2_dirs, channel)  # FIXME: channel
    all_files = gp1_f_list + gp2_f_list
    sizes = [os.path.getsize(f) for f in all_files]
    tolerance = 1024  # 1 KB
    comparable = all(math.isclose(s, sizes[0], abs_tol=tolerance) for s in sizes)
    if comparable:
        return True
    else:
        raise GroupStatsError(f'Could not compare files, sizes differ\n\n'
                              f'Group 1: {gp1_f_list}\n'
                              f'Group 2: {gp2_f_list}\n'
                              f'Sizes 1: {[os.path.getsize(f) for f in gp1_f_list]}\n'
                              f'Sizes 2: {[os.path.getsize(f) for f in gp2_f_list]}\n')


# REFACTOR: move to separate module
def compare_groups(directory, gp1_name, gp2_name, gp1_dirs, gp2_dirs, prefix='p_val_colors', advanced=True):
    directory = Path(directory)

    sample_manager = SampleManager()
    sample_manager.setup(src_dir=directory / gp1_dirs[0])
    result = {}
    for channel in sample_manager.channels:
        if not sample_manager.get('density', channel=channel).exists:
            print(f'No density files found for channel {channel}, skipping')
            continue

        gp1_f_list = dirs_to_density_files(directory, gp1_dirs, channel)
        gp2_f_list = dirs_to_density_files(directory, gp2_dirs, channel)

        gp1_stacked_voxelizations = stack_voxelizations(directory, gp1_f_list, channel=channel, suffix=gp1_name)
        average_voxelization_groups(gp1_stacked_voxelizations, directory, gp1_name, channel=channel, compute_sd=advanced)
        gp2_stacked_voxelizations = stack_voxelizations(directory, gp2_f_list, channel=channel, suffix=gp2_name)
        average_voxelization_groups(gp2_stacked_voxelizations, directory, gp2_name, channel=channel, compute_sd=advanced)

        t_vals, p_vals = stats.ttest_ind(gp1_stacked_voxelizations, gp2_stacked_voxelizations, axis=3, equal_var=False)
        p_vals, t_vals = remove_p_val_nans(p_vals, t_vals)

        colored_p_vals_05 = get_colored_p_vals(p_vals, t_vals, 0.05, ('red', 'green'))
        colored_p_vals_01 = get_colored_p_vals(p_vals, t_vals, 0.01, ('green', 'blue'))
        colored_p_vals = np.maximum(colored_p_vals_05, colored_p_vals_01).astype(np.uint8)

        output_f_name = f'{channel}_{prefix}_{gp1_name}_{gp2_name}.tif'
        output_file_path = directory / output_f_name
        clearmap_io.write(output_file_path, colored_p_vals, photometric='rgb', imagej=True)

        if advanced:
            effect_size = np.abs(np.mean(gp1_stacked_voxelizations, axis=3).astype(int) -
                                 np.mean(gp2_stacked_voxelizations, axis=3).astype(int))
            effect_size = effect_size.astype(np.uint16)  # for imagej compatibility
            output_f_name = f'{channel}_effect_size_{gp1_name}_{gp2_name}.tif'
            output_file_path = directory / output_f_name
            clearmap_io.write(output_file_path, effect_size, imagej=True)

        result[channel] = colored_p_vals

    return result


# def test_completed_cumulatives_in_spheres(points1, intensities1, points2, intensities2,
# shape=ano.default_annotation_file, radius = 100, method = 'AndresonDarling'):
#   """Performs completed cumulative distribution tests for each pixel using points in a ball
#   centered at that cooridnates, returns 4 arrays p value, statistic value, number in each group"""
#
#   # TODO: simple implementation -> slow -> speed up
#   if not isinstance(shape, tuple):
#     shape = io.shape(shape)
#   if len(shape) != 3:
#       raise RuntimeError('Shape expected to be 3d, found %r' % (shape,))
#
#   # distances^2 to origin
#   x1= points1[:,0]; y1 = points1[:,1]; z1 = points1[:,2]; i1 = intensities1
#   d1 = x1 * x1 + y1 * y1 + z1 * z1
#
#   x2 = points2[:,0]; y2 = points2[:,1]; z2 = points2[:,2]; i2 = intensities2
#   d2 = x2 * x2 + y2 * y2 + z2 * z2
#
#   r2 = radius * radius  # WARNING: inhomogenous in 3d !
#
#   p = np.zeros(dataSize)
#   s = np.zeros(dataSize)
#   n1 = np.zeros(dataSize, dtype='int')
#   n2 = np.zeros(dataSize, dtype='int')
#
#   for x in range(dataSize[0]):
#   #print x
#     for y in range(dataSize[1]):
#       #print y
#       for z in range(dataSize[2]):
#         #print z
#         d11 = d1 - 2 * (x * x1 + y * y1 + z * z1) + (x*x + y*y + z*z)
#         d22 = d2 - 2 * (x * x2 + y * y2 + z * z2) + (x*x + y*y + z*z)
#
#         ii1 = d11 < r2
#         ii2 = d22 < r2
#
#         n1[x,y,z] = ii1.sum()
#         n2[x,y,z] = ii2.sum()
#
#         if n1[x,y,z] > 0 and n2[x,y,z] > 0:
#             (pp, ss) = self.testCompletedCumulatives((i1[ii1], i2[ii2]), method = method)
#         else:
#             pp = 0; ss = 0
#
#         p[x,y,z] = pp
#         s[x,y,z] = ss
#
#   return (p,s,n1,n2)
#
#
# ###############################################################################
# ### Tests
# ###############################################################################
#
# def _test():
#     """Test the statistics array"""
#     import numpy as np
#     import ClearMap.Analysis.Statistics.GroupStatistics as st
#
#     s = np.ones((5,4,20))
#     s[:, 0:3, :] = - 1
#
#     x = np.random.rand(4,4,20)
#     y = np.random.rand(5,4,20) + s
#
#     pvals, psign = st.t_test_voxelization(x,y, signed = True)
#
#     pvalscol = st.color_p_values(pvals, psign, positive = [255,0,0], negative = [0,255,0])
#
#     import ClearMap.Visualization.Plot3d as p3d
#     p3d.plot(pvalscol)
