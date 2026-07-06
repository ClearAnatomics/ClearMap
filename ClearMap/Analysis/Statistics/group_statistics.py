# -*- coding: utf-8 -*-
"""
Statistics
==========

Create some statistics to test significant changes in voxelized and labeled 
data.
"""
__author__ = ('Christoph Kirst <christoph.kirst.ck@gmail.com>, '
              'Sophie Skriabine <sophie.skriabine@icm-institute.org>, '
              'Charly Rousseau <charly.rousseau@icm-institute.org>')
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__ = 'https://idisco.info'
__download__ = 'https://github.com/ClearAnatomics/ClearMap'

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from ClearMap.IO import IO as clearmap_io

from ClearMap.Analysis.Statistics import MultipleComparisonCorrection as clearmap_FDR

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
# REFACTOR: move to visualisation module
P_VALUE_COLORS = {
    'red':   [255, 0, 0],
    'green': [0, 255, 0],
    'blue':  [0, 0, 255]
}

# ---------------------------------------------------------------------------
#  Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LoadedPValueResults:
    gp1_avg: np.ndarray
    gp1_sd: Optional[np.ndarray]
    gp2_avg: np.ndarray
    gp2_sd: Optional[np.ndarray]
    p_vals: np.ndarray
    effect_size: Optional[np.ndarray]

    @property
    def has_sd(self) -> bool:
        return (self.gp1_sd is not None) and (self.gp2_sd is not None)

    @property
    def has_effect(self) -> bool:
        return self.effect_size is not None

    @property
    def gp1_imgs(self):
        return [self.gp1_avg, self.gp1_sd] if self.has_sd else self.gp1_avg

    @property
    def gp2_imgs(self):
        return [self.gp2_avg, self.gp2_sd] if self.has_sd else self.gp2_avg

    @property
    def stats_imgs(self):
        return [self.p_vals, self.effect_size] if self.has_effect else self.p_vals


# ---------------------------------------------------------------------------
#  Pure statistics
# ---------------------------------------------------------------------------

def remove_p_val_nans(p_vals, t_vals):
    invalid_idx = np.isnan(p_vals)
    p_vals_c = p_vals.copy()
    p_vals_c[invalid_idx] = 1.0
    t_vals_c = t_vals.copy()
    t_vals_c[invalid_idx] = 0.0
    return p_vals_c, t_vals_c


def t_test_voxelization(group1, group2, *, signed=False, remove_nan=True, p_cutoff=None):
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
def t_test_region_counts(counts1, counts2, *, signed=False, remove_nan=True, p_cutoff=None, equal_var=False):
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


# ---------------------------------------------------------------------------
#  Data loading / stacking
# ---------------------------------------------------------------------------

# TODO: group sources in IO
def read_group(sources, combine=True, **args):
    """
    Turn a list of sources for data into a numpy stack.

    Arguments
    ---------
    sources : list of str or sources
        The sources to combine.
    combine : bool
        If true combine the sources to ndarray, otherwise return a list.

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
        data = clearmap_io.read(f)
        group.append(data[np.newaxis, ...])
    if combine:
        return np.vstack(group)
    else:
        return group

def stack_voxelizations(arrays: list[np.ndarray]) -> np.ndarray:
    """
    Stack a list of 3-D voxelization arrays into a single (X, Y, Z, N) float32 array.

    Parameters
    ----------
    arrays : list of np.ndarray
        Per-sample voxelization volumes, each shaped (X, Y, Z).

    Returns
    -------
    np.ndarray
        Shape (X, Y, Z, N).
    """
    expanded_arrays = [a[:, :, :, np.newaxis] for a in arrays]  # Add concatenation dimension
    return np.concatenate(expanded_arrays, axis=3).astype(np.float32)


# ---------------------------------------------------------------------------
#  P-value coloring  (REFACTOR: move to visualisation module eventually)
# ---------------------------------------------------------------------------

def __validate_colors(positive_color, negative_color):
    if len(positive_color) != len(negative_color):
        raise ValueError(f'Length of positive and negative colors do not match, '
                         f'got {len(positive_color)} and {len(negative_color)}')


def color_p_values(p_vals, p_sign, positive_color=(1, 0), negative_color=(0, 1), p_cutoff=None,
                   positive_trend=(0, 0, 1, 0), negative_trend=(0, 0, 0, 1), p_max=None):
    """

    Parameters
    ----------
    p_vals : np.ndarray
    p_sign : np.ndarray
    positive_color : tuple
    negative_color : tuple
    p_cutoff : float, optional
    positive_trend : tuple
    negative_trend : tuple
    p_max : float, optional

    Returns
    -------
    np.ndarray
    """
    if p_max is None:
        p_max = p_vals.max()

    p_vals_inv = p_max - p_vals

    if p_cutoff is None:  # color given p values
        __validate_colors(positive_color, negative_color)

        # 3D + color output array
        d = len(positive_color)  # 3D
        output_shape = p_vals.shape + (d,)  # 3D + color
        colored_p_vals = np.zeros(output_shape)  # FIXME: simplify with newaxis

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
                          positive_color=P_VALUE_COLORS[color_names[0]],
                          negative_color=P_VALUE_COLORS[color_names[1]])


# ---------------------------------------------------------------------------
#  Region-level cell counting
# ---------------------------------------------------------------------------

def group_region_counts(annotator, region_ids, group_dfs, sample_ids, volume_map) -> pd.DataFrame:
    """
    Count entities (cells, tracts, …) per region per hemisphere for each sample in a group.

    .. note::

        Works for any labeled DataFrame that has an 'id' column.
        'hemisphere' is optional — when absent all entities are treated as
        belonging to a single synthetic hemisphere (value 0).

    Parameters
    ----------
    annotator : Annotator
        Atlas annotator for structure name lookup.
    region_ids : array-like
        Region IDs to count.
    group_dfs : list[pd.DataFrame]
        One DataFrame per sample, with 'id' and (optional) 'hemisphere' columns.
    sample_ids : list
        Sample identifiers (strings or ints).
    volume_map : dict
        Maps (id, hemisphere) to structure volume in pixels.

    Returns
    -------
    pd.DataFrame
    """
    has_hemisphere = all('hemisphere' in df.columns for df in group_dfs)
    hemispheres = (0, 255) if has_hemisphere else (0,)

    all_ints = all(isinstance(sid, int) or (isinstance(sid, str) and sid.isdigit())
                   for sid in sample_ids)
    col_ids = [str(sid).zfill(2) if all_ints else str(sid) for sid in sample_ids]
    count_cols = [f'counts_{cid}' for cid in col_ids]

    output = pd.DataFrame(columns=['id', 'hemisphere'] + count_cols)
    output['id'] = np.tile(region_ids, len(hemispheres))
    output['name'] = np.tile([annotator.find(id_, key='id')['name'] for id_ in region_ids], len(hemispheres))
    output['hemisphere'] = np.repeat(hemispheres, len(region_ids))  # FIXME: translate hemisphere to plain text
    output['volume'] = output.set_index(['id', 'hemisphere']).index.map(volume_map.get)
    output = output[output['volume'].notna()]

    for hem_id in hemispheres:
        for j, sample_df in enumerate(group_dfs):
            hem_df = (sample_df[sample_df['hemisphere'] == hem_id]
                      if has_hemisphere else sample_df)
            for struct_id in region_ids:
                mask = ((output['id'] == struct_id) &
                        (output['hemisphere'] == hem_id))
                output.loc[mask, count_cols[j]] = len(hem_df[hem_df['id'] == struct_id])

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
    """
    bad_idx = total_df[f'mean_{gp_names[0]}'] == 0  # FIXME: check that either not and
    bad_idx = np.logical_or(bad_idx, total_df[f'mean_{gp_names[1]}'] == 0)
    bad_idx = np.logical_or(bad_idx, np.isnan(total_df[f'mean_{gp_names[0]}']))
    bad_idx = np.logical_or(bad_idx, np.isnan(total_df[f'mean_{gp_names[1]}']))

    return total_df[~bad_idx], [grouped_counts[0][~bad_idx], grouped_counts[1][~bad_idx]]


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