# -*- coding: utf-8 -*-
"""
Statistics
==========

Create some statistics to test significant changes in voxelized and labeled 
data.
"""
__author__ = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__ = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__ = 'https://idisco.info'
__download__ = 'https://www.github.com/ChristophKirst/ClearMap2'


import numpy as np
from scipy import stats

import ClearMap.IO.IO as clearmap_io
import ClearMap.Alignment.Annotation as annotation
import ClearMap.Analysis.Statistics.StatisticalTests as clearmap_stat_tests


def t_test_voxelization(group1, group2, signed=False, remove_nan=True, p_cutoff=None):
    """t-Test on differences between the individual voxels in group1 and group2

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
        The gorup data.
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


def color_p_values(p_vals, p_sign, positive_color=[1, 0], negative_color=[0, 1], p_cutoff=None,
                   positive_trend=[0, 0, 1, 0], negative_trend=[0, 0, 0, 1], p_max=None):
    """

    Parameters
    ----------
    p_vals np.array
    p_sign np.array
    positive list
    negative list
    pcutoff float
    positivetrend list
    negativetrend list
    pmax float

    Returns
    -------

    """
    if p_max is None:
        p_max = p_vals.max()
    p_vals_inv = p_max - p_vals

    if p_cutoff is None:  # color given p values
        if len(positive_color) != len(negative_color):
            raise ValueError(f'Length of positive and negative colors do not match, '
                             f'got {len(positive_color)} and {len(negative_color)}')
        d = len(positive_color)  # 3D
        output_shape = p_vals.shape + (d,)  # 3D + color
        colored_p_vals = np.zeros(output_shape)

        # color
        for sign, col in ((1, positive_color), (-1, negative_color)):
            ids = sign * (p_sign > 0)  # positive of inverse for negative
            p_vals_i = p_vals_inv[ids]
            for i in range(len(col)):
                colored_p_vals[ids, i] = p_vals_i * col[i]
    else:  # split p_values according to cutoff
        if any([len(positive_color) != len(v) for v in (negative_color, positive_trend, negative_trend)]):
            raise ValueError('color_p_values: positive, negative, positive_trend and '
                             'negative_trend option must be equal length !')
        output_shape = p_vals.shape + (len(positive_color),)
        colored_p_vals = np.zeros(output_shape)

        idc = p_vals < p_cutoff
        ids = p_sign > 0
        # significant positive, non sig positive, sig neg, non sig neg
        for id_sign, idc_sign, w in ((1, 1, positive_color), (1, -1, positive_trend),
                                     (-1, 1, negative_color), (-1, -1, negative_trend)):
            ii = np.logical_and(id_sign * ids, idc_sign * idc)
            p_vals_i = p_vals_inv[ii]
            for i in range(len(w)):
                colored_p_vals[ii, i] = p_vals_i * w[i]

    return colored_p_vals


def weights_from_precentiles(intensities, percentiles=[25, 50, 75, 100]):
    perc = np.percentiles(intensities, percentiles)
    weights = np.zeros(intensities.shape)
    for p in perc:
        ii = intensities > p
        weights[ii] = weights[ii] + 1

    return weights


# # needs clean up
# def count_points_group_in_regions(point_group, annotation_file=annotation.default_annotation_file,
#                                   weight_group=None, invalid=0, hierarchical=True):
#     """Generates a table of counts for the various point datasets in pointGroup"""
#
#     if intensity_group is None:
#         counts = [annotation.count_points(point_group[i], annotation_file=annotation_file, invalid=invalid, hierarchical=hierarchical) for i in range(len(point_group))]
#     else:
#         counts = [annotation.count_points(point_group[i], weight=weight_group[i], annotation_file=annotation_file, invalid=invalid, hierarchical=hierarchical) for i in range(len(point_group))]
#
#     counts = np.vstack(counts).T
#
#     if returnIds:
#         ids = np.sort(lbl.Label.ids)
#         if intensities is None:
#             return ids, counts
#         else:
#             return ids, counts, intensities
#     else:
#         if intensities is None:
#             return counts
#         else:
#             return counts, intensities
#
#
# def countPointsGroupInRegions(pointGroup, labeledImage=lbl.DefaultLabeledImageFile, intensityGroup=None, intensityRow=0,  # FIXME: V1
#                               returnIds=True, returnCounts=False, collapse=None):
#     """Generates a table of counts for the various point datasets in pointGroup"""
#
#     if intensityGroup is None:
#         counts = [
#             lbl.countPointsInRegions(pointGroup[i], labeledImage=labeledImage, sort=True, allIds=True, returnIds=False,
#                                      returnCounts=returnCounts, intensities=None, collapse=collapse) for i in
#             range(len(pointGroup))]
#     else:
#         counts = [
#             lbl.countPointsInRegions(pointGroup[i], labeledImage=labeledImage, sort=True, allIds=True, returnIds=False,
#                                      returnCounts=returnCounts,
#                                      intensities=intensityGroup[i], intensityRow=intensityRow, collapse=collapse) for i
#             in range(len(pointGroup))]
#
#     if returnCounts and intensityGroup is not None:
#         countsi = (c[1] for c in counts)
#         counts = (c[0] for c in counts)
#     else:
#         countsi = None
#
#     counts = np.vstack((c for c in counts)).T
#     if not countsi is None:
#         countsi = np.vstack((c for c in countsi)).T
#
#     if returnIds:
#         ids = np.sort(lbl.Label.ids)
#         if countsi is None:
#             return ids, counts
#         else:
#             return ids, counts, countsi
#     else:
#         if countsi is None:
#             return counts
#         else:
#             return counts, countsi


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


# def test_completed_cumulatives_in_spheres(points1, intensities1, points2, intensities2, shape = ano.default_annotation_file, radius = 100, method = 'AndresonDarling'):
#   """Performs completed cumulative distribution tests for each pixel using points in a ball centered at that cooridnates, returns 4 arrays p value, statistic value, number in each group"""
#
#   #TODO: sinple implementation -> slow -> speed up
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
#   r2 = radius * radius  # TODO: inhomogenous in 3d !
#
#   p = np.zeros(dataSize)
#   s = np.zeros(dataSize)
#   n1 = np.zeros(dataSize, dtype = 'int')
#   n2 = np.zeros(dataSize, dtype = 'int')
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


def remove_p_val_nans(p_vals, t_vals):
    invalid_idx = np.isnan(p_vals)
    p_vals_c = p_vals.copy()
    t_vals_c = t_vals.copy()
    p_vals_c[invalid_idx] = 1.0
    t_vals_c[invalid_idx] = 0
    return p_vals_c, t_vals_c
