import numpy as np
from scipy import stats

from ClearMap.Analysis.Statistics import StatisticalTests as clearmap_stat_tests


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
