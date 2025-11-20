# -*- coding: utf-8 -*-
"""
MultipleComparisonCorrection
============================

Correction methods for multiple comparison tests.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>, Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'https://idisco.info'
__download__  = 'https://www.github.com/ChristophKirst/ClearMap2'


import numpy as np
import scipy
import scipy.interpolate


###############################################################################
# ## Benjamini Hochberg correction
############################################################################### 

def correct_p_values(p_values, method='BH'):
    """Corrects p-values for multiple testing using various methods.

    Arguments
    ---------
    p_values : array
      List of p values to be corrected.
    method : str
      Optional method to use: 'BH' = 'FDR' = 'Benjamini-Hochberg',
      'B' = 'FWER' = 'Bonferoni'.

    Returns
    -------
    qvalues : array
      Corrected p values.

    References
    ----------
    - `Benjamini Hochberg, 1995 <https://www.jstor.org/stable/2346101?seq=1#page_scan_tab_contents>`_
    - `Bonferoni correction <https://www.tandfonline.com/doi/abs/10.1080/01621459.1961.10482090#.VmHWUHbH6KE>`_
    - `R statistics package <https://www.r-project.org/>`_

    Notes
    -----
    Modified from http://statsmodels.sourceforge.net/ipdirective/generated/scikits.statsmodels.sandbox.stats.multicomp.multipletests.html.
    """

    p_vals = np.asarray(p_values)
    n = len(p_vals)
    if method.lower() in ['bh', 'fdr']:
        pvals_sorted_ids = np.argsort(p_vals)
        pvals_sorted = p_vals[pvals_sorted_ids]
        sorted_ids_inv = pvals_sorted_ids.argsort()

        bhfactor = np.arange(1, n+1) / float(n)

        pvals_corrected_raw = pvals_sorted / bhfactor
        pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
        pvals_corrected[pvals_corrected > 1] = 1

        return pvals_corrected[sorted_ids_inv]
    elif method.lower() in ['b', 'fwer']:
        pvals_corrected = n * p_vals
        pvals_corrected[pvals_corrected > 1] = 1
        return pvals_corrected
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'BH' or 'B' "
                         f"for Benjamini Hochberg and Bonferoni respectively.")

    #return reject[pvals_sortind.argsort()]


def estimate_q_values(p_values, m=None, pi0=None, verbose=False, low_memory=False):
    """Estimates q-values from p-values.

    Arguments
    ---------
    p_values : array
      List of p-values.
    m : int or None
      Number of tests. If None, m = p_values.size
    pi0 : float or None
      Estimate of m_0 / m which is the (true null / total tests) ratio.
      If None estimation via cubic spline.
    verbose : bool
      Print info during execution
    low_memory : bool
      If true, use low memory version.

    Returns
    -------
    q_values : array
      The q values.

    Notes
    -----
    - The q-value of a particular feature can be described as the expected
      proportion of false  positives  among  all  features  as  or  more
      extreme  than  the observed one.
    - The estimated q-values are increasing in the same order as the p-values.

    References
    ----------
    - `Storey and Tibshirani, 2003 <https://www.pnas.org/content/100/16/9440.full>`_
    - modified from https://github.com/nfusi/qvalue
    """

    if not (p_values.min() >= 0 and p_values.max() <= 1):
        raise RuntimeError('estimateQValues: p-values should be between 0 and 1')

    original_shape = p_values.shape
    p_values = p_values.ravel() # flattens the array in place, more efficient than flatten()

    if m is None:
        m = float(len(p_values))
    else:
        # the user has supplied an m
        m *= 1.0

    # if the number of hypotheses is small, just set pi0 to 1
    if len(p_values) < 100 and pi0 is None:
        pi0 = 1.0
    elif pi0 is not None:
        pi0 = pi0
    else:
        # evaluate pi0 for different lambdas
        pi0 = []
        lam = np.arange(0, 0.90, 0.01)
        counts = np.array([(p_values > i).sum() for i in lam])

        for l in range(len(lam)):
            pi0.append(counts[l]/(m*(1-lam[l])))

        pi0 = np.array(pi0)

        # fit natural cubic scipy line
        tck = scipy.interpolate.splrep(lam, pi0, k = 3)
        pi0 = scipy.interpolate.splev(lam[-1], tck)

        if pi0 > 1:
            if verbose:
                raise Warning(f'estimateQValues: got pi0 > 1 ({pi0:.3f}) while estimating qvalues, setting it to 1')
            pi0 = 1.0

    if not (0 <= pi0 <= 1):
        raise RuntimeError(f'estimateQValues: pi0 is not between 0 and 1: {pi0:f}')

    if low_memory:
        # low memory version, only uses 1 p_values and 1 qv matrices
        qv = scipy.zeros((len(p_values),))
        last_p_values = p_values.argmax()
        qv[last_p_values] = (pi0 * p_values[last_p_values] * m) / float(m)
        p_values[last_p_values] = -scipy.inf
        prev_qv = last_p_values
        for i in range(int(len(p_values)) - 2, -1, -1):
            cur_max = p_values.argmax()
            qv_i = (pi0 * m * p_values[cur_max] / float(i + 1))
            p_values[cur_max] = -scipy.inf
            qv_i1 = prev_qv
            qv[cur_max] = min(qv_i, qv_i1)
            prev_qv = qv[cur_max]

    else:
        p_ordered = p_values.argsort()
        p_values = p_values[p_ordered]
        qv = pi0 * m / len(p_values) * p_values
        qv[-1] = min(qv[-1],1.0)

        for i in range(len(p_values) - 2, -1, -1):
            qv[i] = min(pi0 * m * p_values[i] / (i + 1.0), qv[i + 1])

        # reorder q_values
        qv_temp = qv.copy()
        qv = np.zeros_like(qv)
        qv[p_ordered] = qv_temp

        # reshape q_values
        qv = qv.reshape(original_shape)

    return qv
