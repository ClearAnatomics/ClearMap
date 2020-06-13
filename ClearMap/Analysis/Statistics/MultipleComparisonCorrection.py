# -*- coding: utf-8 -*-
"""
MultipleComparisonCorrection
============================

Correction methods for multiple comparison tests.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


import numpy
import scipy
import scipy.interpolate
   

###############################################################################
### Bnejamini Hochberg correction
############################################################################### 
    
def correct_p_values(pvalues, method = 'BH'):
  """Corrects p-values for multiple testing using various methods. 
  
  Arguments
  ---------
  pvalues : array
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
  - `Benjamini Hochberg, 1995 <http://www.jstor.org/stable/2346101?seq=1#page_scan_tab_contents>`_
  - `Bonferoni correction <http://www.tandfonline.com/doi/abs/10.1080/01621459.1961.10482090#.VmHWUHbH6KE>`_
  - `R statistics package <https://www.r-project.org/>`_
  
  Notes
  -----
  Modified from http://statsmodels.sourceforge.net/ipdirective/generated/scikits.statsmodels.sandbox.stats.multicomp.multipletests.html.
  """
  
  pvals = numpy.asarray(pvalues);

  if method.lower() in ['bh', 'fdr']:
    pvals_sorted_ids = numpy.argsort(pvals);
    pvals_sorted = pvals[pvals_sorted_ids]
    sorted_ids_inv = pvals_sorted_ids.argsort()

    n = len(pvals);
    bhfactor = numpy.arange(1,n+1)/float(n);

    pvals_corrected_raw = pvals_sorted / bhfactor;
    pvals_corrected = numpy.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
    pvals_corrected[pvals_corrected>1] = 1;

    return pvals_corrected[sorted_ids_inv];
  
  elif method.lower() in ['b', 'fwer']:
    n = len(pvals);        
    
    pvals_corrected = n * pvals;
    pvals_corrected[pvals_corrected>1] = 1;\
    
    return pvals_corrected;
      
  #return reject[pvals_sortind.argsort()]


def estimate_q_values(pvalues, m = None, pi0 = None, verbose = False, low_memory = False):
  """Estimates q-values from p-values.

  Arguments
  ---------
  pvalues : array
    List of p-values.
  m : int or None
    Number of tests. If None, m = pvalues.size
  pi0 : float or None
    Estimate of m_0 / m which is the (true null / total tests) ratio.
    If None estimation via cubic spline.
  verbose : bool
    Print info during execution
  low_memory : bool
    If true, use low memory version.
    
  Returns
  -------
  qvalues : array
    The q values.
  
  Notes
  -----
  - The q-value of a particular feature can be described as the expected 
    proportion of false  positives  among  all  features  as  or  more  
    extreme  than  the observed one.
  - The estimated q-values are increasing in the same order as the p-values.  
      
  References
  ----------
  - `Storey and Tibshirani, 2003 <http://www.pnas.org/content/100/16/9440.full>`_
  - modified from https://github.com/nfusi/qvalue
  """

  if not (pvalues.min() >= 0 and pvalues.max() <= 1):
    raise RuntimeError("estimateQValues: p-values should be between 0 and 1");

  original_shape = pvalues.shape
  pvalues = pvalues.ravel() # flattens the array in place, more efficient than flatten() 

  if m == None:
    m = float(len(pvalues))
  else:
    # the user has supplied an m
    m *= 1.0

  # if the number of hypotheses is small, just set pi0 to 1
  if len(pvalues) < 100 and pi0 == None:
    pi0 = 1.0
  elif pi0 != None:
    pi0 = pi0
  else:
    # evaluate pi0 for different lambdas
    pi0 = []
    lam = scipy.arange(0, 0.90, 0.01)
    counts = scipy.array([(pvalues > i).sum() for i in lam])
      
    for l in range(len(lam)):
      pi0.append(counts[l]/(m*(1-lam[l])))

    pi0 = scipy.array(pi0)

    # fit natural cubic scipyline
    tck = scipy.interpolate.splrep(lam, pi0, k = 3)
    pi0 = scipy.interpolate.splev(lam[-1], tck)
    
    if pi0 > 1:
      if verbose:
        raise Warning("estimateQValues: got pi0 > 1 (%.3f) while estimating qvalues, setting it to 1" % pi0);
      pi0 = 1.0
      
  if not (pi0 >= 0 and pi0 <= 1):
      raise RuntimeError("estimateQValues: pi0 is not between 0 and 1: %f" % pi0);


  if low_memory:
    # low memory version, only uses 1 pvalues and 1 qv matrices
    qv = scipy.zeros((len(pvalues),))
    last_pvalues = pvalues.argmax()
    qv[last_pvalues] = (pi0*pvalues[last_pvalues]*m)/float(m)
    pvalues[last_pvalues] = -scipy.inf
    prev_qv = last_pvalues
    for i in range(int(len(pvalues))-2, -1, -1):
      cur_max = pvalues.argmax()
      qv_i = (pi0*m*pvalues[cur_max]/float(i+1))
      pvalues[cur_max] = -scipy.inf
      qv_i1 = prev_qv
      qv[cur_max] = min(qv_i, qv_i1)
      prev_qv = qv[cur_max]

  else:
    p_ordered = scipy.argsort(pvalues)    
    pvalues = pvalues[p_ordered]
    qv = pi0 * m/len(pvalues) * pvalues
    qv[-1] = min(qv[-1],1.0)

    for i in range(len(pvalues)-2, -1, -1):
        qv[i] = min(pi0*m*pvalues[i]/(i+1.0), qv[i+1])
    
    # reorder qvalues
    qv_temp = qv.copy()
    qv = scipy.zeros_like(qv)
    qv[p_ordered] = qv_temp

    # reshape qvalues
    qv = qv.reshape(original_shape)
      
  return qv
  

###############################################################################
### Tests
###############################################################################

def _test():
  import numpy as np
  import ClearMap.Analysis.Statistics.MultipleComparisonCorrection as mcc


