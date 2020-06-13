# -*- coding: utf-8 -*-
"""
Statistics
==========

Create some statistics to test significant changes in voxelized and labeled 
data.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


import numpy as np
from scipy import stats

import ClearMap.IO.IO as io

import ClearMap.Alignment.Annotation as ano
import ClearMap.Analysis.Statistics.StatisticalTests as st


def t_test_voxelization(group1, group2, signed = False, remove_nan = True, p_cutoff = None):
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
  group1 = read_group(group1);  
  group2 = read_group(group2);  
  
  tvals, pvals = stats.ttest_ind(group2, group2, axis=0, equal_var=True);

  #remove nans
  if remove_nan: 
    pi = np.isnan(pvals);
    pvals[pi] = 1.0;
    tvals[pi] = 0;

  pvals = cutoff_p_values(pvals, p_cutoff=p_cutoff);

  #return
  if signed:
      return pvals, np.sign(tvals);
  else:
      return pvals;



#TODO: group sources in IO
def read_group(sources, combine = True, **args):
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
  
  #check if stack already:
  if isinstance(sources, np.ndarray):
    return sources;
  
  #read the individual files
  group = [];
  for f in sources:
    data = io.as_source(f, **args).array;
    data = np.reshape(data, (1,) + data.shape);
    group.append(data);
  
  if combine:
    return np.vstack(group);
  else:
    return group;
        
def cutoff_p_values(pvals, p_cutoff = 0.05):
  """cutt of p-values above a threshold.
  
  Arguments
  ---------
  p_valiues : array
    The p values to truncate.
  p_cutoff : float or None
    The p-value cutoff. If None, do not cut off.

  Returns
  -------
  p_values : array
    Cut off p-values.
  """
  pvals2 = pvals.copy();
  pvals2[pvals2 > p_cutoff]  = p_cutoff;
  return pvals2;
    

def color_p_values(pvals, psign, positive = [1,0], negative = [0,1], p_cutoff = None, positive_trend = [0,0,1,0], negative_trend = [0,0,0,1], pmax = None):
    
    pvalsinv = pvals.copy();
    if pmax is None:
        pmax = pvals.max();    
    pvalsinv = pmax - pvalsinv;    
    
    if p_cutoff is None:  # color given p values
        
        d = len(positive);
        ds = pvals.shape + (d,);
        pvc = np.zeros(ds);
    
        #color
        ids = psign > 0;
        pvalsi = pvalsinv[ids];
        for i in range(d):
            pvc[ids, i] = pvalsi * positive[i];
    
        ids = psign < 0;
        pvalsi = pvalsinv[ids];
        for i in range(d):
            pvc[ids, i] = pvalsi * negative[i];
        
        return pvc;
        
    else:  # split pvalues according to cutoff
    
        d = len(positive_trend);
        
        if d != len(positive) or  d != len(negative) or  d != len(negative_trend) :
            raise RuntimeError('colorPValues: postive, negative, postivetrend and negativetrend option must be equal length!');
        
        ds = pvals.shape + (d,);
        pvc = np.zeros(ds);
    
        idc = pvals < p_cutoff;
        ids = psign > 0;

        ##color 
        # significant postive
        ii = np.logical_and(ids, idc);
        pvalsi = pvalsinv[ii];
        w = positive;
        for i in range(d):
            pvc[ii, i] = pvalsi * w[i];
    
        #non significant postive
        ii = np.logical_and(ids, np.negative(idc));
        pvalsi = pvalsinv[ii];
        w = positive_trend;
        for i in range(d):
            pvc[ii, i] = pvalsi * w[i];
            
         # significant negative
        ii = np.logical_and(np.negative(ids), idc);
        pvalsi = pvalsinv[ii];
        w = negative;
        for i in range(d):
            pvc[ii, i] = pvalsi * w[i];
    
        #non significant postive
        ii = np.logical_and(np.negative(ids), np.negative(idc))
        pvalsi = pvalsinv[ii];
        w = negative_trend;
        for i in range(d):
            pvc[ii, i] = pvalsi * w[i];
        
        return pvc;
    

    
    
def mean(group, **args):
    g = read_group(group, **args);  
    return g.mean(axis = 0);


def std(group, **args):
    g = read_group(group, **args);  
    return g.std(axis = 0);

   
def var(group, **args):
    g = read_group(group, **args);  
    return g.var(axis = 0);    
    

def weights_from_precentiles(intensities, percentiles = [25,50,75,100]):
    perc = np.percentiles(intensities, percentiles);
    weights = np.zeros(intensities.shape);
    for p in perc:
        ii = intensities > p;
        weights[ii] = weights[ii] + 1;
    
    return weights;
        

# needs clean up
def count_points_group_in_regions(point_group, annotation_file = ano.default_annotation_file, weight_group = None, invalid = 0, hierarchical = True):
  """Generates a table of counts for the various point datasets in pointGroup"""

  if intensity_group is None: 
    counts = [ano.count_points(point_group[i], annotation_file=annotation_file, invalid=invalid, hierarchical=heirarchical) for i in range(len(point_group))];
  else:
    counts = [ano.count_points(point_group[i], weight=weight_group[i], annotation_file=annotation_file, invalid=invalid, hierarchical=heirarchical) for i in range(len(point_group))];
  
  counts = np.vstack(counts).T;

  return counts;
         


# needs clean up
def t_test_region_countss(counts1, counts2, annotation_file = ano.default_annotation_file, signed = False, remove_nan = True, p_cutoff = None, equal_var = False):
  """t-Test on differences in counts of points in labeled regions"""
  
  #ids, p1 = countPointsGroupInRegions(pointGroup1, labeledImage = labeledImage, withIds = True);
  #p2 = countPointsGroupInRegions(pointGroup2,  labeledImage = labeledImage, withIds = False);   
  
  tvals, pvals = st.ttest_ind(counts1, counts2, axis=1, equal_var=equal_var);
  
  #remove nans
  if remove_nan: 
      pi = np.isnan(pvals);
      pvals[pi] = 1.0;
      tvals[pi] = 0;

  pvals = cutoff_p_values(pvals, p_cutoff = p_cutoff);
  
  #pvals.shape = (1,) + pvals.shape;
  #ids.shape = (1,) + ids.shape;
  #pvals = numpy.concatenate((ids.T, pvals.T), axis = 1);
  
  if signed:
    return pvals, np.sign(tvals);
  else:
    return pvals;

    

def test_completed_cumulatives(data, method = 'AndersonDarling', offset = None, plot = False):
  """Test if data sets have the same number / intensity distribution by adding max intensity counts to the smaller sized data sets and performing a distribution comparison test"""
  
  #idea: fill up data points to the same numbers at the high intensity values and use KS test
  #cf. work in progress on thoouroghly testing the differences in histograms
  
  #fill up the low count data
  n = np.array([x.size for x in data]);
  nm = n.max();
  m = np.array([x.max() for x in data]);
  mm = m.max();
  k = n.size;
  #print nm, mm, k
  
  if offset is None:
    #assume data starts at 0 !
    offset = mm / nm; #ideall for all statistics this should be mm + eps to have as little influence as possible.
  

  datac = [x.copy() for x in data];
  for i in range(m.size):
    if n[i] < nm:
        datac[i] = np.concatenate((datac[i], np.ones(nm-n[i], dtype = datac[i].dtype) * (mm + offset))); # + 10E-5 * numpy.random.rand(nm-n[i])));
        
  #test by plotting
  if plot is True:
    import matplotlib.pyplot as plt;
    for i in range(m.size):
      datac[i].sort();
      plt.step(datac[i], np.arange(datac[i].size));
  
  #perfomr the tests
  if method == 'KolmogorovSmirnov' or method == 'KS':
    if k == 2:
      (s, p) = stats.ks_2samp(datac[0], datac[1]);
    else:
      raise RuntimeError('KolmogorovSmirnov only for 2 samples not %d' % k);
      
  elif method == 'CramervonMises' or method == 'CM':
    if k == 2:
      (s,p) = st.test_cramer_von_mises_2_sample(datac[0], datac[1]);
    else:
      raise RuntimeError('CramervonMises only for 2 samples not %d' % k);
    
  elif method == 'AndersonDarling' or method == 'AD':
    (s,a,p) = stats.anderson_ksamp(datac);

  return (p,s);





def test_completed_inverted_cumulatives(data, method = 'AndersonDarling', offset = None, plot = False):
  """Test if data sets have the same number / intensity distribution by adding zero intensity counts to the smaller sized data sets and performing a distribution comparison test on the reversed cumulative distribution"""
  
  #idea: fill up data points to the same numbers at the high intensity values and use KS test
  #cf. work in progress on thoouroghly testing the differences in histograms
  
  #fill up the low count data
  n = np.array([x.size for x in data]);
  nm = n.max();
  m = np.array([x.max() for x in data]);
  mm = m.max();
  k = n.size;
  #print nm, mm, k
  
  if offset is None:
    #assume data starts at 0 !
    offset = mm / nm; #ideall for all statistics this should be mm + eps to have as little influence as possible.
  

  datac = [x.copy() for x in data];
  for i in range(m.size):
    if n[i] < nm:
      datac[i] = np.concatenate((-datac[i], np.ones(nm-n[i], dtype = datac[i].dtype) * (offset))); # + 10E-5 * numpy.random.rand(nm-n[i])));
    else:
      datac[i] = -datac[i];
        
  #test by plotting
  if plot is True:
    import matplotlib.pyplot as plt;
    for i in range(m.size):
      datac[i].sort();
      plt.step(datac[i], np.arange(datac[i].size));
  
  #perfomr the tests
  if method == 'KolmogorovSmirnov' or method == 'KS':
    if k == 2:
      (s, p) = stats.ks_2samp(datac[0], datac[1]);
    else:
      raise RuntimeError('KolmogorovSmirnov only for 2 samples not %d' % k);
      
  elif method == 'CramervonMises' or method == 'CM':
    if k == 2:
      (s,p) = st.test_cramer_von_mises_2_sample(datac[0], datac[1]);
    else:
      raise RuntimeError('CramervonMises only for 2 samples not %d' % k);
    
  elif method == 'AndersonDarling' or method == 'AD':
    (s,a,p) = stats.anderson_ksamp(datac);

  return (p,s);




def test_completed_cumulatives_in_spheres(points1, intensities1, points2, intensities2, shape = ano.default_annotation_file, radius = 100, method = 'AndresonDarling'):
  """Performs completed cumulative distribution tests for each pixel using points in a ball centered at that cooridnates, returns 4 arrays p value, statistic value, number in each group"""
  
  #TODO: sinple implementation -> slow -> speed up
  if not isinstance(shape, tuple):
    shape = io.shape(shape);
  if len(shape) != 3:
      raise RuntimeError('Shape expected to be 3d, found %r' % (shape,));
  
  # distances^2 to origin
  x1= points1[:,0]; y1 = points1[:,1]; z1 = points1[:,2]; i1 = intensities1;
  d1 = x1 * x1 + y1 * y1 + z1 * z1;
  
  x2 = points2[:,0]; y2 = points2[:,1]; z2 = points2[:,2]; i2 = intensities2;
  d2 = x2 * x2 + y2 * y2 + z2 * z2;
      
  r2 = radius * radius; # TODO: inhomogenous in 3d !
  
  p = np.zeros(dataSize);
  s = np.zeros(dataSize);
  n1 = np.zeros(dataSize, dtype = 'int');
  n2 = np.zeros(dataSize, dtype = 'int');
  
  for x in range(dataSize[0]):
  #print x
    for y in range(dataSize[1]):
      #print y
      for z in range(dataSize[2]):
        #print z
        d11 = d1 - 2 * (x * x1 + y * y1 + z * z1) + (x*x + y*y + z*z);
        d22 = d2 - 2 * (x * x2 + y * y2 + z * z2) + (x*x + y*y + z*z);
        
        ii1 = d11 < r2;
        ii2 = d22 < r2;

        n1[x,y,z] = ii1.sum();
        n2[x,y,z] = ii2.sum();
        
        if n1[x,y,z] > 0 and n2[x,y,z] > 0:
            (pp, ss) = self.testCompletedCumulatives((i1[ii1], i2[ii2]), method = method);
        else:
            pp = 0; ss = 0;
        
        p[x,y,z] = pp; 
        s[x,y,z] = ss;
  
  return (p,s,n1,n2);
        
###############################################################################
### Tests
###############################################################################

def _test():
    """Test the statistics array"""
    import numpy as np
    import ClearMap.Analysis.Statistics.GroupStatistics as st
    
    s = np.ones((5,4,20));
    s[:, 0:3, :] = - 1;
    
    x = np.random.rand(4,4,20);
    y = np.random.rand(5,4,20) + s;
    
    pvals, psign = st.t_test_voxelization(x,y, signed = True);

    pvalscol = st.color_p_values(pvals, psign, positive = [255,0,0], negative = [0,255,0])
    
    import ClearMap.Visualization.Plot3d as p3d
    p3d.plot(pvalscol)

    
