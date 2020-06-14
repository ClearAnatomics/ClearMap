"""
SkeletonProcessing
==================

Utils to post process skeletons.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'

import numpy as np

import ClearMap.ParallelProcessing.DataProcessing.ConvolvePointList as cpl;
import ClearMap.ImageProcessing.Topology.Topology3d as t3d

import ClearMap.Utils.Timer as tmr;

###############################################################################
### Topology
###############################################################################

def clean_open_branches(skeleton, skelton_copy, points, radii, length, clean = True, verbose = False):
  """Branch cleaning via subsequent erosion of end points."""
  
  assert np.isfortran(skeleton);
  assert np.isfortran(skelton_copy);
  
  timer = tmr.Timer();
  timer_all = tmr.Timer();
  
  # find branch and end points
  deg = cpl.convolve_3d_indices(skeleton, t3d.n26, points, sink_dtype = 'uint8');
  branchpoints = points[deg >= 3];
  e_pts = points[deg == 1];
  
  if verbose:
    timer.printElapsedTime('Detected %d branch and %d endpoints' % (branchpoints.shape[0], e_pts.shape[0]));
    timer.reset();
  
  #prepare temps
  #skel = skeleton.copy();
  skel_flat = np.reshape(skelton_copy, -1, order = 'A');
  strides = np.array(skelton_copy.strides);
    
  
  if verbose:
    timer.printElapsedTime('Detected %d branch and %d endpoints' % (branchpoints.shape[0], e_pts.shape[0]));
    timer.reset();
  
  label = np.arange(27);
  label = label.reshape([3,3,3]);
  label[1,1,1] = 0;
  
  critical_points = [e_pts];
  delete_points = [];
  
  for l in range(1, length + 1):
    #neighbours of end points
    e_pts_label = cpl.convolve_3d_indices(skelton_copy, label, e_pts);
    
    if verbose:
      timer.printElapsedTime('Done labeling %d / %d' % (l, length));
      timer.reset();
    
    #label zero points are non-critical short isolated branches
    e_pts_zero = e_pts_label == 0;
    #print 'zero length:', np.unravel_index(e_pts[e_pts_zero], skel.shape)
    if e_pts_zero.sum() > 0:
      keep = np.logical_not(e_pts_zero);
      for m in range(l):
        critical_points[m] = critical_points[m][keep];
      e_pts_label = e_pts_label[keep];
      e_pts = e_pts[keep];
      
    if verbose:
      timer.printElapsedTime('Ignored %d small branches' % (keep.sum()));
      timer.reset();
    
    e_pts_new = e_pts + np.sum((np.vstack(np.unravel_index(e_pts_label, label.shape)) - 1).T * strides, axis = 1)
    
    # did we hit a branch point
    delete = np.in1d(e_pts_new, branchpoints); #, assume_unique = True);
    keep   = np.logical_not(delete);
    #print delete.shape, keep.shape, e_pts_new.shape
    
    #delete all path that hit a branch point
    if delete.sum() > 0:
      for m in range(l):
        delete_points.append(critical_points[m][delete]);
        #print 'deleting:', np.unravel_index(critical_points[m][delete], skel.shape)
        critical_points[m] = critical_points[m][keep];
      e_pts_new = e_pts_new[keep];
      
    if verbose:
      timer.printElapsedTime('Deleted %d points' % (delete.sum()));
      timer.reset();
    
    if l < length:
      skel_flat[e_pts] = False; # remove endpoints for new neighbour detection
      critical_points.append(e_pts_new);
      e_pts = e_pts_new;
      
    if verbose:
      timer.printElapsedTime('Cleanup iteration %d / %d done.' % (l, length));
    
  #gather all points
  if len(delete_points) > 0:
    delete_points = np.hstack(delete_points);
    delete_points = np.unique(delete_points);
  else:
    delete_points = np.zeros(0);
 
  if verbose:
    timer_all.printElapsedTime('Cleanup');
 
  if clean:
    skel_flat = np.reshape(skeleton, -1, order = 'F');
    skel_flat[delete_points] = False;
    keep_ids = np.logical_not(np.in1d(points, delete_points, assume_unique = True))
    points = points[keep_ids];
    radii  = radii[keep_ids];
    return skeleton, points, radii
  
  return delete_points;
    


###############################################################################
### Tests
###############################################################################

def _test():
  """Test"""
  pass
  #%%
#  import numpy as np
#  import ClearMap.Visualization.Plot3d as p3d;
#  import ClearMap.DataProcessing.ConvolvePointList as cpl;
#  import ClearMap.ImageProcessing.Skeletonization.SkeletonCleanUp as scu
#  reload(scu);
#  
#  data = np.load('/home/ckirst/Desktop/data.npy');
#  skel = np.load('/home/ckirst/Desktop/skel.npy');
#  points = np.load('/home/ckirst/Desktop/pts.npy');
#  
#  #data = data[:50,:50,:50];
#  #skel = skel[:50,:50,:50];
#  #t3d.deleteBorder(skel);
#  #points = np.where(np.reshape(skel,-1))[0];
#  skelfor = np.asarray(skel, order = 'F');
#  skelfor_copy = np.asarray(skel, order = 'F');
#  points = np.ravel_multi_index(np.where(skelfor), skelfor.shape, order = 'F');
#  
#  #%%
#  reload(scu);
#  
#  clean = scu.cleanOpenBranches(skelfor, skelfor_copy, points = points, length = 3, clean=False, verbose = True)
#  
#  skel_clean = np.zeros_like(skelfor, dtype = bool, order = 'A');
#  skel_clean_f = np.reshape(skel_clean, -1, order = 'A');
#  skel_clean_f[clean] = True;
#  
#  deg = cpl.convolve_3d_indices(skelfor, t3d.n26, points);
#  branchpoints = points[deg >= 3];
#  skel_3 = np.zeros_like(skelfor, dtype = bool, order = 'A');
#  skel_3_f = np.reshape(skel_3, -1, order = 'A');
#  skel_3_f[branchpoints] = True;
#  
#  endpoints = points[deg ==1];
#  skel_1 = np.zeros_like(skelfor, dtype = bool, order = 'A');
#  skel_1_f = np.reshape(skel_1, -1, order = 'A');
#  skel_1_f[endpoints] = True;
#  
#  #data_f = np.reshape(data, -1);   
#  #data_f[points] = 160; 
#  p3d.multi_plot([np.asarray(skel, dtype = int) + 2 * skel_3 + 4 * skel_1, 
#              np.asarray(skel, dtype = int) + 2 * skel_clean, minMax = [0,7]]);
#  





#%%


#clean2 = scu.cleanOpenBranches2(skel.copy(), points = points, length = 3, clean=False, verbose = True)
#  
#  skel_clean2 = np.zeros_like(skelfor, dtype = bool, order = 'A');
#  skel_clean2_f = np.reshape(skel_clean2, -1, order = 'A');
#  skel_clean2_f[clean2] = True;
#def cleanOpenBranches2(skeleton, points, length, clean = False, processes = cpu_count(), verbose = False):
#    """Remove open branches of length smaller than sepcifed
#    
#    Arguments
#    ---------
#      skeleton : array
#          binary 3d image of skeleton.
#      points : array
#          flat indices of non-zero entries in skeleton
#      length : int
#          maximal branch length to remove
#
#    Returns
#    -------
#      cleaned : array
#          flat indices of cleaned skeleton 
#    """
#    timer = tmr.Timer();
#    timer_all = tmr.Timer();
#
#    #prepare
#    kernel = np.ones((3,3,3), dtype = bool);
#    kernel[1,1,1] = False;
#    deg = cpl.convolve_3d_indices(skeleton, kernel, points);
#    
#    if verbose:
#      timer.printElapsedTime('Degree calcualtion');
#      timer.reset();
#    
#    branchpoints = deg >= 3;
#    branchpoints = branchpoints.view(dtype = 'uint8');
#    
#    if verbose:
#      timer.printElapsedTime('Branch points');
#      timer.reset();
#      
#    endpoints = np.where(deg == 1)[0];
#    
#    if verbose:
#      timer.printElapsedTime('Branch and end point detection');
#      timer.reset();
#
#    
#    keep = code.cleanBranchesIndex(points, strides = np.array(skeleton.strides), 
#                                   length = length, 
#                                   startpoints = endpoints, stoppoints = branchpoints, 
#                                   processes = processes); 
#    keep = keep.view(dtype = bool);
#    delete = points[np.logical_not(keep)];
#    
#    if verbose:
#      timer.printElapsedTime('Cleanup detection');   
#    
#    if clean:
#      skeleton_flat = np.reshape(skeleton, -1);
#      skeleton_flat[delete] = False;
#      delete = skeleton, delete;
#
#    if verbose:
#      timer_all.printElapsedTime('Cleanup');
#      
#    return delete;


#%%
#  
#  ss = con.extractNeighbourhood(skel, [84, 125, 58], 5);
#  ss = t3d.deleteBorder(ss);
#  pp = np.where(np.reshape(ss, -1))[0];
#  
#  ee, ii = con.findEndpoints(ss, pp, border = None)
#  
#  img = np.asarray(ss, dtype = int);
#  imgf = np.reshape(img, -1);
#  imgf[pp] = cpl.convolve_3d)indices(ss, t3d.n26, pp)
#  
#  ee_xyz = np.array(np.unravel_index(ee, ss.shape)).T;
#  pp_xyz = np.array(np.unravel_index(pp, ss.shape)).T;
#  
#  kernel = np.ones((3,3,3), dtype = bool);
#  kernel[1,1,1] = False;
#  deg = cpl.convolve3DIndex(ss, kernel, pp);
#  branchpoints = deg >= 3;
#  
#  s2, p2 = scu.cleanOpenBranches(ss, pp, length = 1);
#  
#  dv.dualPlot(img, s2)
  
  
