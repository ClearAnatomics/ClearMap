# -*- coding: utf-8 -*-
"""
Module to Postprocess skeletonized vasculature data
"""
__author__    = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__   = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__copyright__ = 'Copyright 2017 by Christoph Kirst, The Rockefeller University, New York City'


import numpy as np

import scipy.ndimage as ndi

#TODO:
import ClearMap.ParallelProcessing.DataProcessing.ArrayProcessing as ap
import ClearMap.ParallelProcessing.DataProcessing.ConvolvePointList as cpl

import ClearMap.ImageProcessing.Tracing.Trace as trc
import ClearMap.ImageProcessing.Differentiation.Hessian as cur

import ClearMap.ImageProcessing.Topology.Topology3d as t3d

import ClearMap.Visualization.Plot3d as p3d


def extractNeighbourhood(data, center, radius):
  """Extract local neighborhood with specific radius, if to close to border pad with zeros"""
  order = 'C';
  if np.isfortran(data):
    order = 'F';
  
  if isinstance(radius, int):
    radius = (radius,);
  radius = radius * 3;
  radius = np.array(radius[:3]);
  
  dlo = [max(0, c-r) for c,r in zip(center, radius)];
  dhi = [min(s, c+r+1) for c,r,s in zip(center, radius, data.shape)];
  
  nlo = [-min(0, c-r) for c,r in zip(center, radius)];
  nhi = [2*r+1 + min(0, s-c-r-1) for c,r,s in zip(center, radius, data.shape)];
  
  #print center
  #print radius
  #print dlo, dhi
  #print nlo, nhi  
  
  nbh = np.zeros(2*radius + 1, dtype = data.dtype, order = order);
  nbh[nlo[0]:nhi[0],nlo[1]:nhi[1],nlo[2]:nhi[2]] = data[dlo[0]:dhi[0],dlo[1]:dhi[1],dlo[2]:dhi[2]];
  
  return nbh;


def findEndpoints(skel, points, border = None):
  """Find endpoints in skeleton to try to reconnect"""
  order = 'C';
  if np.isfortran(skel):
    order = 'F'; 
  
  #find node degrees
  deg = cpl.convolve_3d_indices(skel, t3d.n26, points, out_dtype = 'uint8')
  
  #isolated and end points
  isolated = points[deg == 0];
  ends = points[deg == 1];
  
  # remove border points
  if border is not None:
    if not isinstance(border, tuple):
      border = (border,);
    border = border * 3;
    border = border[:3];
  
    x,y,z = np.unravel_index(isolated, skel.shape, order = order)
    ids = np.ones(isolated.shape, dtype = bool);
    for xx,ss,bb in zip((x,y,z), skel.shape, border):
      ids = np.logical_and(ids, np.logical_and(bb < xx, xx < ss - bb));
    isolated = isolated[ids];
    
    x,y,z = np.unravel_index(ends, skel.shape, order = order)
    ids = np.ones(ends.shape, dtype = bool);
    for xx,ss,bb in zip((x,y,z), skel.shape, border):
      ids = np.logical_and(ids, np.logical_and(bb < xx, xx < ss - bb));
    ends = ends[ids];
  
  return ends, isolated


def addPathToMask(mask, path, value = True):
  order = 'C';
  if np.isfortran(mask):
    order = 'F'; 
  m = np.reshape(mask, -1, order = order);
  ids = np.ravel_multi_index(path.T, mask.shape, order = order);
  m[ids] = value;
  #return mask;


def addDilatedPathToMask(mask, path, iterations = 1):
  order = 'C';
  if np.isfortran(mask):
    order = 'F'; 
  ids = np.ravel_multi_index(path.T, mask.shape, order = order);
  m = np.zeros(mask.shape, dtype = bool, order = order);
  mf = np.reshape(m, -1, order = order);
  mf[ids] = True;
  m = ndi.morphology.binary_dilation(m, structure = np.ones((3,3,3), dtype = bool), iterations = iterations);
  m = np.asarray(m, order = order);
  #return np.logical_or(mask, m);


viewer = None;

def plotData(data, skel, binary, ends = None, isolated = None, replot = True):
  global viewer
  
  img = np.zeros(data.shape, dtype = int);
  img[:] = img + binary + skel;
  
  if ends is not None:
    xyz = np.array(np.unravel_index(ends, skel.shape, order = 'C')).T;
    for x,y,z in xyz:
      img[x,y,z] = 3;
  
  if isolated is not None:  
    xyz = np.array(np.unravel_index(isolated, skel.shape, order = 'C')).T;
    for x,y,z in xyz:
      img[x,y,z] = 4;
  
  if replot:
    try:
      viewer[0].setSource(data);
      viewer[1].setSource(data);
    except:
      viewer = p3d.plot([data, img]);
  else:
    return p3d.plot([data, img]);

 
def plotTracingResult(path, data_nbh, mask_nbh, center, radius, tubeness_nbh, skeleton = None, distance_nbh = None):
  global viewer;
  
  img_nbh = np.asarray(data_nbh, dtype = float);
  bin_nbh = np.zeros(mask_nbh.shape, dtype = int);   
  
  if skeleton is not None:
    bin_nbh += extractNeighbourhood(skeleton, center, radius); 

  addPathToMask(bin_nbh, path, value = 4);
  if points is not None:
    pts = points - center + radius;
    for d in range(3):
      ids = np.logical_and(0 <= pts[:,d], pts[:,d] < data_nbh.shape[d]);
      pts = pts[ids];
    addPathToMask(bin_nbh, pts, value = 3);
  
  addPathToMask(img_nbh, path, value = 512);
  addPathToMask(tubeness_nbh, path, value = 50);
  if distance_nbh is not None:
    addPathToMask(distance_nbh, path, value = 50);
  
  try:
    viewer[0].setSource(bin_nbh);
    viewer[1].setSource(img_nbh);
    viewer[2].setSource(tubeness_nbh);
    if distance_nbh is not None:
      viewer[3].setSource(distance_nbh);
  except:
    if distance_nbh is not None:
      viewer = p3d.plot([bin_nbh, img_nbh, tubeness_nbh, distance_nbh])
    else:
      viewer = p3d.plot([bin_nbh, img_nbh, tubeness_nbh])








def tracePointToMask(data, mask, center, radius, points = None, plot = False, skeleton = None, tubeness = None, 
                     removeLocalMask = True,
                     maxSteps = 500, verbose = False, **trace_parameter):
  """Trace an endpoint to a mask"""
    
  # cut out neighbourhood from data
  center_nbh = np.array(center) - center + radius; 
  data_nbh = extractNeighbourhood(data, center, radius);
  mask_nbh = extractNeighbourhood(mask, center, radius);
  
  if tubeness is None:
    tubeness_nbh = cur.tubeness(ndi.gaussian_filter(np.asarray(data_nbh, dtype = float), sigma = 1.0));
  else:
    tubeness_nbh = extractNeighbourhood(tubeness, center, radius);
  
  
  if removeLocalMask is not None:
    mask_nbh_label, mask_label = ndi.label(mask_nbh, structure = np.ones((3,3,3), dtype = bool));
    ids = mask_nbh_label[tuple(center_nbh)] == mask_nbh_label;
    mask_nbh[ids] = False;
  
  distance_nbh = ndi.distance_transform_edt(np.logical_not(mask_nbh))
  
  path, quality = trc.traceToMask(np.asarray(data_nbh, dtype = float), tubeness_nbh, center_nbh, distance_nbh, 
                                  maxSteps = maxSteps, verbose = verbose, returnQuality = True, **trace_parameter);
  
  if verbose:
    if len(path) > 0:
      print('Path of length = %d with quality = %f (per length = %f)' % (len(path), quality, quality / len(path)));
    else:
      print('No path found!');
  
  if plot:
    plotTracingResult(path, data_nbh, mask_nbh, center, radius, tubeness_nbh, skeleton = skeleton, distance_nbh = distance_nbh);
  
  return path + center - center_nbh, quality;


def tracePointToNeighbor(data, mask, center, neighbor, radius, points = None, plot = False, skeleton = None, tubeness = None, 
                         removeLocalMask = True,
                         maxSteps = 500, verbose = False, **trace_parameter):
  """Trace an endpoint to a neighbour"""
  # cut out neighbourhood from data
  center_nbh = np.array(center) - center + radius; 
  neighbor_nbh = np.array(neighbor) - center + radius;   
  
  data_nbh = extractNeighbourhood(data, center, radius);
  mask_nbh = extractNeighbourhood(mask, center, radius);
  
  if tubeness is None:
    tubeness_nbh = cur.tubeness(ndi.gaussian_filter(np.asarray(data_nbh, dtype = float), sigma = 1.0));
  else:
    tubeness_nbh = extractNeighbourhood(tubeness, center, radius);
  
  if removeLocalMask is not None:
    mask_nbh_label, mask_label = ndi.label(mask_nbh, structure = np.ones((3,3,3), dtype = bool));
    ids = mask_nbh_label[tuple(center_nbh)] == mask_nbh_label;
    mask_nbh[ids] = False;
  
  #distance_nbh = ndi.distance_transform_edt(np.logical_not(mask_nbh))
  
  path, quality = trc.trace(np.asarray(data_nbh, dtype = float), tubeness_nbh, 
                   center_nbh, neighbor_nbh, 
                   maxSteps = maxSteps, verbose = verbose, returnQuality = True, **trace_parameter);
  
  if verbose:
    if len(path) > 0:
      print('Path of length = %d with quality = %f (per length = %f)' % (len(path), quality, quality / len(path)));
    else:
      print('No path found!');
  
  if plot:
    plotTracingResult(path, data_nbh, mask_nbh, center, radius, tubeness_nbh, skeleton = skeleton);
  
  return path + center - center_nbh, quality;


def connectPoint(data, mask, endpoints, start_index, radius, 
                 tubeness = None, min_quality = None, remove_local_mask = True,
                 skeleton = None,
                 verbose = False, **trace_parameter):
  """Tries to connect an end point"""
  
  #outine:
  # find neighbour end points and try to connect to nearest one
  # if path score good enough add path and remove two endpoints
  # else try to connect to binarized image
  # if path score good enugh connect to closest skeleton point
  # else not connectable
  
  #assumes everything is in fotran order
  strides = np.array(data.strides) / data.itemsize;
  shape = data.shape;
  #print strides, shape
  
  center_flat = endpoints[start_index];
  center_xyz  = np.array(np.unravel_index(center_flat, data.shape, order = 'F'));
  
  mask_nbh = extractNeighbourhood(mask, center_xyz, radius);
  data_nbh = np.asarray(extractNeighbourhood(data, center_xyz, radius), dtype = float, order = 'F');
  shape_nbh = mask_nbh.shape;
  
  center_nbh_xyz = np.zeros(3, dtype = int) + radius; 
  #center_nbh_flat = np.ravel_multi_index(center_nbh_xyz, shape_nbh, order = 'F');
  
  if tubeness is None:
    tubeness_nbh = cur.tubeness(ndi.gaussian_filter(np.asarray(data_nbh, dtype = float), sigma = 1.0));
    tubeness_nbh = np.asarray(tubeness_nbh, order = 'F');
  else:
    tubeness_nbh = extractNeighbourhood(tubeness, center_xyz, radius);
  
  mask_nbh_label = np.empty(shape_nbh, dtype = 'int32', order = 'F');
  _ = ndi.label(mask_nbh, structure = np.ones((3,3,3), dtype = bool), output = mask_nbh_label);
  local_nbh = mask_nbh_label[tuple(center_nbh_xyz)] == mask_nbh_label;  
  
  # end point neighbours
  nbs_flat = ap.findNeighbours(endpoints, start_index, shape, strides, radius);
  
  if len(nbs_flat) > 0:
    nbs_nbh_xyz  = np.vstack(np.unravel_index(nbs_flat, shape, order = 'F')).T - center_xyz + center_nbh_xyz;
    nbs_nbh_flat = np.ravel_multi_index(nbs_nbh_xyz.T, shape_nbh, order = 'F');
    
    # remove connected neighbours
    non_local_nbh_flat = np.reshape(np.logical_not(local_nbh), -1, order = 'F');
    nbs_nbh_non_local_flat = nbs_nbh_flat[non_local_nbh_flat[nbs_nbh_flat]];
    
    if len(nbs_nbh_non_local_flat) > 0:
      #find nearest neighbour
      nbs_nbh_non_local_xyz  = np.vstack(np.unravel_index(nbs_nbh_non_local_flat, shape, order = 'F')).T;
      
      nbs_nbh_non_local_dist = nbs_nbh_non_local_xyz - center_nbh_xyz;
      nbs_nbh_non_local_dist = np.sum(nbs_nbh_non_local_dist * nbs_nbh_non_local_dist, axis = 1);
      
      neighbor_nbh_xyz = nbs_nbh_non_local_xyz[np.argmin(nbs_nbh_non_local_dist)];
      
      path, quality = trc.trace(data_nbh, tubeness_nbh, 
                                center_nbh_xyz, neighbor_nbh_xyz, 
                                verbose = False, returnQuality = True, **trace_parameter);
      
      if len(path) > 0:
        if quality / len(path) < min_quality:
          if verbose:
            print('Found good path to neighbour of length = %d with quality = %f (per length = %f) [%d / %d nonlocal neighbours]' % (len(path), quality, quality / len(path), len(nbs_nbh_non_local_flat), len(nbs_flat)));  #print path
          return path + center_xyz - center_nbh_xyz, quality;
        else:
          if verbose:
            print('Found bad  path to neighbour of length = %d with quality = %f (per length = %f) [%d / %d nonlocal neighbours]' % (len(path), quality, quality / len(path), len(nbs_nbh_non_local_flat), len(nbs_flat)));  #print path
      else:
        if verbose:
          print('Found no path to neighbour [%d / %d nonlocal neighbours]' % (len(nbs_nbh_non_local_flat), len(nbs_flat)));  #print path
    
    
  #tracing to neares neighbour failed    
  if verbose:
    print('Found no valid path to neighbour, now tracing to binary!');  #print path

  # Tracing to next binary
  if remove_local_mask:
    mask_nbh[local_nbh] = False; 
  
  distance_nbh = ndi.distance_transform_edt(np.logical_not(mask_nbh))
  distance_nbh = np.asarray(distance_nbh, order = 'F');
  
  path, quality = trc.traceToMask(data_nbh, tubeness_nbh, center_nbh_xyz, distance_nbh, 
                                  verbose = False, returnQuality = True, **trace_parameter);
  
  if len(path) > 0:
    if quality / len(path) < min_quality:
      if verbose:
        print('Found good path to binary of length = %d with quality = %f (per length = %f)' % (len(path), quality, quality / len(path)));  #print path
            
      # trace to skeleton
      if skeleton is not None:
        #find closest point on skeleton
        final_xyz = path[0];
        skeleton_nbh = extractNeighbourhood(skeleton, center_xyz, radius);
        local_end_path_nbh = mask_nbh_label[tuple(final_xyz)] == mask_nbh_label;
        skeleton_nbh_dxyz = np.vstack(np.where(np.logical_and(skeleton_nbh, local_end_path_nbh))).T - final_xyz;
        if len(skeleton_nbh_dxyz) == 0: # could not find skeleton nearby -> give up for now
          return path + center_xyz - center_nbh_xyz, quality;
        
        skeleton_nbh_dist = np.sum(skeleton_nbh_dxyz * skeleton_nbh_dxyz, axis = 1);
        closest_dxyz =  skeleton_nbh_dxyz[np.argmin(skeleton_nbh_dist)];
        closest_xyz = closest_dxyz + final_xyz;
        #print path[0], path[-1]
        #print center_nbh_xyz, closest_dxyz
        
        #generate pixel path
        max_l = np.max(np.abs(closest_dxyz)) + 1;
        path_add_xyz  = np.vstack([np.asarray(np.linspace(f, c, max_l), dtype = int) for f,c in zip(final_xyz, closest_xyz)]).T;
        path_add_flat = np.ravel_multi_index(path_add_xyz.T, shape_nbh);
        _, ids = np.unique(path_add_flat, return_index = True);
        path_add_xyz = path_add_xyz[ids];
        #print path_add_xyz;
        path = np.vstack([path, path_add_xyz]); # note: this is not an ordered path anymore!
      
      return path + center_xyz - center_nbh_xyz, quality;
    else:
      if verbose:
        print('Found bad  path to binary of length = %d with quality = %f (per length = %f)' % (len(path), quality, quality / len(path)));  #print path
   
  if verbose:
    print('Found no valid path to binary!');
    
  return np.zeros((0,3)), 0;


import ClearMap.ParallelProcessing.SharedMemoryManager as smm;
#import ClearMap.ParallelProcessing.SharedMemoryProcessing as smp;

import multiprocessing as mp

import tempfile as tmpf


def order(array):
  if np.isfortran(array):
    return 'F';
  else:
    return 'C';


def processSingleConnection(args):
  global temporary_folder;
  i = args;
  
  #steps_done += 1;
  #if steps_done % 100 == 0:
  #  print('Processing step %d / %d...' % (steps_done, steps_total));
  if i % 100 == 0:
    print('Processing step %d...' % i);

  try:  
    fid = mp.current_process()._identity[0];
  except: # mostlikely sequential mode
    fid = 0;
  
  data = smm.get(0);
  mask = smm.get(1);
  skel = smm.get(2);
  spts = smm.get(3);
  
  res = connectPoint(data, mask, spts, i, skeleton = skel, 
                     radius = 20, 
                     tubeness = None, remove_local_mask = True, 
                     min_quality = 15.0,
                     verbose = False,
                     maxSteps = 12000, costPerDistance = 1.0);
  #return res;
  
  path, score = res;
  if len(path) > 0: 
    #convert path to indices
    path_flat = np.ravel_multi_index(path.T, data.shape, order = order(data));
     
    #append to file
    try:
      fid = mp.current_process()._identity[0];
    except:
      fid = 0; # sequential mode
    fn  = '%s/path_%03d.tmp' % (temporary_folder, fid);
    #print fn;
    
    fb = open(fn,"ab");
    path_flat.tofile(fb);
    fb.close();                     
   
  return;


import os;
import shutil;
import ClearMap.Utils.Timer as tmr

import gc

temporary_folder = None;  

def addConnections(data, mask, skeleton, points, radius = 20, 
                   start_points = None, 
                   remove_local_mask = True, min_quality = 15.0,
                   add_to_skeleton = True, add_to_mask = False,
                   verbose = True, processes = mp.cpu_count(), block_size = 5000, debug = False):
  global temporary_folder;
  
  timer = tmr.Timer();
  timer_total = tmr.Timer();
  
  if start_points is None:
    ends, isolated = findEndpoints(skeleton, points, border = 20);
    start_points = np.hstack([ends, isolated]);
    start_points = smm.asShared(start_points);
  npts = len(start_points);
  
  if verbose:
    timer.printElapsedTime('Found %d endpoints' % (npts,));
    timer.reset();
  
  assert smm.isShared(data);
  assert smm.isShared(mask);
  assert smm.isShared(skeleton);
  #steps_total = npts / processes;
  #steps_done = 0;
  
  smm.clean();
  data_hdl = smm.insert(data);
  mask_hdl = smm.insert(mask);
  skel_hdl = smm.insert(skeleton);
  spts_hdl = smm.insert(start_points);
  if not (data_hdl == 0 and mask_hdl == 1 and skel_hdl == 2 and spts_hdl == 3):
    raise RuntimeError('The shared memory handles are invalid: %d, %d, %d, %d' % (data_hdl, mask_hdl, skel_hdl, spts_hdl));
  
  #generate temporary folder to write path too
  temporary_folder = tmpf.mkdtemp();
  
  if verbose:
    timer.printElapsedTime('Preparation of %d connections' % (npts,));
    timer.reset();
    
  # process in parallel / block processing is to clean up memory leaks for now
  nblocks = max(1, int(np.ceil(1.0 * npts / processes / block_size)));
  ranges = np.asarray(np.linspace(0, npts, nblocks+1), dtype = int);
  #nblocks = 1;
  #ranges = [0, 100];
  for b in range(nblocks):
    argdata = np.arange(ranges[b], ranges[b+1]);
    if debug:
      result = [processSingleConnection(a) for a in argdata];
    else:
      #mp.process.current_process()._counter = mp.process.itertools.count(1); #reset worker counter
      pool = mp.Pool(processes = processes);
      #for i,_ in enumerate(pool.imap_unordered(processSingleConnection, argdata)):
      #  if i % 100 == 0:
      #    timer.printElapsedTime('Iteration %d / %d' % (i + ranges[b], npts));
      pool.map(processSingleConnection, argdata)
          
      pool.close();
      pool.join();
    gc.collect();
  
  smm.free(data_hdl);
  smm.free(mask_hdl);
  smm.free(skel_hdl);
  smm.free(spts_hdl);
  
  if verbose:
    timer.printElapsedTime('Processing of %d connections' % (npts,));
    timer.reset();

  #return result;

  #get paths from temporay files
  result = [];
  for f in os.listdir(temporary_folder):
    result.append(np.fromfile(os.path.join(temporary_folder,f), dtype = int));
  result = np.hstack(result);
  
  #clean up
  shutil.rmtree(temporary_folder);
  temporary_folder = None;
  
  if verbose:
    timer.printElapsedTime('Loading paths with %d points' % (len(result),));
    timer.reset();
  
  #add dilated version to skeleton
  if add_to_skeleton:
    skeleton_f = np.reshape(skeleton, -1, order = 'A');
    strides = skeleton.strides;
    skeleton_f_len = len(skeleton_f);
    for dx in [-1,0,1]:
      for dy in [-1,0,1]:
        for dz in [-1,0,1]:
          offset= dx * strides[0] + dy * strides[1] + dz * strides[2];
          pos = result + offset;
          pos = pos[np.logical_and(pos >= 0, pos < skeleton_f_len)];
          skeleton_f[pos] = True;
  
    if verbose:
      timer.printElapsedTime('Added paths with %d points to skeleton' % (len(result),));
      #add dilated version to skeleton
  
  if add_to_mask:
    mask_f = np.reshape(mask, -1, order = 'A');
    strides = mask.strides;
    for dx in [-1,0,1]:
      for dy in [-1,0,1]:
        for dz in [-1,0,1]:
          offset= dx * strides[0] + dy * strides[1] + dz * strides[2];
          pos = result + offset;
          pos = pos[np.logical_and(pos >= 0, pos < skeleton_f_len)];
          mask_f[pos] = True;
  
    if verbose:
      timer.printElapsedTime('Added paths with %d points to mask' % (len(result),));  
      
      
    timer_total.printElapsedTime('Connection post-processing added %d points' % (len(result),));
  
  return result;


def _test():
  #%%
  import numpy as np
  import scipy.ndimage as ndi
  import ClearMap.DataProcessing.LargeData as ld
  import ClearMap.Visualization.Plot3d as p3d
  import ClearMap.DataProcessing.ConvolvePointList as cpl
  import ClearMap.ImageProcessing.Skeletonization.Topology3d as t3d
  import ClearMap.ImageProcessing.Skeletonization.SkeletonCleanUp as scu
  
  import ClearMap.ImageProcessing.Tracing.Connect as con;
  reload(con);  
  
  data   = np.load('/home/ckirst/Desktop/data.npy');
  binary = np.load('/home/ckirst/Desktop/binarized.npy');
  skel   = np.load('/home/ckirst/Desktop/skel.npy');
  #points = np.load('/home/ckirst/Desktop/pts.npy');
  
  data   = np.copy(data,    order = 'F');
  binary = np.copy(binary,  order = 'F');
  skel   = np.copy(skel,    order = 'F');
  skel_copy = np.copy(skel, order = 'F');
  points = np.ravel_multi_index(np.where(skel), skel.shape, order = 'F');
  
  skel, points = scu.cleanOpenBranches(skel, skel_copy, points, length = 3, clean = True);
  deg = cpl.convolve3DIndex(skel, t3d.n26, points);
  
  ends, isolated = con.findEndpoints(skel, points, border = 25);
  special = np.sort(np.hstack([ends, isolated]))
  
  ends_xyz = np.array(np.unravel_index(ends, data.shape, order = 'F')).T;
  isolated_xyz = np.array(np.unravel_index(isolated, data.shape, order = 'F')).T;
  special_xyz = np.vstack([ends_xyz, isolated_xyz]);
  
  
  #%%
  import ClearMap.ParallelProcessing.SharedMemoryManager as smm
  data_s   = smm.asShared(data, order = 'F');
  binary_s = smm.asShared(binary.view('uint8'), order = 'F');
  skel_s   = smm.asShared(skel.view('uint8'),   order = 'F');
  
  smm.clean();
  res = con.addConnections(data_s, binary_s, skel_s, points, radius = 20, 
                           start_points = None,
                           add_to_skeleton=True, add_to_mask = True,
                           verbose = True, processes = 4, debug = False, block_size=10)
  
  skel_s = skel_s.view(bool);
  binary_s = binary_s.view(bool);

  #%%
  mask_img = np.asarray(binary, dtype= int, order = 'A'); 
  mask_img[:] = mask_img + binary_s;
  mask_img[:] = mask_img + skel;
  
  data_img = np.copy(data, order = 'A');
  data_img[skel] = 120;
  
  mask_img_f = np.reshape(mask_img, -1, order = 'A');
  data_img_f = np.reshape(data_img, -1, order = 'A');
  
  mask_img_f[res] = 7;
  data_img_f[res] = 512;
  
  mask_img_f[special] = 8;
  data_img_f[special] = 150;
  
  for d in [3,4,5]:
    mask_img_f[points[deg == d]] = d+1;
  
  try:
    con.viewer[0].setSource(mask_img);
    con.viewer[1].setSource(data_img);
  except:
    con.viewer = p3d.plot([mask_img, data_img]);
  
  con.viewer[0].setMinMax([0, 8]);
  con.viewer[1].setMinMax([24, 160]);          

  
  #%%
  mask = binary; 
  data_new = np.copy(data, order = 'A');
  data_new[skel] = 120;
  
  skel_new = np.asarray(skel, dtype= int, order = 'A');
  skel_new[:] = skel_new + binary;
  
  binary_new = np.copy(binary, order = 'A');
  qs = [];
  for i,e in enumerate(special):
    print('------');
    print('%d / %d'  % (i, len(special)));
    path, quality = con.connectPoint(data, mask, special, i, radius = 25,
                                     skeleton = skel, 
                                     tubeness = None, remove_local_mask = True, 
                                     min_quality = 15.0,
                                     verbose = True,
                                     maxSteps = 15000, costPerDistance = 1.0);

    
    #print path, quality
    if len(path) > 0:
      qs.append(quality * 1.0/len(path));      
      
      q = con.addPathToMask(skel_new, path, value = 7);
      q = con.addPathToMask(data_new, path, value = 512);
      binary_new = con.addDilatedPathToMask(binary_new, path, iterations = 1);
 
  skel_new[:] = skel_new + binary_new;
  q = con.addPathToMask(skel_new, special_xyz, value = 6);
  for d in [3,4,5]:
    xyz = np.array(np.unravel_index(points[deg == d], data.shape, order = 'F')).T;
    q = con.addPathToMask(skel_new, xyz, value = d);  
  q = con.addPathToMask(data_new, special_xyz, value = 150);
    
  try:
    con.viewer[0].setSource(skel_new);
    con.viewer[1].setSource(data_new);
  except:
    con.viewer = p3d.plot([skel_new, data_new]);
  
  con.viewer[0].setMinMax([0, 8]);
  con.viewer[1].setMinMax([24, 160]);          
  
    

  
  #%%
  import matplotlib.pyplot as plt;
  plt.figure(1); plt.clf();
  #plt.plot(qs);
  plt.hist(qs)
  
  
  
  
  
  #%%
  i = 20;
  i = 21;
  i = 30;
  i = 40;
  r = 25;
  center = np.unravel_index(ends[i], data.shape)
  print(center, data.shape)
  mask = binary; 
  path = con.tracePointToMask(data, mask, center, radius= r, points = special_xyz, 
                              plot = True, skel = skel, binary = binary, 
                              tubeness = None, removeLocalMask = True, maxSteps = None, verbose = False,
                              costPerDistance = 0.0);
       

  #%%

  nbs = ap.findNeighbours(ends, i, skel.shape, skel.strides, r);
  center = np.unravel_index(ends[i], skel.shape)

  nbs_xyz = np.array(np.unravel_index(nbs, skel.shape)).T
  dists = nbs_xyz - center;
  dists = np.sum(dists*dists, axis = 1);
  
  nb = np.argmin(dists)
    
  center = np.unravel_index(ends[i], data.shape)
  print(center, data.shape)
  mask = binary; 
  path = con.tracePointToNeighbor(data, mask, center, nbs_xyz[nb],
                                  radius= r, points = special_xyz, 
                                  plot = True, skel = skel, binary = binary, 
                                  tubeness = None, removeLocalMask = True, maxSteps = None, verbose = False,
                                  costPerDistance = 0.0);


  #%%

  import ClearMap.ImageProcessing.Filter.FilterKernel as fkr;
  dog = fkr.filterKernel('DoG', size = (13,13,13));
  dv.plot(dog)
  
  data_filter = ndi.correlate(np.asarray(data, dtype = float), dog);
  data_filter -= data_filter.min();
  data_filter = data_filter / 3.0;
  #dv.dualPlot(data, data_filter);
                       
  #%%add all paths
  reload(con)                              
                              
  r = 25;
  mask = binary; 
  data_new = data.copy();
  data_new[skel] = 120;
  
  skel_new = np.asarray(skel, dtype= int);
  skel_new = skel_new + binary;
  
  binary_new = binary.copy();
  
  for i,e in enumerate(special):
    center = np.unravel_index(e, data.shape);
    
    print(i, e, center)
    path   = con.tracePointToMask(data, mask, center, radius= r, points = special_xyz, 
                                  plot = False, skel = skel, binary = binary, 
                                  tubeness = None, removeLocalMask = True, maxSteps = 15000,
                                  costPerDistance = 1.0);
    
    q = con.addPathToMask(skel_new, path, value = 7);
    q = con.addPathToMask(data_new, path, value = 512);
    binary_new = con.addDilatedPathToMask(binary_new, path, iterations = 1);
 
  q = con.addPathToMask(skel_new, special_xyz, value = 6);
  for d in [3,4,5]:
    xyz = np.array(np.unravel_index(points[deg == d], data.shape)).T;
    q = con.addPathToMask(skel_new, xyz, value = d);  
  q = con.addPathToMask(data_new, special_xyz, value = 150);
  
  skel_new = skel_new + binary_new;
  try:
    con.viewer[0].setSource(skel_new);
    con.viewer[1].setSource(data_new);
  except:
    con.viewer = dv.dualPlot(skel_new, data_new);
  
  con.viewer[0].setMinMax([0, 8]);
  con.viewer[1].setMinMax([24, 160]);                
    
  
  #%%
  
  import ClearMap.ImageProcessing.Skeletonization.Skeletonize as skl
  
  skel_2 = skl.skeletonize3D(binary_new.copy());
  
  
  #%%
  
  np.save('/home/ckirst/Desktop/binarized_con.npy', binary_new)
  #%%
  
  # write image
  
  import ClearMap.IO.IO as io
  
  #r = np.asarray(128 * binary_new, dtype = 'uint8');
  #g = r.copy(); b = r.copy();
  #r[:] = r + 127 * skel_2[0];
  #g[:] = g - 128 * skel_2[0];
  #b[:] = b - 128 * skel_2[0];
  #img = np.stack((r,g,b), axis = 3)
  
  img = np.asarray(128 * binary_new, dtype = 'uint8');
  img[:] = img + 127 * skel_2[0];
  
  io.writeData('/home/ckirst/Desktop/3d.tif', img)
  
  
  #%%


  #%%

  

  
#  if removeLocalMask is not False:
#    if not removeLocalMask is True:
#      if not isinstance(removeLocalMask, tuple):
#        removeLocalMask = (removeLocalMask,);
#      removeLocalMask = removeLocalMask*3;
#      removeLocalMask = removeLocalMask[:3];
#      local_mask = extractNeighbourhood(mask_nbh, center_nbh, removeLocalMask);
#      sl = tuple([slice(c - r, c + r) for c,r in zip(center_nbh, removeLocalMask)]);
#    else:
#      local_mask = mask_nbh;
#      sl = (slice(None),) * 3;
#    
#    mask_nbh_label, mask_label = ndi.label(local_mask, structure = np.ones((3,3,3), dtype = bool));
#    ids = local_mask_label[tuple(center_nbh)] == local_mask_label;
#    
#    mask_nbh[ids] = False;
#    
#    if removeLocalTubness is not None:
#
#      
#      ker = sel.
#      ids = np.logical_and(ids, )
#      tubeness_nbh[sls] = ;
