# -*- coding: utf-8 -*-
"""
Tracking
========

Simple particle tracker based on liner programming.

Note
----
This module is used by :mod:`~ClearMap.Alignment.Stitching.StitchingWobbly`
to trace wobbly stacks.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


import numpy as np

import scipy.spatial.distance as ssd


###############################################################################
### Tracker
###############################################################################

def track_positions(positions, new_trajectory_cost = None, cutoff = None):
  """Track positions of multiple particles.
  
  Arguments
  ---------
  positionss : array
    The particle positions as a list.
  new_trajectory_cost : float or None
    The cost for a new trajectory, if None, maximal distance+1
  cutoff : float or None:
    The maximal distance allowed to connect particles.
     
  Returns
  -------
  trajectories : list
    The trajectories as a list of list of (time,particle) tuples.
    
  References
  ----------
  A shortest augmenting path algorithm for dense and sparse linear assignment problems
  Jonker, R, Volgenant, A, Computing 1987  
  """
  
  #positions: list of list of tuples of of minima [[(min_x1,1,min_y1,2,...),...], [(min_x2,1,...),...],...])
  n_steps = len(positions)-1;
  
  #match steps
  matches = [match(positions[i], positions[i+1], new_trajectory_cost=new_trajectory_cost, cutoff=cutoff) for i in range(n_steps)];
  
  #build trajectories
  n_pre = len(positions[0]);
  trajectories = [[(0,i)] for i in range(n_pre)];
  active = [i for i in range(n_pre)];                          
  for t in range(1, n_steps+1):   
    #n_pre = len(positions[t]);
    n_post = len(positions[t]);
     
    pre = [trajectories[a][-1][1] for a in active];
    m = matches[t-1];
    #if len(active) != len(positions[t-1]):
    #  raise ValueError()
             
    post = [m[p] for p in pre];
                        
    p_post = [];
    p_end = [];              
    for a, p in enumerate(post):
      if p == n_post:
        p_end.append(a);
      else: # extend trajectory
        trajectories[active[a]].append((t,p));
        p_post.append(p)
    
    #end trajectories
    for i in sorted(p_end, reverse=True):
      del active[i];
     
    #start new one
    new = np.setdiff1d(range(n_post), p_post);                        
    for p in new:
      active.append(len(trajectories));
      trajectories.append([(t,p)])
      
    #print '-------'
    #print 'pre=%d,%r, post=%d,%r' % (len(pre), pre, len(post), post);
    #print 'n_pre, n_post = %d, %d' % (len(positions[t-1]), len(positions[t]))                                     
    #print 'match=%r' % (m,)                               
    #print 'p_post = %r, p_end = %r, new = %r' % (p_post, p_end, new)                       
    #print 'active=%d,%r' % (len(active), active)
    #print 'new pre = %r' %  ([trajectories[a][-1][1] for a in active],)
    #if len(active) != n_post:
    #  raise ValueError();                            
                                  
  return trajectories;
         

def match(positions_pre, positions_post, new_trajectory_cost = None, cutoff = None):
  """Matches two set of positions.
  
  Arguments
  ---------
  positions_pre : array
    The initial particle positions.
  positions_post : array
    The final particle positions.
  new_trajectory_cost : float or None
    The cost for a new trajectory, if None, maximal distance+1
  cutoff : float or None:
    The maximal distance allowed to connect particles.
     
  Returns
  -------
  match : dict
    The optimal association as a dictionary {index_pre : index_post}.
  """
  #create distance matrices
  cost = ssd.cdist(positions_pre, positions_post);
  
  if cutoff:
    cost[cost > cutoff] = np.inf;

  if new_trajectory_cost is None:
    new_trajectory_cost = np.max(cost) + 1.0;
  
  cost = np.pad(cost, [(0,1), (0,1)], 'constant', constant_values = new_trajectory_cost);
  
  #match points
  A = optimal_association_matrix(cost);
  
  return { i : j for i,j in zip(*np.where(A[:-1,:]))}
  

def optimal_association_matrix(cost):
  """Optimizes the association matrix A given the cost matrix cost.
  
  Arguments
  ---------
  cost : array
    The cost matrix, the last row and colum represent csot for
    particle creation/destruction.
     
  Returns
  -------
  association : array
    The optimal association matrix.
    
  Note
  ----
  It is assumed that creation/deletion of objects are the last 
  row and column in cost.
    
  References
  ----------
  A shortest augmenting path algorithm for dense and sparse linear assignment problems
  Jonker, R, Volgenant, A, Computing 1987 
  """
  
  A = _init_association_matrix(cost);
  
  Cs = np.where(cost[:-1,:-1].flatten()<np.inf);
                           
  finished = False;
  while not finished:
    A, finished = _do_one_move(A, cost, Cs);
                             
  return A;


###############################################################################
### Helper
###############################################################################

def _init_association_matrix(cost):
  """Association matrix A""" 
  osize = cost.shape[0] - 1;
  nsize = cost.shape[1] - 1;
  A = np.zeros((osize+1, nsize+1), dtype = bool);
                    
  for i in range(osize):
    # sort costs of real particles
    srtidx = np.argsort(cost[i,:]);
    # index of dummy particle
    dumidx = np.where(srtidx==nsize)[0];
    
    # search for available particle of smallest cost or dummy
    # particle must not be taken and cost must be less than dummy
    iidx = 0;
    while np.sum(A[:,srtidx[iidx]]) != 0 and iidx<dumidx: 
      iidx = iidx + 1;                               
    A[i,srtidx[iidx]] = True;
  
  # set dummy particle for columns with no entry
  s = np.sum(A,axis=0);
  A[osize,s < 1] = True;
  # dummy always corresponds to dummy
  A[osize,nsize] = True;
  
  #import matplotlib.pyplot as plt   
  #plt.figure(7); plt.clf();
  #plt.subplot(1,2,1)            
  #plt.imshow(A, origin = 'lower')            
  #plt.subplot(1,2,2)
  #plt.imshow(cost, origin='lower')
  
  return A;


def _do_one_move(A, C, Cs):  
  """Optimize single association in A."""
  osize = A.shape[0]-1;
  nsize = A.shape[1]-1;
                 
  # find unmade links with finite cost
  todo = np.intersect1d(np.where(np.logical_not(A[:osize, :nsize].flatten()))[0], Cs);    
  if len(todo) == 0:
    return A, True
  # determine induced changes and reduced cost cRed for each
  # candidate link insertion
  iCand, jCand = np.unravel_index(todo, (osize, nsize));                             
  yCand = [np.where(A[ic,:])[0][0] for ic in iCand];
  xCand = [np.where(A[:,jc])[0][0] for jc in jCand]; 
  cRed = [C[i,j] + C[x,y] - C[i,y] - C[x,j] for i,j,x,y in zip(iCand,jCand, xCand, yCand)];
  rMin = np.argmin(cRed);
  rCost = cRed[rMin];             
              
  # if minimum is < 0, link addition is favorable
  if rCost < -1e-10:
    A[iCand[rMin],jCand[rMin]] = 1;
    A[xCand[rMin],jCand[rMin]] = 0;
    A[iCand[rMin],yCand[rMin]] = 0;
    A[xCand[rMin],yCand[rMin]] = 1;
    finished = False;
  else:
    finished = True;
  
  return A, finished;


###############################################################################
### Tests
###############################################################################

def _test():
  import ClearMap.Alignment.Tracking as trk
  from importlib import reload
  reload(trk)

  import ClearMap.Alignment.StitchingWobbly as stw
  
  positions = [[(5,6),(10,10)], [(5,7), (11,10),(30,10)],[(9,10),(31,9)]]
  
  tr = trk.track_positions(positions, creation_destruction_cost = None, cutoff = None)
  
  
  # realistic test
  positions = [stw.detect_local_minima(c, distance = 1)[0] for c in correlation.transpose([2,0,1])]
  positions = positions[1230:];

  k = 20;
  plt.figure(1); plt.clf();
  plt.imshow(correlation[:,:,k].T, origin='lower')
  plt.plot([p[0] for p in positions[k]],[p[1] for p in positions[k]], '*', c='r')
  
  
  tr = trk.track_positions(positions, creation_destruction_cost = np.sqrt(np.sum(np.power(correlation[:,:,0].shape, 2))), cutoff = np.sqrt(2 * 3**2))
  
  tr = trk.track_positions(positions, creation_destruction_cost = np.sqrt(np.sum(np.power(correlation[:,:,0].shape, 2))), cutoff = None)

  import matplotlib as mpl
  from mpl_toolkits.mplot3d import Axes3D
  import matplotlib.pyplot as plt
  plt.figure(1); plt.clf();       
  for t in tr:
    plt.plot([p[0] for p in t], [p[1] for p in t]) 
  
  fig = plt.figure(2); plt.clf();
  ax = fig.gca(projection='3d')
  for t in tr:
    plt.plot([positions[p[0]][p[1]][0] for p in t], [positions[p[0]][p[1]][1] for p in t], [p[0] for p in t])    
    
    
  tr[np.argmax([len(t) for t in tr])]

  # possibility to join with other trajectories ?

