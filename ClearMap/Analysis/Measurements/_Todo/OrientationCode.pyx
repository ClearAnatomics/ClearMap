# -*- coding: utf-8 -*-
"""
Cython code to convert point data into voxel image data for visulaization and analysis

"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'

#cimport cython 

import numpy as np
cimport numpy as np

from multiprocessing import cpu_count;
ncpus = cpu_count();

cimport cython
from cython.parallel import prange, parallel


@cython.boundscheck(False)
@cython.wraparound(False)
def voxelizeOrientations(np.ndarray[np.float_t, ndim=2] points, np.ndarray[np.float_t, ndim=2] orientations, int xsize, int ysize, int zsize, float xdiam, float ydiam, float zdiam, int processes = ncpus):
    """Converts a list of points into an volumetric image array using uniformly filled spheres at the center of each point"""
    
    cdef np.ndarray[np.float_t, ndim = 4] voximg = np.zeros([xsize, ysize, zsize, 3], dtype=np.float);

    cdef int iCentroid = 0
    cdef int nCentroid = points.shape[0]
    cdef int nSphereIndices = int(xdiam * ydiam * zdiam)

    # precompute indices centered at 0,0,0
    cdef np.ndarray[np.int_t, ndim = 1] xs = np.zeros([nSphereIndices], dtype=np.int)
    cdef np.ndarray[np.int_t, ndim = 1] ys = np.zeros([nSphereIndices], dtype=np.int)
    cdef np.ndarray[np.int_t, ndim = 1] zs = np.zeros([nSphereIndices], dtype=np.int)
    cdef int ns = 0

    cdef float xdiam2 = (xdiam - 1) * (xdiam - 1) / 4
    cdef float ydiam2 = (ydiam - 1) * (ydiam - 1) / 4
    cdef float zdiam2 = (zdiam - 1) * (zdiam - 1) / 4
    
    for x in range(int(-xdiam/2 + 1), int(xdiam/2 + 1)):
      for y in range(int(-ydiam/2 + 1), int(ydiam/2 + 1)):
        for z in range(int(-zdiam/2 + 1), int(zdiam/2 + 1)):
          if x*x / xdiam2 + y*y / ydiam2 + z*z / zdiam2 < 1:
            xs[ns] = x; ys[ns] = y; zs[ns] = z;
            ns += 1;
                    
    cdef int iss = 0
    cdef float cx0
    cdef float cy0
    cdef float cz0
    
    cdef float cxf
    cdef float cyf
    cdef float czf
    
    cdef int cx
    cdef int cy
    cdef int cz
    

    #with nogil, parallel(num_threads = processes):                 
    #  for iCentroid in prange(nCentroid):
    for iCentroid in range(nCentroid):
      #if ((iCentroid % 25000) == 0):
      #    print "\nProcessed %d/%d\n" % (iCentroid, nCentroid);
  
      cx0 = points[iCentroid, 0];
      cy0 = points[iCentroid, 1];
      cz0 = points[iCentroid, 2];
      
      for iss in range(ns):
        cxf = cx0 + xs[iss];
        cyf = cy0 + ys[iss];
        czf = cz0 + zs[iss];
        
        if cxf >= 0 and cxf < xsize:
          if cyf >= 0 and cyf < ysize:
            if czf >= 0 and czf < zsize:
              cx = int(cxf);
              cy = int(cyf);
              cz = int(czf);
              
              if orientations[iCentroid,0] >= 0: 
                voximg[cx,cy,cz,0] += orientations[iCentroid,0];
                voximg[cx,cy,cz,1] += orientations[iCentroid,1];
                voximg[cx,cy,cz,2] += orientations[iCentroid,2];
              else:
                voximg[cx,cy,cz,0] -= orientations[iCentroid,0];
                voximg[cx,cy,cz,1] -= orientations[iCentroid,1];
                voximg[cx,cy,cz,2] -= orientations[iCentroid,2];
                
    return voximg;