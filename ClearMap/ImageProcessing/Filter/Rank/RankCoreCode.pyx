#cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False, cdivision=True

"""
RankCoreCode
============

Cython code for the core rank filter functions.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


#TODO: masked version with skipping non-valid region borders !

import numpy as np

cimport numpy as cnp
from libc.stdlib cimport malloc, free

from . cimport RankCoreCode

cdef extern from "stdio.h":
  int printf(char *format, ...) nogil


cdef inline source_t _max(source_t a, source_t b) nogil:
  return a if a >= b else b

cdef inline source_t _min(source_t a, source_t b) nogil:
  return a if a <= b else b


cdef inline void histogram_increment(index_t* histo, index_t* pop, source_t value) nogil:
  histo[value] += 1
  pop[0] += 1

cdef inline void histogram_decrement(index_t* histo, index_t* pop, source_t value) nogil:
  histo[value] -= 1
  pop[0] -= 1


cdef inline char is_in_source(index_t nx, index_t ny, index_t nz, 
                             index_t x,  index_t y,  index_t z) nogil:
  if x < 0 or x > nx - 1 or y < 0 or y > ny - 1 or z < 0 or z > nz - 1:
    return 0;
  else:
    return 1;


cdef void rank_core(void kernel(sink_t*, index_t*, index_t, source_t, index_t, index_t*, double*) nogil,
                    source_t[:, :, :] source, char[:, :, :] selem,
                    sink_t[:, :, :, :] sink,
                    index_t max_bin, index_t[:] parameter_index, double[:] parameter_double) except *:
  """Compute histogram for each pixel, azzly kernel function to calculate sinkput."""

  cdef index_t nx = source.shape[0]
  cdef index_t ny = source.shape[1]
  cdef index_t nz = source.shape[2]
  
  cdef index_t snx = selem.shape[0]
  cdef index_t sny = selem.shape[1]
  cdef index_t snz = selem.shape[2]
  cdef index_t max_se  = snx * sny * snz

  cdef index_t scx = <index_t>(selem.shape[0] / 2)
  cdef index_t scy = <index_t>(selem.shape[1] / 2) 
  cdef index_t scz = <index_t>(selem.shape[2] / 2)
  
  #local variables
  cdef index_t i, xx, yy, zz
  cdef index_t x, y, z, dir_x, dir_y, mode

  #number of pixels in heighbourhood
  cdef index_t pop = 0

  # the cuxxent local histogram distribution
  cdef index_t* histo = <index_t*>malloc(max_bin * sizeof(index_t))
  for i in range(max_bin):
    histo[i] = 0

  # attack borders 
  cdef index_t num_se_n, num_se_s, num_se_e, num_se_w, num_se_d, num_se_u
  num_se_n = num_se_s = num_se_e = num_se_w = num_se_d = num_se_u = 0

  cdef index_t* se_e_x = <index_t*>malloc(max_se * sizeof(index_t))
  cdef index_t* se_e_y = <index_t*>malloc(max_se * sizeof(index_t))
  cdef index_t* se_e_z = <index_t*>malloc(max_se * sizeof(index_t))
  
  cdef index_t* se_w_x = <index_t*>malloc(max_se * sizeof(index_t))
  cdef index_t* se_w_y = <index_t*>malloc(max_se * sizeof(index_t))
  cdef index_t* se_w_z = <index_t*>malloc(max_se * sizeof(index_t)) 

  
  cdef index_t* se_n_x = <index_t*>malloc(max_se * sizeof(index_t))
  cdef index_t* se_n_y = <index_t*>malloc(max_se * sizeof(index_t))
  cdef index_t* se_n_z = <index_t*>malloc(max_se * sizeof(index_t))
  
  cdef index_t* se_s_x = <index_t*>malloc(max_se * sizeof(index_t))
  cdef index_t* se_s_y = <index_t*>malloc(max_se * sizeof(index_t))
  cdef index_t* se_s_z = <index_t*>malloc(max_se * sizeof(index_t))
  
  
  cdef index_t* se_u_x = <index_t*>malloc(max_se * sizeof(index_t))
  cdef index_t* se_u_y = <index_t*>malloc(max_se * sizeof(index_t))
  cdef index_t* se_u_z = <index_t*>malloc(max_se * sizeof(index_t))
  
  cdef index_t* se_d_x = <index_t*>malloc(max_se * sizeof(index_t))
  cdef index_t* se_d_y = <index_t*>malloc(max_se * sizeof(index_t))
  cdef index_t* se_d_z = <index_t*>malloc(max_se * sizeof(index_t))

  # build attack and release borders by using difference along axis
  t = np.concatenate((selem, np.zeros((1, selem.shape[1], selem.shape[2]))), axis = 0)
  cdef unsigned char[:, :, :] t_e = (np.diff(t, axis=0) < 0).view(np.uint8)

  t = np.concatenate((np.zeros((1, selem.shape[1], selem.shape[2])), selem), axis = 0)
  cdef unsigned char[:, :, :] t_w = (np.diff(t, axis=0) > 0).view(np.uint8)
  
  
  t = np.concatenate((selem, np.zeros((selem.shape[0], 1, selem.shape[2]))), axis = 1)
  cdef unsigned char[:, :, :] t_n = (np.diff(t, axis=1) < 0).view(np.uint8)

  t = np.concatenate((np.zeros((selem.shape[0], 1, selem.shape[2])), selem), axis = 1)
  cdef unsigned char[:, :, :] t_s = (np.diff(t, axis=1) > 0).view(np.uint8)
  
  
  t = np.concatenate((selem, np.zeros((selem.shape[0], selem.shape[1], 1))), axis = 2)
  cdef unsigned char[:, :, :] t_u = (np.diff(t, axis=2) < 0).view(np.uint8)

  t = np.concatenate((np.zeros((selem.shape[0], selem.shape[1], 1)), selem), axis = 2)
  cdef unsigned char[:, :, :] t_d = (np.diff(t, axis=2) > 0).view(np.uint8)
  
  cdef index_t* par_index  = &parameter_index[0];
  cdef double*  par_double = &parameter_double[0];
  
  with nogil:
    #define attack borders
    for x in range(snx):
      for y in range(sny):
        for z in range(snz):
          if t_e[x, y, z]:
              se_e_x[num_se_e] = x - scx
              se_e_y[num_se_e] = y - scy
              se_e_z[num_se_e] = z - scz         
              num_se_e += 1
          if t_w[x, y, z]:
              se_w_x[num_se_w] = x - scx
              se_w_y[num_se_w] = y - scy
              se_w_z[num_se_w] = z - scz
              num_se_w += 1
          
          if t_n[x, y, z]:
              se_n_x[num_se_n] = x - scx
              se_n_y[num_se_n] = y - scy
              se_n_z[num_se_n] = z - scz  
              num_se_n += 1
          if t_s[x, y, z]:
              se_s_x[num_se_s] = x - scx
              se_s_y[num_se_s] = y - scy
              se_s_z[num_se_s] = z - scz 
              num_se_s += 1
              
          if t_u[x, y, z]:
              se_u_x[num_se_u] = x - scx
              se_u_y[num_se_u] = y - scy
              se_u_z[num_se_u] = z - scz  
              num_se_u += 1 
          if t_d[x, y, z]:
              se_d_x[num_se_d] = x - scx
              se_d_y[num_se_d] = y - scy
              se_d_z[num_se_d] = z - scz  
              num_se_d += 1
    
    # create historgram
    for x in range(snx):
      for y in range(sny):
        for z in range(snz):
          xx = x - scx
          yy = y - scy
          zz = z - scz
          if selem[x, y, z]:
            if is_in_source(nx, ny, nz, xx, yy, zz):
              histogram_increment(histo, &pop, source[xx, yy, zz])
              #printf('hist adding (%d,%d,%d) [%d]\n', xx, yy, zz, source[xx,yy,zz]);    
    x = 0
    y = 0
    z = 0;
    kernel(&sink[x, y, z, 0], histo, pop, source[x, y, z], max_bin, par_index, par_double)

    # main loop
    dir_x = 1;
    dir_y = 1;
    mode = 0; # modes 0 r+=1, 1 r-=1, 2 c+=1, 3 c-=1, 4 p+=1
      
    while True:
      #printf('-----\n');          
      
      if dir_x == 1 and x < nx - 1:
        x += 1;
        mode = 0;
      elif dir_x == -1 and x > 0:
        x -= 1;
        mode = 1;
      else:
        if dir_y == 1 and y < ny - 1:
          y += 1;
          dir_x *= -1;
          mode = 2;
        elif dir_y == -1 and y > 0:
          y -= 1;
          dir_x *= -1;
          mode = 3;
        else:
          z += 1
          if z == nz:
            break;
          dir_x *= -1;
          dir_y *= -1;
          mode = 4;

      if mode == 0:
        for s in range(num_se_e):
          xx = x + se_e_x[s]
          yy = y + se_e_y[s]
          zz = z + se_e_z[s]
          if is_in_source(nx, ny, nz, xx, yy, zz):
            histogram_increment(histo, &pop, source[xx, yy, zz])
            #printf('mode %d adding (%d,%d,%d) [%d]\n', mode, xx, yy, zz, source[xx,yy,zz]);

        for s in range(num_se_w):
          xx = x + se_w_x[s] - 1
          yy = y + se_w_y[s] 
          zz = z + se_w_z[s]
          if is_in_source(nx, ny, nz, xx, yy, zz):
            histogram_decrement(histo, &pop, source[xx, yy, zz])
            #printf('mode %d removing (%d,%d,%d) [%d]\n', mode, xx, yy, zz, source[xx,yy,zz]);
      
      elif mode == 1:
        for s in range(num_se_w):
          xx = x + se_w_x[s]
          yy = y + se_w_y[s]
          zz = z + se_w_z[s] 
          if is_in_source(nx, ny, nz, xx, yy, zz):
            histogram_increment(histo, &pop, source[xx, yy, zz])
            #printf('mode %d adding (%d,%d,%d) [%d]\n', mode, xx, yy, zz, source[xx,yy,zz]);
        
        for s in range(num_se_e):
          xx = x + se_e_x[s] + 1
          yy = y + se_e_y[s]
          zz = z + se_e_z[s] 
          if is_in_source(nx, ny, nz, xx, yy, zz):
            histogram_decrement(histo, &pop, source[xx, yy, zz])
            #printf('mode %d removing (%d,%d,%d) [%d]\n', mode, xx, yy, zz, source[xx,yy,zz]);

      elif mode == 2:
        for s in range(num_se_n):
          xx = x + se_n_x[s]
          yy = y + se_n_y[s]
          zz = z + se_n_z[s] 
          if is_in_source(nx, ny, nz, xx, yy, zz):
            histogram_increment(histo, &pop, source[xx, yy, zz])
            #printf('mode %d adding (%d,%d,%d) [%d]\n', mode, xx, yy, zz, source[xx,yy,zz]);

        for s in range(num_se_s):
          xx = x + se_s_x[s] 
          yy = y + se_s_y[s] - 1
          zz = z + se_s_z[s] 
          if is_in_source(nx, ny, nz, xx, yy, zz):
            histogram_decrement(histo, &pop, source[xx, yy, zz])
            #printf('mode %d removing (%d,%d,%d) [%d]\n', mode, xx, yy, zz, source[xx,yy,zz]);
      
      elif mode == 3:
        for s in range(num_se_s):
          xx = x + se_s_x[s]
          yy = y + se_s_y[s]
          zz = z + se_s_z[s] 
          if is_in_source(nx, ny, nz, xx, yy, zz):
            histogram_increment(histo, &pop, source[xx, yy, zz])
            #printf('mode %d adding (%d,%d,%d) [%d]\n', mode, xx, yy, zz, source[xx,yy,zz]);

        for s in range(num_se_n):
          xx = x + se_n_x[s] 
          yy = y + se_n_y[s] + 1
          zz = z + se_n_z[s] 
          if is_in_source(nx, ny, nz, xx, yy, zz):
            histogram_decrement(histo, &pop, source[xx, yy, zz])
            #printf('mode %d removing (%d,%d,%d) [%d]\n', mode, xx, yy, zz, source[xx,yy,zz]);
      
      elif mode == 4:
        for s in range(num_se_u):
          xx = x + se_u_x[s]
          yy = y + se_u_y[s]
          zz = z + se_u_z[s] 
          if is_in_source(nx, ny, nz, xx, yy, zz):
            histogram_increment(histo, &pop, source[xx, yy, zz])
            #printf('mode %d adding (%d,%d,%d) [%d]\n', mode, xx, yy, zz, source[xx,yy,zz]);

        for s in range(num_se_d):
          xx = x + se_d_x[s]
          yy = y + se_d_y[s]
          zz = z + se_d_z[s] - 1
          if is_in_source(nx, ny, nz, xx, yy, zz):
            histogram_decrement(histo, &pop, source[xx, yy, zz])
            #printf('mode %d removing (%d,%d,%d) [%d]\n', mode, xx, yy, zz, source[xx,yy,zz]);
      
      kernel(&sink[x, y, z, 0], histo, pop, source[x, y, z], max_bin, par_index, par_double)
    #while True
  #with nogil
        
  # release memory allocated by malloc
  free(se_e_x)
  free(se_e_y)
  free(se_e_z)
  
  free(se_w_x)
  free(se_w_y)
  free(se_w_z)
  
  free(se_n_x)
  free(se_n_y)
  free(se_n_z)
  
  free(se_s_x)
  free(se_s_y)
  free(se_s_z)
  
  free(se_u_x)
  free(se_u_y)
  free(se_u_z)        
  
  free(se_d_x)
  free(se_d_y)
  free(se_d_z)
  
  free(histo)





cdef inline char is_in_masked_source(index_t nx, index_t ny, index_t nz, 
                                     index_t x,  index_t y,  index_t z, char[:,:,:] mask) nogil:
  if x < 0 or x > nx - 1 or y < 0 or y > ny - 1 or z < 0 or z > nz - 1:
    return 0;
  else:
    return mask[x,y,z];


cdef inline void move(index_t nx, index_t ny, index_t nz, index_t* x, index_t* y, index_t* z, index_t* dir_x, index_t* dir_y, index_t* mode, index_t* done) nogil:
  if dir_x[0] == 1 and x[0] < nx - 1:
    x[0] += 1;
    mode[0] = 0;
  elif dir_x[0] == -1 and x[0] > 0:
    x[0] -= 1;
    mode[0] = 1;  
  else:
    if dir_y[0] == 1 and y[0] < ny - 1:
      y[0] += 1;
      dir_x[0] *= -1;
      mode[0] = 2;
    elif dir_y[0] == -1 and y[0] > 0:
      y[0] -= 1;
      dir_x[0] *= -1;
      mode[0] = 3;
    else:
      z[0] += 1
      if z[0] == nz:
        done[0] = 1;
      dir_x[0] *= -1;
      dir_y[0] *= -1;
      mode[0] = 4;


cdef inline void move_histo(index_t nx, index_t ny, index_t nz, index_t x, index_t y, index_t z, index_t mode,
                            index_t* se_e_x, index_t* se_e_y, index_t* se_e_z, 
                            index_t* se_w_x, index_t* se_w_y, index_t* se_w_z, 
                            index_t* se_n_x, index_t* se_n_y, index_t* se_n_z, 
                            index_t* se_s_x, index_t* se_s_y, index_t* se_s_z, 
                            index_t* se_u_x, index_t* se_u_y, index_t* se_u_z, 
                            index_t* se_d_x, index_t* se_d_y, index_t* se_d_z,
                            index_t num_se_e, index_t num_se_w, index_t num_se_n, index_t num_se_s, index_t num_se_u, index_t num_se_d,
                            index_t* histo, index_t* pop, source_t[:, :, :] source, char[:,:,:] mask) nogil:
      cdef index_t xx,yy,zz,s
  
      if mode == 0:
        for s in range(num_se_e):
          xx = x + se_e_x[s]
          yy = y + se_e_y[s]
          zz = z + se_e_z[s]
          if is_in_masked_source(nx, ny, nz, xx, yy, zz, mask):
            histogram_increment(histo, pop, source[xx, yy, zz])
            #printf('mode %d adding (%d,%d,%d) [%d]\n', mode, xx, yy, zz, source[xx,yy,zz]);

        for s in range(num_se_w):
          xx = x + se_w_x[s] - 1
          yy = y + se_w_y[s] 
          zz = z + se_w_z[s]
          if is_in_masked_source(nx, ny, nz, xx, yy, zz, mask):
            histogram_decrement(histo, pop, source[xx, yy, zz])
            #printf('mode %d removing (%d,%d,%d) [%d]\n', mode, xx, yy, zz, source[xx,yy,zz]);
      
      elif mode == 1:
        for s in range(num_se_w):
          xx = x + se_w_x[s]
          yy = y + se_w_y[s]
          zz = z + se_w_z[s] 
          if is_in_masked_source(nx, ny, nz, xx, yy, zz, mask):
            histogram_increment(histo, pop, source[xx, yy, zz])
            #printf('mode %d adding (%d,%d,%d) [%d]\n', mode, xx, yy, zz, source[xx,yy,zz]);
        
        for s in range(num_se_e):
          xx = x + se_e_x[s] + 1
          yy = y + se_e_y[s]
          zz = z + se_e_z[s] 
          if is_in_masked_source(nx, ny, nz, xx, yy, zz, mask):
            histogram_decrement(histo, pop, source[xx, yy, zz])
            #printf('mode %d removing (%d,%d,%d) [%d]\n', mode, xx, yy, zz, source[xx,yy,zz]);

      elif mode == 2:
        for s in range(num_se_n):
          xx = x + se_n_x[s]
          yy = y + se_n_y[s]
          zz = z + se_n_z[s] 
          if is_in_masked_source(nx, ny, nz, xx, yy, zz, mask):
            histogram_increment(histo, pop, source[xx, yy, zz])
            #printf('mode %d adding (%d,%d,%d) [%d]\n', mode, xx, yy, zz, source[xx,yy,zz]);

        for s in range(num_se_s):
          xx = x + se_s_x[s] 
          yy = y + se_s_y[s] - 1
          zz = z + se_s_z[s]
          if is_in_masked_source(nx, ny, nz, xx, yy, zz, mask):
            histogram_decrement(histo, pop, source[xx, yy, zz])
            #printf('mode %d removing (%d,%d,%d) [%d]\n', mode, xx, yy, zz, source[xx,yy,zz]);
      
      elif mode == 3:
        for s in range(num_se_s):
          xx = x + se_s_x[s]
          yy = y + se_s_y[s]
          zz = z + se_s_z[s] 
          if is_in_masked_source(nx, ny, nz, xx, yy, zz, mask):
            histogram_increment(histo, pop, source[xx, yy, zz])
            #printf('mode %d adding (%d,%d,%d) [%d]\n', mode, xx, yy, zz, source[xx,yy,zz]);

        for s in range(num_se_n):
          xx = x + se_n_x[s] 
          yy = y + se_n_y[s] + 1
          zz = z + se_n_z[s] 
          if is_in_masked_source(nx, ny, nz, xx, yy, zz, mask):
            histogram_decrement(histo, pop, source[xx, yy, zz])
            #printf('mode %d removing (%d,%d,%d) [%d]\n', mode, xx, yy, zz, source[xx,yy,zz]);
      
      elif mode == 4:
        for s in range(num_se_u):
          xx = x + se_u_x[s]
          yy = y + se_u_y[s]
          zz = z + se_u_z[s] 
          if is_in_masked_source(nx, ny, nz, xx, yy, zz, mask):
            histogram_increment(histo, pop, source[xx, yy, zz])
            #printf('mode %d adding (%d,%d,%d) [%d]\n', mode, xx, yy, zz, source[xx,yy,zz]);

        for s in range(num_se_d):
          xx = x + se_d_x[s]
          yy = y + se_d_y[s]
          zz = z + se_d_z[s] - 1
          if is_in_masked_source(nx, ny, nz, xx, yy, zz, mask):
            histogram_decrement(histo, pop, source[xx, yy, zz])
            #printf('mode %d removing (%d,%d,%d) [%d]\n', mode, xx, yy, zz, source[xx,yy,zz]);



cdef inline void clean_up(index_t* se_e_x, index_t* se_e_y, index_t* se_e_z, 
                          index_t* se_w_x, index_t* se_w_y, index_t* se_w_z, 
                          index_t* se_n_x, index_t* se_n_y, index_t* se_n_z, 
                          index_t* se_s_x, index_t* se_s_y, index_t* se_s_z, 
                          index_t* se_u_x, index_t* se_u_y, index_t* se_u_z, 
                          index_t* se_d_x, index_t* se_d_y, index_t* se_d_z, index_t* histo) nogil:
  # release memory allocated by malloc
  free(se_e_x); free(se_e_y); free(se_e_z);  
  free(se_w_x); free(se_w_y); free(se_w_z); 
  free(se_n_x); free(se_n_y); free(se_n_z); 
  free(se_s_x); free(se_s_y); free(se_s_z)
  free(se_u_x); free(se_u_y); free(se_u_z)        
  free(se_d_x); free(se_d_y); free(se_d_z) 
  free(histo)


cdef void rank_core_masked(void kernel(sink_t*, index_t*, index_t, source_t, index_t, index_t*, double*) nogil,
                           source_t[:, :, :] source, char[:, :, :] selem, char[:,:,:] mask,
                           sink_t[:, :, :, :] sink, index_t max_bin,
                           index_t[:] parameter_index, double[:] parameter_double) except *:
  """Compute histogram for each pixel, azzly kernel function to calculate sinkput."""

  cdef index_t nx = source.shape[0]
  cdef index_t ny = source.shape[1]
  cdef index_t nz = source.shape[2]
  
  cdef index_t snx = selem.shape[0]
  cdef index_t sny = selem.shape[1]
  cdef index_t snz = selem.shape[2]
  cdef index_t max_se  = snx * sny * snz

  cdef index_t scx = <index_t>(selem.shape[0] / 2) 
  cdef index_t scy = <index_t>(selem.shape[1] / 2)
  cdef index_t scz = <index_t>(selem.shape[2] / 2)

  # define local variable types
  cdef index_t i, d, xx, yy, zz
  cdef index_t x, y, z, dir_x, dir_y, mode
  cdef index_t xp, yp, zp, dir_xp, dir_yp, modep
  cdef index_t done, donep

  # number of pixels in heighbourhood
  cdef index_t pop = 0

  # the cuxxent local histogram distribution
  cdef index_t* histo = <index_t*>malloc(max_bin * sizeof(index_t))
  for i in range(max_bin):
    histo[i] = 0

  # these lists contain the relative pixel row and column for each of the 6
  # attack borders east, west, north and ssinkh, 

  # number of elements in each attack border
  cdef index_t num_se_n, num_se_s, num_se_e, num_se_w, num_se_d, num_se_u
  num_se_n = num_se_s = num_se_e = num_se_w = num_se_d = num_se_u = 0

  cdef index_t* se_e_x = <index_t*>malloc(max_se * sizeof(index_t))
  cdef index_t* se_e_y = <index_t*>malloc(max_se * sizeof(index_t))
  cdef index_t* se_e_z = <index_t*>malloc(max_se * sizeof(index_t))
  
  cdef index_t* se_w_x = <index_t*>malloc(max_se * sizeof(index_t))
  cdef index_t* se_w_y = <index_t*>malloc(max_se * sizeof(index_t))
  cdef index_t* se_w_z = <index_t*>malloc(max_se * sizeof(index_t)) 

  
  cdef index_t* se_n_x = <index_t*>malloc(max_se * sizeof(index_t))
  cdef index_t* se_n_y = <index_t*>malloc(max_se * sizeof(index_t))
  cdef index_t* se_n_z = <index_t*>malloc(max_se * sizeof(index_t))
  
  cdef index_t* se_s_x = <index_t*>malloc(max_se * sizeof(index_t))
  cdef index_t* se_s_y = <index_t*>malloc(max_se * sizeof(index_t))
  cdef index_t* se_s_z = <index_t*>malloc(max_se * sizeof(index_t))
  
  
  cdef index_t* se_u_x = <index_t*>malloc(max_se * sizeof(index_t))
  cdef index_t* se_u_y = <index_t*>malloc(max_se * sizeof(index_t))
  cdef index_t* se_u_z = <index_t*>malloc(max_se * sizeof(index_t))
  
  cdef index_t* se_d_x = <index_t*>malloc(max_se * sizeof(index_t))
  cdef index_t* se_d_y = <index_t*>malloc(max_se * sizeof(index_t))
  cdef index_t* se_d_z = <index_t*>malloc(max_se * sizeof(index_t))

  # build attack and release borders by using difference along axis
  t = np.concatenate((selem, np.zeros((1, selem.shape[1], selem.shape[2]))), axis = 0)
  cdef unsigned char[:, :, :] t_e = (np.diff(t, axis=0) < 0).view(np.uint8)

  t = np.concatenate((np.zeros((1, selem.shape[1], selem.shape[2])), selem), axis = 0)
  cdef unsigned char[:, :, :] t_w = (np.diff(t, axis=0) > 0).view(np.uint8)
  
  
  t = np.concatenate((selem, np.zeros((selem.shape[0], 1, selem.shape[2]))), axis = 1)
  cdef unsigned char[:, :, :] t_n = (np.diff(t, axis=1) < 0).view(np.uint8)

  t = np.concatenate((np.zeros((selem.shape[0], 1, selem.shape[2])), selem), axis = 1)
  cdef unsigned char[:, :, :] t_s = (np.diff(t, axis=1) > 0).view(np.uint8)
  
  
  t = np.concatenate((selem, np.zeros((selem.shape[0], selem.shape[1], 1))), axis = 2)
  cdef unsigned char[:, :, :] t_u = (np.diff(t, axis=2) < 0).view(np.uint8)

  t = np.concatenate((np.zeros((selem.shape[0], selem.shape[1], 1)), selem), axis = 2)
  cdef unsigned char[:, :, :] t_d = (np.diff(t, axis=2) > 0).view(np.uint8)
  
  cdef index_t* par_index  = &parameter_index[0];
  cdef double*  par_double = &parameter_double[0];
  
  with nogil:
    #define attack borders
    for x in range(snx):
      for y in range(sny):
        for z in range(snz):
          if t_e[x, y, z]:
              se_e_x[num_se_e] = x - scx
              se_e_y[num_se_e] = y - scy
              se_e_z[num_se_e] = z - scz         
              num_se_e += 1
          if t_w[x, y, z]:
              se_w_x[num_se_w] = x - scx
              se_w_y[num_se_w] = y - scy
              se_w_z[num_se_w] = z - scz
              num_se_w += 1
          
          if t_n[x, y, z]:
              se_n_x[num_se_n] = x - scx
              se_n_y[num_se_n] = y - scy
              se_n_z[num_se_n] = z - scz  
              num_se_n += 1
          if t_s[x, y, z]:
              se_s_x[num_se_s] = x - scx
              se_s_y[num_se_s] = y - scy
              se_s_z[num_se_s] = z - scz 
              num_se_s += 1
              
          if t_u[x, y, z]:
              se_u_x[num_se_u] = x - scx
              se_u_y[num_se_u] = y - scy
              se_u_z[num_se_u] = z - scz  
              num_se_u += 1 
          if t_d[x, y, z]:
              se_d_x[num_se_d] = x - scx
              se_d_y[num_se_d] = y - scy
              se_d_z[num_se_d] = z - scz  
              num_se_d += 1
    
    #printf('e=%d, w=%d, n=%d, s=%d, u=%d, d=%d\n', num_se_e, num_se_w, num_se_n, num_se_s, num_se_u, num_se_d);
    #for i in range(num_se_e):
    #  printf('se_e = (%d,%d,%d)\n' , se_e_x[i], se_e_y[i], se_e_z[i]);
    #for i in range(num_se_w):
    #  printf('se_w = (%d,%d,%d)\n' , se_w_x[i], se_w_y[i], se_w_z[i]);

    
    # move to first non masked pixel
    x = 0
    y = 0
    z = 0; 
    
    dir_x = 1;
    dir_y = 1;
    mode = 0;  # modes 0 x+=1, 1 x-=1, 2 y+=1, 3 y-=1, 4 z+=1
    done = 0; 
    while not is_in_masked_source(nx, ny, nz, x, y, z, mask):
      move(nx, ny, nz, &x, &y, &z, &dir_x, &dir_y, &mode, &done); 
      if done:
        clean_up(se_e_x, se_e_y, se_e_z, se_w_x, se_w_y, se_w_z, se_n_x, se_n_y, se_n_z, se_s_x, se_s_y, se_s_z, se_u_x, se_u_y, se_u_z, se_d_x, se_d_y, se_d_z, histo);
        return;
    #printf('x=%d, y=%d, z=%d, dirx=%d, diry=%d, mode=%d\n', x,y,z,dir_x,dir_y,mode);    
    
    # create historgram
    for xp in range(snx):
      for yp in range(sny):
        for zp in range(snz):
          if selem[xp, yp, zp]:
            xx = x + xp - scx;
            yy = y + yp - scy;
            zz = z + zp - scz;
            if is_in_masked_source(nx, ny, nz, xx, yy, zz, mask):
              histogram_increment(histo, &pop, source[xx, yy, zz])
              #printf('hist adding (%d,%d,%d) [%d]\n', xx, yy, zz, source[xx,yy,zz]); 
    
    kernel(&sink[x, y, z, 0], histo, pop, source[x, y, z], max_bin, par_index, par_double)

    # main loop 
    while True:
      xp = x; yp = y; zp = z; # save previous positions
      move(nx, ny, nz, &x, &y, &z, &dir_x, &dir_y, &mode, &done);
      if done:
        clean_up(se_e_x, se_e_y, se_e_z, se_w_x, se_w_y, se_w_z, se_n_x, se_n_y, se_n_z, se_s_x, se_s_y, se_s_z, se_u_x, se_u_y, se_u_z, se_d_x, se_d_y, se_d_z, histo);
        return;
      #printf('main move: x=%d, y=%d, z=%d, dirx=%d, diry=%d, mode=%d\n', x,y,z,dir_x, dir_y, mode);    
      #printf('pop=%d\n', pop);
      
      #move to next valid 
      if not is_in_masked_source(nx,ny,nz,x,y,z,mask):
        x,xp = xp,x; y,yp = yp,y; z,zp = zp,z;
        dir_xp = dir_x; dir_yp = dir_y;
        modep = mode; donep = done;
        while not is_in_masked_source(nx,ny,nz,xp,yp,zp,mask):
          move(nx, ny, nz, &xp, &yp, &zp, &dir_xp, &dir_yp, &modep, &donep);
          if donep:
            clean_up(se_e_x, se_e_y, se_e_z, se_w_x, se_w_y, se_w_z, se_n_x, se_n_y, se_n_z, se_s_x, se_s_y, se_s_z, se_u_x, se_u_y, se_u_z, se_d_x, se_d_y, se_d_z, histo);
            return;
        #printf('masked move: x=%d, y=%d, z=%d, dirx=%d, diry=%d, mode=%d\n', xp,yp,zp,dir_xp, dir_yp, modep);    
 
        
        #move to next valid coordinate via shortest path -> speed up for a object surrounded by background 
        d = xp - x;
        if d > 0:
          for i in range(d):
            mode = 0; 
            x += 1;
            move_histo(nx, ny, nz, x, y, z, mode,
                       se_e_x, se_e_y, se_e_z, se_w_x, se_w_y, se_w_z, 
                       se_n_x, se_n_y, se_n_z, se_s_x, se_s_y, se_s_z, 
                       se_u_x, se_u_y, se_u_z, se_d_x, se_d_y, se_d_z, 
                       num_se_e, num_se_w, num_se_n, num_se_s, num_se_u, num_se_d,
                       histo, &pop, source, mask);

        elif d < 0:
          for i in range(-d):
            mode = 1; 
            x -= 1;
            move_histo(nx, ny, nz, x, y, z, mode,
                       se_e_x, se_e_y, se_e_z, se_w_x, se_w_y, se_w_z, 
                       se_n_x, se_n_y, se_n_z, se_s_x, se_s_y, se_s_z, 
                       se_u_x, se_u_y, se_u_z, se_d_x, se_d_y, se_d_z,
                       num_se_e, num_se_w, num_se_n, num_se_s, num_se_u, num_se_d,
                       histo, &pop, source, mask);

        
        d = yp - y;
        if d > 0:
          for i in range(d):
            mode = 2; 
            y += 1;
            move_histo(nx, ny, nz, x, y, z, mode,
                       se_e_x, se_e_y, se_e_z, se_w_x, se_w_y, se_w_z, 
                       se_n_x, se_n_y, se_n_z, se_s_x, se_s_y, se_s_z, 
                       se_u_x, se_u_y, se_u_z, se_d_x, se_d_y, se_d_z,
                       num_se_e, num_se_w, num_se_n, num_se_s, num_se_u, num_se_d,
                       histo, &pop, source, mask);

        elif d < 0:
          for i in range(-d):
            mode = 3; 
            y -= 1;
            move_histo(nx, ny, nz, x, y, z, mode,
                       se_e_x, se_e_y, se_e_z, se_w_x, se_w_y, se_w_z, 
                       se_n_x, se_n_y, se_n_z, se_s_x, se_s_y, se_s_z, 
                       se_u_x, se_u_y, se_u_z, se_d_x, se_d_y, se_d_z,
                       num_se_e, num_se_w, num_se_n, num_se_s, num_se_u, num_se_d,
                       histo, &pop, source, mask);


        d = zp - z;
        if d > 0:
          for i in range(d):
            mode = 4; 
            z += 1;
            move_histo(nx, ny, nz, x, y, z, mode,
                       se_e_x, se_e_y, se_e_z, se_w_x, se_w_y, se_w_z, 
                       se_n_x, se_n_y, se_n_z, se_s_x, se_s_y, se_s_z, 
                       se_u_x, se_u_y, se_u_z, se_d_x, se_d_y, se_d_z,
                       num_se_e, num_se_w, num_se_n, num_se_s, num_se_u, num_se_d,
                       histo, &pop, source, mask);

      
        dir_x = dir_xp; dir_y = dir_yp; mode = modep;
        #printf('after masked move: x=%d, y=%d, z=%d, dirx=%d, diry=%d, mode=%d\n', x,y,z,dir_x, dir_y, mode); 
        #printf('pop=%d\n', pop)
      
      else: #if is_in_masked_source
        move_histo(nx, ny, nz, x, y, z, mode,
                   se_e_x, se_e_y, se_e_z, se_w_x, se_w_y, se_w_z, 
                   se_n_x, se_n_y, se_n_z, se_s_x, se_s_y, se_s_z, 
                   se_u_x, se_u_y, se_u_z, se_d_x, se_d_y, se_d_z,
                   num_se_e, num_se_w, num_se_n, num_se_s, num_se_u, num_se_d,
                   histo, &pop, source, mask);
      
      #printf('after main move: x=%d, y=%d, z=%d, dirx=%d, diry=%d, mode=%d\n', x,y,z,dir_x, dir_y, mode); 
      #printf('pop=%d\n', pop) 
      
      kernel(&sink[x, y, z, 0], histo, pop, source[x, y, z], max_bin, par_index, par_double)
    #while True
  #with nogil
 
  clean_up(se_e_x, se_e_y, se_e_z, se_w_x, se_w_y, se_w_z, se_n_x, se_n_y, se_n_z, se_s_x, se_s_y, se_s_z, se_u_x, se_u_y, se_u_z, se_d_x, se_d_y, se_d_z, histo);
