#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
GraphVisualization
==================

Module providing tools to create meshes and visualize graphs.

"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


import numpy as np

import functools as ft

import vispy.util.transforms as trf

import ClearMap.ParallelProcessing.SharedMemoryManager as smm
import ClearMap.ParallelProcessing.ParallelTraceback as ptb



###############################################################################
### Mesh generation
###############################################################################

def mesh(graph = None, method = 'tubes', **kwargs):
  if method == 'tubes':
    return mesh_tube(graph, **kwargs);
  else:
    ValueError('Method n%r not valid!' % method);

###############################################################################
### Graph mesh using tubes
###############################################################################

def mesh_tube(graph = None, 
              coordinates = None, radii = None, indices = None,
              color = None, vertex_colors = None, edge_colors = None,
              n_tube_points = None, 
              default_radius = 1, default_color = (0.8, 0.1, 0.1, 1.0),
              processes = None, verbose = False):
  """Construct mesh from edge geometry of a graph."""
  if graph is not None:
    if graph.has_edge_geometry():
      if coordinates is None:
        name = 'coordinates';
      if isinstance(coordinates, str):
        coordinates, indices = graph.edge_geometry(name=name, return_indices=True, as_list=False);
      else:
        raise ValueError('Expected coordinates to by None or str, found %r!' % coordinates)
      
      try:
        if radii is None:
          name = 'radii';
        else:
          name = radii;
        radii = graph.edge_geometry(name=name, return_indices=False, as_list=False);
      except:
        print('No radii found in the graph, using uniform radii = %r!' % default_radius);
        radii = default_radius * np.ones(coordinates.shape[0]);
    else:
      coordinates = graph.vertex_coordinates();
      indices = graph.edge_connectivity().flatten();
      coordinates = np.vstack(coordinates[indices]);
      try:
        radii = graph.vertex_radii();
        radii = radii[indices];
      except:
        print('No radii found in the graph, using uniform radii = %r!' % default_radius);
        radii = default_radius * np.ones(coordinates.shape[0]);
      
      n_edges = graph.n_edges;
      indices = np.array([2*np.arange(0,n_edges), 2*np.arange(1,n_edges+1)]).T;
  
  if vertex_colors is not None or edge_colors is not None:
    color = None;
  
  if color is None and vertex_colors is None:
      color = default_color;  
   
  if vertex_colors is not None:
    connectivity = graph.edge_connectivity();
    edge_colors = (vertex_colors[connectivity[:,0]] + vertex_colors[connectivity[:,1]])/2.0;
  
  vertices, faces, colors = mesh_tube_from_coordinates_and_radii(coordinates, radii, indices, n_tube_points=n_tube_points, edge_colors=edge_colors, processes=None);
  
  return vertices, faces, colors


def mesh_tube_from_coordinates_and_radii(coordinates, radii, indices, n_tube_points = None, edge_colors = None, processes = None, verbose = False):
  """Construct a mesh from the edge geometry of a graph."""  
  coordinates_hdl = smm.insert(coordinates);
  radii_hdl       = smm.insert(radii);
  indices_hdl     = smm.insert(indices);
  
  if n_tube_points is None:
    n_tube_points = 8;
  
  func = ft.partial(_parallel_mesh, coordinates_hdl=coordinates_hdl, radii_hdl=radii_hdl, indices_hdl=indices_hdl, n_tube_points=n_tube_points, verbose=verbose);
  argdata = np.arange(len(indices));
  
  # process in parallel
  pool = smm.mp.Pool(processes = processes);    
  results = pool.map(func, argdata);
  pool.close();
  pool.join();
  
  smm.free(coordinates_hdl);
  smm.free(radii_hdl);
  smm.free(indices_hdl);  

  n_results = len(results);
  
  vertices = [np.reshape(r[0],(-1,3)) for r in results];
  n_indices = [len(v) for v in vertices];
  n_indices = np.cumsum(n_indices);
  n_indices = np.hstack([[0], n_indices]);
  
  if edge_colors is not None:
    colors = [len(v) * [c] for v,c in zip(vertices, edge_colors)];
    colors = np.concatenate(colors);
    #print(colors.shape)
  else:
    colors = None;
  
  vertices = np.concatenate(vertices);
  #print(grid.shape)
  faces = np.concatenate([results[i][1] + n_indices[i] for i in range(n_results)]);
  
  return vertices, faces, colors


def _mesh(coordinates, radii, n_tube_points = 15, dtype = 'uint32'):
  n_coordinates = len(coordinates);
  tangents, normals, binormals = _frenet_frames(coordinates)

  #circular tube
  v = np.arange(n_tube_points, dtype = float) / n_tube_points * 2 * np.pi;
  c = np.cos(v);
  s = np.sin(v);
  
  r = radii[:, np.newaxis];
  n = normals * r;
  b = binormals * r;
  
  #grid shape (npoints, ntube, 3)
  grid = coordinates[:, np.newaxis, :] + c[np.newaxis, :, np.newaxis] * n[:, np.newaxis, :] + s[np.newaxis, :, np.newaxis] * b[:, np.newaxis, :];

  # construct the mesh
  n_segments = n_coordinates - 1;
  jp = np.ones(n_segments*n_tube_points, dtype = int);
  jp[n_tube_points * np.arange(n_segments, dtype = int) - 1] -= n_tube_points;
  
  i1 = np.arange(n_segments*n_tube_points, dtype = int);
  i2 = i1 + n_tube_points;
  i3 = i2 + jp;
  i4 = i1 + jp;
  
  indices = np.array(np.vstack([np.array([i1,i2,i4]).T, np.array([i2, i3, i4]).T]), dtype=dtype);
  
  return grid, indices


def _frenet_frames(coordinates):
  """Calculates and returns the tangents, normals and binormals for a chain of coordinates."""
  npoints = len(coordinates);
  
  epsilon = 0.0001

  # compute tangent vectors for each segment
  tangents = np.roll(coordinates, -1, axis=0) - np.roll(coordinates, 1, axis=0)
  tangents[0]  = coordinates[1] - coordinates[0]
  tangents[-1] = coordinates[-1] - coordinates[-2]
  tangents = (tangents.T / np.linalg.norm(tangents, axis = 1)).T;

  # get initial normal and binormal
  t = np.abs(tangents[0])
  smallest = np.argmin(t)
  normal = np.zeros(3)
  normal[smallest] = 1.

  vec = np.cross(tangents[0], normal)

  normals = np.zeros((npoints, 3));
  normals[0] = np.cross(tangents[0], vec)

  # compute changea along trajectory 
  theta = np.arccos(np.clip(np.sum(tangents[:-1] * tangents[1:], axis = 1),-1,1));
  vec = np.cross(tangents[:-1], tangents[1:]);
  nrm = np.linalg.norm(vec, axis = 1);

  # compute normal and binormal vectors along the path
  for i in range(npoints-1):
    normals[i+1] = normals[i]
    
    if nrm[i] > epsilon:
      v = vec[i] / nrm[i];
      normals[i+1] = trf.rotate(-np.degrees(theta[i]), v)[:3, :3].dot(normals[i+1])
  
  binormals = np.cross(tangents, normals)

  return tangents, normals, binormals


@ptb.parallel_traceback
def _parallel_mesh(i, coordinates_hdl, radii_hdl, indices_hdl, n_tube_points = 15, verbose = False):

  coordinates = smm.get(coordinates_hdl);
  radii       = smm.get(radii_hdl);
  start,end   = smm.get(indices_hdl)[i];
  
  coordinates = coordinates[start:end];
  radii  = radii[start:end];
  
  if verbose:
    if i % 1000 == 0:
      print('Mesh calculation %d / %d.' % (i, len(smm.get(indices_hdl))));
  
  return _mesh(coordinates, radii, n_tube_points)


###############################################################################
### Render graph using intrepolation
###############################################################################

def interpolate_edge_geometry(graph, smooth = 5, order = 2,
                              points_per_pixel = 0.5, 
                              processes = None, verbose = False):
  """Smooth center lines and radii of the edge geometry."""
  if not graph.has_edge_geometry():
    raise ValueError('Graph has no edge geometry!')
  
  coordinates, indices = graph.edge_geometry('coordinates', return_indices=True, as_list=False);
  radii = graph.edge_geometry('radii', as_list=False);
  
  # prepare result arrays
  #indices_interp = np.array([_n_points_per_edge(i[1]-i[0], points_per_pixel=points_per_pixel) for i in indices]);
  #indices_interp = np.cumsum(indices_interp);
  #indices_interp = np.hstack([[0], indices_interp])
  #indices_interp = np.array([indices_interp[:-1], indices_interp[1:]]).T;  
  
  #coordinates_interp = np.zeros((indices_interp[-1,1], coordinates.shape[1]), dtype=float);
  #radii_interp = np.zeros(indices_interp[-1,1], dtype=float);
    
  # process in parallel
  coordinates_hdl = smm.insert(coordinates);
  radii_hdl       = smm.insert(radii);
  indices_hdl     = smm.insert(indices);
  
#  coordinates_interp_hdl = smm.insert(coordinates_interp);
#  radii_interp_hdl       = smm.insert(radii_interp);
#  indices_interp_hdl     = smm.insert(indices_interp);
  
  func = ft.partial(_parallel_interpolate, 
                    coordinates_hdl=coordinates_hdl, radii_hdl=radii_hdl, indices_hdl=indices_hdl,
#                   coordinates_interp_hdl=coordinates_interp_hdl, radii_interp_hdl=radii_interp_hdl, indices_interp_hdl=indices_interp_hdl,   
                    smooth=smooth, order=order, points_per_pixel=points_per_pixel, verbose=verbose);
  argdata = np.arange(len(indices));
  
  pool = smm.mp.Pool(processes = processes);    
  results = pool.map(func, argdata);
  pool.close();
  pool.join();
  
  smm.free(coordinates_hdl);
  smm.free(radii_hdl);
  smm.free(indices_hdl);  
  
#  smm.free(coordinates_interp_hdl);
#  smm.free(radii_interp_hdl);
#  smm.free(indices_interp_hdl);
  
  indices_interp = np.array([len(r[1]) for r in results]);
  indices_interp = np.cumsum(indices_interp);
  indices_interp = np.hstack([[0], indices_interp])
  indices_interp = np.array([indices_interp[:-1], indices_interp[1:]]).T;  
  
  coordinates_interp = np.vstack([r[0] for r in results]);
  radii_interp = np.hstack([r[1] for r in results]);

  return coordinates_interp, radii_interp, indices_interp
   
 
@ptb.parallel_traceback
def _parallel_interpolate(i, coordinates_hdl, radii_hdl, indices_hdl, 
#                             coordinates_interp_hdl, radii_interp_hdl, indices_interp_hdl,
                          smooth = 5, order = 2, points_per_pixel = 0.5, verbose = False):
  coordinates = smm.get(coordinates_hdl);
  radii       = smm.get(radii_hdl);
  start,end   = smm.get(indices_hdl)[i];
  
#  coordinates_interp = smm.get(coordinates_interp_hdl);
#  radii_interp       = smm.get(radii_interp_hdl);
#  start_interp,end_interp = smm.get(indices_interp_hdl)[i];
  
  coordinates = coordinates[start:end];
  radii  = radii[start:end];
  
  if verbose:
    if i % 1000 == 0:
      print('Mesh interpolation %d / %d.' % (i, len(smm.get(indices_hdl))));
  
  #coordinates_interp[start:end], radii_interp[start:end]=
  
  n_points = _n_points_per_edge(end-start, points_per_pixel=points_per_pixel)
  #n_points = end_interp - start_interp;
  
  #coordinates_interp[start_interp:end_interp], radii_interp[start_interp:end_interp] = 
  return _interpolate_edge(coordinates, radii, n_points=n_points,
                           smooth=smooth, order=order);
   
 
import ClearMap.Analysis.Curves.Resampling as crs
  
def _n_points_per_edge(n_pixel, points_per_pixel):
  n_points = int(np.ceil(n_pixel * points_per_pixel));
  n_points = max(2, n_points);
  return n_points;

def _interpolate_edge(coordinates, radii, n_points, smooth = 5, order = 2):
  order = min(coordinates.shape[0]-1, order);
  coordinates_interp = crs.resample(coordinates, n_points=n_points, smooth=smooth, order=order);
  radii_interp = crs.resample(radii, n_points=n_points, smooth=smooth, order=order);
  return coordinates_interp, radii_interp;


###############################################################################
### Tests
###############################################################################

def _test():
  import numpy as np
  import ClearMap.Tests.Files as tf
  import ClearMap.Analysis.Graphs.GraphProcessing as gp
  #reload(gp)

  skeleton = tf.source('skeleton');
  
  #import ClearMap.Visualization.Plot3d as p3d
  #p3d.plot(skeleton)

  #reload(gp)
  g = gp.graph_from_skeleton(skeleton)

  g.has_edge_geometry()
  
  g.vertex_coordinates()
  
  s = g.skeleton()
  assert np.all(s==skeleton)


  gc = gp.clean_graph(g, verbose=True)  
  
  
  gr = gp.reduce_graph(gc, verbose=True)
  
  coordinates, indices = gr.edge_geometry(return_indices=True, as_list=False);
 
  gr.set_edge_geometry(name='radii', values=np.ones(len(coordinates)))

  radii = gr.edge_geometry('radii', as_list=False);

  import ClearMap.Analysis.Graphs.GraphVisualization as gv
  reload(gv)
  
  grid, grid_indices = gv.mesh_from_edge_geometry(coordinates=coordinates, radii=radii, indices=indices);
 
  
  import ClearMap.Visualization.GraphVisual as gvi

  gvi.plot_mesh(grid, grid_indices)  
  