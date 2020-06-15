"""
Icosphere
=========

Generates a mesh of a icosphere.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


import numpy as np


def icosphere(radius = 1.0, center = (0,0,0), n_refinements = 0):
  #construct icosahedron 
  r = (1.0 + np.sqrt(5.0)) / 2.0;
  vertices = np.array([
      [-1.0,   r, 0.0],
      [ 1.0,   r, 0.0],
      [-1.0,  -r, 0.0],
      [ 1.0,  -r, 0.0],
      [0.0, -1.0,   r],
      [0.0,  1.0,   r],
      [0.0, -1.0,  -r],
      [0.0,  1.0,  -r],
      [  r, 0.0, -1.0],
      [  r, 0.0,  1.0],
      [ -r, 0.0, -1.0],
      [ -r, 0.0,  1.0],
      ], dtype=float);
  vertices = list(vertices)
  
  faces = [
      [0, 11, 5],
      [0, 5, 1],
      [0, 1, 7],
      [0, 7, 10],
      [0, 10, 11],
      [1, 5, 9],
      [5, 11, 4],
      [11, 10, 2],
      [10, 7, 6],
      [7, 1, 8],
      [3, 9, 4],
      [3, 4, 2],
      [3, 2, 6],
      [3, 6, 8],
      [3, 8, 9],
      [5, 4, 9],
      [2, 4, 11],
      [6, 2, 10],
      [8, 6, 7],
      [9, 8, 1],
      ];
  
  for n in range(n_refinements):
    vertices, faces = _sub_divide_faces(vertices, faces);
  
  length = np.linalg.norm(vertices, axis=1);
  vertices = (np.array(vertices).T / length * radius).T + center;
  faces = np.array(faces);
  
  return vertices, faces

    
def _sub_divide_faces(vertices, faces):  
  new_vertices = list(vertices);
  new_faces = [];
  n = len(new_vertices);
  
  for face in faces:
    new_vertices += [0.5 * (vertices[face[i]] + vertices[face[j]]) for i,j in zip([0,1,2],[1,2,0])];
    new_faces += [[face[0], n, n+2], [n, face[1], n+1], [n+1, face[2], n+2],[n, n+1, n+2]];
    n += 3;
  
  return new_vertices, new_faces;

  
def _test():
  import ClearMap.Visualization.Mesh.IcoSphere as ics
  from importlib import reload
  reload(ics)
  
  vertices, faces = ics.icosphere(n_refinements = 2);
  
  import ClearMap.Visualization.Plot3d as p3d
  p3d.plot_mesh_3d(vertices, faces)


