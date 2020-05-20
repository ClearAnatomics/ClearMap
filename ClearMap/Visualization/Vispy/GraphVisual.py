# -*- coding: utf-8 -*-
"""
GraphVisual
===========

Module providing Graph visuals for rendering graphs.

Note
----
This module is porviding vispy visuals only. 
See :mod:`PlotGraph3d` module for plotting.
"""
__author__    = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__   = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__copyright__ = 'Copyright (c) 2017 by Christoph Kirst, The Rockefeller University, New York City'


import numpy as np

import vispy.visuals as visuals

import ClearMap.Analysis.Graphs.GraphRendering as gr  
import ClearMap.Visualization.Color as col


###############################################################################
### Graph visuals
###############################################################################

default_color = (0.8, 0.1, 0.1, 1.0);

class GraphLineVisual(visuals.LineVisual):
    """Displays a graph in 3d using tube rendering for the edges
    """
    
    def __init__(self, graph, 
                 coordinates = None,
                 color = None, vertex_colors = None, edge_colors = None, 
                 width = None, mode = 'gl'):
     
      connectivity = graph.edge_connectivity();
      
      if coordinates is None:
        name = 'coordinates'
      else:
        name = coordinates;
      coordinates = graph.vertex_property(name);
      
      if vertex_colors is not None or edge_colors is not None:
        color = None;
      if color is None:
        if edge_colors is None and vertex_colors is None:
          color = default_color;
        elif vertex_colors is not None:
          if isinstance(vertex_colors, np.ndarray) and vertex_colors.ndim == 2:
            color = vertex_colors;
          else:
            color = col.color(vertex_colors, alpha=True);
        elif edge_colors is not None:
          if isinstance(edge_colors, np.ndarray) and edge_colors.ndim == 2:
            #need a vertex pair for every edge if the color is different
            coordinates = coordinates[connectivity.flatten()];
            connectivity = np.arange(coordinates.shape[0]);
            connectivity = connectivity.reshape((-1,2));
            indices = np.arange(len(edge_colors));
            indices = np.array([indices, indices]).T.flatten();
            color = edge_colors[indices];
          else:
            color = col.color(edge_colors, alpha=True); 
      else:
        color = col.color(color, alpha=True);

      if width is None:
        width = 1;
      
      visuals.LineVisual.__init__(self, coordinates, connect=connectivity, 
                                  color=color, width=width, method=mode) 
  

class GraphMeshVisual(visuals.mesh.MeshVisual):
  """Displays a graph in 3d using tube rendering for the edges
  """
  
  def __init__(self, graph, 
               coordinates = None, radii = None,
               n_tube_points = 8, default_radius = 1, 
               color = None, vertex_colors = None, edge_colors = None, 
               mode = 'triangles', shading = 'smooth'):
    
    if vertex_colors is not None or edge_colors is not None:
      color = None;
    
    if color is None and vertex_colors is None:
        color = default_color;
    
    if graph.has_edge_geometry():
      if coordinates is None:
        name = 'coordinates';
      else:
        name = coordinates;
      coordinates, indices = graph.edge_geometry(name=name, return_indices=True, as_list=False);

      #calculate mesh
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
      #print(radii.shape, indices.shape, coordinates.shape)
    
    if vertex_colors is not None:
      connectivity = graph.edge_connectivity();
      edge_colors = (vertex_colors[connectivity[:,0]] + vertex_colors[connectivity[:,1]])/2.0;
    
    vertices, faces, vertex_colors = gr.mesh_tube_from_coordinates_and_radii(coordinates, radii, indices, n_tube_points=n_tube_points, edge_colors=edge_colors, processes=None);
    
    visuals.mesh.MeshVisual.__init__(self, vertices, faces, 
                                     color=color, vertex_colors=vertex_colors, 
                                     shading=shading, mode=mode)


###############################################################################
### Tests
###############################################################################

def _test():
  import numpy as np
  import vispy
  import ClearMap.Analysis.Graphs.GraphGt as ggt
  import ClearMap.Visualization.Vispy.Plot3d as p3d
  import ClearMap.Visualization.Vispy.GraphVisual as gv
  #reload(gv)
  
  g = ggt.Graph();
  g.add_vertex(5);
  g.add_edge([[0,1],[1,2],[2,3],[3,4],[4,0]]);
  g.set_vertex_coordinates(20*np.random.rand(5,3))
  
  v = vispy.scene.visuals.create_visual_node(gv.GraphLineVisual)
  p = v(g, parent=p3d.initialize_view().scene)
  p3d.center(p)

  v = vispy.scene.visuals.create_visual_node(gv.GraphMeshVisual)
  p = v(g, parent=p3d.initialize_view().scene)
  p3d.center(p)


  