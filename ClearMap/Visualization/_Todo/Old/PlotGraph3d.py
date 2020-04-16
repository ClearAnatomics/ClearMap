# -*- coding: utf-8 -*-
"""
Plot graphs in 3d
"""

#%%

import numpy as np

import vispy
import vispy.scene


import matplotlib.pyplot as plt

import ClearMap.Analysis.Graphs as gr

from ClearMap.Visualization.Plot3d import getView

import ClearMap.Visualization.Color as col


### plotting
def plot(graph, color = (0.8, 0.1, 0.1, 0.25), center = np.mean, color_map = None, view = None):
  
  #extract graph data  
  pos = graph.vertexCoordinates();
  if center is not None:
    pos = pos - center(pos, axis = 0); 

  connect = np.asarray(graph.edges(), dtype = np.int32);

  # build visuals
  GraphPlot3D = vispy.scene.visuals.create_visual_node(vispy.visuals.LineVisual)

  if view is None:
    # build canvas
    canvas = vispy.scene.SceneCanvas(keys='interactive', title='plot3d', show=True)

    # Add a ViewBox to let the user zoom/rotate
    view = canvas.central_widget.add_view()
    view.camera = 'turntable'
    view.camera.fov = 85
    view.camera.distance = 0
    view.camera.elevation = 0
    view.camera.azimuth = 0;
  else:
    view = getView(view);
  
  color_map = col.colormap(color_map);
    
  if color is not None and color in graph.vertexProperties().keys():
    color = np.array(graph.vertexProperty(color), dtype = float);
  
  if isinstance(color, np.ndarray) and color.ndim == 1:
    color = color - color.min();
    color = color / color.max();
    color = color_map(color);
  
  return GraphPlot3D(pos = pos, connect = connect,
                     width=2.0, color = color, parent=view.scene)




if __name__ == '__main__':
  import ClearMap.Analysis.Graphs.Graph as gr
  import ClearMap.Visualization.PlotGraph3d as pg3
  reload(pg3)
  #g = gr.load('/home/ckirst/Desktop/Vasculature/Analysis_2018_03_27/stitched_graph_transformed.gt')
  g = gr.load('/home/ckirst/Science/Projects/WholeBrainClearing/Vasculature/Experiment/Graphs_2018_05/graph_reduced.gt')
  
  pg3.plot(g)