# -*- coding: utf-8 -*-
"""
Plot3d Module
=============

Plotting routines for 3d display of data.

Note
----
This module is using vispy.
"""
__author__    = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__   = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__copyright__ = 'Copyright (c) 2017 by Christoph Kirst, The Rockefeller University, New York City'

import numpy as np

import vispy

#try:
#  vispy.use(app='PyQt5');  #avoid use of ipynb_webgl
#except:
#  print('qt')
#  try:
#    vispy.use(app='PySide')
#  except:
#    print('no')
#    print(vispy.app.backends.BACKENDMAP)

import vispy.app
import vispy.scene
#vispy.app.run();


import ClearMap.Visualization.Vispy.TurntableCamera as ttc
import ClearMap.Visualization.Vispy.VolumeVisual as vvi

import ClearMap.Visualization.Color as col

import ClearMap.IO.IO as io

###############################################################################
### 3d plotting
###############################################################################

def list_line_plot_3d(coordinates, view = None, title = None, center_view = True, **kwargs):
  """Plot lines between coordinates in 3d.
  
  Arguments
  ---------
  coordaintes : array
    Coordinate nx3 array.
  title : str or None
    Window title.
  view : view or None
    Add plot to this view. if given.
    
  Returns
  -------
  view : view
    The view of the plot.
  """ 
  #visual
  Plot3D = vispy.scene.visuals.create_visual_node(vispy.visuals.LinePlotVisual)

  #view
  title = title if title is not None else 'list_line_plot_3d';  
  view = initialize_view(view, title=title);
  
  # style
  style = dict(width=2.0, color='red', symbol='o',
               edge_color='w', face_color = 'blue');
  style.update(**kwargs);
  
  #plot
  p = Plot3D(coordinates, parent=view.scene, **style)
  p.set_gl_state('translucent', blend=True, depth_test=True);
  #view.camera.set_range();
  
  if center_view:
    _center_view(view);
  
  return p;


def list_plot_3d(coordinates, view = None, title = None, center_view = True, color = None, **kwargs):
  """Scatter plot of points in 3d.
  
  Arguments
  ---------
  coordaintes : array
    Coordinatenx3 array.
  title : str or None
    Window title.
  view : view or None
    Add plot to this view. if given.
    
  Returns
  -------
  view : view
    The view of the plot.
  """ 
  #visual
  Scatter3D = vispy.scene.visuals.create_visual_node(vispy.visuals.MarkersVisual)
  
  #view
  title = title if title is not None else 'list_plot_3d';
  view = initialize_view(view, title=title);
  
  #style
  if color and 'face_color' not in kwargs.keys():
    kwargs.update(face_color=color);
  if color and 'edge_color' not in kwargs.keys():
    kwargs.update(edge_color=color);
  style = dict(face_color='blue', symbol='o', size=10,
               edge_width=0.5, edge_color='blue')
  style.update(**kwargs)
  
  # plot
  p = Scatter3D(parent=view.scene);
  p.set_gl_state('translucent', blend=True, depth_test=True);
  p.set_data(coordinates, **style)
  #view.camera.set_range();
  
  if center_view:
    _center_view(view);
  
  return p;
    

def plot_3d(source, colormap = None, view = None, title = None, center_view = True, **kwargs):  
  """Plot 3d volume.
  
  Arguments
  ---------
  source : array
    The 3d volume.
  title : str or None
    Window title.
  view : view or None
    Add plot to this view. if given.
    
  Returns
  -------
  view : view
    The view of the plot.
  """
  #visual
  #VolumePlot3D = vispy.scene.visuals.create_visual_node(vispy.visuals.VolumeVisual)
  VolumePlot3D = vispy.scene.visuals.create_visual_node(vvi.VolumeVisual)
  
  #view
  title = title if title is not None else 'plot_3d';
  #center = (np.array(source.shape) // 2);
  view = initialize_view(view, title=title, fov=0, depth_value=10**8)#, center=center)
  
  #style
  style = dict(cmap = grays_alpha(), method = 'translucent', relative_step_size=0.5)
  style.update(**kwargs);
  
  #source
  source = io.as_source(source)[:];
  if source.dtype==bool:
    source = source.view(dtype='uint8');
  
  #orient 
  source = source.transpose([2,1,0]);
  
  #plot
  p = VolumePlot3D(source, parent=view.scene, **style);
  #view.camera.set_range();
  
  if center_view:
    _center_view(view);
  
  return p;


def plot_mesh_3d(coordinates, faces, view = None, shading='smooth', color = None, face_colors = None, vertex_colors = None, mode = 'triangles', center_view = True,  title = None, **kwargs):
  """Plot a 3d mesh.
  
  Arguments
  ---------
  coordinates : array
    Coordinate nx3 array.
  faces : array
    Indices of triangular faces, nx3 array.
  title : str or None
    Window title.
  view : view or None
    Add plot to this view. if given.
    
  Returns
  -------
  view : view
    The view of the plot.
  """ 

   # build visuals
  MeshVisual = vispy.scene.visuals.create_visual_node(vispy.visuals.mesh.MeshVisual)

  #view
  title = title if title is not None else 'plot_mesh_3d';  
  view = initialize_view(view, title=title);
  
  #print view
  if color is None and face_colors is None and vertex_colors is None:
    color ='red';
  
  p = MeshVisual(coordinates, faces, parent=view.scene, shading=shading, color=color, face_colors=face_colors, vertex_colors=vertex_colors, mode=mode, **kwargs)
  
  if center_view:
    view.camera.center = np.mean(coordinates, axis=0);
  
  return p;


def plot_regular_polygon(center, sides=4, title=None, view=None, color='red', border_color=None, border_width=1, radius=1.0,  **kwargs):
  
  # build visuals
  PolyVisual = vispy.scene.visuals.create_visual_node(vispy.visuals.RegularPolygonVisual)

  #view
  title = title if title is not None else 'plot_cube';  
  view = initialize_view(view, title=title);
  
  #print view
  
  p = PolyVisual(center,color=color, border_color=border_color, border_width=border_width, radius=radius, sides=sides, parent=view.scene, **kwargs)
  
  if center_view:
    view.camera.center = center;
  
  return p;



def plot_box(lower, upper, face_color=(1,0,0,0.5), line_color=None, line_width=1, 
             line_padding = 0, shading='smooth', mode = 'triangles', 
             title=None, view=None, center_view = True, **kwargs):
  """Plots a box in 3d."""
  #TODO: return compound visual
  
  #view
  title = title if title is not None else 'plot_box';  
  view = initialize_view(view, title=title);
  
  #corners
  corners = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],
             [0,0,1],[1,0,1],[1,1,1],[0,1,1]];
  corners = [[lower[d] if c[d] == 0 else upper[d] for d in range(3)] for c in corners]             
  corners = np.array(corners, dtype=float);
  
  visuals = [];
  
  if face_color is not None:
    # build visuals
    MeshVisual = vispy.scene.visuals.create_visual_node(vispy.visuals.mesh.MeshVisual)
    
    faces = [[0,1,5],[0,5,4],
             [1,2,6],[1,6,5],
             [4,5,6],[4,6,7],
             [3,0,4],[3,4,7],
             [2,3,7],[2,7,6],
             [3,2,1],[3,1,0]]
    faces = np.array(faces);
    
    mv = MeshVisual(corners, faces, parent=view.scene, shading=shading, color=face_color, mode=mode, **kwargs);
    visuals.append(mv);
  
  if line_color is not None:
    #visual
    LineVisual = vispy.scene.visuals.create_visual_node(vispy.visuals.LineVisual)
    
    lines = np.array([[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]);
    
    if line_padding != 0:
      corners = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                 [0,0,1],[1,0,1],[1,1,1],[0,1,1]];
      corners = [[lower[d]-line_padding if c[d] == 0 else upper[d]+line_padding for d in range(3)] for c in corners]             
      corners = np.array(corners, dtype=float);
    
    lv = LineVisual(corners, parent=view.scene, connect=lines, color=line_color, width=line_width);
    visuals.append(lv);
  
  if center_view:
    view.camera.center = 0.5 * (np.array(lower) + np.array(upper));
  
  return view;


###############################################################################
### Helpers
###############################################################################

#TODO: vispy.color.ColorMap([col1,col2,...]) is much easier!

def single_color_colormap(color = (1,1,1), alpha = 0.075, inverse_alpha = False):
  
  color = tuple(col.color(color, alpha=alpha));
  if inverse_alpha:
    opacity = lambda t : 1-t;
  else:
    opacity = lambda t : t
  
  class SingleColor(vispy.color.colormap.BaseColormap):
    glsl_map = """
    vec4 grays(float t) {
      return vec4(%g * t, %g * t, %g * t, %g * %s);
    }
    """  % (color + ('(1-t)' if inverse_alpha else 't',))
    
    def map(self, t):
      if isinstance(t, np.ndarray):
        return np.hstack([color[0] * t, color[1] * t, color[2] * t, color[3] * opacity(t)]).astype(np.float32)
      else:
        return np.array([color[0] * t, color[1] * t, color[2] * t, color[3] * opacity(t)], dtype=np.float32)
      
  return SingleColor();


def grays_alpha(alpha = 0.075, inverse=False):
  return single_color_colormap(color = (1,1,1), alpha=alpha, inverse_alpha=inverse);





def get_view(view):
  """Return the view of the argument.
  
  Arguments
  ---------
  view : view
    The vispy window.
    
  Returns
  -------
  view : view
    The vispy view of the plot.
  """
  while not isinstance(view, vispy.scene.widgets.viewbox.ViewBox) and hasattr(view, 'parent'):
    view = view.parent;
  
  return view;


def add_axes(view):
  """Add axes to a plot.
  
  Arguments
  ---------
  view : view
    The vispy window.
    
  Returns
  -------
  view : view
    The vispy view of the plot.
  """
  view = initialize_view(view);
  axes = vispy.scene.visuals.XYZAxis(parent=view.scene);

  return axes;


def set_light_to_camera(mesh):
  """Set the light direction to the crrent camera position.
  
  Arguments
  ---------
  mesh : mesh
    A vispy mesh.
  """
  v = get_view(mesh);
  a = np.deg2rad(v.camera.azimuth);
  e = np.deg2rad(v.camera.elevation);
  direction = (np.cos(a) * np.cos(e), np.sin(a) * np.cos(e), np.sin(e));
  #direction = (np.sin(a) * np.cos(e), np.cos(a) * np.cos(e), np.sin(e));
  direction = tuple(-d for d in direction)
  mesh.light_dir = direction;


def set_background(view, color):
  """Set the background color of the view.
  
  Arguments
  ---------
  view : view
    The vispy window.
  color : color specification
    The color for the background.
    
  Returns
  -------
  view : view
    The vispy view of the plot.
  """
  view = get_view(view);
  view.canvas.bgcolor = col.color(color, alpha=True, as_int=False);
  return view;


def center_view(view):
  """Center the camera in a plot.
  
  Arguments
  ---------
  view : view
    The vispy window.
    
  Returns
  -------
  view : view
    The vispy view of the plot.
  """ 
  view = get_view(view);
  bounds = view.get_scene_bounds();
  center = [(b[1] - b[0])*0.5 for b in bounds]
  view.camera.center = center;

_center_view = center_view;


def get_view_parameter(view):
  view = get_view(view);
  return view.camera.get_state();

def set_view_parameter(view, parameter):
  view = get_view(view);
  view.camera.set_state(parameter);


def initialize_view(view=None, title=None, fov=None, distance=None, elevation=None, azimuth=None, center=None, depth_value=None):
  """Return a deafult view."""
  if view is None:
    # build canvas
    canvas = vispy.scene.SceneCanvas(keys='interactive', title=title, show=True, bgcolor='white')
    
    view = canvas.central_widget.add_view(camera = ttc.TurntableCamera())
    view.camera = 'turntable'
    if fov is not None:
      view.camera.fov = fov
    if distance is not None:
      view.camera.distance = distance
    if elevation is not None:
      view.camera.elevation = elevation
    if azimuth is not None:
      view.camera.azimuth = azimuth;
    if center is not None:
      view.camera.center = center;
    if depth_value is not None:
      view.camera.depth_value = depth_value;
      
  else:
    view = get_view(view);

  return view;


def save(location, view, transparent = None, *args, **kwargs):
  """Save the current view to a file."""
  canvas = get_view(view).canvas;
  img = canvas.render(*args, **kwargs);
  if transparent is not None:
      t = np.logical_and(img[:,:,0] >= transparent, img[:,:,1] >= transparent);
      t = np.logical_and(img[:,:,2] >= transparent, t)
      img[:,:,3][t]=0;
      img = img.transpose([0,1,2]);
  else:
    img = img[:,:,:3].T;
  return io.write(location, img);



###############################################################################
### Tests
###############################################################################
    
def _test():
  import numpy as np
  import ClearMap.Visualization.Vispy.Plot3d as p3d;
  #reload(p3d)
  
  #plot
  coordinates = np.random.rand(40,3)
  p1 = p3d.list_plot_3d(coordinates, title='test')  
  
  #plot something on top
  coordinates = np.random.rand(30,3);
  p2 = p3d.list_line_plot_3d(coordinates, color = 'white', face_color='red', view = p1) #analysis:ignore
  
  #volumetric plot
  import numpy as np
  shape = (31,31,31);
  binary = np.zeros(shape, dtype = bool);
  grid = np.meshgrid(*[range(s) for s in shape], indexing='ij');
  center = tuple(s/2 for s in shape);
  distance = np.sum([(g-c)**2 for g,c in zip(grid, center)], axis=0);
  binary[distance <= 10**2] = True
  p3 = p3d.plot_3d(binary) #analysis:ignore

  #reload(p3d)
  v = p3d.plot_box((0,0,0),(1,2,3), face_color = (1,0,0,0), line_color='white', line_width=2, line_padding=0.015)
  print(v)


