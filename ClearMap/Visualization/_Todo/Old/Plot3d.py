# -*- coding: utf-8 -*-
"""
Plotting routines for 3d display of data
"""
__author__    = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__   = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'

import numpy as np

import vispy
import vispy.app
import vispy.scene
vispy.app.run();


def getView(view):
  while not isinstance(view, vispy.scene.widgets.viewbox.ViewBox) and hasattr(view, 'parent'):
    view = view.parent;
  
  return view;
    

class graysAlpha(vispy.color.colormap.BaseColormap):
    glsl_map = """
    vec4 grays(float t) {
        return vec4(t, t, t, 0.075 * t);
    }
    """
    
    def map(self, t):
        if isinstance(t, np.ndarray):
            return np.hstack([t, t, t, 0.075 * t]).astype(np.float32)
        else:
            return np.array([t, t, t, 0.075 * t], dtype=np.float32)



def listPlot3d(x,y,z, view = None):
  Plot3D = vispy.scene.visuals.create_visual_node(vispy.visuals.LinePlotVisual)

  # Add a ViewBox to let the user zoom/rotate
  if view is None:
    # build canvas
    canvas = vispy.scene.SceneCanvas(keys='interactive', title='listplot3d', show=True)
    
    view = canvas.central_widget.add_view()
    view.camera = 'turntable'
    view.camera.fov = 45
    view.camera.distance = 6
  else:
    view = getView(view);
  
  # plot
  pos = np.c_[x, y, z]
  return Plot3D(pos, width=2.0, color='red',
                edge_color='w', symbol='o', face_color=(0.2, 0.2, 1, 0.8),
                parent=view.scene)


def plot3d(data, colormap = graysAlpha(), view = None):  
  VolumePlot3D = vispy.scene.visuals.create_visual_node(vispy.visuals.VolumeVisual)
  
  # Add a ViewBox to let the user zoom/rotate
  if view is None:
    # build canvas
    canvas = vispy.scene.SceneCanvas(keys='interactive', title='plot3d', show=True)

    view = canvas.central_widget.add_view()
    view.camera = 'turntable'
    view.camera.fov = 0
    view.camera.distance = 7200
    view.camera.elevation = 31
    view.camera.azimuth = 0
    
    view.camera.depth_value = 100000000
    
    cc = (np.array(data.shape) // 2);
    #cc = cc[[2,1,0]]
    view.camera.center = cc;
  else:
    view = getView(view);
  
  return VolumePlot3D(data.transpose([2,1,0]), method = 'translucent', relative_step_size=0.5, parent=view.scene, cmap = colormap)
