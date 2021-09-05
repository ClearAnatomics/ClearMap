# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 01:34:09 2018

@author: ckirst
"""

import numpy as np;

import pyqtgraph as pg


def screen_geometry(screen = None):
  if screen is None:
    screen = -1;
  g = pg.QAPP.screens()[screen].geometry()
  return (g.left(), g.top(), g.width(), g.height())


def tiled_layout(n_windows, origin = None, shape = None, percent = None, screen = None):
  
  if origin is None or shape is None:
    geometry = screen_geometry();
  if origin is None:
    origin = geometry[:2];
  if shape is None:
    shape = geometry[2:];
  width, height = shape;
  x0, y0 = origin;
  
  if percent is not None:
    width = int(width / 100.0 * percent);
    height = int(height / 100.0 * percent);
  
  if n_windows <= 3:
    nx = n_windows;
    ny = 1;
  else:
    nx = int(np.ceil(np.sqrt(n_windows)));
    ny = int(np.ceil(n_windows*1.0/nx));
  
  x = np.array(np.linspace(0, width, nx+1), dtype = int);
  y = np.array(np.linspace(0, height, ny+1), dtype = int);
  
  geo = [];
  ix = 0;
  iy = 0;
  for i in range(n_windows):
    geo.append([x[ix] + x0, y[iy] + y0, x[ix+1]-x[ix], y[iy+1]-y[iy]]);
    ix+=1;
    if ix == nx:
      ix = 0;
      iy += 1;
  
  return geo;
  
  


if __name__ == '__main__':
  import ClearMap.GUI.Utility as guiu;
  from importlib import reload
  reload(guiu)
  
  print(guiu.screenResolution())
  
  w,h = guiu.screenResolution();
  
  print(guiu.tiledLayout(3))