# -*- coding: utf-8 -*-
"""
PlotUtils Module
================

Plotting routines for ClearMap based on matplotlib.

Note
----
This module is using matplotlib.
"""
__author__    = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__   = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__copyright__ = 'Copyright (c) 2017 by Christoph Kirst, The Rockefeller University, New York City'

import numpy as np

import matplotlib as mpl; # analysis:ignore
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # analysis:ignore

import scipy.stats as st

#%%############################################################################
### Density and Countour plots
###############################################################################

def plot_density(points, plot_points=True, plot_contour=True, plot_contour_lines=10, n_bins = 100, color = None, cmap = plt.cm.gray, xlim = None, ylim = None, verbose = False):
  """Plot point distributions."""
  
  if xlim is None:
    xlim = (np.min(points[:,0]), np.max(points[:,0]));
  if ylim is None:
    ylim = (np.min(points[:,1]), np.max(points[:,1])); 
  xmin,xmax = xlim;
  ymin,ymax = ylim;
  
  # kernel density estimates
  dx = float(xmax-xmin) / n_bins;
  dy = float(ymax-ymin) / n_bins;
  xx, yy = np.mgrid[xmin:xmax:dx, ymin:ymax:dy]
  positions = np.vstack([xx.ravel(), yy.ravel()])
  kernel = st.gaussian_kde(points.T);
  if verbose:
    print('plot_density: kernel estimation done.');
  density = kernel(positions)
  density = np.reshape(density, xx.shape)
  if verbose:
    print('plot_density: density estimation done.');

  # figure setup
  ax = plt.gca()
  ax.set_xlim(xmin, xmax)
  ax.set_ylim(ymin, ymax)
  
  # contour plot
  if plot_contour:
    cfset = ax.contourf(xx, yy, density, cmap=cmap);
  if plot_points:
    plt.scatter(points[:, 0], points[:, 1], c=color, cmap=cmap, s=1)
  if plot_contour_lines is not None:
    cset = ax.contour(xx, yy, density, plot_contour_lines, cmap=cmap)
  
  return (kernel, (xx,yy,density))


def plot_curve(coordinates, **kwargs):
  """Plot a curve in 3d.
  
  Arguments
  ---------
  coordinates : nx3 array.
    The coordinates of the curve.
  **kwargs : matplotlib parameter.
  
  Returns
  -------
  ax : ax
    3d axes object.
  """
  ax = plt.gca(projection='3d');
  x,y,z = coordinates.T;
  ax.plot(x, y, z, **kwargs)
  return ax;

