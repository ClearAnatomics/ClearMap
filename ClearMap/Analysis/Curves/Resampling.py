# -*- coding: utf-8 -*-
"""
Resampling
==========

Module with routines for data and curve resampling and smoothing based on splines.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


import numpy as np
from scipy.interpolate import splprep,splrep, splev


def resample_nd(curve, n_points = None, smooth = 0, periodic = False, derivative = 0, order = 5, iterations = 1):
  """Resample n points using n equidistant points along a curve
  
  Arguments
  ---------
  curve : mxd array
    Coordinates for the reference points of the curve.
  n_points : int
    Number of equidistant points to evaluate the curve on.
  smooth : int or float
    Smoothness factor.
  periodic : bool
    If True, assume the curve is a closed loop.
  derivative : int
    If > 0, return the n-th derivative of the curve.
  order : int
    The spline order to use to interpolate the curve.
  iterations : int
    Iterate the resampling process this amount of time.
  
  Returns
  -------
  curve : nxd array
    Resampled curve along n_points equidistant points.
  """
  if n_points is None or n_points is all:
    n_points = curve.shape[0];  
  for i in range(iterations):  
    cinterp, u = splprep(curve.T, u=None, s=smooth, per=periodic, k=order);
    us = np.linspace(u.min(), u.max(), n_points)
    curve = np.vstack(splev(us, cinterp, der=derivative)).T;
  return curve;


def resample_1d(data, n_points = None, smooth = 0, periodic = False, derivative = 0, order = 5, iterations = 0):
  """Resample 1d data using n equidistant points
  
  Arguments
  ---------
  curve : mxd array
    Coordinates for the reference points of the curve.
  n_points : int
    Number of equidistant points to evaluate the curve on.
  smooth : int or float
    Smoothness factor.
  periodic : bool
    If True, assume the curve is a closed loop.
  derivative : int
    If > 0, return the n-th derivative of the curve.
  order : int
    The spline order to use to interpolate the curve.
  iterations : int
    Iterate the resampling process this amount of time.
  
  Returns
  -------
  curve : nxd array
    Resampled curve along n_points equidistant points.
  """
  if n_points is None or n_points is None:
    n_points = data.shape[0];

  u0 = np.linspace(0, 1, data.shape[0]);
  us = np.linspace(0, 1, n_points);
  for i in range(iterations):
    dinterp = splrep(u0, data, s=smooth, per=periodic, k=order);
    data = splev(us, dinterp, der = derivative);
  return data;


def resample(curve, n_points = None, smooth = 0, periodic = False, derivative = 0, order = 5, iterations = 1):
  """Resample a curve using equidistant points along a curve.
  
  Arguments
  ---------
  curve : mxd array
    Coordinates for the reference points of the curve.
  n_points : int
    Number of equidistant points to evaluate the curve on.
  smooth : int or float
    Smoothness factor.
  periodic : bool
    If True, assume the curve is a closed loop.
  derivative : int
    If > 0, return the n-th derivative of the curve.
  order : int
    The spline order to use to interpolate the curve.
  iterations : int
    Iterate the resampling process this amount of time.
  
  Returns
  -------
  curve : nxd array
    Resampled curve along n_points equidistant points.
  """
  if curve.ndim > 1:
    return resample_nd(curve, n_points, smooth=smooth, periodic=periodic, derivative=derivative, order=order, iterations=iterations);
  else:
    return resample_1d(curve, n_points, smooth=smooth, periodic=periodic, derivative=derivative, order=order, iterations=iterations);



def test():
  import numpy as np
  import matplotlib.pyplot as plt
  import ClearMap.Analysis.Curves.Resampling as res
  #reload(res)
  
  curve = np.linspace(0,10,50);
  curve = np.vstack([curve, np.sin(curve)]).T;
  
  rcurve = res.resample(curve, n_points=150, smooth=0);
  plt.figure(1); plt.clf();
  plt.plot(rcurve[:,0], rcurve[:,1], 'red');
  plt.plot(curve[:,0], curve[:,1], 'blue');
  
  curve1d = np.sin(np.linspace(0,1,50) * 2 * np.pi);
  rcurve1d = res.resample(curve1d, n_points=150, smooth=0);
  plt.figure(2); plt.clf();
  plt.plot(curve1d);
  plt.plot(rcurve1d);
  
