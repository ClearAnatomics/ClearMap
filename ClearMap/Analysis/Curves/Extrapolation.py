# -*- coding: utf-8 -*-
"""
Extrapolation
=============

Method to extend interpolation objects to constantly / linearly extrapolate.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


import numpy as np

import scipy.interpolate

###############################################################################
### Extrapolation
###############################################################################

def extrapolate_1d(x, y, interpolation = 'linear', exterpolation = 'constant'):
    """Interpolate on given values and extrapolate outside the given data
    
    Arguments
    ---------
    x : array
      x values of the data to interpolate.
    y : array
      y values of the data to interpolate.
    interpolation : str 
      Optional interpolation metho. See :func:`scipy.interpolate.interp1d`.
      Default is 'linear'.
    exterpolation :
      Optional interpolation method, either "linear" or "constant".
    
    Returns
    -------
    extrapolator : function
      Inter- and extra-polation function.
    """
        
    interpolator = scipy.interpolate.interp1d(x, y, kind = interpolation);
    return extrapolate_from_interpolator_1d(interpolator, exterpolation);   


def  extrapolate_from_interpolator_1d(interpolator, exterpolation = 'constant'):
    """Extend interpolation function to extrapolate outside the given data
    
    Arguments
    ---------
    interpolator : function
      Interpolating function, see e.g. :func:`scipy.interpolate.interp1d`.
    exterpolation : str
      Optional interpolation method, either "linear" or "constant".
    
    Returns
    -------
    extrapolator : function
      Inter- and extra-polation function.
    """    
    xs = interpolator.x
    ys = interpolator.y
    cs = (exterpolation == 'constant');

    def pointwise(x):
        if cs:   #constant extrapolation
            if x < xs[0]:
                return ys[0];
            elif x > xs[-1]:
                return ys[-1];
            else:
                return interpolator(x);
        else:  # linear extrapolation
            if x < xs[0]:
                return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
            elif x > xs[-1]:
                return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
            else:
                return interpolator(x)

    def extrapfunc(xs):
        return np.array(map(pointwise, np.array(xs)))

    return extrapfunc

