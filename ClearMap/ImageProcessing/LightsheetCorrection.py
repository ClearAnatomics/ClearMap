# -*- coding: utf-8 -*-
"""
LightsheetCorrection
====================

Module to remove lightsheet artifacts in images.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'

import numpy as np

import ClearMap.ImageProcessing.Filter.Rank as rnk
import ClearMap.ImageProcessing.LocalStatistics as ls

import ClearMap.Utils.Timer as tmr

###############################################################################
### Lightsheet correction
###############################################################################

def correct_lightsheet(source, percentile = 0.25, max_bin=2**12, mask=None,
                       lightsheet = dict(selem = (150,1,1)), 
                       background = dict(selem = (200,200,1), spacing = (25,25,1), interpolate = 1, dtype = float, step = (2,2,1)),
                       lightsheet_vs_background = 2, return_lightsheet = False, return_background = False, verbose = True):
  """Removes lightsheet artifacts.
  
  Arguments
  ---------
  source : array
    The source to correct.
  percentile : float in [0,1]
    Ther percentile to base the lightsheet correction on.
  max_bin : int 
    The maximal bin to use. Max_bin needs to be >= the maximal value in the 
    source.
  mask : array or None
    Optional mask.
  lightsheet : dict
    Parameter to pass to the percentile routine for the lightsheet artifact
    estimate. See :func:`ImageProcessing.Filter.Rank.percentile`.
  background : dict
    Parameter to pass to the percentile rouitne for the background estimation.
  lightsheet_vs_background : float
    The background is multiplied by this weight before comparing to the
    lightsheet artifact estimate.
  return_lightsheet : bool
    If True, return the lightsheeet artifact estimate.
  return_background : bool
    If True, return the background estimate.
  verbose : bool
    If True, print progress information.
  
  Returns
  -------
  corrected : array
    Lightsheet artifact corrected image.
  lightsheet : array
    The lightsheet artifact estimate.
  background : array
    The background estimate.
  
  Note
  ----
  The routine implements a fast but efftice way to remove lightsheet artifacts.
  Effectively the percentile in an eleoganted structural element along the 
  lightsheet direction centered around each pixel is calculated and then
  compared to the percentile in a symmetrical box like structural element 
  at the same pixel. The former is an estimate of the lightsheet artifact 
  the latter of the backgrond. The background is multiplied by the factor 
  lightsheet_vs_background and then the minimum of both results is subtracted
  from the source.
  Adding an overall background estimate helps to not accidentally remove
  vessesl like structures along the light-sheet direction.
  """
  if verbose:
    timer = tmr.Timer();
  
  #lightsheet artifact estimate
  l =  rnk.per.percentile(source, percentile=percentile, max_bin=max_bin, mask=mask, **lightsheet);
  if verbose:
    timer.print_elapsed_time('LightsheetCorrection: lightsheet artifact done')
  
  #background estimate                         
  b = ls.local_percentile(source, percentile=percentile, mask=mask, **background);
  if verbose:
    timer.print_elapsed_time('LightsheetCorrection: background done')
    
  #combined estimate                                                                                    
  lb = np.minimum(l, lightsheet_vs_background * b);
  
  #corrected image                                           
  c = source - np.minimum(source, lb);
  
  if verbose:
    timer.print_elapsed_time('LightsheetCorrection: done')
  
  result = (c,); 
  if return_lightsheet:
    result += (l,)                    
  if return_background:
    result += (b,) 
  if len(result) == 1:
    result = result[0];        
  return result;


###############################################################################
### Tests
###############################################################################

def _test():
  """Tests"""
  import ClearMap.Tests.Files as tsf
  import ClearMap.Visualization.Plot3d as p3d
  
  import ClearMap.ImageProcessing.LightsheetCorrection as ls
  from importlib import reload
  reload(ls)
  
  s = tsf.source('vasculature_lightsheet_raw')
  #p3d.plot(s)
  
  import ClearMap.ImageProcessing.Experts.Vasculature as vasc
  clipped, mask, high, low = vasc.clip(s[:,:,80:120], clip_range=(400,60000));
  
  corrected = ls.correct_lightsheet(clipped, mask=mask, percentile=0.25, 
                                    lightsheet=dict(selem=(150,1,1)),
                                    background=dict(selem = (200,200,1), 
                                                    spacing = (25,25,1), 
                                                    step=(2,2,1), 
                                                    interpolate=1),
                                    lightsheet_vs_background = 2);
                                                    
  p3d.plot([clipped, corrected])
  
  
