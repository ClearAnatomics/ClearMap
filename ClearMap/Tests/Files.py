# -*- coding: utf-8 -*-
"""
Files
=====

Module defining test data files.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'

import ClearMap.IO.IO as io

###############################################################################
### Files
###############################################################################

base_directory = io.split(io.abspath(__file__))[0];
"""The base directory for test data."""

data_directory = io.join(base_directory, 'Data');
"""The base directory for test data."""


vasculature_directory = io.join(data_directory, 'Vasculature');
"""Directory for vasculature sample data."""
                                    
vasculature_raw = io.join(vasculature_directory, 'vasculature_raw.npy');                                    
"""Vasculature sample data."""

vasculature_pre = io.join(vasculature_directory, 'vasculature_pre.npy');                                    
"""Vasculature sample data."""

vasculature_surface_raw = io.join(vasculature_directory, 'vasculature_surface_raw.npy');                                    
"""Vasculature sample data."""

vasculature_surface_pre = io.join(vasculature_directory, 'vasculature_surface_pre.npy');                                    
"""Vasculature sample data."""

vasculature_arteries_raw = io.join(vasculature_directory, 'vasculature_arteries_raw.npy');                                    
"""Vasculature sample data."""

vasculature_arteries_pre = io.join(vasculature_directory, 'vasculature_arteries_pre.npy');                                    
"""Vasculature sample data."""

vasculature_surface_arteries_raw = io.join(vasculature_directory, 'vasculature_surface_arteries_raw.npy');                                    
"""Vasculature sample data."""

vasculature_surface_arteries_pre = io.join(vasculature_directory, 'vasculature_surface_arteries_pre.npy');                                    
"""Vasculature sample data."""

vasculature_weak_raw = io.join(vasculature_directory, 'vasculature_weak_raw.npy');                                    
"""Vasculature sample data."""

vasculature_weak_pre = io.join(vasculature_directory, 'vasculature_weak_pre.npy');                                    
"""Vasculature sample data."""

vasculature_weak_high_raw = io.join(vasculature_directory, 'vasculature_weak_high_raw.npy');     
"""Vasculature sample data."""

vasculature_lightsheet_raw = io.join(vasculature_directory, 'vasculature_lightsheet_raw.npy');     
"""Vasculature sample data."""

vasculature_low_contrast_raw = io.join(vasculature_directory, 'vasculature_low_contrast_raw.npy');     
"""Vasculature sample data."""

vasculature_low_contrast_surface_raw = io.join(vasculature_directory, 'vasculature_low_contrast_surface_raw.npy');     
"""Vasculature sample data."""


tif_directory = io.join(data_directory, 'Tif');
"""Directory for tif sample data."""

tif_2d = io.join(tif_directory, '2d.tif');
"""2d tif file."""

tif_2d_color = io.join(tif_directory, '2d_color.tif');
"""2d color tif file."""

tif_3d = io.join(tif_directory, '3d.tif');
"""3d tif file."""

tif_sequence = io.join(tif_directory, 'sequence');
"""tif file sequence."""


skeleton_directory = io.join(data_directory, 'Skeletonization');
"""Directory for smaple data to test skeletonization."""

skeleton = io.join(skeleton_directory, 'skeleton.npy');
"""Skeleton sample file."""

skeleton_binary = io.join(skeleton_directory, 'binary.npy');
"""Binary sample data for skeleton generation."""


temp = io.join(data_directory, 'temp.npy');
"""Temporary file"""              


###############################################################################
### Filenames and sources
###############################################################################

def filename(data, postfix = None):
  """Returns a test data file name.
  
  Arguments
  ---------
  data : str
    The type of the sample data to initialize.
    {'vasculature'}
    
  Returns
  -------
  source : Source
    A data source of the sample data.
  """
  if data in ['v', 'vasculature_raw']:
    return vasculature_raw;
  elif data in ['vp', 'vasculature_pre']:
    return vasculature_pre;
  elif data in ['vs', 'vasculature_surface_raw']:
    return vasculature_surface_raw;
  elif data in ['vsp', 'vasculature_surfaace_pre']:
    return vasculature_surface_pre;
  elif data in ['va', 'vasculature_arteries_raw']:
    return vasculature_arteries_raw;
  elif data in ['vap', 'vasculature_arteries_pre']:
    return vasculature_arteries_pre;
  elif data in ['vsa', 'vasculature_surface_arteries_raw']:
    return vasculature_surface_arteries_raw;
  elif data in ['vsap', 'vasculature_surfaace_arteries_pre']:
    return vasculature_surface_arteries_pre;
  elif data in ['vw', 'vasculature_weak_raw']:
    return vasculature_weak_raw;
  elif data in ['vwp', 'vasculature_weak_pre']:
    return vasculature_weak_pre;
  elif data in ['vwh']:
    return vasculature_weak_high_raw;
  elif data in ['vls', 'vasculature_lightsheet_raw']:
    return vasculature_lightsheet_raw;
  elif data in ['vlc']:
    return vasculature_low_contrast_raw;
  elif data in ['vlcs']:
    return vasculature_low_contrast_surface_raw;
    
  elif data in ['tif_2d', 't2d']:
    return tif_2d;
  elif data in ['tif_2d_color', 't2dc']:
    return tif_2d_color;
  elif data in ['tif_3d', 't3d']:
    return tif_3d;
  elif data in ['tif_sequence', 'ts']:
    return tif_sequence
  
  elif data in ['skeleton', 'sk']:
    return skeleton
  elif data in ['skeleton_binary', 'skb']:
    return skeleton_binary
  
  elif data in ['temp', 'tmp', 't']:
    if postfix is not None:
      return temp[:-4] + '_' + postfix + '.npy';
    else:
      return temp;
  
  else:
    raise ValueError('Data sample type %r not found' % data); 


def source(*args, **kwargs):
  """Initializes a test data source.
  
  Arguments
  ---------
  As in :func:`filename`
    
  Returns
  -------
  source : Source
    A data source of the sample data.
  """
  return io.source(filename(*args, **kwargs));