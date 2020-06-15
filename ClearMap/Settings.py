# -*- coding: utf-8 -*-
"""
Setings
=======

Module to set *ClearMap's* internal parameter and paths to external programs.

Note
----
Edit the :func:`setup` routine to point to set paths for specific hosts.

See Also
--------
    * :const:`clearmap_path`
    * :const:`resources_path`
    * :const:`external_path`
    * :const:`ilastik_path`
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


import os
import socket

###############################################################################
### Paths
###############################################################################

def path():
  """Returns root path to the ClearMap software.
  
  Returns
  -------
  path : str 
    Root path to *ClearMap*.
  """
  fn = os.path.split(__file__)
  fn = os.path.abspath(fn[0]);
  return fn;


clearmap_path = path();
"""Absolute path to the ClearMap's root folder."""

resources_path = os.path.join(path(), 'Resources');
"""Absolute path to the ClearMap's resources folder."""

external_path = os.path.join(path(), 'External');
"""Absolute path to the ClearMap's external program folder."""

test_path = os.path.join(clearmap_path, 'Tests');
"""Absolute path to the ClearMap's test folder."""

test_data_path = os.path.join(test_path, 'Data');
"""Absolute path to the ClearMap's test data folder."""


###############################################################################
### Paths to external programs and resources
###############################################################################

elastix_path = os.path.join(external_path, 'elastix/build');
"""Absolue path to the elastix installation

Note
----
  `Elastix Webpage <http://elastix.isi.uu.nl/>`_
"""

tera_stitcher_path = None;
"""Absolue path to the TeraStitcher installation

Note
----
  `TeraSticher Webpage <http://abria.github.io/TeraStitcher/>`_
"""

imagej_path = None; # '/usr/local/Fiji.app';
"""Absolue path to the ImageJ/Fiji installation

Note
----
  `ImageJ/Fiji Webpage <https://fiji.sc/>`_
"""

ilastik_path = None; #'/usr/local/ilastik/ilastik-1.2.0-Linux';
"""Absolute path to the Ilastik installation

Note
----
  `Ilastik Webpage <http://ilastik.org/>`_ 
  `Ilastik Download <http://old.ilastik.org/>`_
"""


###############################################################################
## Setup
###############################################################################

def setup():
    """Setup ClearMap settings for specific hosts.
    
    Note
    ----
      Edit this routine to include special setttings for specific hosts.
        
    See Also
    --------
        :const:`elastix_path`, :const:`tera_stitcher_path`, :const:`imagej_path`, :const:`ilastik_path`
    """
    global elastix_path, tera_stitcher_path, imagej_path, ilastik_path
    
    hostname = socket.gethostname();
    
    if hostname == 'ChristophsComputer.rockefeller.edu':  #Christoph's Dektop 
        imagej_path        = '/home/ckirst/Programs/Fiji.app';
        ilastik_path       = '/home/ckirst/Programs/ilastik';
    
    elif hostname == 'ChristophsLaptop':  #Christoph's Laptop 
        imagej_path        = '/home/ckirst/Programs/Fiji.app';
        ilastik_path       = None;

    elif hostname == 'ihu-renie-lf001' or hostname == 'icm-renie-lf004.icm-institute.org':  #Nico's Desktop 
        elastix_path       = os.path.join(external_path, 'elastix/build');
        tera_stitcher_path = os.path.join(external_path, 'TeraStitcher'); 
    
    elif hostname == 'mtllab-Ubuntu': #MTL workstation
        elastix_path       = '/usr/local/elastix'; 
        tera_stitcher_path = '/home/mtllab/Programs/TeraStitcher';
        imagej_path        = None;
        ilastik_path       = '/usr/local/ilastik-1.1.9-Linux';

    # check existence:
    if not elastix_path is None:
        if not os.path.exists(elastix_path):
            #raise RuntimeWarning('Settings: elastix path %s does not exists, cf. Settings.py or type help(Settings) for details.' % elastix_path);
            print('Settings: Elastix path %s does not exists, cf. Settings.py or type help(Settings) for details.' % elastix_path);
            elastix_path = None;

    if not tera_stitcher_path is None:
        if not os.path.exists(tera_stitcher_path):
            #raise RuntimeWarning('Settings: TeraStitcher path %s does not exists, cf. Settings.py or type help(Settings) for details.' % tera_stitcher_path);
            print('Settings: TeraStitcher path %s does not exists, cf. Settings.py or type help(Settings) for details.' % tera_stitcher_path);
            tera_stitcher_path = None;
            
    if not imagej_path is None:
        if not os.path.exists(imagej_path):
            #raise RuntimeWarning('Settings: TeraStitcher path %s does not exists, cf. Settings.py or type help(Settings) for details.' % imagej_path);
            print('Settings: ImageJ path %s does not exists, cf. Settings.py or type help(Settings) for details.' % imagej_path);
            imagej_path = None;
    
    if not ilastik_path is None:
        if not os.path.exists(ilastik_path):
            #raise RuntimeWarning('Settings: ilastik path %s does not exists, cf. Settings.py or type help(Settings) for details.' % ilastik_path);
            print('Settings: ilastik path %s does not exists, cf. Settings.py or type help(Settings) for details.' % ilastik_path);
            ilastik_path = None;

setup();



