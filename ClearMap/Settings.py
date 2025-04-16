# -*- coding: utf-8 -*-
"""
Settings
========

Module to set *ClearMap's* internal parameter and paths to external programs.

Note
----
Edit the :func:`setup` routine to point to set paths for specific hosts.

See Also
--------
    * :const:`clearmap_path`
    * :const:`atlas_folder`
    * :const:`resources_path`
    * :const:`external_path`
    * :const:`ilastik_path`
"""
__author__ = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__ = 'https://idisco.info'
__download__ = 'https://github.com/ClearAnatomics/ClearMap'


import os

import configobj


###############################################################################
# ## Paths
###############################################################################


def path():  # FIXME: use inspect and rename
    """Returns root path to the ClearMap software.
  
    Returns
    -------
    path : str
        Root path to *ClearMap*.
    """
    f_name = os.path.split(__file__)
    f_name = os.path.abspath(f_name[0])
    return f_name


clearmap_path = path()
"""Absolute path to the ClearMap's root folder."""

resources_path = os.path.join(clearmap_path, 'Resources')
"""Absolute path to the ClearMap's resources folder."""

atlas_folder = os.path.join(resources_path, 'Atlas')  # FIXME: rename or rename others to be consistent
"""Absolute path to the ClearMap's atlas folder."""

external_path = os.path.join(clearmap_path, 'External')
"""Absolute path to the ClearMap's external program folder."""

test_path = os.path.join(clearmap_path, 'Tests')
"""Absolute path to the ClearMap's test folder."""

test_data_path = os.path.join(test_path, 'Data')
"""Absolute path to the ClearMap's test data folder."""


###############################################################################
# ## Paths to external programs and resources
###############################################################################

cfg_path = os.path.expanduser('~/.clearmap/machine_params_v3_0.cfg')
if os.path.exists(cfg_path):
    config = configobj.ConfigObj(cfg_path, encoding="UTF8", indent_type='    ', unrepr=True, file_error=True)
else:
    package_machine_path = os.path.join(clearmap_path, 'config', 'machine_params_v3_0.cfg')
    if os.path.exists(package_machine_path):
        config = configobj.ConfigObj(package_machine_path, encoding="UTF8", indent_type='    ',
                                     unrepr=True, file_error=True)
    else:
        config = {}

default_elastix_path = os.path.join(external_path, 'elastix', 'build')
elastix_path = config.get('elastix_path', default_elastix_path)
if elastix_path is None:
    elastix_path = default_elastix_path
"""Absolute path to the elastix installation

Note
----
  `Elastix Webpage <https://elastix.lumc.nl/>`_
"""

tera_stitcher_path = config.get('tera_stitcher_path')
"""Absolute path to the TeraStitcher installation

Note
----
  `TeraSticher Webpage <https://abria.github.io/TeraStitcher/>`_
"""

imagej_path = config.get('imagej_path')
"""Absolute path to the ImageJ/Fiji installation

Note
----
  `ImageJ/Fiji Webpage <https://fiji.sc/>`_
"""

ilastik_path = config.get('ilastik_path')
"""Absolute path to the Ilastik installation

Note
----
  `Ilastik Webpage <https://ilastik.org/>`_ 
  `Ilastik Download <https://www.ilastik.org/download.html>`_
"""
