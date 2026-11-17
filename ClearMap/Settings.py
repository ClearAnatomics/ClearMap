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
    * ``elastix_path``: the ``elastix_path`` of the machine configuration (see :func:`__getattr__`), else the elastix
      bundled with ClearMap
    * :const:`ilastik_path`
"""
__author__ = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__ = 'https://idisco.info'
__download__ = 'https://github.com/ClearAnatomics/ClearMap'


import os
from pathlib import Path


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


CFG_DIR_ENV_VAR = 'CLEARMAP_CFG_DIR'
"""Environment variable that sets the folder of the user configuration (default: ``~/.clearmap``).

On a cluster, the home folder of a job is often read-only or absent, or shared by jobs that need different settings.
"""

# WARNING: an empty variable means unset: Path('') is the current folder, not an error
clearmap_cfg_dir = Path(os.environ.get(CFG_DIR_ENV_VAR) or '~/.clearmap').expanduser().absolute()
"""Absolute :class:`~pathlib.Path` of the folder of the user configuration (machine parameters, user defaults, ...)."""

clearmap_cache_dir = clearmap_cfg_dir / 'cache'
"""Absolute :class:`~pathlib.Path` of the folder for the files that ClearMap derives from its own data (look-up tables, ...)
when the folder of the program cannot be written."""


###############################################################################
# ## Paths to external programs and resources
###############################################################################

default_elastix_path = os.path.join(external_path, 'elastix', 'build')
"""Absolute path to the elastix bundled with ClearMap, used when the machine configuration has no ``elastix_path``

Note
----
  `Elastix Webpage <https://elastix.lumc.nl/>`_
"""

EXTERNAL_PROGRAM_PATH_NAMES = ('elastix_path', 'tera_stitcher_path', 'imagej_path', 'ilastik_path')
"""The attributes of this module that the machine configuration sets, resolved at their first access"""


def __getattr__(name):
    """
    The paths to external programs that the machine configuration sets, read at their first access, then kept (PEP 562).

    ``elastix_path``
        Absolute path to the elastix installation (the bundled one if the machine configuration has none).
        `Elastix Webpage <https://elastix.lumc.nl/>`_
    ``tera_stitcher_path``
        Absolute path to the TeraStitcher installation, None if the machine configuration has none.
        `TeraStitcher Webpage <https://abria.github.io/TeraStitcher/>`_
    ``imagej_path``
        Absolute path to the ImageJ/Fiji installation, None if the machine configuration has none.
        `ImageJ/Fiji Webpage <https://fiji.sc/>`_
    ``ilastik_path``
        Absolute path to the Ilastik installation, None if the machine configuration has none.
        `Ilastik Webpage <https://ilastik.org/>`_, `Ilastik Download <https://www.ilastik.org/download.html>`_

    Not at import: :mod:`ClearMap.config` imports this module, and the installation script imports it before the
    machine configuration exists.
    """
    if name not in EXTERNAL_PROGRAM_PATH_NAMES:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    value = _machine_configuration_value(name)
    if value is None and name == 'elastix_path':
        value = default_elastix_path
    globals()[name] = value  # The next access finds it without coming here
    return value


def _machine_configuration_value(key):
    """
    The value of key in the machine configuration of this version, else in the 3.0 machine parameters when the machine
    configuration of this version does not exist (not upgraded yet), else None (also for a ``null`` value or a missing
    key).
    """
    # Not at the top: ClearMap.config imports this module
    from ClearMap.config.early_boot import MachineConfig

    machine_config = MachineConfig()
    if machine_config.cfg_path.exists():
        return machine_config.get(key)

    from ClearMap.config.config_handler import ConfigHandler  # Only to read the machine parameters of 3.0
    legacy_path = ConfigHandler.get_global_path('machine', must_exist=False, version='3.0')
    if not legacy_path.exists():
        return None
    return ConfigHandler.get_cfg_from_path(legacy_path).get(key)
