"""
The version of ClearMap and the version of its config files: the one place where they are written (to release,
change them here only).

This module imports nothing, so ``setup.py`` can read it before ClearMap and its dependencies are built or installed.
"""
__version__ = '3.1.2'

# The version of the config files: the vX.Y folders of the defaults, schemas and user configs, and the clearmap_version
# that the configs carry. It moves only when the config files change, not with each release of ClearMap.
CURRENT_CONFIG_VERSION = (3, 1)
