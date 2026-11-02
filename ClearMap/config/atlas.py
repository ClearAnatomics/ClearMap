"""
A simple configuration module for the atlases. This is parsed by the ClearMap GUI to
display the available atlases, and says where the atlases are kept.
"""
import re
from pathlib import Path

from ClearMap import Settings as settings

ATLAS_NAMES_MAP = {
    'ABA 2017 - adult mouse - 25µm': {'resolution': 25, 'base_name': 'ABA_25um_2017'},
    'ABA - adult mouse - 25µm': {'resolution': 25, 'base_name': 'ABA_25um'},
    # 'LAMBADA - P1 - 25µm': {'resolution': 25, 'base_name': 'LAMBADA_P1_25um'},
    # 'LAMBADA - P3 - 25µm': {'resolution': 25, 'base_name': 'LAMBADA_P3_25um'},
    # 'LAMBADA - P5 - 25µm': {'resolution': 25, 'base_name': 'LAMBADA_P5_25um'},
    # 'LAMBADA - P7 - 25µm': {'resolution': 25, 'base_name': 'LAMBADA_P7_25um'},
    # 'LAMBADA - P9 - 25µm': {'resolution': 25, 'base_name': 'LAMBADA_P9_25um'},
    # 'LAMBADA - P10 - 25µm': {'resolution': 25, 'base_name': 'LAMBADA_P10_25um'},
    # 'LAMBADA - P12 - 25µm': {'resolution': 25, 'base_name': 'LAMBADA_P12_25um'},
    # 'LAMBADA - P14 - 25µm': {'resolution': 25, 'base_name': 'LAMBADA_P14_25um'},
    # 'LAMBADA - P21 - 25µm': {'resolution': 25, 'base_name': 'LAMBADA_P21_25um'},
}


STRUCTURE_TREE_NAMES_MAP = {
    'ABA json 2022': 'ABA_annotation_last.json',
    'ABA json clearmap': 'ABA_annotation.json',
}


ATLAS_COMPONENT_NAMES = ('annotation', 'hemispheres', 'reference', 'distance_to_surface')
"""The files of an atlas: ``<base name>_<component>.tif``."""

ATLAS_CACHE_SUBFOLDER_NAME = 'atlases'
"""In the cache folder of ClearMap: where the atlases shipped compressed are decompressed."""

DEFAULT_ATLAS_CACHE_FOLDER_NAME = 'atlas_cache'
"""In the ClearMap config folder: where the reoriented / cropped atlases are, when the machine config has no folder."""


DERIVED_ATLAS_NAME_RE = re.compile(rf'^.+_(?:{"|".join(ATLAS_COMPONENT_NAMES)})_-?\d+_-?\d+_-?\d+_.+\.tif$')
"""
The name of an atlas file derived from a shipped one, reoriented or cropped:
``<base name>_<component>_<x>_<y>_<z>_<slicing>.tif`` (see ``Annotation.get_atlas_filepath``), for any atlas, the ones
that were added to ClearMap included. The slicing is not matched by what it contains (``None`` is there only when a step
is not set), but by being there: the shipped files are ``<base name>_<component>.tif``.
"""


def get_decompressed_atlases_dir() -> Path:
    """Where the atlases shipped compressed are decompressed, when they are not in the atlas folder."""
    return Path(settings.clearmap_cache_dir) / ATLAS_CACHE_SUBFOLDER_NAME


def get_atlas_cache_dir(configured_folder) -> Path:
    """
    Where the reoriented / cropped atlases are stored.

    Parameters
    ----------
    configured_folder : str or None
        The machine config 'atlas_cache_folder'. When empty: the folder "atlas_cache" of the ClearMap config folder,
        so that it follows ``CLEARMAP_CFG_DIR`` and is not in the home of a user who has no space there.
    """
    if not configured_folder:
        return Path(settings.clearmap_cfg_dir) / DEFAULT_ATLAS_CACHE_FOLDER_NAME
    return Path(configured_folder).expanduser()
