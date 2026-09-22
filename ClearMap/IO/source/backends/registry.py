# -*- coding: utf-8 -*-
"""
registry
========

Declarative registry of IO backends.

This module only lazily imports backend modules.
It is imported by :mod:`ClearMap.IO.dispatch`.

Backend modules must follow this nomenclature convention:
# FIXME
"""
__author__ = 'Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'

import importlib
import importlib.util

from enum import Enum
from typing import cast

from ClearMap.IO.source.protocol import SourceModule


PACKAGE = __name__.rpartition('.')[0]  # ClearMap.IO.source.backends


class Backend(Enum):
    """Every backend known to ClearMap.

    .. WARNING::
        Extensions must be unique across backends.
    """

    NPY      = ('npy',       (),                 ())
    MMP      = ('mmp',       ('npy',),           ())
    SMA      = ('sma',       (),                 ())
    TIF      = ('tif',       ('tif', 'tiff'),    ())
    NRRD     = ('nrrd',      ('nrrd', 'nrdh'),   ())
    MHD      = ('mhd',       ('mhd',),           ())
    CSV      = ('csv',       ('csv',),           ())
    FEATHER  = ('feather',   ('feather',),       ('pyarrow',))
    FILELIST = ('file_list', (),                 ())
    GT       = ('gt',        ('gt',),            ('graph_tool',))

    def __init__(self, module_name: str, extensions: tuple[str, ...], requirements: tuple[str, ...]):
        self.module_name = module_name
        self.extensions = extensions
        self.requirements = requirements

    @property
    def key(self) -> str:
        return self.name.lower()

    @property
    def module_path(self) -> str:
        return f'{PACKAGE}.{self.module_name}'

    @property
    def module(self) -> SourceModule:
        return cast(SourceModule, importlib.import_module(self.module_path))

    @property
    def source_class(self) -> type:
        return self.module.SOURCE_CLASS

    @property
    def is_available(self) -> bool:
        return all(importlib.util.find_spec(requirement) is not None for requirement in self.requirements)

    def __str__(self):
        return self.key


BY_KEY = {backend.key: backend for backend in Backend}
BY_MODULE_NAME = {backend.module_name: backend for backend in Backend}
BY_EXTENSION = {ext: backend for backend in Backend for ext in backend.extensions}

_extensions = [ext for backend in Backend for ext in backend.extensions]
if len(_extensions) != len(set(_extensions)):
    raise RuntimeError('Duplicate file extensions declared by IO backends.')

SOURCE_EXTENSIONS = _extensions

def supports_extension(extension: str) -> bool:
    return extension.lstrip('.').lower() in BY_EXTENSION
