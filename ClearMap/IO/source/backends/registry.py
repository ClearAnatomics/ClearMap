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
from dataclasses import dataclass
from enum import StrEnum

PACKAGE = __name__.rpartition('.')[0]  # 'ClearMap.IO'


class BackendName(StrEnum):
    """Stable identifiers for IO backends.

    Used as registry keys and as the value of ``Source.backend``. An enum
    rather than bare strings so references are trackable by the IDE, typos
    raise ``AttributeError`` at import, and the set is enumerable for tests.

    Note
    ----
    Inherits from ``str`` so that legacy code comparing against plain strings
    keeps working during migration. Prefer ``BackendName.NRRD`` over ``'nrrd'``
    in new code.
    """
    NPY = 'npy'
    MMP = 'mmp'
    SMA = 'sma'
    TIF = 'tif'
    NRRD = 'nrrd'
    MHD = 'mhd'
    CSV = 'csv'
    FEATHER = 'feather'
    FILELIST = 'filelist'
    GT = 'gt'


@dataclass(frozen=True)
class Backend:
    """One IO backend: where its module lives and which files it claims.

    Arguments
    ---------
    key : BackendName
        Registry identifier.
    module_stem : str
        File stem inside ``ClearMap/IO/``, e.g. ``'NRRD'`` for ``NRRD.py``.
    extensions : tuple of str
        File extensions this backend owns, without the leading dot. A backend
        reachable only through a Source object or an in-memory array has none.
    optional : bool
        True if the backend depends on a package that may not be installed.
    """
    key: BackendName
    module_stem: str
    extensions: tuple[str, ...] = ()
    optional: bool = False

    def __post_init__(self):
        if importlib.util.find_spec(self.module_path) is None:
            raise ImportError(f'Backend {self.key.value!r} declares a missing module '
                              f'{self.module_path!r}. Was {self.module_stem}.py renamed or moved?')

    @property
    def module_path(self):
        return f'{PACKAGE}.{self.module_stem}'

    @property
    def module(self):
        """The backend module, imported on first access.

        ``import_module`` is a ``sys.modules`` lookup after the first call, so
        no caching is needed here.
        """
        return importlib.import_module(self.module_path)

    @property
    def source_class(self):
        """The concrete Source subclass this backend produces."""
        return self.module.SOURCE_CLASS

    @property
    def is_available(self):
        """Whether the backend can be imported; False for absent optional deps."""
        try:
            self.module
        except ImportError:
            if self.optional:
                return False
            raise
        return True

    def __str__(self):
        return self.key.value


_N = BackendName

BACKENDS = (
    Backend(_N.NPY,      'NPY'),
    Backend(_N.MMP,      'MMP',      extensions=('npy',)),
    Backend(_N.SMA,      'SMA'),
    Backend(_N.TIF,      'TIF',      extensions=('tif', 'tiff')),
    Backend(_N.NRRD,     'NRRD',     extensions=('nrrd', 'nrdh')),
    Backend(_N.MHD,      'MHD',      extensions=('mhd',)),
    Backend(_N.CSV,      'CSV',      extensions=('csv',)),
    Backend(_N.FEATHER,  'Feather',  extensions=('feather',), optional=True),
    Backend(_N.FILELIST, 'FileList'),
    Backend(_N.GT,       'GT',       extensions=('gt',),      optional=True),
)
"""Every backend known to ClearMap.
 WARNING: extensions must be unique."""


BY_KEY = {b.key: b for b in BACKENDS}
BY_MODULE_STEM = {b.module_stem: b for b in BACKENDS}
BY_EXTENSION = {ext: b for b in BACKENDS for ext in b.extensions}
