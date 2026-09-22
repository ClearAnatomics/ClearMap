from __future__ import annotations

import importlib
import importlib.util
import os
from enum import Enum
from typing import Any, Protocol, cast


PathSpec = str | bytes | os.PathLike[str] | os.PathLike[bytes]
BACKENDS_PACKAGE = 'ClearMap.IO.source.backends'


class CreateFunction(Protocol):
    def __call__(self, location: PathSpec | None = None, shape: Any = None, dtype: Any = None, order: Any = None,
                 mode: str | None = None, array: Any = None, as_source: bool = True, **kwargs: Any) -> Any: ...


class OpenReadOnlyFunction(Protocol):
    def __call__(self, source_: Any, **kwargs: Any) -> Any: ...


class ReadFunction(Protocol):
    def __call__(self, source_: Any, slicing: Any = None, **kwargs: Any) -> Any: ...


class WriteFunction(Protocol):
    def __call__(self, sink: Any, data: Any = None, slicing: Any = None, overwrite: bool = False, **kwargs: Any) -> Any: ...


class SourceModule(Protocol):
    """Structural interface implemented by an IO backend module."""
    SOURCE_CLASS: type

    create: CreateFunction
    open_ro: OpenReadOnlyFunction
    read: ReadFunction
    write: WriteFunction


class Backend(Enum):
    """Every backend known to ClearMap.

    .. WARNING::
        Extensions must be unique across backends.
    """

    NPY      = ('npy_backend',       (),                 ())
    MMP      = ('mmp_backend',       ('npy',),           ())
    SMA      = ('sma_backend',       (),                 ())
    TIF      = ('tif_backend',       ('tif', 'tiff'),    ())
    NRRD     = ('nrrd_backend',      ('nrrd', 'nrdh'),   ())
    MHD      = ('mhd_backend',       ('mhd',),           ())
    CSV      = ('csv_backend',       ('csv',),           ())
    FEATHER  = ('feather_backend',   ('feather',),       ('pyarrow',))
    FILELIST = ('file_list_backend', (),                 ())
    GT       = ('gt_backend',        ('gt',),            ('graph_tool',))

    def __init__(self, module_name: str, extensions: tuple[str, ...], requirements: tuple[str, ...]):
        self.module_name = module_name
        self.extensions = extensions
        self.requirements = requirements

    @property
    def key(self) -> str:
        return self.name.lower()

    @property
    def module_path(self) -> str:
        return f'{BACKENDS_PACKAGE}.{self.module_name}'

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
