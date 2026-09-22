from __future__ import annotations

import importlib
import importlib.util
from enum import Enum
from pathlib import Path
from typing import Any, Protocol


class CreateFunction(Protocol):
    def __call__(self, location: str | Path | None = None, shape: Any = None, dtype: Any = None, order: Any = None,
                 mode: str | None = None, array: Any = None, as_source: bool = True, **kwargs: Any) -> Any:
        ...


class OpenReadOnlyFunction(Protocol):
    def __call__(self, source_: Any, **kwargs: Any) -> Any:
        ...


class ReadFunction(Protocol):
    def __call__(self, source_: Any, **kwargs: Any) -> Any:
        ...


class WriteFunction(Protocol):
    def __call__(self, sink: Any, data: Any = None, slicing: Any = None, overwrite: bool = False, **kwargs: Any) -> Any:
        ...


class SourceModule(Protocol):
    """Structural interface implemented by an IO backend module."""
    SOURCE_CLASS: type

    create: CreateFunction
    open_ro: OpenReadOnlyFunction
    read: ReadFunction
    write: WriteFunction