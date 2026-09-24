import os

import numpy as np

import ClearMap.ParallelProcessing.SharedMemoryArray as sma
from ClearMap.IO import FileUtils as fu
from ClearMap.IO.source import Slice as slc, Source as source_mod
from ClearMap.IO.source.protocol import Backend
from ClearMap.IO.source.backends.registry import BY_EXTENSION
from ClearMap.Utils import tag_expression as te
from ClearMap.Utils.exceptions import SourceModuleNotFoundError, ClearMapValueError


_LOCATION_SPEC = (str, bytes, os.PathLike, te.Expression)


def normalize_source_spec(source_):
    """Normalise *source_* if it is a location (path, bytes, Expression); pass anything else through.

    Unlike :func:`FileUtils.normalize_location_spec`, this accepts every source
    specification (arrays, Sources, None, ...), which is what the IO entry points receive.
    """
    if isinstance(source_, _LOCATION_SPEC):
        return fu.normalize_location_spec(source_)
    return source_


def source_to_backend(source_) -> Backend:
    if isinstance(source_, slc.Slice):
        return source_to_backend(source_.source)

    if isinstance(source_, source_mod.Source):
        if source_.backend is None:
            raise ClearMapValueError(f'Source {source_!r} has no backend identity.')
        return source_.backend

    source_ = normalize_source_spec(source_)

    if isinstance(source_, (str, te.Expression)):
        return location_to_backend(source_)

    if isinstance(source_, np.memmap):
        return Backend.MMP

    if isinstance(source_, (np.ndarray, list, tuple)) or source_ is None:
        return Backend.SMA if sma.is_shared(source_) else Backend.NPY

    raise ClearMapValueError(f'{source_!r} is not a valid source specification.')


def filename_to_backend(filename) -> Backend:
    filename = fu.normalize_location_spec(filename)
    extension = fu.file_extension(filename)
    backend = BY_EXTENSION.get(extension.lower() if extension else extension)

    if backend is None:
        raise SourceModuleNotFoundError(filename, extension)

    return backend


def location_to_backend(location_) -> Backend:
    location_ = fu.normalize_location_spec(location_)

    if isinstance(location_, te.Expression):
        return Backend.FILELIST

    if not isinstance(location_, str):  # REFACTOR: ClearMapTypeError
        raise TypeError(f'Expected a location string or Expression, got {type(location_).__name__}.')

    if fu.is_directory(location_) or te.Expression.is_expression(location_):
        return Backend.FILELIST

    return filename_to_backend(location_)


def source_to_module(source_):
    """
    Returns IO module associated with a source.

    Parameters
    ----------
    source_ : object
        The source specification.

    Returns
    -------
    type : module
        The module that handles the IO of the source.
    """
    return source_to_backend(source_).module


def source_to_class(source_):
    return source_to_module(source_).SOURCE_CLASS


def location_to_module(location_):
    """
    Returns the IO module associated with a location string.

    Parameters
    ----------
    location_ : str or te.Expression or pathlib.Path
        Location of the source.

    Returns
    -------
    module : module
        The module that handles the IO of the source specified by its location.
    """
    return location_to_backend(location_).module


def filename_to_module(filename):
    """
    Returns the IO module associated with a filename.

    Parameters
    ----------
    filename : str
       The file name.

    Returns
    -------
    module : module
       The module that handles the IO of the file.
    """
    return filename_to_backend(filename).module


def as_source(source_, slicing=None, *args, **kwargs):
    """
    Convert source specification to a Source class.

    Parameters
    ----------
    source_ : object
        The source specification.
    slicing : int, slice, list of slices or None
        Optional slicing to apply to the source after opening.

    Returns
    -------
    source : Source class
        The source class.
    """
    if not isinstance(source_, source_mod.Source):
        source_ = normalize_source_spec(source_)
        source_ = source_to_backend(source_).source_class(source_, *args, **kwargs)

    if slicing is not None:
        if isinstance(source_, (source_mod.TableSource, source_mod.GraphSource)):  # Slice is array arithmetic
            raise ClearMapValueError(f'{type(source_).__name__} cannot be sliced into a Source; '
                                     f'use io.read(source, slicing=...) to select data instead.')
        source_ = slc.Slice(source=source_, slicing=slicing)

    return source_
