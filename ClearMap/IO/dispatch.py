import importlib
import pathlib

import numpy as np

from ClearMap.IO import FileUtils as fu
from ClearMap.IO.source import Slice as slc, Source as source_mod
from ClearMap.IO.source.backends import file_list_backend as fl, mmp_backend, sma_backend, npy_backend
from ClearMap.Utils import tag_expression as te
from ClearMap.Utils.exceptions import SourceModuleNotFoundError


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
    source_ = fu.normalize_location_spec(source_)

    # FIXME: add Slice sources unwrapping (recursive call to source_to_module of source_.base

    if isinstance(source_, source_mod.Source):
        return importlib.import_module(source_.__module__)
    elif isinstance(source_, (str, te.Expression)):
        return location_to_module(source_)
    elif isinstance(source_, np.memmap):
        return mmp_backend
    elif isinstance(source_, (np.ndarray, list, tuple)) or source_ is None:
        if sma_backend.is_shared(source_):
            return sma_backend
        else:
            return npy_backend
    else:
        raise ValueError(f'The source {source_} is not a valid source!')


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
    location_ = fu.normalize_location_spec(location_)
    if fl.is_file_list(location_):
        return fl
    else:
        return filename_to_module(location_)


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
    filename = fu.normalize_location_spec(filename)
    ext = fu.file_extension(filename)
    mod = file_extension_to_module.get(ext, None)
    if mod is None:
        raise SourceModuleNotFoundError(filename, ext)

    return mod


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
    source_ = fu.normalize_location_spec(source_)

    if not isinstance(source_, source_mod.Source):
        mod = source_to_module(source_)
        source_ = module_to_source_cls[mod](source_, *args, **kwargs)  # FIXME: bypass mod altogether
    if slicing is not None:
        source_ = slc.Slice(source=source_, slicing=slicing)
    return source_
