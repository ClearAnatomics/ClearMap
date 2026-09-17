# -*- coding: utf-8 -*-
"""
IO
==

Unified IO interface for all ClearMap data sources.

This module is the single entry point for reading, writing, and inspecting
data in any format that ClearMap understands.  It dispatches every call to the
appropriate format-specific sub-module based on the type or file extension of
the source, so calling code never needs to import ``TIF``, ``MMP``, ``NPY``,
etc. directly.

Supported formats
-----------------

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - Extension
     - Module
     - Notes
   * - ``.npy``
     - :mod:`~ClearMap.IO.MMP`
     - Memory-mapped NumPy arrays (default for large 3-D volumes)
   * - ``.tif`` / ``.tiff``
     - :mod:`~ClearMap.IO.TIF`
     - Single files and file-list expressions
   * - ``.nrrd`` / ``.nrdh``
     - :mod:`~ClearMap.IO.NRRD`
     -
   * - ``.mhd``
     - :mod:`~ClearMap.IO.MHD`
     - MetaImage header + raw data
   * - ``.csv``
     - :mod:`~ClearMap.IO.CSV`
     - Point / coordinate tables
   * - ``.gt``
     - :mod:`~ClearMap.IO.GT`
     - graph-tool graphs (optional dependency)
   * - ``<tag expression>``
     - :mod:`~ClearMap.IO.FileList`
     - Ordered lists of files matched by a tag expression
   * - ``np.ndarray``
     - :mod:`~ClearMap.IO.NPY`
     - In-memory NumPy arrays
   * - ``np.memmap``
     - :mod:`~ClearMap.IO.MMP`
     - Memory maps passed directly
   * - shared memory
     - :mod:`~ClearMap.IO.SMA`
     - Shared-memory arrays for parallel processing

Source routing
--------------
:func:`source_to_module` maps any source specification to its handler module:

* A :class:`~ClearMap.IO.Source.Source` instance → the module that created it.
* A ``str`` or :class:`~ClearMap.Utils.tag_expression.Expression` →
  :func:`location_to_module` (file-list expression, or extension lookup).
* A ``np.ndarray`` / ``list`` / ``tuple`` →
  :mod:`~ClearMap.IO.SMA` if in shared memory, else :mod:`~ClearMap.IO.NPY`.
* A ``np.memmap`` → :mod:`~ClearMap.IO.MMP`.

Core functions
--------------

**Reading and writing**

.. code-block:: python

    import ClearMap.IO.IO as io

    data  = io.read('signal.tif')            # returns np.ndarray
    data  = io.read('volume.npy',
                    slicing=(slice(0, 100),)) # sub-slice
    io.write('output.tif', data)

**Source objects** — richer than raw arrays; carry shape, dtype, and location:

.. code-block:: python

    src = io.as_source('volume.npy')
    print(src.shape, src.dtype, src.order)
    data = src[10:20, :, :]               # lazy slicing

**Initialising a sink** before parallel workers write into it:

.. code-block:: python

    # Open existing file or create it if absent
    sink = io.initialize('counts.npy',
                         shape_=(512, 512, 256),
                         dtype_=np.uint16,
                         order_='F')

**File-list expressions** — match tiles with tag patterns:

.. code-block:: python

    files = io.file_list('raw/tile_<X,2>_<Y,2>.tif')

**Bulk conversion** between formats (parallelised):

.. code-block:: python

    io.convert_files(files, extension='.npy', processes=8)

Property helpers
----------------
:func:`shape`, :func:`dtype`, :func:`order`, :func:`location`,
:func:`element_strides`, :func:`memory`, :func:`buffer` —
each accepts any source specification (path, array, or
:class:`~ClearMap.IO.Source.Source`) and returns the corresponding attribute
without requiring the caller to construct a Source explicitly.

File-path utilities
-------------------
The following functions from :mod:`~ClearMap.IO.FileUtils` are re-exported
here for convenience:
``is_file``, ``is_directory``, ``file_extension``, ``join``, ``split``,
``abspath``, ``create_directory``, ``delete_directory``, ``copy_file``,
``link_file``, ``delete_file``.

See also
--------
:mod:`ClearMap.IO.Source` : Base Source class and AbstractSource / VirtualSource.
:mod:`ClearMap.IO.MMP`    : Memory-mapped arrays (primary large-data format).
:mod:`ClearMap.IO.FileList` : Tag-expression file lists.
:mod:`ClearMap.IO.workspace2` : High-level asset management built on top of this module.
"""
__author__ = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__ = 'https://idisco.info'
__download__ = 'https://github.com/ClearAnatomics/ClearMap'


import importlib
import functools
import math
import pathlib
import multiprocessing as mp
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd

import ClearMap.IO.Source as source_mod
import ClearMap.IO.Slice as slc
import ClearMap.IO.TIF as tif
import ClearMap.IO.NRRD as nrrd
import ClearMap.IO.CSV as csv
import ClearMap.IO.NPY as npy
import ClearMap.IO.MMP as mmp
import ClearMap.IO.SMA as sma
import ClearMap.IO.MHD as mhd
from ClearMap.Utils.exceptions import (IncompatibleSource, SourceModuleNotFoundError, ClearMapRuntimeError,
                                       ClearMapException, SourceNotFoundError, AssetNotFoundError, ClearMapValueError)

try:
    import ClearMap.IO.GT as gt
    gt_loaded = True
except ImportError:
    gt_loaded = False
import ClearMap.IO.FileList as fl
import ClearMap.IO.FileUtils as fu

import ClearMap.Utils.tag_expression as te
import ClearMap.Utils.Timer as tmr

import ClearMap.ParallelProcessing.ParallelTraceback as ptb

from ClearMap.Utils.utilities import CancelableProcessPoolExecutor


###############################################################################
# ## File manipulation
###############################################################################
# WARNING: imported just for module level access. REFACTOR: should be in subpackage __init__
# noinspection PyUnusedImports
from ClearMap.IO.FileUtils import (is_file, is_directory, file_extension,
                                   join, split, abspath, create_directory, 
                                   delete_directory, copy_file, link_file, delete_file)

###############################################################################
# ## Source associations
###############################################################################

source_modules = [npy, tif, mmp, sma, fl, nrrd, mhd, csv]
"""The valid source modules."""

file_extension_to_module = {'npy': mmp,
                            'tif': tif,
                            'tiff': tif,
                            'nrrd': nrrd,
                            'nrdh': nrrd,
                            'csv': csv,
                            'mhd': mhd}

# FIXME: there MUST be a better way
module_to_source_cls = {
    npy: npy.NumpySource,
    tif: tif.TifSource,
    mmp: mmp.MMPSource,
    sma: sma.SMASource,
    fl: fl.FileListSource,
    nrrd: nrrd.NrrdSource,
    mhd: mhd.MhdSource,
    csv: csv.CSVSource
}


if gt_loaded:
    file_extension_to_module['gt'] = gt
    source_modules += [gt]
"""Map between file extensions and modules that handle this file type."""


class AssetBase:
    @property
    def path(self):
        raise NotImplementedError('AssetBase is an abstract class, cannot get path!')


###############################################################################
# ## Source to module conversions
###############################################################################
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
    if isinstance(source_, AssetBase):
        source_ = source_.path
    if isinstance(source_, pathlib.Path):
        source_ = str(source_)

    # FIXME: add Slice sources unwrapping (recursive call to source_to_module of source_.base

    if isinstance(source_, source_mod.Source):
        return importlib.import_module(source_.__module__)
    elif isinstance(source_, (str, te.Expression)):
        return location_to_module(source_)
    elif isinstance(source_, np.memmap):
        return mmp
    elif isinstance(source_, (np.ndarray, list, tuple)) or source_ is None:
        if sma.is_shared(source_):
            return sma
        else:
            return npy
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
    if isinstance(location_, pathlib.Path):
        location_ = str(location_)
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
    if isinstance(filename, pathlib.Path):
        filename = str(filename)

    ext = fu.file_extension(filename)

    mod = file_extension_to_module.get(ext, None)
    if mod is None:
        raise SourceModuleNotFoundError(filename, ext)

    return mod

##############################################################################
# ## IO Interface
##############################################################################
# FIXME: add support for Assets

# read write interface: specialized modules can assume the following
# read(source, slicing=None, **kwargs)
#  source is a valid source for the module as determined by the module's is_xxx function
# write(sink, data, slicing=None, *kwargs)
#  sink is a valid source for the module as determined by the module's is_xxx function
#  data is a Source class

def is_source(source_, exists=True):
    """
    Checks if `source_` is a valid Source.

    Parameters
    ----------
    source_ : object
        Source to check.
    exists : bool
        If True, check if source exists in case it has a location.

    Returns
    -------
    is_source : bool
       True if source is a valid source.
    """
    if isinstance(source_, pathlib.Path):
        source_ = str(source_)

    if isinstance(source_, source_mod.Source):
        if exists:
            return source_.exists()
        else:
            return True
    elif isinstance(source_, str):
        try:
            mod = location_to_module(source_)
        except SourceModuleNotFoundError:
            return False
        if exists:
            return module_to_source_cls[mod](source_).exists()  # FIXME: bypass module altogether with separate dict
        else:
            return True
    elif isinstance(source_, (np.memmap, np.ndarray, list, tuple)):
        return True
    else:
        return False


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
    if isinstance(source_, pathlib.Path):
        source_ = str(source_)

    if not isinstance(source_, source_mod.Source):
        mod = source_to_module(source_)
        source_ = module_to_source_cls[mod](source_, *args, **kwargs)  # FIXME: bypass mod altogether
    if slicing is not None:
        source_ = slc.Slice(source=source_, slicing=slicing)
    return source_


def source(source_, slicing=None, *args, **kwargs):
    """
    Convert source specification to a Source class.

    Parameters
    ----------
    source_ : object
        The source specification.

    Returns
    -------
    source : Source class
        The source class.
    """
    return as_source(source_, slicing=slicing, *args, **kwargs)


def ndim(source_):
    """
    Returns number of dimensions of a source.

    Parameters
    ----------
    source_ : str, array or Source
        The source specification.

    Returns
    -------
    ndim : int
        The number of dimensions in the source.
    """
    source_ = open_ro(source_)
    return source_.ndim


def shape(source_):
    """
    Returns shape of a source.

    Parameters
    ----------
    source_: str, array or Source
       The source specification.

    Returns
    -------
    shape : tuple of ints
       The shape of the source.
    """
    source_ = open_ro(source_)
    return source_.shape


def size(source_):
    """
    Returns size of a source.

    Parameters
    ----------
    source_ : str, array or Source
        The source specification.

    Returns
    -------
    size : int
        The size of the source.
    """
    source_ = open_ro(source_)
    return source_.size


def dtype(source_):
    """
    Returns dtype of a source.

    Parameters
    ----------
    source_ : str, array or Source
        The source specification.

    Returns
    -------
    dtype : dtype
        The data type of the source.
    """
    source_ = open_ro(source_)
    return source_.dtype


def order(source_):
    """
    Returns order of a source.

    Parameters
    ----------
    source_ : str, array or Source
        The source specification.

    Returns
    -------
    order : 'C', 'F', or None
        The order of the source data items.
    """
    source_ = open_ro(source_)
    return source_.order


def location(source_):
    """
    Returns the location of a source.

    Parameters
    ----------
    source_ : str, array or Source
        The source specification.

    Returns
    -------
    location : str or None
        The location of the source.
    """
    if isinstance(source_, (str, pathlib.Path)) and not pathlib.Path(source_).exists():  # TODO: check if we **want** that bhv
        return source_
    source_ = open_ro(source_)
    return source_.location


def memory(source_):
    """
    Returns the memory type of `source_`.

    Parameters
    ----------
    source_ : str, array or Source
        The source specification.

    Returns
    -------
    memory : str or None
        The memory type of the source.
    """
    if isinstance(source_, pathlib.Path):
        source_ = str(source_)

    if sma.is_shared(source_):
        return 'shared'


def element_strides(source_):
    """
    Returns the strides of the data array of a source.

    Parameters
    ----------
    source_ : str, array, dtype or Source
        The source specification.

    Returns
    -------
    strides : tuple of int
        The strides of the source.
    """
    try:
        source_ = open_ro(source_)
        strides = source_.element_strides
    except Exception as e:
        raise ValueError(f'Cannot determine the strides for the source!; {e}')

    return strides


def buffer(source_):
    """
    Returns an io buffer of the data array of a source for use with e.g. cython.

    Parameters
    ----------
    source_ : source specification
      The source specification.

    Returns
    -------
    buffer : array or memmap
      A buffer to read and write data.
    """
    try:
        source_ = as_source(source_)
        buffer_ = source_.as_buffer()
    except Exception as e:
        raise ValueError(f'Cannot get a io buffer for the source!; {e}')

    return buffer_


def _is_feather_path(source_) -> bool:
    return isinstance(source_, (str, pathlib.Path)) and str(source_).endswith('.feather')


# TODO: arg memory= to specify which kind of array is created, better use device=
# TODO: arg processes= in order to use ParallelIO -> can combine with buffer=
def read(source_, slicing=None, *args, **kwargs):
    """
    Read data from a data source.

    .. warning::
        For some modules (file types) this does an active read
        for others, it just does an open and returns a Source class
        that can be used to read the data.

    Parameters
    ----------
    source_ : str, pathlib.Path, array, Source class
       The source to read the data from.

    Returns
    -------
    data : array
        The data of the source.
    """
    if isinstance(source_, pathlib.Path):
        source_ = str(source_)
    if _is_feather_path(source_):
        return pd.read_feather(source_)
    elif isinstance(source_, np.ndarray):  # Already materialised — nothing to do
        return source_
    elif isinstance(source_, source_mod.Source):  # Source-like with .array (Block, Slice, NPY.Source, MMP.Source, ...)
        if not hasattr(source_, 'array'):
            raise ClearMapValueError(f'Source {source_} has no array property and cannot be read directly')
        if args or kwargs:
            warnings.warn(f'Ignoring unsupported read arguments {args=} and {kwargs=}'
                          f' for materialised source {source_!r}.', stacklevel=2)
        return source_.array  if slicing is None else source_[slicing]

    # File path or expression — dispatch to the right module
    mod = source_to_module(source_)
    return mod.read(source_, *args, **kwargs)


def open_ro(source_, **kwargs):
    """Open a source strictly read-only for metadata queries."""
    if isinstance(source_, pathlib.Path):
        source_ = str(source_)
    if _is_feather_path(source_):
        return pd.read_feather(source_)
    mod = source_to_module(source_)
    if isinstance(source_, source_mod.Source):
        if source_.mode == 'r':
            return source_
        if hasattr(mod, 'open_ro'):
            return mod.open_ro(source_, **kwargs)
        return source_  # non-MMP Sources are read-only at API level already
    else:  # e.g. string
        if hasattr(mod, 'open_ro'):
            return mod.open_ro(source_, **kwargs)
    # no open_ro available -> fallback: construct with mode='r'
    return module_to_source_cls[mod](source_, mode='r', **kwargs)


@contextmanager
def peek_into(source_, **kwargs):
    """Temporarily open *source_* for metadata inspection.

    Existing Source objects remain owned by the caller. Sources opened from
    paths or other descriptors are closed on exit when possible.
    """
    source = open_ro(source_, **kwargs)
    owns_source = source is not source_

    try:
        yield source
    finally:
        if owns_source and hasattr(source, 'close'):
            source.close()  # FIXME: no `close` API in Source


def edit(source_, **kwargs):
    """Open a source for in-place editing (mode='r+')."""
    if isinstance(source_, pathlib.Path):
        source_ = str(source_)
    if _is_feather_path(source_):
        return pd.read_feather(source_)
    mod = source_to_module(source_)
    if hasattr(mod, 'edit'):
        return mod.edit(source_, **kwargs)
    return as_source(source_, **kwargs)  # FIXME: explicit mode = r+


def write(sink, data, *args, **kwargs):
    """
    Write data to a data source.

    Parameters
    ----------
    sink : str, pathlib.Path, array, Source class
        The source to write data to.
    data : array
        The data to write to the sink.
    slicing : slice specification or None
        Optional sub-slice to write data to.

    Returns
    -------
    sink : str, array or Source class
        The sink to which the data was written.
    """
    if isinstance(sink, pathlib.Path):
        sink = str(sink)
    if _is_feather_path(sink):
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)  # backward compat: structured array
        if not isinstance(data.index, pd.RangeIndex) or data.index[0] != 0:
            data = data.reset_index(drop=True)  # feather requires default RangeIndex
        data.to_feather(sink)
        return sink
    mod = source_to_module(sink)
    return mod.write(sink, open_ro(data), *args, **kwargs)


def create(source_, *args, **kwargs):
    """
    Create a data source on disk.

    Parameters
    ----------
    source_ : str, pathlib.Path, array, Source class
        The source to write data to.

    Returns
    -------
    sink : str, array or Source class
       The sink to which the data was written.
    """
    if isinstance(source_, pathlib.Path):
        source_ = str(source_)
    mod = source_to_module(source_)
    return mod.create(source_, *args, **kwargs)


def initialize(source_=None, shape_=None, dtype_=None,
               order_=None, location_=None, memory_=None, like=None, hint=None, **kwargs):
    """
    Initialize (open to edit or create if missing) a source with specified properties.

    Note
    ----
    The source is created on disk or in memory if it does not exist so processes
    can start writing into it.

    Parameters
    ----------
    source_ : str, array, Source class
        The source to write data to.
    shape_ : tuple or None
        The desired shape of the source.
        If None, inferred from existing file or from the like parameter.
        If not None and source has a valid shape shapes are tested to match.
    dtype_ : type, str or None
        The desired dtype of the source.
        If None, inferred from existing file or from the like parameter.
        If not None and source has a valid dtype the types are tested to match.
    order_ : 'C', 'F' or None
        The desired order of the source.
        If None, inferred from existing file or from the like parameter.
        If not None and source has a valid order the orders are tested to match.
    location_ : str or None
        The desired location of the source.
        If None, inferred from existing file or from the like parameter.
        If not None and source has a valid location the locations need to match.
    memory_ : 'shared' or None
        The memory type of the source. If 'shared' a shared array is created.
    like : str, array or Source class
        Infer the source parameter from this source.
    hint : str, array or Source class
        If parameters for source creation are missing use the ones from this
        hint source.

    Returns
    -------
    source: Source class
        The initialized source.
    """
    if isinstance(source_, pathlib.Path):
        source_ = str(source_)

    if isinstance(source_, (str, te.Expression)):  # If the source is a path (location)
        location_ = source_
        source_ = None

    if like is not None:
        shape_, dtype_, order_ = _from_like(like, shape_, dtype_, order_)

    if source_ is None:
        if location_ is None:  # No source and no path: array in memory, regular or shared
            shape_, dtype_, order_ = _from_hint(hint, shape_, dtype_, order_)
            if memory_ in ['shared', 'automatic']:
                return sma.create(shape=shape_, dtype=dtype_, order=order_, **kwargs)
            else:
                return npy.create(shape=shape_, dtype=dtype_, order=order_, **kwargs)
        else:  # No source but a path
            # Before try because missing module != missing file so shouldn't fall through to creation.
            mod = location_to_module(location_)
            try:  # First, attempt to open the existing source in 'edit' mode
                if hasattr(mod, 'edit'):
                    source_ = mod.edit(location_)
                else:
                    source_ = as_source(location_, mode=source_mod.DEFAULT_EDIT_MODE)
            except AssetNotFoundError:  # workspace-level failure is not a reason to create a file
                raise  # WARNING: must stay before FileNotFoundError
            except (SourceNotFoundError, FileNotFoundError): # TODO: narrow to SourceNotFoundError once every IO module raises it.
                source_ = None
            # Anything else (IncompleteSourceSpecError, corrupt header, shape mismatch) != missing file -> propagates

            if source_ is None:  # Opening existing failed -> creation path
                if isinstance(location_, str) and pathlib.Path(location_).is_file():  # Just belt and braces
                    raise ClearMapRuntimeError(f'{location_} exists but could not be opened for editing; refusing overwrite.')
                shape_, dtype_, order_ = _from_hint(hint, shape_, dtype_, order_)
                try:
                    return mod.create(location=location_, shape=shape_, dtype=dtype_, order=order_, **kwargs)
                except ClearMapException:  # Catch and raise specific to avoid generic path
                    raise
                except Exception as error:
                    raise ClearMapRuntimeError(f'Cannot initialize source for location {location_}') from error

    if isinstance(source_, np.ndarray):
        source_ = as_source(source_)

    # ######## Exception handling ##############
    if not isinstance(source_, source_mod.Source):
        raise ClearMapValueError(f'Source specification {source_} not a valid location, array or Source class!')

    current_vars = locals()
    for attr in ('shape_', 'dtype_', 'order_'):
        base_attr = attr[:-1]
        if current_vars.get(attr) is not None and current_vars[attr] != getattr(source_, base_attr, None):
            raise IncompatibleSource(source_, attr, current_vars)

    if location_ is not None and abspath(location_) != abspath(source_.location):
        raise IncompatibleSource(source_, 'location', current_vars)
    if memory_ == 'shared' and not sma.is_shared(source_):
        raise ClearMapValueError(f'Incompatible memory type, the source {source_} is not shared!')

    return source_


def _from_like(like, shape, dtype, order):
    """Resolve geometry from a template (``like``) source.

    Parameters
    ----------
    like : object or None
        Source, source location, or other object from which missing geometry
        properties can be inferred.
    shape : tuple-like or None
        Requested shape. If ``None``, infer it from ``like``.
    dtype : dtype-like or None
        Requested data type. If ``None``, infer it from ``like``.
    order : {'C', 'F'} or None
        Requested memory order. If ``None``, infer it from ``like``.

    Returns
    -------
    shape : tuple-like or None
        Explicitly supplied or inferred shape.
    dtype : numpy.dtype or None
        Explicitly supplied or inferred data type.
    order : {'C', 'F'} or None
        Explicitly supplied or inferred memory order.

    Notes
    -----
    Explicit values take precedence over values inferred from ``like``.
    The source is opened read-only for metadata inspection and any temporary
    source opened for that purpose is closed before returning.
    """
    if like is None:
        return shape, dtype, order
    else:
        with peek_into(like) as source:
            return source_mod.resolve_geometry(shape=shape, dtype=dtype, order_=order, like=source)


def _from_hint(hint, shape, dtype, order):
    """Best-effort geometry inference from an initialization hint.
    Contrary to _from_like, this does not Raise

    Parameters
    ----------
    hint : object or None
        Source, source location, or other object from which missing geometry
        properties should be inferred.
    shape : tuple-like or None
        Requested shape, or ``None`` to infer it from ``hint``.
    dtype : dtype-like or None
        Requested data type, or ``None`` to infer it from ``hint``.
    order : {'C', 'F'} or None
        Requested memory order, or ``None`` to infer it from ``hint``.

    Returns
    -------
    shape : tuple-like or None
        Explicitly supplied or inferred shape.
    dtype : numpy.dtype or None
        Explicitly supplied or inferred data type.
    order : {'C', 'F'} or None
        Explicitly supplied or inferred memory order.

    Notes
    -----
    This is a non-raising wrapper around :func:`_from_like`. If inference
    fails for any reason, a warning is emitted and the original values are
    returned unchanged.

    Explicit values take precedence over values inferred from ``hint``.
    """
    try:
        return _from_like(hint, shape, dtype, order)
    except Exception as err:
        warnings.warn(f'Cannot infer shape, dtype and order from hint {hint}, keeping defaults; {err}', stacklevel=2)
        return shape, dtype, order
  

def initialize_buffer(source_, shape=None, dtype=None, order=None, location=None, memory=None, like=None, **kwargs):
    """
    Initialize a buffer with specific properties.

    Parameters
    ----------
    source_ : str, array, Source class
        The source to write data to.
    shape : tuple or None
        The desired shape of the source.
        If None, inferred from existing file or from the like parameter.
        If not None and source has a valid shape shapes are tested to match.
    dtype : type, str or None
        The desired dtype of the source.
        If None, inferred from existing file or from the like parameter.
        If not None and source has a valid dtype the types are tested to match.
    order : 'C', 'F' or None
        The desired order of the source.
        If None, inferred from existing file or from the like parameter.
        If not None and source has a valid order the orders are tested to match.
    location : str or None
        The desired location of the source.
        If None, inferred from existing file or from the like parameter.
        If not None and source has a valid location the locations need to match.
    memory : 'shared' or None
        The memory type of the source. If 'shared' a shared array is created.
    like : str, array or Source class
        Infer the source parameter from this source.

    Returns
    -------
    buffer : array
        The initialized buffer to use tih e.g. cython.

    Note
    ----
    The buffer is created if it does not exist.
    """
    source_ = initialize(source_, shape_=shape, dtype_=dtype, order_=order, location_=location, memory_=memory, **kwargs)
    return source_.as_buffer()


###############################################################################
# ## Utils
###############################################################################

def file_list(expression=None, file_list=None, sort=True, verbose=False):
    """
    Returns the list of files that match the tag expression.

    Parameters
    ----------
    expression :str | Path | te.Expression | None
        The regular expression the file names should match.
    sort : bool
        If True, sort files naturally.
    verbose : bool
        If True, print warning if no files exists.

    Returns
    -------
    file_list : list of str
        The list of files that matched the expression.
    """
    return fl._file_list(expression=expression, file_list=file_list, sort=sort, verbose=verbose)


def get_info(d_type):
    """
    Get the numpy info object for a data type. (automatically determines if integer or float)

    Parameters
    ----------
    d_type: dtype
        The data type to get the info for.

    Returns
    -------
    info: numpy info object
        The info object for the data type.
    """
    try:
        return np.iinfo(d_type)
    except ValueError:
        return np.finfo(d_type)


def get_value(source_, value_type):  # REFACTOR: should be moved to io_utils or Source module
    """
    Get the minimal or maximal value of a source data type.

    Parameters
    ----------
    source_: str, array, dtype or Source
        The source specification.
    value_type: str
        The value type to get, either 'min' or 'max'.

    Returns
    -------
    value: number
        The value of the data type.
    """
    if isinstance(source_, pathlib.Path):
        source_ = str(source_)

    if value_type not in ['min', 'max']:
        raise ValueError(f'Unknown value type {value_type}, accepted Parameters are "min" and "max"!')

    if isinstance(source_, (source_mod.Source, np.ndarray)):
        source_ = source_.dtype

    if isinstance(source_, str):
        try:
            source_ = np.dtype(source_)
        except TypeError:
            pass

    if not isinstance(source_, (type, np.dtype)):
        source_ = dtype(source_)

    try:
        info = get_info(source_)
        return getattr(info, value_type)
    except ValueError as e:
        raise ValueError(f'Cannot determine the {value_type} value for the type {source_}!; {e}')


def min_value(source_):
    """
    Returns the minimal value of a source data type.

    Parameters
    ----------
    source_ : str, array, dtype or Source
        The source specification.

    Returns
    -------
    min_value : number
        The minimal value for the data type of the source
    """
    return get_value(source_, 'min')


def max_value(source_):
    """
    Returns the maximal value of a source data type.

    Parameters
    ----------
    source_ : str, array, dtype or Source
        The source specification.

    Returns
    -------
    max_value : number
       The maximal value for the data type of the source
    """
    max_value = get_value(source_, 'max')
    return max_value


def convert(source_, sink, processes=None, verbose=False, **kwargs):
    """
    Transforms a source into another format.

    Parameters
    ----------
    source_ : source specification
        The source or list of sources.
    sink : source specification
        The sink or list of sinks.

    Returns
    -------
    sink : sink specification
        The sink or list of sinks.
    """
    if isinstance(sink, pathlib.Path):
        sink = str(sink)
    source_ = open_ro(source_)
    if verbose:
        print(f'converting {source_} -> {sink}')
    mod = source_to_module(source_)
    if hasattr(mod, 'convert'):
        return mod.convert(source_, sink, processes=processes, verbose=verbose, **kwargs)
    else:
        return write(sink, source_)


def convert_files(filenames, extension=None, path=None, processes=None, verbose=False, workspace=None, verify=False):
    """
    Transforms list of files to their sink format in parallel.

    Parameters
    ----------
    filenames : list of str | list of pathlib.Path
        The filenames to convert
    extension : str
        The new file format extension.
    path : str or None
        Optional path specification.
    processes : int, 'serial' or None
        The number of processes to use for parallel conversion.
    verbose : bool
        If True, print progress information.

    Returns
    -------
    filenames : list of str
        The new file names.
    """
    if extension.startswith('.'):  # FIXME: downstream code should handle extension with or without dot
        extension = extension[1:]
    if not isinstance(filenames, (tuple, list)):
        filenames = [filenames]
    if len(filenames) == 0:
        return []
    n_files = len(filenames)

    if path is not None:
        filenames = [fu.join(path, fu.split(f)[1]) for f in filenames]  # TODO: replace with pathlib
    sinks = [str(pathlib.Path(f).with_suffix('.'+extension)) for f in filenames]

    if verbose:
        timer = tmr.Timer()
        print(f'Converting {n_files} files to {extension}!')

    if not isinstance(processes, int) and processes != 'serial':
        processes = mp.cpu_count()

    # print(n_files, extension, filenames, sinks)
    _convert = functools.partial(_convert_files, n_files=n_files, extension=extension, verbose=verbose, verify=verify)

    if processes == 'serial':
        [_convert(source_, sink, i) for i, source_, sink in zip(range(n_files), filenames, sinks)]
    else:
        with CancelableProcessPoolExecutor(processes) as executor:
            results = executor.map(_convert, filenames, sinks, range(n_files))
            if workspace is not None:
                workspace.executor = executor
            _ = list(results)  # to catch exceptions
        if workspace is not None:
            workspace.executor = None

    if verbose:
        timer.print_elapsed_time(f'Converting {n_files} files to {extension}')

    return sinks


@ptb.parallel_traceback
def _convert_files(source_, sink, fid, n_files, extension, verbose, verify=False):
    source_ = open_ro(source_)
    if verbose:
        print(f'Converting file {fid}/{n_files} {source_} -> {sink}')
    mod = file_extension_to_module[extension]  # FIXME: ammend file_name_to_module to handle extension for SST
    if mod is None:
        raise ValueError(f"Cannot determine module for extension {extension}!")
    mod.write(sink, source_)
    if verify:
        src_mean = source_.array.mean()
        sink_mean = mod.read(sink).mean()
        if not math.isclose(src_mean, sink_mean, rel_tol=1e-5):
            raise RuntimeError(f"Conversion of {source_} to {sink} failed, means differ")


###############################################################################
# ## Helpers
###############################################################################

_shape = shape
_dtype = dtype
_order = order
_location = location
_memory = memory


###############################################################################
# ## Tests
###############################################################################
def _test():
    import ClearMap.IO.IO as io

    print(io.abspath('.'))
    # reload(io)


# TODO:
#
# def copy(source, sink):
#    """Copy a data file from source to sink, which can consist of multiple files
#    
#    Parameters:
#        source (str): file name of source
#        sink (str): file name of sink
#    
#    Returns:
#        str: name of the copied file
#    
#    See Also:
#        :func:`copyImage`, :func:`copyArray`, :func:`copyData`, :func:`convert`
#    """     
#    
#    return copyData(source, sink);
#
#
# def convert(source, sink, **args):
#    """Transforms data from source format to sink format
#    
#    Parameters:
#        source (str): file name of source
#        sink (str): file name of sink
#    
#    Returns:
#        str: name of the copied file
#        
#    Warning:
#        Not optimized for large image data sets yet
#    
#    See Also:
#        :func:`copyImage`, :func:`combineImage`
#    """      
#
#    if source is None:
#        return None;
#    
#    elif isinstance(source, str):
#        if sink is None:        
#            return read(source, **args);
#        elif isinstance(sink, str):
#            #if args == {} and dataFileNameToType(source) == dataFileNameToType(sink):
#            #    return copy(source, sink);
#            #else:
#            data = read(source, **args); #TODO: improve for large data sets
#            return write(sink, data);
#        else:
#            raise RuntimeError('convert: unknown sink format!');
#            
#    elif isinstance(source, numpy.ndarray):
#        if sink is None:
#            return dataFromRegion(source, **args);
#        elif isinstance(sink,  str):
#            data = dataFromRange(source, **args);
#            return writeData(sink, data);
#        else:
#            raise RuntimeError('convert: unknown sink format!');
#    
#    else:
#      raise RuntimeError('convert: unknown source format!');
    
#
###############################################################################
# # Other
###############################################################################
#
# def writeTable(filename, table):
#    """Writes a numpy array with column names to a csv file.
#    
#    Parameters:
#        filename (str): filename to save table to
#        table (annotated array): table to write to file
#        
#    Returns:
#        str: file name
#    """
#    with open(filename,'w') as f:
#        for sublist in table:
#            f.write(', '.join([str(item) for item in sublist]));
#            f.write('\n');
#        f.close();
#
#    return filename;
#

# Temp

# def isFileExpression(source):
#    """Checks if filename is a regular expression denoting a file list
#    
#    Parameters:
#        source (str): source file name
#        
#    Returns:
#        bool: True if source is regular expression with a digit placeholder
#    """    
#
#    if not isinstance(source, str):
#      return False;    
#      
#    ext = fileExtension(source);
#    if not ext in dataFileExtensions:
#      return False
#    
#    #sepcified number of digits
#    searchRegex = re.compile('.*\\\\d\{(?P<digit>\d)\}.*').search
#    m = searchRegex(source); 
#    if not m is None:
#      return True;
#    
#    #digits without trailing zeros \d* or 
#    searchRegex = re.compile('.*\\\\d\*.*').search
#    m = searchRegex(source); 
#    if not m is None:
#      return True;
#    
#    #digits without trailing zeros \d{} or 
#    searchRegex = re.compile('.*\\\\d\{\}.*').search
#    m = searchRegex(source); 
#    if not m is None:
#      return True;
#      
#    return False;
#
#     
# def isDataFile(source, exists = False):
#    """Checks if a file has a valid data file extension usable in *ClearMap*
#     
#    Parameters:
#        source (str): source file name
#        exists (bool): if true also checks if source exists 
#        
#    Returns:
#        bool: true if source is an data file usable in *ClearMap*
#    """   
#    
#    if not isinstance(source, str):
#        return False;    
#    
#    fext = fileExtension(source);
#    if fext in dataFileExtensions:
#      if not exists:
#        return True;
#      else:
#        return exitsFile(source) or existsFileExpression(source);
#    else:
#        return False;
#        
#
# def isImageFile(source, exists = False):
#    """Checks if a file has a valid image file extension usable in *ClearMap*
#     
#    Parameters:
#        source (str): source file name
#        exists (bool): if true also checks if source exists 
#        
#    Returns:
#        bool: true if source is an image file usable in *ClearMap*
#    """   
#    
#    if not isinstance(source, str):
#        return False;    
#    
#    fext = fileExtension(source);
#    if fext in imageFileExtensions:
#      if not exists:
#        return True;
#      else:
#        return exitsFile(source) or existsFileExpression(source);
#    else:
#        return False;
#
#
# def isArrayFile(source, exists =False):
#    """Checks if a file is a valid array data file
#     
#    Parameters:
#        source (str): source file name
#        exists (bool): if true also checks if source exists 
#        
#    Returns:
#        bool: true if source is a array data file
#    """     
#    
#    if not isinstance(source, str):
#        return False;
#    
#    fext = fileExtension(source);
#    if fext in pointFileExtensions:
#      if not exists:
#        return True;
#      else:
#        return exitsFile(source) or existsFileExpression(source);
#    else:
#        return False;
#
#
# def isDataSource(source, exists = False):
#  """Checks if source is a valid data source for use in *ClearMap*
#   
#  Parameters:
#      source (str): source file name or array
#      exists (bool): if true also checks if source exists 
#      
#  Returns:
#      bool: true if source is an data source usable in *ClearMap*
#  """  
#  
#  return (not exists and source is None) or isinstance(source, numpy.ndarray) or isDataFile(source, exists = exists);
