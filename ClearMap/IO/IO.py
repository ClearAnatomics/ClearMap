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

data = io.read('signal.tif')  # returns np.ndarray
data = io.read('volume.npy',
               slicing=(slice(0, 100),))  # sub-slice
io.write('output.tif', data)

**Source objects** — richer than raw arrays; carry shape, dtype, and location:

.. code-block:: python

    src = dispatch.as_source('volume.npy')
print(src.shape, src.dtype, src.order)
data = src[10:20, :, :]  # lazy slicing

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

    conversion.convert_files(files, extension='.npy', processes=8)

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

import pathlib
from contextlib import contextmanager

import numpy as np

import ClearMap.IO.source.Source as source_mod
import ClearMap.IO.source.backends.TIF as tif
import ClearMap.IO.source.backends.NRRD as nrrd
import ClearMap.IO.source.backends.CSV as csv
import ClearMap.IO.source.backends.NPY as npy
import ClearMap.IO.source.backends.MMP as mmp
import ClearMap.IO.source.backends.SMA as sma
import ClearMap.IO.source.backends.MHD as mhd

from ClearMap.IO.dispatch import location_to_module, as_source
from ClearMap.IO.io_ops import open_ro

from ClearMap.Utils.exceptions import (SourceModuleNotFoundError)

try:
    import ClearMap.IO.source.backends.GT as gt
    gt_loaded = True
except ImportError:
    gt_loaded = False
import ClearMap.IO.source.backends.FileList as fl

import ClearMap.Utils.tag_expression as te

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


file_extension_to_module = {
    'npy': mmp,
    'tif': tif,
    'tiff': tif,
    'nrrd': nrrd,
    'nrdh': nrrd,
    'csv': csv,
    'mhd': mhd
}

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


###############################################################################
# ## Tests
###############################################################################
def _test():
    import ClearMap.IO.IO as io

    print(io.abspath('.'))
    # reload(io)
