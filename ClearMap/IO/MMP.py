# -*- coding: utf-8 -*-
"""
MMP
===

Interface to numpy memmaps

Note
----
For image processing we use [x,y,z] order of arrays. 
To speed up access to z-planes memmaps are created in Fortran order by default.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>, Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__   = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'https://idisco.info'
__download__  = 'httpss://github.com/ClearAnatomics/ClearMap'

import pathlib
import warnings

import numpy as np

import ClearMap.IO.Source as src
import ClearMap.IO.Slice as slc
import ClearMap.IO.NPY as npy
import ClearMap.IO.FileUtils as fu
from ClearMap.Utils.exceptions import (ClearMapPermissionError, ClearMapFileNotFoundError, ClearMapValueError,
                                       ClearMapRuntimeError, ClearMapIoException)


###############################################################################
### Source class
###############################################################################

class Source(npy.Source):
    """Memory mapped array source."""

    def __init__(self, location = None, shape = None, dtype = None, order = None, array = None, mode = None, name = None):
        """Memory mapped source constructor.

        Arguments
        ---------
        array : array
            The underlying data array of this source.
        """
        memmap = _memmap(location=location, shape=shape, dtype=dtype, order=order, mode=mode, array=array)  # FIXME: dangerous location
        super(Source, self).__init__(array=memmap, name=name)

    @property
    def name(self):
        return "Memmap-Source"

    @property
    def array(self):
        """The underlying data array.

        Returns
        -------
        array : array or np.ndarray
            The underlying data array of this source.
        """
        return self._array

    @array.setter
    def array(self, value):
        if not isinstance(value, np.memmap):
            array = np.asarray(value)
            value = _memmap(location=self.location, array=array, mode='w+')  # Explicit write
        self._array = value

    @property
    def dtype(self):
        """The data type of the source.

        Returns
        -------
        dtype : dtype
            The data type of the source.
        """
        return self._array.dtype

    @dtype.setter
    def dtype(self, value):
        if np.dtype(value) != self.dtype:
            self.array = np.asarray(self.array, dtype=value)

    @property
    def order(self):
        """The order of how the data is stored in the source.

        Returns
        -------
        order : str
            Returns 'C' for C contiguous and 'F' for Fortran contiguous, None otherwise.
        """
        return npy.order(self.array)

    @order.setter
    def order(self, value):
        if value != self.order:
            self.array = np.asarray(self.array, order=value)


    @property
    def location(self):
        """The location where the data of the source is stored.

        Returns
        -------
        location : str or None
            Returns the location of the data source or None if this source lives in memory only.
        """
        return self._array.filename

    @location.setter
    def location(self, value):  # FIXME: should only accept path
        if value != self.location:
            # mode is None → _memmap infers: file exists -> 'r+';  file missing + shape given → 'w+' (creates)
            memmap = _memmap(location=value, shape=self.shape, dtype=self.dtype, order=self.order)
            self.array = memmap

    @property
    def offset(self):
        """The offset of the memory map in the file.

        Returns
        -------
        offset : int
            Offset of the memory map in the file.
        """
        return self._array.offset

    def as_virtual(self):
        return VirtualSource(source=self)

    def as_buffer(self):
        return self._array


class VirtualSource(src.VirtualSource):
    """Virtual memory map source."""
    _real_class = Source

    def __init__(self, source = None, shape = None, dtype = None, order = None, name = None):
        super(VirtualSource, self).__init__(source=source, shape=shape, dtype=dtype, order=order, name=name)
        super().__init__(source=source, shape=shape, dtype=dtype, order=order, name=name, mode=mode)

    def as_real(self):
        return Source(location=self.location, shape=self.shape, dtype=self.dtype, order=self.order, name=self.name)

    @property
    def array(self):
       return self.as_real().array


###############################################################################
### IO Interface
###############################################################################

def is_memmap(source):
    if isinstance(source, (np.memmap, Source)):
        return True
    elif isinstance(source, str):
        if fu.is_file(source):
            try:
                _ = np.memmap(source)
            except:
                return False
        return True
    else:
        return False


def read(source, slicing=None, mode=None, **kwargs):
    """Read data from a memory mapped source.

    Arguments
    ---------
    source : str, memmap, or Source
        The source to read the data from.
    slicing : slice specification
        Optional slice specification of memmap to read from.
    mode : str
        Optional mode specification of how to open the memmap.

    Returns
    -------
    source : Source
        The read memmap source.
    """
    mode = mode if mode is not None else 'r+'  # FIXME: this is edit, not read

    if isinstance(source, Source):
        if slicing is None:
            return source
        else:
            return source.__getitem__(slicing)
    elif isinstance(source, np.memmap):
        if slicing is None:
            memmap = source
        else:
            memmap = source.__getitem__(slicing)
        return Source(array = memmap)
    elif isinstance(source, str):
        try:
            memmap = _memmap(location=source, mode=mode)
        except FileNotFoundError:
            raise
        except Exception as err:
            raise ValueError(f'Cannot read memmap from location {source!r}!') from err

        if slicing is not None:
            memmap = memmap.__getitem__(slicing)

        return Source(array = memmap)
    else:
        raise ValueError(f'Cannot read memmap from source {source!r}!')


def write(sink, data, slicing=None, **kwargs):
    """Write data to a memory map.

    Arguments
    ---------
    sink : str, memmap, or Source
        The sink to write the data to.
    data : array
        The data to write int the sink.
    slicing : slice specification or None
        Optional slice specification of an existing memmap to write to.

    Returns
    -------
    sink : str, memmap, or Source
        The sink.
    """
    if slc.is_trivial(slicing):
        slicing = (slice(None),)

    if isinstance(sink, (Source, np.memmap)):
        sink.__setitem__(slicing, data.array)
    elif isinstance(sink, str):
        if slicing == (slice(None),):
            create(location=sink, array=data.array)
        else:
            try:
                memmap = _memmap(location=sink, mode='r+')  # FIXME: r+ = edit (or read_and_write)
            except:
                raise ValueError(f'Cannot write slice into non-existent memmap at {sink=!r}!')
            memmap.__setitem__(slicing, data.array)
    else:
        raise ValueError(f'Cannot write memmap to {sink=!r}!')

    return sink


def create(location = None, shape = None, dtype = None, order = None,
           mode = None, array = None, as_source = True, **kwargs):
    """Create a memory map.

    Arguments
    ---------
    location : str
        The filename of the memory mapped array.
    shape : tuple or None
        The shape of the memory map to create.
    dtype : dtype
        The data type of the memory map.
    order : 'C', 'F', or None
        The contiguous order of the memmap.
    mode : 'r', 'w', 'w+', None
        The mode to open the memory map.
    array : array, Source or None
        Optional source with data to fill the memory map with.
    as_source : bool
        If True, return as Source class.

    Returns
    -------
    memmap : np.memmap
        The memory map.

    Note
    ----
    By default memmaps are initialized as Fortran contiguous if order is None.
    """
    if mode is not None and mode != 'w+':
        raise ValueError(f'create() only supports mode="w+", got {mode!r}. '
                         f'Use read() to open existing files or initialize() for read-or-create behaviour.')
    memmap = _memmap(location=location, shape=shape, dtype=dtype, order=order, mode='w+', array=array)
    if as_source:
        return Source(memmap)
    else:
       return memmap


###############################################################################
### Helpers
###############################################################################

def _open_memmap(location, mode=None, shape=None, dtype=None, order=None, context=''):
    """
    Thin wrapper around ``np.lib.format.open_memmap`` that converts
    any unexpected exception into a :class:`ClearMapRuntimeError` with
    full argument context.

    Parameters
    ----------
    location : str | Path
        File path.
    mode : str or None
        File mode (``'r'``, ``'r+'``, ``'w+'``, etc.).
    shape : tuple or None
        Array shape (required when ``mode='w+'``).
    dtype : dtype or None
        Array dtype (required when ``mode='w+'``).
    order : 'C', 'F', or None
        Whether to use Fortran (column-major) memory layout.
    context : str
        Short description of what the caller was trying to do,
        included in the error message (e.g. ``'creating from array'``,
        ``'reopening as r+'``).

    Returns
    -------
    memmap : np.memmap

    Raises
    ------
    ClearMapPermissionError
    ClearMapFileNotFoundError
    ClearMapValueError
    ClearMapRuntimeError

    """
    kwargs = dict(mode=mode)
    if shape is not None:
        kwargs['shape'] = shape
    if dtype is not None:
        kwargs['dtype'] = dtype
    if mode == 'w+':
        kwargs['fortran_order'] = order in ('F', None)

    try:
        location = str(location)
    except TypeError as err:
        action = f' while {context}' if context else ''
        raise ClearMapValueError(f'Invalid location type{action}.',
                                 value=type(location).__name__, expected='str or pathlib.Path') from err

    try:
        return np.lib.format.open_memmap(location, **kwargs)
    except PermissionError as err:
        detail = f'{location=!r}, {mode=!r}'
        action = f' while {context}' if context else ''
        raise ClearMapPermissionError(f'Permission denied{action}: {detail}') from err
    except FileNotFoundError as err:
        raise ClearMapFileNotFoundError(f'File not found while {context}: {location!r}') from err
    except ValueError as err:
        detail = f'{location=!r}, {mode=!r}, {shape=}, {dtype=}, {order=}'
        raise ClearMapValueError(f'Invalid parameters for memmap while {context}:'
                                 f' {detail}\nNumpy says: {err}') from err
    except Exception as err:  # Unexpected errors fall through to RuntimeError
        detail = f'{location=!r}, {mode=!r}, {shape=}, {dtype=}, {order=}'
        action = f' while {context}' if context else ''
        raise ClearMapRuntimeError(f'Unexpected error opening memmap{action}: {detail}\n{err}') from err


def _try_read_existing(location, mode):
    """Attempt to read an existing memmap, handling the read-only fallback on permission errors."""
    try:
        if mode:
            return _open_memmap(location, mode=mode, context='reading existing file')
        else:
            try:
                return _open_memmap(location, context='reading existing file')
            except ClearMapPermissionError:  # Read fallback
                return _open_memmap(location, mode='r', context='reading existing file (fallback read-only)')
    except (ClearMapRuntimeError, ClearMapValueError, ClearMapFileNotFoundError):
        # ignore general runtime/value errors here and fall through to creation
        return None


def _resolve_params(array, location=None, shape=None, dtype=None, order=None):
    """Extract missing parameters from a source array (ndarray or memmap)."""
    if isinstance(array, np.memmap):
        location = location if location is not None else array.filename
        location = fu.abspath(location) if location is not None else location

    shape = shape if shape is not None else array.shape
    dtype = dtype if dtype is not None else array.dtype
    order = order if order is not None else npy.order(array)

    return location, shape, dtype, order


def _is_exact_memmap_match(array, location, shape, dtype, order):
    """Check if an existing memmap exactly matches the requested target parameters."""
    return (shape == array.shape and
            dtype == array.dtype and
            order == npy.order(array) and
            fu.abspath(location) == fu.abspath(array.filename))


def _reopen_with_mode(location: str | pathlib.Path | None, memmap: np.memmap, mode: str | None) -> np.memmap:
    # reopen in requested mode if different from 'w+' or current mode
    """
    Reopen the memmap with the desired mode if it doesn't already match.
    i.e. if different from 'w+' or current mode
    """
    desired_mode = mode or 'r+'
    if desired_mode == memmap.mode:  # Already open in the requested mode — nothing to do
        return memmap
    elif desired_mode == 'w+':  # Reopening as 'w+' would truncate the file we just wrote — refuse
        return memmap
    else:
        return _open_memmap(location, mode=desired_mode, context=f'reopening as {desired_mode!r}')


def _memmap(location=None, shape=None, dtype=None, order=None,
            mode=None, array=None):
    """
    Create a memory map.

    Arguments
    ---------
    location : str
        The filename of the memory mapped array.
    shape : tuple or None
        The shape of the memory map to create.
    dtype : dtype
        The data type of the memory map.
    order : 'C', 'F', or None
        The contiguous order of the memmap.
    mode : 'r', 'w', 'w+', None
        The mode to open the memory map.
    array : array, Source or None
        Optional source with data to fill the memory map with.

    Returns
    -------
    memmap : np.memmap
        The memory map.

    Raises
    ------
    ClearMapIoException
        When the memmap cannot be created or opened for a known reason.
    ClearMapValueError
        When arguments are invalid (missing location, shape mismatch, etc.).
    ClearMapRuntimeError
        When an unexpected error occurs inside numpy's memmap machinery
        (via :func:`_open_memmap`).

    Note
    ----
    By default memmaps are initialised as Fortran contiguous if order is None.
    """
    # ── 1. Normalize Inputs ───────────────────────────────────────────
    if isinstance(location, pathlib.Path):
        location = str(location)
    if isinstance(location, np.memmap):
        array = location
        location = None

    # Try reading existing file if no array is provided
    if array is None:
        if isinstance(location, str):
            if mode != 'w+' and fu.is_file(location):
                # if fails, array stays None, fall through to creation
                array = _try_read_existing(location, mode)
        else:
            raise ClearMapIoException(f'Cannot create memmap without a location! '
                                      f'Args: {location=} (type={type(location).__name__}), '
                                      f'{shape}, {dtype=}, {order=}, {mode=}')

    # Still no array -> Create entirely new file if we have enough info, raise otherwise
    if array is None:
        if shape is not None:
            mode = 'w+' if mode is None else mode
            memmap = _open_memmap(location, mode=mode, shape=shape, dtype=dtype, order=order, context='creating new file')
        else:
            if isinstance(location, str) and not fu.is_file(location) and mode != 'w+':
                # We have location and mode is not EXPLICITLY write. Then we infer we tried to read
                raise ClearMapFileNotFoundError(f'Memmap file not found at {location!r}. '
                                                f'Cannot read source (and cannot create without shape).')
            else:  # we had no array, no shape, and no existing file -> probable creation attempt but not enough info
                raise ClearMapIoException(f'Cannot create memmap without shape at location {location!r}!')  # FIXME: message
    elif isinstance(array, np.memmap):    # Existing memmap source -> check and use, or fallback to copy
        location, shape, dtype, order = _resolve_params(array, location, shape, dtype, order)

        if _is_exact_memmap_match(array, location, shape, dtype, order):
            memmap = array  # shape=array.shape already checked above
        else:  # Fallback: create a new memmap and copy data if shapes align
            if shape == array.shape:
                memmap = _open_memmap(location, mode='w+', shape=shape, dtype=dtype, order=order,
                                      context='creating from memmap source')
                memmap[:] = array
            else:
                raise ClearMapValueError(f'Cannot create memmap from source: shape {shape!r} does not match '
                                         f'array shape {array.shape!r}. Explicit reshaping or slicing is required.',
                                         value=array.shape, expected=shape)
        memmap = _reopen_with_mode(location, memmap, mode)
    elif isinstance(array, np.ndarray):    # Existing ndarray source -> cast to memmap and use
        location, shape, dtype, order = _resolve_params(array, location, shape, dtype, order)

        if isinstance(location, str):
            if shape == array.shape:
                memmap = _open_memmap(location, mode='w+', shape=shape, dtype=dtype, order=order,
                                      context='creating from ndarray')
                memmap[:] = array
            else:
                raise ClearMapValueError(f'Cannot create memmap from source: shape {shape!r} does not match '
                                         f'array shape {array.shape!r}. Explicit reshaping or slicing is required.',
                                         value=array.shape, expected=shape)
        else:
            raise ClearMapIoException(f'Cannot create memmap without a location! Got {location} of type {type(location).__name__}')
        memmap = _reopen_with_mode(location, memmap, mode)
    else:  # No valid input found -> raise
        raise ClearMapValueError(f'Array type {type(array).__name__} is not valid for memmap creation!',
                                 value=type(array).__name__, expected='np.ndarray, np.memmap, or None')

    return memmap


def header_size(filename):
    """Return the offset of a header in a memmaped file.

    Arguments
    ---------
    filename : str
        Filename of the npy fie.

    Returns
    -------
    offset : int
        The offest due to the header.
    """
    with open(filename, 'rb') as f:
        major, minor = np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        offset = f.tell()

    return offset


###############################################################################
### Tests
###############################################################################

def _test():
    import ClearMap.IO.MMP as mmp
    # reload(mmp)

    m = mmp.Source(location = 'test.npy', shape = 4)
    print(m)

    m[:] = 5
    print(m)

    import ClearMap.IO.Slice as slc

    s = slc.Slice(source = m, slicing = slice(1,3))
    print(s)

    s[:] = 3
    print(s)
    print(m)
#  
# # extract info from source if given
# if isinstance(location, src.Source):
#     shape = source.shape
#     if dtype is None:
#         dtype = source.dtype
#     if order is None:
#         order = npy.order(source)
#     if location is None:
#         location = source.location
# elif isinstance(source, np.ndarray):
#     memmap = read(location=location, mode=mode)
#        if shape != memmap.shape or dtype != memmap.dtype or order != order:
#            memmap = create()
#      else:
#          if shape is None and dtype is None and order is None:
#              raise ValueError('No way to initialize the source, a location is needed!')
#      memmap = create(location = location, shape = shape, dtype = dtype, order = order, mode = mode, source = array, as_source = False)
# # write data if given
# if isinstance(source, src.Source) and source.array is not None:
#     memmap[:] = source.array
# elif isinstance(source, np.ndarray):
#     memmap[:] = source
#  
# if as_source:
#     return Source(array = memmap)
# else:
#     return memmap


# class Source(np.memmap):
#     """Memory map source class"""
#
#     def __new__(cls, filename, shape=None, dtype=None, order=None, mode=None):
#         if isinstance(filename, np.memmap):
#             self = filename
#         elif fu.is_file(filename):
#             self = read(filename, mode = mode)
#         else:
#             self = create(filename, dtype=dtype, shape=shape, order=order, mode=mode)
#         self = self.view(cls)
#         return self
#
#     def name(self):
#         return "Source-Memmap"
#
#     @property
#     def order(self):
#         return npy.order(self)
#
#     @property
#     def array_strides(self):
#         return tuple(np.array(self.strides, dtype = int) / self.itemsize)
#
#     def array(self, *args, **kwargs):
#         return self.view(np.memmap)
#
#     def __str__(self):
#         if hasattr(self, 'filename') and self.filename is not None:
#             info = f'{{0}}'
#         else:
#             info = ''
#
#         dtype = self.dtype
#         if hasattr(dtype, 'name'):
#            dtype = dtype.name
#
#         return f"{self.name()}{self.shape!r}[{dtype!r}]{info}"
#
#     def __repr__(self):
#         return self.__str__()
