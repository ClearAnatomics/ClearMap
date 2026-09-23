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

import os
import pathlib
import tempfile
import warnings

import numpy as np

from ClearMap.IO.source import Source as source_mod, geometry_utils
from ClearMap.IO.source.geometry_utils import resolve_geometry, properties_match
from ClearMap.IO.source.protocol import Backend

from ClearMap.IO.source.source_modes import (VALID_MODES, EXISTING_FILE_MODES, PERSISTABLE_MODES,
                                             DEFAULT_EDIT_MODE,  mode_after_create)
from ClearMap.IO.source.backends.npy_backend import NumpySource
import ClearMap.IO.source.Slice as slc
import ClearMap.IO.FileUtils as fu

from ClearMap.Utils.tag_expression import Expression
from ClearMap.Utils.exceptions import (ClearMapPermissionError, ClearMapFileNotFoundError, ClearMapValueError,
                                       ClearMapRuntimeError, ClearMapIoException, SourceNotFoundError,
                                       IncompleteSourceSpecError, IncompatibleSource, SourceExistsError)


###############################################################################
### Source class
###############################################################################

def _close_memmap(array):
    """Flush and release the OS mapping behind *array*.

    Only call this on a memmap you own. Closing the mapping invalidates every view
    derived from it; subsequent access raises ``ValueError: mmap closed or invalid``.
    Required before renaming or unlinking on Windows, where an open mapping locks
    the file.
    """
    if isinstance(array, MMPSource):
        array = array.array
    if not isinstance(array, np.memmap):
        return

    if array.flags['OWNDATA'] or isinstance(array.base, np.ndarray):
        raise ClearMapRuntimeError(f'Refusing to close the mapping behind a derived array ({type(array).__name__}, '
                                   f'owndata={array.flags["OWNDATA"]}); close the base memmap instead.')

    try:
        if array.mode in PERSISTABLE_MODES:
            array.flush()
    finally:
        handle = getattr(array, '_mmap', None)
        if handle is not None:
            handle.close()


def _assert_durable_sink(sink, context: str = ''):
    """Raise unless writes through ``sink[...] = ...`` will actually reach disk."""
    ctx = f' ({context})' if context else ''

    # Unwrap slice to their base source, if any, to check the underlying array's persistability
    while isinstance(sink, slc.Slice):
        sink = sink.source

    if isinstance(sink, MMPSource):
        if not sink.is_persistable:
            raise ClearMapPermissionError(f'Cannot write to {sink!r}{ctx}: opened with mode={sink.mode!r}. '
                                          f'One of {PERSISTABLE_MODES} is required; use io.edit() to reopen for writing.')
        return

    if isinstance(sink, np.memmap):                      # before ndarray because mmemap is a subclass of ndarray!
        if sink.mode not in PERSISTABLE_MODES:
            hint = ("mode='c' writes go to a private copy-on-write page and are silently discarded"
                    if sink.mode == 'c' else 'the file is mapped read-only')
            raise ClearMapPermissionError(f'Cannot write to memmap {sink.filename!r}{ctx}: mode={sink.mode!r} — {hint}.')
        return

    if isinstance(sink, np.ndarray):
        if not sink.flags.writeable:
            raise ClearMapPermissionError(f'Cannot write to a read-only array{ctx}.')
        return


def _as_array(data):
    """Array view of *data*, accepting an MMPSource, a memmap or anything array-like."""
    if isinstance(data, MMPSource):
        return data.array
    return np.asanyarray(data)


def _resolve_params(array, location=None, shape=None, dtype=None, order=None):
    """Resolve memmap target location and geometry from an input array."""
    if isinstance(array, np.memmap) and location is None:
        location = array.filename

    if location is not None:
        location = fu.abspath(location)

    shape, dtype, order = resolve_geometry(shape=shape, dtype=dtype, order_=order, array=array, default_order='F')
    return location, shape, dtype, order


def _flush_sink(sink):
    array = sink.array if isinstance(sink, MMPSource) else sink
    if isinstance(array, np.memmap):
        array.flush()


def _validate_write_request(sink, data, kwargs):
    """Reject requests that cannot be honoured, before anything is opened or created."""
    if isinstance(sink, Expression):
        raise ClearMapValueError(f'Cannot write an expression sink {sink} with MMP; use FileList.',
                                 value=sink, expected='a single-file location, a Source or a numpy array')
    if data is None:
        raise ClearMapValueError('write() requires data; use create() to make a blank sink.',
                                 value=data, expected='an array, memmap or Source')
    if 'mode' in kwargs:
        raise ClearMapValueError('write() controls the mode of its sink; remove mode= from the call.',
                                 value=kwargs['mode'], expected=None)
    if kwargs and not isinstance(sink, (str, pathlib.Path)):
        raise ClearMapValueError(f'Unexpected keyword arguments {sorted(kwargs)} when writing into an existing sink; '
                                 f'kwargs are only supported for file creation.')


def _create_sink(path, array, overwrite, kwargs):
    """Create (or replace) a file sized for *array*. The data is written by the caller."""
    if path.exists() and not overwrite:
        raise SourceExistsError(location=str(path),
                                message=f'Cannot write to {path}: pass overwrite=True to replace it.')
    kwargs.setdefault('shape', array.shape)
    kwargs.setdefault('dtype', array.dtype)
    kwargs.setdefault('order', geometry_utils.order(array))
    return create(location=str(path), mode='w+', **kwargs)


def _create_block_sink(path, array, slicing, overwrite, kwargs):
    """Open the existing file to be patched, or create one if its geometry was declared."""
    if not path.is_file():
        if 'shape' not in kwargs:
            raise SourceNotFoundError(location=str(path),
                                      message=(f'Cannot write slicing {slicing} to {path}: file does not exist and '
                                               f'shape was not supplied. Create the sink first, or pass shape= to '
                                               f'write().'))
        kwargs.setdefault('dtype', array.dtype)
        return create(location=str(path), mode='w+', **kwargs)

    sink = MMPSource(str(path), mode=DEFAULT_EDIT_MODE)
    if properties_match(sink, **kwargs):
        return sink
    return _recreate_sink(sink, path, overwrite, kwargs)


def _recreate_sink(sink, path, overwrite, kwargs):
    """Replace *sink* to satisfy *kwargs*; the .npy header cannot be rewritten in place."""
    existing = {'shape': sink.shape, 'dtype': sink.dtype, 'order': sink.order}
    requested = {key: kwargs[key] for key in existing if kwargs.get(key) is not None}
    if not overwrite:
        raise SourceExistsError(location=str(path),
                                message=(f'{path} exists with {existing}, but {requested} was requested. Pass '
                                         f'overwrite=True to recreate it, or drop the mismatched arguments to write '
                                         f'into the file as it is'))
    warnings.warn(f'Recreating {path}: requested {requested} differs from the existing file ({existing}). '
                  f'Existing contents are lost.', stacklevel=4)
    _close_memmap(sink.array)  # sink is local; the mapping is released on return (Windows lock)
    for key, value in existing.items():  # keep the properties that were not overridden
        kwargs.setdefault(key, value)
    return create(location=str(path), mode='w+', **kwargs)


def _assign(sink, array, slicing):
    """Assign into *sink*, translating numpy's refusals into ClearMap exceptions."""
    slicing_ = slice(None) if slicing is None else slicing  # sink[None] is np.newaxis
    try:
        sink[slicing_] = array
    except ValueError as err:
        if 'read-only' in str(err) or 'assignment destination' in str(err):
            # Shouldn't reach but ensure ClearMap context added to pure np exceptions
            raise ClearMapPermissionError(f'Write to {sink!r} was refused by numpy: {err}') from err
        raise ClearMapValueError(f'Cannot write data of shape {np.shape(array)} and dtype {np.dtype(array.dtype)} '
                                 f'into {sink!r}[{slicing_}]: {err}') from err


class MMPSource(NumpySource):
    """Memory mapped array source."""
    backend = Backend.MMP

    def __init__(self, location=None, shape=None, dtype=None, order=None,
                 array=None, mode=None, name=None):
        """Memory mapped source constructor.

        Arguments
        ---------
        array : array
            The underlying data array of this source.
        """
        location = fu.normalize_location_spec(location)

        if mode is None and array is None and location is not None:
            mode = 'r+' if fu.is_file(location) else 'w+'
            warnings.warn(f'Constructing MMPSource without explicit mode is deprecated. Inferred mode={mode!r}. '
                          f'Use mode="r" to read, mode="r+" to edit, mode="w+" to create.', FutureWarning, stacklevel=2)

        if mode is None and array is not None:
            mode = 'w+'

        if mode not in VALID_MODES:
            raise ClearMapValueError(f'Invalid mode {mode!r}.', value=mode, expected=VALID_MODES)

        if mode in EXISTING_FILE_MODES:
            memmap = self._open_existing(location, mode=mode)
        else:  # 'w+'
            memmap = self._create_new(location, shape=shape, dtype=dtype, order=order, array=array)

        final_mode = mode_after_create(self._mode)
        if memmap.mode != final_mode:
            memmap = _reopen_with_mode(location, memmap, mode=final_mode)

        super().__init__(array=memmap, name=name, mode=final_mode)
        self._check_mode_invariant()

    @property
    def mode(self):
        return self._mode

    @staticmethod
    def _open_existing(location, mode):
        if not isinstance(location, str):
            raise ClearMapValueError(f'Cannot open memmap: location must be a string, got {type(location).__name__}',
                                     value=type(location).__name__, expected='str')
        if not fu.is_file(location):
            raise SourceNotFoundError(f'Cannot open memmap in mode {mode!r}', location=location)
        return _open_memmap(location, mode=mode, context=f'opening existing file as {mode!r}')

    def _check_mode_invariant(self):
        array_mode = getattr(self._array, 'mode', None)
        if array_mode is not None and array_mode != self._mode:
            raise ClearMapRuntimeError(f'Mode divergence on {self.location}: source says {self._mode!r},'
                                       f' mapping says {array_mode!r}.')

    @staticmethod
    def _create_new(location, shape, dtype, order, array):
        if array is not None:
            return _create_from_array(location, array, shape, dtype, order, mode='w+', context='creating in __init__')

        shape, dtype, order = resolve_geometry(shape=shape, dtype=dtype, order_=order, default_order='F')
        if shape is None:
            raise IncompleteSourceSpecError('Cannot create memmap without shape or source array!',
                                            value=None, expected='shape or array')
        if dtype is None:
            raise IncompleteSourceSpecError('Cannot create memmap without dtype or source array!',
                                            value=None, expected='dtype or array')

        return _open_memmap(location, mode='w+', shape=shape, dtype=dtype, order=order, context='creating empty in __init__')

    @property
    def array(self):
        """The underlying data array.

        Returns
        -------
        array : array or np.ndarray
            The underlying data array of this source.
        """
        return self._array

    # FIXME:  truncates and rewrites the file on disk even when self._mode == 'r' or 'c', and leaves self._array.mode == 'w+' while self._mode still says 'r'
    @array.setter
    def array(self, value):
        if not isinstance(value, np.memmap):
            array = np.asarray(value)
            value = _create_from_array(location=self.location, array=array, mode='w+', context='Source.array setter')
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
        return geometry_utils.order(self.array)

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
        if value == self.location:
            return
        self.copy_to(value, repoint=True)

    def copy_to(self, location, if_exists='error', repoint=False):
        """Materialise this source's data at *location*.

        repoint=False (default) returns a new Source; ``self`` keeps its file.
        repoint=True repoints ``self`` at the new file and leaves the old one on disk.
        """
        return self._relocate(location, if_exists=if_exists, delete_source=False, repoint=repoint)

    def move_to(self, location, if_exists='error'):
        """Relocate the backing file and repoint this source at it."""
        return self._relocate(location, if_exists=if_exists, delete_source=True, repoint=True)

    def _relocate(self, location, *, if_exists='error', delete_source=False, repoint=True):
        target = pathlib.Path(location)
        old_array, old_location = self._array, pathlib.Path(self.location)
        if if_exists not in ('error', 'overwrite', 'adopt'):
            raise ClearMapValueError('Invalid if_exists', value=if_exists, expected=('error', 'overwrite', 'adopt'))
        if self.location is not None and target.resolve() == old_location.resolve():  # WARNING: keep self.location here because of bad Falseness of pathlib
            return self

        # 1. validate before touching anything
        if delete_source and self._mode not in PERSISTABLE_MODES:
            raise ClearMapPermissionError(f'Cannot move {self!r}: opened with mode={self._mode!r}. '
                                          f'Reopen with io.edit() before moving it.')

        dtype, shape = old_array.dtype, old_array.shape
        order = 'F' if old_array.flags['F_CONTIGUOUS'] else 'C'
        # 'w+' must never be carried forward; a read-only source stays read-only.
        new_mode = mode_after_create(self._mode) or DEFAULT_EDIT_MODE

        if target.exists():
            if if_exists == 'error':
                raise SourceExistsError(location=str(target), message=(f'{target} already exists; '
                                                                       f'pass if_exists="overwrite" to replace '
                                                                       f'it or if_exists="adopt" to open it instead of writing.'))
            if if_exists == 'adopt':
                adopted = MMPSource(target, mode=new_mode)
                if not properties_match(adopted, shape=shape, dtype=dtype, order=order):
                    raise IncompatibleSource(adopted, 'geomtetry',{'shape': shape, 'dtype': dtype, 'order': order})

                warnings.warn(f'Adopting existing file {location}; the current contents of '
                              f'{self!r} are discarded, not copied.', stacklevel=3)
                if not repoint:
                    return adopted
                _close_memmap(old_array)
                self._array, self.location, self._mode = adopted.array, target, new_mode
                return self

        # 2. stage into a sibling temp file: a failure never clobbers the destination
        tmp_fd, tmp_name = tempfile.mkstemp(prefix='.cm-relocate-', dir=str(target.parent))
        os.close(tmp_fd)
        tmp_path = pathlib.Path(tmp_name)
        try:
            staged = _create_from_array(location=str(tmp_path), array=old_array, mode='w+', context='MMPSource._relocate')
            _close_memmap(staged)
            del staged                      # release the mapping before the rename (Windows)
            tmp_path.replace(target)        # atomic within one filesystem
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise                           # self is completely untouched

        if not repoint:
            return MMPSource(target, mode=new_mode)

        # 3. only now mutate self
        _close_memmap(old_array)
        self._array = _open_memmap(target, mode=new_mode, context='reopening relocated memmap')
        self.location = target
        self._mode = new_mode
        if delete_source and old_location is not None:
            pathlib.Path(old_location).unlink(missing_ok=True)
        self._check_mode_invariant()
        return self


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
        return MMPVirtualSource(source=self)

    def as_buffer(self):
        return self._array


class MMPVirtualSource(source_mod.VirtualSource):
    """Virtual memory map source."""
    _real_class = MMPSource

    def __init__(self, source=None, shape=None,
                 dtype=None, order=None, name=None, mode=None):
        super().__init__(source=source, shape=shape, dtype=dtype, order=order, name=name, mode=mode)

    def as_real(self):
        return self._real_class(location=self.location, shape=self.shape, dtype=self.dtype, order=self.order,
                                name=self.name, mode=self.mode)

    @property
    def array(self):
       return self.as_real().array

SOURCE_CLASS = MMPSource
###############################################################################
### IO Interface
###############################################################################

def is_memmap(source):
    if isinstance(source, (np.memmap, MMPSource)):
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
    if mode == 'r+':
        warnings.warn('read() does not support mode="r+" (edit mode). Use edit() instead.',
                       FutureWarning, stacklevel=2)
    mode = mode if mode is not None else 'r'

    if isinstance(source, MMPSource):
        src = source if source.mode == mode else MMPSource(location=source.location, mode=mode)
    elif isinstance(source, np.memmap):
        src = MMPSource(location=source.filename, mode=mode)
    elif isinstance(source, np.ndarray):
        src = NumpySource(array=source)
    elif isinstance(source, str):  # TOOD: early raise ?
        try:
            src = MMPSource(location=source, mode=mode)
        except FileNotFoundError as err:
            raise ClearMapFileNotFoundError(f'Memmap file not found: {source!r}') from err
        except Exception as err:
            raise ClearMapValueError(f'Cannot read memmap from location {source!r}!') from err  # FIXME: specific
    else:
        raise ValueError(f'Cannot read memmap from {source=!r}!')

    return src if slicing is None else NumpySource(array=(src.__getitem__(slicing)))


def edit(source, **kwargs):
    """Open an existing memmap for in-place modification."""
    if isinstance(source, MMPSource):
        if source.mode == 'r+':
            return source
        location = source.location
    elif isinstance(source, np.memmap):
        location=source.filename
    elif isinstance(source, str):
        location=source
    else:
        raise ValueError(f'Cannot edit {source!r} as memmap')

    return MMPSource(location=location, mode='r+')


def open_ro(source, **kwargs):
    """Open a source strictly read-only for metadata queries."""
    if isinstance(source, MMPSource):
        if source.mode == 'r':
            return source
        return MMPSource(location=source.location, mode='r')
    elif isinstance(source, np.memmap):
        return MMPSource(location=source.filename, mode='r')
    elif isinstance(source, np.ndarray):
        return NumpySource(array=source)  # already in memory, inherently read-only-ish
    elif isinstance(source, str):
        return MMPSource(location=source, mode='r')
    else:
        raise ValueError(f'Cannot inspect {source!r} as memmap source')


def write(sink, data, slicing=None, overwrite=True, flush=None, **kwargs):
    """Write data to a memory map.

    Arguments
    ---------
    sink : str, memmap, or Source
        The sink to write the data to.
    data : array
        The data to write int the sink.
    slicing : slice specification or None
        Optional slice specification of an existing memmap to write to.
    overwrite : bool
        Whether an existing file may be replaced. For a sliced write this only applies
        when the requested shape/dtype/order disagree with the existing file, since the
        .npy header cannot be changed in place.
    flush : bool or None
        Whether to msync before returning. ``None`` (default) flushes whole-array writes,
        where the barrier is once per file and the pages are dirty anyway, and does not
        flush sliced writes, where a per-slice barrier serialises writeback. Pass
        ``flush=True`` at a block boundary instead.

    Returns
    -------
    sink : memmap or Source
        The sink that was written to. For a location sink this is the opened or created
        Source, not the input path.
    """
    _validate_write_request(sink, data, kwargs)
    array = _as_array(data)
    whole = slicing is None or slc.is_trivial(slicing)

    # ---- 1. location sinks ------------------------------------------------
    if isinstance(sink, (str, pathlib.Path)):
        path = pathlib.Path(sink)
        if whole:
            sink= _create_sink(path, array, overwrite, kwargs)
        else:  # Block write: full shape must come from args or existing file
            sink =_create_block_sink(path, array, slicing, overwrite, kwargs)

    # ---- 2. object sinks --------------------------------------------------
    if not isinstance(sink, (MMPSource, np.ndarray)):
        raise ClearMapValueError(f'Cannot write to sink of type {type(sink).__name__}!',
                                 value=sink, expected='a location, a Source or a numpy array')

    _assert_durable_sink(sink, context='MMP.write')  # if we don't persist to disk, it's not a "write"
    _assign(sink, array, slicing)

    if flush or (flush is None and whole):
        _flush_sink(sink)
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
    if kwargs:
        raise ClearMapValueError(f'Unexpected keyword arguments {sorted(kwargs)} for create().',
                                 value=sorted(kwargs), expected=None)
    if mode is not None and mode != 'w+':
        raise ClearMapValueError(f'create() only supports mode="w+", got {mode!r}. '
                                 f'Use read() to open existing files or initialize() for read-or-create behaviour.')
    # param validation happens in ctor
    source = MMPSource(location=location, shape=shape, dtype=dtype, order=order, array=array, mode='w+')
    return source if as_source else source.array


###############################################################################
### Helpers
###############################################################################

def _create_from_array(location, array, shape=None, dtype=None, order=None,
                       mode=None, slicing=None, context=''):
    """Create or edit a memmap at location, populate from array, return in desired mode."""
    if slicing is not None:
        # Editing an existing file at a specific slice
        if not isinstance(location, str) or not fu.is_file(location):
            raise ClearMapValueError(f'Cannot write slice into non-existent memmap at {location!r}!',
                                     value=location, expected='existing file path')
        memmap = _open_memmap(location, mode='r+', context=context)
        memmap.__setitem__(slicing, array)
        return _reopen_with_mode(location, memmap, mode)

    # Full write — resolve, validate, create, populate
    location, shape, dtype, order = _resolve_params(array, location, shape, dtype, order)

    if isinstance(location, str):
        if shape == array.shape:
            memmap = _open_memmap(location, mode='w+', shape=shape, dtype=dtype, order=order, context=context)
            memmap[:] = array
        else:
            raise ClearMapValueError(f'Shape mismatch to create memmap: '
                                     f'requested {shape!r}, array has {array.shape!r}.',
                                     value=array.shape, expected=shape)
    else:
        raise ClearMapIoException(f'Cannot create memmap without a location! '
                                  f'Got {location!r} (type={type(location).__name__})')

    return _reopen_with_mode(location, memmap, mode)

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
        if order is None:
            raise ClearMapValueError(f'order must be explicitly provided in w+ mode, got None')
        kwargs['fortran_order'] = order == 'F'

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
        mode = mode or 'r'
        return _open_memmap(location, mode=mode, context='reading existing file')
    except ClearMapPermissionError:  # Read fallback
        return _open_memmap(location, mode='r', context='reading existing file (fallback read-only)')
    # WARNING: catching ClearMapValueError catches np ValueError (after reraise) which swallows corrupt files here
    #    fine with us, recreate corrupt from previous run.
    except (ClearMapRuntimeError, ClearMapValueError, ClearMapFileNotFoundError):
        # ignore general runtime/value errors here and fall through to creation
        return None


def _is_exact_memmap_match(array, location, shape, dtype, order):
    """Check if an existing memmap exactly matches the requested target parameters."""
    return (properties_match(array, shape=shape, dtype=dtype, order=order) and
            fu.abspath(location) == fu.abspath(array.filename))


def _reopen_with_mode(location: str | pathlib.Path | None, memmap: np.memmap, mode: str | None) -> np.memmap:
    # reopen in requested mode if different from 'w+' or current mode
    """
    Reopen the memmap with the desired mode if it doesn't already match.
    i.e. if different from 'w+' or current mode
    """
    desired_mode = mode or memmap.mode
    if desired_mode == memmap.mode:  # Already open in the requested mode — nothing to do
        return memmap
    elif desired_mode == 'w+':  # Reopening as 'w+' would truncate the file we just wrote — refuse
        return memmap
    else:
        _close_memmap(memmap)  # flush + release before remapping
        return _open_memmap(location, mode=desired_mode, context=f'reopening as {desired_mode!r}')


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
    # reload(mmp)
    from ClearMap.IO.source.backends.mmp_backend import MMPSource

    m = MMPSource(location='test.npy', shape=4)
    print(m)

    m[:] = 5
    print(m)

    import ClearMap.IO.source.Slice as slc

    s = slc.Slice(source=m, slicing=slice(1,3))
    print(s)

    s[:] = 3
    print(s)
    print(m)
