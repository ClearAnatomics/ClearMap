# -*- coding: utf-8 -*-
"""
Slice
=====

Virtual slices of array sources.

A :class:`Slice` pairs an array source with a slice specification. It reports
the shape, dtype, order, strides and offset of the sliced data without reading
it, and reads or writes through to the underlying source only on item access.

Slices are cheap to send to worker processes: :meth:`Slice.as_virtual` keeps the
slicing and virtualizes the source. This makes them the handles that
:mod:`~ClearMap.ParallelProcessing.Block` and
:mod:`~ClearMap.ParallelProcessing.BlockProcessing` pass around.

Only array sources can be sliced this way. Tables and graphs are selected with
``io_ops.read(source, slicing=...)``.

The module also provides the slicing arithmetic used throughout ClearMap:
:func:`unpack_slicing`, :func:`sliced_shape`, :func:`sliced_order`,
:func:`sliced_slicing` and related helpers.

Example
-------
>>> import numpy as np
>>> from ClearMap.IO import dispatch
>>> from ClearMap.IO.source.Slice import Slice
>>> source = dispatch.as_source(np.random.rand(30, 40))
>>> sliced = Slice(source, slicing=(slice(None), slice(10, 20)))
>>> sliced.shape
(30, 10)
>>> sliced.base is source
True

The same slice can be made in one step with
``dispatch.as_source(array, slicing=(slice(None), slice(10, 20)))``.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'https://idisco.info'
__download__  = 'https://www.github.com/ChristophKirst/ClearMap2'


import numbers
import numpy as np

import ClearMap.IO.source.Source as source_mod


class Slice(source_mod.BaseArraySource):
    """A virtual slice of a source."""

    def __init__(self, source=None, slicing=None, name=None):
        """Slice class construtor.

        Arguments
        ---------
        source : class
            The underlying source of this slice.
        slicing : int, slice, list of slices or None
            The slice specification.
        """
        super(Slice, self).__init__(name=name)

        self._source = source
        self._slicing = slicing

    @property
    def name(self):
        return f'{type(self).__name__}[{self.source.name}]'

    @property
    def mode(self):
        """The mode of the underlying source
        Delegate since a slice has no mode of its own."""
        return self.source.mode

    @property
    def data_model(self):
        """A slice has the data model of the source it slices."""
        return self.source.data_model

    @property
    def source(self):
        """The source of this sliced source.

        Returns
        -------
        source : Source class
           The source of this slice.
        """
        return self._source

    @source.setter
    def source(self, source):
        self.__init__(source = source, slicing = self.slicing)

    @property
    def slicing(self):
        """The slice spcification of the source.

        Returns
        -------
        slicing : slice, list or None
            Returns the slice specification of this sliced source.
        """
        return self._slicing

    @slicing.setter
    def slicing(self, value):
        self.__init__(source = self.source, slicing = value)

    @property
    def shape(self):
        """The shape of the source.

        Returns
        -------
        shape : tuple
           The shape of the source.
        """
        return sliced_shape(self.slicing, self.source.shape)

    @property
    def dtype(self):
        """The data type of the source.

        Returns
        -------
        dtype : dtype
           The data type of the source.
        """
        return self.source.dtype

    @property
    def order(self):
        """The order of how the data is stored in the source.

        Returns
        -------
        order : str
           Returns 'C' for C contigous and 'F' for fortran contigous, None otherwise.
        """
        return sliced_order(self.slicing, self.source.order, self.source.shape)

    @property
    def strides(self):
        """The strides of the data source.

        Returns
        -------
        strides : tuple of ints
            The strides of the sliced source.
        """
        return sliced_strides(self.slicing, self.source.strides)

    @property
    def element_strides(self):
        """The strides of the data source.

        Returns
        -------
        strides : tuple of ints
           The strides of the sliced source.
        """
        return sliced_strides(self.slicing, self.source.element_strides)

    @property
    def offset(self):
        """The offset of the memory map in the file.

        Returns
        -------
        offset : int
           Offset of the memeory map in the file.
        """
        return self.source.offset + sliced_offset(self.slicing, self.source.strides)

    @property
    def location(self):
        """The location of the source's data.

        Returns
        -------
        location : str
            Returns the location of the data source or None if there is none.
        """
        return self.source.location

    @property
    def unpacked_slicing(self):
        """The unpacked slicing specification.

        Returns
        -------
        slicing : tuple
            Returns the unpacked slice spcification of this sliced source.
        """
        return unpack_slicing(self.slicing, self.source.ndim)

    @property
    def base_shape(self):
        """The shape of the underlying base source in case of nested slicing.

        Returns
        -------
        shape : tuple
            Returns the shape for the base of this source.
        """
        if isinstance(self.source, Slice):
            return self.source.base_shape
        else:
            return self.source.shape

    @property
    def base_slicing(self):
        """The direct slicing specification of the underlying base source in case of nested slicing.

        Returns
        -------
        slicing : tuple
            Returns the slice specification for the base of this source.
        """
        if isinstance(self.source, Slice):
            return sliced_slicing(self.slicing, self.source.base_slicing, self.base_shape)
        else:
            return self.slicing

    @property
    def base(self):
        """The underlying base source in case of nested slicing.

        Returns
        -------
        source : Source
            Returns the underlying base source of this sliced source.
        """
        if isinstance(self.source, Slice):
            return self.source.base
        else:
            return self.source

    @property
    def position(self):
        """Returns the indices of the lower corner of this slice in the underlying source.

        Returns
        -------
        position : tuple of int
            The coordinates of the lower corner of this slice within the source.
        """
        return tuple(sl.indices(s)[0] for sl,s in zip(self.slicing, self.source.shape))

    @property
    def base_position(self):
        """Returns the indices of the lower corner of this slice in the underlying base source.

        Returns
        -------
        position : tuple of int
           The coordinates of the lower corner of this slice within the base source.
        """
        return tuple(sl.indices(s)[0] for sl,s in zip(self.base_slicing, self.base.shape))

    @property
    def lower(self):
        """Returns the indices of the lower corner of this slice in the underlying source.

        Returns
        -------
        position : tuple of int
            The coordinates of the lower corner of this slice within the source.
        """
        return self.position

    @property
    def base_lower(self):
        """Returns the indices of the lower corner of this slice in the underlying base source.

        Returns
        -------
        position : tuple of int
            The coordinates of the lower corner of this slice within the base source.
        """
        return self.base_position

    @property
    def upper(self):
        """Returns the indices of the upper corner of this slice in the underlying source.

        Returns
        -------
        position : tuple of int
            The coordinates of the upper corner of this slice within the source.
        """
        return tuple(sl.indices(s)[1] for sl,s in zip(self.slicing, self.source.shape))

    @property
    def base_upper(self):
        """Returns the indices of the upper corner of this slice in the underlying base source.

        Returns
        -------
        position : tuple of int
            The coordinates of the upper corner of this slice within the base source.
        """
        return tuple(sl.indices(s)[1] for sl,s in zip(self.base_slicing, self.base.shape))

    @property
    def array(self):
        """The sliced array.

        Returns
        -------
        array : array
            An array representing the slice, if the underling source has an slicable array property, else None.
        """
        return self.source.__getitem__(self.slicing)

    def __getitem__(self, slicing):
        slicing = sliced_slicing(slicing, self.slicing, self.source.shape)
        return self.source.__getitem__(slicing)

    def __setitem__(self, slicing, data):
        slicing = sliced_slicing(slicing, self.slicing, self.source.shape)
        self.source.__setitem__(slicing, data)

    def as_virtual(self):
        """Returns a virtual handle to this source striped of any big array data useful for parallel processing.

        Returns
        -------
        source : Source class
            The source class with out any cached array data.

        Note
        ----
        The slicing structure is kept here to be able to appropiately take slices in a source when processsing in parallel.
        """
        return Slice(source=self.source.as_virtual(), slicing=self.slicing)

    def as_real(self):
        return Slice(source=self.source.as_real(), slicing=self.slicing)

    def as_buffer(self):
        return self.array


###############################################################################
### Functionality
###############################################################################

allow_index_arrays = False
"""Default value to allow index arrays in slicing.

Note
----
If True, allows indexing to also be integer or boolean arrays. 
In numpy this type of indexing triggers copying of the array and the 
sliced array is not a view into the original array.
"""

def slice_to_range(slicing, shape = None):
    """Transforms a slice object to a range.

    Arguments
    ---------
    slicing : slice
        A sinlge slice class.
    shape : tupe of ints or None
        The shape of the source, if None, try to determine from slice.

    Returns
    -------
    range : range
       The range corrresponding to the slice.
    """
    if not isinstance(slicing, slice):
        raise ValueError('A slice is expected!')

    if shape is not None:
        return np.arange(*slicing.indices(shape))
    elif slicing.stop is not None:
        return np.arange(*slicing.indices(slicing.stop))
    else:
        raise ValueError(f'No way to determine the range from the slice {slicing!r} without a source shape!')


def unpack_slicing(slicing, ndim):
    """Convert slice specification to a slice specification that matches the dimension of the sliced array.

    Arguments
    ---------
    slicing : object
        The slice specification.
    ndim : int
        The dimension of the source to slice.

    Returns
    -------
    slicing : object
        The full slice specification.
    """
    if not isinstance(slicing, tuple):
        slicing = (slicing,)
    slicing = list(slicing)

    n_no_newaxis = len([s for s in slicing if not (s is np.newaxis or s is None)])

    is_ellipsis = [s is Ellipsis for s in slicing]
    n_ellipsis = np.sum(is_ellipsis)
    if n_ellipsis > 1:
        raise IndexError(f'Only a single ellipsis allowed in a slice specification, found {np.sum(is_ellipsis):d}!')
    elif n_ellipsis == 1 and n_no_newaxis - 1 >= ndim:
        slicing.pop(is_ellipsis.index(True))
        n_no_newaxis -= 1

    if n_no_newaxis > ndim:
        raise IndexError(f'Slice specification has more dimensions {n_no_newaxis:d} than array {ndim:d}.')

    left  = []
    right = []
    take_from_left = True
    while slicing:
        if take_from_left:
            next_s = slicing.pop(0)
            list_s = left
        else:
            next_s = slicing.pop(-1)
            list_s = right

        if next_s is Ellipsis:
            next_s = slice(None)
            take_from_left = not take_from_left

        list_s.append(next_s)

    middle = [slice(None)] * (ndim - n_no_newaxis)

    return tuple(left + middle + right[::-1])


def simplify_slicing(slicing, ndim = None):
    """Simplifies slice specification to avoid fancy indexing if possible.

    Arguments
    ---------
    slicing : object
        The slice specification.
    ndim : int
        The dimension of the source to slice.

    Returns
    -------
    slicing : object
        The full slice specification.
    """
    if not isinstance(slicing, tuple):
        slicing = (slicing,)

    if ndim is not None:
        slicing = unpack_slicing(slicing, ndim)

    simple = []
    for s in slicing:
        s = _standard_slice(s)

        if isinstance(s, np.ndarray):
            if s.dtype == bool:
                s = np.where(s)[0]
            if len(s) == 0:
                simple.append(slice(0, 0))
                continue
            elif len(s) == 1 :
                simple.append(slice(s[0], s[0]+1))
                continue
            else:
                step = np.unique(np.diff(s))
                if len(step) == 1:
                    simple.append(slice(s[0], s[-1] + 1, step[0]))
                    continue

        simple.append(s)

    return tuple(simple)


def is_view(slicing):
    """Returns True if the slicing results in a view of the original array.

    Arguments
    ---------
    slicing : object
        The slice specification.

    Returns
    -------
    is_view : bool
        True if the sliced array is a view.
    """
    for s in slicing:
        if not isinstance(s, (slice, numbers.Integral)) and not (s is Ellipsis or s is None or s is np.newaxis):
            return False
    return True


def is_trivial(slicing):
    """Returns True if the slicing is not generating a real sub-slice.

    Arguments
    ---------
    slicing : object
        The slice specification.

    Returns
    -------
    is_trivial : bool
        True if the sliced array is changed form the original one.
    """
    if slicing is None:
        return True
    for s in slicing:
        if s is Ellipsis:
            continue
        elif isinstance(s, slice) and s.start is None and s.stop is None and s.step is None:
            continue
        else:
            return False
    return True


class _ArrayBudget:
    """How many axes may still use fancy indexing (a 1-d boolean mask or 1-d integer array).

    At most one such axis in total, of either kind, and none at all unless
    allow_index_arrays is set. This is a single shared counter, not one per
    kind: combining a mask on one axis with an index array on another axis
    requires numpy to broadcast the two together into one shared output axis
    (or raise, if their lengths disagree) -- something sliced_shape and its
    siblings do not compute. A separate per-kind budget would silently let
    that combination through and report a wrong shape/offset/start for it, so
    a second fancy axis is refused regardless of what kind it is.

    Note
    ----
    BUG FIX relative to the original module: the original tracked a bool-array
    count and an int-array count separately, each capped at 1, which let one of
    each through simultaneously -- a combination the module's own arithmetic
    computes incorrectly (see the class docstring above). This is the last
    commit meant to carry allow_index_arrays support at all; see the module
    docstring/changelog.
    """
    def __init__(self, allow_index_arrays):
        self._max = 1 if allow_index_arrays else 0
        self._count = 0

    def check_ndim(self, s, d):
        if s.ndim != 1:
            raise IndexError(f'Fancy slicing in dimension {d} not supported!')

    def check_limit(self, s, d):
        self._count += 1
        if self._count > self._max:
            name = 'Boolean' if s.dtype == bool else 'Index'
            raise IndexError(f'{name} array slicing in dimension {d} not supported!')


def sliced_ndim(slicing, ndim, allow_index_arrays=allow_index_arrays):
    """Returns the dimension of a slicing of an array with given dimension.

    Arguments
    ---------
    slicing : object
        Slice specification.
    ndim : int
       Dimension of the array.

    Returns
    -------
    ndim : int
        The dimension of the sliced array.

    Note
    ----
    Like :func:`sliced_shape`, only integers, slices and new axes are accepted:
    anything else, including boolean or integer arrays, raises IndexError.
    """
    budget = _ArrayBudget(allow_index_arrays)
    d = 0
    for s in unpack_slicing(slicing, ndim):
        s = _standard_slice(s)

        if isinstance(s, int):
            continue
        elif isinstance(s, slice) or s is np.newaxis:  # np.newaxis is None. a slice keeps its axis, a new axis adds one
            d += 1
        elif isinstance(s, np.ndarray):
            budget.check_ndim(s, d)
            budget.check_limit(s, d)
            d += 1
        else:
            raise IndexError(f'Invalid indexing object {s!r}')

    return d


def sliced_shape(slicing, shape, allow_index_arrays=allow_index_arrays):
    """Returns the shape that results from slicing.

    Arguments
    ---------
    slicing : object
       Slice specification.
    shape : tuple
       Shape of the original array.

    Returns
    -------
    shape : tuple
       The shape of the sliced array.

    Note
    ----
    Only integers, slices and new axes are accepted: anything else, including
    boolean or integer arrays (fancy indexing), raises IndexError.
    """
    if shape is None:
        return None

    sliced = []
    budget = _ArrayBudget(allow_index_arrays)
    axis = -1
    for s in unpack_slicing(slicing, len(shape)):
        s = _standard_slice(s)
        if s is np.newaxis:  # np.new_axis is None. A new axis consumes no axis of the source
            sliced.append(1)
            continue
        axis += 1

        if isinstance(s, int):
            if s >= shape[axis] or -s > shape[axis]:
                raise IndexError(f'Index out of range in dimension {axis:d}!')
        elif isinstance(s, slice):
            start, stop, step = s.indices(shape[axis])
            sliced.append((stop - start - 1) // step + 1)
        elif isinstance(s, np.ndarray):
            budget.check_ndim(s, axis)
            if s.dtype == bool:
                if len(s) != shape[axis]:
                    raise IndexError(f'The boolean indexing has different shape {len(s)} than the source {shape[axis]} in dimension {axis}!')
            else:
                if np.any(s >= shape[axis]) or np.any(-s > shape[axis]):
                    raise IndexError(f'Index out of range in dimension {axis}!')
            budget.check_limit(s, axis)
            sliced.append(np.sum(s) if s.dtype == bool else len(s))
        else:
            raise IndexError(f'Invalid indexing object {s!r}')

    return tuple(sliced)


def _contiguity_step(size, shape_d, is_initial, is_subslice):
    """Advance the is_initial/is_subslice state machine by one axis of the given size.

    The same 3-way branch (size == 1 / size == the full axis / anything else) that
    sliced_order ran once for a scalar int, once for a slice, and once each for a
    bool array and an int array. Returns the updated (is_initial, is_subslice), or
    None to signal "order broken, sliced_order should return None".
    """
    if size == 1:
        if is_initial:
            return is_initial, is_subslice
        if is_subslice:
            return None
        return is_initial, True
    if size == shape_d:
        return False, True
    if is_subslice:
        return None
    return False, True


def sliced_order(slicing, order, shape,  allow_index_arrays =allow_index_arrays):
    """Returns the contiguous order of a sliced array.

    Arguments
    ---------
    slicing : object
        The slice specification.
    order : 'C', 'F' or None
       The order of the source to be sliced.
    shape : tuple of ints
        The shape of the source to be sliced.

    Returns
    -------
    order : 'C', 'F' or None
        The order of the sliced source.
    """
    if order is None:
        return None

    if shape is None:
        return None

    slicing = unpack_slicing(slicing, len(shape))

    if order == 'F':
        slicing = slicing[::-1]
        shape   = shape[::-1]

    # check order
    budget = _ArrayBudget(allow_index_arrays)
    is_subslice = False
    is_initial = True
    d = -1
    for s in slicing:
        d += 1
        s = _standard_slice(s)

        if isinstance(s, int):
            if s >= shape[d] or -s > shape[d]:
                raise IndexError(f'Index out of range in dimension {d:d}!')
            size = 1
        elif isinstance(s, slice):
            if s == slice(None):
                is_initial = False
                is_subslice = True
                continue
            start, stop, step = s.indices(shape[d])
            size = (stop - start - 1) // step + 1
            if size != 1 and step > 1:
                return None
        elif isinstance(s, np.ndarray) and s.dtype == bool:
            budget.check_ndim(s, d)
            if len(s) != shape[d]:
                raise IndexError(f'The boolean indexing has different shape {len(s)} than the source {shape[d]} in dimension {d}!')
            budget.check_limit(s, d)
            size = np.sum(s)
        elif isinstance(s, np.ndarray) and s.dtype == int:
            budget.check_ndim(s, d)
            if np.any(s >= shape[d]) or np.any(-s > shape[d]):
                raise IndexError(f'Index out of range in dimension {d}!')
            budget.check_limit(s, d)
            size = len(s)
        elif s is np.newaxis or s is None:
            continue
        else:
            raise IndexError(f'Invalid indexing object {s!r}')

        result = _contiguity_step(size, shape[d], is_initial, is_subslice)
        if result is None:
            return None
        is_initial, is_subslice = result

    return order


def sliced_offset(slicing, strides, shape=None, allow_index_arrays=allow_index_arrays):
    """Returns the offset to the first element of the slicing into a buffer with given strides.

    Arguments
    ---------
    slicing : object
        Slice specification.
    strides : tuple
        Strides of the array.

    Returns
    -------
    offset : int
        Offset into the sliced array.
    """
    offset = 0
    budget = _ArrayBudget(allow_index_arrays)
    axis = -1
    for s in unpack_slicing(slicing, len(strides)):
        s = _standard_slice(s)
        if s is np.newaxis:
            continue
        axis += 1

        if isinstance(s, int):
            if s < 0:
                if shape is None:
                    raise IndexError('Cannot determine offset without shape!')
                s = shape[axis] + s
                if s < 0:
                    raise IndexError(f'Index out of bounds in dimension {axis:d}!')
            offset += s * strides[axis]
        elif isinstance(s, slice):
            start = s.start or 0
            if start < 0:
                if shape is None:
                    raise IndexError('Cannot determine offset without shape!')
                start = shape[axis] + start
                start = 0 if start < 0 else start
            offset += start * strides[axis]
        elif isinstance(s, np.ndarray):
            budget.check_ndim(s, axis)
            budget.check_limit(s, axis)
            if s.dtype == bool:
                first = np.where(s)[0]
                if len(first) == 0:
                    raise IndexError(f'There is not True value in boolean array slicing in dimension {axis}!')
                first = first[0]
            else:
                if len(s) == 0:
                    raise IndexError(f'There is no index in array slicing in dimension {axis}!')
                first = s[0]
                if first < 0:
                    if shape is None:
                        raise IndexError('Cannot determine offset without shape!')
                    first = shape[axis] + first
                    if first < 0:
                        raise IndexError(f'Index out of bounds in dimension {axis}!')
            offset += first * strides[axis]
        else:
            raise IndexError(f'Invalid indexing object {s!r}')

    return offset


def sliced_strides(slicing, strides):
    """Returns the strides of the slicing of a buffer with given strides if possible.

    Arguments
    ---------
    slicing : object
        Slice specification.
    strides : tuple
        Strides of the original array.

    Returns
    -------
    strides : tuple
        Strides into the sliced array.
    """
    sliced = []
    d = -1
    for s in simplify_slicing(slicing, len(strides)):
        s = _standard_slice(s)
        if s is np.newaxis:  # a new axis consumes no axis of the source and has stride 0
            sliced.append(0)
            continue
        d += 1
        if isinstance(s, int):
            pass
        elif isinstance(s, slice):
            sliced.append(strides[d] * (s.step or 1))
        elif isinstance(s, np.ndarray):
            raise ValueError('Fancy slicing does not result in valid strides!')
        else:
            raise IndexError(f'Invalid indexing object {s!r}')

    return tuple(sliced)


def sliced_start(slicing, shape, allow_index_arrays=allow_index_arrays):
    """Returns the starting position of the slicing in the original source.

    Arguments
    ---------
    slicing : object
       Slice specification.
    shape : tuple
        Shape of the array.

    Returns
    -------
    start : tuple of int
        Start position of the slicing in the original source.
    """
    start = []
    budget = _ArrayBudget(allow_index_arrays)
    axis = -1
    for s in unpack_slicing(slicing, len(shape)):
        s = _standard_slice(s)
        if s is np.newaxis:
            continue
        axis += 1

        if isinstance(s, int):
            if s < 0:
                s = shape[axis] + s
            if s < 0:
                raise IndexError(f'Index out of bounds in dimension {s:d}!')  # FIXME: should be {axis}, not {s}
            start.append(s)
        elif isinstance(s, slice):
            first = s.start or 0
            if first < 0:
                first = shape[axis] + first
                first = 0 if first < 0 else first
            start.append(first)
        elif isinstance(s, np.ndarray):
            budget.check_ndim(s, axis)
            budget.check_limit(s, axis)
            if s.dtype == bool:
                start = np.where(s)[0]  # FIXME: (kept): shadows the accumulator `start`
                if len(start) == 0:
                    raise IndexError(f'There is not True value in boolean array slicing in dimension {axis}!')
                start.append(start[0])  # FIXME: ndarray has no .append -> always raises AttributeError here
            else:
                if len(s) == 0:
                    raise IndexError(f'There is no index in array slicing in dimension {axis}!')
                first = s[0]
                if first < 0:
                    first = shape[axis] + first
                    if first < 0:
                        raise IndexError(f'Index out of bounds in dimension {axis}')
                start.append(first)
        else:
            raise IndexError(f'Invalid indexing object {s!r}')

    return tuple(start)


def sliced_slicing(slicing_second, slicing_first, shape, allow_index_arrays = allow_index_arrays):
    """Returns a slicing of a slicing if possible.

    Arguments
    ---------
    slicing_second : object
        Slice specification followed by first slicing.
    slicing_first : object
        First slicing.
    shape : tuple of ints
        Shape of the original source to be sliced twice.

    Returns
    -------
    slicing : object
        The reduced slicing.

    Note
    ----
    After simplification, slicing_first may only contain integers, slices and new axes:
    sliced_shape rejects anything else. Boolean masks in slicing_second are turned into
    index arrays by simplify_slicing.
    """
    shape1 = shape
    slicing1 = simplify_slicing(slicing_first, len(shape1))

    shape2 = sliced_shape(slicing1, shape1)  # also validates slicing1, see Note
    slicing2 = simplify_slicing(slicing_second, len(shape2))

    d1 = -1
    d2 = -1
    slicing = []
    index_array1 = 0
    bool_array1 = 0
    index_array2 = 0
    bool_array2 = 0
    max_arrays = 1 if allow_index_arrays else 0

    for s1 in slicing1:
        s1 = _standard_slice(s1)

        if isinstance(s1, int):
            d1 += 1
            if s1 > shape1[d1] or -s1 > shape1[d1]:
                raise ValueError(f'Integer index {s1:d} out of range in dimension {d1:d}!')
            slicing.append(s1)
        elif isinstance(s1, slice):
            d1 += 1
            start1,stop1,step1 = s1.indices(shape1[d1])

            d2 += 1
            s2 = _standard_slice(slicing2[d2])

            while s2 is np.newaxis or s2 is None:
                d2 += 1
                s2 = _standard_slice(slicing2[d2])
                slicing.append(np.newaxis)

            if isinstance(s2, int):
                if s2 < 0:
                    s = stop1 + step1 * s2
                else:
                    s = start1 + step1 * s2
                if s < start1 or s > stop1 or s > shape1[d1]:
                    raise ValueError(f'Index {s:d} in second slicing out of range in dimension {d2:d}!')
                slicing.append(s)
            elif isinstance(s2, slice):
                start2, stop2, step2 = s2.indices(shape2[d2])
                start = start1 + start2
                stop = start1 +  stop2
                stop = min(stop1, stop)
                step = step1 * step2
                start = None if start == 0 else start
                stop = None if stop == shape1[d1] else stop
                step = None if step == 1 else step
                slicing.append(slice(start, stop, step))
            elif isinstance(s2, np.ndarray) and s2.dtype == bool:
                if s2.ndim != 1:
                    raise IndexError(f'Fancy second slicing in dimension {d2:d} not supported!')
                bool_array2 += 1
                if bool_array2 > max_arrays:
                    raise IndexError(f'Second boolean array slicing in dimension {d2:d} not supported!')
                if len(s2) != shape2[d2]:
                    raise IndexError(f'Second boolean array slicing with shape {len(s2):d} is not of shape {shape2[d2]:d} in dimension {d2:d}!')
                slicing.append(np.arange(start1, stop1, step1)[s2])
            elif isinstance(s2, np.ndarray) and s2.dtype == int:
                if s2.ndim != 1:
                    raise IndexError(f'Fancy second slicing in dimension {d2:d} not supported!')
                index_array2 += 1
                if index_array2 > max_arrays:
                    raise IndexError(f'Second index array slicing in dimension {d2:d} not supported!')
                try:
                    slicing.append(np.arange(start1, stop1, step1)[s2])
                except:
                    raise IndexError(f'Index out of range in second array slicing in dimension {d2:d}!')
            else:
                raise IndexError(f'The index at dimension {d2:d} in second slicing is invalid!')
        elif isinstance(s1, np.ndarray) and s1.dtype == bool:
            d1 += 1
            if s1.ndim != 1:
                raise IndexError(f'Fancy slicing in dimension {d1:d} not supported!')
            bool_array1 += 1
            if bool_array1 > max_arrays:
                raise IndexError(f'Boolean array slicing in dimension {d1:d} not supported!')
            if len(s1) != shape1[d1]:
                raise IndexError(f'Boolean array slicing with shape {len(s1):d} is not of shape {shape1[d1]:d} in dimension {d1:d}!')
            s1 = np.where(s1)[0]

            d2 += 1
            s2 = _standard_slice(slicing2[d2])

            while s2 is np.newaxis or s2 is None:
                d2 += 1
                s2 = _standard_slice(slicing2[d2])
                slicing.append(np.newaxis)

            if isinstance(s2, int):
                try:
                    s = s1[s2]
                except:
                    raise IndexError(f'Index {s:d} in second slicing out of range in dimension {d2:d}!')
                if s > shape1[d1] or -s > shape1[d1]:
                    raise ValueError(f'Index {s:d} in second slicing out of range in dimension {d2:d}!')
                slicing.append(s)
            elif isinstance(s2, slice):
                try:
                    slicing.append(s1[s2])
                except:
                    raise IndexError(f'Index out of range in second slicing in dimension {d2:d}!')
            elif isinstance(s2, np.ndarray) and s2.dtype == bool:
                if s2.ndim != 1:
                    raise IndexError(f'Fancy second slicing in dimension {d2:d} not supported!')
                bool_array2 += 1
                if bool_array2 > max_arrays:
                    raise IndexError(f'Second boolean array slicing in dimension {d2:d} not supported!')
                if len(s2) != shape2[d2]:
                    raise IndexError(f'Second boolean array slicing with shape {len(s2):d} is not an array'
                                     f' with shape {shape2[d2]:d} in dimension {d2:d}!')
                s1 = np.where(s1)[0]
                try:
                    slicing.append(s1[s2])
                except:
                    raise IndexError(f'Index out of range in second slicing in dimension {d2:d}!')
            elif isinstance(s2, np.ndarray) and s2.dtype == int:
                if s2.ndim != 1:
                    raise IndexError(f'Fancy second slicing in dimension {d2:d} not supported!')
                index_array2 += 1
                if index_array2 > max_arrays:
                    raise IndexError(f'Second index array slicing in dimension {d2:d} not supported!')
                if len(s2) == 0:
                    raise IndexError(f'There is no index in second array slicing in dimension {d2:d}!')
                try:
                    slicing.append(s1[s2])
                except:
                    raise IndexError(f'Index out of range in second slicing in dimension {d2:d}!')
            else:
                raise IndexError(f'The index at dimension {d2:d} in second slicing is invalid!')
        elif isinstance(s1, np.ndarray) and s1.dtype == int:
            d1 += 1
            if s1.ndim != 1:
                raise IndexError(f'Fancy slicing in dimension {d1:d} not supported!')
            index_array1 += 1
            if index_array1 > max_arrays:
                raise IndexError(f'Integer array slicing in dimension {d1:d} not supported!')
            if len(s1) == 0:
                raise IndexError(f'There is no index in array slicing in dimension {d1:d}!')

            d2 += 1
            s2 = _standard_slice(slicing2[d2])

            while s2 is np.newaxis or s2 is None:
                d2 += 1
                s2 = _standard_slice(slicing2[d2])
                slicing.append(np.newaxis)

            if isinstance(s2, int):
                try:
                    s = s1[s2]
                except:
                    raise IndexError('Index %d in second slicing out of range in dimension %d!' % (s, d2))
                if s > shape1[d1] or -s > shape1[d1]:
                    raise ValueError('Index %d in second slicing out of range in dimension %d!' % (s, d2))
                slicing.append(s)
            elif isinstance(s2, slice):
                try:
                    slicing.append(s1[s2])
                except:
                    raise IndexError('Index out of range in second slicing in dimension %d!' % d2)
            elif isinstance(s2, np.ndarray) and s2.dtype == bool:
                if s2.ndim != 1:
                    raise IndexError('Fancy second slicing in dimension %d not supported!' % d2)
                bool_array2 += 1
                if bool_array2 > max_arrays:
                    raise IndexError('Second boolean array slicing in dimension %d not supported!' % d2)
                if len(s2) != shape2[d2]:
                    raise IndexError('Second boolean array slicing with shape %d is not an array with shape %d in dimension %d!' % (len(s2), shape2[d2], d2))
                try:
                    slicing.append(s1[s2])
                except:
                    raise IndexError('Index out of range in second slicing in dimension %d!' % d2)
            elif isinstance(s2, np.ndarray) and s2.dtype == int:
                if s2.ndim != 1:
                    raise IndexError('Fancy second slicing in dimension %d not supported!' % d2)
                index_array2 += 1
                if index_array2 > max_arrays:
                    raise IndexError('Second index array slicing in dimension %d not supported!' % d2)
                if len(s2) == 0:
                    raise IndexError('There is no index in second array slicing in dimension %d!' % d2)
                try:
                    slicing.append(s1[s2])
                except:
                    raise IndexError('Index out of range in second slicing in dimension %d!' % d2)
            else:
                raise IndexError('The index at dimension %d in second slicing is invalid!' % d2)

        elif s1 is np.newaxis or s2 is None:
            d2 += 1
            s2 = _standard_slice(slicing2[d2])

            while s2 is np.newaxis or s2 is None:
                d2 += 1
                s2 = _standard_slice(slicing2[d2])
                slicing.append(np.newaxis)

            assert shape2[d2] == 1

            if isinstance(s2, int):
                if s2 not in [0,-1]:
                    raise IndexError('Index %d in second slicing out of range in dimension %d!' % (s, d2))
            elif isinstance(s2, slice):
                start, stop, step = s2.indices(1)
                if start == 1 or stop == 0:
                    raise IndexError('Empty slice of a new axis cannot be reduced in dimension %d!' % d2)
                else:
                    slicing.append(np.newaxis)
            elif isinstance(s2, np.ndarray) and s2.dtype == bool:
                if s2.ndim != 1:
                    raise IndexError('Fancy second slicing in dimension %d not supported!' % d2)
                bool_array2 += 1
                if bool_array2 > max_arrays:
                    raise IndexError('Second boolean array slicing in dimension %d not supported!' % d2)
                if len(s2) != shape2[d2]:
                    raise IndexError('Second boolean array slicing with shape %d is not of shape %d in dimension %d!' % (len(s2), shape2[d2], d2))
                if s2[0] == False:
                    raise IndexError('Empty slice of a new axis cannot be reduced in dimension %d!' % d2)
                else:
                    slicing.append(np.newaxis)
            elif isinstance(s2, np.ndarray) and s2.dtype == int:
                if s2.ndim != 1:
                    raise IndexError('Fancy second slicing in dimension %d not supported!' % d2)
                index_array2 += 1
                if index_array2 > max_arrays:
                    raise IndexError('Second index array slicing in dimension %d not supported!' % d2)
                if len(s2) != shape2[d2]:
                    raise IndexError('Second boolean array slicing with shape %d is not of shape %d in dimension %d!' % (len(s2), shape2[d2], d2))
                if s2[0] not in [0,-1]:
                    raise IndexError('Index %d in second slicing out of range in dimension %d!' % (s, d2))
                slicing.append(np.newaxis)
            else:
                raise IndexError('The index at dimension %d in second slicing is invalid!' % d2)
        else:
            raise IndexError(f'The index at dimension {d1:d} in first slicing is invalid!')

    d2 += 1
    while d2 < len(slicing2):
        if not (slicing2[d2] is np.newaxis or slicing2[d2] is None):
            raise IndexError(f'The index at dimension {d2:d} in second slicing is invalid!')
        else:
            slicing.append(np.newaxis)

    return tuple(slicing)


def sliced_reduction(slicing, ndim, allow_index_arrays = allow_index_arrays):
    """Returns a slicing that slices a list retaining only full dimensions in the slice.

    Arguments
    ---------
    slicing : object
        The slice specification.
    ndim : int
       The dinension of the source.

    Returns
    -------
    slicing : object
        Slice specification that reduces a list of length ndim to the new dimensions of the slice.
    """
    slicing = unpack_slicing(slicing, ndim)

    reduction = []
    index_array = 0
    bool_array = 0
    max_arrays = 1 if allow_index_arrays else 0
    d = -1
    for s in slicing:
        s = _standard_slice(s)
        if s is np.newaxis:
            continue
        d += 1

        if isinstance(s, int):
            continue
        elif isinstance(s, slice):
            reduction.append(d)
        elif isinstance(s, np.ndarray) and s.dtype == bool:
            if s.ndim != 1:
                raise IndexError(f'Fancy slicing in dimension {d:d} not supported!')
            bool_array += 1
            if bool_array > max_arrays:
                raise IndexError(f'Boolean array slicing in dimension {d:d} not supported!')
            reduction.append(d)
        elif isinstance(s, np.ndarray) and s.dtype == int:
            if s.ndim != 1:
                raise IndexError(f'Fancy slicing in dimension {d:d} not supported!')
            index_array += 1
            if index_array > max_arrays:
                raise IndexError(f'Index array slicing in dimension {d:d} not supported!')
            if len(s) == 0:
                raise IndexError(f'There is no index in array slicing in dimension {d:d}!')
            reduction.append(d)
        else:
            raise IndexError(f'Invalid indexing object {s!r}')

    return reduction


###############################################################################
### Helpers
###############################################################################

def _standard_slice(s):
    if s is Ellipsis:
        return s
    if isinstance(s, numbers.Integral):
        return int(s)
    if isinstance(s, slice):
        return s
    if isinstance(s, (list, tuple, np.ndarray)):
        s = np.asarray(s)
    else:
        try:
            iter(s)
        except:
            return s
        else:
            s = np.asarray(s)
    if not s.dtype in [bool, int]:
        s = np.asarray(s, dtype = int)
    return s


def _slicing_to_str(slicing, ndim):
    slicing = unpack_slicing(slicing, ndim)
    info = '('
    for s in slicing:
        s = _standard_slice(s)
        if s is Ellipsis:
            info += ':'
        elif isinstance(s, slice):
            if s.start is None and s.stop is None and s.step is None:
                info += ':'
            else:
                for r in [s.start, s.stop, s.step]:
                    if r is not None:
                        info += '%d' % r
                    info += ':'
                if s.step is None:
                    info = info[:-1]
                info = info[:-1]
        else:
            info += '%r' % s
        info += ','
    info = info[:-1] + ')'
    return info


###############################################################################
### Tests
###############################################################################

def _test():
    import numpy as np  # analysis:ignore
    import ClearMap.IO.source.Slice as slc
    from importlib import reload
    reload(slc)

    s1 = (slice(1,4), [1,2,3,4,5], None, Ellipsis)
    ss = slc.simplify_slicing(s1, ndim = 5)
    print(ss)

    shape = (7,6,2,3,5)

    d1 = slc.sliced_ndim(s1, 5, allow_index_arrays = True)
    shape1 = slc.sliced_shape(s1, shape, allow_index_arrays=True)
    print(d1, shape1)

    x = np.random.rand(*shape)
    x1 = x[s1]
    x1.shape == shape1


    s2 = (slice(None, None, 2), slice(3,4), slice(None), 1, [0,2,1])
    s12 = slc.sliced_slicing(s2, s1, shape, allow_index_arrays=True)

    np.all(x[s12] == x[s1][s2])

    slc.is_view(s1)

    s1s = slc.simplify_slicing(s1)
    slc.is_view(s1s)

    s2 = (slice(None, None, 2), slice(3,4), slice(None), 1, slice(0,2))

    y = x[s1s][s2]
    y.base is x

    s12 = slc.sliced_slicing(s2, s1s, shape)
    slc.is_view(s12)
    x[s12].base is x


    x = slc.src.VirtualSource(shape=(5,10,15), dtype=float, order='F', location='/home/test.src')

    s = slc.Slice(source = x, slicing = (Ellipsis, 1))
    print(s)
    print(s.source)
    print(s.unpacked_slicing)

    reload(slc)
    shape = (50,100,200)
    s1 = (slice(None), slice(None), slice(0, 38))
    s2 = (slice(None), slice(None), slice(None, -10))

    slc.sliced_slicing(s2,s1,shape)
