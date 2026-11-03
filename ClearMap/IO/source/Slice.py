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
        return self.source.offset + sliced_offset(self.slicing, self.source.strides, self.source.shape)

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

    n_no_newaxis = len([s for s in slicing if not (s is np.newaxis)])

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

    Note
    ----
    An index array is turned into a slice only when that is exact without knowing
    the axis length: evenly spaced, strictly monotonic, and all of one sign. Any
    other array (repeated indices, mixed signs, irregular spacing) is returned
    unchanged, and the functions below then reject it as fancy indexing.
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
            as_slice = _indices_to_slice(s)
            if as_slice is not None:
                simple.append(as_slice)
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
        if not isinstance(s, (slice, numbers.Integral)) and not (s is Ellipsis or s is np.newaxis):
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


def sliced_ndim(slicing, ndim):
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
    Only integers, slices and new axes are accepted: arrays (fancy indexing)
    are no longer supported and raise IndexError, like anything else invalid.
    """
    d = 0
    for s in unpack_slicing(slicing, ndim):
        s = _standard_slice(s)

        if isinstance(s, int):
            continue
        elif isinstance(s, slice) or s is np.newaxis:  # np.newaxis is None. a slice keeps its axis, a new axis adds one
            d += 1
        else:
            raise IndexError(f'Invalid indexing object {s!r}')

    return d


def _iter_axes(slicing, ndim):
    """Yield (axis, s) for each standardized entry of the unpacked slicing.

    axis is the position being consumed in the source being sliced; a new axis
    consumes none of it, so it is yielded as (None, np.newaxis) rather than
    advancing axis. Shared by every function below that walks a slicing axis
    by axis and skips new axes (sliced_shape also needs to see them, to record
    the extra length-1 dimension they add).
    """
    axis = -1
    for s in unpack_slicing(slicing, ndim):
        s = _standard_slice(s)
        if s is np.newaxis:
            yield None, s
            continue
        axis += 1
        yield axis, s


def sliced_shape(slicing, shape):
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
    Only integers, slices and new axes are accepted: arrays (fancy indexing)
    are no longer supported and raise IndexError, like anything else invalid.
    """
    if shape is None:
        return None

    sliced = []
    for axis, s in _iter_axes(slicing, len(shape)):
        if axis is None:  # a new axis consumes no axis of the source
            sliced.append(1)
            continue

        if isinstance(s, int):
            if s >= shape[axis] or -s > shape[axis]:
                raise IndexError(f'Index out of range in dimension {axis:d}!')
        elif isinstance(s, slice):
            sliced.append(_slice_length(s, shape[axis]))
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


def sliced_order(slicing, order, shape):
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
    if order is None or shape is None:
        return None

    slicing = unpack_slicing(slicing, len(shape))

    if order == 'F':
        slicing = slicing[::-1]
        shape   = shape[::-1]

    # check order
    is_subslice = False
    is_initial = True
    for axis, s in _iter_axes(slicing, len(shape)):
        if axis is None:  # a new axis consumes no axis of the source and does not affect contiguity
            continue

        if isinstance(s, int):
            if s >= shape[axis] or -s > shape[axis]:
                raise IndexError(f'Index out of range in dimension {axis:d}!')
            size = 1
        elif isinstance(s, slice):
            if s == slice(None):
                is_initial = False
                is_subslice = True
                continue
            size = _slice_length(s, shape[axis])
            step = s.indices(shape[axis])[2]
            if size > 1 and step != 1:  # strided or reversed
                return None
        else:
            raise IndexError(f'Invalid indexing object {s!r}')

        result = _contiguity_step(size, shape[axis], is_initial, is_subslice)
        if result is None:
            return None
        is_initial, is_subslice = result

    return order


def sliced_offset(slicing, strides, shape=None):
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
    for axis, s in _iter_axes(slicing, len(strides)):
        if axis is None:
            continue

        if isinstance(s, int):
            if s < 0:
                if shape is None:
                    raise IndexError('Cannot determine offset without shape!')
                s = shape[axis] + s
                if s < 0:
                    raise IndexError(f'Index out of bounds in dimension {axis:d}!')
            offset += s * strides[axis]
        elif isinstance(s, slice):
            offset += _slice_first(s, shape, axis, 'offset') * strides[axis]
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


def sliced_start(slicing, shape):
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
        Start position of the slicing in the original source: along each axis, the
        position of the first element read (for a reversed slice, its high end).
    """
    start = []
    for axis, s in _iter_axes(slicing, len(shape)):
        if axis is None:
            continue

        if isinstance(s, int):
            if s < 0:
                s = shape[axis] + s
            if s < 0:
                raise IndexError(f'Index out of bounds in dimension {axis:d}!')
            start.append(s)
        elif isinstance(s, slice):
            start.append(_slice_first(s, shape, axis, 'start'))
        else:
            raise IndexError(f'Invalid indexing object {s!r}')

    return tuple(start)


def sliced_slicing(slicing_second, slicing_first, shape):
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
        The reduced slicing: source[slicing] selects the same elements as
        source[slicing_first][slicing_second].

    Note
    ----
    After simplification, both slicing_first and slicing_second may only contain
    integers, slices and new axes: an irregular array or mask that simplify_slicing
    cannot turn into an equivalent slice is invalid in either one.

    Each axis of slicing_first is resolved to a range of source positions, and the
    matching entry of slicing_second is applied to that range: Python's own range
    indexing does the composition, for any sign of step and with numpy's
    out-of-range rules for integers. The one case that cannot be reduced is an
    empty slice of a new axis made by slicing_first; it raises IndexError.
    """
    shape1 = shape
    slicing1 = simplify_slicing(slicing_first, len(shape1))

    shape2 = sliced_shape(slicing1, shape1)  # also validates slicing1, see Note
    slicing2 = simplify_slicing(slicing_second, len(shape2))

    def next_second_index(d2):
        """Advance to the next non-newaxis entry of slicing2, appending np.newaxis to
        the (enclosing) `slicing` output for every new axis skipped along the way.
        Returns the new d2 and that entry. d2 is a position in slicing2 only; the
        length an entry is resolved against comes from slicing1, never shape2[d2].
        """
        d2 += 1
        s2 = _standard_slice(slicing2[d2])
        while s2 is np.newaxis:
            slicing.append(np.newaxis)
            d2 += 1
            s2 = _standard_slice(slicing2[d2])
        return d2, s2

    def resolve_second(selected, s2, d2):
        """Apply second index s2 to `selected`, the range of source positions one axis
        of slicing1 selected. An int gives one position, a slice a sub-range."""
        if isinstance(s2, int):
            try:
                return selected[s2]
            except IndexError:
                raise IndexError(f'Index {s2:d} in second slicing out of range in dimension {d2:d}!') from None
        elif isinstance(s2, slice):
            return selected[s2]
        raise IndexError(f'The index at dimension {d2:d} in second slicing is invalid!')

    d1 = -1
    d2 = -1
    slicing = []

    for s1 in slicing1:
        s1 = _standard_slice(s1)

        if isinstance(s1, int):
            d1 += 1
            if s1 > shape1[d1] or -s1 > shape1[d1]:
                raise ValueError(f'Integer index {s1:d} out of range in dimension {d1:d}!')
            slicing.append(s1)
        elif isinstance(s1, slice):
            d1 += 1
            d2, s2 = next_second_index(d2)
            selected = resolve_second(range(*s1.indices(shape1[d1])), s2, d2)
            if isinstance(selected, range):
                slicing.append(_range_to_slice(selected, shape1[d1]))
            else:
                slicing.append(selected)
        elif s1 is np.newaxis: # slicing1 made a length-1 axis here that the source does not have
            d2, s2 = next_second_index(d2)
            selected = resolve_second(range(1), s2, d2)
            if isinstance(selected, range):
                if len(selected) == 0:
                    raise IndexError(f'Empty slice of a new axis cannot be reduced in dimension {d2}!')
                slicing.append(np.newaxis)
            # an int (0 or -1) takes the new axis away again: nothing to append
        else:
            raise IndexError(f'The index at dimension {d1} in first slicing is invalid!')

    # anything left in slicing2 must be trailing new axes
    for s2 in slicing2[d2 + 1:]:
        if s2 is not np.newaxis:
            raise IndexError(f'The index at dimension {d2} in second slicing is invalid!')
        slicing.append(np.newaxis)

    return tuple(slicing)


def sliced_reduction(slicing, ndim):
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
    reduction = []
    for d, s in _iter_axes(slicing, ndim):
        if d is None:
            continue

        if isinstance(s, int):
            continue
        elif isinstance(s, slice):
            reduction.append(d)
        else:
            raise IndexError(f'Invalid indexing object {s!r}')

    return reduction


###############################################################################
### Helpers
###############################################################################

def _slice_length(s, axis_length):
    """Number of elements slice s selects from an axis of the given length (0 if empty, any step)."""
    return len(range(*s.indices(axis_length)))


def _clip_range_val(value, low, high):
    """value as a slice bound, or None if it is at or beyond either edge of the axis.

    A start or stop at the edge a slice defaults to, or past it, selects the same
    elements as None, so None is the shorter spelling. low and high are those
    edges for the slice's direction:

    * positive step: (0, axis_length) -- runs from 0 up to axis_length;
    * negative step: (-1, axis_length - 1) -- runs from axis_length - 1 down to
      past 0. Here the edge value -1 must never be written as a bound: a slice
      reads -1 as "the last element", which is why it becomes None.
    """
    return None if value <= low or value >= high else value


def _range_to_slice(r, axis_length):
    """The shortest slice that selects exactly the elements of range r from an axis of the given length.

    r comes from slicing a range(*s.indices(axis_length)), so its elements are
    valid, non-negative positions. Its own stop is not reused: stepping backwards
    past index 0 gives a negative stop, which a slice would read as counting from
    the end. The stop is rebuilt one step past the last element instead, and set
    to None where that falls off either end of the axis.
    """
    if len(r) == 0:
        return slice(0, 0)
    step = r.step
    if step > 0:
        low, high, stop = 0, axis_length, r[-1] + 1
    else:
        low, high, stop = -1, axis_length - 1, r[-1] - 1
    return slice(_clip_range_val(r[0], low, high),
                 _clip_range_val(stop, low, high),
                 None if step == 1 else step)


def _slice_first(s, shape, axis, what):
    """Position along the axis of the first element slice s reads: where numpy's view of it starts.

    For a negative step that is the high end of the range, not s.start, and it can
    only be found from the axis length; without a shape it raises rather than guess.
    """
    if shape is not None:
        return max(s.indices(shape[axis])[0], 0)  # max: a reversed slice of an empty axis gives -1
    if s.step is not None and s.step < 0:
        raise IndexError(f'Cannot determine {what} of a reversed slice without shape!')
    first = s.start or 0
    if first < 0:  # FIXME: improve error message.
        raise IndexError(f'Cannot determine {what} without shape!')
    return first


def _indices_to_slice(indices):
    """A slice selecting exactly the given 1-d integer indices, in order, or None if there is none.

    Only exact without knowing the axis length when the indices are evenly spaced
    with a non-zero step and all of one sign (all >= 0, or all < 0 i.e. all counted
    from the end). The stop is placed one step past the last index; if that crosses
    zero it would change meaning (e.g. -1 + 1 == 0 is the first element, not "past
    the end"), so it becomes None there.
    """
    if indices.ndim != 1 or len(indices) == 0:
        return None
    first, last = int(indices[0]), int(indices[-1])
    if len(indices) == 1:
        step = 1
    else:
        steps = np.unique(np.diff(indices))
        if len(steps) != 1 or steps[0] == 0:
            return None
        step = int(steps[0])
    if not (np.all(indices >= 0) or np.all(indices < 0)):
        return None
    stop = last + (1 if step > 0 else -1)
    if (last >= 0) != (stop >= 0):  # stepped across 0: past the end of the axis
        stop = None
    return slice(first, stop, None if step == 1 else step)


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
                        info += f'{r:d}'
                    info += ':'
                if s.step is None:
                    info = info[:-1]
                info = info[:-1]
        else:
            info += f'{s!r}'
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

    # NOTE: allow_index_arrays has been removed; [1,2,3,4,5] below is a regular run so
    # simplify_slicing still turns it into a slice, but an irregular array like the old
    # [0,2,1] example is no longer valid input anywhere in this module.
    s1 = (slice(1,4), [1,2,3,4,5], None, Ellipsis)
    ss = slc.simplify_slicing(s1, ndim = 5)
    print(ss)

    shape = (7,6,2,3,5)

    d1 = slc.sliced_ndim(s1, 5)
    shape1 = slc.sliced_shape(s1, shape)
    print(d1, shape1)

    x = np.random.rand(*shape)
    x1 = x[s1]
    x1.shape == shape1


    s2 = (slice(None, None, 2), slice(3,4), slice(None), 1, slice(0, 3, 2))
    s12 = slc.sliced_slicing(s2, s1, shape)

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
