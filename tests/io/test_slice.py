"""Tests for ClearMap.IO.source.Slice.

Four groups:

* the behaviour BlockProcessing relies on, checked against numpy;
* pickling of virtual slices and blocks, the handles sent to worker processes;
* the slicing arithmetic on arbitrary slicings (negative steps, empty slices,
  new axes), checked against numpy;
* regressions: one test per bug fixed in the slicing arithmetic, and the cases
  that are refused on purpose because they have no exact answer.
"""
import pickle

import numpy as np
import pytest

import ClearMap.IO.source.Slice as slc
from ClearMap.IO.source.Source import ArraySource, VirtualSource
from ClearMap.ParallelProcessing import BlockProcessing as bp
from ClearMap.ParallelProcessing.Block import Block


###############################################################################
### Test sources
###############################################################################

class MemorySource(ArraySource):
    """An array source over an in-memory array."""

    def __init__(self, array):
        super().__init__()
        self._array = array

    @property
    def array(self):
        return self._array

    @property
    def strides(self):
        return self._array.strides


class NpySource(ArraySource):
    """A file-backed array source that refuses to be pickled, like a real memory map should."""

    def __init__(self, location, mode='r'):
        super().__init__(mode=mode)
        self.location = location
        self._array = np.load(location, mmap_mode=mode)

    @property
    def array(self):
        return self._array

    @property
    def strides(self):
        return self._array.strides

    def __getstate__(self):
        raise pickle.PicklingError('a real source crossed the process barrier')


class NpyVirtualSource(VirtualSource):
    _real_class = NpySource


NpySource._virtual_class = NpyVirtualSource


def _block_like_index(rng, n):
    """An index of the kind block processing produces: full, bounded or negative-stop slices, or an int."""
    r = rng.random()
    if r < 0.25:
        return slice(None)
    if r < 0.6:
        lo = int(rng.integers(0, n))
        return slice(lo, int(rng.integers(lo + 1, n + 1)))
    if r < 0.85:
        lo = int(rng.integers(0, n))
        return slice(lo or None, -int(rng.integers(1, n - lo)) if n - lo > 1 else None)
    return int(rng.integers(0, n))


def _block_like_slicing(rng, shape):
    return tuple(_block_like_index(rng, n) for n in shape)


def _random_array(rng, shape, order):
    return np.asarray(rng.random(shape), order=order)


###############################################################################
### Behaviour BlockProcessing relies on, against numpy
###############################################################################

@pytest.mark.parametrize('order', ['C', 'F'])
def test_slice_matches_numpy(order):
    rng = np.random.default_rng(0)
    for _ in range(300):
        shape = tuple(int(n) for n in rng.integers(2, 7, size=rng.integers(1, 4)))
        x = _random_array(rng, shape, order)
        s = _block_like_slicing(rng, shape)
        sliced = slc.Slice(MemorySource(x), s)
        expected = x[s]

        assert sliced.shape == expected.shape
        np.testing.assert_array_equal(sliced.array, expected)
        assert sliced.strides == expected.strides
        if isinstance(expected, np.ndarray) and expected.size:  # all-int indexing gives a scalar, not a view
            offset = expected.__array_interface__['data'][0] - x.__array_interface__['data'][0]
            assert sliced.offset == offset
        if sliced.order is not None:  # the prediction may be conservative, but must not be wrong
            assert expected.flags[f'{sliced.order}_CONTIGUOUS']


@pytest.mark.parametrize('order', ['C', 'F'])
def test_nested_slice_matches_numpy(order):
    rng = np.random.default_rng(1)
    for _ in range(300):
        shape = tuple(int(n) for n in rng.integers(3, 8, size=rng.integers(1, 4)))
        x = _random_array(rng, shape, order)
        s1 = tuple(i if isinstance(i, slice) else slice(i, i + 1) for i in _block_like_slicing(rng, shape))
        s2 = _block_like_slicing(rng, x[s1].shape)
        nested = slc.Slice(slc.Slice(MemorySource(x), s1), s2)
        expected = x[s1][s2]

        assert nested.shape == expected.shape
        np.testing.assert_array_equal(nested.array, expected)
        np.testing.assert_array_equal(x[nested.base_slicing], expected)

        before = x.copy()
        nested[...] = -1.0
        before[s1][s2] = -1.0  # s1 holds only slices, so before[s1] is a view
        np.testing.assert_array_equal(x, before)


@pytest.mark.parametrize('axes, overlap', [([2], 3), ([0, 2], 1), ('all', 0)])
def test_blocks_tile_the_source(axes, overlap):
    """What process_block_source does: every element is written once, from the block that owns it."""
    rng = np.random.default_rng(2)
    x = _random_array(rng, (9, 7, 40), 'F')
    sink = np.full(x.shape, np.nan)
    split = dict(processes=3, axes=axes, size_min=overlap + 2, size_max=12, overlap=overlap)

    blocks = bp.split_into_blocks(MemorySource(x), **split)
    sink_blocks = bp.split_into_blocks(MemorySource(sink), **split)
    assert len(blocks) > 1
    for block, sink_block in zip(blocks, sink_blocks):
        result = 2 * block.array + 1
        sink_block.valid[:] = result[block.valid.slicing]

    np.testing.assert_array_equal(sink, 2 * x + 1)


###############################################################################
### Pickling: what crosses the process barrier
###############################################################################

def test_virtual_blocks_pickle_light_and_write_through(tmp_path):
    """The BlockProcessing round trip: virtual blocks are pickled, reopened in the worker, and write to disk."""
    rng = np.random.default_rng(3)
    x = rng.random((20, 30, 64))
    np.save(tmp_path / 'source.npy', x)
    np.save(tmp_path / 'sink.npy', np.zeros_like(x))
    source = NpySource(str(tmp_path / 'source.npy')).as_virtual()
    sink = NpySource(str(tmp_path / 'sink.npy'), mode='r+').as_virtual()

    split = dict(processes=4, axes=[2], size_min=10, size_max=20, overlap=4)
    for block, sink_block in zip(bp.split_into_blocks(source, **split), bp.split_into_blocks(sink, **split)):
        payload = pickle.dumps((block, sink_block))
        assert len(payload) < 4096 < x.nbytes  # handles, not data
        block, sink_block = pickle.loads(payload)
        sink_block.valid[:] = (2 * block.array)[block.valid.slicing]

    np.testing.assert_array_equal(np.load(tmp_path / 'sink.npy'), 2 * x)


def test_as_virtual_and_as_real_keep_the_slicing(tmp_path):
    np.save(tmp_path / 'a.npy', np.arange(60.).reshape(6, 10))
    real = slc.Slice(NpySource(str(tmp_path / 'a.npy')), (slice(1, 4), 2))

    virtual = real.as_virtual()
    assert isinstance(virtual.source, NpyVirtualSource)
    assert virtual.slicing == real.slicing
    assert virtual.shape == real.shape == (3,)
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(virtual)).as_real().array, real.array)

    block = Block(source=NpySource(str(tmp_path / 'a.npy')), slicing=(slice(0, 5),), valid_slicing=(slice(1, -1),))
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(block.as_virtual())).valid.array, block.valid.array)


###############################################################################
### Slice class details
###############################################################################

def test_slice_properties():
    x = np.arange(60.).reshape(6, 10)
    source = MemorySource(x)
    sliced = slc.Slice(source, (slice(1, 4), slice(2, 8)))
    nested = slc.Slice(sliced, (slice(1, None), slice(None, 3)))

    assert sliced.position == sliced.lower == (1, 2)
    assert sliced.upper == (4, 8)
    assert nested.base is source
    assert nested.base_shape == (6, 10)
    assert nested.base_position == nested.base_lower == (2, 2)
    assert nested.base_upper == (4, 5)
    assert nested.name == 'Slice[Slice[MemorySource]]'
    assert nested.dtype == x.dtype
    assert nested.unpacked_slicing == (slice(1, None), slice(None, 3))
    np.testing.assert_array_equal(nested.as_buffer(), x[2:4, 2:5])

    sliced.slicing = (0,)
    assert sliced.shape == (10,)
    sliced.source = MemorySource(x[:3])
    assert sliced.base_shape == (3, 10)


###############################################################################
### Slicing arithmetic on arbitrary slicings, against numpy
###############################################################################

def _any_index(rng, n):
    """Any basic index of an axis of length n: a slice with any sign of step, possibly empty or out of range, an int, or a new axis."""
    r = rng.random()
    if r < 0.15 and n:
        return int(rng.integers(-n, n))
    if r < 0.25:
        return None
    bounds = [None] + list(range(-n - 2, n + 3))
    return slice(rng.choice(bounds), rng.choice(bounds), rng.choice([None, 1, 2, 3, -1, -2, -3]))


def _any_slicing(rng, shape):
    slicing = []
    for n in shape:
        while rng.random() < 0.15:
            slicing.append(None)
        slicing.append(_any_index(rng, n))
    while rng.random() < 0.2:
        slicing.append(None)
    return tuple(slicing)


def test_composition_matches_numpy():
    """x[sliced_slicing(s2, s1)] is x[s1][s2]; where numpy raises, so does sliced_slicing."""
    rng = np.random.default_rng(4)
    checked = 0
    for _ in range(20000):
        shape = tuple(int(n) for n in rng.integers(0, 6, size=rng.integers(1, 4)))
        x = np.arange(int(np.prod(shape))).reshape(shape)
        first = _any_slicing(rng, shape)
        try:
            y = x[first]
        except IndexError:
            continue
        if np.ndim(y) == 0:
            continue
        second = _any_slicing(rng, y.shape)
        try:
            expected = y[second]
        except IndexError:
            with pytest.raises(IndexError):
                slc.sliced_slicing(second, first, shape)
            continue
        try:
            composed = slc.sliced_slicing(second, first, shape)
        except IndexError as error:  # the one refusal, see test_empty_slice_of_new_axis_is_refused
            assert 'Empty slice of a new axis' in str(error)
            continue
        result = x[composed]
        assert result.shape == expected.shape, (shape, first, second, composed)
        np.testing.assert_array_equal(result, expected)
        checked += 1
    assert checked > 10000


@pytest.mark.parametrize('order', ['C', 'F'])
def test_slice_with_any_step_matches_numpy(order):
    """Shape, strides, offset and order of a Slice, now including reversed and empty slices."""
    rng = np.random.default_rng(5)
    for _ in range(3000):
        shape = tuple(int(n) for n in rng.integers(1, 6, size=rng.integers(1, 4)))
        x = _random_array(rng, shape, order)
        s = _any_slicing(rng, shape)
        try:
            expected = x[s]
        except IndexError:
            continue
        if np.ndim(expected) == 0:
            continue
        sliced = slc.Slice(MemorySource(x), s)

        assert sliced.shape == expected.shape
        if expected.size > 1:  # numpy may report any stride for an axis it never steps along
            assert sliced.strides == expected.strides
        if expected.size:
            offset = expected.__array_interface__['data'][0] - x.__array_interface__['data'][0]
            assert sliced.offset == offset
            if sliced.order is not None:
                assert expected.flags[f'{sliced.order}_CONTIGUOUS']


def test_simplify_slicing_matches_numpy():
    """An index array is either turned into an exactly equivalent slice or left alone."""
    rng = np.random.default_rng(6)
    converted = 0
    for _ in range(20000):
        n = int(rng.integers(1, 9))
        if rng.random() < 0.5:  # evenly spaced, any sign of step and start
            step = int(rng.choice([-3, -2, -1, 1, 2, 3]))
            indices = int(rng.integers(-n, n)) + step * np.arange(rng.integers(1, 6))
        else:
            indices = rng.integers(-n, n, size=rng.integers(1, 6))
        x = np.arange(n)
        try:
            expected = x[indices]
        except IndexError:
            continue
        (simplified,) = slc.simplify_slicing((indices,))
        converted += isinstance(simplified, slice)
        np.testing.assert_array_equal(x[simplified], expected)
    assert converted > 1000


###############################################################################
### Regressions: bugs fixed in the slicing arithmetic
###############################################################################

def _compose(x, second, first):
    return x[slc.sliced_slicing(second, first, x.shape)]


def test_trailing_new_axes_in_second_slicing():
    """Used to loop forever."""
    x = np.arange(10)
    composed = slc.sliced_slicing((slice(None), None, None), slice(2, 5), x.shape)
    assert x[composed].shape == (3, 1, 1)


def test_bad_index_on_new_axis():
    """Used to raise NameError from the error message itself."""
    with pytest.raises(IndexError):
        slc.sliced_slicing((5, slice(None)), (None, slice(None)), (3,))


def test_shape_of_reversed_slice():
    assert slc.sliced_shape(slice(None, None, -1), (5,)) == (5,)


def test_shape_of_empty_slice():
    assert slc.sliced_shape(slice(3, 1), (5,)) == (0,)


def test_order_after_new_axis():
    """Used to look up the wrong axis length once a new axis had been seen."""
    assert slc.sliced_order((None, slice(None), 0), 'C', (3, 4)) is None


def test_order_of_reversed_slice():
    """A reversed slice used to be reported as contiguous."""
    assert slc.sliced_order(slice(None, None, -1), 'C', (5,)) is None


def test_offset_of_reversed_slice():
    """numpy's view of a reversed slice starts at its last element, not at s.start."""
    assert slc.sliced_offset(slice(None, None, -1), (8,), (5,)) == 32


def test_offset_of_slice_with_negative_start():
    """Slice.offset used to call sliced_offset without the shape, so a negative start raised."""
    x = np.arange(10.)
    sliced = slc.Slice(MemorySource(x), (slice(-3, None),))
    assert sliced.offset == 7 * x.itemsize


def test_start_of_reversed_slice():
    assert slc.sliced_start((slice(None, None, -1), slice(1, 3)), (5, 4)) == (4, 1)


def test_compose_slice_of_strided_slice():
    np.testing.assert_array_equal(_compose(np.arange(10), slice(1, 3), slice(None, None, 2)), [2, 4])


def test_compose_index_past_the_end():
    with pytest.raises(IndexError):
        _compose(np.arange(10), 3, slice(2, 5))  # np.arange(10)[2:5][3] raises; this used to read x[5]


def test_compose_negative_index_of_strided_slice():
    assert _compose(np.arange(10), -1, slice(None, None, 3)) == 9


def test_compose_slice_of_reversed_slice():
    np.testing.assert_array_equal(_compose(np.arange(10), slice(0, 2), slice(None, None, -1)), [9, 8])


def test_compose_reversed_slice_down_to_zero():
    """The composed range steps past index 0: its stop must become None, not a negative index."""
    np.testing.assert_array_equal(_compose(np.arange(10), slice(None, None, -1), slice(0, 5)), [4, 3, 2, 1, 0])


def test_compose_after_new_axis_in_second_slicing():
    y = np.arange(30).reshape(5, 6)
    second, first = (None, slice(-2, None), slice(None)), (slice(None), slice(None))
    np.testing.assert_array_equal(_compose(y, second, first), y[first][second])


@pytest.mark.parametrize('indices', [[5, 3, 1], [-1, 0], [2, 2], [-3, -2, -1], [2, 1, 0], [-1]])
def test_simplify_irregular_index_arrays(indices):
    x = np.arange(10)
    np.testing.assert_array_equal(x[slc.simplify_slicing((indices,))], x[indices])


###############################################################################
### Refused on purpose
###############################################################################

def test_empty_slice_of_new_axis_is_refused():
    """x[None][1:] has an empty axis where the source has none: no single slicing of x selects that."""
    with pytest.raises(IndexError, match='Empty slice of a new axis'):
        slc.sliced_slicing((slice(1, None), slice(None)), (None, slice(None)), (4,))


def test_offset_of_reversed_slice_needs_shape():
    """Where a reversed slice starts depends on the axis length."""
    with pytest.raises(IndexError, match='reversed slice without shape'):
        slc.sliced_offset(slice(None, None, -1), (8,))


@pytest.mark.parametrize('index', [np.array([True, False, True, True]), [0, 2, 3], [2, 2]])
def test_index_arrays_that_are_not_slices_are_rejected(index):
    """Fancy indexing is not supported. Arrays simplify_slicing cannot make exact slices of are refused, not guessed."""
    with pytest.raises(IndexError):
        slc.sliced_start((index,), (4,))
    with pytest.raises(IndexError):
        slc.sliced_slicing((index,), slice(None), (4,))
