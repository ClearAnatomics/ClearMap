"""Regression tests for writing through a Slice.

A Slice has no mode of its own; it must answer permission questions by delegating
to the source it wraps. Getting this wrong makes every block write in
BlockProcessing raise, or (worse) lets a read-only source accept writes.
"""
import numpy as np
import pytest

from ClearMap.IO.source.backends.mmp_backend import MMPSource
import ClearMap.IO.source.Slice as slc
from ClearMap.IO.source.source_modes import DEFAULT_READ_MODE, DEFAULT_EDIT_MODE
from ClearMap.Utils.exceptions import ClearMapPermissionError

SHAPE = (8, 8)
SLICING = (slice(None), slice(2, 6))


@pytest.fixture
def sink_path(tmp_path):
    path = tmp_path / 'sink.npy'
    source = MMP.create(location=str(path), shape=SHAPE, dtype='uint16')
    del source                       # release the mapping; each test opens its own
    return str(path)


@pytest.fixture
def editable(sink_path):
    return MMPSource(sink_path, mode=DEFAULT_EDIT_MODE)


@pytest.fixture
def read_only(sink_path):
    return MMPSource(sink_path, mode=DEFAULT_READ_MODE)


class TestSliceDelegatesMode:
    def test_slice_exposes_the_mode_of_its_source(self, editable):
        sliced = slc.Slice(editable, slicing=SLICING)
        assert sliced.mode == editable.mode      # AttributeError today if _mode is unset

    def test_slice_of_editable_source_is_persistable(self, editable):
        sliced = slc.Slice(editable, slicing=SLICING)
        assert sliced.is_writable
        assert sliced.is_persistable             # the BlockProcessing regression

    def test_slice_of_read_only_source_is_not_writable(self, read_only):
        sliced = slc.Slice(read_only, slicing=SLICING)
        assert not sliced.is_writable
        assert not sliced.is_persistable

    def test_nested_slice_delegates_through_both_levels(self, editable):
        sliced = slc.Slice(slc.Slice(editable, slicing=SLICING),
                           slicing=(slice(0, 4), slice(None)))
        assert sliced.mode == editable.mode
        assert sliced.is_persistable

    def test_mode_is_delegated_not_copied(self, editable):
        """A stale copy taken at construction is the failure this pins."""
        sliced = slc.Slice(editable, slicing=SLICING)
        editable._mode = DEFAULT_READ_MODE       # simulate a reopen under the slice
        assert sliced.mode == DEFAULT_READ_MODE
        assert not sliced.is_persistable


class TestWriteThroughSlice:
    def test_write_reaches_disk(self, editable, sink_path):
        data = np.full((8, 4), 7, dtype='uint16')
        sliced = slc.Slice(editable, slicing=SLICING)
        MMP.write(sliced, data, flush=True)
        del sliced, editable
        assert np.all(MMP.read(sink_path)[SLICING] == 7)

    def test_write_to_slice_of_read_only_source_is_refused(self, read_only):
        sliced = slc.Slice(read_only, slicing=SLICING)
        with pytest.raises(ClearMapPermissionError):
            MMP.write(sliced, np.ones((8, 4), dtype='uint16'))

    def test_refusal_names_the_underlying_file(self, read_only, sink_path):
        sliced = slc.Slice(read_only, slicing=SLICING)
        with pytest.raises(ClearMapPermissionError, match=r'sink\.npy'):
            MMP._assert_durable_sink(sliced, context='test')

    @pytest.mark.parametrize('mode, persistable', [('r', False), ('c', False),
                                                   ('r+', True)])
    def test_durability_matches_mode(self, sink_path, mode, persistable):
        """'c' accepts writes and silently discards them: not durable."""
        sliced = slc.Slice(MMPSource(sink_path, mode=mode), slicing=SLICING)
        assert sliced.is_persistable is persistable

    def test_copy_on_write_slice_does_not_reach_disk(self, sink_path):
        sliced = slc.Slice(MMPSource(sink_path, mode='c'), slicing=SLICING)
        with pytest.raises(ClearMapPermissionError):
            MMP.write(sliced, np.ones((8, 4), dtype='uint16'))
        assert np.all(MMP.read(sink_path) == 0)
