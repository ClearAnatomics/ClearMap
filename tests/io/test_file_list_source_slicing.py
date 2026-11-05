"""
A stack split in one file per plane is sliced along the planes like an array.
"""
import numpy as np
import pytest

from ClearMap.IO.source.backends.file_list_backend import FileListSource

PLANE_SHAPE = (4, 3)
PLANE_NUMBERS = [1, 2, 3, 4]


def _stack_of_planes(directory):
    """One npy file per plane, whose pixels all equal the number of the plane"""
    for number in PLANE_NUMBERS:
        np.save(directory / f'Z{number:04d}.npy', np.full(PLANE_SHAPE, number, dtype=np.uint16))
    return FileListSource(expression=str(directory / 'Z<Z,4>.npy'))


def _plane_numbers(stack):
    return [int(plane[0, 0]) for plane in np.moveaxis(np.asarray(stack), -1, 0)]


def test_a_negative_position_or_slice_counts_from_the_last_plane(tmp_path):
    source = _stack_of_planes(tmp_path)

    np.testing.assert_array_equal(source[:, :, -1], np.full(PLANE_SHAPE, 4))
    assert _plane_numbers(source[:, :, -2:]) == [3, 4]
