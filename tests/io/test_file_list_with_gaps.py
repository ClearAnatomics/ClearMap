"""
A stack split in one file per plane is read by position along the planes. When the numbering has gaps
(planes removed or not exported, Z0001, Z0002, Z0004), the i-th plane is the i-th file found (#73).
"""
import numpy as np
import pytest

from ClearMap.IO.source.backends.file_list_backend import FileListSource

PLANE_SHAPE = (4, 3)


def _write_planes(directory, numbers):
    """One npy file per number, whose pixels all equal that number"""
    for number in numbers:
        np.save(directory / f'Z{number:04d}.npy', np.full(PLANE_SHAPE, number, dtype=np.uint16))
    return str(directory / 'Z<Z,4>.npy')


def _plane_numbers(stack):
    return [int(plane[0, 0]) for plane in np.moveaxis(np.asarray(stack), -1, 0)]


@pytest.mark.parametrize('numbers', [[1, 2, 3, 4], [1, 2, 4], [3, 5, 6, 10], [0, 7]], ids=str)
def test_every_plane_is_read_in_the_order_of_the_numbering(tmp_path, numbers):
    source = FileListSource(expression=_write_planes(tmp_path, numbers))

    assert source.shape == PLANE_SHAPE + (len(numbers),)
    assert _plane_numbers(source[:]) == numbers


def test_a_slice_of_planes_skips_the_gaps(tmp_path):
    source = FileListSource(expression=_write_planes(tmp_path, [1, 2, 4, 8]))

    assert _plane_numbers(source[:, :, 1:3]) == [2, 4]
    assert _plane_numbers(source[:, :, ::2]) == [1, 4]


def test_a_single_plane_is_read_by_its_position(tmp_path):
    source = FileListSource(expression=_write_planes(tmp_path, [1, 2, 4, 8]))

    np.testing.assert_array_equal(source[:, :, 2], np.full(PLANE_SHAPE, 4))


def test_two_tags_with_gaps_are_read_by_position_in_the_order_of_the_axes(tmp_path):
    """Tiles X in (1, 3), Y in (2, 5, 6); the value of each tile tells where it comes from"""
    x_values, y_values = (1, 3), (2, 5, 6)
    for x in x_values:
        for y in y_values:
            np.save(tmp_path / f'tile_{x:02d}_{y:02d}.npy', np.full(PLANE_SHAPE, 10 * x + y, dtype=np.uint16))

    source = FileListSource(expression=str(tmp_path / 'tile_<X,2>_<Y,2>.npy'), axes_order=['Y', 'X'])

    assert source.shape == PLANE_SHAPE + (len(y_values), len(x_values))
    tiles = np.asarray(source[:])
    assert [[int(tiles[0, 0, j, i]) for i in range(2)] for j in range(3)] == [[12, 32], [15, 35], [16, 36]]
    assert int(np.asarray(source[:, :, 1:, 1])[0, 0, 0]) == 35


def test_the_values_of_the_tags_per_axis_are_those_of_the_docstring_example(tmp_path):
    """Tiles X in (1, 3) and Y in (2, 5, 6): the numbers of the docstring of _tag_values_per_axis"""
    for x in (1, 3):
        for y in (2, 5, 6):
            np.save(tmp_path / f'tile_{x:02d}_{y:02d}.npy', np.zeros(PLANE_SHAPE, dtype=np.uint16))
    expression = str(tmp_path / 'tile_<X,2>_<Y,2>.npy')

    assert FileListSource(expression=expression)._tag_values_per_axis() == [[1, 3], [2, 5, 6]]
    assert FileListSource(expression=expression, axes_order=['Y', 'X'])._tag_values_per_axis() == [[2, 5, 6], [1, 3]]


def test_the_values_of_the_tag_of_planes_with_a_gap_are_those_of_the_docstring_example(tmp_path):
    source = FileListSource(expression=_write_planes(tmp_path, [1, 2, 4]))

    assert source._tag_values_per_axis() == [[1, 2, 4]]


def test_the_names_of_the_files_are_parsed_once_whatever_the_number_of_reads(tmp_path, monkeypatch):
    """Parsing every name at every read made reading a stack plane by plane quadratic in the number of planes"""
    source = FileListSource(expression=_write_planes(tmp_path, [1, 2, 4, 8]))
    names_parsed = []
    parse = source.expression.indices

    def parse_and_count(name):
        names_parsed.append(name)
        return parse(name)
    monkeypatch.setattr(source.expression, 'indices', parse_and_count)

    for position in range(4):
        source[:, :, position]

    assert len(names_parsed) == 4  # Each of the 4 files once, at the first read


def test_changing_the_order_of_the_axes_after_a_read_is_still_followed(tmp_path):
    for x in (1, 3):
        for y in (2, 5, 6):
            np.save(tmp_path / f'tile_{x:02d}_{y:02d}.npy', np.zeros(PLANE_SHAPE, dtype=np.uint16))
    source = FileListSource(expression=str(tmp_path / 'tile_<X,2>_<Y,2>.npy'))
    assert source._tag_values_per_axis() == [[1, 3], [2, 5, 6]]

    source.axes_order = ['Y', 'X']

    assert source._tag_values_per_axis() == [[2, 5, 6], [1, 3]]
