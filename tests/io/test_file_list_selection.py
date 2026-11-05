"""
Which files of a file list a slicing reads, and where their data goes in the array which is returned.
The planes below are Z0001, Z0002, Z0004, Z0008: the position i is the i-th file found, its tag value is
PLANE_NUMBERS[i].
"""
import numpy as np
import pytest

from ClearMap.IO.source.backends.file_list_backend import AxisSelection, FileListSource, FileToRead

PLANE_NUMBERS = [1, 2, 4, 8]


def test_a_slice_selects_the_files_at_these_positions_and_numbers_them_from_zero():
    selection = FileListSource._select_along_axis(slice(1, 3), len(PLANE_NUMBERS), PLANE_NUMBERS)

    assert selection == AxisSelection([2, 4], range(2), True)


def test_a_slice_with_a_step_selects_every_other_file():
    selection = FileListSource._select_along_axis(slice(None, None, 2), len(PLANE_NUMBERS), PLANE_NUMBERS)

    assert selection.tag_values == [1, 4]
    assert list(selection.positions) == [0, 1]


def test_a_negative_slice_counts_from_the_last_file():
    assert FileListSource._select_along_axis(slice(-2, None), len(PLANE_NUMBERS), PLANE_NUMBERS).tag_values == [4, 8]


def test_a_list_selects_the_files_at_these_positions():
    assert FileListSource._select_along_axis([0, 3], len(PLANE_NUMBERS), PLANE_NUMBERS).tag_values == [1, 8]


def test_the_data_of_a_list_goes_to_the_positions_from_zero_not_to_the_positions_selected():
    """[2, 3] selects two files: the result has two planes, at 0 and 1 (not at 2 and 3, which it does not have)"""
    selection = FileListSource._select_along_axis([2, 3], len(PLANE_NUMBERS), PLANE_NUMBERS)

    assert list(selection.positions) == [0, 1]


def test_an_array_of_positions_is_a_list():
    assert FileListSource._select_along_axis(np.array([3, 1]), len(PLANE_NUMBERS), PLANE_NUMBERS).tag_values == [8, 2]


def test_an_integer_selects_one_file_and_removes_the_axis():
    assert FileListSource._select_along_axis(2, len(PLANE_NUMBERS), PLANE_NUMBERS) == AxisSelection([4], [0], False)


def test_a_negative_integer_counts_from_the_last_file():
    assert FileListSource._select_along_axis(-1, len(PLANE_NUMBERS), PLANE_NUMBERS).tag_values == [8]


def test_something_else_than_a_slicing_is_refused():
    with pytest.raises(IndexError, match="Invalid slice specification 'a'"):
        FileListSource._select_along_axis('a', len(PLANE_NUMBERS), PLANE_NUMBERS)


def test_one_axis_gives_one_file_per_selected_position():
    selections = [AxisSelection([2, 4], range(2), True)]

    assert FileListSource._files_to_read(selections, axes_to_tags=[0]) == [FileToRead((2,), (0,)),
                                                                           FileToRead((4,), (1,))]


def test_every_combination_of_one_file_per_axis_is_read_with_the_position_of_its_own_data():
    """Tiles X in (1, 3), Y in (2, 5, 6): the order of the files does not matter, their position does"""
    selections = [AxisSelection([1, 3], range(2), True), AxisSelection([2, 5, 6], range(3), True)]

    files_to_read = FileListSource._files_to_read(selections, axes_to_tags=[0, 1])

    assert len(files_to_read) == 6
    assert set(files_to_read) == {FileToRead((1, 2), (0, 0)), FileToRead((3, 2), (1, 0)),
                                  FileToRead((1, 5), (0, 1)), FileToRead((3, 5), (1, 1)),
                                  FileToRead((1, 6), (0, 2)), FileToRead((3, 6), (1, 2))}


def test_the_values_follow_the_order_of_the_tags_when_the_axes_are_in_another_order():
    """Axes (Y, X) for the tags (X, Y): the file of Y=5, X=3 is tile_03_05, its data goes to Y position 0, X 0"""
    selections = [AxisSelection([5, 6], range(2), True), AxisSelection([3], [0], False)]

    assert FileListSource._files_to_read(selections, axes_to_tags=[1, 0]) == [FileToRead((3, 5), (0, 0)),
                                                                              FileToRead((3, 6), (1, 0))]
