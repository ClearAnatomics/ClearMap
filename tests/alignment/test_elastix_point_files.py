"""
The ``.bin`` point files of the patched elastix build start with two int64 (index flag, number of points), then the
coordinates as float64; ``read_points`` reads them back.
"""
import numpy as np
import pytest

from ClearMap.Alignment import Elastix as elastix

POINTS = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


@pytest.mark.parametrize('indices, expected_index_flag', [(False, 0), (True, 1)])
def test_a_bin_point_file_starts_with_the_index_flag_and_the_number_of_points(tmp_path, indices,
                                                                               expected_index_flag):
    point_file = tmp_path / 'points.bin'

    elastix.write_points(str(point_file), POINTS, indices=indices, binary=True)

    content = point_file.read_bytes()
    assert np.frombuffer(content[:16], dtype=np.int64).tolist() == [expected_index_flag, len(POINTS)]
    np.testing.assert_array_equal(np.frombuffer(content[16:], dtype=np.double).reshape(-1, 3), POINTS)


@pytest.mark.parametrize('points', [POINTS, np.zeros((0, 3))])
def test_a_bin_point_file_reads_back_the_points(tmp_path, points):
    point_file = tmp_path / 'points.bin'
    elastix.write_points(str(point_file), points, binary=True)

    np.testing.assert_array_equal(elastix.read_points(str(point_file), binary=True), points)
