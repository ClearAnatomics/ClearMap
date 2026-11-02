"""
The ``.bin`` point files of the patched elastix build start with two int64 (index flag, number of points), then the
coordinates as float64; ``read_points`` reads them back.
"""
import numpy as np
import pytest

from ClearMap.Alignment import Elastix as elastix
from ClearMap.Utils.exceptions import ClearMapIoException

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


def bin_point_file_bytes(header, header_dtype=np.int64, points_dtype=np.double):
    return np.array(header, dtype=header_dtype).tobytes() + POINTS.astype(points_dtype).tobytes()


INVALID_BIN_POINT_FILES = {
    'a negative count': bin_point_file_bytes([0, -1]),
    'a flag that is neither 0 nor 1': bin_point_file_bytes([2, len(POINTS)]),
    'a header of 4 byte integers': bin_point_file_bytes([0, len(POINTS)], header_dtype=np.int32),
    'coordinates of 4 bytes': bin_point_file_bytes([0, len(POINTS)], points_dtype=np.float32),
    'a count larger than the points present': bin_point_file_bytes([0, len(POINTS) + 1]),
    'a count smaller than the points present': bin_point_file_bytes([0, len(POINTS) - 1]),
    'a last point cut in the middle': bin_point_file_bytes([0, len(POINTS)])[:-4],
    'points although the count is zero': bin_point_file_bytes([0, 0]),
    'less bytes than the header': np.array([0], dtype=np.int64).tobytes(),
    'nothing': b'',
}


@pytest.mark.parametrize('content', INVALID_BIN_POINT_FILES.values(), ids=INVALID_BIN_POINT_FILES.keys())
def test_a_bin_point_file_whose_header_does_not_match_its_size_is_refused_with_its_name(tmp_path, content):
    point_file = tmp_path / 'wrong_points.bin'
    point_file.write_bytes(content)

    with pytest.raises(ClearMapIoException, match='wrong_points.bin'):
        elastix.read_points(str(point_file), binary=True)


@pytest.mark.parametrize('header_dtype', [np.int64, np.uint64])
def test_a_bin_point_file_reads_the_same_whether_its_header_is_signed_or_not(tmp_path, header_dtype):
    point_file = tmp_path / 'points.bin'
    point_file.write_bytes(bin_point_file_bytes([0, len(POINTS)], header_dtype=header_dtype))

    np.testing.assert_array_equal(elastix.read_points(str(point_file), binary=True), POINTS)
